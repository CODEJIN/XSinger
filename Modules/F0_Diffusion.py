import torch
import math
from argparse import Namespace
from typing import Optional, List, Dict, Union
from tqdm import tqdm
from torchdiffeq import odeint

from .Layer import Conv_Init, Lambda, FFT_Block, Positional_Encoding

class F0_Diffusion(torch.nn.Module):
    def __init__(
        self,
        hyper_parameters: Namespace
        ):
        super().__init__()
        self.hp = hyper_parameters

        self.network = Network(
            hyper_parameters= self.hp
            )
        self.cosmap_calc = lambda x: 1.0 - (1 / (torch.tan(torch.pi / 2.0 * x) + 1))

    def forward(
        self,
        encodings: torch.FloatTensor,
        f0s: torch.FloatTensor,
        lengths: torch.IntTensor
        ):
        '''
        encodings: [Batch, Enc_d, Dec_t]
        f0s: [Batch, Dec_t]
        '''
        diffusion_times = torch.rand(encodings.size(0), device= encodings.device)   # [Batch]
        cosmap_schedule_times = self.cosmap_calc(diffusion_times)[:, None]

        noises = torch.randn_like(f0s)  # [Batch, Dec_t]

        noised_f0s = cosmap_schedule_times * f0s + (1.0 - cosmap_schedule_times) * noises
        flows = f0s - noises

        # predict flow
        network_output = self.network(
            noised_f0s= noised_f0s,
            encodings= encodings,
            lengths= lengths,
            diffusion_steps= cosmap_schedule_times[:, 0] # [Batch]
            )
        
        if self.hp.F0_Diffusion.Network_Prediction == 'Flow':
            prediction_flows = network_output
            prediction_noises = noised_f0s - prediction_flows * cosmap_schedule_times.clamp(1e-7)   # is it working?
        elif self.hp.F0_Diffusion.Network_Prediction == 'Noise':
            prediction_noises = network_output
            prediction_flows = (noised_f0s - prediction_noises) / cosmap_schedule_times.clamp(1e-7)
        else:
            NotImplementedError(f'Unsupported diffusion network prediction type: {self.hp.F0_Diffusion.Network_Prediction}')

        return flows, prediction_flows, noises, prediction_noises
    
    def Inference(
        self,
        encodings: torch.FloatTensor,
        lengths: torch.IntTensor,
        steps: int
        ):
        noises = torch.randn(
            encodings.size(0), encodings.size(2),
            device= encodings.device,
            dtype= encodings.dtype
            )  # [Batch, Latent_d, Latent_t]
        
        def ode_func(diffusion_times: torch.Tensor, noised_f0s: torch.Tensor):
            diffusion_times = diffusion_times[None].expand(noised_f0s.size(0))
            cosmap_schedule_times = self.cosmap_calc(diffusion_times)[:, None]

            # predict flow
            network_output = self.network(
                noised_f0s= noised_f0s,
                encodings= encodings,
                lengths= lengths,
                diffusion_steps= cosmap_schedule_times[:, 0] # [Batch]
                )
            
            if self.hp.F0_Diffusion.Network_Prediction == 'Flow':
                prediction_flows = network_output                
            elif self.hp.F0_Diffusion.Network_Prediction == 'Noise':
                prediction_noises = network_output
                prediction_flows = (noised_f0s - prediction_noises) / cosmap_schedule_times.clamp(1e-7)
            else:
                NotImplementedError(f'Unsupported diffusion network prediction type: {self.hp.F0_Diffusion.Network_Prediction}')

            return prediction_flows
        
        f0s = odeint(
            func= ode_func,
            y0= noises,
            t= torch.linspace(0.0, 1.0, steps, device= encodings.device),
            atol= 1e-5,
            rtol= 1e-5,
            method= 'midpoint'
            )[-1]

        return f0s

class Network(torch.nn.Module):
    def __init__(
        self,
        hyper_parameters: Namespace
        ):
        super().__init__()
        self.hp = hyper_parameters

        self.prenet = torch.nn.Sequential(
            Lambda(lambda x: x[:, None, :]),
            Conv_Init(torch.nn.Conv1d(
                in_channels= 1,
                out_channels= self.hp.F0_Diffusion.Size,
                kernel_size= 1,
                ), w_init_gain= 'gelu'),
            torch.nn.GELU(approximate= 'tanh')
            )
        
        self.encoding_ffn = torch.nn.Sequential(
            Conv_Init(torch.nn.Conv1d(
                in_channels= self.hp.Encoder.Size,
                out_channels= self.hp.F0_Diffusion.Size * 4,
                kernel_size= 1,
                ), w_init_gain= 'gelu'),
            torch.nn.GELU(approximate= 'tanh'),
            Conv_Init(torch.nn.Conv1d(
                in_channels= self.hp.F0_Diffusion.Size * 4,
                out_channels= self.hp.F0_Diffusion.Size,
                kernel_size= 1,
                ), w_init_gain= 'linear')
            )
        
        self.step_ffn = torch.nn.Sequential(
            Step_Embedding(
                embedding_dim= self.hp.F0_Diffusion.Size
                ),            
            Conv_Init(torch.nn.Linear(
                in_features= self.hp.F0_Diffusion.Size,
                out_features= self.hp.F0_Diffusion.Size * 4,
                ), w_init_gain= 'gelu'),
            torch.nn.GELU(approximate= 'tanh'),
            Conv_Init(torch.nn.Linear(
                in_features= self.hp.F0_Diffusion.Size * 4,
                out_features= self.hp.F0_Diffusion.Size,
                ), w_init_gain= 'linear'),
            Lambda(lambda x: x[:, :, None])
            )

        self.positional_encoding = Positional_Encoding(
            num_embeddings= self.hp.Durations,
            embedding_dim= self.hp.F0_Diffusion.Size
            )

        self.blocks: List[FFT_Block] = torch.nn.ModuleList([
            FFT_Block(
                channels= self.hp.F0_Diffusion.Size,
                num_head= self.hp.F0_Diffusion.Transformer.Head,
                residual_conv_stack= self.hp.F0_Diffusion.Transformer.Residual_Conv.Stack,
                residual_conv_kernel_size= self.hp.F0_Diffusion.Transformer.Residual_Conv.Kernel_Size,
                ffn_kernel_size= self.hp.F0_Diffusion.Transformer.FFN.Kernel_Size,
                residual_conv_dropout_rate= self.hp.F0_Diffusion.Transformer.Residual_Conv.Dropout_Rate,
                ffn_dropout_rate= self.hp.F0_Diffusion.Transformer.FFN.Dropout_Rate,
                )
            for _ in range(self.hp.F0_Diffusion.Transformer.Stack)
            ])  # real type: torch.nn.ModuleList[FFT_BLock]

        self.projection = torch.nn.Sequential(
            Conv_Init(torch.nn.Conv1d(
                in_channels= self.hp.F0_Diffusion.Size,
                out_channels= self.hp.F0_Diffusion.Size,
                kernel_size= 1
                ), w_init_gain= 'gelu'),
            torch.nn.GELU(approximate= 'tanh'),
            Conv_Init(torch.nn.Conv1d(
                in_channels= self.hp.F0_Diffusion.Size,
                out_channels= 1,
                kernel_size= 1), w_init_gain= 'zero'),
            Lambda(lambda x: x[:, 0, :])
            )

    def forward(
        self,
        noised_f0s: torch.Tensor,
        encodings: torch.Tensor,
        lengths: torch.Tensor,
        diffusion_steps: torch.Tensor
        ):
        '''
        noised_f0s: [Batch, Dec_t]        
        encodings: [Batch, Enc_d, Dec_t]
        diffusion_steps: [Batch]
        '''
        x = self.prenet(noised_f0s)
        encodings = self.encoding_ffn(encodings)
        diffusion_steps = self.step_ffn(diffusion_steps) # [Batch, Res_d, 1]

        x = x + encodings + diffusion_steps + self.positional_encoding(
            position_ids= torch.arange(x.size(2), device= encodings.device)[None]
            ).mT

        for block in self.blocks:
            x = block(x, lengths)

        x = self.projection(x)

        return x

class Step_Embedding(torch.nn.Module):
    '''
    sinusoidal for float input
    '''
    def __init__(
        self,
        embedding_dim: int,
        ):
        super().__init__()
        assert embedding_dim % 2 == 0
        self.embedding_dim = embedding_dim

        half_dim = embedding_dim // 2
        div_term = math.log(10000) / (half_dim - 1)
        div_term = torch.exp(torch.arange(half_dim, dtype= torch.int64).float() * -div_term)

        self.register_buffer('div_term', div_term, persistent= False)

    def forward(self, x: torch.Tensor):
        x = x[:, None] * self.div_term[None]
        x = torch.cat([x.sin(), x.cos()], dim= 1)

        return x

@torch.jit.script
def Fused_Gate(x):
    x_tanh, x_sigmoid = x.chunk(chunks= 2, dim= 1)
    x = x_tanh.tanh() * x_sigmoid.sigmoid()

    return x