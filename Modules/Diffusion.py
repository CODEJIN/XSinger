import torch
import math
from argparse import Namespace
from typing import Optional, List, Dict, Union
from tqdm import tqdm
from torchdiffeq import odeint

from .Layer import Conv_Init, Lambda, FFT_Block, Positional_Encoding

class Diffusion(torch.nn.Module):
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
        latents: torch.FloatTensor,
        lengths: torch.IntTensor
        ):
        '''
        encodings: [Batch, Enc_d, Dec_t]
        latents: [Batch, Latent_d, Latent_t]
        '''
        diffusion_times = torch.rand(encodings.size(0), device= encodings.device)   # [Batch]
        cosmap_schedule_times = self.cosmap_calc(diffusion_times)[:, None, None]

        noises = torch.randn_like(latents)  # [Batch, Latent_d, Latent_t]

        noised_latents = cosmap_schedule_times * latents + (1.0 - cosmap_schedule_times) * noises
        flows = latents - noises

        # predict flow
        network_output = self.network(
            noised_latents= noised_latents,
            encodings= encodings,
            lengths= lengths,
            diffusion_steps= cosmap_schedule_times[:, 0, 0] # [Batch]
            )
        
        if self.hp.Diffusion.Network_Prediction == 'Flow':
            prediction_flows = network_output
            prediction_noises = noised_latents - prediction_flows * cosmap_schedule_times.clamp(1e-7)   # is it working?
        elif self.hp.Diffusion.Network_Prediction == 'Noise':
            prediction_noises = network_output
            prediction_flows = (noised_latents - prediction_noises) / cosmap_schedule_times.clamp(1e-7)
        else:
            NotImplementedError(f'Unsupported diffusion network prediction type: {self.hp.Diffusion.Network_Prediction}')

        # prediction_latents = noised_latents + prediction_flows * (1.0 - cosmap_schedule_times)    # not using

        return flows, prediction_flows, noises, prediction_noises
    
    def Inference(
        self,
        encodings: torch.FloatTensor,
        lengths: torch.IntTensor,
        steps: int
        ):
        noises = torch.randn(
            encodings.size(0), self.hp.Audio_Codec_Size, encodings.size(2),
            device= encodings.device,
            dtype= encodings.dtype
            )  # [Batch, Latent_d, Latent_t]
        
        def ode_func(diffusion_times: torch.Tensor, noised_latents: torch.Tensor):
            diffusion_times = diffusion_times[None].expand(noised_latents.size(0))
            cosmap_schedule_times = self.cosmap_calc(diffusion_times)[:, None, None]

            # predict flow
            network_output = self.network(
                noised_latents= noised_latents,
                encodings= encodings,
                lengths= lengths,
                diffusion_steps= cosmap_schedule_times[:, 0, 0] # [Batch]
                )
            
            if self.hp.Diffusion.Network_Prediction == 'Flow':
                prediction_flows = network_output                
            elif self.hp.Diffusion.Network_Prediction == 'Noise':
                prediction_noises = network_output
                prediction_flows = (noised_latents - prediction_noises) / cosmap_schedule_times.clamp(1e-7)
            else:
                NotImplementedError(f'Unsupported diffusion network prediction type: {self.hp.Diffusion.Network_Prediction}')

            return prediction_flows
        
        latents = odeint(
            func= ode_func,
            y0= noises,
            t= torch.linspace(0.0, 1.0, steps, device= encodings.device),
            atol= 1e-5,
            rtol= 1e-5,
            method= 'midpoint'
            )[-1]

        return latents

class Network(torch.nn.Module):
    def __init__(
        self,
        hyper_parameters: Namespace
        ):
        super().__init__()
        self.hp = hyper_parameters

        self.prenet = torch.nn.Sequential(
            Conv_Init(torch.nn.Conv1d(
                in_channels= self.hp.Audio_Codec_Size,
                out_channels= self.hp.Diffusion.Size,
                kernel_size= 1,
                ), w_init_gain= 'gelu'),
            torch.nn.GELU(approximate= 'tanh')
            )
        
        self.encoding_ffn = torch.nn.Sequential(
            Conv_Init(torch.nn.Conv1d(
                in_channels= self.hp.Encoder.Size,
                out_channels= self.hp.Diffusion.Size * 4,
                kernel_size= 1,
                ), w_init_gain= 'gelu'),
            torch.nn.GELU(approximate= 'tanh'),
            Conv_Init(torch.nn.Conv1d(
                in_channels= self.hp.Diffusion.Size * 4,
                out_channels= self.hp.Diffusion.Size,
                kernel_size= 1,
                ), w_init_gain= 'linear')
            )
        
        self.step_ffn = torch.nn.Sequential(
            Step_Embedding(
                embedding_dim= self.hp.Diffusion.Size
                ),            
            Conv_Init(torch.nn.Linear(
                in_features= self.hp.Diffusion.Size,
                out_features= self.hp.Diffusion.Size * 4,
                ), w_init_gain= 'gelu'),
            torch.nn.GELU(approximate= 'tanh'),
            Conv_Init(torch.nn.Linear(
                in_features= self.hp.Diffusion.Size * 4,
                out_features= self.hp.Diffusion.Size,
                ), w_init_gain= 'linear'),
            Lambda(lambda x: x[:, :, None])
            )

        self.positional_encoding = Positional_Encoding(
            num_embeddings= self.hp.Durations,
            embedding_dim= self.hp.Diffusion.Size
            )

        self.blocks: List[FFT_Block] = torch.nn.ModuleList([
            FFT_Block(
                channels= self.hp.Diffusion.Size,
                num_head= self.hp.Diffusion.Transformer.Head,
                residual_conv_stack= self.hp.Diffusion.Transformer.Residual_Conv.Stack,
                residual_conv_kernel_size= self.hp.Diffusion.Transformer.Residual_Conv.Kernel_Size,
                ffn_kernel_size= self.hp.Diffusion.Transformer.FFN.Kernel_Size,
                residual_conv_dropout_rate= self.hp.Diffusion.Transformer.Residual_Conv.Dropout_Rate,
                ffn_dropout_rate= self.hp.Diffusion.Transformer.FFN.Dropout_Rate,
                )
            for _ in range(self.hp.Diffusion.Transformer.Stack)
            ])  # real type: torch.nn.ModuleList[FFT_BLock]

        self.projection = torch.nn.Sequential(
            Conv_Init(torch.nn.Conv1d(
                in_channels= self.hp.Diffusion.Size,
                out_channels= self.hp.Diffusion.Size,
                kernel_size= 1
                ), w_init_gain= 'gelu'),
            torch.nn.GELU(approximate= 'tanh'),
            Conv_Init(torch.nn.Conv1d(
                in_channels= self.hp.Diffusion.Size,
                out_channels= self.hp.Audio_Codec_Size,
                kernel_size= 1), w_init_gain= 'zero'),
            )

    def forward(
        self,
        noised_latents: torch.Tensor,
        encodings: torch.Tensor,
        lengths: torch.Tensor,
        diffusion_steps: torch.Tensor
        ):
        '''
        noised_latents: [Batch, Latent_d, Dec_t]
        encodings: [Batch, Enc_d, Dec_t]
        diffusion_steps: [Batch]
        '''
        x = self.prenet(noised_latents)
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