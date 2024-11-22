import torch
import math
from argparse import Namespace
from typing import Optional, List, Dict, Union
from tqdm import tqdm
from torchdiffeq import odeint

from .Layer import Conv_Init, Lambda, FFT_Block

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
        singers: torch.FloatTensor,
        mels: torch.FloatTensor,
        lengths: torch.IntTensor
        ):
        '''
        encodings: [Batch, Enc_d, Dec_t]
        mels: [Batch, Mel_d, Mel_t]
        '''
        diffusion_times = torch.rand(encodings.size(0), device= encodings.device)   # [Batch]
        cosmap_schedule_times = self.cosmap_calc(diffusion_times)[:, None, None]

        noises = torch.randn_like(mels)  # [Batch, Mel_d, Mel_t]

        noised_mels = cosmap_schedule_times * mels + (1.0 - cosmap_schedule_times) * noises
        flows = mels - noises

        # predict flow
        network_output = self.network(
            noised_mels= noised_mels,
            encodings= encodings,
            singers= singers,
            lengths= lengths,
            diffusion_steps= cosmap_schedule_times[:, 0, 0] # [Batch]
            )
        
        if self.hp.Diffusion.Network_Prediction == 'Flow':
            prediction_flows = network_output
            prediction_noises = noised_mels - prediction_flows * cosmap_schedule_times.clamp(1e-7)   # is it working?
        elif self.hp.Diffusion.Network_Prediction == 'Noise':
            prediction_noises = network_output
            prediction_flows = (noised_mels - prediction_noises) / cosmap_schedule_times.clamp(1e-7)
        else:
            NotImplementedError(f'Unsupported diffusion network prediction type: {self.hp.Diffusion.Network_Prediction}')

        # prediction_mels = noised_mels + prediction_flows * (1.0 - cosmap_schedule_times)    # not using

        return flows, prediction_flows, noises, prediction_noises
    
    def Inference(
        self,
        encodings: torch.FloatTensor,
        singers: torch.FloatTensor,
        lengths: torch.IntTensor,
        steps: int
        ):
        noises = torch.randn(
            encodings.size(0), self.hp.Sound.N_Mel, encodings.size(2),
            device= encodings.device,
            dtype= encodings.dtype
            )  # [Batch, Mel_d, Mel_t]
        
        def ode_func(diffusion_times: torch.Tensor, noised_mels: torch.Tensor):
            diffusion_times = diffusion_times[None].expand(noised_mels.size(0))
            cosmap_schedule_times = self.cosmap_calc(diffusion_times)[:, None, None]

            # predict flow
            network_output = self.network(
                noised_mels= noised_mels,
                encodings= encodings,
                singers= singers,
                lengths= lengths,
                diffusion_steps= cosmap_schedule_times[:, 0, 0] # [Batch]
                )
            
            if self.hp.Diffusion.Network_Prediction == 'Flow':
                prediction_flows = network_output                
            elif self.hp.Diffusion.Network_Prediction == 'Noise':
                prediction_noises = network_output
                prediction_flows = (noised_mels - prediction_noises) / cosmap_schedule_times.clamp(1e-7)
            else:
                NotImplementedError(f'Unsupported diffusion network prediction type: {self.hp.Diffusion.Network_Prediction}')

            return prediction_flows
        
        mels = odeint(
            func= ode_func,
            y0= noises,
            t= torch.linspace(0.0, 1.0, steps, device= encodings.device),
            atol= 1e-5,
            rtol= 1e-5,
            method= 'midpoint'
            )[-1]

        return mels

class Network(torch.nn.Module):
    def __init__(
        self,
        hyper_parameters: Namespace
        ):
        super().__init__()
        self.hp = hyper_parameters

        self.prenet = torch.nn.Sequential(
            Conv_Init(torch.nn.Conv1d(
                in_channels= self.hp.Sound.N_Mel,
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
        
        self.singer_ffn = torch.nn.Sequential(
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

        self.blocks = torch.nn.ModuleList([
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
                out_channels= self.hp.Sound.N_Mel,
                kernel_size= 1), w_init_gain= 'zero'),
            )

    def forward(
        self,
        noised_mels: torch.FloatTensor,
        encodings: torch.FloatTensor,
        singers: torch.FloatTensor,
        lengths: torch.FloatTensor,
        diffusion_steps: torch.FloatTensor
        ):
        '''
        noised_mels: [Batch, Mel_d, Dec_t]
        encodings: [Batch, Enc_d, Dec_t]
        singers: [Batch, Enc_d, Dec_t]
        diffusion_steps: [Batch]
        '''
        x = self.prenet(noised_mels)
        encodings = self.encoding_ffn(encodings)
        singers = self.singer_ffn(singers)
        diffusion_steps = self.step_ffn(diffusion_steps) # [Batch, Res_d, 1]

        x = x + encodings + singers + diffusion_steps

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