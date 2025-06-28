import torch
import math
from argparse import Namespace
from typing import Optional, List, Dict, Union
from tqdm import tqdm
import ot

from .Layer import Conv_Init, ConvTranspose_Init, Lambda, FFT_Block, Norm_Type, Mask_Generate

class RectifiedFlow(torch.nn.Module):
    def __init__(
        self,
        hyper_parameters: Namespace
        ):
        super().__init__()
        self.hp = hyper_parameters

        self.network = Network(
            hyper_parameters= self.hp
            )

        if self.hp.F0_Predictor.Scheduler.upper() == 'Uniform'.upper():
            self.timestep_scheduler = lambda x: x
        elif self.hp.F0_Predictor.Scheduler.upper() == 'Cosmap'.upper():
            self.timestep_scheduler = lambda x: 1.0 - (1 / (torch.tan(torch.pi / 2.0 * x) + 1))
        else:
            raise NotImplementedError(f'Unsupported RectifiedFlow scheduler: {self.hp.F0_Predictor.Scheduler}')

    def forward(
        self,
        encodings: torch.FloatTensor,
        f0s: torch.FloatTensor
        ):
        '''
        encodings: [Batch, Enc_d, Dec_t]
        f0s: [Batch, F0_t]
        '''        
        if self.hp.F0_Predictor.Use_CFG:
            cfg_filters = (torch.rand(
                encodings.size(0),
                device= encodings.device
                )[:, None, None] > self.hp.Train.CFG_Alpha).float()  # [Batch, 1, 1]            
            encodings = encodings * cfg_filters

        timesteps = self.timestep_scheduler(
            torch.rand(encodings.size(0), device= encodings.device)
            )[:, None]

        if self.hp.F0_Predictor.Use_OT:
            noises = self.OT_CFM_Sampling(f0s)
        else:
            noises = torch.randn_like(f0s)  # [Batch, F0_t]

        noised_f0s = timesteps * f0s + (1.0 - timesteps) * noises  # [Batch, F0_t]
        # Rectified Flow target: v(x_t, t) = x_1 - x_0
        flows = f0s - noises

        # predict flow
        network_outputs = self.network(
            f0s= noised_f0s,
            encodings= encodings,
            steps= timesteps[:, 0]  # [Batch]
            )
        prediction_flows = network_outputs

        return flows, prediction_flows
    
    def Inference(
        self,
        encodings: torch.FloatTensor,        
        steps: int,
        cfg_guidance_scale: Optional[float]= 2.0
        ):  # Euler solver
        x = torch.randn(
            encodings.size(0), encodings.size(2),
            device= encodings.device,
            dtype= encodings.dtype
            )  # [Batch, F0_t]

        timesteps_list = self.timestep_scheduler(torch.linspace(
            start= 0.0,
            end= 1.0,
            steps= steps + 1,
            device= encodings.device,
            dtype= encodings.dtype
            )) # [Step + 1]
        delta_timesteps_list = timesteps_list[1:] - timesteps_list[:-1]  # [Step]

        timesteps_list = timesteps_list[:-1, None].expand(-1, x.size(0))  # [Step, Batch]
        delta_timesteps_list = delta_timesteps_list[:, None].expand(-1, x.size(0))   # [Step, Batch]

        # Prepare inputs for potential CFG (duplicate batch for unconditional call)
        if self.hp.F0_Predictor.Use_CFG:
            batch_size = x.size(0)
            x_double = torch.cat([x, x], dim=0)
            encodings_double = torch.cat([encodings, torch.zeros_like(encodings)], dim=0)

        for timesteps, delta_timesteps in zip(timesteps_list, delta_timesteps_list):
            if self.hp.F0_Predictor.Use_CFG:                
                delta_timesteps_double = torch.cat([delta_timesteps, delta_timesteps], dim= 0)
                timesteps_double = torch.cat([timesteps, timesteps], dim=0) # timesteps are the same for both
                network_outputs_double = self.network(
                    f0s= x_double,
                    encodings= encodings_double,
                    steps= timesteps_double
                    )
                # Split outputs and apply CFG
                network_outputs_with_condition, network_outputs_without_condition = network_outputs_double.chunk(2, dim=0)
                network_outputs_with_condition = network_outputs_with_condition + cfg_guidance_scale * (network_outputs_with_condition - network_outputs_without_condition)

                # Combine the guided flow for the conditional part and the original flow for the unconditional part
                network_outputs_double = torch.cat([network_outputs_with_condition, network_outputs_without_condition], dim=0)

                # Update x_double for the next step
                x_double = x_double + delta_timesteps_double[:, None] * network_outputs_double # Euler update
                x = x_double[:batch_size] # Update x for the next iteration (only the conditional part)
            else:
                network_outputs = self.network(
                    f0s= x,
                    encodings= encodings,
                    steps= timesteps
                    )
                x = x + delta_timesteps[:, None] * network_outputs # Euler update

        return x
    
    # @torch.compile
    def OT_CFM_Sampling(
        self,
        f0s: torch.FloatTensor,
        reg: float= 0.05
        ):
        noises = torch.randn(
            size= (f0s.size(0) * self.hp.Train.OT_Noise_Multiplier, f0s.size(1)),
            device= f0s.device
            )

        distances = torch.cdist(
            noises.view(noises.size(0), -1),
            f0s.view(f0s.size(0), -1)
            ).detach() ** 2.0
        distances = distances / distances.max() # without normalize_cost, sinkhorn is failed.

        ot_maps = ot.bregman.sinkhorn_log(
            a= torch.full(
                (noises.size(0), ),
                fill_value= 1.0 / noises.size(0),
                device= noises.device
                ),
            b= torch.full(
                (f0s.size(0), ),
                fill_value= 1.0 / f0s.size(0),
                device= f0s.device
                ),
            M= distances,
            reg= reg
            )
        probabilities = ot_maps / ot_maps.sum(dim= 0, keepdim= True)
        noise_indices = torch.multinomial(
            probabilities.T,
            num_samples= 1,
            replacement= True
            )[:, 0]
        sampled_noises = noises[noise_indices]

        return sampled_noises


class Network(torch.nn.Module):
    def __init__(
        self,
        hyper_parameters: Namespace
        ):
        super().__init__()
        self.hp = hyper_parameters

        self.prenet = torch.nn.Sequential(
            Conv_Init(torch.nn.Conv1d(
                in_channels= 1,
                out_channels= self.hp.F0_Predictor.Size,
                kernel_size= 1
                ), w_init_gain= 'gelu'),
            torch.nn.GELU(approximate= 'tanh')
            )

        self.step_ffn = torch.nn.Sequential(
            Step_Embedding(
                embedding_dim= self.hp.F0_Predictor.Size
                ),
            Lambda(lambda x: x.unsqueeze(2)),
            Conv_Init(torch.nn.Conv1d(
                in_channels= self.hp.F0_Predictor.Size,
                out_channels= self.hp.F0_Predictor.Size * 4,
                kernel_size= 1
                ), w_init_gain= 'gelu'),
            torch.nn.GELU(),
            Conv_Init(torch.nn.Conv1d(
                in_channels= self.hp.F0_Predictor.Size * 4,
                out_channels= self.hp.F0_Predictor.Size,
                kernel_size= 1
                ), w_init_gain= 'linear')
            )

        self.residual_blocks = torch.nn.ModuleList([
            Residual_Block(
                in_channels= self.hp.F0_Predictor.Size,
                kernel_size= self.hp.F0_Predictor.Kernel_Size,
                dilation= 2 ** (stack_index % self.hp.F0_Predictor.Dilation_Cycle),
                condition_channels= self.hp.Encoder.Size
                )
            for stack_index in range(self.hp.F0_Predictor.Stack)
            ])
        
        self.projection =  torch.nn.Sequential(
            Conv_Init(torch.nn.Conv1d(
                in_channels= self.hp.F0_Predictor.Size,
                out_channels= self.hp.F0_Predictor.Size,
                kernel_size= 1
                ), w_init_gain= 'gelu'),
            torch.nn.GELU(approximate= 'tanh'),
            Conv_Init(torch.nn.Conv1d(
                in_channels= self.hp.F0_Predictor.Size,
                out_channels= 1,
                kernel_size= 1
                ), w_init_gain= 'gelu')
            )
        torch.nn.init.zeros_(self.projection[-1].weight)    # This is key factor....

    def forward(
        self,
        encodings: torch.FloatTensor,
        f0s: torch.FloatTensor,
        steps: torch.FloatTensor
        ):
        '''
        f0s: [Batch, F0_t]
        encodings: [Batch, Enc_d, Feature_t], condition
        steps: [Batch]
        '''
        x = self.prenet(f0s[:, None, :])    # [Batch, F0_Flow_d, F0_t]        
        steps = self.step_ffn(steps) # [Batch, F0_Flow_d, 1]
        
        skips_list = []
        for residual_block in self.residual_blocks:
            x, skips = residual_block(
                x= x,
                conditions= encodings,
                steps= steps
                )   # [Batch, F0_Flow_d, F0_t], [Batch, F0_Flow_d, F0_t]
            skips_list.append(skips)

        x = torch.stack(skips_list, dim= 0).sum(dim= 0) / math.sqrt(self.hp.F0_Predictor.Stack)
        x = self.projection(x)

        return x[:, 0, :]   # [Batch, F0_t]

class Step_Embedding(torch.nn.Module):
    '''
    sinusoidal for float input
    '''
    def __init__(
        self,
        embedding_dim: int,
        scale: float= 1000.0
        ):
        super().__init__()
        assert embedding_dim % 2 == 0
        self.embedding_dim = embedding_dim
        self.scale = scale

        half_dim = embedding_dim // 2
        div_term = math.log(10000) / (half_dim - 1)
        div_term = torch.exp(torch.arange(half_dim, dtype= torch.int64).float() * -div_term)

        self.register_buffer('div_term', div_term, persistent= False)

    def forward(self, x: torch.Tensor):
        x = self.scale * x[:, None] * self.div_term[None]
        x = torch.cat([x.sin(), x.cos()], dim= 1)

        return x

class Residual_Block(torch.nn.Module):
    def __init__(
        self,
        in_channels: int,
        kernel_size: int,
        dilation: int,
        condition_channels: int
        ):
        super().__init__()
        self.in_channels = in_channels
        
        self.condition = Conv_Init(torch.nn.Conv1d(
            in_channels= condition_channels,
            out_channels= in_channels * 2,
            kernel_size= 1
            ), w_init_gain= 'gate')
        self.step = Conv_Init(torch.nn.Conv1d(
            in_channels= in_channels,
            out_channels= in_channels,
            kernel_size= 1
            ), w_init_gain= 'gelu')

        self.conv = Conv_Init(torch.nn.Conv1d(
            in_channels= in_channels,
            out_channels= in_channels * 2,
            kernel_size= kernel_size,
            padding = dilation * (kernel_size - 1) // 2,
            dilation= dilation,
            ), w_init_gain= 'gate')

        self.projection = Conv_Init(torch.nn.Conv1d(
            in_channels= in_channels,
            out_channels= in_channels * 2,
            kernel_size= 1
            ), w_init_gain= 'linear')

    def forward(
        self,
        x: torch.Tensor,
        conditions: torch.Tensor,
        steps: torch.Tensor
        ):
        residuals = x

        conditions = self.condition(conditions)
        steps = self.step(steps)

        x = self.conv(x + steps) + conditions
        x = Fused_Gate(x)
        
        x = self.projection(x)
        x, skips = x.chunk(chunks= 2, dim= 1)

        return (x + residuals) / math.sqrt(2.0), skips

@torch.jit.script
def Fused_Gate(x):
    x_tanh, x_sigmoid = x.chunk(chunks= 2, dim= 1)
    x = x_tanh.tanh() * x_sigmoid.sigmoid()

    return x
