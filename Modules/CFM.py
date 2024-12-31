import torch
import math
from argparse import Namespace
from typing import Optional, List, Dict, Union
from tqdm import tqdm
from torchdiffeq import odeint
import ot

from .Layer import Conv_Init, Lambda, FFT_Block, Norm_Type

class CFM(torch.nn.Module):
    def __init__(
        self,
        hyper_parameters: Namespace
        ):
        super().__init__()
        self.hp = hyper_parameters

        self.network = Network(
            hyper_parameters= self.hp
            )
        
        if self.hp.CFM.Scheduler.upper() == 'Uniform'.upper():
            self.timestep_scheduler = lambda x: x
        elif self.hp.CFM.Scheduler.upper() == 'Cosmap'.upper():
            self.timestep_scheduler = lambda x: 1.0 - (1 / (torch.tan(torch.pi / 2.0 * x) + 1))
        else:
            raise NotImplementedError(f'Unsupported CFM scheduler: {self.hp.CFM.Scheduler}')

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
        if self.hp.CFM.Use_CFG:
            cfg_filters = (torch.rand(
                encodings.size(0),
                device= encodings.device
                )[:, None, None] > self.hp.Train.CFG_Alpha).float()  # [Batch, 1, 1]            
            encodings = encodings * cfg_filters
            singers = singers * cfg_filters

        timesteps = self.timestep_scheduler(
            torch.rand(encodings.size(0), device= encodings.device)
            )[:, None, None]

        noises = torch.randn_like(mels)  # [Batch, Mel_d, Mel_t]
        if self.hp.CFM.Use_OT:
            noises = self.OT_CFM_Sampling(mels)
        else:
            noises = torch.randn_like(mels)  # [Batch, Mel_d, Mel_t]
        
        noised_mels = timesteps * mels + (1.0 - timesteps) * noises  # [Batch, Mel_d, Mel_t]
        flows = mels - noises

        # predict flow
        network_output = self.network.forward(
            noised_mels= noised_mels,
            encodings= encodings,
            singers= singers,
            lengths= lengths,
            timesteps= timesteps[:, 0, 0] # [Batch]
            )        
        prediction_flows = network_output

        return flows, prediction_flows
    
    def Inference(
        self,
        encodings: torch.FloatTensor,
        singers: torch.FloatTensor,
        lengths: torch.IntTensor,
        steps: int,
        cfg_guidance_scale: Optional[float]= 4.0
        ):
        noises = torch.randn(
            encodings.size(0), self.hp.Sound.N_Mel, encodings.size(2),
            device= encodings.device,
            dtype= encodings.dtype
            )  # [Batch, Mel_d, Mel_t]
        
        def ode_func(timesteps: torch.Tensor, noised_mels: torch.Tensor):
            timesteps = timesteps[None].expand(noised_mels.size(0))
            schedule_times = self.timestep_scheduler(timesteps)[:, None, None]

            # predict flow
            network_outputs = self.network.forward(
                noised_mels= noised_mels,
                encodings= encodings,
                singers= singers,
                lengths= lengths,
                timesteps= schedule_times[:, 0, 0] # [Batch]
                )
            if self.hp.CFM.Use_CFG:
                network_outputs_without_condition = self.network.forward(
                    noised_mels= noised_mels,
                    encodings= torch.zeros_like(encodings),
                    singers= torch.zeros_like(singers),
                    lengths= lengths,
                    timesteps= schedule_times[:, 0, 0] # [Batch]
                    )
                network_outputs = network_outputs + cfg_guidance_scale * (network_outputs - network_outputs_without_condition)
            
            prediction_flows = network_outputs

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
    
    @torch.compile
    def OT_CFM_Sampling(
        self,
        mels: torch.FloatTensor,
        reg: float= 0.05
        ):
        noises = torch.randn(
            size= (mels.size(0) * self.hp.Train.OT_Noise_Multiplier, mels.size(1), mels.size(2)),
            device= mels.device
            )

        distances = torch.cdist(
            noises.view(noises.size(0), -1),
            mels.view(mels.size(0), -1)
            ).detach() ** 2.0        
        distances = distances / distances.max() # without normalize_cost, sinkhorn is failed.

        ot_maps = ot.bregman.sinkhorn_log(
            a= torch.full(
                (noises.size(0), ),
                fill_value= 1.0 / noises.size(0),
                device= noises.device
                ),
            b= torch.full(
                (mels.size(0), ),
                fill_value= 1.0 / mels.size(0),
                device= mels.device
                ),
            M= distances,
            reg= reg
            )
        probabilities = ot_maps / ot_maps.sum(dim= 0, keepdim= True)
        noise_indices = torch.cat([
            torch.multinomial(
                probability,
                num_samples= 1,
                replacement= True
                )
            for probability in probabilities.T
            ])

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
                in_channels= self.hp.Sound.N_Mel,
                out_channels= self.hp.CFM.Size,
                kernel_size= 1,
                ), w_init_gain= 'gelu'),
            torch.nn.GELU(approximate= 'tanh')
            )
        
        self.encoding_ffn = torch.nn.Sequential(
            Conv_Init(torch.nn.Conv1d(
                in_channels= self.hp.Sound.N_Mel, # linear prediction mels inserted.
                out_channels= self.hp.CFM.Size * 4,
                kernel_size= 1,
                ), w_init_gain= 'gelu'),
            torch.nn.GELU(approximate= 'tanh'),
            Conv_Init(torch.nn.Conv1d(
                in_channels= self.hp.CFM.Size * 4,
                out_channels= self.hp.CFM.Size,
                kernel_size= 1,
                ), w_init_gain= 'linear')
            )
        
        self.singer_ffn = torch.nn.Sequential(
            Conv_Init(torch.nn.Conv1d(
                in_channels= self.hp.Encoder.Size,
                out_channels= self.hp.CFM.Size * 4,
                kernel_size= 1,
                ), w_init_gain= 'gelu'),
            torch.nn.GELU(approximate= 'tanh'),
            Conv_Init(torch.nn.Conv1d(
                in_channels= self.hp.CFM.Size * 4,
                out_channels= self.hp.CFM.Size,
                kernel_size= 1,
                ), w_init_gain= 'linear')
            )
        
        self.step_ffn = torch.nn.Sequential(
            Step_Embedding(
                embedding_dim= self.hp.CFM.Size
                ),
            Conv_Init(torch.nn.Linear(
                in_features= self.hp.CFM.Size,
                out_features= self.hp.CFM.Size * 4,
                ), w_init_gain= 'gelu'),
            torch.nn.GELU(approximate= 'tanh'),
            Conv_Init(torch.nn.Linear(
                in_features= self.hp.CFM.Size * 4,
                out_features= self.hp.CFM.Size,
                ), w_init_gain= 'linear'),
            Lambda(lambda x: x[:, :, None])
            )

        self.blocks = torch.nn.ModuleList([
            FFT_Block(
                channels= self.hp.CFM.Size,
                num_head= self.hp.CFM.Transformer.Head,
                ffn_kernel_size= self.hp.CFM.Transformer.FFN.Kernel_Size,
                ffn_dropout_rate= self.hp.CFM.Transformer.FFN.Dropout_Rate,
                norm_type= Norm_Type.Conditional_LayerNorm if index == 0 else Norm_Type.LayerNorm,
                layer_norm_condition_channels= self.hp.CFM.Size if index == 0 else None
                )
            for index in range(self.hp.CFM.Transformer.Stack)
            ])  # real type: torch.nn.ModuleList[FFT_BLock]

        self.projection = torch.nn.Sequential(
            Conv_Init(torch.nn.Conv1d(
                in_channels= self.hp.CFM.Size,
                out_channels= self.hp.CFM.Size,
                kernel_size= 1
                ), w_init_gain= 'gelu'),
            torch.nn.GELU(approximate= 'tanh'),
            Conv_Init(torch.nn.Conv1d(
                in_channels= self.hp.CFM.Size,
                out_channels= self.hp.Sound.N_Mel,
                kernel_size= 1), w_init_gain= 'zero'),
            )

    def forward(
        self,
        noised_mels: torch.FloatTensor,
        encodings: torch.FloatTensor,
        singers: torch.FloatTensor,
        lengths: torch.FloatTensor,
        timesteps: torch.FloatTensor
        ):
        '''
        noised_mels: [Batch, Mel_d, Dec_t]
        encodings: [Batch, Enc_d, Dec_t]
        singers: [Batch, Enc_d, Dec_t]
        timesteps: [Batch]
        '''
        x = self.prenet(noised_mels)
        encodings = self.encoding_ffn(encodings)
        singers = self.singer_ffn(singers)
        timesteps = self.step_ffn(timesteps) # [Batch, Res_d, 1]

        conditions = encodings + singers + timesteps

        for block in self.blocks:
            x = block(
                x= x,
                lengths= lengths,
                layer_norm_conditions= conditions
                )

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