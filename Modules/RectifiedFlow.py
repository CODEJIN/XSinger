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

        if self.hp.RectifiedFlow.Scheduler.upper() == 'Uniform'.upper():
            self.timestep_scheduler = lambda x: x
        elif self.hp.RectifiedFlow.Scheduler.upper() == 'Cosmap'.upper():
            self.timestep_scheduler = lambda x: 1.0 - (1 / (torch.tan(torch.pi / 2.0 * x) + 1))
        else:
            raise NotImplementedError(f'Unsupported CFM scheduler: {self.hp.RectifiedFlow.Scheduler}')

    def forward(
        self,
        encodings: torch.FloatTensor,
        mels: torch.FloatTensor,
        lengths: torch.IntTensor
        ):
        '''
        encodings: [Batch, Enc_d, Dec_t]
        mels: [Batch, Mel_d, Mel_t]
        '''        
        if self.hp.RectifiedFlow.Use_CFG:
            cfg_filters = (torch.rand(
                encodings.size(0),
                device= encodings.device
                )[:, None, None] > self.hp.Train.CFG_Alpha).float()  # [Batch, 1, 1]            
            encodings = encodings * cfg_filters
        
        timesteps = self.timestep_scheduler(
            torch.rand(encodings.size(0), device= encodings.device)
            )[:, None, None]

        if self.hp.RectifiedFlow.Use_OT:
            noises = self.OT_CFM_Sampling(mels)
        else:
            noises = torch.randn_like(mels)  # [Batch, Mel_d, Mel_t]

        noised_mels = timesteps * mels + (1.0 - timesteps) * noises  # [Batch, Mel_d, Mel_t]
        # Rectified Flow target: v(x_t, t) = x_1 - x_0
        flows = mels - noises

        # predict flow
        network_outputs = self.network.forward(
            noised_mels= noised_mels,
            encodings= encodings,
            lengths= lengths,
            timesteps= timesteps[:, 0, 0]  # [Batch]
            )
        prediction_flows = network_outputs

        return flows, prediction_flows
    
    def Inference(
        self,
        encodings: torch.FloatTensor,
        lengths: torch.IntTensor,
        steps: int,
        cfg_guidance_scale: Optional[float]= 2.0
        ):  # Euler solver
        x = torch.randn(
            encodings.size(0), self.hp.Sound.N_Mel, encodings.size(2),
            device= encodings.device,
            dtype= encodings.dtype
            )  # [Batch, Mel_d, Mel_t]

        timesteps_list = self.timestep_scheduler(torch.linspace(
            start= 0.0,
            end= 1.0,
            steps= steps + 1,
            device= encodings.device,
            dtype= encodings.dtype
            )) # [Step + 1]
        delta_timesteps_list = timesteps_list[1:] - timesteps_list[:-1]  # [Step]

        timesteps_list = timesteps_list[:-1, None].expand(-1, x.size(0))  # [Step, Batch]
        delta_timesteps_list = delta_timesteps_list[:, None].expand(-1, x.size(0))   # [Batch, Step]

        # Prepare inputs for potential CFG (duplicate batch for unconditional call)
        if self.hp.RectifiedFlow.Use_CFG:
            batch_size = x.size(0)
            x_double = torch.cat([x, x], dim=0)
            encodings_double = torch.cat([encodings, torch.zeros_like(encodings)], dim=0)
            lengths_double = torch.cat([lengths, lengths], dim=0) # lengths are the same for both

        for timesteps, delta_timesteps in zip(timesteps_list, delta_timesteps_list):
            if self.hp.RectifiedFlow.Use_CFG:
                delta_timesteps_double = torch.cat([delta_timesteps, delta_timesteps], dim= 0)
                timesteps_double = torch.cat([timesteps, timesteps], dim=0) # timesteps are the same for both
                network_outputs_double = self.network.forward(
                    noised_mels= x_double,
                    encodings= encodings_double,
                    lengths= lengths_double,
                    timesteps= timesteps_double
                    )
                # Split outputs and apply CFG
                network_outputs_with_condition, network_outputs_without_condition = network_outputs_double.chunk(2, dim=0)
                network_outputs_with_condition = network_outputs_with_condition + cfg_guidance_scale * (network_outputs_with_condition - network_outputs_without_condition)

                # Combine the guided flow for the conditional part and the original flow for the unconditional part
                network_outputs_double = torch.cat([network_outputs_with_condition, network_outputs_without_condition], dim=0)

                # Update x_double for the next step
                x_double = x_double + delta_timesteps_double[:, None, None] * network_outputs_double # Euler update
                x = x_double[:batch_size] # Update x for the next iteration (only the conditional part)
            else:
                network_outputs = self.network.forward(
                    noised_mels= x,
                    encodings= encodings,
                    lengths= lengths,
                    timesteps= timesteps
                    )
                x = x + delta_timesteps[:, None, None] * network_outputs # Euler update

        return x
    
    # @torch.compile
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
                in_channels= self.hp.Sound.N_Mel,
                out_channels= self.hp.RectifiedFlow.Size,
                kernel_size= 1,
                ), w_init_gain= 'gelu'),
            torch.nn.GELU(approximate= 'tanh')
            )        
        self.encoding_ffn = torch.nn.Sequential(
            Conv_Init(torch.nn.Conv1d(
                in_channels= self.hp.Sound.N_Mel,
                out_channels= self.hp.RectifiedFlow.Size * 4,
                kernel_size= 1,
                ), w_init_gain= 'gelu'),
            torch.nn.GELU(approximate= 'tanh'),
            Conv_Init(torch.nn.Conv1d(
                in_channels= self.hp.RectifiedFlow.Size * 4,
                out_channels= self.hp.RectifiedFlow.Size,
                kernel_size= 1,
                ), w_init_gain= 'linear')
            )

        self.step_ffn = torch.nn.Sequential(
            Step_Embedding(
                embedding_dim= self.hp.RectifiedFlow.Size
                ),            
            Conv_Init(torch.nn.Linear(
                in_features= self.hp.RectifiedFlow.Size,
                out_features= self.hp.RectifiedFlow.Size * 4,
                ), w_init_gain= 'gelu'),
            torch.nn.GELU(approximate= 'tanh'),
            Conv_Init(torch.nn.Linear(
                in_features= self.hp.RectifiedFlow.Size * 4,
                out_features= self.hp.RectifiedFlow.Size,
                ), w_init_gain= 'linear')
            )

        self.down_blocks = torch.nn.ModuleList()
        self.mid_blocks = torch.nn.ModuleList()
        self.up_blocks = torch.nn.ModuleList()

        for unet_index in range(self.hp.RectifiedFlow.UNet.Rescale_Stack):
            resnet_block = ResnetBlock(
                in_channels= self.hp.RectifiedFlow.Size * (2 if unet_index == 0 else 1),
                out_channels= self.hp.RectifiedFlow.Size,
                kernel_size= self.hp.RectifiedFlow.UNet.Kernel_Size,
                groups= self.hp.RectifiedFlow.UNet.Groups,
                step_channels= self.hp.RectifiedFlow.Size
                )
            transformer_blocks = torch.nn.ModuleList([
                FFT_Block(
                    channels= self.hp.RectifiedFlow.Size,
                    num_head= self.hp.RectifiedFlow.UNet.Transformer.Head,
                    ffn_kernel_size= self.hp.RectifiedFlow.UNet.Transformer.FFN.Kernel_Size,
                    ffn_dropout_rate= self.hp.RectifiedFlow.UNet.Transformer.FFN.Dropout_Rate,
                    norm_type= Norm_Type.LayerNorm
                    )
                for index in range(self.hp.RectifiedFlow.UNet.Transformer.Stack)
                ])
            downsample_block = Downsample(
                in_channels= self.hp.RectifiedFlow.Size,
                out_channels= self.hp.RectifiedFlow.Size,
                kernel_size= self.hp.RectifiedFlow.UNet.Kernel_Size + (unet_index < self.hp.RectifiedFlow.UNet.Rescale_Stack - 1),
                scale_factor= 2 if unet_index < self.hp.RectifiedFlow.UNet.Rescale_Stack - 1 else 1
                )

            self.down_blocks.append(torch.nn.ModuleList([
                resnet_block,
                transformer_blocks,
                downsample_block
                ]))
            
        for unet_index in range(self.hp.RectifiedFlow.UNet.Mid_Stack):
            resnet_block = ResnetBlock(
                in_channels= self.hp.RectifiedFlow.Size,
                out_channels= self.hp.RectifiedFlow.Size,
                kernel_size= self.hp.RectifiedFlow.UNet.Kernel_Size,
                groups= self.hp.RectifiedFlow.UNet.Groups,
                step_channels= self.hp.RectifiedFlow.Size
                )
            transformer_blocks = torch.nn.ModuleList([
                FFT_Block(
                    channels= self.hp.RectifiedFlow.Size,
                    num_head= self.hp.RectifiedFlow.UNet.Transformer.Head,
                    ffn_kernel_size= self.hp.RectifiedFlow.UNet.Transformer.FFN.Kernel_Size,
                    ffn_dropout_rate= self.hp.RectifiedFlow.UNet.Transformer.FFN.Dropout_Rate,
                    norm_type= Norm_Type.LayerNorm
                    )
                for index in range(self.hp.RectifiedFlow.UNet.Transformer.Stack)
                ])

            self.mid_blocks.append(torch.nn.ModuleList([
                resnet_block,
                transformer_blocks,
                ]))
            
        for unet_index in range(self.hp.RectifiedFlow.UNet.Rescale_Stack):
            resnet_block = ResnetBlock(
                in_channels= self.hp.RectifiedFlow.Size * 2,  # concat with downsample
                out_channels= self.hp.RectifiedFlow.Size,
                kernel_size= self.hp.RectifiedFlow.UNet.Kernel_Size,
                groups= self.hp.RectifiedFlow.UNet.Groups,
                step_channels= self.hp.RectifiedFlow.Size
                )
            transformer_blocks = torch.nn.ModuleList([
                FFT_Block(
                    channels= self.hp.RectifiedFlow.Size,
                    num_head= self.hp.RectifiedFlow.UNet.Transformer.Head,
                    ffn_kernel_size= self.hp.RectifiedFlow.UNet.Transformer.FFN.Kernel_Size,
                    ffn_dropout_rate= self.hp.RectifiedFlow.UNet.Transformer.FFN.Dropout_Rate,
                    norm_type= Norm_Type.LayerNorm
                    )
                for index in range(self.hp.RectifiedFlow.UNet.Transformer.Stack)
                ])
            upsample_block = Upsample(
                in_channels= self.hp.RectifiedFlow.Size,
                out_channels= self.hp.RectifiedFlow.Size,
                kernel_size= self.hp.RectifiedFlow.UNet.Kernel_Size + (unet_index < self.hp.RectifiedFlow.UNet.Rescale_Stack - 1),
                scale_factor= 2 if unet_index < self.hp.RectifiedFlow.UNet.Rescale_Stack - 1 else 1
                )

            self.up_blocks.append(torch.nn.ModuleList([
                resnet_block,
                transformer_blocks,
                upsample_block
                ]))

        self.projection = torch.nn.Sequential(
            Conv_Init(torch.nn.Conv1d(
                in_channels= self.hp.RectifiedFlow.Size,
                out_channels= self.hp.RectifiedFlow.Size,
                kernel_size= 1
                ), w_init_gain= 'gelu'),
            torch.nn.GroupNorm(
                num_groups= self.hp.RectifiedFlow.UNet.Groups,
                num_channels= self.hp.RectifiedFlow.Size,
                ),
            torch.nn.GELU(approximate= 'tanh'),
            Conv_Init(torch.nn.Conv1d(
                in_channels= self.hp.RectifiedFlow.Size,
                out_channels= self.hp.Sound.N_Mel,
                kernel_size= 1), w_init_gain= 'zero'),
            )

    def forward(
        self,
        noised_mels: torch.Tensor,
        encodings: torch.Tensor,
        lengths: torch.Tensor,
        timesteps: torch.Tensor
        ):
        '''
        noised_mels: [Batch, Mel_d, Dec_t]
        encodings: [Batch, Enc_d, Dec_t]
        timesteps: [Batch]
        '''
        assert noised_mels.size(2) % 2 ** (self.hp.RectifiedFlow.UNet.Rescale_Stack - 1) == 0, \
            f'input size must be dividable by \'2 ** (self.hp.RectifiedFlow.UNet.Rescale_Stack - 1)\': {noised_mels.size(2)}'
        assert encodings.size(2) % 2 ** (self.hp.RectifiedFlow.UNet.Rescale_Stack - 1) == 0, \
            f'input size must be dividable by \'2 ** (self.hp.RectifiedFlow.UNet.Rescale_Stack - 1)\': {encodings.size(2)}'

        x = torch.cat(
            [
                self.prenet(noised_mels),
                self.encoding_ffn(encodings)
                ],
            dim= 1
            )   # [Batch, CFM_d * 2, Dec_t]
        timesteps = self.step_ffn(timesteps) # [Batch, Res_d]

        skips_list = []
        for resnet_block, transformer_blocks, downsample_block in self.down_blocks:
            x = resnet_block(
                x= x,
                lengths= lengths,
                timesteps= timesteps
                )
            for transformer_block_index, transformer_block in enumerate(transformer_blocks):
                x = transformer_block(
                    x= x,
                    lengths= lengths
                    )
            skips_list.append((x, lengths))
            x, lengths = downsample_block(x, lengths)

        for resnet_block, transformer_blocks in self.mid_blocks:
            x = resnet_block(
                x= x,
                lengths= lengths,
                timesteps= timesteps
                )
            for transformer_block_index, transformer_block in enumerate(transformer_blocks):
                x = transformer_block(
                    x= x,
                    lengths= lengths
                    )

        for (resnet_block, transformer_blocks, up_block), (skips, skip_lengths) in zip(self.up_blocks, reversed(skips_list)):
            lengths = skip_lengths
            x = resnet_block(
                x= torch.cat([x, skips], dim= 1),
                lengths= lengths,
                timesteps= timesteps
                )
            for transformer_block_index, transformer_block in enumerate(transformer_blocks):
                x = transformer_block(
                    x= x,
                    lengths= lengths
                    )
            x = up_block(x)

        x = self.projection(x)

        return x

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


@torch.jit.script
def Fused_Gate(x):
    x_tanh, x_sigmoid = x.chunk(chunks= 2, dim= 1)
    x = x_tanh.tanh() * x_sigmoid.sigmoid()

    return x


class ResnetBlock(torch.nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        groups: int,
        step_channels: int,
        ):
        super().__init__()

        self.conv_0 = Conv_Init(torch.nn.Conv1d(
            in_channels= in_channels,
            out_channels= out_channels,
            kernel_size= kernel_size,
            padding= (kernel_size - 1) // 2,
            ), w_init_gain= 'gelu')
        self.norm_0 = torch.nn.GroupNorm(
            num_groups= groups,
            num_channels= out_channels
            )
        self.conv_1 = Conv_Init(torch.nn.Conv1d(
            in_channels= out_channels,
            out_channels= out_channels,
            kernel_size= kernel_size,
            padding= (kernel_size - 1) // 2,
            ), w_init_gain= 'gelu')
        self.norm_1 = torch.nn.GroupNorm(
            num_groups= groups,
            num_channels= out_channels
            )
        self.gelu = torch.nn.GELU(approximate= 'tanh')
        
        self.residual = Conv_Init(torch.nn.Conv1d(
            in_channels= in_channels,
            out_channels= out_channels,
            kernel_size= 1
            ), w_init_gain= 'linear')
        self.step = Conv_Init(torch.nn.Linear(
            in_features= step_channels,
            out_features= out_channels
            ), w_init_gain= 'linear')

    def forward(
        self,
        x: torch.FloatTensor,
        lengths: torch.LongTensor,
        timesteps: torch.FloatTensor
        ):
        '''
        x: [Batch, CFM_d, Dec_t]
        lengths: [Batch]
        timesteps: [Batch, step_channels] # Corrected annotation
        '''
        masks = None
        float_masks = 1.0
        if not lengths is None:
            masks = Mask_Generate(lengths= lengths, max_length= x.size(2))    # [Batch, X_t]
            float_masks = (~masks)[:, None, :].float()   # float mask, [Batch, 1, X_t]

        residuals = self.residual(x * float_masks)
        timesteps = self.step(timesteps)

        x = self.conv_0(x * float_masks)
        x = self.norm_0(x)
        x = self.gelu(x)
        x = x + timesteps[:, :, None]
        x = self.conv_1(x * float_masks)
        x = self.norm_1(x)
        x = self.gelu(x)
        x = x + residuals

        return x * float_masks

class Downsample(torch.nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        scale_factor: int= 2
        ):
        super().__init__()
        assert kernel_size >= scale_factor, \
            'kernel_size must be bigger than scale_factor'
        assert (kernel_size - scale_factor) % 2 == 0, \
            '\'kernel_size - scale_factor\' must be an even.'

        self.scale_factor = scale_factor

        self.conv = Conv_Init(torch.nn.Conv1d(
            in_channels= in_channels,
            out_channels= out_channels,
            kernel_size= kernel_size,
            stride= scale_factor,
            padding= (kernel_size - scale_factor) // 2
            ), w_init_gain= 'linear')

    def forward(
        self,
        x: torch.FloatTensor,
        lengths: torch.IntTensor
        ):
        '''
        x: [Batch, CFM_d, Dec_t]
        '''
        assert x.size(2) % self.scale_factor == 0, \
            f'input length must be diviable by scale factor {self.scale_factor}'

        x = self.conv(x)
        lengths = (lengths.float() / self.scale_factor).ceil().to(lengths.dtype)

        return x, lengths

class Upsample(torch.nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        scale_factor: int= 2
        ):
        super().__init__()
        assert kernel_size >= scale_factor, 'kernel_size must be bigger than scale_factor'
        assert (kernel_size - scale_factor) % 2 == 0, '\'kernel_size - scale_factor\' must be an even.'

        self.convtranspose = ConvTranspose_Init(torch.nn.ConvTranspose1d(
            in_channels= in_channels,
            out_channels= out_channels,
            kernel_size= kernel_size,
            stride= scale_factor,
            padding= (kernel_size - scale_factor) // 2
            ), w_init_gain= 'linear')

    def forward(self, x: torch.FloatTensor):
        '''
        x: [Batch, CFM_d, Dec_t]
        '''
        return self.convtranspose(x)