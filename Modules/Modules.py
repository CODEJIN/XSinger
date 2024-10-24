from argparse import Namespace
import torch
import math
from typing import Union, List, Optional

from .Layer import Conv_Init, Embedding_Initialize_, FFT_Block, Positional_Encoding
from .GRL import GRL
from .F0_Diffusion import F0_Diffusion
from .Diffusion import Diffusion

from hificodec.vqvae import VQVAE

class RectifiedFlowSVS(torch.nn.Module):
    def __init__(
        self,
        hyper_parameters: Namespace,
        hificodec: VQVAE
        ):
        super().__init__()
        self.hp = hyper_parameters

        self.encoder = Encoder(self.hp)

        self.singer_eliminator = Eliminator(self.hp, self.hp.Singers)

        self.f0_diffusion = F0_Diffusion(self.hp)
        self.diffusion = Diffusion(self.hp)

        self.hificodec = hificodec
        self.hificodec.eval()

    def forward(
        self,
        tokens: torch.IntTensor,
        notes: torch.IntTensor,
        lyric_durations: torch.IntTensor,
        note_durations: torch.IntTensor,
        lengths: torch.IntTensor,
        singers: torch.IntTensor,
        languages: torch.IntTensor,
        f0s: torch.FloatTensor,
        latent_codes: torch.LongTensor,
        ):
        with torch.no_grad():
            latents = self.hificodec.quantizer.embed(latent_codes.mT).detach()

        encodings = self.encoder(
            tokens= tokens,
            notes= notes,
            lyric_durations= lyric_durations,
            note_durations= note_durations,
            lengths= lengths,
            singers= singers,
            languages= languages
            )    # [Batch, Enc_d, Enc_t]

        prediction_singers = self.singer_eliminator(encodings)

        f0_flows, prediction_f0_flows, _, _ = self.f0_diffusion(
            encodings= encodings,
            f0s= f0s,
            lengths= lengths,
            )

        flows, prediction_flows, _, _ = self.diffusion(
            encodings= encodings,
            f0s= f0s,
            latents= latents,
            lengths= lengths,
            )

        return flows, prediction_flows, f0_flows, prediction_f0_flows, prediction_singers
    
    def Inference(
        self,
        tokens: torch.IntTensor,
        notes: torch.IntTensor,
        lyric_durations: torch.IntTensor,
        note_durations: torch.IntTensor,
        lengths: torch.IntTensor,
        singers: torch.IntTensor,
        languages: torch.IntTensor,
        diffusion_steps: int= 16
        ):
        encodings = self.encoder(
            tokens= tokens,
            notes= notes,
            lyric_durations= lyric_durations,
            note_durations= note_durations,
            lengths= lengths,
            singers= singers,
            languages= languages
            )    # [Batch, Enc_d, Enc_t]

        f0s = self.f0_diffusion.Inference(
            encodings= encodings,
            lengths= lengths,
            steps= diffusion_steps,
            )   # [Batch, Dec_t]

        latents = self.diffusion.Inference(
            encodings= encodings,
            f0s= f0s,
            lengths= lengths,
            steps= diffusion_steps,
            )

        # Performing VQ to correct the incomplete predictions of diffusion.
        *_, latent_codes = self.hificodec.quantizer(latents)
        latent_codes = [code.reshape(tokens.size(0), -1) for code in latent_codes]
        latent_codes = torch.stack(latent_codes, 2)
        audios = self.hificodec(latent_codes)[:, 0, :]

        return audios, f0s

    def train(self, mode: bool= True):
        super().train(mode= mode)
        self.hificodec.eval()


class Encoder(torch.nn.Module):
    def __init__(
        self,
        hyper_parameters: Namespace
        ):
        super().__init__()
        self.hp = hyper_parameters

        self.token_embedding = torch.nn.Embedding(
            num_embeddings= self.hp.Tokens,
            embedding_dim= self.hp.Encoder.Size
            )
        Embedding_Initialize_(self.token_embedding)

        self.note_embedding = torch.nn.Embedding(
            num_embeddings= self.hp.Notes,
            embedding_dim= self.hp.Encoder.Size
            )
        Embedding_Initialize_(self.note_embedding)

        self.lyric_duration_embedding = Duration_Positional_Encoding(
            num_embeddings= self.hp.Durations,
            embedding_dim= self.hp.Encoder.Size
            )

        self.note_duration_embedding = Duration_Positional_Encoding(
            num_embeddings= self.hp.Durations,
            embedding_dim= self.hp.Encoder.Size
            )
        
        self.singer_embedding = torch.nn.Embedding(
            num_embeddings= self.hp.Singers,
            embedding_dim= self.hp.Encoder.Size
            )
        Embedding_Initialize_(self.singer_embedding)
        
        self.language_embedding = torch.nn.Embedding(
            num_embeddings= self.hp.Languages,
            embedding_dim= self.hp.Encoder.Size
            )
        Embedding_Initialize_(self.language_embedding)
        
        self.blocks: List[FFT_Block] = torch.nn.ModuleList([
            FFT_Block(
                channels= self.hp.Encoder.Size,
                num_head= self.hp.Encoder.Transformer.Head,
                residual_conv_stack= self.hp.Encoder.Transformer.Residual_Conv.Stack,
                residual_conv_kernel_size= self.hp.Encoder.Transformer.Residual_Conv.Kernel_Size,
                ffn_kernel_size= self.hp.Encoder.Transformer.FFN.Kernel_Size,
                residual_conv_dropout_rate= self.hp.Encoder.Transformer.Residual_Conv.Dropout_Rate,
                ffn_dropout_rate= self.hp.Encoder.Transformer.FFN.Dropout_Rate,
                )
            for _ in range(self.hp.Encoder.Transformer.Stack)
            ])  # real type: torch.nn.ModuleList[FFT_BLock]

    def forward(
        self,
        tokens: torch.Tensor,
        notes: torch.Tensor,
        lyric_durations: torch.Tensor,
        note_durations: torch.Tensor,
        lengths: torch.Tensor,
        singers: torch.Tensor,
        languages: torch.Tensor,
        ) -> torch.FloatTensor:
        '''
        tokens: [Batch, Enc_t]
        languages: [Batch] or [Batch, Enc_t]
        lengths: [Batch], token length
        '''        
        masks = None
        float_masks = 1.0
        if not lengths is None:
            masks = Mask_Generate(lengths= lengths, max_length= torch.ones_like(tokens[0]).sum())    # [Batch, Time]
            float_masks = (~masks)[:, None].float()   # float mask, [Batch, 1, X_t]

        encodings = \
            self.token_embedding(tokens) + \
            self.note_embedding(notes) + \
            self.lyric_duration_embedding(lyric_durations) + \
            self.note_duration_embedding(note_durations) + \
            self.singer_embedding(singers)[:, None, :] + \
            self.language_embedding(languages)[:, None, :]        
        encodings = encodings.mT    # [Batch, Enc_d, Enc_t]

        for block in self.blocks:
            encodings = block(encodings, lengths)

        return encodings

class Eliminator(torch.nn.Module): 
    def __init__(
        self,
        hyper_parameters: Namespace,
        num_classes: int
        ):
        super().__init__()
        self.hp = hyper_parameters
        
        self.grl = GRL()

        self.conv_0 = Conv_Init(torch.nn.Conv1d(
            in_channels= self.hp.Encoder.Size,
            out_channels= self.hp.Eliminator.Size,
            kernel_size= self.hp.Eliminator.Kernel_Size
            ), w_init_gain= 'relu')
        self.norm_0 = torch.nn.LayerNorm(self.hp.Eliminator.Size)

        self.conv_1 = Conv_Init(torch.nn.Conv1d(
            in_channels= self.hp.Eliminator.Size,
            out_channels= self.hp.Eliminator.Size,
            kernel_size= self.hp.Eliminator.Kernel_Size
            ), w_init_gain= 'relu')
        self.norm_1 = torch.nn.LayerNorm(self.hp.Eliminator.Size)

        self.conv_2 = Conv_Init(torch.nn.Conv1d(
            in_channels= self.hp.Eliminator.Size,
            out_channels= self.hp.Eliminator.Size,
            kernel_size= self.hp.Eliminator.Kernel_Size
            ), w_init_gain= 'relu')
        self.norm_2 = torch.nn.LayerNorm(self.hp.Eliminator.Size)

        self.projection = Conv_Init(torch.nn.Linear(
            in_features= self.hp.Eliminator.Size,
            out_features= num_classes
            ), w_init_gain= 'linear')

        self.gelu = torch.nn.GELU(approximate= 'tanh')
            
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.grl(x)
        x = self.conv_0(x)
        x = self.norm_0(x.mT).mT
        x = self.gelu(x)
        x = self.conv_1(x)
        x = self.norm_1(x.mT).mT
        x = self.gelu(x)
        x = self.conv_2(x)
        x = self.norm_2(x.mT).mT
        x = self.gelu(x).mean(dim= 2)
        x = self.projection(x)

        return x

# https://pytorch.org/tutorials/beginner/transformer_tutorial.html
# https://github.com/soobinseo/Transformer-TTS/blob/master/network.py
class Duration_Positional_Encoding(torch.nn.Embedding):
    def __init__(
        self,        
        num_embeddings: int,
        embedding_dim: int,
        ):        
        positional_embedding = torch.zeros(num_embeddings, embedding_dim)
        position = torch.arange(0, num_embeddings, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, embedding_dim, 2).float() * (-math.log(10000.0) / embedding_dim))
        positional_embedding[:, 0::2] = torch.sin(position * div_term)
        positional_embedding[:, 1::2] = torch.cos(position * div_term)
        super().__init__(
            num_embeddings= num_embeddings,
            embedding_dim= embedding_dim,
            _weight= positional_embedding
            )
        self.weight.requires_grad = False

        self.alpha = torch.nn.Parameter(
            data= torch.ones(1) * 0.01,
            requires_grad= True
            )

    def forward(self, durations):
        '''
        durations: [Batch, Length]
        '''
        return self.alpha * super().forward(durations)  # [Batch, Dim, Length]

    @torch.jit.script
    def get_pe(x: torch.Tensor, pe: torch.Tensor):
        pe = pe.repeat(1, 1, math.ceil(x.size(2) / pe.size(2))) 
        return pe[:, :, :x.size(2)]


def Mask_Generate(lengths: torch.Tensor, max_length: Union[torch.Tensor, int, None]= None):
    '''
    lengths: [Batch]
    max_lengths: an int value. If None, max_lengths == max(lengths)
    '''
    max_length = max_length or torch.max(lengths)
    sequence = torch.arange(max_length)[None, :].to(lengths.device)
    return sequence >= lengths[:, None]    # [Batch, Time]