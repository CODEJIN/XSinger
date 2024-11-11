from argparse import Namespace
import torch
import math
from typing import Union, List, Optional, Tuple

from .Layer import Conv_Init, Embedding_Initialize_, FFT_Block, Norm_Type, Positional_Encoding
from .GRL import GRL
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

        self.diffusion = Diffusion(self.hp)

        self.hificodec = hificodec
        self.hificodec.eval()

    def forward(
        self,
        tokens: torch.IntTensor,
        languages: torch.IntTensor,
        token_lengths: torch.IntTensor,
        token_on_note_lengths: torch.IntTensor,
        notes: torch.IntTensor,
        durations: torch.IntTensor,
        note_lengths: torch.IntTensor,
        singers: torch.IntTensor,
        latent_codes: torch.IntTensor,
        latent_code_lengths: torch.IntTensor,
        ):
        with torch.no_grad():
            latents = self.hificodec.quantizer.embed(latent_codes.mT).detach()

        encodings, singers, cross_attention_alignments = self.encoder(
            tokens= tokens,
            languages= languages,
            token_lengths= token_lengths,
            token_on_note_lengths= token_on_note_lengths,
            notes= notes,
            durations= durations,
            note_lengths= note_lengths,
            singers= singers,
            latent_code_lengths= latent_code_lengths
            )    # [Batch, Enc_d, Dec_t], [Batch, Enc_d, Dec_t], [Batch, Dec_t, Token_t]

        flows, prediction_flows, _, _ = self.diffusion(
            encodings= encodings,
            singers= singers,
            latents= latents,
            lengths= latent_code_lengths,
            )

        return flows, prediction_flows, cross_attention_alignments
    
    def Inference(
        self,
        tokens: torch.IntTensor,
        languages: torch.IntTensor,
        token_lengths: torch.IntTensor,
        token_on_note_lengths: torch.IntTensor,
        notes: torch.IntTensor,
        durations: torch.IntTensor,
        note_lengths: torch.IntTensor,
        singers: torch.IntTensor,
        latent_code_lengths: torch.IntTensor,
        diffusion_steps: int= 16
        ):
        encodings, singers, _ = self.encoder(
            tokens= tokens,
            languages= languages,
            token_lengths= token_lengths,
            token_on_note_lengths= token_on_note_lengths,
            notes= notes,
            durations= durations,
            note_lengths= note_lengths,
            singers= singers,
            latent_code_lengths= latent_code_lengths
            )    # [Batch, Enc_d, Enc_t], [Batch, Enc_d, Dec_t]

        latents = self.diffusion.Inference(
            encodings= encodings,
            singers= singers,
            lengths= latent_code_lengths,
            steps= diffusion_steps,
            )

        # Performing VQ to correct the incomplete predictions of diffusion.
        *_, latent_codes = self.hificodec.quantizer(latents)
        latent_codes = [code.reshape(tokens.size(0), -1) for code in latent_codes]
        latent_codes = torch.stack(latent_codes, 2)
        audios = self.hificodec(latent_codes)[:, 0, :]

        return audios

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

        self.singer_embedding = torch.nn.Embedding(
            num_embeddings= self.hp.Singers,
            embedding_dim= self.hp.Encoder.Size
            )
        Embedding_Initialize_(self.singer_embedding)

        self.lyric_encoder = Lyric_Encoder(self.hp)
        self.melody_encoder = Melody_Encoder(self.hp)
        self.phoneme_to_note_encoder = Phoneme_to_Note_Encoder(self.hp)

        self.positional_encoding = Positional_Encoding(
            num_embeddings= self.hp.Durations,
            embedding_dim= self.hp.Encoder.Size
            )
        self.cross_attention = torch.nn.MultiheadAttention(
            embed_dim= self.hp.Encoder.Size,
            num_heads= self.hp.Encoder.Cross_Attention.Head
            )

        self.prior_encoder = Prior_Encoder(self.hp)

    def forward(
        self,
        tokens: torch.IntTensor,
        languages: torch.IntTensor,
        token_lengths: torch.IntTensor,
        token_on_note_lengths: torch.IntTensor,
        notes: torch.IntTensor,
        durations: torch.IntTensor,
        note_lengths: torch.IntTensor,
        singers: torch.IntTensor,
        latent_code_lengths: torch.IntTensor
        ) -> Tuple[torch.FloatTensor, torch.FloatTensor]:
        '''
        tokens: [Batch, Enc_t]
        languages: [Batch, Enc_t]

        lengths: [Batch], token length
        '''
        singers: torch.FloatTensor = self.singer_embedding(singers).mT # [Batch, Enc_d, Token_t]

        lyric_encodings: torch.FloatTensor = self.lyric_encoder(
            tokens= tokens,
            languages= languages,
            singers= singers,
            lengths= token_lengths
            )   # [Batch, Enc_d, Token_t]
        
        melody_encodings: torch.FloatTensor = self.melody_encoder(
            notes= notes,
            durations= durations,
            lengths= note_lengths,
            )   # [Batch, Enc_d, Note_t]
        
        # average pooling
        token_to_note_alignments = (Length_Regulate(token_on_note_lengths, lyric_encodings.size(2)) / token_on_note_lengths[:, None, :].clamp(1.0))  # [Batch, Token_t, Note_t]
        
        pooled_lyric_encodings: torch.FloatTensor = lyric_encodings @ token_to_note_alignments  # [Batch, Enc_d, Note_t]
        pooled_lyric_encodings: torch.FloatTensor = self.phoneme_to_note_encoder.forward(
            encodings= pooled_lyric_encodings,
            lengths= note_lengths
            )   # [Batch, Enc_d, Note_t]

        lyric_encodings = lyric_encodings + singers # [Batch, Enc_d, Token_t]

        note_to_decoding_alignments = Length_Regulate(durations).mT # [Batch, Note_t, Dec_t]
        melody_encodings = (melody_encodings + pooled_lyric_encodings) @ note_to_decoding_alignments # [Batch, Enc_d, Dec_t]

        lyric_encodings = lyric_encodings + self.positional_encoding(
            position_ids= torch.arange(lyric_encodings.size(2), device= lyric_encodings.device)[None]
            ).mT
        melody_encodings = melody_encodings + self.positional_encoding(
            position_ids= torch.arange(melody_encodings.size(2), device= melody_encodings.device)[None]
            ).mT
        
        encodings, cross_attention_alignments = self.cross_attention(
            query= melody_encodings.permute(2, 0, 1),
            key= lyric_encodings.permute(2, 0, 1),
            value= lyric_encodings.permute(2, 0, 1),
            key_padding_mask= Mask_Generate(
                lengths= token_lengths,
                max_length= torch.ones_like(lyric_encodings[0, 0]).sum()
                )
            )   # [Dec_t, Batch, Enc_d], [Batch, Dec_t, Token_t]
        encodings = encodings.permute(1, 2, 0) # [Batch, Enc_d, Dec_t]
        
        singers = singers @ token_to_note_alignments @ note_to_decoding_alignments  # [Batch, Enc_d, Dec_t]

        encodings: torch.FloatTensor = self.prior_encoder(
            encodings= encodings,
            singers= singers,
            lengths= latent_code_lengths
            )   # [Batch, Enc_d, Dec_t]
  
        return encodings, singers, cross_attention_alignments   # singer: using in diffusion.

class Lyric_Encoder(torch.nn.Module):
    def __init__(
        self,
        hyper_parameters: Namespace
        ):
        super().__init__()
        self.hp = hyper_parameters
        assert self.hp.Encoder.Size % 2 == 0, f'hp.Encoder.Size must be even: {self.hp.Encoder.Size} % 2 != 0'

        self.token_embedding = torch.nn.Embedding(
            num_embeddings= self.hp.Tokens,
            embedding_dim= self.hp.Encoder.Size // 2
            )
        Embedding_Initialize_(self.token_embedding)

        self.language_embedding = torch.nn.Embedding(
            num_embeddings= self.hp.Languages,
            embedding_dim= self.hp.Encoder.Size // 2
            )
        Embedding_Initialize_(self.language_embedding)
        
        self.positional_encoding = Positional_Encoding(
            num_embeddings= self.hp.Durations,
            embedding_dim= self.hp.Encoder.Size
            )

        self.blocks = torch.nn.ModuleList([
            FFT_Block(
                channels= self.hp.Encoder.Size,
                num_head= self.hp.Encoder.Lyric_Encoder.Head,
                residual_conv_stack= self.hp.Encoder.Lyric_Encoder.Residual_Conv.Stack,
                residual_conv_kernel_size= self.hp.Encoder.Lyric_Encoder.Residual_Conv.Kernel_Size,
                ffn_kernel_size= self.hp.Encoder.Lyric_Encoder.FFN.Kernel_Size,
                residual_conv_dropout_rate= self.hp.Encoder.Lyric_Encoder.Residual_Conv.Dropout_Rate,
                ffn_dropout_rate= self.hp.Encoder.Lyric_Encoder.FFN.Dropout_Rate,
                norm_type= Norm_Type.Mixed_LayerNorm if index == 0 else Norm_Type.LayerNorm,
                condition_channels= self.hp.Encoder.Size,
                beta_distribution_concentration= self.hp.Encoder.Lyric_Encoder.Beta_Distribution_Concentration
                )
            for index in range(self.hp.Encoder.Lyric_Encoder.Stack)
            ])  # real type: torch.nn.ModuleList[FFT_BLock]

    def forward(
        self,
        tokens: torch.IntTensor,
        languages: torch.IntTensor,
        singers: torch.FloatTensor,
        lengths: torch.IntTensor,
        ) -> torch.FloatTensor:
        '''
        tokens: [Batch, Enc_t]
        languages: [Batch, Enc_t]
        singers: [Batch, Enc_d, Enc_t]
        lengths: [Batch], token length
        '''        
        masks = None
        float_masks = 1.0
        if not lengths is None:
            masks = Mask_Generate(lengths= lengths, max_length= torch.ones_like(tokens[0]).sum())    # [Batch, Time]
            float_masks = (~masks)[:, None].float()   # float mask, [Batch, 1, X_t]

        tokens = self.token_embedding(tokens)   # [Batch, Token_t, Enc_d / 2]
        languages = self.language_embedding(languages)  # [Batch, Token_t, Enc_d / 2]

        encodings = torch.cat([tokens, languages], dim= 2).mT * float_masks # [Batch, Enc_d, Token_t]
    
        encodings = encodings + self.positional_encoding(
            position_ids= torch.arange(encodings.size(2), device= encodings.device)[None]
            ).mT

        for block in self.blocks:
            encodings = block(
                x= encodings,
                conditions= singers,
                lengths= lengths
                )

        return encodings * float_masks

class Melody_Encoder(torch.nn.Module):
    def __init__(
        self,
        hyper_parameters: Namespace
        ):
        super().__init__()
        self.hp = hyper_parameters
        assert self.hp.Encoder.Size % 2 == 0, f'hp.Encoder.Size must be even: {self.hp.Encoder.Size} % 2 != 0'

        self.note_embedding = torch.nn.Embedding(
            num_embeddings= self.hp.Notes,
            embedding_dim= self.hp.Encoder.Size // 2
            )
        Embedding_Initialize_(self.note_embedding)

        self.duration_embedding = Duration_Positional_Encoding(
            num_embeddings= self.hp.Durations,
            embedding_dim= self.hp.Encoder.Size // 2
            )
        
        self.positional_encoding = Positional_Encoding(
            num_embeddings= self.hp.Durations,
            embedding_dim= self.hp.Encoder.Size
            )

        self.blocks = torch.nn.ModuleList([
            FFT_Block(
                channels= self.hp.Encoder.Size,
                num_head= self.hp.Encoder.Melody_Encoder.Head,
                residual_conv_stack= self.hp.Encoder.Melody_Encoder.Residual_Conv.Stack,
                residual_conv_kernel_size= self.hp.Encoder.Melody_Encoder.Residual_Conv.Kernel_Size,
                ffn_kernel_size= self.hp.Encoder.Melody_Encoder.FFN.Kernel_Size,
                residual_conv_dropout_rate= self.hp.Encoder.Melody_Encoder.Residual_Conv.Dropout_Rate,
                ffn_dropout_rate= self.hp.Encoder.Melody_Encoder.FFN.Dropout_Rate,
                )
            for _ in range(self.hp.Encoder.Melody_Encoder.Stack)
            ])  # real type: torch.nn.ModuleList[FFT_BLock]

    def forward(
        self,
        notes: torch.IntTensor,
        durations: torch.IntTensor,
        lengths: torch.IntTensor,
        ) -> torch.FloatTensor:
        '''
        notes: [Batch, Note_t]
        durations: [Batch, Note_t]
        lengths: [Batch], note length
        '''        
        masks = None
        float_masks = 1.0
        if not lengths is None:
            masks = Mask_Generate(lengths= lengths, max_length= torch.ones_like(notes[0]).sum())    # [Batch, Time]
            float_masks = (~masks)[:, None].float()   # float mask, [Batch, 1, X_t]

        notes = self.note_embedding(notes)   # [Batch, Note_t, Enc_d / 2]
        durations = self.duration_embedding(durations)   # [Batch, Note_t, Enc_d / 2]
        
        encodings = torch.cat([notes, durations], dim= 2).mT * float_masks # [Batch, Enc_d, Token_t]
    
        encodings = encodings + self.positional_encoding(
            position_ids= torch.arange(encodings.size(2), device= encodings.device)[None]
            ).mT

        for block in self.blocks:
            encodings = block(
                x= encodings,
                lengths= lengths
                )

        return encodings * float_masks

class Phoneme_to_Note_Encoder(torch.nn.Module):
    def __init__(
        self,
        hyper_parameters: Namespace
        ):
        super().__init__()
        self.hp = hyper_parameters
        
        self.positional_encoding = Positional_Encoding(
            num_embeddings= self.hp.Durations,
            embedding_dim= self.hp.Encoder.Size
            )
        
        self.blocks = torch.nn.ModuleList([
            FFT_Block(
                channels= self.hp.Encoder.Size,
                num_head= self.hp.Encoder.Phoneme_to_Note_Encoder.Head,
                residual_conv_stack= self.hp.Encoder.Phoneme_to_Note_Encoder.Residual_Conv.Stack,
                residual_conv_kernel_size= self.hp.Encoder.Phoneme_to_Note_Encoder.Residual_Conv.Kernel_Size,
                ffn_kernel_size= self.hp.Encoder.Phoneme_to_Note_Encoder.FFN.Kernel_Size,
                residual_conv_dropout_rate= self.hp.Encoder.Phoneme_to_Note_Encoder.Residual_Conv.Dropout_Rate,
                ffn_dropout_rate= self.hp.Encoder.Phoneme_to_Note_Encoder.FFN.Dropout_Rate,
                )
            for _ in range(self.hp.Encoder.Phoneme_to_Note_Encoder.Stack)
            ])  # real type: torch.nn.ModuleList[FFT_BLock]

    def forward(
        self,
        encodings: torch.FloatTensor,
        lengths: torch.IntTensor
        ) -> torch.FloatTensor:
        '''
        encodings: [Batch, Enc_d, Token_t], pooled lyric encodings
        note_lengths: [Batch]
        '''
        masks = None
        float_masks = 1.0
        if not lengths is None:
            masks = Mask_Generate(lengths= lengths, max_length= torch.ones_like(encodings[0, 0]).sum())    # [Batch, Time]
            float_masks = (~masks)[:, None].float()   # float mask, [Batch, 1, X_t]

        encodings = encodings + self.positional_encoding(
            position_ids= torch.arange(encodings.size(2), device= encodings.device)[None]
            ).mT

        for block in self.blocks:
            encodings = block(
                x= encodings,
                lengths= lengths
                )

        return encodings * float_masks

class Prior_Encoder(torch.nn.Module):
    def __init__(
        self,
        hyper_parameters: Namespace
        ):
        super().__init__()
        self.hp = hyper_parameters
        
        self.positional_encoding = Positional_Encoding(
            num_embeddings= self.hp.Durations,
            embedding_dim= self.hp.Encoder.Size
            )
        
        self.blocks = torch.nn.ModuleList([
            FFT_Block(
                channels= self.hp.Encoder.Size,
                num_head= self.hp.Encoder.Prior_Encoder.Head,
                residual_conv_stack= self.hp.Encoder.Prior_Encoder.Residual_Conv.Stack,
                residual_conv_kernel_size= self.hp.Encoder.Prior_Encoder.Residual_Conv.Kernel_Size,
                ffn_kernel_size= self.hp.Encoder.Prior_Encoder.FFN.Kernel_Size,
                residual_conv_dropout_rate= self.hp.Encoder.Prior_Encoder.Residual_Conv.Dropout_Rate,
                ffn_dropout_rate= self.hp.Encoder.Prior_Encoder.FFN.Dropout_Rate,
                norm_type= Norm_Type.Conditional_LayerNorm,
                condition_channels= self.hp.Encoder.Size
                )
            for _ in range(self.hp.Encoder.Prior_Encoder.Stack)
            ])  # real type: torch.nn.ModuleList[FFT_BLock]

    def forward(
        self,
        encodings: torch.FloatTensor,
        singers: torch.FloatTensor,
        lengths: torch.IntTensor
        ) -> torch.FloatTensor:
        '''
        encodings: [Batch, Enc_d, Latent_Code_t]
        lengths: [Batch], latent_code_lengths
        '''
        encodings = encodings + self.positional_encoding(
            position_ids= torch.arange(encodings.size(2), device= encodings.device)[None]
            ).mT

        for block in self.blocks:
            encodings = block(
                x= encodings,
                conditions= singers,
                lengths= lengths
                )

        return encodings


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

def Length_Regulate(
    durations: torch.IntTensor,
    max_decoding_length: Optional[int]= None
    ) -> torch.FloatTensor:
    repeats = (durations.float() + 0.5).long()
    decoding_lengths = repeats.sum(dim=1)

    max_decoding_length = max_decoding_length or decoding_lengths.max()
    reps_cumsum = torch.cumsum(torch.nn.functional.pad(repeats, (1, 0, 0, 0), value=0.0), dim=1)[:, None, :]

    range_ = torch.arange(max_decoding_length)[None, :, None].to(durations.device)
    alignments = (reps_cumsum[:, :, :-1] <= range_) & (reps_cumsum[:, :, 1:] > range_)

    return alignments.float()

def Mask_Generate(lengths: torch.Tensor, max_length: Union[torch.Tensor, int, None]= None):
    '''
    lengths: [Batch]
    max_lengths: an int value. If None, max_lengths == max(lengths)
    '''
    max_length = max_length or torch.max(lengths)
    sequence = torch.arange(max_length)[None, :].to(lengths.device)
    return sequence >= lengths[:, None]    # [Batch, Time]