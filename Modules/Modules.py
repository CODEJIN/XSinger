from argparse import Namespace
import torch
import math
from typing import Union, List, Optional, Tuple

from .Layer import Conv_Init, Embedding_Initialize_, FFT_Block, Norm_Type, Lambda
from .RectifiedFlow_F0 import RectifiedFlow as F0_Predictor

class TechSinger_Linear(torch.nn.Module):
    def __init__(
        self,
        hyper_parameters: Namespace,
        mel_mean: float,
        mel_std: float,
        f0_mean: float,
        f0_std: float,
        ):
        super().__init__()
        self.hp = hyper_parameters
        self.mel_mean = mel_mean
        self.mel_std = mel_std
        self.f0_mean = f0_mean
        self.f0_std = f0_std

        self.encoder = Encoder(self.hp)
        self.f0_predictor = F0_Predictor(self.hp)
        self.f0_embedding = torch.nn.Embedding(
            num_embeddings= self.hp.F0_Predictor.Coarse_Bin,
            embedding_dim= self.hp.Encoder.Size
            )
        
        self.tech_predictor = Tech_Predictor(self.hp)
        self.tech_weight = torch.nn.Parameter(torch.empty(self.hp.Techniques, self.hp.Encoder.Size))
        weight_variance = math.sqrt(3.0) * math.sqrt(2.0 / (self.hp.Techniques + self.hp.Encoder.Size))
        self.tech_weight.data.uniform_(-weight_variance, weight_variance)
        
        self.prior_encoder = Prior_Encoder(self.hp)

        self.token_predictor = Token_Predictor(self.hp)

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
        mels: torch.FloatTensor,
        f0s: torch.FloatTensor,
        techs: torch.IntTensor,
        mel_lengths: torch.IntTensor,
        ):
        normalized_mels = (mels - self.mel_mean) / self.mel_std
        normalized_f0s = (f0s - self.f0_mean) / self.f0_std

        encodings, singers, cross_attention_alignments, _ = self.encoder(
            tokens= tokens,
            languages= languages,
            token_lengths= token_lengths,
            token_on_note_lengths= token_on_note_lengths,
            notes= notes,
            durations= durations,
            note_lengths= note_lengths,
            singers= singers,
            mel_lengths= mel_lengths
            )    # [Batch, Enc_d, Dec_t], [Batch, Enc_d, Dec_t], [Batch, Dec_t, Token_t]
        
        f0_flows, prediction_f0_flows = self.f0_predictor(
            encodings= encodings,
            f0s= normalized_f0s
            )   # [Batch, Dec_t], [Batch, Dec_t]
        f0s_coarse = self.F0_Coarse(
            f0s= f0s
            )   # [Batch, Dec_t]
        f0_encodings = self.f0_embedding(f0s_coarse).mT # [Batch, Enc_d, Dec_t]

        prediction_techs = self.tech_predictor(encodings, mel_lengths)
        techs = torch.einsum('bnt,ne->bet', techs.float(), self.tech_weight)
        
        prediction_mels = self.prior_encoder(
            encodings= encodings + f0_encodings + techs,
            lengths= mel_lengths
            )
        
        prediction_tokens = self.token_predictor(
            encodings= prediction_mels
            )

        return \
            normalized_mels, prediction_mels, \
            f0_flows, prediction_f0_flows, \
            prediction_techs, prediction_tokens
    
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
        mel_lengths: torch.IntTensor,
        rectifiedflow_steps: int= 100,
        use_f0_bound: bool= False
        ):
        encodings, singers, cross_attention_alignments, notes_expanded = self.encoder(
            tokens= tokens,
            languages= languages,
            token_lengths= token_lengths,
            token_on_note_lengths= token_on_note_lengths,
            notes= notes,
            durations= durations,
            note_lengths= note_lengths,
            singers= singers,
            mel_lengths= mel_lengths
            )    # [Batch, Enc_d, Enc_t], [Batch, Enc_d, Dec_t]
        
        normalized_f0s = self.f0_predictor.Inference(
            encodings= encodings,
            steps= rectifiedflow_steps
            )        
        f0s = (normalized_f0s * self.f0_std + self.f0_mean).clamp_min(0.0)
        if use_f0_bound:    # need dynamic clipping
            f0s_lower_bound = (440 * 2 ** ((notes_expanded - 3 - 69) / 12))
            f0s_lower_bound = torch.where(notes_expanded > 0, f0s_lower_bound, torch.zeros_like(f0s_lower_bound))
            f0s_upper_bound = 440 * 2 ** ((notes_expanded + 3 - 69) / 12)
            f0s = torch.where(f0s < f0s_lower_bound, f0s_lower_bound, f0s)
            f0s = torch.where(f0s > f0s_upper_bound, f0s_upper_bound, f0s)
        f0s_coarse = self.F0_Coarse(
            f0s= f0s
            )   # [Batch, Dec_t]
        f0_encodings = self.f0_embedding(f0s_coarse).mT

        techs = self.tech_predictor(encodings, mel_lengths)
        techs = (techs > 0.0).float()
        techs = torch.einsum('bnt,ne->bet', techs, self.tech_weight)
        
        prediction_mels = self.prior_encoder(
            encodings= encodings + f0_encodings + techs,
            lengths= mel_lengths
            )
        prediction_mels = prediction_mels * self.mel_std + self.mel_mean

        return prediction_mels, f0s, cross_attention_alignments

    def F0_Coarse(
        self,
        f0s: torch.FloatTensor
        ) -> torch.IntTensor:
        f0_min_mel = 1127.0 * math.log(1.0 + self.hp.Sound.F0_Min / 700.0)
        f0_max_mel = 1127.0 * math.log(1.0 + self.hp.Sound.F0_Max / 700.0)        
        f0s_mel = 1127.0 * (1.0 + f0s / 700.0).log()
        
        f0s_mel[f0s_mel > 0.0] = (f0s_mel[f0s_mel > 0.0] - f0_min_mel) / (f0_max_mel - f0_min_mel) * (self.hp.F0_Predictor.Coarse_Bin - 1)  # 0 to Bin-1
        f0s_mel = f0s_mel.clamp(min= 0.0, max= self.hp.F0_Predictor.Coarse_Bin - 1)
        
        f0s_coarse = (f0s_mel + 0.5).int()
        
        return f0s_coarse


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
        self.phoneme_to_note_cross_encoder = Phoneme_to_Note_Cross_Encoder(self.hp)

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
        mel_lengths: torch.IntTensor
        ) -> Tuple[torch.FloatTensor, torch.FloatTensor, torch.FloatTensor, torch.IntTensor]:
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
        token_to_note_alignments = \
            Length_Regulate(token_on_note_lengths, lyric_encodings.size(2)) / token_on_note_lengths[:, None, :].clamp(1.0)  # [Batch, Token_t, Note_t]

        pooled_lyric_encodings: torch.FloatTensor = lyric_encodings @ token_to_note_alignments  # [Batch, Enc_d, Note_t]
        pooled_lyric_encodings: torch.FloatTensor = self.phoneme_to_note_encoder(
            encodings= pooled_lyric_encodings,
            lengths= note_lengths
            )   # [Batch, Enc_d, Note_t]

        lyric_encodings = lyric_encodings + singers # [Batch, Enc_d, Token_t]

        note_to_decoding_alignments = Length_Regulate(durations).mT # [Batch, Note_t, Dec_t]
        melody_encodings = (melody_encodings + pooled_lyric_encodings) @ note_to_decoding_alignments # [Batch, Enc_d, Dec_t]

        cross_attention_masks = (token_to_note_alignments > 0.0).to(note_to_decoding_alignments.dtype) @ note_to_decoding_alignments
        cross_attention_masks = (cross_attention_masks == 0.0).to(note_to_decoding_alignments.dtype).mT # [Batch, Dec_t, Token_t]
        encodings, cross_attention_alignments = self.phoneme_to_note_cross_encoder(
            melody_encodings= melody_encodings,
            melody_lengths= mel_lengths,
            lyric_encodings= lyric_encodings,
            lyric_lengths= token_lengths,
            cross_attention_masks= cross_attention_masks
            )

        singers = singers @ token_to_note_alignments @ note_to_decoding_alignments  # [Batch, Enc_d, Dec_t]
        
        notes_expaned = notes[:, None, :].to(note_to_decoding_alignments.dtype) @ note_to_decoding_alignments
        notes_expaned = notes_expaned[:, 0, :].to(notes.dtype)  # [Batch, Dec_t], for f0 correction.
  
        return encodings, singers, cross_attention_alignments, notes_expaned

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

        self.blocks = torch.nn.ModuleList([
            FFT_Block(
                channels= self.hp.Encoder.Size,
                num_head= self.hp.Encoder.Lyric_Encoder.Head,
                ffn_kernel_size= self.hp.Encoder.Lyric_Encoder.FFN.Kernel_Size,
                ffn_dropout_rate= self.hp.Encoder.Lyric_Encoder.FFN.Dropout_Rate
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
        
        for block in self.blocks:
            encodings = block(
                x= encodings,
                lengths= lengths,
                layer_norm_conditions= singers,
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
        
        self.blocks = torch.nn.ModuleList([
            FFT_Block(
                channels= self.hp.Encoder.Size,
                num_head= self.hp.Encoder.Melody_Encoder.Head,
                ffn_kernel_size= self.hp.Encoder.Melody_Encoder.FFN.Kernel_Size,
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
        
        self.blocks = torch.nn.ModuleList([
            FFT_Block(
                channels= self.hp.Encoder.Size,
                num_head= self.hp.Encoder.Phoneme_to_Note_Encoder.Head,
                ffn_kernel_size= self.hp.Encoder.Phoneme_to_Note_Encoder.FFN.Kernel_Size,
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

        for block in self.blocks:
            encodings = block(
                x= encodings,
                lengths= lengths
                )

        return encodings * float_masks

class Phoneme_to_Note_Cross_Encoder(torch.nn.Module):
    def __init__(
        self,
        hyper_parameters: Namespace
        ):
        super().__init__()
        self.hp = hyper_parameters
        
        self.blocks = torch.nn.ModuleList([
            FFT_Block(
                channels= self.hp.Encoder.Size,
                num_head= self.hp.Encoder.Phoneme_to_Note_Cross_Encoder.Head,
                ffn_kernel_size= self.hp.Encoder.Phoneme_to_Note_Cross_Encoder.FFN.Kernel_Size,
                ffn_dropout_rate= self.hp.Encoder.Phoneme_to_Note_Cross_Encoder.FFN.Dropout_Rate,
                norm_type= Norm_Type.LayerNorm,
                cross_attention_condition_channels= self.hp.Encoder.Size
                )
            for _ in range(self.hp.Encoder.Phoneme_to_Note_Cross_Encoder.Stack)
            ])  # real type: torch.nn.ModuleList[FFT_BLock]

    def forward(
        self,
        melody_encodings: torch.FloatTensor,
        melody_lengths: torch.IntTensor,
        lyric_encodings: torch.FloatTensor,
        lyric_lengths: torch.IntTensor,
        cross_attention_masks: Optional[torch.FloatTensor]= None,
        ) -> Tuple[torch.FloatTensor, torch.FloatTensor]:
        '''
        melody_encodings: [Batch, Enc_d, Dec_t]
        melody_lengths: [Batch]
        lyric_encodings: [Batch, Enc_d, Token_t]
        lyric_lengths: [Batch]
        '''
        masks = None
        float_masks = 1.0
        if not melody_lengths is None:
            masks = Mask_Generate(lengths= melody_lengths, max_length= torch.ones_like(melody_encodings[0, 0]).sum())    # [Batch, Time]
            float_masks = (~masks)[:, None].float()   # float mask, [Batch, 1, X_t]

        encodings = melody_encodings
        
        cross_alignments = []
        for block in self.blocks:
            cross_alignments.append(block.Get_Cross_Alignments(
                x= encodings,
                cross_attention_conditions= lyric_encodings,
                cross_attention_condition_lengths= lyric_lengths,
                cross_attention_masks= cross_attention_masks,
                average_attn_weights= False
                ))
            encodings = block(
                x= encodings,
                lengths= melody_lengths,
                cross_attention_conditions= lyric_encodings,
                cross_attention_condition_lengths= lyric_lengths,
                cross_attention_masks= cross_attention_masks,
                )
        
        encodings = encodings * float_masks
        cross_alignments = torch.cat(cross_alignments, dim= 1).mean(dim= 1) # [Batch, Melody_d, Lyric_d]

        return encodings, cross_alignments

class Tech_Predictor(torch.nn.Module):
    def __init__(
        self,
        hyper_parameters: Namespace
        ):
        super().__init__()
        self.hp = hyper_parameters
        
        self.blocks = torch.nn.ModuleList([
            FFT_Block(
                channels= self.hp.Encoder.Size,
                num_head= self.hp.Tech_Predictor.Head,
                ffn_kernel_size= self.hp.Tech_Predictor.FFN.Kernel_Size,
                ffn_dropout_rate= self.hp.Tech_Predictor.FFN.Dropout_Rate
                )
            for _ in range(self.hp.Tech_Predictor.Stack)
            ])        

        self.norm = torch.nn.LayerNorm(self.hp.Encoder.Size)
        self.projection = Conv_Init(torch.nn.Conv1d(
            in_channels= self.hp.Encoder.Size,
            out_channels= self.hp.Techniques,
            kernel_size= 1
            ), w_init_gain= 'linear')

    def forward(
        self,
        encodings: torch.FloatTensor,
        lengths: torch.IntTensor
        ) -> torch.FloatTensor:
        '''
        encodings: [Batch, Enc_d, Mel_t]
        lengths: [Batch], mel_lengths
        '''
        for block in self.blocks:
            encodings = block(
                x= encodings,
                lengths= lengths                
                )
        encodings = self.norm(encodings.mT).mT
        techs = self.projection(encodings)

        return techs

class Token_Predictor(torch.nn.Module):
    def __init__(
        self,
        hyper_parameters: Namespace
        ):
        super().__init__()
        self.hp = hyper_parameters
        
        self.lstm = torch.nn.LSTM(
            input_size= self.hp.Sound.N_Mel,
            hidden_size= self.hp.Token_Predictor.Size,
            num_layers= self.hp.Token_Predictor.LSTM.Stack,
            )
        self.lstm_dropout = torch.nn.Dropout(
            p= self.hp.Token_Predictor.LSTM.Dropout_Rate,
            )

        self.projection = Conv_Init(torch.nn.Conv1d(
            in_channels= self.hp.Token_Predictor.Size,
            out_channels= self.hp.Tokens + 1,
            kernel_size= 1,
            ), w_init_gain= 'linear')
            
    def forward(
        self,
        encodings: torch.Tensor,
        ) -> torch.Tensor:
        '''
        features: [Batch, Feature_d, Feature_t], Spectrogram
        lengths: [Batch]
        '''
        encodings = encodings.permute(2, 0, 1)    # [Feature_t, Batch, Enc_d]
        
        self.lstm.flatten_parameters()
        encodings = self.lstm(encodings)[0] # [Feature_t, Batch, LSTM_d]
        
        predictions = self.projection(encodings.permute(1, 2, 0))
        predictions = torch.nn.functional.log_softmax(predictions, dim= 1)

        return predictions

class Prior_Encoder(torch.nn.Module):
    def __init__(
        self,
        hyper_parameters: Namespace
        ):
        super().__init__()
        self.hp = hyper_parameters
        
        self.blocks = torch.nn.ModuleList([
            Prior_Residual_Block(
                channels= self.hp.Encoder.Size,
                kernel_size= self.hp.Prior_Encoder.Kernel_Size,
                dilation= 1
                )
            for _ in range(self.hp.Prior_Encoder.Stack)
            ])        
        
        self.norm = torch.nn.LayerNorm(self.hp.Encoder.Size)
        self.projection = Conv_Init(torch.nn.Conv1d(
            in_channels= self.hp.Encoder.Size,
            out_channels= self.hp.Sound.N_Mel,
            kernel_size= self.hp.Prior_Encoder.Projection_Kernel_Size,
            padding = (self.hp.Prior_Encoder.Projection_Kernel_Size - 1) // 2,
            ), w_init_gain= 'linear')

    def forward(
        self,
        encodings: torch.FloatTensor,
        lengths: torch.IntTensor
        ) -> torch.FloatTensor:
        '''
        encodings: [Batch, Enc_d, Mel_t]
        lengths: [Batch], mel_lengths
        '''
        for block in self.blocks:
            encodings = block(
                x= encodings,
                lengths= lengths                
                )
        encodings = self.norm(encodings.mT).mT
        mels = self.projection(encodings)

        return mels

class Prior_Residual_Block(torch.nn.Module):
    def __init__(
        self,
        channels: int,
        kernel_size: int,
        dilation: int
        ):
        super().__init__()
        self.rescaler = kernel_size ** -0.5
        
        self.norm = torch.nn.LayerNorm(channels)
        self.conv_0 = Conv_Init(torch.nn.Conv1d(
            in_channels= channels,
            out_channels= channels * 2,
            kernel_size= kernel_size,
            padding = dilation * (kernel_size - 1) // 2,
            dilation= dilation,
            ), w_init_gain= 'gelu')
        
        self.gelu = torch.nn.GELU(approximate= 'tanh')
        self.conv_1 = Conv_Init(torch.nn.Conv1d(
            in_channels= channels * 2,
            out_channels= channels,
            kernel_size= 1,
            dilation= dilation,
            ), w_init_gain= 'linear')

    def forward(
        self,
        x: torch.Tensor,
        lengths: torch.Tensor
        ):
        masks = None
        float_masks = 1.0
        if not lengths is None:
            masks = Mask_Generate(lengths= lengths, max_length= x.size(2))    # [Batch, Time]
            float_masks = (~masks)[:, None].float()   # float mask, [Batch, 1, X_t]
        
        residuals = x
        
        x = self.norm(x.mT).mT * float_masks
        x = self.conv_0(x) * float_masks
        x = self.gelu(x * self.rescaler) * float_masks
        x = self.conv_1(x) * float_masks
        x = (x + residuals) * float_masks
        
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