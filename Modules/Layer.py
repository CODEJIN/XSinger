import torch
import math
from typing import Optional

class Lambda(torch.nn.Module):
    def __init__(self, lambd):
        super().__init__()
        self.lambd = lambd

    def forward(self, x):
        return self.lambd(x)

class Residual(torch.nn.Module):
    def __init__(self, module):
        super().__init__()
        self.module = module

    def forward(self, *args, **kwargs):
        return self.module(*args, **kwargs)

class RMSNorm(torch.nn.Module):
    def __init__(self, num_features: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.scale = torch.nn.Parameter(torch.ones(num_features))

    def _norm(self, x):
        return x * (x.pow(2.0).mean(dim= 1, keepdim=True) + self.eps).rsqrt()

    def forward(self, x: torch.Tensor):
        '''
        x: [Batch, Dim, Time]
        '''
        output = self._norm(x.float()).to(x.dtype)

        shape = [1, -1] + [1] * (x.ndim - 2)

        return output * self.scale.view(*shape)

class Scale(torch.nn.Module):
    def __init__(
        self,
        condition_features: int,
        *args,
        **kwargs
        ):
        super().__init__(*args, **kwargs)
        self.condition = torch.nn.Sequential(
            Lambda(lambda x: x.unsqueeze(2)),
            Conv_Init(torch.nn.Conv1d(
                in_channels= condition_features,
                out_channels= self.num_features * 2,
                kernel_size= 1,
                ), w_init_gain= 'relu'),
            torch.nn.SiLU()
            )

    def forward(
        self,
        x: torch.Tensor,
        conditions: torch.Tensor
        ):
        x = super().forward(x)
        betas, gammas = self.condition(conditions).chunk(chunks= 2, dim= 1)
        x = (gammas + 1) * x + betas

        return x
    
class Scaling(torch.nn.Module):
    def __init__(
        self,
        channels: int,
        condition_channels: int,
        use_shift: bool= True
        ):
        super().__init__()
        self.use_shift = use_shift

        self.silu = torch.nn.SiLU()
        self.conv = Conv_Init(torch.nn.Conv1d(
            in_channels= condition_channels,
            out_channels= channels * (1 + use_shift),
            kernel_size= 1,
            ), w_init_gain= 'zero')

    def forward(
        self,
        x: torch.Tensor,
        conditions: torch.Tensor
        ):
        conditions = self.silu(conditions)
        conditions = self.conv(conditions)

        if self.use_shift:
            scales, shifts = conditions.chunk(chunks= 2, dim= 1)
        else:
            scales = conditions
            shifts = torch.zeros_like(scales)

        return x * (1 + scales) + shifts

class LightweightConv1d(torch.nn.Module):
    '''
    Args:
        input_size: # of channels of the input and output
        kernel_size: convolution channels
        padding: padding
        num_heads: number of heads used. The weight is of shape
            `(num_heads, 1, kernel_size)`
        weight_softmax: normalize the weight with softmax before the convolution

    Shape:
        Input: BxCxT, i.e. (batch_size, input_size, timesteps)
        Output: BxCxT, i.e. (batch_size, input_size, timesteps)

    Attributes:
        weight: the learnable weights of the module of shape
            `(num_heads, 1, kernel_size)`
        bias: the learnable bias of the module of shape `(input_size)`
    '''

    def __init__(
        self,
        input_size,
        kernel_size=1,
        padding=0,
        num_heads=1,
        weight_softmax=False,
        bias=False,
        weight_dropout=0.0,
        w_init_gain= 'linear'
    ):
        super().__init__()
        self.input_size = input_size
        self.kernel_size = kernel_size
        self.num_heads = num_heads
        self.padding = padding
        self.weight_softmax = weight_softmax
        self.weight = torch.nn.Parameter(torch.Tensor(num_heads, 1, kernel_size))
        self.w_init_gain = w_init_gain

        if bias:
            self.bias = torch.nn.Parameter(torch.Tensor(input_size))
        else:
            self.bias = None
        self.weight_dropout_module = FairseqDropout(
            weight_dropout, module_name=self.__class__.__name__
        )
        self.reset_parameters()

    def reset_parameters(self):
        if self.w_init_gain in ['relu', 'leaky_relu']:
            torch.nn.init.kaiming_uniform_(self.weight, nonlinearity= self.w_init_gain)
        elif self.w_init_gain == 'glu':
            assert self.out_channels % 2 == 0, 'The out_channels of GLU requires even number.'
            torch.nn.init.kaiming_uniform_(self.weight[:self.out_channels // 2], nonlinearity= 'linear')
            torch.nn.init.xavier_uniform_(self.weight[self.out_channels // 2:], gain= torch.nn.init.calculate_gain('sigmoid'))
        else:
            torch.nn.init.xavier_uniform_(self.weight, gain= torch.nn.init.calculate_gain(self.w_init_gain))
        if not self.bias is None:
            torch.nn.init.zeros_(self.bias)

    def forward(self, input):
        """
        input size: B x C x T
        output size: B x C x T
        """
        B, C, T = input.size()
        H = self.num_heads

        weight = self.weight
        if self.weight_softmax:
            weight = weight.softmax(dim=-1)

        weight = self.weight_dropout_module(weight)
        # Merge every C/H entries into the batch dimension (C = self.input_size)
        # B x C x T -> (B * C/H) x H x T
        # One can also expand the weight to C x 1 x K by a factor of C/H
        # and do not reshape the input instead, which is slow though
        input = input.view(-1, H, T)
        output = torch.nn.functional.conv1d(input, weight, padding=self.padding, groups=self.num_heads)
        output = output.view(B, C, T)
        if self.bias is not None:
            output = output + self.bias.view(1, -1, 1)

        return output

class FairseqDropout(torch.nn.Module):
    def __init__(self, p, module_name=None):
        super().__init__()
        self.p = p
        self.module_name = module_name
        self.apply_during_inference = False

    def forward(self, x, inplace: bool = False):
        if self.training or self.apply_during_inference:
            return torch.nn.functional.dropout(x, p=self.p, training=True, inplace=inplace)
        else:
            return x

class FFT_Block(torch.nn.Module):
    def __init__(
        self,
        channels: int,
        num_head: int,        
        residual_conv_stack: int,
        residual_conv_kernel_size: int,        
        ffn_kernel_size: int,
        residual_conv_dropout_rate: float= 0.1,
        ffn_dropout_rate: float= 0.1
        ) -> None:
        super().__init__()

        self.attention = torch.nn.MultiheadAttention(
            embed_dim= channels,
            num_heads= num_head
            )        
        self.attention_norm = torch.nn.LayerNorm(channels)
        
        self.residual_conv_blocks = torch.nn.ModuleList([
            FFN(
                channels= channels,
                kernel_size= residual_conv_kernel_size,
                droput_rate= residual_conv_dropout_rate
                )
            for _ in range(residual_conv_stack)
            ])
        
        self.norm = torch.nn.LayerNorm(channels)
        self.ffn = FFN(
            channels= channels,
            kernel_size= ffn_kernel_size,
            droput_rate= ffn_dropout_rate
            )

    def forward(
        self,
        x: torch.Tensor,
        lengths: Optional[torch.Tensor]= None
        ) -> torch.Tensor:
        '''
        x: [B, X_c, X_t]
        lengths: [B]

        '''
        masks = None
        float_masks = 1.0
        if not lengths is None:
            masks = Mask_Generate(lengths= lengths, max_length= torch.ones_like(x[0, 0]).sum())    # [Batch, Time]
            float_masks = (~masks).unsqueeze(1).float()   # float mask, [Batch, 1, X_t]

        # attention block
        residuals = x_attentions = x.permute(2,0,1).contiguous()
        x_attentions, _ = self.attention(
            query= x_attentions,
            key= x_attentions,
            value= x_attentions,
            key_padding_mask= masks
            )
        x_attentions = self.attention_norm(x_attentions + residuals).permute(1, 2, 0) * float_masks
        
        # conv block
        x_convs = x
        for block in self.residual_conv_blocks:
            x_convs = block(x_convs, float_masks)
        
        residuals = x = self.norm((x_attentions + x_convs).mT).mT * float_masks
        
        # feed forward
        x = self.ffn(x, float_masks)
        
        return x    # [B, X_c, X_t]

class FFN(torch.nn.Module):
    def __init__(
        self,
        channels: int,
        kernel_size: int,
        droput_rate: float= 0.0
        ) -> None:
        super().__init__()
        self.conv_0 = Conv_Init(torch.nn.Conv1d(
            in_channels= channels,
            out_channels= channels,
            kernel_size= kernel_size,
            padding= (kernel_size - 1) // 2,
            bias= False
            ), w_init_gain= 'gelu')
        self.gelu = torch.nn.GELU(approximate= 'tanh')
        self.conv_1 = torch.nn.Conv1d(
            in_channels= channels,
            out_channels= channels,
            kernel_size= kernel_size,
            padding= (kernel_size - 1) // 2,
            bias= False
            )
        self.norm = torch.nn.LayerNorm(channels)
        
        self.dropout = torch.nn.Dropout(p= droput_rate)

    def forward(
        self,
        x: torch.Tensor,
        float_masks: torch.FloatTensor
        ):
        residuals = x

        x = self.conv_0(x * float_masks)
        x = self.gelu(x)
        x = self.dropout(x)
        x = self.conv_1(x * float_masks)
        x = self.dropout(x)
        x = self.norm((x + residuals).mT).mT * float_masks

        return x

class Prompt_Block(FFT_Block):
    def forward(
        self,
        x: torch.FloatTensor,
        prompts: torch.FloatTensor,
        lengths: Optional[torch.Tensor]= None,
        prompt_lengths: Optional[torch.Tensor]= None
        ) -> torch.Tensor:
        '''
        x: [B, X_c, X_t]
        lengths: [B]

        '''
        masks = None
        float_masks = 1.0
        if not lengths is None:
            masks = Mask_Generate(lengths= lengths, max_length= torch.ones_like(x[0, 0]).sum())    # [Batch, Time]
            float_masks = (~masks).unsqueeze(1).float()   # float mask, [Batch, 1, X_t]

        prompt_masks = None
        prompt_float_masks = 1.0
        if not lengths is None:
            prompt_masks = Mask_Generate(lengths= prompt_lengths, max_length= torch.ones_like(prompts[0, 0]).sum())    # [Batch, Time]
            prompt_float_masks = (~prompt_masks).unsqueeze(1).float()   # float mask, [Batch, 1, X_t]

        residuals = x_attentions = x.permute(2,0,1).contiguous()
        prompts = prompts.permute(2, 0, 1).contiguous()
        x_attentions, _ = self.attention(
            query= x_attentions,
            key= prompts,
            value= prompts,
            key_padding_mask= prompt_masks
            )
        x_attentions = self.attention_norm(x_attentions + residuals).permute(1, 2, 0) * float_masks

        # conv block
        x_convs = x
        for block in self.residual_conv_blocks:
            x_convs = block(x_convs, float_masks)
        
        residuals = x = self.norm((x_attentions + x_convs).mT).mT * float_masks
        
        # feed forward
        x = self.ffn(x, float_masks)
        
        return x    # [B, X_c, X_t]



class RotaryPositionalEncoding(torch.nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.channels = channels
        inv_freq = 1.0 / (10000 ** (torch.arange(0, channels, 2).float() / channels))
        self.register_buffer('inv_freq', inv_freq)

    def forward(self, x):
        seq_length = x.size(0)
        t = torch.arange(
            seq_length,
            dtype= self.inv_freq.dtype,
            device=x.device
            )
        freqs = torch.einsum('i,j->ij', t, self.inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)  # (seq_length, dim)
        
        cos_emb = emb.cos().unsqueeze(1)  # (seq_length, 1, dim)
        sin_emb = emb.sin().unsqueeze(1)  # (seq_length, 1, dim)

        x_rotated = x * cos_emb + self.rotate_half(x) * sin_emb

        return x_rotated

    def rotate_half(self, x):
        x1, x2 = x[..., :self.channels // 2], x[..., self.channels // 2:]
        return torch.cat((-x2, x1), dim=-1)


class Positional_Encoding(torch.nn.Module):
    def __init__(
        self,
        num_embeddings: int,
        embedding_dim: int,
        chunks: int= 1
        ):
        super().__init__()
        assert embedding_dim % 2 == 0

        self.embedding_dim = embedding_dim

        positions = torch.arange(num_embeddings)

        half_dim = embedding_dim // (2 * chunks)
        div_term = math.log(10000) / (half_dim - 1)
        div_term = torch.exp(torch.arange(half_dim, dtype= torch.int64).float() * -div_term)
        div_term = positions[:, None] * div_term[None]
        weights = torch.zeros(num_embeddings, embedding_dim)

        chunk_dim = embedding_dim // chunks
        for chunk_index in range(chunks):
            weights[:, chunk_dim * chunk_index + 1:chunk_dim * (chunk_index + 1) + 0:2] = div_term.sin()
            weights[:, chunk_dim * chunk_index + 1:chunk_dim * (chunk_index + 1) + 0:2] = div_term.cos()
        self.register_buffer('weights', weights, persistent= False)

    @torch.no_grad()
    def forward(self, position_ids: torch.Tensor):
        batch_size, length = position_ids.size()

        encodings = self.weights.index_select(dim= 0, index= position_ids.view(-1))

        return encodings.view(batch_size, length, self.embedding_dim).detach()  # [Batch, Length, Dim]
    

class Learnable_Positional_Encoding(torch.nn.Module):
    def __init__(
        self,
        num_embeddings: int,
        embedding_dim: int,
        chunks: int= 1
        ) -> None:
        super().__init__()
        assert embedding_dim % 2 == 0

        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.chunks = chunks

        positions = torch.arange(num_embeddings)

        half_dim = embedding_dim // (2 * chunks)
        div_term = math.log(10000) / (half_dim - 1)
        div_term = torch.exp(torch.arange(half_dim, dtype= torch.int64).float() * -div_term)
        
        self.register_buffer('div_term', div_term, persistent= False)
        self.register_buffer('position', positions, persistent= False)

        self.base = torch.nn.Parameter(torch.ones(1))
        self.alpha = torch.nn.Parameter(torch.zeros(1))
        self.beta = torch.nn.Parameter(torch.zeros(1))

    def forward(self, position_ids: torch.Tensor):
        batch_size, lengths = position_ids.size()

        div_term_sin = (self.position[:, None] * (self.base * self.div_term + self.alpha)[None]).sin()
        div_term_cos = (self.position[:, None] * (self.base * self.div_term + self.beta)[None]).cos()
        weights = torch.zeros(
            self.num_embeddings,
            self.embedding_dim,
            device= position_ids.device
            )

        chunk_dim = self.embedding_dim // self.chunks
        for chunk_index in range(self.chunks):
            weights[:, chunk_dim * chunk_index + 1:chunk_dim * (chunk_index + 1) + 0:2] = div_term_sin
            weights[:, chunk_dim * chunk_index + 1:chunk_dim * (chunk_index + 1) + 0:2] = div_term_cos
        encodings = weights.index_select(dim= 0, index= position_ids.view(-1))

        return encodings.view(batch_size, lengths, self.embedding_dim)  # [Batch, Length, Dim]



def Mask_Generate(lengths: torch.Tensor, max_length: int= None):
    '''
    lengths: [Batch]
    max_lengths: an int value. If None, max_lengths == max(lengths)
    '''
    max_length = max_length or torch.max(lengths)
    sequence = torch.arange(max_length, device= lengths.device)[None, :]
    return sequence >= lengths[:, None]    # [Batch, Time]


def Conv_Init(
    module: torch.nn.Module,
    w_init_gain: str
    ):
    if w_init_gain in ['zero']:
        torch.nn.init.zeros_(module.weight)
    elif w_init_gain in ['relu', 'leaky_relu']:
        torch.nn.init.kaiming_uniform_(module.weight, nonlinearity= w_init_gain)
    elif w_init_gain == 'glu':
        assert module.out_channels % 2 == 0, 'The out_channels of GLU requires even number.'
        torch.nn.init.kaiming_uniform_(module.weight[:module.out_channels // 2], nonlinearity= 'linear')
        torch.nn.init.xavier_uniform_(module.weight[module.out_channels // 2:], gain= torch.nn.init.calculate_gain('sigmoid'))
    elif w_init_gain == 'gate':
        assert module.out_channels % 2 == 0, 'The out_channels of GLU requires even number.'
        torch.nn.init.xavier_uniform_(module.weight[:module.out_channels // 2], gain= torch.nn.init.calculate_gain('tanh'))
        torch.nn.init.xavier_uniform_(module.weight[module.out_channels // 2:], gain= torch.nn.init.calculate_gain('sigmoid'))
    else:
        try:
            torch.nn.init.xavier_uniform_(module.weight, gain= torch.nn.init.calculate_gain(w_init_gain))
        except ValueError:
            torch.nn.init.xavier_uniform_(module.weight)

    if not module.bias is None:
        torch.nn.init.zeros_(module.bias)

    return module

def ConvTranspose_Init(
    module: torch.nn.Module,
    w_init_gain: str
    ):
    if w_init_gain in ['zero']:
        torch.nn.init.zeros_(module.weight)
    elif w_init_gain in ['relu', 'leaky_relu']:
        torch.nn.init.kaiming_uniform_(module.weight, nonlinearity= w_init_gain)
    elif w_init_gain == 'glu':
        assert module.out_channels % 2 == 0, 'The out_channels of GLU requires even number.'
        torch.nn.init.kaiming_uniform_(module.weight[:, :module.out_channels // 2], nonlinearity= 'linear')
        torch.nn.init.xavier_uniform_(module.weight[:, module.out_channels // 2:], gain= torch.nn.init.calculate_gain('sigmoid'))
    elif w_init_gain == 'gate':
        assert module.out_channels % 2 == 0, 'The out_channels of GLU requires even number.'
        torch.nn.init.xavier_uniform_(module.weight[:, :module.out_channels // 2], gain= torch.nn.init.calculate_gain('tanh'))
        torch.nn.init.xavier_uniform_(module.weight[:, module.out_channels // 2:], gain= torch.nn.init.calculate_gain('sigmoid'))
    else:
        try:
            torch.nn.init.xavier_uniform_(module.weight, gain= torch.nn.init.calculate_gain(w_init_gain))
        except ValueError:
            torch.nn.init.xavier_uniform_(module.weight)
    if not module.bias is None:
        torch.nn.init.zeros_(module.bias)

    return module

def Embedding_Initialize_(
    embedding_module: torch.nn.Embedding
    ) -> None:
    embedding_variance = math.sqrt(3.0) * math.sqrt(2.0 / (embedding_module.num_embeddings + embedding_module.embedding_dim))
    embedding_module.weight.data.uniform_(-embedding_variance, embedding_variance)
