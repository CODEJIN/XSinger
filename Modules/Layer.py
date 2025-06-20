import torch
import math
from enum import Enum
from typing import Optional

from einops import einsum, rearrange

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

class Conditional_LayerNorm(torch.nn.LayerNorm):
    def __init__(
        self,
        channels: int,
        condition_channels: int,
        eps: float= 1e-5,
        bias: bool= True,
        device= None,
        dtype= None
        ):
        super().__init__(
            normalized_shape= channels,
            eps= eps,
            elementwise_affine= False,
            bias= bias,
            device= device,
            dtype= dtype,
            )
        
        self.affine = Conv_Init(torch.nn.Linear(
            in_features= condition_channels,
            out_features= channels * 2,
            bias= True
            ), w_init_gain= 'linear')
        self.affine.bias.data[:channels] = 1
        self.affine.bias.data[channels:] = 0        

    def forward(
        self,
        x: torch.FloatTensor,
        conditions: torch.FloatTensor
        ) -> torch.FloatTensor:
        '''
        x: [Batch, Time, X_d],
        conditions: [Batch, Time or 1, Cond_d]
        '''
        x = super().forward(x)  # [Batch, Time, X_d]

        gammas, betas = self.affine(conditions).chunk(chunks= 2, dim= 2)    # [Batch, Time or 1, X_d] * 2
        x = gammas * x + betas

        return x

class Mixed_LayerNorm(torch.nn.LayerNorm):
    def __init__(
        self,
        channels: int,
        condition_channels: int,
        beta_distribution_concentration: float,
        eps: float= 1e-5,
        bias: bool= True,
        device= None,
        dtype= None
        ):
        super().__init__(
            normalized_shape= channels,
            eps= eps,
            elementwise_affine= False,
            bias= bias,
            device= device,
            dtype= dtype,
            )
        
        self.beta_distribution = torch.distributions.Beta(
            beta_distribution_concentration,
            beta_distribution_concentration
            )

        self.affine = Conv_Init(torch.nn.Linear(
            in_features= condition_channels,
            out_features= channels * 2,
            bias= True
            ), w_init_gain= 'linear')
        self.affine.bias.data[:channels] = 1
        self.affine.bias.data[channels:] = 0        

    def forward(
        self,
        x: torch.FloatTensor,
        conditions: torch.FloatTensor
        ) -> torch.FloatTensor:
        '''
        x: [Batch, Time, X_d],
        conditions: [Batch, Time, Cond_d]
        '''
        x = super().forward(x)  # [Batch, Time, X_d]

        betas, gammas = self.affine(conditions).chunk(chunks= 2, dim= 2)    # [Batch, Time, X_d] * 2
        
        if not self.training:
            return gammas * x + betas

        suffile_indices = torch.randperm(conditions.size(1))
        shuffled_betas = betas[:, suffile_indices, :]
        shuffled_gammas = gammas[:, suffile_indices, :]

        beta_samples = self.beta_distribution.sample((x.size(0), 1, 1)).to(x.device) # [Batch, 1, 1]
        
        mixed_betas = beta_samples * betas + (1 - beta_samples) * shuffled_betas
        mixed_gammas = beta_samples * gammas + (1 - beta_samples) * shuffled_gammas

        x = mixed_gammas * x + mixed_betas

        return x

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

class Norm_Type(Enum):
    LayerNorm= 0
    Conditional_LayerNorm= 1
    Mixed_LayerNorm= 2


class FFT_Block(torch.nn.Module):
    def __init__(
        self,
        channels: int,
        num_head: int,        
        ffn_kernel_size: int,
        ffn_dropout_rate: float= 0.1,
        norm_type: Norm_Type= Norm_Type.LayerNorm,
        use_prenorm: bool= True,
        cross_attention_condition_channels: Optional[float]= None,
        layer_norm_condition_channels: Optional[float]= None,
        beta_distribution_concentration: Optional[float]= None
        ) -> None:
        '''
        use_prenorm
            - The order of residual addition and normalization.
            - If True, 'x_attn = LN(Attention(x)) + x' and 'x_out = LN(FFN(x_attn)) + x_attn'.
            - If False, 'x_attn = LN(Attention(x) + x)' and 'x_out = LN(FFN(x_attn) + x_attn)'.
        cross_attention_condition_channels
            - If None, only self-attention is calculated.
                - 'Attention(x) = Self_Attention(x)'
            - If int, both self and cross-attention are calculated in parallel.
                - 'Attention(x) = LN(Self_Attention(x)) + LN(Cross_Attention(x, cond))'
                - LN is applied to each attention mechanism.
        layer_norm_condition_channels
            - If the norm_type is LayerNorm, this parameter is ignored.
            - Only Conditional_LayerNorm or Mixed_LayerNorm can be used with a layer norm condition.
        beta_distribution_concentration
            - If the norm_type isn't Mixed_LayerNorm, this parameter is ignored.
            - Mixed_LayerNorm can only be used with beta_distribution_concentration.
        '''
        super().__init__()
        self.norm_type = norm_type
        self.use_prenorm = use_prenorm
        self.use_cross_attention_condition = not cross_attention_condition_channels is None
        self.use_layer_norm_condition = not layer_norm_condition_channels is None

        if all([
            not self.use_layer_norm_condition,
            norm_type in [Norm_Type.Conditional_LayerNorm, Norm_Type.Mixed_LayerNorm]
            ]):
            raise ValueError('CLN or MLN only can be used with layer norm condition.')
        
        self.self_attention = MultiHeadAttentionWithRoPE(
            embed_dim= channels,
            num_heads= num_head
            )

        if norm_type == Norm_Type.LayerNorm:
            self.residual_norm = torch.nn.LayerNorm(channels)
        elif norm_type == Norm_Type.Conditional_LayerNorm:
            self.residual_norm = Conditional_LayerNorm(
                channels= channels,
                condition_channels= layer_norm_condition_channels
                )
        elif norm_type == Norm_Type.Mixed_LayerNorm:
            self.residual_norm = Mixed_LayerNorm(
                channels= channels,
                condition_channels= layer_norm_condition_channels,
                beta_distribution_concentration= beta_distribution_concentration
                )
            
        if self.use_cross_attention_condition:
            self.cross_attention = MultiHeadAttentionWithRoPE(
                embed_dim= channels,
                num_heads= num_head,
                k_dim= cross_attention_condition_channels,
                v_dim= cross_attention_condition_channels
                )
            
            if norm_type == Norm_Type.LayerNorm:
                self.self_attention_norm = torch.nn.LayerNorm(channels)
                self.cross_attention_norm = torch.nn.LayerNorm(channels)
            elif norm_type == Norm_Type.Conditional_LayerNorm:
                self.self_attention_norm = Conditional_LayerNorm(
                    channels= channels,
                    condition_channels= layer_norm_condition_channels
                    )
                self.cross_attention_norm = Conditional_LayerNorm(
                    channels= channels,
                    condition_channels= layer_norm_condition_channels
                    )
            elif norm_type == Norm_Type.Mixed_LayerNorm:
                self.self_attention_norm = Mixed_LayerNorm(
                    channels= channels,
                    condition_channels= layer_norm_condition_channels,
                    beta_distribution_concentration= beta_distribution_concentration
                    )
                self.cross_attention_norm = Mixed_LayerNorm(
                    channels= channels,
                    condition_channels= layer_norm_condition_channels,
                    beta_distribution_concentration= beta_distribution_concentration
                    )

        self.ffn = FFN(
            channels= channels,
            kernel_size= ffn_kernel_size,
            droput_rate= ffn_dropout_rate,
            norm_type= norm_type,
            use_prenorm= use_prenorm,
            condition_channels= layer_norm_condition_channels,
            beta_distribution_concentration= beta_distribution_concentration
            )

    def forward(
        self,
        x: torch.FloatTensor,
        lengths: Optional[torch.IntTensor]= None,
        cross_attention_conditions: Optional[torch.FloatTensor]= None,
        cross_attention_condition_lengths: Optional[torch.IntTensor]= None,
        layer_norm_conditions: Optional[torch.FloatTensor]= None,
        cross_attention_masks: Optional[torch.FloatTensor]= None,   # masking is 1.0, don't use boolean.
        ) -> torch.FloatTensor:
        '''
        x: [B, X_d, X_t]
        lengths: [B]
        cross_attention_conditions: [B, CAC_d, CAC_t]
        cross_attention_condition_lengths: [B]
        layer_norm_conditions: [B, LNC_d, X_t or 1], X_t-> local condition, 1 -> global condition
        cross_attention_masks: [B, X_t, CAC_t]
        '''
        if not layer_norm_conditions is None:
            layer_norm_conditions = layer_norm_conditions.mT    # [B, X_t or 1, LNC_d]

        masks = None
        float_masks = 1.0
        if not lengths is None:
            masks = Mask_Generate(lengths= lengths, max_length= torch.ones_like(x[0, 0]).sum())    # [Batch, X_t]
            float_masks = (~masks)[:, None, :].float()   # float mask, [Batch, 1, X_t]

        residuals = x.mT    # [Batch, X_t, X_d]
        x_self_attentions = x_cross_attentions = x.permute(2, 0, 1).contiguous()    # [X_t, Batch, X_d]
        x_self_attentions, _ = self.self_attention(
            query= x_self_attentions,
            key= x_self_attentions,
            value= x_self_attentions,
            key_padding_mask= masks
            )   # [X_t, Batch, X_d]
        x_self_attentions = x_self_attentions.permute(1, 0, 2)  # [Batch, X_t, X_d]
        
        if self.use_cross_attention_condition:
            cross_attention_condition_masks = None
            if not cross_attention_condition_lengths is None:
                cross_attention_condition_masks = Mask_Generate(
                    lengths= cross_attention_condition_lengths,
                    max_length= torch.ones_like(cross_attention_conditions[0, 0]).sum()
                    )    # [Batch, C_t]

            cross_attention_conditions = cross_attention_conditions.permute(2, 0, 1).contiguous()    # [CAC_t, Batch, CAC_d]
            x_cross_attentions, _ = self.cross_attention(
                query= x_cross_attentions,
                key= cross_attention_conditions,
                value= cross_attention_conditions,
                key_padding_mask= cross_attention_condition_masks,
                attn_mask= cross_attention_masks
                )   # [X_t, Batch, X_d]
            x_cross_attentions = x_cross_attentions.permute(1, 0, 2)  # [Batch, X_t, X_d]

            if self.norm_type == Norm_Type.LayerNorm:
                x_self_attentions = self.self_attention_norm(x_self_attentions) # [Batch, X_t, X_d]
                x_cross_attentions = self.cross_attention_norm(x_cross_attentions) # [Batch, X_t, X_d]
            if self.norm_type in [Norm_Type.Conditional_LayerNorm, Norm_Type.Mixed_LayerNorm]:
                x_self_attentions = self.self_attention_norm(x_self_attentions, layer_norm_conditions) # [Batch, X_t, X_d]
                x_cross_attentions = self.cross_attention_norm(x_cross_attentions, layer_norm_conditions) # [Batch, X_t, X_d]

            x = x_self_attentions + x_cross_attentions
        else:
            x = x_self_attentions

        # apply residual norm
        if self.use_prenorm:
            if self.norm_type == Norm_Type.LayerNorm:
                x = (self.residual_norm(x) + residuals).mT * float_masks  # [Batch, X_d, X_t]
            elif self.norm_type in [Norm_Type.Conditional_LayerNorm, Norm_Type.Mixed_LayerNorm]:
                x = (self.residual_norm(x, layer_norm_conditions) + residuals).mT * float_masks   # [Batch, X_d, X_t]
        else:
            if self.norm_type == Norm_Type.LayerNorm:
                x = self.residual_norm(x + residuals).mT * float_masks  # [Batch, X_d, X_t]
            elif self.norm_type in [Norm_Type.Conditional_LayerNorm, Norm_Type.Mixed_LayerNorm]:
                x = self.residual_norm(x + residuals, layer_norm_conditions).mT * float_masks   # [Batch, X_d, X_t]

        # feed forward
        if not layer_norm_conditions is None:
            layer_norm_conditions = layer_norm_conditions.mT
        x = self.ffn(
            x= x,
            conditions= layer_norm_conditions,
            float_masks= float_masks
            )
        
        return x    # [B, X_c, X_t]

    def Get_Cross_Alignments(
        self,
        x: torch.FloatTensor,
        cross_attention_conditions: torch.FloatTensor,
        cross_attention_condition_lengths: Optional[torch.IntTensor]= None,
        cross_attention_masks: Optional[torch.FloatTensor]= None,   # masking is 1.0, don't use boolean.
        average_attn_weights: bool= True,
        ) -> torch.FloatTensor:
        '''
        x: [B, X_d, X_t]
        cross_attention_conditions: [B, CAC_d, CAC_t]
        cross_attention_condition_lengths: [B]
        '''
        cross_attention_condition_masks = None
        if not cross_attention_condition_lengths is None:
            cross_attention_condition_masks = Mask_Generate(
                lengths= cross_attention_condition_lengths,
                max_length= torch.ones_like(cross_attention_conditions[0, 0]).sum()
                )    # [Batch, C_t]

        x = x.permute(2, 0, 1).contiguous()
        cross_attention_conditions = cross_attention_conditions.permute(2, 0, 1).contiguous()    # [CAC_t, Batch, CAC_d]
        
        _, alignments = self.cross_attention(
            query= x,
            key= cross_attention_conditions,
            value= cross_attention_conditions,
            key_padding_mask= cross_attention_condition_masks,
            attn_mask= cross_attention_masks,
            need_weights= True,
            average_attn_weights= average_attn_weights
            )   # [Batch, X_t, CAC_t] or [Batch, Head, X_t, CAC_t]

        return alignments   # [Batch, X_t, CAC_t] or [Batch, Head, X_t, CAC_t]

class FFN(torch.nn.Module):
    def __init__(
        self,
        channels: int,
        kernel_size: int,
        droput_rate: float= 0.0,
        norm_type: Norm_Type= Norm_Type.LayerNorm,
        use_prenorm: bool= True,
        condition_channels: Optional[int]= None,
        beta_distribution_concentration: Optional[float]= None,
        ) -> None:
        super().__init__()
        self.norm_type = norm_type
        self.use_prenorm = use_prenorm

        self.conv_0 = Conv_Init(torch.nn.Conv1d(
            in_channels= channels,
            out_channels= channels,
            kernel_size= kernel_size,
            padding= (kernel_size - 1) // 2,
            bias= False
            ), w_init_gain= 'gelu')
        self.gelu = torch.nn.GELU(approximate= 'tanh')
        self.conv_1 = Conv_Init(torch.nn.Conv1d(
            in_channels= channels,
            out_channels= channels,
            kernel_size= kernel_size,
            padding= (kernel_size - 1) // 2,
            bias= False
            ), w_init_gain= 'linear')

        if norm_type == Norm_Type.LayerNorm:
            self.norm = torch.nn.LayerNorm(channels)
        elif norm_type == Norm_Type.Conditional_LayerNorm:
            self.norm = Conditional_LayerNorm(
                channels= channels,
                condition_channels= condition_channels
                )
        elif norm_type == Norm_Type.Mixed_LayerNorm:
            self.norm = Mixed_LayerNorm(
                channels= channels,
                condition_channels= condition_channels,
                beta_distribution_concentration= beta_distribution_concentration
                )
        
        self.dropout = torch.nn.Dropout(p= droput_rate)

    def forward(
        self,
        x: torch.Tensor,
        conditions: Optional[torch.FloatTensor]= None,
        float_masks: Optional[torch.FloatTensor]= 1.0,
        ):
        '''
        conditions: [Batch. C_d, X_t or 1]
        '''
        residuals = x
        x = self.conv_0(x * float_masks)
        x = self.gelu(x)
        x = self.dropout(x)
        x = self.conv_1(x * float_masks)
        x = self.dropout(x)

        if self.use_prenorm:
            if self.norm_type == Norm_Type.LayerNorm:
                x = (self.norm(x.mT).mT + residuals) * float_masks
            elif self.norm_type in [Norm_Type.Conditional_LayerNorm, Norm_Type.Mixed_LayerNorm]:
                x = (self.norm(x.mT, conditions.mT).mT + residuals) * float_masks
        else:
            if self.norm_type == Norm_Type.LayerNorm:
                x = self.norm((x + residuals).mT).mT * float_masks  # [Batch, X_d, X_t]
            elif self.norm_type in [Norm_Type.Conditional_LayerNorm, Norm_Type.Mixed_LayerNorm]:
                x = self.norm((x + residuals).mT, conditions.mT).mT * float_masks   # [Batch, X_d, X_t]

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

class LinearAttention(torch.nn.Module):
    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        ) -> None:
        super().__init__()
        assert embed_dim % num_heads == 0, 'embed_dim must be divisible by num_heads.'

        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.scaling = self.head_dim ** -0.5

        self.query = Conv_Init(
            torch.nn.Linear(embed_dim, embed_dim, bias= False),
            w_init_gain= 'linear'
            )
        self.key = Conv_Init(
            torch.nn.Linear(embed_dim, embed_dim, bias= False),
            w_init_gain= 'linear'
            )
        self.value = Conv_Init(
            torch.nn.Linear(embed_dim, embed_dim, bias= False),
            w_init_gain= 'linear'
            )
        
        self.projection = Conv_Init(
            torch.nn.Linear(embed_dim, embed_dim, bias= False),
            w_init_gain= 'linear'
            )

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        key_padding_mask: Optional[torch.Tensor]= None
        ) -> torch.FloatTensor:
        query_seq_len, batch_size, _ = query.size()
        key_seq_len, _, _ = key.size()

        queries = self.query(query).view(query_seq_len, batch_size, self.num_heads, self.head_dim)
        keys = self.key(key).view(key_seq_len, batch_size, self.num_heads, self.head_dim)
        values = self.value(value).view(key_seq_len, batch_size, self.num_heads, self.head_dim)

        queries = (torch.nn.functional.elu(queries) + 1.0) * self.scaling
        keys = (torch.nn.functional.elu(keys) + 1.0)

        if not key_padding_mask is None:
            key_padding_mask = key_padding_mask.T[:, :, None, None] # [Key_t, Batch, 1, 1]
            keys.masked_fill_(key_padding_mask, 0.0)

        keys_and_values = einsum(keys, values, 'key_length batch head key_d, key_length batch head value_d -> batch head key_d value_d')
        keys_sum = keys.sum(dim= 0, keepdim= True)  # [1, Batch, Head_n, Head_d]
        contexts = einsum(queries, keys_and_values, 'query_length batch head query_d, batch head query_d value_d -> query_length batch head value_d')
        normalizer = einsum(queries, keys_sum, 'query_length batch head query_d, x batch head query_d -> query_length batch head x') + 1e-5

        contexts = contexts / normalizer
        contexts = contexts.contiguous().view(query_seq_len, batch_size, self.embed_dim)
        contexts = self.projection(contexts)

        return contexts

class MultiHeadAttentionWithRoPE(torch.nn.Module):
    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        k_dim: Optional[int]= None,
        v_dim: Optional[int]= None,
        ):
        super().__init__()        
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        
        assert embed_dim % num_heads == 0, 'embed_dim must be divisible by num_heads'
        assert self.head_dim % 2 == 0, 'head_dim must be divisible by 2 for RoPE'
        
        k_dim = k_dim or embed_dim
        v_dim = v_dim or k_dim
        
        self.q_proj = Conv_Init(
            torch.nn.Linear(embed_dim, embed_dim, bias= False),
            w_init_gain= 'linear'
            )
        self.k_proj = Conv_Init(
            torch.nn.Linear(k_dim, embed_dim, bias= False),
            w_init_gain= 'linear'
            )
        self.v_proj = Conv_Init(
            torch.nn.Linear(v_dim, embed_dim, bias= False),
            w_init_gain= 'linear'
            )
        
        self.o_proj = Conv_Init(
            torch.nn.Linear(embed_dim, embed_dim, bias= False),
            w_init_gain= 'linear'
            )

    def forward(
        self,
        query: torch.FloatTensor,
        key: torch.FloatTensor,
        value: torch.FloatTensor,
        key_padding_mask: Optional[torch.BoolTensor]= None,
        need_weights: bool= False,
        attn_mask: Optional[torch.BoolTensor]= None,
        average_attn_weights: bool= True
        ):
        '''
        query: [Query_t, Batch, Query_d]
        key: [Key_t, Batch, Key_d]
        value: [Value_t, Batch, Value_d], Key_t == Value_t
        attn_mask: [Batch, Query_t, Key_t]
        '''
        query_lengths, batch_size, _ = query.size()
        key_lengths, _, _ = key.size()

        # Linear projections
        Q = self.q_proj(query)  # [Query_t, Batch, Query_d]

        K = self.k_proj(key)    # [Key_t, Batch, Query_d]
        V = self.v_proj(value)  # [Key_t, Batch, Query_d]

        # Reshape
        Q = rearrange(
            tensor= Q,
            pattern= 'query_t batch (num_head head_d) -> batch num_head query_t head_d',
            num_head= self.num_heads,
            head_d= self.head_dim
            )   # [Batch, Head, Query_t, Head_d]
        K = rearrange(
            tensor= K,
            pattern= 'key_t batch (num_head head_d) -> batch num_head key_t head_d',
            num_head= self.num_heads,
            head_d= self.head_dim
            )   # [Batch, Head, Key_t, Head_d]
        V = rearrange(
            tensor= V,
            pattern= 'key_t batch (num_head head_d) -> batch num_head key_t head_d',
            num_head= self.num_heads,
            head_d= self.head_dim
            )   # [Batch, Head, Key_t, Head_d]

        # Apply RoPE
        Q = self.apply_rope(Q)
        K = self.apply_rope(K)

        # Scaled Dot-Product Attention using PyTorch's native function
        Q = rearrange(
            tensor= Q,
            pattern= 'batch num_head query_t head_d -> (batch num_head) query_t head_d',
            )   # [Batch * Head, Query_t, Head_d]
        K = rearrange(
            tensor= K,
            pattern= 'batch num_head key_t head_d -> (batch num_head) key_t head_d',
            )   # [Batch * Head, Key_t, Head_d]
        V = rearrange(
            tensor= V,
            pattern= 'batch num_head key_t head_d -> (batch num_head) key_t head_d',
            )   # [Batch * Head, Key_t, Head_d]

        if not attn_mask is None:
            attn_mask = rearrange(
                tensor= attn_mask[:, None, :query_lengths, :key_lengths].expand(-1, self.num_heads, -1 , -1).float(),   # flash attention use float type mask, indexing is because of eval inference problem
                pattern= 'batch num_head query_t key_t -> (batch num_head) query_t key_t',
                )   # [Batch * Head, Query_t, Key_t]
        else:
            # make attn_mask
            attn_mask = torch.zeros(
                size= (batch_size * self.num_heads, query_lengths, key_lengths),
                device= Q.device
                )   # [Batch * Head, Query_t, Key_t]

        if not key_padding_mask is None:
            key_padding_mask = key_padding_mask[:, None, None, :].expand(
                batch_size,
                self.num_heads,
                1,
                key_lengths
                )
            key_padding_mask = rearrange(
                key_padding_mask,
                'batch_size num_head x key_t -> (batch_size num_head) x key_t'
                )
            attn_mask = attn_mask + key_padding_mask.float()    # [Batch * Head, 1, Key_t], 
        attn_mask = attn_mask * -1e+5 # float(torch.finfo(attn_mask.dtype).min)

        output = torch.nn.functional.scaled_dot_product_attention(Q, K, V, attn_mask=attn_mask) # [Batch * Head, Query_t, Head_d]

        # Reshape
        output = rearrange(
            tensor= output,
            pattern= '(batch num_head) query_t head_d -> query_t batch (num_head head_d)',
            batch= batch_size,
            num_head= self.num_heads
            )   # [Batch * Head, Query_t, Head_d] -> [Query_t, Batch, Query_d]
        
        output = self.o_proj(output)

        alignments = None
        if need_weights:
            scaling = float(self.head_dim) ** -0.5
            attn_weights = torch.bmm(Q * scaling, K.mT)
            # if not key_padding_mask is None:
            # attn_weights = attn_weights.masked_fill(
            #     attn_mask.bool(),
            #     float(torch.finfo(attn_weights.dtype).min)
            #     )
            attn_weights = attn_weights + attn_mask
            alignments = torch.softmax(attn_weights, dim=-1)    # [Batch * Head, Query_t, Key_t]
            alignments = rearrange(
                alignments,
                '(batch num_head) query_t key_t -> batch num_head query_t key_t',
                batch= batch_size,
                num_head = self.num_heads
                )
            if average_attn_weights:
                alignments = alignments.mean(dim= 1)    # [Batch, Query_t, Key_t]

        return output, alignments


    def apply_rope(self, x):
        '''
        Apply Rotary Position Embedding (RoPE) to the given tensor.
        Args:
            x: Tensor of shape (batch_size, num_heads, seq_len, head_dim)
        Returns:
            Tensor with RoPE applied.
        '''
        _, _, seq_len, head_dim = x.size()

        # Create position index and scaling factors
        theta = 1.0 / (10000 ** (torch.arange(0, head_dim // 2, device=x.device).float() / (head_dim // 2)))    # [Head_d / 2]
        pos = torch.arange(0, seq_len, device= x.device).float()    # [X_t]
        
        # Compute sinusoidal embeddings
        sinusoid = einsum(pos, theta, 'lengths, dim -> lengths dim')    # [X_t, Head_d / 2]
        sin, cos = sinusoid.sin(), sinusoid.cos()
        sin = sin[None, None, :, :] # [1, 1, X_t, Head_d / 2]
        cos = cos[None, None, :, :] # [1, 1, X_t, Head_d / 2]

        x1, x2 = x.chunk(chunks= 2, dim= 3) # [Batch, Head, X_t, Head_d / 2] * 2
        x_rope = torch.cat([
            x1 * cos - x2 * sin,
            x1 * sin + x2 * cos
            ], dim= 3)  # [Batch, Head, X_t, Head_d]

        return x_rope

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
