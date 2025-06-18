import torch
from .Layer import Conv_Init, Mask_Generate

class Location_Sensitive_Stepwise_Monotonic_Attention(torch.nn.Module):
    def __init__(
        self,
        embed_dim: int,
        attention_location_channels: int,
        attention_location_kernel_size: int,
        sigmoid_noise: float= 2.0,
        k_dim: int | None= None,
        v_dim: int | None= None,
        ):
        super().__init__()
        self.sigmoid_noise = sigmoid_noise

        k_dim = k_dim or embed_dim
        v_dim = v_dim or k_dim

        self.query = Conv_Init(
            torch.nn.Linear(embed_dim, embed_dim, bias= False),
            w_init_gain= 'linear'
            )
        self.key = Conv_Init(
            torch.nn.Linear(k_dim, embed_dim, bias= False),
            w_init_gain= 'linear'
            )
        self.value = Conv_Init(
            torch.nn.Linear(v_dim, embed_dim, bias= False),
            w_init_gain= 'linear'
            )

        self.location_conv_0 = Conv_Init(torch.nn.Conv1d(
            in_channels= 2,
            out_channels= attention_location_channels,
            kernel_size= attention_location_kernel_size,
            padding= (attention_location_kernel_size - 1) // 2,
            bias= False
            ), w_init_gain= 'gelu')
        self.gelu = torch.nn.GELU(approximate= 'tanh')
        self.location_conv_1 = Conv_Init(torch.nn.Conv1d(
            in_channels= attention_location_channels,
            out_channels= embed_dim,
            kernel_size= 1,
            bias= False
            ), w_init_gain= 'linear')
        self.location_norm = torch.nn.LayerNorm(embed_dim)

        self.tanh = torch.nn.Tanh()

        self.score = Conv_Init(torch.nn.Conv1d(
            in_channels= embed_dim,
            out_channels= 1,
            kernel_size= 1,
            bias= False
            ), w_init_gain= 'linear')
    
    def forward(
        self,
        queries: torch.FloatTensor,
        values: torch.FloatTensor,
        keys: torch.FloatTensor | None= None,
        key_padding_mask: torch.BoolTensor | None= None
        ):
        '''
        queries: [Batch, Query_d, Query_t]
        keys: [Batch, Key_d, Key_t]
        values: [Batch, Key_d, Key_t] or None
        key_padding_mask: [Batch, Key_t]
        '''
        queries = self.query(queries.mT).mT   # [Batch, Query_d, Query_t]
        keys = self.key(keys.mT if not keys is None else values.mT).mT   # [Batch, Query_d, Key_t]
        values = self.value(values.mT).mT # [Batch, Query_d, Key_t]
        
        alignments_step = self.Get_Initial_Alignments(keys= keys)
        cumulated_alignments_step = alignments_step

        alignments = []
        for queries_step in queries.permute(2, 0, 1):
            alignments_step = self.Step(
                queries_step= queries_step,
                keys= keys,
                previous_alignments= alignments_step,
                cumulated_alignments= cumulated_alignments_step,
                key_padding_mask= key_padding_mask
                )   # [Batch, Key_t]
            cumulated_alignments_step = cumulated_alignments_step + alignments_step
            alignments.append(alignments_step)

        alignments = torch.stack(alignments, dim= 2)    # [Batch, Key_t, Query_t]
        contexts = values @ alignments  # [Batch, Query_d, Query_t]

        return contexts, alignments

    def Step(
        self,
        queries_step: torch.FloatTensor,
        keys: torch.FloatTensor,
        previous_alignments: torch.FloatTensor,
        cumulated_alignments: torch.FloatTensor,
        key_padding_mask: torch.BoolTensor | None= None
        ):
        '''
        queries_step: [Batch, Query_d]
        keys: [Batch, Query_d, Key_t]
        values: [Batch, Query_d, Key_t]
        previous_alignments: [Batch, Key_t]
        cumulated_alignments: [Batch, Key_t]
        '''
        locations = torch.stack(
            [previous_alignments, cumulated_alignments],
            dim= 1
            )   # [Batch, 2, Key_t]
        locations = self.location_conv_0(locations) # [Batch, Query_d, Key_t]
        locations = self.gelu(locations)
        locations = self.location_conv_1(locations) # [Batch, Query_d, Key_t]
        locations = self.location_norm(locations.mT).mT # [Batch, Query_d, Key_t]
        
        # print(queries_step[:, :, None].shape)
        # print(keys.shape)
        # print(locations.shape)

        scores = self.tanh(queries_step[:, :, None] + keys + locations)
        scores = self.score(scores)[:, 0, :] # [Batch, Key_t]

        previous_alignments = previous_alignments[:, None, :]   # [Batch, Key_t]
        if self.training and self.sigmoid_noise > 0.0:
            scores =  scores + torch.randn_like(scores) * self.sigmoid_noise
            
        if not key_padding_mask is None:
            scores.masked_fill_(key_padding_mask, -float('inf'))
        
        p_choose_i = scores[:, None, :].sigmoid()   # [Batch, 1, Key_t]
        pad = torch.zeros(p_choose_i.size(0), 1, 1).to(
            device= p_choose_i.device,
            dtype= p_choose_i.dtype
            )    # [Batch, 1, 1]
        
        alignments = previous_alignments * p_choose_i + torch.cat(
            [pad, previous_alignments[..., :-1] * (1.0 - p_choose_i[..., :-1])],
            dim= -1
            )   # [Batch, 1, Key_t]
        
        return alignments[:, 0, :]  # [Batch, Key_t]
    
    def Get_Initial_Alignments(
        self,
        keys: torch.FloatTensor
        ):
        return torch.nn.functional.one_hot(
            keys.new_zeros(keys.size(0)).long(),
            num_classes= keys.size(2)
            ).to(dtype= keys.dtype)