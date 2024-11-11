import torch
import numpy as np
from numba import jit

class Guided_Attention_Loss(torch.nn.Module):
    def __init__(self, sigma= 0.2):
        super().__init__()
        self.sigma = sigma

    def forward(self, alignments, query_lengths, key_lengths):
        '''
        alignments: [Batch, Query_t, Key_t]
        query_lengths: [Batch]
        key_lengths: [Batch]
        '''
        return (alignments * self.Get_Soft_Masks(query_lengths, key_lengths).to(alignments.device)).mean()

    def Get_Soft_Masks(self, query_lengths, key_lengths):
        query_lengths = query_lengths.cpu().numpy()
        key_lengths = key_lengths.cpu().numpy()
        max_query_length = max(query_lengths)
        max_key_length = max(key_lengths)
        masks = np.stack([
            _Calc_Soft_Masks(query_Length, max_query_length, key_Length, max_key_length, self.sigma).T
            for query_Length, key_Length in zip(query_lengths, key_lengths)
            ], axis= 0)

        return torch.FloatTensor(masks).mT

@jit(nopython=True)
def _Calc_Soft_Masks(query_length, max_query_length, key_length, max_key_length, sigma= 0.2):
    mask = np.zeros((max_query_length, max_key_length), dtype=np.float32)
    for query_index in range(query_length):
        for key_index in range(key_length):
            mask[query_index, key_index] = 1 - np.exp(-(query_index / query_length - key_index / key_length) ** 2 / (2 * sigma ** 2))

    return mask