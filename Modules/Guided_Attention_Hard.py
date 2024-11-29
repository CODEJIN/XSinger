import torch
import numpy as np
from numba import jit

class Guided_Attention_Loss(torch.nn.Module):
    def forward(
        self,
        alignments,
        token_on_note_lengths,
        note_durations
        ):
        '''
        alignments: [Batch, Query_t, Key_t]
        query_lengths: [Batch]
        key_lengths: [Batch]
        '''
        masks = self.Get_Masks(token_on_note_lengths, note_durations).to(alignments.device)
        loss = (alignments * masks).sum() / (token_on_note_lengths.sum(dim= 1) * note_durations.sum(dim= 1)).sum()

        return loss

    def Get_Masks(
        self,
        token_on_note_lengths: torch.IntTensor, # type: ignore
        note_durations: torch.IntTensor # type: ignore
        ):
        token_on_note_lengths: np.ndarray = token_on_note_lengths.cpu().numpy()
        note_durations: np.ndarray = note_durations.cpu().numpy()
        max_token_length = token_on_note_lengths.sum(axis= 1).max()
        max_duration_length = note_durations.sum(axis= 1).max()

        masks = np.stack([
            _Calc_Masks(
                token_on_note_length,
                note_duration,
                max_token_length,
                max_duration_length
                )
            for token_on_note_length, note_duration in zip(token_on_note_lengths, note_durations)
            ], axis= 0)

        return torch.FloatTensor(masks).mT

@jit(nopython=True)
def _Calc_Masks(
    token_on_note_length: np.ndarray,
    note_duration: np.ndarray,
    max_token_length: int,
    max_latent_length: int,
    extra_width_factor: int= 10
    ):
    all_token_on_note_length = token_on_note_length.sum()
    all_note_duration = note_duration.sum()
    token_extra_width = max(1, token_on_note_length.sum() // extra_width_factor)
    latent_extra_width = max(1, note_duration.sum() // extra_width_factor)

    mask = np.ones((max_token_length, max_latent_length))
    mask[all_token_on_note_length:] = 0
    mask[all_note_duration:] = 0
    token_start_index, latent_start_index = 0, 0
    for token_length, latent_length in zip(token_on_note_length, note_duration):
        mask[
            max(0, token_start_index - token_extra_width):min(token_start_index + token_length + token_extra_width, all_token_on_note_length),
            max(0, latent_start_index - latent_extra_width):min(latent_start_index + latent_length + latent_extra_width, all_note_duration)
            ] = 0.0
        token_start_index += token_length
        latent_start_index += latent_length

    return mask

# @jit(nopython=True)
# def _Calc_Soft_Masks(query_length, max_query_length, key_length, max_key_length, sigma= 0.2):
#     mask = np.zeros((max_query_length, max_key_length), dtype=np.float32)
#     for query_index in range(query_length):
#         for key_index in range(key_length):
#             mask[query_index, key_index] = 1 - np.exp(-(query_index / query_length - key_index / key_length) ** 2 / (2 * sigma ** 2))

#     return mask