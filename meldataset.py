###############################################################################
# MIT License
#
# Copyright (c) 2020 Jungil Kong
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
###############################################################################

import math
import os
import random
import logging
import torch
import torch.utils.data
import numpy as np
from librosa.util import normalize
from scipy.io.wavfile import read
from librosa.filters import mel as librosa_mel_fn
# import pyworld as pw

MAX_WAV_VALUE = 32768.0


def load_wav(full_path):
    sampling_rate, data = read(full_path)
    return data, sampling_rate


def dynamic_range_compression(x, C=1, clip_val=1e-5):
    return np.log(np.clip(x, a_min=clip_val, a_max=None) * C)


def dynamic_range_decompression(x, C=1):
    return np.exp(x) / C


def dynamic_range_compression_torch(x, C=1, clip_val=1e-5):
    return torch.log(torch.clamp(x, min=clip_val) * C)


def dynamic_range_decompression_torch(x, C=1):
    return torch.exp(x) / C


def spectral_normalize_torch(magnitudes):
    output = dynamic_range_compression_torch(magnitudes)
    return output


def spectral_de_normalize_torch(magnitudes):
    output = dynamic_range_decompression_torch(magnitudes)
    return output

mel_basis_caches = {}
hann_window_caches = {}

def mel_spectrogram(
    y: torch.Tensor,
    n_fft: int,
    num_mels: int,
    sampling_rate: int,
    hop_size: int,
    win_size: int,
    fmin: int,
    fmax: int= None,
    center: bool= False,
    use_normalize= True
    ) -> torch.Tensor:
    if torch.min(y) < -1.0:
        logging.warning('min value is {}'.format(torch.min(y)))
    if torch.max(y) > 1.:
        logging.warning('max value is {}'.format(torch.max(y)))

    device = y.device
    mel_key = f'{n_fft}_{num_mels}_{sampling_rate}_{fmin}_{fmax}_{device}_{torch.is_inference_mode_enabled()}'
    hann_window_key = f'{win_size}_{device}_{torch.is_inference_mode_enabled()}'
    
    if mel_key not in mel_basis_caches:
        mel = librosa_mel_fn(
            sr= sampling_rate,
            n_fft= n_fft,
            n_mels= num_mels,
            fmin= fmin,
            fmax= fmax
            )
        mel_basis_caches[mel_key] = torch.from_numpy(mel).float().to(device)
    if hann_window_key not in hann_window_caches:
        hann_window_caches[hann_window_key] = torch.hann_window(win_size).to(device)

    mel_basis = mel_basis_caches[mel_key]
    hann_window = hann_window_caches[hann_window_key]

    padding = (n_fft - hop_size) // 2
    y = torch.nn.functional.pad(
        y[:, None, :],
        (padding, padding),
        mode='reflect'
        )[:, 0, :]
    
    spec = torch.stft(
        y,
        n_fft,
        hop_length= hop_size,
        win_length= win_size,
        window= hann_window,
        center= center,
        pad_mode= 'reflect',
        normalized= False,
        onesided= True,
        return_complex= True
        )
    spec = torch.sqrt(torch.view_as_real(spec).pow(2).sum(-1) + 1e-9)

    mel_spec = torch.matmul(mel_basis, spec)
    if use_normalize:
        mel_spec = spectral_normalize_torch(mel_spec)

    return mel_spec

def cepstral_liftering(y, n_fft, feature_size, hop_size, win_size, cutoff= 3, center=False):
    if torch.min(y) < -1.:
        logging.warning('min value is {}'.format(torch.min(y)))
    if torch.max(y) > 1.:
        logging.warning('max value is {}'.format(torch.max(y)))

    device = y.device
    hann_window_key = f'{win_size}_{device}_{torch.is_inference_mode_enabled()}'
    if hann_window_key not in hann_window_caches:
        hann_window_caches[hann_window_key] = torch.hann_window(win_size).to(device)
    hann_window = hann_window_caches[hann_window_key]

    y = torch.nn.functional.pad(y.unsqueeze(1), (int((n_fft-hop_size)/2), int((n_fft-hop_size)/2)), mode='reflect')
    y = y.squeeze(1)

    spec = torch.stft(y, n_fft, hop_length=hop_size, win_length=win_size, window=hann_window,
                      center=center, pad_mode='reflect', normalized=False, onesided=True, return_complex= True)
    spec = torch.fft.irfft(torch.log(spec+1e-6), axis= 1)
    
    lifter = torch.zeros(spec.size(1))
    lifter[:cutoff] = 1
    lifter[cutoff] = 0.5
    lifter = torch.diag(lifter).unsqueeze(0).expand(spec.size(0), -1, -1)

    spec = torch.matmul(lifter, spec)
    spec = torch.fft.rfft(spec, dim= 1).exp().abs()
    spec = spectral_normalize_torch(spec)
    spec = torch.nn.functional.interpolate(spec.unsqueeze(1), size= (feature_size, spec.size(2)), mode= 'bilinear').squeeze(1)

    return spec

def spectrogram(y, n_fft, hop_size, win_size, center=False, use_normalize= True):
    if torch.min(y) < -1.:
        logging.warning('min value is {}'.format(torch.min(y)))
    if torch.max(y) > 1.:
        logging.warning('max value is {}'.format(torch.max(y)))

    device = y.device
    hann_window_key = f'{win_size}_{device}_{torch.is_inference_mode_enabled()}'
    if hann_window_key not in hann_window_caches:
        hann_window_caches[hann_window_key] = torch.hann_window(win_size).to(device)
    hann_window = hann_window_key

    y = torch.nn.functional.pad(y.unsqueeze(1), (int((n_fft-hop_size)/2), int((n_fft-hop_size)/2)), mode='reflect')
    y = y.squeeze(1)

    spec = torch.stft(y, n_fft, hop_length=hop_size, win_length=win_size, window=hann_window,
                      center=center, pad_mode='reflect', normalized=False, onesided=True, return_complex=False)

    spec = torch.sqrt(spec.pow(2).sum(-1)+(1e-9))
    if use_normalize:
        spec = spectral_normalize_torch(spec)

    return spec

def spectrogram_to_mel(spec, n_fft, num_mels, sampling_rate, win_size, fmin, fmax, use_denorm= False):
    spec = spectral_de_normalize_torch(spec) if use_denorm else spec
    
    device = spec.device
    mel_key = f'{n_fft}_{num_mels}_{sampling_rate}_{fmin}_{fmax}_{device}_{torch.is_inference_mode_enabled()}'
    
    if mel_key not in mel_basis_caches:
        mel = librosa_mel_fn(
            sr= sampling_rate,
            n_fft= n_fft,
            n_mels= num_mels,
            fmin= fmin,
            fmax= fmax
            )
        mel_basis_caches[mel_key] = torch.from_numpy(mel).float().to(device)

    mel_basis = mel_basis_caches[mel_key]

    spec = torch.matmul(mel_basis, spec)
    spec = spectral_normalize_torch(spec)

    return spec

def spec_energy(y, n_fft, hop_size, win_size, center=False):
    if torch.min(y) < -1.:
        logging.warning('min value is {}'.format(torch.min(y)))
    if torch.max(y) > 1.:
        logging.warning('max value is {}'.format(torch.max(y)))

    device = y.device
    hann_window_key = f'{win_size}_{device}_{torch.is_inference_mode_enabled()}'
    
    if hann_window_key not in hann_window_caches:
        hann_window_caches[hann_window_key] = torch.hann_window(win_size).to(device)

    hann_window = hann_window_caches[hann_window_key]

    y = torch.nn.functional.pad(y.unsqueeze(1), (int((n_fft-hop_size)/2), int((n_fft-hop_size)/2)), mode='reflect')
    y = y.squeeze(1)

    spec = torch.stft(y, n_fft, hop_length=hop_size, win_length=win_size, window=hann_window,
                      center=center, pad_mode='reflect', normalized=False, onesided=True, return_complex=False)
    spec = torch.sqrt(spec.pow(2).sum(-1)+(1e-9))
    energy = torch.norm(spec, dim= 1)

    return energy

def get_dataset_filelist(a):
    with open(a.input_training_file, 'r', encoding='utf-8') as fi:
        training_files = [os.path.join(a.input_wavs_dir, x.split('|')[0] + '.wav')
                          for x in fi.read().split('\n') if len(x) > 0]

    with open(a.input_validation_file, 'r', encoding='utf-8') as fi:
        validation_files = [os.path.join(a.input_wavs_dir, x.split('|')[0] + '.wav')
                            for x in fi.read().split('\n') if len(x) > 0]
    return training_files, validation_files


# https://github.com/biggytruck/SpeechSplit2/blob/b67354aa74b252003c8e644176fc964ad1a241ad/utils.py#L233
def get_frequency_warp(n_fft: int, sampling_rate: int, fhi: int= 4800, alpha: float= 0.9):
    bins = torch.linspace(0, 1, n_fft)
    scale = fhi * min(alpha, 1.0)
    frequency_boundary = scale / alpha
    sampling_rate_half = sampling_rate // 2

    frequency_original = bins * sampling_rate
    frequency_warp = torch.where(
        frequency_original <= frequency_boundary,
        frequency_original * alpha,
        sampling_rate_half - (sampling_rate_half - scale) / (sampling_rate_half - scale / alpha) * (sampling_rate_half - frequency_original)
        )
    
    return frequency_warp

# https://github.com/biggytruck/SpeechSplit2/blob/b67354aa74b252003c8e644176fc964ad1a241ad/utils.py#L252
def vtlp(y: torch.Tensor, n_fft: int, sampling_rate: int, hop_size: int, win_size: int, alpha: float, center: bool=False):
    if torch.min(y) < -1.:
        logging.warning('min value is {}'.format(torch.min(y)))
    if torch.max(y) > 1.:
        logging.warning('max value is {}'.format(torch.max(y)))

    device = y.device
    hann_window_key = f'{win_size}_{device}_{torch.is_inference_mode_enabled()}'
    
    if hann_window_key not in hann_window_caches:
        hann_window_caches[hann_window_key] = torch.hann_window(win_size).to(device)

    hann_window = hann_window_caches[hann_window_key]
    
    y_padded = torch.nn.functional.pad(y.unsqueeze(1), (int((n_fft-hop_size)/2), int((n_fft-hop_size)/2)), mode='reflect')
    y_padded = y_padded.squeeze(1)

    spec = torch.stft(y_padded, n_fft, hop_length=hop_size, win_length=win_size, window=hann_window,
                      center=center, pad_mode='reflect', normalized=False, onesided=True, return_complex= True)

    frequency_warp = get_frequency_warp(n_fft= spec.size(1), sampling_rate= sampling_rate, alpha= alpha)
    frequency_warp *= (spec.size(1) - 1) / frequency_warp.max()
    
    spec_warp = torch.zeros_like(spec)
    for index in range(spec.size(1)):
        if index == 0 or index == spec.size(1) - 1:
            spec_warp[:, index] += spec[:, index]
        else:
            warp_up = frequency_warp[index] - frequency_warp[index].floor()
            warp_down = 1 - warp_up
            position = int(frequency_warp[index].floor())
            spec_warp[:, position] += warp_down * spec[:, index]
            spec_warp[:, position + 1] += warp_up * spec[:, index]

    y_warp = torch.istft(spec_warp, n_fft= n_fft, hop_length= hop_size, win_length= win_size, window=hann_window[str(y_padded.device)])
    y_warp = torch.nn.functional.pad(y_warp.unsqueeze(1), (0, y.size(1) - y_warp.size(1))).squeeze(1)
    y_warp = np.clip(y_warp, -1.0, 1.0)

    return y_warp

# def get_monotonic_wav(audio: np.array, sampling_rate: int, mean: float):
#     audio = audio.astype('double')
#     _f0, t = pw.dio(audio, sampling_rate)
#     f0 = pw.stonemask(audio, _f0, t, sampling_rate)  # pitch refinement
#     sp = pw.cheaptrick(audio, f0, t, sampling_rate)  # extract smoothed spectrogram
#     ap = pw.d4c(audio, f0, t, sampling_rate)         # extract aperiodicity
    
#     f0 = np.where(f0 > 0, mean, 0.0)
#     audio_monotonic = pw.synthesize(f0, sp, ap, sampling_rate) # synthesize an utterance using the parameters
#     audio_monotonic = np.pad(audio_monotonic, (0, max(0, audio.shape[0] - audio_monotonic.shape[0])))
#     audio_monotonic = np.clip(audio_monotonic, -1.0, 1.0)

#     return audio_monotonic[:audio.shape[0]].astype(audio.dtype)



    
