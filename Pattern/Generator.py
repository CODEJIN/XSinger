import numpy as np
import pandas as pd
import mido, os, pickle, yaml, argparse, math, librosa, hgtk, logging, re, sys, warnings, textgrid as tg, music21, json
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
from tqdm import tqdm
from pysptk.sptk import rapt
from typing import List, Tuple
from argparse import Namespace  # for type
import torch, torchaudio
from itertools import cycle
from collections import deque
from typing import Dict, Union, Optional



from meldataset import mel_spectrogram, spec_energy
from Arg_Parser import Recursive_Parse
from . import CSD, GTSinger, AIHub_Mediazen, AIHub_Metabuild, M4Singer, OpenCPop, Kiritan, Ofuton
from . import Note, Note_to_Dict, Dict_to_Note

logging.basicConfig(
    level=logging.INFO, stream=sys.stdout,
    format= '%(asctime)s (%(module)s:%(lineno)d) %(levelname)s: %(message)s'
    )
warnings.filterwarnings('ignore')


def Load_Audio(
    path: str,
    sample_rate: int,
    hop_size: int
    ):
    audio = librosa.load(path, sr= sample_rate)[0]
    audio = audio[:audio.shape[0] - (audio.shape[0] % hop_size)]
    audio = librosa.util.normalize(audio) * 0.95

    return audio

def Note_Predictor(f0):
    '''
    f0: [F0_t]
    '''
    notes = np.arange(0, 128)
    # f0s = 440 * 2 ** ((notes - 69 - 12) / 12)
    f0s = 440 * 2 ** ((notes - 69) / 12)
    f0s[0] = 0.0
    criterion = np.expand_dims(f0s, axis= 0)   # [1, 128]

    return np.argmin(
        np.abs(np.expand_dims(f0, axis= 1) - criterion),
        axis= 1
        )

def Pattern_File_Generate_by_Full_Song(
    music: List[Note],
    audio: np.ndarray,
    singer: str,
    genre: str,
    language: str,
    dataset: str,
    music_label: str,
    is_eval_music: bool,
    hyper_parameters: Namespace,
    tech_from_dataset: Optional[np.ndarray]= None,
    note_error_criterion: float= 1.0,
    min_length_seconds: float= 5.0,    
    verbose: bool= False
    ):
    hp = hyper_parameters
    min_length_frame = min_length_seconds * hp.Sound.Sample_Rate // hp.Sound.Hop_Size

    audio_tensor = torch.from_numpy(audio).to(device).float()
    with torch.inference_mode():
        mel = mel_spectrogram(
            y= audio_tensor[None],
            n_fft= hp.Sound.Hop_Size * 4,
            num_mels= hp.Sound.N_Mel,
            sampling_rate= hp.Sound.Sample_Rate,
            hop_size= hp.Sound.Hop_Size,
            win_size= hp.Sound.Hop_Size * 4,
            fmin= 0,
            fmax= None,
            center= False
            )[0].cpu().numpy() # [Batch, N_Mel, Audio_t / Hop_Size]
        
        f0 = rmvpe_model(torchaudio.functional.resample(
            audio_tensor,
            orig_freq= hp.Sound.Sample_Rate,
            new_freq= 16_000
            ))
        f0 = torch.nn.functional.interpolate(
            input= f0[None, None, :],
            size= mel.shape[1],
            mode= 'linear',
            align_corners= False
            )[0, 0, :].cpu().numpy()

    music_length = sum([x.Duration for x in music])

    if mel.shape[1] > music_length:
        # print('Case1')
        mel = mel[:, math.floor((mel.shape[1] - music_length) / 2.0):-math.ceil((mel.shape[1] - music_length) / 2.0)]
        f0 = f0[math.floor((f0.shape[0] - music_length) / 2.0):-math.ceil((f0.shape[0] - music_length) / 2.0)]
    elif music_length > mel.shape[1]:
        # print('Case2')
        fix_length = music_length - mel.shape[1]
        fix_index = None
        for fix_index in reversed(range(len(music))):
            if music[fix_index].Duration <= fix_length:
                continue
            break
        music[fix_index].Duration = music[fix_index].Duration - fix_length
        
    note_from_f0 = Note_Predictor(f0).astype(np.float16)

    def Calc_Note_from_Music(
        music: List[Note],
        correction: int= 0
        ):
        return np.array([
            note.Pitch + (correction if note.Pitch > 0 else 0)
            for note in music
            for _ in range(note.Duration)
            ]).astype(np.float16)

    note_from_midi = Calc_Note_from_Music(music)
    note_from_f0_without_0, note_from_midi_without_0 = note_from_f0[(note_from_f0 > 0) * (note_from_midi > 0)], note_from_midi[(note_from_f0 > 0) * (note_from_midi > 0)]
    note_error_value = np.abs(note_from_f0_without_0 - note_from_midi_without_0).mean()

    # octave problem fix    
    if note_error_value > note_error_criterion:
        is_fixed = False
        if np.abs(note_from_f0_without_0 - note_from_midi_without_0 - 12.0).mean() < note_error_criterion:
            for note in music:
                note.Pitch = note.Pitch + 12 if note.Pitch > 0 else 0
            note_fix_from_midi = Calc_Note_from_Music(music)
            note_fix_from_midi_without_0 = note_fix_from_midi[(note_from_f0 > 0) * (note_from_midi > 0)]
            note_fix_error_value = np.abs(note_from_f0_without_0 - note_fix_from_midi_without_0).mean()
            is_fixed = True
        elif np.abs(note_from_f0_without_0 - note_from_midi_without_0 + 12.0).mean() < note_error_criterion:
            for note in music:
                note.Pitch = note.Pitch - 12 if note.Pitch > 0 else 0
            note_fix_from_midi = Calc_Note_from_Music(music)
            note_fix_from_midi_without_0 = note_fix_from_midi[(note_from_f0 > 0) * (note_from_midi > 0)]
            note_fix_error_value = np.abs(note_from_f0_without_0 - note_fix_from_midi_without_0).mean()
            is_fixed = True
        if is_fixed:
            if verbose:
                logging.warning(
                    f'the note octave of \'{dataset}-{singer}-{music_label}\' is fixed because the audio and midi note incompatible'
                    f'(note error = {note_error_value:.3f}, fixed note error = {note_fix_error_value:.3f}).'
                    'Note graph is exported at \'./note_fix\' path.'
                    )
                os.makedirs(os.path.join('./note_fix', dataset, singer), exist_ok= True)
                plt.figure(figsize= (50, 10))
                plt.plot(note_from_f0, label= 'F0 Note')
                plt.plot(note_from_midi, label= 'MIDI Note')
                plt.plot(note_fix_from_midi, label= 'Fixed MIDI Note')
                plt.xticks(
                    ticks= [x for x in [0.0] + np.cumsum([note.Duration for note in music]).tolist()[:-1]],
                    labels= [note.Text for note in music],
                    )
                for x in [x for x in [0.0] + np.cumsum([note.Duration for note in music]).tolist()[:-1]]:
                    plt.axvline(x= x, linewidth= 0.5)
                plt.margins(x= 0)
                plt.legend()
                plt.tight_layout()
                plt.savefig(os.path.join('./note_fix', dataset, singer, f'{music_label}.png'))
                plt.close()
        else:
            if verbose:
                logging.warning(
                    f'\'{dataset}-{singer}-{music_label}\' is skipped because the audio and midi note incompatible(note error = {note_error_value:.3f}).'
                    'This could be due to a misaligned octave in the MIDI or a synchronization issue between the audio and MIDI.'
                    'Note graph is exported at \'./note_error\' path.'
                    )
                os.makedirs(os.path.join('./note_error', dataset, singer), exist_ok= True)
                plt.figure(figsize= (50, 10))
                plt.plot(note_from_f0, label= 'F0 Note')
                plt.plot(note_from_midi, label= 'MIDI Note')
                plt.xticks(
                    ticks= [x for x in [0.0] + np.cumsum([note.Duration for note in music]).tolist()[:-1]],
                    labels= [note.Text for note in music],
                    )
                # for x in [x for x in [0.0] + np.cumsum(lyric_duration).tolist()[:-1]]:
                #     plt.axvline(x= x, linewidth= 0.5)
                plt.margins(x= 0)
                plt.legend()            
                plt.tight_layout()
                plt.savefig(os.path.join('./note_error', dataset, singer, f'{music_label}.png'))
                plt.close()
                return False

    current_start_mel_index = 0
    current_start_music_index = 0
    for music_current_index in range(len(music)):
        if music[music_current_index].Pitch != 0:   # not silence.
            continue
        elif current_start_music_index == music_current_index:  # avoid silence start.
            current_start_mel_index += music[music_current_index].Duration
            current_start_music_index = music_current_index + 1
            continue
        
        # note is 0(silence).
        current_mel_length_with_silence = sum([
            note.Duration
            for note in music[current_start_music_index:music_current_index + 1]
            ])
        if current_mel_length_with_silence < min_length_frame:  # mel length is shorter than min length.
            continue

        # current note is 0(silence) and mel length is over the min length.        
        current_mel_length_without_silence = sum([
            note.Duration
            for note in music[current_start_music_index:music_current_index]
            ])  # except last silence.
        
        mel_slice = mel[:, current_start_mel_index:current_start_mel_index + current_mel_length_without_silence]
        f0_slice = f0[current_start_mel_index:current_start_mel_index + current_mel_length_without_silence]
        music_slice = music[current_start_music_index:music_current_index]
        
        if tech_from_dataset is None:
            mel_slice_length = torch.IntTensor([mel_slice.shape[1]]).to(device)
            mel_slice_tensor = torch.from_numpy(
                np.pad(mel_slice, [[0, 0], [0, math.ceil(mel_slice.shape[1] / 16) * 16 - mel_slice.shape[1]]], constant_values= mel_slice.min())
                ).to(device)[None]
            f0_slice_tensor = torch.from_numpy(
                np.pad(f0_slice, [0, math.ceil(f0_slice.shape[0] / 16) * 16 - f0_slice.shape[0]], constant_values= 0)
                ).to(device)[None]
            energy_slice = spec_energy(
                y= audio_tensor[None, current_start_mel_index * hp.Sound.Hop_Size:(current_start_mel_index + current_mel_length_without_silence) * hp.Sound.Hop_Size],
                n_fft= hp.Sound.Hop_Size * 4,
                hop_size= hp.Sound.Hop_Size,
                win_size= hp.Sound.Hop_Size * 4,
                )[:, :mel_slice.shape[1]].cpu().numpy()
            energy_slice_tensor = torch.from_numpy(
                np.pad(energy_slice, [[0, 0], [0, math.ceil(energy_slice.shape[1] / 16) * 16 - energy_slice.shape[1]]], constant_values= 0)
                ).to(device)
            with torch.inference_mode():
                tech_based_mel = tech_detector_model(
                    mel_slice_tensor,
                    f0_slice_tensor,
                    energy_slice_tensor,
                    mel_slice_length
                    ).cpu().numpy()[0, :, :mel.shape[1]]
            start_index = 0
            tech = []
            for note in music:
                tech.append(tech_based_mel[:, start_index:start_index + note.Duration].sum(axis= 1))
                start_index += note.Duration
            tech = np.stack(tech, axis= 1).clip(0, 1)
        else:
            tech = tech_from_dataset

        for note, tech_note in zip(music, tech.T):
            note.Tech = tech_note

        if current_mel_length_without_silence > 0:  # silence가 엄청나게 긴 케이스는 pattern을 만들면 안됨.
            pattern = {
                'Mel': mel_slice,
                'F0': f0_slice,
                'Music': [Note_to_Dict(note) for note in music_slice],
                'Singer': singer,
                'Genre': genre,
                'Language': language,
                'Dataset': dataset
                }
            
            pattern_path = os.path.join(
                hp.Dataset_Path,
                'Train' if not is_eval_music else 'Eval',
                dataset,
                singer,
                f'{music_label}_{current_start_mel_index:08d}.pickle'
                )
            
            os.makedirs(os.path.dirname(pattern_path), exist_ok= True)
            pickle.dump(
                pattern,
                open(pattern_path, 'wb'),
                protocol= 4
                )

            torch.cuda.empty_cache()
        
        # silence를 skip해야되므로 current start는 silence를 포함해서 갱신함
        current_start_mel_index += current_mel_length_with_silence
        current_start_music_index = music_current_index + 1

    return True

def Pattern_File_Generate_by_Pattern(
    music: List[Note],
    audio: np.ndarray,
    singer: str,
    genre: str,
    language: str,
    dataset: str,
    music_label: str,
    is_eval_music: bool,
    hyper_parameters: Namespace,
    tech_from_dataset: Optional[np.ndarray]= None,
    note_error_criterion: float= 1.0,
    verbose: bool= False
    ):
    hp = hyper_parameters

    audio_tensor = torch.from_numpy(audio).to(device).float()
    with torch.inference_mode():
        mel = mel_spectrogram(
            y= audio_tensor[None],
            n_fft= hp.Sound.Hop_Size * 4,
            num_mels= hp.Sound.N_Mel,
            sampling_rate= hp.Sound.Sample_Rate,
            hop_size= hp.Sound.Hop_Size,
            win_size= hp.Sound.Hop_Size * 4,
            fmin= 0,
            fmax= None,
            center= False
            )[0].cpu().numpy() # [Batch, N_Mel, Audio_t / Hop_Size]
        
        f0 = rmvpe_model(torchaudio.functional.resample(
            audio_tensor,
            orig_freq= hp.Sound.Sample_Rate,
            new_freq= 16_000
            ))
        f0 = torch.nn.functional.interpolate(
            input= f0[None, None, :],
            size= mel.shape[1],
            mode= 'linear',
            align_corners= False
            )[0, 0, :].cpu().numpy()        
    
    music_length = sum([x.Duration for x in music])

    if mel.shape[1] > music_length:
        # print('Case1')
        mel = mel[:, math.floor((mel.shape[1] - music_length) / 2.0):-math.ceil((mel.shape[1] - music_length) / 2.0)]
        f0 = f0[math.floor((f0.shape[0] - music_length) / 2.0):-math.ceil((f0.shape[0] - music_length) / 2.0)]
    elif music_length > mel.shape[1]:
        # print('Case2')
        fix_length = music_length - mel.shape[1]
        fix_index = None
        for fix_index in reversed(range(len(music))):
            if music[fix_index].Duration <= fix_length:
                continue
            break
        music[fix_index].Duration = music[fix_index].Duration - fix_length
        
    note_from_f0 = Note_Predictor(f0).astype(np.float16)

    def Calc_Note_from_Music(
        music: List[Note],
        correction: int= 0
        ):
        return np.array([
            note.Pitch + (correction if note.Pitch > 0 else 0)
            for note in music
            for _ in range(note.Duration)
            ]).astype(np.float16)

    note_from_midi = Calc_Note_from_Music(music)
    note_from_f0_without_0, note_from_midi_without_0 = note_from_f0[(note_from_f0 > 0) * (note_from_midi > 0)], note_from_midi[(note_from_f0 > 0) * (note_from_midi > 0)]
    note_error_value = np.abs(note_from_f0_without_0 - note_from_midi_without_0).mean()

    # octave problem fix    
    if note_error_value > note_error_criterion:
        is_fixed = False
        if np.abs(note_from_f0_without_0 - note_from_midi_without_0 - 12.0).mean() < note_error_criterion:
            for note in music:
                note.Pitch = note.Pitch + 12 if note.Pitch > 0 else 0
            note_fix_from_midi = Calc_Note_from_Music(music)
            note_fix_from_midi_without_0 = note_fix_from_midi[(note_from_f0 > 0) * (note_from_midi > 0)]
            note_fix_error_value = np.abs(note_from_f0_without_0 - note_fix_from_midi_without_0).mean()
            is_fixed = True
        elif np.abs(note_from_f0_without_0 - note_from_midi_without_0 + 12.0).mean() < note_error_criterion:
            for note in music:
                note.Pitch = note.Pitch - 12 if note.Pitch > 0 else 0
            note_fix_from_midi = Calc_Note_from_Music(music)
            note_fix_from_midi_without_0 = note_fix_from_midi[(note_from_f0 > 0) * (note_from_midi > 0)]
            note_fix_error_value = np.abs(note_from_f0_without_0 - note_fix_from_midi_without_0).mean()
            is_fixed = True
        if is_fixed:
            if verbose:
                logging.warning(
                    f'the note octave of \'{dataset}-{singer}-{music_label}\' is fixed because the audio and midi note incompatible'
                    f'(note error = {note_error_value:.3f}, fixed note error = {note_fix_error_value:.3f}).'
                    'Note graph is exported at \'./note_fix\' path.'
                    )
                os.makedirs(os.path.join('./note_fix', dataset, singer), exist_ok= True)
                plt.figure(figsize= (50, 10))
                plt.plot(note_from_f0, label= 'F0 Note')
                plt.plot(note_from_midi, label= 'MIDI Note')
                plt.plot(note_fix_from_midi, label= 'Fixed MIDI Note')
                plt.xticks(
                    ticks= [x for x in [0.0] + np.cumsum([note.Duration for note in music]).tolist()[:-1]],
                    labels= [note.Text for note in music],
                    )
                for x in [x for x in [0.0] + np.cumsum([note.Duration for note in music]).tolist()[:-1]]:
                    plt.axvline(x= x, linewidth= 0.5)
                plt.margins(x= 0)
                plt.legend()
                plt.tight_layout()
                plt.savefig(os.path.join('./note_fix', dataset, singer, f'{music_label}.png'))
                plt.close()
        else:
            if verbose:
                logging.warning(
                    f'\'{dataset}-{singer}-{music_label}\' is skipped because the audio and midi note incompatible(note error = {note_error_value:.3f}).'
                    'This could be due to a misaligned octave in the MIDI or a synchronization issue between the audio and MIDI.'
                    'Note graph is exported at \'./note_error\' path.'
                    )
                os.makedirs(os.path.join('./note_error', dataset, singer), exist_ok= True)
                plt.figure(figsize= (50, 10))
                plt.plot(note_from_f0, label= 'F0 Note')
                plt.plot(note_from_midi, label= 'MIDI Note')
                plt.xticks(
                    ticks= [x for x in [0.0] + np.cumsum([note.Duration for note in music]).tolist()[:-1]],
                    labels= [note.Text for note in music],
                    )
                # for x in [x for x in [0.0] + np.cumsum(lyric_duration).tolist()[:-1]]:
                #     plt.axvline(x= x, linewidth= 0.5)
                plt.margins(x= 0)
                plt.legend()            
                plt.tight_layout()
                plt.savefig(os.path.join('./note_error', dataset, singer, f'{music_label}.png'))
                plt.close()
                return False

    # tech
    if tech_from_dataset is None:
        mel_length = torch.IntTensor([mel.shape[1]]).to(device)
        mel_tensor = torch.from_numpy(
            np.pad(mel, [[0, 0], [0, math.ceil(mel.shape[1] / 16) * 16 - mel.shape[1]]], constant_values= mel.min())
            ).to(device)[None]
        f0_tensor = torch.from_numpy(
            np.pad(f0, [0, math.ceil(f0.shape[0] / 16) * 16 - f0.shape[0]], constant_values= 0)
            ).to(device)[None]
        energy = spec_energy(
            y= audio_tensor[None],
            n_fft= hp.Sound.Hop_Size * 4,
            hop_size= hp.Sound.Hop_Size,
            win_size= hp.Sound.Hop_Size * 4,
            )[:, :mel.shape[1]].cpu().numpy()
        energy_tensor = torch.from_numpy(
            np.pad(energy, [[0, 0], [0, math.ceil(energy.shape[1] / 16) * 16 - energy.shape[1]]], constant_values= 0)
            ).to(device)
        with torch.inference_mode():
            tech_based_mel = tech_detector_model(
                mel_tensor,
                f0_tensor,
                energy_tensor,
                mel_length
                ).cpu().numpy()[0, :, :mel.shape[1]]
        start_index = 0
        tech = []
        for note in music:
            tech.append(tech_based_mel[:, start_index:start_index + note.Duration].sum(axis= 1))
            start_index += note.Duration
        tech = np.stack(tech, axis= 1).clip(0, 1)
    else:
        tech = tech_from_dataset

    for note, tech_note in zip(music, tech.T):
        note.Tech = tech_note

    pattern = {
        'Mel': mel,
        'F0': f0,
        'Music': [Note_to_Dict(note) for note in music],
        'Singer': singer,
        'Genre': genre,
        'Language': language,
        'Dataset': dataset
        }

    pattern_path = os.path.join(
        hp.Dataset_Path,
        'Train' if not is_eval_music else 'Eval',
        dataset,
        singer,
        f'{music_label}.pickle'
        )
    
    os.makedirs(os.path.dirname(pattern_path), exist_ok= True)
    pickle.dump(
        pattern,
        open(pattern_path, 'wb'),
        protocol= 4
        )

    torch.cuda.empty_cache()

    return True

def Token_Dict_Generate(hp: Namespace, tokens: List[str]):
    tokens = [x for x in tokens if x != '<X>']
    
    os.makedirs(os.path.dirname(hp.Dataset_Path), exist_ok= True)
    yaml.dump(
        {token: index for index, token in enumerate(['<S>', '<E>', '<X>'] + sorted(tokens))},
        open(os.path.join(hp.Dataset_Path, hp.Token_Path), 'w', encoding='utf-8-sig'),
        allow_unicode= True
        )


def CSD_Process(
    hyper_parameters: Namespace,
    dataset_path: str,
    verbose: bool= False
    ):
    hp = hyper_parameters

    processed_list_path = os.path.join(hp.Dataset_Path, 'Processed.txt')
    skipped_list_path = os.path.join(hp.Dataset_Path, 'Skipped.txt')
    processed_list = []
    if os.path.exists(processed_list_path):
        processed_list = \
            [x.strip() for x in open(processed_list_path, 'r', encoding= 'utf-8-sig').readlines()] + \
            [x.strip() for x in open(skipped_list_path, 'r', encoding= 'utf-8-sig').readlines()]

    korean_paths = []
    for root, _, files in os.walk(os.path.join(dataset_path, 'korean', 'wav')):
        for file in sorted(files):
            if os.path.splitext(file)[1] != '.wav':
                continue
            wav_path = os.path.join(root, file)
            midi_path = wav_path.replace('wav', 'csv')
            lyric_path = os.path.join(root.replace('wav', 'lyric'), file.replace('wav', 'txt'))
            
            if not os.path.exists(midi_path):
                raise FileExistsError(midi_path)

            korean_paths.append((wav_path, midi_path, lyric_path))

    for index, (wav_path, midi_path, lyric_path) in tqdm(
        enumerate(korean_paths),
        total= len(korean_paths),
        desc= 'CSD Kor'
        ):
        if wav_path in processed_list:
            continue

        music_label = os.path.splitext(os.path.basename(wav_path))[0]

        music = CSD.Korean(
            midi_path= midi_path,
            lyric_path= lyric_path,
            sample_rate= hp.Sound.Sample_Rate,
            hop_size= hp.Sound.Hop_Size
            )
        
        audio = Load_Audio(
            path= wav_path,
            sample_rate= hp.Sound.Sample_Rate,
            hop_size= hp.Sound.Hop_Size
            )
        
        music_length = sum([note.Duration for note in music]) * hyper_parameters.Sound.Hop_Size
        audio_length = audio.shape[0]
        
        if music_length < audio_length:
            audio = audio[:music_length]
        elif music_length > audio_length:
            cut_frame_size = (music_length - audio_length) // hyper_parameters.Sound.Hop_Size
            if music[-1].Pitch == 0 and music[-1].Duration > cut_frame_size:    # last silence cutoff
                music[-1].Duration = music[-1].Duration - cut_frame_size
            else:   # distributed cutoff
                for index in cycle(reversed([x for x in range(len(music))])):
                    music[index].Duration = music[index].Duration - 1
                    cut_frame_size -= 1
                    if cut_frame_size <= 0:
                        break

        is_generated = Pattern_File_Generate_by_Full_Song(
            music= music,
            audio= audio,
            music_label= music_label,
            singer= 'CSD',
            genre= 'Children',
            language= 'Korean',
            dataset= 'CSD',
            is_eval_music= index == 0,
            hyper_parameters= hp,
            verbose= verbose
            )

        if is_generated:
            with open(processed_list_path, 'a', encoding= 'utf-8-sig') as f:
                f.write(wav_path + '\n')

    english_paths = []
    for root, _, files in os.walk(os.path.join(dataset_path, 'english', 'wav')):
        for file in sorted(files):
            if os.path.splitext(file)[1] != '.wav':
                continue
            wav_path = os.path.join(root, file)
            midi_path = wav_path.replace('wav', 'csv')
            lyric_path = wav_path.replace('wav/', 'ipa/').replace('.wav', '.txt')
            
            if not os.path.exists(midi_path):
                raise FileExistsError(midi_path)
            if not os.path.exists(midi_path):
                raise FileExistsError(lyric_path)

            english_paths.append((wav_path, midi_path, lyric_path))

    for index, (wav_path, midi_path, lyric_path) in tqdm(
        enumerate(english_paths),
        total= len(english_paths),
        desc= 'CSD Eng'
        ):
        if wav_path in processed_list:
            continue

        music_label = os.path.splitext(os.path.basename(wav_path))[0]

        music = CSD.English(
            midi_path= midi_path,
            lyric_path= lyric_path,
            sample_rate= hp.Sound.Sample_Rate,
            hop_size= hp.Sound.Hop_Size
            )
        
        audio = Load_Audio(
            path= wav_path,
            sample_rate= hp.Sound.Sample_Rate,
            hop_size= hp.Sound.Hop_Size
            )
        
        music_length = sum([note.Duration for note in music]) * hyper_parameters.Sound.Hop_Size
        audio_length = audio.shape[0]
        
        if music_length < audio_length:
            audio = audio[:music_length]
        elif music_length > audio_length:
            cut_frame_size = (music_length - audio_length) // hyper_parameters.Sound.Hop_Size
            if music[-1].Pitch == 0 and music[-1].Duration > cut_frame_size:    # last silence cutoff
                music[-1].Duration = music[-1].Duration - cut_frame_size
            else:   # distributed cutoff
                for index in cycle(reversed([x for x in range(len(music))])):
                    music[index].Duration = music[index].Duration - 1
                    cut_frame_size -= 1
                    if cut_frame_size <= 0:
                        break

        is_generated = Pattern_File_Generate_by_Full_Song(
            music= music,
            audio= audio,
            music_label= music_label,
            singer= 'CSD',
            genre= 'Children',
            language= 'English',
            dataset= 'CSD',
            is_eval_music= index == 0,
            hyper_parameters= hp,
            verbose= verbose
            )

        if is_generated:
            with open(processed_list_path, 'a', encoding= 'utf-8-sig') as f:
                f.write(wav_path + '\n')

def AIHub_Mediazen_Process(
    hyper_parameters: Namespace,
    dataset_path: str,
    verbose: bool= False
    ):
    hp = hyper_parameters

    processed_list_path = os.path.join(hp.Dataset_Path, 'Processed.txt')
    skipped_list_path = os.path.join(hp.Dataset_Path, 'Skipped.txt')
    processed_list = []
    if os.path.exists(processed_list_path):
        processed_list = \
            [x.strip() for x in open(processed_list_path, 'r', encoding= 'utf-8-sig').readlines()] + \
            [x.strip() for x in open(skipped_list_path, 'r', encoding= 'utf-8-sig').readlines()]

    path_dict = {}
    for root, _, files in os.walk(os.path.join(dataset_path)):
        for file in files:
            key, extension = os.path.splitext(file)
            if not extension.upper() in ['.WAV', '.MID']:
                continue
            
            if not key in path_dict.keys():
                path_dict[key] = {}
            path_dict[key]['wav' if extension.upper() == '.WAV' else 'mid'] = os.path.join(root, file)            

    paths = [
        (value['wav'], value['mid'], key.strip().split('_')[0].upper(), 'AMZ_' + key.strip().split('_')[4].upper())
        for key, value in path_dict.items()
        if 'wav' in value.keys() and 'mid' in value.keys()
        ]
    paths = [
        path for path in paths
        if not os.path.basename(path[0]) in [
            'ba_10754_+2_s_yej_f_04.wav',
            'ro_15699_+0_s_s18_m_04.wav',
            'ro_23930_-2_a_lsb_f_02.wav'
            ]
        ]   # invalid patterns
    genre_dict = {
        'BA': 'Ballade',
        'RO': 'Rock',
        'TR': 'Trot'
        }
    
    eval_cycle = len(paths) // int(len(paths) * 0.001)
    for pattern_index, (wav_path, midi_path, genre, singer) in tqdm(
        enumerate(paths),
        total= len(paths),
        desc= 'AIHub_Mediazen'
        ):
        if wav_path in processed_list:
            continue

        music_label = os.path.splitext(os.path.basename(wav_path))[0]        
        genre = genre_dict[genre]

        music = AIHub_Mediazen.Process(
            midi_path= midi_path,
            sample_rate= hp.Sound.Sample_Rate,
            hop_size= hp.Sound.Hop_Size
            )
        
        audio = Load_Audio(
            path= wav_path,
            sample_rate= hp.Sound.Sample_Rate,
            hop_size= hp.Sound.Hop_Size
            )
        
        music_length = sum([note.Duration for note in music]) * hyper_parameters.Sound.Hop_Size
        audio_length = audio.shape[0]

        if abs(music_length - audio_length) > hyper_parameters.Sound.Sample_Rate:
            logging.warning(f'{midi_path} is skipped because of async between audio and midi.({music_length} vs {audio_length})')
            with open(skipped_list_path, 'a', encoding= 'utf-8-sig') as f:
                f.write(wav_path + '\n')
            continue
        elif music_length < audio_length:
            audio = audio[:music_length]
        elif music_length > audio_length:
            cut_frame_size = (music_length - audio_length) // hyper_parameters.Sound.Hop_Size
            if music[-1].Pitch == 0 and music[-1].Duration > cut_frame_size:    # last silence cutoff
                music[-1].Duration = music[-1].Duration - cut_frame_size
            else:   # distributed cutoff
                for index in cycle(reversed([x for x in range(len(music))])):
                    if music[index].Duration > 10:
                        music[index].Duration = music[index].Duration - 1
                        cut_frame_size -= 1
                        if cut_frame_size <= 0:
                            break

        is_generated = Pattern_File_Generate_by_Full_Song(
            music= music,
            audio= audio,
            music_label= music_label,
            singer= singer,
            genre= genre,
            language= 'Korean',
            dataset= 'AIHub_Mediazen',
            is_eval_music= pattern_index % eval_cycle == 0,
            hyper_parameters= hp,
            verbose= verbose
            )

        if is_generated:
            with open(processed_list_path, 'a', encoding= 'utf-8-sig') as f:
                f.write(wav_path + '\n')

def AIHub_Metabuild_Process(
    hyper_parameters: Namespace,
    dataset_path: str,
    verbose: bool= False
    ):
    hp = hyper_parameters

    processed_list_path = os.path.join(hp.Dataset_Path, 'Processed.txt')
    skipped_list_path = os.path.join(hp.Dataset_Path, 'Skipped.txt')
    processed_list = []
    if os.path.exists(processed_list_path):
        processed_list = \
            [x.strip() for x in open(processed_list_path, 'r', encoding= 'utf-8-sig').readlines()] + \
            [x.strip() for x in open(skipped_list_path, 'r', encoding= 'utf-8-sig').readlines()]

    wav_path_dict, json_path_dict = {}, {}
    for root, _, files in os.walk(dataset_path):
        for file in sorted(files):
            if os.path.splitext(file)[1].lower() == '.wav':
                wav_path_dict[os.path.splitext(file)[0]] = os.path.join(root, file)
            elif os.path.splitext(file)[1].lower() == '.json':
                json_path_dict[os.path.splitext(file)[0]] = os.path.join(root, file)

    paths = sorted([
        (wav_path, json_path_dict[key])
        for key, wav_path in wav_path_dict.items()
        if key in json_path_dict.keys()
        ])
    

    eval_cycle = len(paths) // int(len(paths) * 0.001)
    for pattern_index, (wav_path, json_path) in tqdm(
        enumerate(paths),
        total= len(paths),
        desc= 'AIHub_Metabuild'
        ):
        if wav_path in processed_list:
            continue

        _, singer, _, _, _, genre, music_label = os.path.splitext(os.path.basename(wav_path))[0].strip().split('_')
        singer = 'AMB_S' + singer
        genre = genre.capitalize()

        music = AIHub_Metabuild.Process(
            json_path= json_path,
            sample_rate= hp.Sound.Sample_Rate,
            hop_size= hp.Sound.Hop_Size
            )
        
        audio = Load_Audio(
            path= wav_path,
            sample_rate= hp.Sound.Sample_Rate,
            hop_size= hp.Sound.Hop_Size
            )
        
        music_length = sum([note.Duration for note in music]) * hyper_parameters.Sound.Hop_Size
        audio_length = audio.shape[0]

        if music_length < audio_length:
            audio = audio[:music_length]    # AIHub Metabuild's last silence is no tagged.
        elif music_length - audio_length > hyper_parameters.Sound.Sample_Rate:
            logging.warning(f'{json_path} is skipped because of async between audio and midi.({music_length} vs {audio_length})')
            with open(skipped_list_path, 'a', encoding= 'utf-8-sig') as f:
                f.write(wav_path + '\n')
            continue
        elif music_length > audio_length:
            cut_frame_size = (music_length - audio_length) // hyper_parameters.Sound.Hop_Size
            if music[-1].Pitch == 0 and music[-1].Duration > cut_frame_size:    # last silence cutoff
                music[-1].Duration = music[-1].Duration - cut_frame_size
            else:   # distributed cutoff
                for index in cycle(reversed([x for x in range(len(music))])):
                    if music[index].Duration > 10:
                        music[index].Duration = music[index].Duration - 1
                        cut_frame_size -= 1
                        if cut_frame_size <= 0:
                            break

        is_generated = Pattern_File_Generate_by_Full_Song(
            music= music,
            audio= audio,
            music_label= music_label,
            singer= singer,
            genre= genre,
            language= 'Korean',
            dataset= 'AIHub_Metabuild',
            is_eval_music= pattern_index % eval_cycle == 0,
            hyper_parameters= hp,
            verbose= verbose
            )

        if is_generated:
            with open(processed_list_path, 'a', encoding= 'utf-8-sig') as f:
                f.write(wav_path + '\n')

def M4Singer_Process(
    hyper_parameters: Namespace,
    dataset_path: str,
    verbose: bool= False
    ):
    hp = hyper_parameters

    processed_list_path = os.path.join(hp.Dataset_Path, 'Processed.txt')
    skipped_list_path = os.path.join(hp.Dataset_Path, 'Skipped.txt')
    processed_list = []
    if os.path.exists(processed_list_path):
        processed_list = \
            [x.strip() for x in open(processed_list_path, 'r', encoding= 'utf-8-sig').readlines()] + \
            [x.strip() for x in open(skipped_list_path, 'r', encoding= 'utf-8-sig').readlines()]

    wav_paths = sorted([
        os.path.join(root, file)
        for root, _, files in os.walk(dataset_path)
        for file in files
        if os.path.splitext(file)[1] == '.wav'
        ])
    midi_paths = [
        wav_path.replace('.wav', '.mid')
        for wav_path in wav_paths
        ]
    tg_paths = [
        wav_path.replace('.wav', '.TextGrid')
        for wav_path in wav_paths
        ]
    
    eval_cycle = len(wav_paths) // int(len(wav_paths) * 0.001)
    for pattern_index, (wav_path, midi_path, tg_path) in tqdm(
        enumerate(zip(wav_paths, midi_paths, tg_paths)),
        total= len(wav_paths),
        desc= 'M4Singer'
        ):
        if wav_path in processed_list:
            continue

        singer = 'M4' + wav_path.split('/')[-2].split('#')[0].replace('-', '').strip()
        music_label = \
            wav_path.split('/')[-2].split('#')[1].strip() + \
            '_' + os.path.splitext(os.path.basename(wav_path))[0]

        if not os.path.exists(midi_path) or not os.path.exists(tg_path):
            print(f'There is no other file about wav file: {wav_path}')
            continue

        music = M4Singer.Process(
            midi_path= midi_path,
            tg_path= tg_path,
            sample_rate= hp.Sound.Sample_Rate,
            hop_size= hp.Sound.Hop_Size
            )
        
        audio = Load_Audio(
            path= wav_path,
            sample_rate= hp.Sound.Sample_Rate,
            hop_size= hp.Sound.Hop_Size
            )
        
        music_length = sum([note.Duration for note in music]) * hyper_parameters.Sound.Hop_Size
        audio_length = audio.shape[0]
        
        if abs(music_length - audio_length) > hp.Sound.Sample_Rate:
            logging.warning(f'{midi_path} is skipped because of async between audio and midi.({music_length} vs {audio_length})')
            with open(skipped_list_path, 'a', encoding= 'utf-8-sig') as f:
                f.write(wav_path + '\n')
            continue
        elif music_length < audio_length:
            audio = audio[:music_length]
        elif music_length > audio_length:
            cut_frame_size = (music_length - audio_length) // hp.Sound.Hop_Size
            if music[-1].Pitch == 0 and music[-1].Duration > cut_frame_size:    # last silence cutoff
                music[-1].Duration = music[-1].Duration - cut_frame_size
            else:   # distributed cutoff
                for index in cycle(reversed([x for x in range(len(music))])):
                    music[index].Duration = music[index].Duration - 1
                    cut_frame_size -= 1
                    if cut_frame_size <= 0:
                        break

        is_generated = Pattern_File_Generate_by_Pattern(
            music= music,
            audio= audio,
            music_label= music_label,
            singer= singer,
            genre= 'Pop',
            language= 'Chinese',
            dataset= 'M4Singer',
            is_eval_music= pattern_index % eval_cycle == 0,
            hyper_parameters= hp,
            note_error_criterion= 1.5,
            verbose= verbose
            )

        if is_generated:
            with open(processed_list_path, 'a', encoding= 'utf-8-sig') as f:
                f.write(wav_path + '\n')

def OpenCPop_Process(
    hyper_parameters: Namespace,
    dataset_path: str,
    verbose: bool= False
    ):
    hp = hyper_parameters

    processed_list_path = os.path.join(hp.Dataset_Path, 'Processed.txt')
    skipped_list_path = os.path.join(hp.Dataset_Path, 'Skipped.txt')
    processed_list = []
    if os.path.exists(processed_list_path):
        processed_list = \
            [x.strip() for x in open(processed_list_path, 'r', encoding= 'utf-8-sig').readlines()] + \
            [x.strip() for x in open(skipped_list_path, 'r', encoding= 'utf-8-sig').readlines()]

    for script_file, is_eval in [('train.txt', False), ('test.txt', True)]:
        lines = open(
            os.path.join(dataset_path, 'segments', script_file),
            mode= 'r',
            encoding= 'utf-8-sig'
            ).readlines()
        for line in tqdm(
            lines,
            total= len(lines),
            desc= 'OpenCPop Eval' if is_eval else 'OpenCPop Train'
            ):
            wav_path, text, phoneme, note, duration, _, slur = line.strip().split('|')
            wav_path = os.path.join(dataset_path, 'segments', 'wavs', f'{wav_path}.wav')
            if wav_path in processed_list:
                continue

            music_label = os.path.splitext(os.path.basename(wav_path))[0]

            music = OpenCPop.Process(
                text= text,
                phoneme= phoneme,
                note= note,
                duration= duration,
                slur= slur,
                music_label= music_label,
                sample_rate= hp.Sound.Sample_Rate,
                hop_size= hp.Sound.Hop_Size
                )

            audio = librosa.load(wav_path, sr= hyper_parameters.Sound.Sample_Rate)[0]
            audio = audio[:audio.shape[0] - (audio.shape[0] % hyper_parameters.Sound.Hop_Size)]

            music_length = sum([note.Duration for note in music]) * hp.Sound.Hop_Size
            audio_length = audio.shape[0]

            if abs(music_length - audio_length) > hp.Sound.Sample_Rate:
                logging.warning(f'{script_file} is skipped because of async between audio and midi.({music_length} vs {audio_length})')
                with open(skipped_list_path, 'a', encoding= 'utf-8-sig') as f:
                    f.write(wav_path + '\n')
                continue
            elif music_length < audio_length:
                audio = audio[:music_length]
            elif music_length > audio_length:
                cut_frame_size = (music_length - audio_length) // hp.Sound.Hop_Size
                if music[-1].Pitch == 0 and music[-1].Duration > cut_frame_size:    # last silence cutoff
                    music[-1].Duration = music[-1].Duration - cut_frame_size
                else:   # distributed cutoff
                    for index in cycle(reversed([x for x in range(len(music))])):
                        music[index].Duration = music[index].Duration - 1
                        cut_frame_size -= 1
                        if cut_frame_size <= 0:
                            break

            is_generated = Pattern_File_Generate_by_Pattern(
                music= music,
                audio= audio,
                music_label= music_label,
                singer= 'OpenCPop',
                genre= 'Pop',
                language= 'Chinese',
                dataset= 'OpenCPop',
                is_eval_music= is_eval,
                hyper_parameters= hyper_parameters,
                note_error_criterion= 1.5,
                verbose= verbose
                )
            
            if is_generated:
                with open(processed_list_path, 'a', encoding= 'utf-8-sig') as f:
                    f.write(wav_path + '\n')

def Kiritan_Process(
    hyper_parameters: Namespace,
    dataset_path: str,
    verbose: bool= False
    ):
    hp = hyper_parameters

    processed_list_path = os.path.join(hp.Dataset_Path, 'Processed.txt')
    skipped_list_path = os.path.join(hp.Dataset_Path, 'Skipped.txt')
    processed_list = []
    if os.path.exists(processed_list_path):
        processed_list = \
            [x.strip() for x in open(processed_list_path, 'r', encoding= 'utf-8-sig').readlines()] + \
            [x.strip() for x in open(skipped_list_path, 'r', encoding= 'utf-8-sig').readlines()]

    music_label_dict = {
        1: 'Color',
        2: 'らむねサンセット',
        3: 'イチズ',
        4: 'i☆Doloid',
        5: 'Get Over',
        6: '§Rainbow',
        7: '進めアバンチュール',
        8: 'もってけ!セーラーふく',
        9: '幻想曲WONDERLAND',
        10: 'Last Moment',
        11: 'EDGE OF HEAVEN',
        12: '徒太陽',
        13: 'Happy New World☆',
        14: 'Dream☆Land',
        15: 'Make it !',
        16: 'ユメノツバサ',
        17: 'Secret Pure Love',
        18: 'ミラクル☆パラダイス',
        19: 'ココロノヲト',
        20: 'Special Kiss',
        21: 'Realize!',
        22: 'ミライノナマエ',
        23: 'わくドキしたいっ!',
        24: 'ドリームパレード',
        25: 'Summer Vacation Love',
        26: 'ブライトファンタジー',
        27: 'NEXTAGE',
        28: 'ハチャメチャ×ストライク',
        29: 'Goin\'on',
        30: 'Baby…',
        31: 'キラリ',
        32: 'Ready Smile!!',
        33: 'trust',
        34: 'Garnet ',
        35: 'Re：Call',
        36: '鏡のLabyrinth',
        37: '真夏の花火と秘密基地',
        38: 'Shining Star',
        39: 'jewel',
        40: 'HERO ',
        41: 'Memorial',
        42: 'キラキラGood day',
        43: '漢',
        44: 'Changing point',
        45: 'アメコイ',
        46: '卒業式',
        47: 'Endless Notes',
        48: 'イノセントイノベーション',
        49: 'アルティメット☆MAGIC',
        50: 'ありえんほどフィーバー',
        }

    paths = []
    for index in range(1, 51):
        wav_path = os.path.join(dataset_path, 'wav', f'{index:02d}.wav')
        score_path = os.path.join(dataset_path, 'musicxml', f'{index:02d}.xml')
        if not os.path.exists(wav_path):
            raise FileExistsError(wav_path)
        if not os.path.exists(score_path):
            raise FileExistsError(score_path)

        paths.append((wav_path, score_path))
    paths = sorted(paths, key= lambda x: x[0])

    for pattern_index, (wav_path, score_path) in tqdm(
        enumerate(paths),
        total= len(paths),
        desc= 'Kiritan'
        ):
        if wav_path in processed_list:
            continue

        music_label = os.path.splitext(os.path.basename(wav_path))[0]
        music_label = music_label_dict[int(music_label)]

        music = Kiritan.Process(
            score_path= score_path,
            sample_rate= hp.Sound.Sample_Rate,
            hop_size= hp.Sound.Hop_Size
            )
        
        audio = Load_Audio(
            path= wav_path,
            sample_rate= hp.Sound.Sample_Rate,
            hop_size= hp.Sound.Hop_Size
            )

        music_length = sum([note.Duration for note in music]) * hp.Sound.Hop_Size
        audio_length = audio.shape[0]
        
        if music_length < audio_length:
            audio = audio[:music_length]
        elif music_length > audio_length:
            cut_frame_size = (music_length - audio_length) // hyper_parameters.Sound.Hop_Size
            if music[-1].Pitch == 0 and music[-1].Duration > cut_frame_size:    # last silence cutoff
                music[-1].Duration = music[-1].Duration - cut_frame_size
            else:   # distributed cutoff
                for index in cycle(reversed([x for x in range(len(music))])):
                    music[index].Duration = music[index].Duration - 1
                    cut_frame_size -= 1
                    if cut_frame_size <= 0:
                        break

        is_generated = Pattern_File_Generate_by_Full_Song(
            music= music,
            audio= audio,
            music_label= music_label,
            singer= 'Kiritan',
            genre= 'Pop',
            language= 'Japanese',
            dataset= 'Kiritan',
            is_eval_music= pattern_index == 0,
            hyper_parameters= hp,
            verbose= verbose
            )

        if is_generated:
            with open(processed_list_path, 'a', encoding= 'utf-8-sig') as f:
                f.write(wav_path + '\n')

def Ofuton_Process(
    hyper_parameters: Namespace,
    dataset_path: str,
    verbose: bool= False
    ):
    hp = hyper_parameters

    processed_list_path = os.path.join(hp.Dataset_Path, 'Processed.txt')
    skipped_list_path = os.path.join(hp.Dataset_Path, 'Skipped.txt')
    processed_list = []
    if os.path.exists(processed_list_path):
        processed_list = \
            [x.strip() for x in open(processed_list_path, 'r', encoding= 'utf-8-sig').readlines()] + \
            [x.strip() for x in open(skipped_list_path, 'r', encoding= 'utf-8-sig').readlines()]

    paths = []
    for root, _, files in os.walk(dataset_path):
        for file in files:
            if os.path.splitext(file)[1] != '.wav':
                continue
            wav_path = os.path.join(root, file)
            score_path = wav_path.replace('.wav', '.musicxml')
            paths.append((wav_path, score_path))

    paths = sorted(paths, key= lambda x: x[0])
    
    for pattern_index, (wav_path, score_path) in tqdm(
        enumerate(paths),
        total= len(paths),
        desc= 'Ofuton'
        ):
        if wav_path in processed_list:
            continue
        
        music_label = os.path.splitext(os.path.basename(wav_path))[0]     

        music = Ofuton.Process(
            score_path= score_path,
            sample_rate= hp.Sound.Sample_Rate,
            hop_size= hp.Sound.Hop_Size
            )
        
        audio = Load_Audio(
            path= wav_path,
            sample_rate= hp.Sound.Sample_Rate,
            hop_size= hp.Sound.Hop_Size
            )

        music_length = sum([note.Duration for note in music]) * hp.Sound.Hop_Size
        audio_length = audio.shape[0]

        if music_length < audio_length:
            audio = audio[:music_length]
        elif music_length > audio_length:
            cut_frame_size = (music_length - audio_length) // hyper_parameters.Sound.Hop_Size
            if music[-1].Pitch == 0 and music[-1].Duration > cut_frame_size:    # last silence cutoff
                music[-1].Duration = music[-1].Duration - cut_frame_size
            else:   # distributed cutoff
                for index in cycle(reversed([x for x in range(len(music))])):
                    music[index].Duration = music[index].Duration - 1
                    cut_frame_size -= 1
                    if cut_frame_size <= 0:
                        break

        is_generated = Pattern_File_Generate_by_Full_Song(
            music= music,
            audio= audio,
            music_label= music_label,
            singer= 'Ofuton',
            genre= 'Pop',
            language= 'Japanese',
            dataset= 'Ofuton',
            is_eval_music= pattern_index == 0,
            hyper_parameters= hp,
            verbose= verbose
            )

        if is_generated:
            with open(processed_list_path, 'a', encoding= 'utf-8-sig') as f:
                f.write(wav_path + '\n')

def GTSinger_Process(
    hyper_parameters: Namespace,
    dataset_path: str,
    verbose: bool= False
    ):
    hp = hyper_parameters

    processed_list_path = os.path.join(hp.Dataset_Path, 'Processed.txt')
    skipped_list_path = os.path.join(hp.Dataset_Path, 'Skipped.txt')
    processed_list = []
    if os.path.exists(processed_list_path):
        processed_list = \
            [x.strip() for x in open(processed_list_path, 'r', encoding= 'utf-8-sig').readlines()] + \
            [x.strip() for x in open(skipped_list_path, 'r', encoding= 'utf-8-sig').readlines()]

    with open(os.path.join(dataset_path, 'processed', 'All', 'metadata.json'), 'r', encoding= 'utf-8-sig') as f:
        metadata_list = json.load(f)
    metadata_list = [
        metadata
        for metadata in metadata_list
        if metadata['language'] in [
            'Chinese', 'English', 'Korean',
            # 'Spanish', 'French', 'German', 'Italian', 'Russian',
            # 'Japanese', -> the Japanese kanji phonemes in GTSinger are incorrect. This cannot be use now.
            ]
        ]
    eval_cycle = len(metadata_list) // int(len(metadata_list) * 0.001)

    for pattern_index, metadata in tqdm(
        enumerate(metadata_list),
        total= len(metadata_list),
        desc= 'GTSinger'
        ):
        wav_path = os.path.join(dataset_path, metadata['wav_fn'])
        if wav_path in processed_list:
            continue
        singer = metadata['singer']
        language = metadata['language']
        music_label = '#'.join(metadata['item_name'].replace(' ', '_').split('#')[2:])

        # music structure: [duration(sec), lyric, note, text] 
        # techs: np.ndarray
        if language == 'Chinese':
            music, tech = GTSinger.Chinese(
                metadata= metadata,
                sample_rate= hp.Sound.Sample_Rate,
                hop_size= hp.Sound.Hop_Size
                )
        elif language == 'English':
            music, tech = GTSinger.English(
                metadata= metadata,
                sample_rate= hp.Sound.Sample_Rate,
                hop_size= hp.Sound.Hop_Size
                )
        elif language == 'Japanese':
            music, tech = GTSinger.Japanese(
                metadata= metadata,
                sample_rate= hp.Sound.Sample_Rate,
                hop_size= hp.Sound.Hop_Size
                )
        elif language == 'Korean':
            music, tech = GTSinger.Korean(
                metadata= metadata,
                sample_rate= hp.Sound.Sample_Rate,
                hop_size= hp.Sound.Hop_Size
                )
        else:
            print(language)
            assert False

        if music is None:
            with open(skipped_list_path, 'a', encoding= 'utf-8-sig') as f:
                f.write(wav_path + '\n')
            continue

        audio = Load_Audio(
            path= wav_path,
            sample_rate= hp.Sound.Sample_Rate,
            hop_size= hp.Sound.Hop_Size
            )
        
        music_length = sum([note.Duration for note in music]) * hyper_parameters.Sound.Hop_Size
        audio_length = audio.shape[0]
        
        if music_length < audio_length:
            audio = audio[:music_length]
        elif music_length > audio_length:
            cut_frame_size = (music_length - audio_length) // hyper_parameters.Sound.Hop_Size
            if music[-1].Pitch == 0 and music[-1].Duration > cut_frame_size:    # last silence cutoff
                music[-1].Duration = music[-1].Duration - cut_frame_size
            else:   # distributed cutoff
                for index in cycle(reversed([x for x in range(len(music))])):
                    music[index].Duration = music[index].Duration - 1
                    cut_frame_size -= 1
                    if cut_frame_size <= 0:
                        break

        is_generated = Pattern_File_Generate_by_Pattern(
            music= music,
            audio= audio,
            music_label= music_label,
            singer= singer,
            genre= 'Pop',
            language= language,
            dataset= 'GTSinger',
            is_eval_music= pattern_index % eval_cycle == 0,
            hyper_parameters= hyper_parameters,
            tech_from_dataset= tech,
            note_error_criterion= 1.0,
            verbose= verbose
            )

        if is_generated:
            with open(processed_list_path, 'a', encoding= 'utf-8-sig') as f:
                f.write(wav_path + '\n')

def Metadata_Generate(
    hp: Namespace,
    eval: bool= False
    ):
    pattern_path = os.path.join(hp.Dataset_Path, 'Train' if not eval else 'Eval')

    mel_dict = {
        'Total': {
            'Min': math.inf,
            'Max': -math.inf,
            'Sum': 0.0,
            'Sum_of_Square': 0.0,
            'Count': 0.0
            }
        }
    f0_dict = {
        'Total': {
            'Min': math.inf,
            'Max': -math.inf,
            'Sum': 0.0,
            'Sum_of_Square': 0.0,
            'Count': 0.0
            }
        }
    tokens = []
    singers = []
    genres = []
    languages = []
    min_pitch, max_pitch = math.inf, -math.inf

    new_metadata_dict = {
        'Hop_Size': hp.Sound.Hop_Size,
        'Sample_Rate': hp.Sound.Sample_Rate,
        'File_List': [],
        'Mel_Length_Dict': {},
        'F0_Length_Dict': {},
        'Music_Length_Dict': {},
        'Singer_Dict': {},
        'Genre_Dict': {},
        'Language_Dict': {},
        'File_List_by_Singer_Dict': {},
        }

    files_tqdm = tqdm(
        total= sum([len(files) for root, _, files in os.walk(pattern_path, followlinks= True)]),
        desc= 'Eval_Pattern' if eval else 'Train_Pattern'
        )

    for root, _, files in os.walk(pattern_path, followlinks= True):
        for file in files:
            # print(os.path.join(root, file))
            with open(os.path.join(root, file), 'rb') as f:
                pattern_dict = pickle.load(f)                
            file = os.path.join(root, file).replace(pattern_path, '').lstrip('/')
            if not all([
                key in pattern_dict.keys()
                for key in ('Mel', 'F0', 'Music', 'Singer', 'Genre', 'Dataset')
                ]):
                print(f'Skipped pickle file \'{file}\' because of insufficient dict_keys: {pattern_dict.keys()}')
                continue
            pattern_dict['Mel'] = pattern_dict['Mel'].astype(object)
            pattern_dict['F0'] = pattern_dict['F0'].astype(object)
            pattern_dict['Music'] = [Dict_to_Note(x) for x in pattern_dict['Music']]

            new_metadata_dict['Mel_Length_Dict'][file] = pattern_dict['Mel'].shape[1]
            new_metadata_dict['F0_Length_Dict'][file] = pattern_dict['F0'].shape[0]
            new_metadata_dict['Music_Length_Dict'][file] = len(pattern_dict['Music'])
            new_metadata_dict['Singer_Dict'][file] = pattern_dict['Singer']
            new_metadata_dict['File_List'].append(file)
            if not pattern_dict['Singer'] in new_metadata_dict['File_List_by_Singer_Dict'].keys():
                new_metadata_dict['File_List_by_Singer_Dict'][pattern_dict['Singer']] = []
            new_metadata_dict['File_List_by_Singer_Dict'][pattern_dict['Singer']].append(file)

            if not pattern_dict['Singer'] in mel_dict.keys():
                mel_dict[pattern_dict['Singer']] = {
                    'Min': math.inf,
                    'Max': -math.inf,
                    'Sum': 0.0,
                    'Sum_of_Square': 0.0,
                    'Count': 0.0
                    }
            if not pattern_dict['Singer'] in f0_dict.keys():
                f0_dict[pattern_dict['Singer']] = {
                    'Min': math.inf,
                    'Max': -math.inf,
                    'Sum': 0.0,
                    'Sum_of_Square': 0.0,
                    'Count': 0.0
                    }
                
            mel_dict['Total']['Min'] = min(mel_dict['Total']['Min'], pattern_dict['Mel'].min())
            mel_dict['Total']['Max'] = max(mel_dict['Total']['Max'], pattern_dict['Mel'].max())
            mel_dict['Total']['Sum'] = mel_dict['Total']['Sum'] + pattern_dict['Mel'].sum()
            mel_dict['Total']['Sum_of_Square'] = mel_dict['Total']['Sum_of_Square'] + (pattern_dict['Mel'] ** 2.0).sum()
            mel_dict['Total']['Count'] = mel_dict['Total']['Count'] + pattern_dict['Mel'].size

            mel_dict[pattern_dict['Singer']]['Min'] = min(mel_dict[pattern_dict['Singer']]['Min'], pattern_dict['Mel'].min())
            mel_dict[pattern_dict['Singer']]['Max'] = max(mel_dict[pattern_dict['Singer']]['Max'], pattern_dict['Mel'].max())
            mel_dict[pattern_dict['Singer']]['Sum'] = mel_dict[pattern_dict['Singer']]['Sum'] + pattern_dict['Mel'].sum()
            mel_dict[pattern_dict['Singer']]['Sum_of_Square'] = mel_dict[pattern_dict['Singer']]['Sum_of_Square'] + (pattern_dict['Mel'] ** 2.0).sum()
            mel_dict[pattern_dict['Singer']]['Count'] = mel_dict[pattern_dict['Singer']]['Count'] + pattern_dict['Mel'].size

            f0_dict['Total']['Min'] = min(f0_dict['Total']['Min'], pattern_dict['F0'].min())
            f0_dict['Total']['Max'] = max(f0_dict['Total']['Max'], pattern_dict['F0'].max())
            f0_dict['Total']['Sum'] = f0_dict['Total']['Sum'] + pattern_dict['F0'].sum()
            f0_dict['Total']['Sum_of_Square'] = f0_dict['Total']['Sum_of_Square'] + (pattern_dict['F0'] ** 2.0).sum()
            f0_dict['Total']['Count'] = f0_dict['Total']['Count'] + pattern_dict['F0'].size

            f0_dict[pattern_dict['Singer']]['Min'] = min(f0_dict[pattern_dict['Singer']]['Min'], pattern_dict['F0'].min())
            f0_dict[pattern_dict['Singer']]['Max'] = max(f0_dict[pattern_dict['Singer']]['Max'], pattern_dict['F0'].max())
            f0_dict[pattern_dict['Singer']]['Sum'] = f0_dict[pattern_dict['Singer']]['Sum'] + pattern_dict['F0'].sum()
            f0_dict[pattern_dict['Singer']]['Sum_of_Square'] = f0_dict[pattern_dict['Singer']]['Sum_of_Square'] + (pattern_dict['F0'] ** 2.0).sum()
            f0_dict[pattern_dict['Singer']]['Count'] = f0_dict[pattern_dict['Singer']]['Count'] + pattern_dict['F0'].size

            tokens.extend([phoneme for note in pattern_dict['Music'] for phoneme in note.Lyric])
            singers.append(pattern_dict['Singer'])
            genres.append(pattern_dict['Genre'])
            languages.append(pattern_dict['Language'])
            min_pitch = min(min_pitch, *[note.Pitch for note in pattern_dict['Music'] if note.Pitch > 0])
            max_pitch = max(max_pitch, *[note.Pitch for note in pattern_dict['Music'] if note.Pitch > 0])

            files_tqdm.update(1)

    for singer in mel_dict.keys():
        mel_dict[singer]['Mean'] = mel_dict[singer]['Sum'] / mel_dict[singer]['Count']
        mel_dict[singer]['Std'] = math.sqrt(mel_dict[singer]['Sum_of_Square'] / mel_dict[singer]['Count'] - (mel_dict[singer]['Mean'] ** 2))
        del mel_dict[singer]['Sum']
        del mel_dict[singer]['Sum_of_Square']
        del mel_dict[singer]['Count']

    for singer in f0_dict.keys():
        f0_dict[singer]['Mean'] = f0_dict[singer]['Sum'] / f0_dict[singer]['Count']
        f0_dict[singer]['Std'] = math.sqrt(f0_dict[singer]['Sum_of_Square'] / f0_dict[singer]['Count'] - (f0_dict[singer]['Mean'] ** 2))
        del f0_dict[singer]['Sum']
        del f0_dict[singer]['Sum_of_Square']
        del f0_dict[singer]['Count']
    
    new_metadata_dict['Min_Pitch'] = min_pitch
    new_metadata_dict['Max_Pitch'] = max_pitch

    with open(os.path.join(pattern_path, 'METADATA.PICKLE'), 'wb') as f:
        pickle.dump(new_metadata_dict, f, protocol= 4)

    if not eval:
        yaml.dump(
            mel_dict,
            open(os.path.join(hp.Dataset_Path, hp.Mel_Info_Path), 'w')
            )
        
        yaml.dump(
            f0_dict,
            open(os.path.join(hp.Dataset_Path, hp.F0_Info_Path), 'w')
            )
        Token_Dict_Generate(hp, sorted(list(set(tokens))))

        singer_index_dict = {
            singer: index
            for index, singer in enumerate(sorted(set(singers)))
            }
        yaml.dump(
            singer_index_dict,
            open(os.path.join(hp.Dataset_Path, hp.Singer_Info_Path), 'w')
            )

        genre_index_dict = {
            genre: index
            for index, genre in enumerate(sorted(set(genres)))
            }
        yaml.dump(
            genre_index_dict,
            open(os.path.join(hp.Dataset_Path, hp.Genre_Info_Path), 'w')
            )
        
        language_index_dict = {
            language: index
            for index, language in enumerate(sorted(set(languages)))
            }
        yaml.dump(
            language_index_dict,
            open(os.path.join(hp.Dataset_Path, hp.Language_Info_Path), 'w')
            )

    logging.info('Metadata generate done.')

if __name__ == '__main__':
    argparser = argparse.ArgumentParser()
    argparser.add_argument('-csd', '--csd_path', required= False)
    argparser.add_argument('-amz', '--amz_path', required= False)
    argparser.add_argument('-amb', '--amb_path', required= False)
    argparser.add_argument('-m4singer', '--m4singer_path', required= False)
    argparser.add_argument('-opencpop', '--opencpop_path', required= False)

    argparser.add_argument('-kiritan', '--kiritan_path', required= False)
    argparser.add_argument('-ofuton', '--ofuton_path', required= False)

    argparser.add_argument('-gtsinger', '--gtsinger_path', required= False)
    argparser.add_argument('-hp', '--hyper_parameters', required= True)
    argparser.add_argument('-verbose', '--verbose', action= 'store_true')
    args = argparser.parse_args()

    hp = Recursive_Parse(yaml.load(
        open(args.hyper_parameters, encoding='utf-8'),
        Loader=yaml.Loader
        ))

    if not torch.cuda.is_available():
        device = torch.device('cpu')
        rmvpe_model = torch.jit.load('rmvpe_cpu.pts', map_location= 'cpu')
        tech_detector_model = torch.jit.load('Tech_Detector.pts', map_location= 'cpu')
    else:
        device = torch.device('cuda:0')
        rmvpe_model = torch.jit.load('rmvpe_cuda.pts', map_location= 'cuda:0')
        tech_detector_model = torch.jit.load('Tech_Detector.pts', map_location= 'cuda:0')


    processed_list_path = os.path.join(hp.Dataset_Path, 'Processed.txt')
    skipped_list_path = os.path.join(hp.Dataset_Path, 'Skipped.txt')
    os.makedirs(hp.Dataset_Path, exist_ok= True)
    if not os.path.exists(processed_list_path):
        with open(processed_list_path, 'w', encoding= 'utf-8-sig') as f:
            f.write('')
    if not os.path.exists(skipped_list_path):
        with open(skipped_list_path, 'w', encoding= 'utf-8-sig') as f:
            f.write('')

    if args.csd_path:
        CSD_Process(
            hyper_parameters= hp,
            dataset_path= args.csd_path,
            verbose= args.verbose
            )
    
    if args.amz_path:
        AIHub_Mediazen_Process(
            hyper_parameters= hp,
            dataset_path= args.amz_path,
            verbose= args.verbose
            )

    if args.amb_path:
        AIHub_Metabuild_Process(
            hyper_parameters= hp,
            dataset_path= args.amb_path,
            verbose= args.verbose
            )

    if args.m4singer_path:
        M4Singer_Process(
            hyper_parameters= hp,
            dataset_path= args.m4singer_path,
            verbose= args.verbose
            )

    if args.opencpop_path:
        OpenCPop_Process(
            hyper_parameters= hp,
            dataset_path= args.opencpop_path,
            verbose= args.verbose
            )

    if args.kiritan_path:
        Kiritan_Process(
            hyper_parameters= hp,
            dataset_path= args.kiritan_path,
            verbose= args.verbose
            )

    if args.ofuton_path:
        Ofuton_Process(
            hyper_parameters= hp,
            dataset_path= args.ofuton_path,
            verbose= args.verbose
            )

    if args.gtsinger_path:
        GTSinger_Process(
            hyper_parameters= hp,
            dataset_path= args.gtsinger_path,
            verbose= args.verbose
            )

    Metadata_Generate(hp, False)
    Metadata_Generate(hp, True)

# python -m Pattern.Generator -hp Hyper_Parameters.yaml -gtsinger /mnt/e/GTSinger
# python -m Pattern.Generator -hp Hyper_Parameters.yaml -csd /mnt/f/Rawdata_Music/CSD_1.1
# python -m Pattern.Generator -hp Hyper_Parameters.yaml -amz /mnt/f/Rawdata_Music/AIHub_Mediazen
# python -m Pattern.Generator -hp Hyper_Parameters.yaml -amb /mnt/e/AIHub_Metabuild
# python -m Pattern.Generator -hp Hyper_Parameters.yaml -m4 /mnt/f/Rawdata_Music/m4singer
# python -m Pattern.Generator -hp Hyper_Parameters.yaml -opencpop /mnt/e/OpenCPop
# python -m Pattern.Generator -hp Hyper_Parameters.yaml -kiritan /mnt/f/Rawdata_Music/Kiritan
# python -m Pattern.Generator -hp Hyper_Parameters.yaml -ofuton /mnt/f/Rawdata_Music/OFUTON_P_UTAGOE_DB

# python -m Pattern.Generator -hp Hyper_Parameters.yaml -csd /mnt/f/Rawdata_Music/CSD_1.1 -kiritan /mnt/f/Rawdata_Music/Kiritan -opencpop /mnt/e/OpenCPop


# python -m Pattern.Generator -hp Hyper_Parameters.yaml \
#     -csd /nas/SW/heejo.you/rawdata_music/CSD_1.1 \
#     -amz /nas/SW/heejo.you/rawdata_music/AIHub_Mediazen \
#     -amb /nas/SW/heejo.you/rawdata_music/AIHub_Metabuild \
#     -m4 /nas/SW/heejo.you/rawdata_music/m4singer \
#     -opencpop /nas/SW/heejo.you/rawdata_music/OpenCPop \
#     -kiritan /nas/SW/heejo.you/rawdata_music/Kiritan \
#     -ofuton /nas/SW/heejo.you/rawdata_music/OFUTON_P_UTAGOE_DB \
#     -gtsinger /nas/SW/heejo.you/rawdata_music/GTSinger

# python -m Pattern.Generator -hp Hyper_Parameters.yaml \
#     -csd /mnt/f/Rawdata_Music/CSD_1.1 \
#     -amz /mnt/f/Rawdata_Music/AIHub_Mediazen \
#     -amb /mnt/e/AIHub_Metabuild \
#     -m4 /mnt/f/Rawdata_Music/m4singer \
#     -opencpop /mnt/e/OpenCPop \
#     -kiritan /mnt/f/Rawdata_Music/Kiritan \
#     -ofuton /mnt/f/Rawdata_Music/OFUTON_P_UTAGOE_DB \
#     -gtsinger /mnt/e/GTSinger
