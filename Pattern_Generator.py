import numpy as np
import pandas as pd
import mido, os, pickle, yaml, argparse, math, librosa, hgtk, logging, re, sys, warnings, textgrid as tg, music21
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
from tqdm import tqdm
from pysptk.sptk import rapt
from typing import List, Tuple
from argparse import Namespace  # for type
import torch
from itertools import cycle
from typing import Dict, Union

from Phonemize import Phonemize, Language, _Korean_Syllablize, _Chinese_Syllablize, _Japanese_Syllable_to_IPA

from hificodec.vqvae import VQVAE
from Arg_Parser import Recursive_Parse

logging.basicConfig(
    level=logging.INFO, stream=sys.stdout,
    format= '%(asctime)s (%(module)s:%(lineno)d) %(levelname)s: %(message)s'
    )
warnings.filterwarnings('ignore')

if __name__ == '__main__':
    if not torch.cuda.is_available():
        device = torch.device('cpu')
    else:
        device = torch.device('cuda:0')
    hificodec = VQVAE(
        config_path= './hificodec/config_24k_320d.json',
        ckpt_path= './hificodec/HiFi-Codec-24k-320d',
        with_encoder= True
        ).to(device)

def AIHub_Mediazen(
    hyper_paramters: Namespace,
    dataset_path: str,
    verbose: bool= False
    ):
    path_dict = {}
    for root, _, files in os.walk(os.path.join(dataset_path).replace('\\', '/')):
        for file in files:
            key, extension = os.path.splitext(file)
            if not extension.upper() in ['.WAV', '.MID']:
                continue
            
            if not key in path_dict.keys():
                path_dict[key] = {}
            path_dict[key]['wav' if extension.upper() == '.WAV' else 'mid'] = os.path.join(root, file).replace('\\', '/')            

    paths = [
        (value['wav'], value['mid'], key.strip().split('_')[0].upper(), 'AM' + key.strip().split('_')[4].upper())
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

    is_eval_generated = False
    for index, (wav_path, midi_path, genre, singer) in tqdm(
        enumerate(paths),
        total= len(paths),
        desc= 'AIHub_Mediazen'
        ):
        music_label = os.path.splitext(os.path.basename(wav_path))[0]        
        if any([
            os.path.exists(os.path.join(x, 'AIHub_Mediazen', singer, f'{music_label}.pickle').replace('\\', '/'))
            for x in [hyper_paramters.Train.Eval_Pattern.Path, hyper_paramters.Train.Train_Pattern.Path]
            ] + [
            os.path.exists(os.path.join('./note_error/AIHub_Mediazen', singer, f'{music_label}.png'))
            ]):
            continue
        genre = genre_dict[genre]

        mid = mido.MidiFile(midi_path, charset='CP949')

        music = []
        note_states = {}
        last_note = None

        # Note on 쉼표
        # From Lyric to message before note on: real note
        for message in list(mid):
            if message.type == 'note_on' and message.velocity != 0:                
                if len(note_states) == 0:
                    if message.time < 0.1:
                        music[-1][0] += message.time
                    else:
                        if len(music) > 0 and music[-1][1] == '<X>':
                            music[-1][0] += message.time
                        else:
                            music.append([message.time, '<X>', 0])
                else:
                    note_states[last_note]['Time'] += message.time
                note_states[message.note] = {
                    'Lyric': None,
                    'Time': 0.0
                    }
                last_note = message.note                
            elif message.type == 'lyrics':
                if message.text == '\r' or last_note is None:    # If there is a bug in lyric
                    if verbose:
                        logging.warning(f'{wav_path} | {midi_path}')
                    continue
                note_states[last_note]['Lyric'] = message.text.strip()
                note_states[last_note]['Time'] += message.time
            elif message.type == 'note_off' or (message.type == 'note_on' and message.velocity == 0):
                note_states[message.note]['Time'] += message.time
                music.append([note_states[message.note]['Time'], note_states[message.note]['Lyric'], message.note])
                del note_states[message.note]
                last_note = None
            else:
                if len(note_states) == 0:
                    if len(music) > 0 and music[-1][1] == '<X>':
                        music[-1][0] += message.time
                    else:
                        music.append([message.time, '<X>', 0])
                else:
                    note_states[last_note]['Time'] += message.time
        if len(note_states) > 0:
            logging.critical(wav_path, midi_path)
            logging.critical(note_states)
            assert False
        music = [x for x in music if x[0] > 0.0]

        audio, _ = librosa.load(wav_path, sr= hyper_paramters.Sound.Sample_Rate)

        initial_audio_length = audio.shape[0]
        while True:
            if music[0][1] in [None, '', '<X>', 'J']:
                audio = audio[int(music[0][0] * hyper_paramters.Sound.Sample_Rate):]
                music = music[1:]
            else:
                break
        while True:
            if music[-1][1] in ['', '<X>', 'H']:
                music = music[:-1]
            else:
                break
        audio = audio[:int(sum([x[0] for x in music]) * hyper_paramters.Sound.Sample_Rate)]    # remove last silence
        
        # This is to avoid to use wrong data.
        if initial_audio_length * 0.5 > audio.shape[0]:
            continue

        audio = librosa.util.normalize(audio) * 0.95
        music = G2P_Lyric(music)
        if music is None:
            continue
        lyrics, notes, lyric_durations, note_durations = Convert_Feature_Based_Music(
            music= music,
            sample_rate= hyper_paramters.Sound.Sample_Rate,
            hop_size= hyper_paramters.Sound.Hop_Size,
            verbose= verbose
            )
        if lyrics is None:
            continue

        lyrics_expand, notes_expand, lyric_durations_expand, note_durations_expand = Expand_by_Duration(
            lyrics= lyrics,
            notes= notes,
            lyric_durations= lyric_durations,
            note_durations= note_durations
            )
        
        is_generated = Pattern_File_Generate(
            lyric= lyrics,
            note= notes,
            lyric_duration= lyric_durations,
            note_duration= note_durations,
            lyric_expand= lyrics_expand,
            note_expand= notes_expand,
            lyric_duration_expand= lyric_durations_expand,
            note_duration_expand= note_durations_expand,
            audio= audio,
            music_label= music_label,
            singer= singer,
            genre= genre,
            language= 'Korean',
            dataset= 'AIHub_Mediazen',
            is_eval_music= not is_eval_generated or np.random.rand() < 0.001,
            hyper_paramters= hyper_paramters,
            verbose= verbose
            )
        
        is_eval_generated = is_eval_generated or is_generated

def CSD_Fix(
    hyper_paramters: Namespace,
    dataset_path: str,
    verbose: bool= False
    ):
    paths = []
    for root, _, files in os.walk(os.path.join(dataset_path, 'wav').replace('\\', '/')):
        for file in sorted(files):
            if os.path.splitext(file)[1] != '.wav':
                continue
            wav_path = os.path.join(root, file).replace('\\', '/')
            midi_path = wav_path.replace('wav', 'mid')
            
            if not os.path.exists(midi_path):
                raise FileExistsError(midi_path)

            paths.append((wav_path, midi_path))

    for index, (wav_path, midi_path) in tqdm(
        enumerate(paths),
        total= len(paths),
        desc= 'CSD_Fix'
        ):
        music_label = os.path.splitext(os.path.basename(wav_path))[0]
        pattern_path = os.path.join(
            hyper_paramters.Train.Train_Pattern.Path if not index == (len(paths) - 1) else hyper_paramters.Train.Eval_Pattern.Path,
            'CSD',
            'CSD',
            f'{music_label}.pickle'
            ).replace('\\', '/')
        if os.path.exists(pattern_path) or os.path.exists(os.path.join(f'./note_error/CSD/CSD/{music_label}.png')):
            continue

        mid = mido.MidiFile(midi_path, charset='CP949')

        music = []
        note_states = {}
        last_note = None

        # Note on 쉼표
        # From Lyric to message before note on: real note        
        for message in list(mid):
            if message.type == 'note_on' and message.velocity != 0:                
                if len(note_states) == 0:
                    if message.time < 0.1:
                        music[-1][0] += message.time
                    else:
                        music.append([message.time, '<X>', 0])
                else:
                    note_states[last_note]['Time'] += message.time
                note_states[message.note] = {
                    'Lyric': None,
                    'Time': 0.0
                    }
                last_note = message.note                
            elif message.type == 'lyrics':
                if message.text == '\r':    # If there is a bug in lyric                    
                    if verbose:
                        logging.warning(wav_path, midi_path)
                    continue
                note_states[last_note]['Lyric'] = message.text.strip()
                note_states[last_note]['Time'] += message.time
            elif message.type == 'note_off' or (message.type == 'note_on' and message.velocity == 0):
                note_states[message.note]['Time'] += message.time
                music.append([note_states[message.note]['Time'], note_states[message.note]['Lyric'], message.note])
                del note_states[message.note]
                last_note = None
            else:
                if len(note_states) == 0:
                    music.append([message.time, '<X>', 0])
                else:
                    note_states[last_note]['Time'] += message.time
        if len(note_states) > 0:
            logging.critical(wav_path, midi_path)
            logging.critical(note_states)
            assert False
        music = [x for x in music if x[0] > 0.0]

        audio, _ = librosa.load(wav_path, sr= hyper_paramters.Sound.Sample_Rate)
        
        initial_audio_length = audio.shape[0]
        while True:
            if music[0][1] in [None, '', '<X>', 'J']:
                audio = audio[int(music[0][0] * hyper_paramters.Sound.Sample_Rate):]
                music = music[1:]
            else:
                break
        while True:
            if music[-1][1] in [None, '', '<X>', 'H']:
                music = music[:-1]
            else:
                break
        audio = audio[:int(sum([x[0] for x in music]) * hyper_paramters.Sound.Sample_Rate)]    # remove last silence
        
        # This is to avoid to use wrong data.
        if initial_audio_length * 0.5 > audio.shape[0]:
            continue

        audio = librosa.util.normalize(audio) * 0.95
        music = G2P_Lyric(music)
        if music is None:
            continue
        lyrics, notes, lyric_durations, note_durations = Convert_Feature_Based_Music(
            music= music,
            sample_rate= hyper_paramters.Sound.Sample_Rate,
            hop_size= hyper_paramters.Sound.Hop_Size,
            verbose= verbose
            )
        if lyrics is None:
            continue
        lyrics_expand, notes_expand, lyric_durations_expand, note_durations_expand = Expand_by_Duration(
            lyrics= lyrics,
            notes= notes,
            lyric_durations= lyric_durations,
            note_durations= note_durations
            )

        Pattern_File_Generate(
            lyric= lyrics,
            note= notes,
            lyric_duration= lyric_durations,
            note_duration= note_durations,
            lyric_expand= lyrics_expand,
            note_expand= notes_expand,
            lyric_duration_expand= lyric_durations_expand,
            note_duration_expand= note_durations_expand,
            audio= audio,
            music_label= music_label,
            singer= 'CSD',
            genre= 'Children',
            language= 'Korean',
            dataset= 'CSD',
            is_eval_music= index == (len(paths) - 1),
            hyper_paramters= hyper_paramters,
            verbose= verbose
            )

def CSD_Eng_Fix(
    hyper_paramters: Namespace,
    dataset_path: str,
    verbose: bool= False
    ):
    paths = []
    for root, _, files in os.walk(os.path.join(dataset_path, 'wav').replace('\\', '/')):
        for file in sorted(files):
            if os.path.splitext(file)[1] != '.wav':
                continue
            wav_path = os.path.join(root, file).replace('\\', '/')
            midi_path = wav_path.replace('wav', 'csv')
            lyric_path = wav_path.replace('wav/', 'ipa/').replace('.wav', '.txt')
            
            if not os.path.exists(midi_path):
                raise FileExistsError(midi_path)
            elif not os.path.exists(lyric_path):
                raise FileExistsError(lyric_path)

            paths.append((wav_path, midi_path, lyric_path))

    paths = sorted(paths, key= lambda x: x[0])

    for index, (wav_path, midi_path, lyric_path) in tqdm(
        enumerate(paths),
        total= len(paths),
        desc= 'CSD_Eng'
        ):
        music_label = os.path.splitext(os.path.basename(wav_path))[0]
        pattern_path = os.path.join(
            hyper_paramters.Train.Train_Pattern.Path if not index == (len(paths) - 1) else hyper_paramters.Train.Eval_Pattern.Path,
            'CSD',
            'CSD',
            f'{music_label}.pickle'
            ).replace('\\', '/')
        if os.path.exists(pattern_path) or os.path.exists(os.path.join(f'./note_error/CSD/CSD/{music_label}.png')):
            continue


        midi_df = pd.read_csv(midi_path)
        lyrics = [
            line.strip().split('\t')[0]
            for line in open(lyric_path, 'r', encoding= 'utf-8-sig').readlines()
            ]   # deque is better, but I don't use.
        assert len(lyrics) == len(midi_df), f'There is a lyric data has a problem in: \'{lyric_path}\', lyric length: {len(lyrics)}, midi length: {len(midi_df)}'
        midi_df['syllable'] = lyrics

        musics = []
        last_end_time = 0.0
        for _, row in midi_df.iterrows():
            if last_end_time != row['start']:
                musics.append((row['start'] - last_end_time, '<X>', 0))

            musics.append((row['end'] - row['start'], row['syllable'], row['pitch']))
            last_end_time = row['end']

        audio, _ = librosa.load(wav_path, sr= hyper_paramters.Sound.Sample_Rate)
        audio = librosa.util.normalize(audio) * 0.95

        # print(musics)
        musics = Phoneme_Split(musics)
        
        lyrics, notes, lyric_durations, note_durations = Convert_Feature_Based_Music(
            music= musics,
            sample_rate= hyper_paramters.Sound.Sample_Rate,
            hop_size= hyper_paramters.Sound.Hop_Size,
            verbose= verbose
            )
        
        if lyrics is None:
            continue
        lyrics_expand, notes_expand, lyric_durations_expand, note_durations_expand = Expand_by_Duration(
            lyrics= lyrics,
            notes= notes,
            lyric_durations= lyric_durations,
            note_durations= note_durations
            )
        
        audio = audio[:len(lyric_durations_expand) * hyper_paramters.Sound.Hop_Size]

        
        Pattern_File_Generate(
            lyric= lyrics,
            note= notes,
            lyric_duration= lyric_durations,
            note_duration= note_durations,
            lyric_expand= lyrics_expand,
            note_expand= notes_expand,
            lyric_duration_expand= lyric_durations_expand,
            note_duration_expand= note_durations_expand,
            audio= audio,
            music_label= music_label,
            singer= 'CSD',
            genre= 'Children',
            language= 'English',
            dataset= 'CSD',
            is_eval_music= index == (len(paths) - 1),
            hyper_paramters= hyper_paramters,
            verbose= verbose
            )

def M4Singer(
    hyper_paramters: Namespace,
    dataset_path: str,
    verbose: bool= False
    ):
    wav_paths_by_song = sorted([
        sorted([
            os.path.join(root, file).replace('\\', '/')    
            for file in files
            if os.path.splitext(file)[1] == '.wav'
            ])
        for root, _, files in os.walk(dataset_path)
        ])
    wav_paths_by_song = [
        wav_path
        for wav_path in wav_paths_by_song
        if len(wav_path) > 0
        ]
    
    is_eval_generated = False
    for wav_paths in tqdm(
        wav_paths_by_song,
        total= len(wav_paths_by_song),
        desc= 'M4Singer'
        ):
        singer = 'M4' + wav_paths[0].split('/')[-2].split('#')[0].replace('-', '').strip()
        music_label = wav_paths[0].split('/')[-2].split('#')[1].strip()

        if any([
            os.path.exists(os.path.join(
                path,
                'M4Singer',
                singer,
                f'{music_label}.pickle'
                ).replace('\\', '/'))
            for path in [
                hyper_paramters.Train.Train_Pattern.Path,
                hyper_paramters.Train.Eval_Pattern.Path,
                ]] + [os.path.exists(f'./note_error/M4Singer/{singer}/{music_label}.png')
            ]):
            continue

        lyrics, notes, lyric_durations, note_durations, audios = [], [], [], [], []
        for wav_path in wav_paths:
            midi_path = wav_path.replace('.wav', '.mid')
            tg_path = wav_path.replace('.wav', '.TextGrid')
            
            if not os.path.exists(midi_path) or not os.path.exists(tg_path):
                print(f'There is no other file about wav file: {wav_path}')
                continue

            musics = []
            note_states = {}
            last_notes = []
            mid = mido.MidiFile(midi_path)
            for message in list(mid):
                if message.type == 'note_on' and message.velocity != 0:
                    if len(note_states) == 0:
                        if message.time < 0.1:
                            musics[-1][0] += message.time
                        else:
                            if len(musics) > 0 and musics[-1][1] == 0:
                                musics[-1][0] += message.time
                            else:
                                musics.append([message.time, 0])
                    else:
                        note_states[last_notes[-1]]['Time'] += message.time
                    note_states[message.note] = {
                        'Time': 0.0
                        }
                    last_notes.append(message.note)
                elif message.type == 'note_off' or (message.type == 'note_on' and message.velocity == 0):
                    note_states[message.note]['Time'] += message.time
                    musics.append([note_states[message.note]['Time'], message.note])
                    del note_states[message.note]
                    del last_notes[last_notes.index(message.note)]
                else:
                    if len(note_states) == 0:
                        if len(musics) > 0 and musics[-1][1] == 0:
                            musics[-1][0] += message.time
                        else:
                            musics.append([message.time, 0])
                    else:
                        note_states[last_notes]['Time'] += message.time
            if len(note_states) > 0:
                logging.critical(wav_path, midi_path)
                logging.critical(note_states)
                assert False
            musics = [
                (np.round(duration, 3), note)
                for duration, note in musics
                if duration > 0.0
                ]
            
            # all data [start_time, end_time, duration, ...]
            cumulated_musics = []
            cumulated_time = 0
            for duration, note in musics:
                # music: [duration, note]
                cumulated_musics.append((cumulated_time, cumulated_time + duration, duration, note))
                cumulated_time += duration

            musics = cumulated_musics   # [start_time, end_time, duration, note]
            
            textgrids = []
            for textgrid_interval in tg.TextGrid.fromFile(tg_path)[0].intervals:
                chinese_character = textgrid_interval.mark
                if chinese_character in ['<AP>', '<SP>']:
                    chinese_character = '<X>'
                    
                ipa_phoneme = \
                    chinese_character \
                    if chinese_character == '<X>' else \
                    _Chinese_Syllablize(Phonemize(chinese_character, language= Language.Chinese)[0])[0]

                textgrids.append((
                    np.round(textgrid_interval.minTime, 3),
                    np.round(textgrid_interval.maxTime, 3),
                    np.round(textgrid_interval.duration(), 3),
                    chinese_character,
                    ipa_phoneme
                    ))

            def Overlap_Ratio(
                a_range: Union[List[float], Tuple[float, float]],
                b_range: Union[List[float], Tuple[float, float]]
                ):
                '''
                A∩B / A∪B
                '''
                a_range = sorted(a_range)
                b_range = sorted(b_range)
                assert len(a_range) == 2 or len(b_range) == 2
                if a_range[1] < b_range[0] or a_range[0] > b_range[1]:
                    return 0.0
                
                intersection = min(a_range[1], b_range[1]) - max(a_range[0], b_range[0])
                union = max(a_range[1], b_range[1]) - min(a_range[0], b_range[0])

                return intersection / union

            music_index = 0
            textgrid_index = 0
            music_textgrid_list = []            
            while music_index < len(musics) and textgrid_index < len(textgrids):
                # music[0]: start_time
                # music[1]: end_time
                # music[2]: duration
                # music[3]: note

                # textgrid[0]: start_time
                # textgrid[1]: end_time
                # textgrid[2]: duration
                # textgrid[3]: chinese chracter
                # textgrid[4]: ipa phoneme
                current_music = musics[music_index]
                current_textgrid = textgrids[textgrid_index]
                current_textgrid_and_current_music_overlap = Overlap_Ratio(
                    (current_music[0], current_music[1]),
                    (current_textgrid[0], current_textgrid[1]),
                    )
                
                multi_textgrid_and_current_music_overlap = \
                current_textgrid_and_multi_music_overlap = \
                    current_textgrid_and_current_music_overlap
                
                if music_index < len(musics) - 1:
                    current_textgrid_and_multi_music_overlap = Overlap_Ratio(
                        (current_music[0], musics[music_index + 1][1]),
                        (current_textgrid[0], current_textgrid[1]),
                        )
                if textgrid_index < len(textgrids) - 1:
                    multi_textgrid_and_current_music_overlap = Overlap_Ratio(
                        (current_music[0], current_music[1]),
                        (current_textgrid[0], textgrids[textgrid_index + 1][1]),
                        )

                if all([
                    current_textgrid_and_current_music_overlap >= current_textgrid_and_multi_music_overlap,
                    current_textgrid_and_current_music_overlap >= multi_textgrid_and_current_music_overlap,
                    ]): # (1 textgrid, 1 music)
                    music_textgrid_list.append(([current_music], [current_textgrid]))
                    music_index += 1
                    textgrid_index += 1
                elif current_textgrid_and_current_music_overlap < current_textgrid_and_multi_music_overlap: # (1 textgrid, n music)
                    fixed_start_time = current_music[0]
                    current_overlap = current_textgrid_and_current_music_overlap
                    music_list = [current_music]
                    while music_index + 1 < len(musics):
                        next_music = musics[music_index + 1]
                        next_overlap = Overlap_Ratio(
                            (fixed_start_time, next_music[1]),
                            (current_textgrid[0], current_textgrid[1])
                            )
                        if next_overlap > current_overlap:
                            music_list.append(next_music)
                            current_music = next_music
                            current_overlap = next_overlap
                            music_index += 1
                        else:
                            break

                    textgrid_list = [current_textgrid]
                    # last fix check, this managed very small silence
                    if textgrid_index < len(textgrids) - 1:
                        next_textgrid = textgrids[textgrid_index + 1]
                        if next_textgrid[3] == '<X>':
                            fixed_overlap = Overlap_Ratio(
                                (fixed_start_time, current_music[1]),
                                (current_textgrid[0], next_textgrid[1])
                                )
                            if fixed_overlap > current_overlap:
                                textgrid_list.append(next_textgrid)
                                textgrid_index += 1

                    music_textgrid_list.append((music_list, textgrid_list))
                    music_index += 1
                    textgrid_index += 1
                elif current_textgrid_and_current_music_overlap < multi_textgrid_and_current_music_overlap: # (n textgrid, 1 music)
                    fixed_start_time = current_textgrid[0]
                    current_overlap = current_textgrid_and_current_music_overlap
                    textgrid_list = [current_textgrid]
                    while textgrid_index + 1 < len(textgrids):
                        next_textgrid = textgrids[textgrid_index + 1]
                        next_overlap = Overlap_Ratio(
                            (current_music[0], current_music[1]),
                            (fixed_start_time, next_textgrid[1])
                            )
                        if next_overlap > current_overlap:
                            textgrid_list.append(next_textgrid)
                            current_textgrid = next_textgrid
                            current_overlap = next_overlap
                            textgrid_index += 1
                        else:
                            break


                    music_list = [current_music]
                    if music_index < len(musics) - 1:
                        # last fix check, this managed very small silence
                        next_music = musics[music_index + 1]
                        if next_music[3] == 0:
                            fixed_overlap = Overlap_Ratio(
                                (current_music[0], next_music[1]),
                                (fixed_start_time, current_textgrid[1])
                                )
                            if fixed_overlap > current_overlap:
                                music_list.append(next_music)
                                music_index += 1

                    music_textgrid_list.append((music_list, textgrid_list))
                    music_index += 1
                    textgrid_index += 1
                else:
                    raise ValueError(f'sync complexity: {wav_path}')
                
            if music_index < len(musics):
                assert musics[music_index][3] == 0, 'Not processed music info.'
                music_textgrid_list.append((musics[music_index:], []))
            if textgrid_index < len(textgrids): # some midi data does not have last silence info.
                assert textgrids[textgrid_index][3] == '<X>', 'Not processed textgrid info.'
                music_textgrid_list.append(([], textgrids[textgrid_index:]))

            musics = []
            for music_list, textgrid_list in music_textgrid_list:
                if len(music_list) > 1 and len(textgrid_list) > 1:    # special case, this is independent if
                    clean_music_list = []
                    for start_time, end_time, duration, note in music_list:
                        if len(clean_music_list) > 0 and note == 0:
                            clean_music_list[-1] = (
                                clean_music_list[-1][0],
                                end_time,
                                clean_music_list[-1][2] + duration,
                                clean_music_list[-1][3],
                                )
                            continue
                        clean_music_list.append((start_time, end_time, duration, note))
                    music_list = clean_music_list


                    clean_textgrid_list = []
                    for start_time, end_time, duration, chinese_character, ipa_phoneme in textgrid_list:
                        if len(clean_textgrid_list) > 0 and chinese_character == '<X>':
                            clean_textgrid_list[-1] = (
                                clean_textgrid_list[-1][0],
                                end_time,
                                clean_textgrid_list[-1][2] + duration,
                                clean_textgrid_list[-1][3],
                                clean_textgrid_list[-1][4],
                                )
                            continue
                        clean_textgrid_list.append((start_time, end_time, duration, chinese_character, ipa_phoneme))
                    textgrid_list = clean_textgrid_list

                    # if both of music_list and textgrid_list are multiple after cleaning, it is critical.
                    assert not (len(music_list) > 1 and len(textgrid_list) > 1), f'sync complexity: {wav_path}'

                if len(music_list) == 0 and len(textgrid_list) > 0: # special case
                    for textgrid in textgrid_list:
                        _, _, textgrid_duration, chinese_character, ipa_phoneme = textgrid
                        if chinese_character == '<X>':
                            musics.append((textgrid_duration, ipa_phoneme, 0))                
                elif len(music_list) == 1 and len(textgrid_list) == 1:
                    _, _, music_duration, note = music_list[0]
                    _, _, _, _, ipa_phoneme = textgrid_list[0]
                    musics.append((music_duration, ipa_phoneme, note))
                elif len(music_list) == 1 and len(textgrid_list) > 1:
                    _, _, music_duration, note = music_list[0]
                    for textgrid in textgrid_list:
                        _, _, textgrid_duration, chinese_character, ipa_phoneme = textgrid
                        if chinese_character == '<X>' and len(musics) > 0:
                            musics[-1] = ((musics[-1][0] + textgrid_duration, musics[-1][1], musics[-1][2]))
                        else:
                            musics.append((textgrid_duration, ipa_phoneme, note))
                elif len(music_list) > 1 and len(textgrid_list) == 1:
                    _, _, _, chinese_character, (onset, nucleus, coda) = textgrid_list[0]
                    
                    if all([music[3] == 0 for music in music_list]) or all([music[3] != 0 for music in music_list]):    # everything is 0 or nothing is 0.
                        pass
                    else:
                        # 일반 음표에 note 0이 섞인 케이스
                        fixed_music_list = []
                        for music in music_list:
                            if music[3] != 0:
                                fixed_music_list.append(music)
                            else:
                                fixed_music_list[-1] = (
                                    fixed_music_list[-1][0],
                                    fixed_music_list[-1][1] + music[1],
                                    fixed_music_list[-1][2] + music[2],
                                    fixed_music_list[-1][3],
                                    )
                        music_list = fixed_music_list
                    
                    for index, music in enumerate(music_list):
                        _, _, music_duration, note = music
                        if index == 0: ipa_phoneme = (onset, nucleus, [])
                        elif index == len(music) - 1: ipa_phoneme = ([], nucleus, coda)
                        else: ipa_phoneme = ([], nucleus, [])   # middle
                        musics.append((music_duration, ipa_phoneme, note))

            lyric, note, lyric_duration, note_duration = Convert_Feature_Based_Music(
                music= musics,
                sample_rate= hyper_paramters.Sound.Sample_Rate,
                hop_size= hyper_paramters.Sound.Hop_Size,
                verbose= verbose
                )
            if lyric is None:
                logging.warning(f'there is an error in\'{wav_path}\'. This pattern will is skipped.')
                continue

            audio = librosa.load(wav_path, sr= hyper_paramters.Sound.Sample_Rate)[0]
            audio = audio[:audio.shape[0] - (audio.shape[0] % hyper_paramters.Sound.Hop_Size)]
            
            music_length = sum(lyric_duration) * hyper_paramters.Sound.Hop_Size
            audio_length = audio.shape[0]

            if music_length < audio_length:
                audio = audio[:music_length]
            elif music_length > audio_length:
                cut_frame_size = (music_length - audio_length) // hyper_paramters.Sound.Hop_Size

                if note[-1] == 0 and note_duration[-1] > cut_frame_size:    # last silence cutoff
                    lyric_duration[-1] -= cut_frame_size
                    note_duration[-1] -= cut_frame_size
                else:   # distributed cutoff
                    lyric_duration_indices = reversed([index for index, x in enumerate(lyric_duration) if x > 1])
                    note_duration_indices = reversed([index for index, x in enumerate(note_duration) if x > 0])

                    for lyric_duration_index, note_duration_index in cycle(zip(
                        lyric_duration_indices,
                        note_duration_indices
                        )):
                        lyric_duration[lyric_duration_index] -= 1
                        note_duration[note_duration_index] -= 1
                        cut_frame_size -= 1

                        if cut_frame_size <= 0:
                            break

            # music level concat makes big length error between music and audio....
            lyrics.append(lyric)
            notes.append(note)
            lyric_durations.append(lyric_duration)
            note_durations.append(note_duration)
            audios.append(audio)

        lyrics_concat, notes_concat, lyric_durations_concat, note_durations_concat = [], [], [], []
        for lyric, note, lyric_duration, note_duration in zip(lyrics, notes, lyric_durations, note_durations):
            lyrics_concat.extend(lyric)
            notes_concat.extend(note)
            lyric_durations_concat.extend(lyric_duration)
            note_durations_concat.extend(note_duration)

        lyrics = lyrics_concat
        notes = notes_concat
        lyric_durations = lyric_durations_concat
        note_durations = note_durations_concat
        audios = np.hstack(audios)

        assert \
            sum(lyric_durations) == audios.shape[0] // hyper_paramters.Sound.Hop_Size, \
            (wav_paths, sum(lyric_durations), audios.shape[0])
        
        lyrics_expand, notes_expand, lyric_durations_expand, note_durations_expand = Expand_by_Duration(
            lyrics= lyrics,
            notes= notes,
            lyric_durations= lyric_durations,
            note_durations= note_durations
            )

        is_generated = Pattern_File_Generate(
            lyric= lyrics,
            note= notes,
            lyric_duration= lyric_durations,
            note_duration= note_durations,
            lyric_expand= lyrics_expand,
            note_expand= notes_expand,
            lyric_duration_expand= lyric_durations_expand,
            note_duration_expand= note_durations_expand,
            audio= audios,
            music_label= music_label,
            singer= singer,
            genre= 'Pop',
            language= 'Chinese',
            dataset= 'M4Singer',
            is_eval_music= not is_eval_generated or np.random.rand() < 0.001,
            hyper_paramters= hyper_paramters,
            note_error_criterion= 1.5,
            verbose= verbose
            )
        is_eval_generated = is_eval_generated or is_generated

def NUS48E(
    hyper_paramters: Namespace,
    dataset_path: str,
    verbose: bool= False
    ):
    music_label_dict = {
        1: 'Edelweiss',
        2: 'Do_Re_Mi',
        3: 'Jingle_Bells',
        4: 'Silent_Night',
        5: 'Wonderful_Tonight',
        6: 'Moon_River',
        7: 'Rhythm_of_the_Rain',
        8: 'I_Have_a_Dream',
        9: 'Love_Me_Tender',
        10: 'Twinkle_Twinkle_Little_Star',
        11: 'You_Are_My_Sunshine',
        12: 'A_Little_Love',
        13: 'Proud_of_You',
        14: 'Lemon_Tree',
        15: 'Can_You_Feel_the_Love_Tonight',
        16: 'Far_Away_from_Home',
        17: 'Seasons_in_the_Sun',
        18: 'The_Show',
        19: 'The_Rose',
        20: 'Right_Here_Waiting',
        }
    
    paths = []
    for root, _, files in os.walk(os.path.join(dataset_path, 'wav').replace('\\', '/')):
        for file in sorted(files):
            if os.path.splitext(file)[1] != '.wav':
                continue
            wav_path = os.path.join(root, file).replace('\\', '/')
            data_path = wav_path.replace('wav/', 'Data/').replace('.wav', '.pickle')
            
            
            if not os.path.exists(data_path):
                raise FileExistsError(data_path)

            paths.append((wav_path, data_path))

    paths = sorted(paths, key= lambda x: x[0])

    for index, (wav_path, data_path) in tqdm(
        enumerate(paths),
        total= len(paths),
        desc= 'NUS48E'
        ):
        singer, music_label = os.path.splitext(os.path.basename(wav_path))[0].split('_')
        music_label = music_label_dict[int(music_label)]
        pattern_path = os.path.join(
            hyper_paramters.Train.Train_Pattern.Path if not index == (len(paths) - 1) else hyper_paramters.Train.Eval_Pattern.Path,
            'NUS48E',
            singer,
            f'{music_label}.pickle'
            ).replace('\\', '/')
        if os.path.exists(pattern_path) or os.path.exists(os.path.join(f'./note_error/NUS48E/{singer}/{music_label}.png')):
            continue


        musics = pickle.load(open(data_path, 'rb'))
        musics = [
            (end - start, lyric, note)
            for start, end, note, lyric in musics
            ]
        
        lyrics, notes, lyric_durations, note_durations = Convert_Feature_Based_Music(
            music= musics,
            sample_rate= hyper_paramters.Sound.Sample_Rate,
            hop_size= hyper_paramters.Sound.Hop_Size,
            verbose= verbose
            )
        if lyrics is None:
            continue
        lyrics_expand, notes_expand, lyric_durations_expand, note_durations_expand = Expand_by_Duration(
            lyrics= lyrics,
            notes= notes,
            lyric_durations= lyric_durations,
            note_durations= note_durations
            )

        audio, _ = librosa.load(wav_path, sr= hyper_paramters.Sound.Sample_Rate)
        audio = librosa.util.normalize(audio) * 0.95
        audio = audio[:len(lyric_durations_expand) * hyper_paramters.Sound.Hop_Size]
        
        Pattern_File_Generate(
            lyric= lyrics,
            note= notes,
            lyric_duration= lyric_durations,
            note_duration= note_durations,
            lyric_expand= lyrics_expand,
            note_expand= notes_expand,
            lyric_duration_expand= lyric_durations_expand,
            note_duration_expand= note_durations_expand,
            audio= audio,
            music_label= music_label,
            singer= singer,
            genre= 'Pop',
            language= 'English',
            dataset= 'NUS48E',
            is_eval_music= index == (len(paths) - 1),
            hyper_paramters= hyper_paramters,
            verbose= verbose
            )

def Kiritan(
    hyper_paramters: Namespace,
    dataset_path: str,
    verbose: bool= False
    ):
    logging.warning('Kiritan 중간에 쉼표 있을 때의 촉음 처리 추가할 것')
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
    
    def remove_initial_consonant(syllable):
        # 히라가나 모음 매핑
        vowel_map = {
            'あ': 'あ', 'い': 'い', 'う': 'う', 'え': 'え', 'お': 'お',
            'か': 'あ', 'き': 'い', 'く': 'う', 'け': 'え', 'こ': 'お',
            'さ': 'あ', 'し': 'い', 'す': 'う', 'せ': 'え', 'そ': 'お',
            'た': 'あ', 'ち': 'い', 'つ': 'う', 'て': 'え', 'と': 'お',
            'な': 'あ', 'に': 'い', 'ぬ': 'う', 'ね': 'え', 'の': 'お',
            'は': 'あ', 'ひ': 'い', 'ふ': 'う', 'へ': 'え', 'ほ': 'お',
            'ま': 'あ', 'み': 'い', 'む': 'う', 'め': 'え', 'も': 'お',
            'や': 'あ', 'ゆ': 'う', 'よ': 'お',
            'ら': 'あ', 'り': 'い', 'る': 'う', 'れ': 'え', 'ろ': 'お',
            'わ': 'あ', 'を': 'お', 'ん': 'ん',
            'が': 'あ', 'ぎ': 'い', 'ぐ': 'う', 'げ': 'え', 'ご': 'お',
            'ざ': 'あ', 'じ': 'い', 'ず': 'う', 'ぜ': 'え', 'ぞ': 'お',
            'だ': 'あ', 'ぢ': 'い', 'づ': 'う', 'で': 'え', 'ど': 'お',
            'ば': 'あ', 'び': 'い', 'ぶ': 'う', 'べ': 'え', 'ぼ': 'お',
            'ぱ': 'あ', 'ぴ': 'い', 'ぷ': 'う', 'ぺ': 'え', 'ぽ': 'お',

            'ぁ': 'あ', 'ぃ': 'い', 'ぅ': 'う', 'ぇ': 'え', 'ぉ': 'お',
            'ゕ': 'あ', 'ゖ': 'え',
            'ゃ': 'あ', 'ゅ': 'う', 'ょ': 'お',
            'ゎ': 'あ',
            }
        
        if len(syllable) == 2 and syllable[1] in ['ん', 'っ']:
            return remove_initial_consonant(syllable[0]) + syllable[1]

        return vowel_map[syllable[-1]]

    def dakuten_fix(syllable):
        dakuten_map = {
            'う': 'ゔ',
            'か': 'が', 'き': 'ぎ', 'く': 'ぐ', 'け': 'げ', 'こ': 'ご',
            'さ': 'ざ', 'し': 'じ', 'す': 'ず', 'せ': 'ぜ', 'そ': 'ぞ',
            'た': 'だ', 'ち': 'ぢ', 'つ': 'づ', 'て': 'で', 'と': 'ど',
            'は': 'ば', 'ひ': 'び', 'ふ': 'ぶ', 'へ': 'べ', 'ほ': 'ぼ',        
            }
        
        return dakuten_map[syllable]

    def get_duration_seconds(quater_length, current_tempo):
        return quater_length * (60 / current_tempo.number)

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

    for index, (wav_path, score_path) in tqdm(
        enumerate(paths),
        total= len(paths),
        desc= 'Kiritan'
        ):
        music_label = os.path.splitext(os.path.basename(wav_path))[0]
        music_label = music_label_dict[int(music_label)]
        pattern_path = os.path.join(
            hyper_paramters.Train.Train_Pattern.Path if not index == (len(paths) - 1) else hyper_paramters.Train.Eval_Pattern.Path,
            'Kiritan',
            'Kiritan',
            f'{music_label}.pickle'
            ).replace('\\', '/')
        if os.path.exists(pattern_path) or os.path.exists(os.path.join(f'./note_error/Kiritan/Kiritan/{music_label}.png')):
            continue
        
        score = music21.converter.parse(score_path)
        current_tempo = music21.tempo.MetronomeMark(number= 120)

        music = []
        cumulative_time = 0.0

        for part in score.parts:
            previous_split = 1
            for element in part.flatten():
                if isinstance(element, music21.tempo.MetronomeMark):
                    current_tempo = element
                elif isinstance(element, music21.note.Note):
                    note = librosa.note_to_midi(element.nameWithOctave.replace('-', '♭'))
                    if not element.lyric is None and '゛' in element.lyric:
                        if element.lyric[0] == '゛':
                            music[-1][1][2] = music[-1][1][:-1] + dakuten_fix(music[-1][1][-1])
                            element.lyric = element.lyric[:-1]
                        else:
                            hakuten_index = element.lyric.index('゛')
                            element.lyric = element.lyric[:max(0, hakuten_index - 2)] + dakuten_fix(element.lyric[hakuten_index - 1]) + element.lyric[hakuten_index + 1:]

                    if element.lyric is None:                        
                        duration_seconds = get_duration_seconds(element.quarterLength, current_tempo)
                        cumulative_time += duration_seconds
                        if music[-1][2] == note:
                            for previous_split_index in range(1, previous_split + 1):
                                music[-previous_split_index] = (
                                    music[-previous_split_index][0] + duration_seconds / previous_split,
                                    music[-previous_split_index][1],
                                    music[-previous_split_index][2]
                                    )
                        else:   # music[-1][0] != note, 28의 'つう', 47 'ろ' 체크
                            if previous_split == 1:
                                music.append((duration_seconds, remove_initial_consonant(music[-1][1]), note))
                                if music[-2][1][-1] in ['ん', 'っ']:    # -2 is correct because of new score.
                                    music[-2] = (music[-2][0], music[-2][1][:-1], music[-2][2])
                            elif previous_split == 2:
                                if music[-1][1][-1] in ['ん', 'っ']:
                                    print(index+1, music[-2], music[-1])                                    
                                    assert False
                                music[-2] = (music[-2][0] + music[-1][0], music[-2][1], music[-2][2])
                                music[-1] = (duration_seconds, music[-1][1], note)
                            else:
                                for x in music:
                                    print(x)
                                print(index, note, previous_split)
                                raise NotImplementedError(f'Check index {index + 1}')
                    elif len(element.lyric) == 1:
                        duration_seconds = get_duration_seconds(element.quarterLength, current_tempo)
                        cumulative_time += duration_seconds
                        lyric = element.lyric
                        if element.lyric == 'ー':
                            lyric = remove_initial_consonant(music[-1][1])
                        elif element.lyric in ['ん', 'っ']:
                            previous_syllable = [
                                previous_syllable
                                for _, previous_syllable, _ in music
                                if previous_syllable != '<X>'][-1]  # sometime staccato makes problem.
                            lyric = remove_initial_consonant(previous_syllable) + element.lyric                            
                        music.append((duration_seconds, lyric, note))
                        previous_split = 1
                    else:   # len(element.lyric) > 1
                        lyrics = []
                        for syllable in element.lyric:
                            if not syllable in ['ぁ', 'ぃ', 'ぅ', 'ぇ', 'ぉ', 'ゕ', 'ゖ', 'っ', 'ゃ', 'ゅ', 'ょ', 'ゎ', 'ん'] or len(lyrics) == 0:
                                lyrics.append([syllable])
                            else:
                                lyrics[-1].append(syllable)
                        lyrics = [''.join(x) for x in lyrics]
                        notes = [note] * len(lyrics)
                        duration_seconds = [get_duration_seconds(element.quarterLength / len(lyrics), current_tempo)] * len(lyrics)
                        cumulative_time += sum(duration_seconds)
                        for note, duration, lyric in zip(notes, duration_seconds, lyrics):
                            music.append((duration, lyric, note))                        
                        previous_split = len(lyrics)
                elif isinstance(element, music21.note.Rest):
                    duration_seconds = get_duration_seconds(element.quarterLength, current_tempo)
                    cumulative_time += duration_seconds
                    if len(music) > 0 and music[-1][2] == 0:
                        music[-1] = (music[-1][0] + duration_seconds, music[-1][1], music[-1][2])
                    else:
                        music.append((duration_seconds, '<X>', 0))

        music = Japanese_G2P_Lyric(music)
        
        audio, _ = librosa.load(wav_path, sr= hyper_paramters.Sound.Sample_Rate)
        while True:
            if music[0][1] == '<X>' and music[1][1] == '<X>':
                audio = audio[int(music[0][0] * hyper_paramters.Sound.Sample_Rate):]
                music = music[1:]
            elif music[0][1] == '<X>' and music[0][0] > 1.0:  # music[1][1] != '<X>'
                audio = audio[int((music[0][0] - 1.0) * hyper_paramters.Sound.Sample_Rate):]                
                music[0] = (1.0, music[0][1], music[0][2])
            else:
                break
        while True:
            if music[-1][1] == '<X>':
                music = music[:-1]
            else:
                break
        audio = librosa.util.normalize(audio) * 0.95
        
        lyrics, notes, lyric_durations, note_durations = Convert_Feature_Based_Music(
            music= music,
            sample_rate= hyper_paramters.Sound.Sample_Rate,
            hop_size= hyper_paramters.Sound.Hop_Size,
            verbose= verbose
            )

        if lyric is None:
            logging.warning(f'there is an error in\'{wav_path}\'. This pattern will is skipped.')
            continue
        lyrics_expand, notes_expand, lyric_durations_expand, note_durations_expand = Expand_by_Duration(
            lyrics= lyrics,
            notes= notes,
            lyric_durations= lyric_durations,
            note_durations= note_durations
            )
            
        music_length = sum(lyric_durations) * hyper_paramters.Sound.Hop_Size
        audio_length = audio.shape[0]
        if music_length < audio_length:
            audio = audio[:music_length]

        Pattern_File_Generate(
            lyric= lyrics,
            note= notes,
            lyric_duration= lyric_durations,
            note_duration= note_durations,
            lyric_expand= lyrics_expand,
            note_expand= notes_expand,
            lyric_duration_expand= lyric_durations_expand,
            note_duration_expand= note_durations_expand,
            audio= audio,
            music_label= music_label,
            singer= 'Kiritan',
            genre= 'Pop',
            language= 'Japanese',
            dataset= 'Kiritan',
            is_eval_music= index == (len(paths) - 1),
            hyper_paramters= hyper_paramters,
            note_error_criterion= 1.5,
            verbose= verbose
            )

# for English
def Phoneme_Split(
    musics: List[Tuple[float, str, int]]
    ) -> Tuple[float, Tuple[List[str], str, List[str]], int]:
    # /ʔ/ is not vowel.
    # However, in English, the next vowel of /ʔ/ is usually silence /ə/.
    # To rule make, I manage /ʔ/ like a vowel
    diphones = ['tʃ', 'hw', 'aʊ', 'kw', 'eɪ', 'oʊ', 'dʒ', 'ɔɪ', 'aɪ']
    use_vowels = [
        'aʊ', 'eɪ', 'oʊ', 'ɔɪ', 'aɪ', 'a', 'e', 'i', 'o', 'u', 
        'æ', 'ɐ', 'ɑ', 'ɔ', 'ə', 'ɛ', 'ɜ', 'ɪ', 'ʊ', 'ʌ', 'ᵻ', 'ɚ', 'ʔ', 
        ]

    pronunciations = [x[1] for x in musics]
    lyrics = []
    for pronunciation in pronunciations:
        if pronunciation == '<X>':
            lyrics.append(pronunciation)
            continue
        onsets = []
        nucleus = None
        codas = []
        index = 0
        while index < len(pronunciation):
            is_diphone = False
            if pronunciation[index:index + 2] in diphones:
                phoneme = pronunciation[index:index + 2]
                is_diphone = True
            else:
                phoneme = pronunciation[index]
            
            if not phoneme in use_vowels:
                if nucleus is None:
                    onsets.append(phoneme)
                else:   # coda
                    codas.append(phoneme)
            else:   # vowel
                nucleus = phoneme
                if len(codas) > 0:
                    assert False
            index += 1 + is_diphone

        lyrics.append([onsets, nucleus, codas])

    # assert len(lyrics) == len(music)

    if len(lyrics) != len(musics):
        return None
    musics = [
        (music[0], lyric, music[2])
        for music, lyric in zip(musics, lyrics)
        ]

    return musics

def Convert_Feature_Based_Music(
    music: List[Tuple[float, str, int]],
    sample_rate: int,
    hop_size: int,
    verbose: bool= False
    ) -> Tuple[List[str], List[int], List[int], List[int]]:
    previous_used = 0
    lyrics = []
    notes = []
    lyric_durations = []
    note_durations = []

    for message_time, lyric, note in music:
        duration = round(message_time * sample_rate) + previous_used
        previous_used = duration % hop_size
        duration = duration // hop_size

        if lyric == '<X>':
            lyrics.append(lyric)
            notes.append(note)
            lyric_durations.append(duration)
            note_durations.append(duration)
        elif duration < len(lyric[0]) + 1 + len(lyric[2]):            
            if verbose:
                logging.warning(f'too short duration than lyric: {duration} < {len(lyric[0]) + 1 + len(lyric[2])}')
            return None, None, None, None
        else:
            lyrics.extend(lyric[0]) # onset
            lyrics.append(lyric[1]) # nuclues
            lyrics.extend(lyric[2]) # coda
            notes.extend([note] * (len(lyric[0]) + 1 + len(lyric[2])))            
            lyric_durations.extend([1] * len(lyric[0]))
            lyric_durations.append(duration - len(lyric[0]) - len(lyric[2]))
            lyric_durations.extend([1] * len(lyric[2]))
            note_durations.append(duration)
            note_durations.extend([0] * (len(lyric[0]) + len(lyric[2])))

    return lyrics, notes, lyric_durations, note_durations

def G2P_Lyric(music):
    grapheme_string_list = []
    current_grapheme_string = []
    for _, lyric, _ in music:
        if lyric == '<X>':
            grapheme_string_list.append(''.join(current_grapheme_string))
            current_grapheme_string = []
        else:
            current_grapheme_string.append(lyric)
    grapheme_string_list.append(''.join(current_grapheme_string))

    pronunciations = Phonemize(
        texts= grapheme_string_list,
        language= Language.Korean
        )

    lyrics = []   
    for pronunciation in pronunciations:
        lyrics.extend(_Korean_Syllablize(pronunciation))
        lyrics.append('<X>')
    lyrics.pop()    # remove Last <X>

    assert len(lyrics) == len(music)
    if len(lyrics) != len(music):
        return None
    for index, lyric in enumerate(lyrics):
        music[index] = (music[index][0], lyric, music[index][2])

    return music

def Chinese_G2P_Lyric(music):
    ipa = [
        _Chinese_Syllablize(Phonemize(lyric, Language.Chinese)[0])[0] if not lyric in ['<X>', '-'] else lyric
        for _, lyric, _, in music
        ]
    ipa_slur = []
    ipa_index = 0
    current_onset, current_nucleus, current_coda = '', '', ''
    while ipa_index < len(ipa):
        if ipa[ipa_index] == '<X>':
            ipa_slur.append(ipa[ipa_index])
        elif ipa[ipa_index] != '-' and ipa_index < len(ipa) - 1 and ipa[ipa_index + 1] == '-': # initial slur
            current_onset, current_nucleus, current_coda = ipa[ipa_index]
            ipa_slur.append((current_onset, current_nucleus, []))        
        elif ipa[ipa_index] == '-' and ipa_index < len(ipa) - 1 and ipa[ipa_index + 1] == '-':  # middle slur
            ipa_slur.append(([], current_nucleus, []))        
        elif ipa[ipa_index] == '-': # final slur
            ipa_slur.append(([], current_nucleus, current_coda))
        else:
            ipa_slur.append(ipa[ipa_index])
        ipa_index += 1

    music = [
        (duration, syllable, note)
        for (duration, _, note), syllable in zip(music, ipa_slur)
        ]
    
    return music

def Japanese_G2P_Lyric(music):
    ipa_music = []
    for music_index, (current_duration, current_lyric, current_note) in enumerate(music):
        if current_lyric == '<X>':
            ipa_music.append((current_duration, current_lyric, current_note))
            continue
        
        if music_index + 1 < len(music) and music[music_index + 1][1] != '<X>':
            next_syllable = music[music_index + 1][1]
        elif music_index + 2 < len(music) and music[music_index + 1][0] < 0.5: # music[music_index + 1][1] == '<X>' but short
            next_syllable = music[music_index + 2][1]
        else:
            next_syllable = None

        ipa_lyric = _Japanese_Syllable_to_IPA(current_lyric, next_syllable)
        ipa_music.append((current_duration, ipa_lyric, current_note))
    
    return ipa_music


def Expand_by_Duration(
    lyrics: List[str],
    notes: List[int],
    lyric_durations: List[int],
    note_durations: List[int],
    ) -> Tuple[List[str], List[int], List[int], List[int]]:
    lyrics = sum([[lyric] * duration for lyric, duration in zip(lyrics, lyric_durations)], [])
    notes = sum([*[[note] * duration for note, duration in zip(notes, note_durations)]], [])
    lyric_durations = [index for duration in lyric_durations for index in range(duration)]
    note_durations = [index for duration in note_durations for index in range(duration)]

    return lyrics, notes, lyric_durations, note_durations

def Decompose(syllable: str):
    onset, nucleus, coda = hgtk.letter.decompose(syllable)
    coda += '_'

    return onset, nucleus, coda

def Pattern_File_Generate(
    lyric: List[str],
    note: List[int],
    lyric_duration: List[int],
    note_duration: List[int],
    lyric_expand: List[str],
    note_expand: List[int],
    lyric_duration_expand: List[int],
    note_duration_expand: List[int],
    audio: np.array,
    singer: str,
    genre: str,
    language: str,
    dataset: str,
    music_label: str,
    is_eval_music: bool,
    hyper_paramters: Namespace,
    note_error_criterion: float= 1.0,
    verbose: bool= False
    ):
    audio_tensor = torch.from_numpy(audio).to(device).float()
    with torch.inference_mode():
        latent_code = hificodec.encode(audio_tensor[None])[0].T.cpu().numpy() # [4, Audio_t / 320]
    f0 = rapt(
        x= audio * 32768,
        fs= hyper_paramters.Sound.Sample_Rate,
        hopsize= hyper_paramters.Sound.Hop_Size,
        min= hyper_paramters.Sound.F0_Min,
        max= hyper_paramters.Sound.F0_Max,
        otype= 1,
        )[:latent_code.shape[1]]

    if latent_code.shape[1] > len(lyric_expand):
        # print('Case1')
        latent_code = latent_code[math.floor((latent_code.shape[1] - len(lyric_expand)) / 2.0):-math.ceil((latent_code.shape[1] - len(lyric_expand)) / 2.0)]
        f0 = f0[math.floor((f0.shape[0] - len(lyric_expand)) / 2.0):-math.ceil((f0.shape[0] - len(lyric_expand)) / 2.0)]
    elif len(lyric_expand) > latent_code.shape[1]:
        # print('Case2')
        fix_length = len(lyric_expand) - latent_code.shape[1]
        fix_index = None
        for fix_index in reversed(range(len(lyric_duration))):
            if lyric_duration[fix_index] <= fix_length:                
                continue
            break

        lyric_duration[fix_index] -= fix_length
        note_duration[fix_index] -= fix_length
        lyric_expand = lyric_expand[:sum(lyric_duration[:fix_index + 1]) - fix_length + 1] + lyric_expand[sum(lyric_duration[:fix_index + 1]) + 1:]
        note_expand = note_expand[:sum(note_duration[:fix_index + 1]) - fix_length + 1] + note_expand[sum(note_duration[:fix_index + 1]) + 1:]
        lyric_duration_expand = lyric_duration_expand[:sum(lyric_duration[:fix_index + 1]) - fix_length + 1] + lyric_duration_expand[sum(lyric_duration[:fix_index + 1]) + 1:]
        note_duration_expand = note_duration_expand[:sum(note_duration[:fix_index + 1]) - fix_length + 1] + note_duration_expand[sum(note_duration[:fix_index + 1]) + 1:]

    # criterion = 1.0, this is just empirical criterion.
    note_from_f0 = Note_Predictor(f0).astype(np.float16)
    note_from_midi = np.array(note_expand).astype(np.float16)
    note_from_f0_without_0, note_from_midi_without_0 = note_from_f0[(note_from_f0 > 0) * (note_from_midi > 0)], note_from_midi[(note_from_f0 > 0) * (note_from_midi > 0)]
    note_error_value = np.abs(note_from_f0_without_0 - note_from_midi_without_0).mean()

    # octave problem fix    
    if note_error_value > note_error_criterion:
        is_fixed = False
        if np.abs(note_from_f0_without_0 - note_from_midi_without_0 - 12.0).mean() < note_error_criterion:
            note = [(x + 12 if x != 0 else x) for x in note]
            note_expand = [(x + 12 if x != 0 else x) for x in note_expand]
            note_fix_from_midi = np.array(note_expand).astype(np.float16)
            note_fix_from_midi_without_0 = note_fix_from_midi[(note_from_f0 > 0) * (note_from_midi > 0)]
            note_fix_error_value = np.abs(note_from_f0_without_0 - note_fix_from_midi_without_0).mean()
            is_fixed = True
        elif np.abs(note_from_f0_without_0 - note_from_midi_without_0 + 12.0).mean() < note_error_criterion:
            note = [(x - 12 if x != 0 else x) for x in note]
            note_expand = [(x - 12 if x != 0 else x) for x in note_expand]
            note_fix_from_midi = np.array(note_expand).astype(np.float16)
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
                    ticks= [x for x in [0.0] + np.cumsum(lyric_duration).tolist()[:-1]],
                    labels= [x for x in lyric],
                    )
                for x in [x for x in [0.0] + np.cumsum(lyric_duration).tolist()[:-1]]:
                    plt.axvline(x= x, linewidth= 0.5)
                plt.margins(x= 0)
                plt.legend()
                plt.tight_layout()
                plt.savefig(os.path.join('./note_fix', dataset, singer, f'{music_label}.png').replace('\\', '/'))
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
                    ticks= [x for x in [0.0] + np.cumsum(lyric_duration).tolist()[:-1]],
                    labels= [x for x in lyric],
                    )
                # for x in [x for x in [0.0] + np.cumsum(lyric_duration).tolist()[:-1]]:
                #     plt.axvline(x= x, linewidth= 0.5)
                plt.margins(x= 0)
                plt.legend()            
                plt.tight_layout()
                plt.savefig(os.path.join('./note_error', dataset, singer, f'{music_label}.png').replace('\\', '/'))
                plt.close()
                return False

    pattern = {
        'Latent_Code': latent_code.astype(np.int16),
        'F0': f0.astype(np.float16),
        'Lyric': lyric,
        'Note': note,
        'Lyric_Duration': lyric_duration,
        'Note_Duration': note_duration,
        'Lyric_Expand': lyric_expand,
        'Note_Expand': note_expand,
        'Lyric_Duration_Expand': lyric_duration_expand,
        'Note_Duration_Expand': note_duration_expand,
        'Singer': singer,
        'Genre': genre,
        'Language': language,
        'Dataset': dataset,
        }

    pattern_path = os.path.join(
        hyper_paramters.Train.Train_Pattern.Path if not is_eval_music else hyper_paramters.Train.Eval_Pattern.Path,
        dataset,
        singer,
        f'{music_label}.pickle'
        ).replace('\\', '/')

    os.makedirs(os.path.dirname(pattern_path), exist_ok= True)
    pickle.dump(
        pattern,
        open(pattern_path, 'wb'),
        protocol= 4
        )

    return True

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

def Token_Dict_Generate(hyper_parameters: Namespace, tokens: List[str]):
    tokens = [x for x in tokens if x != '<X>']
    
    os.makedirs(os.path.dirname(hyper_parameters.Token_Path), exist_ok= True)
    yaml.dump(
        {token: index for index, token in enumerate(['<S>', '<E>', '<X>'] + sorted(tokens))},
        open(hyper_parameters.Token_Path, 'w', encoding='utf-8-sig'),
        allow_unicode= True
        )

def Metadata_Generate(
    hyper_parameters: Namespace,
    eval: bool= False
    ):
    pattern_path = hyper_parameters.Train.Eval_Pattern.Path if eval else hyper_parameters.Train.Train_Pattern.Path
    metadata_file = hyper_parameters.Train.Eval_Pattern.Metadata_File if eval else hyper_parameters.Train.Train_Pattern.Metadata_File

    latent_dict = {}
    f0_dict = {}
    tokens = []
    singers = []
    genres = []
    languages = []
    min_note, max_note = math.inf, -math.inf

    new_metadata_dict = {
        'Hop_Size': hyper_parameters.Sound.Hop_Size,
        'Sample_Rate': hyper_parameters.Sound.Sample_Rate,
        'File_List': [],
        'Latent_Length_Dict': {},
        'F0_Length_Dict': {},
        'Lyric_Length_Dict': {},
        'Note_Length_Dict': {},
        'Duration_Length_Dict': {},
        'Lyric_Expand_Length_Dict': {},
        'Note_Expand_Length_Dict': {},
        'Duration_Expand_Length_Dict': {},
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
            with open(os.path.join(root, file).replace('\\', '/'), 'rb') as f:
                pattern_dict = pickle.load(f)
            file = os.path.join(root, file).replace('\\', '/').replace(pattern_path, '').lstrip('/')
            try:
                if not all([
                    key in pattern_dict.keys()
                    for key in ('Latent_Code', 'F0', 'Lyric', 'Note', 'Singer', 'Genre', 'Dataset')
                    ]):
                    print(pattern_dict.keys())
                    continue

                new_metadata_dict['F0_Length_Dict'][file] = pattern_dict['F0'].shape[0]
                new_metadata_dict['Lyric_Length_Dict'][file] = len(pattern_dict['Lyric'])
                new_metadata_dict['Note_Length_Dict'][file] = len(pattern_dict['Note'])
                new_metadata_dict['Duration_Length_Dict'][file] = len(pattern_dict['Lyric_Duration'])
                new_metadata_dict['Lyric_Expand_Length_Dict'][file] = len(pattern_dict['Lyric_Expand'])
                new_metadata_dict['Note_Expand_Length_Dict'][file] = len(pattern_dict['Note_Expand'])
                new_metadata_dict['Duration_Expand_Length_Dict'][file] = len(pattern_dict['Lyric_Duration_Expand'])
                new_metadata_dict['Singer_Dict'][file] = pattern_dict['Singer']
                new_metadata_dict['File_List'].append(file)
                if not pattern_dict['Singer'] in new_metadata_dict['File_List_by_Singer_Dict'].keys():
                    new_metadata_dict['File_List_by_Singer_Dict'][pattern_dict['Singer']] = []
                new_metadata_dict['File_List_by_Singer_Dict'][pattern_dict['Singer']].append(file)

                if not pattern_dict['Singer'] in latent_dict.keys():
                    latent_dict[pattern_dict['Singer']] = {'Min': math.inf, 'Max': -math.inf}
                if not pattern_dict['Singer'] in f0_dict.keys():
                    f0_dict[pattern_dict['Singer']] = []
                
                latent = hificodec.quantizer.embed(torch.from_numpy(pattern_dict['Latent_Code']).T[None].long().to(device))[0].cpu()
                latent_dict[pattern_dict['Singer']]['Min'] = min(latent_dict[pattern_dict['Singer']]['Min'], latent.min().item())
                latent_dict[pattern_dict['Singer']]['Max'] = max(latent_dict[pattern_dict['Singer']]['Max'], latent.max().item())
                
                f0_dict[pattern_dict['Singer']].append(pattern_dict['F0'])
                tokens.extend(pattern_dict['Lyric'])
                singers.append(pattern_dict['Singer'])
                genres.append(pattern_dict['Genre'])
                languages.append(pattern_dict['Language'])

                min_note = min(min_note, *[x for x in pattern_dict['Note'] if x > 0])
                max_note = max(max_note, *[x for x in pattern_dict['Note'] if x > 0])
            except Exception as e:
                logging.critical('File \'{}\' is not correct pattern file. This file is ignored. Error: {}'.format(file, e))
            files_tqdm.update(1)

    new_metadata_dict['Min_Note'] = min_note
    new_metadata_dict['Max_Note'] = max_note

    with open(os.path.join(pattern_path, metadata_file.upper()).replace('\\', '/'), 'wb') as f:
        pickle.dump(new_metadata_dict, f, protocol= 4)

    if not eval:
        yaml.dump(
            latent_dict,
            open(hp.Latent_Info_Path, 'w')
            )
        
        f0_info_dict = {}
        for singer, f0_list in f0_dict.items():
            f0 = np.hstack(f0_list).astype(np.float32)
            f0 = np.clip(f0, 0.0, np.inf)
            
            f0_info_dict[singer] = {
                'Mean': f0.mean().item(),
                'Std': f0.std().item(),
                'Min': f0.min().item(),
                'Max': f0.max().item(),
                }
        yaml.dump(
            f0_info_dict,
            open(hp.F0_Info_Path, 'w')
            )

        Token_Dict_Generate(hp, sorted(list(set(tokens))))

        singer_index_dict = {
            singer: index
            for index, singer in enumerate(sorted(set(singers)))
            }
        yaml.dump(
            singer_index_dict,
            open(hyper_parameters.Singer_Info_Path, 'w')
            )

        genre_index_dict = {
            genre: index
            for index, genre in enumerate(sorted(set(genres)))
            }
        yaml.dump(
            genre_index_dict,
            open(hyper_parameters.Genre_Info_Path, 'w')
            )
        
        language_index_dict = {
            language: index
            for index, language in enumerate(sorted(set(languages)))
            }
        yaml.dump(
            language_index_dict,
            open(hyper_parameters.Language_Info_Path, 'w')
            )

    logging.info('Metadata generate done.')


if __name__ == '__main__':
    argparser = argparse.ArgumentParser()
    argparser.add_argument('-csd', '--csd_path', required= False)
    argparser.add_argument('-csde', '--csd_eng_path', required= False)
    argparser.add_argument('-am', '--aihub_mediazen_path', required= False)
    argparser.add_argument('-m4', '--m4singer_path', required= False)
    argparser.add_argument('-nus48e', '--nus48e_path', required= False)
    argparser.add_argument('-kiritan', '--kiritan_path', required= False)
    argparser.add_argument('-hp', '--hyper_paramters', required= True)
    argparser.add_argument('-verbose', '--verbose', action= 'store_true')
    args = argparser.parse_args()

    hp = Recursive_Parse(yaml.load(
        open(args.hyper_paramters, encoding='utf-8'),
        Loader=yaml.Loader
        ))

    if args.csd_path:
        CSD_Fix(
            hyper_paramters= hp,
            dataset_path= args.csd_path,
            verbose= args.verbose
            )
    if args.csd_eng_path:
        CSD_Eng_Fix(
            hyper_paramters= hp,
            dataset_path= args.csd_eng_path,
            verbose= args.verbose
            )
    
    if args.aihub_mediazen_path:
        AIHub_Mediazen(
            hyper_paramters= hp,
            dataset_path= args.aihub_mediazen_path,
            verbose= args.verbose
            )
        
    if args.m4singer_path:
        M4Singer(
            hyper_paramters= hp,
            dataset_path= args.m4singer_path,
            verbose= args.verbose
            )
   
    if args.nus48e_path:
        NUS48E(
            hyper_paramters= hp,
            dataset_path= args.nus48e_path,
            verbose= args.verbose
            )
        
    if args.kiritan_path:
        Kiritan(
            hyper_paramters= hp,
            dataset_path= args.kiritan_path,
            verbose= args.verbose
            )

    Metadata_Generate(hp, False)
    Metadata_Generate(hp, True)

# python -m Pattern_Generator -hp Hyper_Parameters.yaml -csd /mnt/f/Rawdata_Music/CSD_Fix
