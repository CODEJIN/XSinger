import torch
import numpy as np
from tqdm import tqdm
import pickle, yaml, os, librosa
from random import shuffle
from scipy.io import wavfile
from typing import Dict, List, Optional, Tuple, Union
from pesq import pesq
from multiprocessing import Pool
import whisper
import jiwer

from Arg_Parser import Recursive_Parse

from Inference import Inferencer
from Datasets import \
    Lyric_to_Token, Token_Stack, Token_on_Note_Length_Stack, \
    Note_Stack, Duration_Stack, Singer_Stack, Language_Stack, Mel_Stack

hp_path = 'Hyper_Parameters.yaml'
checkpoint_path = './results/Checkpoint/S_47648'
num_patterns = 64
pattern_length_min = 1024
pattern_length_max = 2048
batch_size = 8

hp = Recursive_Parse(yaml.load(
    open(hp_path, encoding='utf-8'),
    Loader=yaml.Loader
    ))

if checkpoint_path is None:
    checkpoint_path = max(
        [
            os.path.join(hp.CheckPoint_Path, path).replace('\\', '/')
            for path in os.listdir(hp.Checkpoint_Path)
            if path.startswith('S_')
            ],
        key= os.path.getctime
        )
    
inferencer = Inferencer(
    hp_path= hp_path,
    checkpoint_path= checkpoint_path
    )

token_dict = yaml.load(open(hp.Token_Path, encoding= 'utf-8-sig'), Loader=yaml.Loader)
singer_dict = yaml.load(open(hp.Singer_Info_Path, 'r', encoding= 'utf-8-sig'), Loader=yaml.Loader)
language_dict = yaml.load(open(hp.Language_Info_Path, encoding= 'utf-8-sig'), Loader=yaml.Loader)

reference_paths = sorted([
    os.path.join(root, file)
    for root, _, files in os.walk(hp.Train.Eval_Pattern.Path)
    for file in files
    if os.path.splitext(file)[1].upper() == '.PICKLE' and file.upper() != hp.Train.Eval_Pattern.Metadata_File
    ])
reference_patterns = [
    pickle.load(open(reference_path, 'rb'))
    for reference_path in reference_paths
    ]

num_patterns = [
    num_patterns // len(reference_paths) + (1 if index < num_patterns % len(reference_paths) else 0)
    for index in range(len(reference_paths))
    ]

tokens, languages, token_lengths, token_on_note_lengths = [], [], [], []
notes, durations, note_lengths, singers, mel_lengths, audio_gts = [], [], [], [], [], []
language_labels = []
for reference_pattern, num_pattern in zip(reference_patterns, num_patterns):
    duration, pronunciation, note, _ = zip(*reference_pattern['Music'])
    token, token_on_note_length = zip(*[
        (Lyric_to_Token(x, token_dict), len(x))
        for x in pronunciation
        ])
    
    singer = singer_dict[reference_pattern['Singer']]
    language = language_dict[reference_pattern['Language']]
    mel = reference_pattern['Mel']
    language_label = reference_pattern['Language']

    note_offsets = []
    note_offset_start = 0
    while len(note_offsets) < num_pattern:
        while True:
            if note[note_offset_start] == 0:
                note_offset_start += 1
                continue
            note_offset_end = None
            for offset_end_candidate in range(note_offset_start + 1, len(note) + 1): # as large as possible
                if sum(duration[note_offset_start:offset_end_candidate]) < pattern_length_min:
                    continue
                elif sum(duration[note_offset_start:offset_end_candidate]) > pattern_length_max:
                    break
                note_offset_end = offset_end_candidate
            if not note_offset_end is None:
                break
            else:
                note_offset_start += 1
        
        note_offsets.append((note_offset_start, note_offset_end))
        note_offset_start = note_offset_end

    for note_offset_start, note_offset_end in note_offsets:
        feature_offset_start, feature_offset_end = sum(duration[:note_offset_start]), sum(duration[:note_offset_end])
        audio_offset_start, audio_offset_end = feature_offset_start * hp.Sound.Hop_Size, feature_offset_end * hp.Sound.Hop_Size

        tokens.append([x for token_on_note in token[note_offset_start:note_offset_end] for x in token_on_note])
        token_lengths.append(len(tokens[-1]))
        token_on_note_lengths.append(token_on_note_length[note_offset_start:note_offset_end])
        notes.append(note[note_offset_start:note_offset_end])
        note_lengths.append(len(notes[-1]))
        durations.append(duration[note_offset_start:note_offset_end])
        singers.append([singer] * len(tokens[-1]))
        languages.append([language] * len(tokens[-1]))
        mel_lengths.append(sum(durations[-1]))
        audio_gts.append(inferencer.vocoder(
            torch.from_numpy(mel[None, :, feature_offset_start:feature_offset_end]).float().to(inferencer.device)
            )[0].cpu().numpy())
        language_labels.append(language_label)

tokens = torch.IntTensor(Token_Stack(tokens, token_dict))
token_lengths = torch.IntTensor(np.array(token_lengths))
token_on_note_lengths = torch.IntTensor(Token_on_Note_Length_Stack(token_on_note_lengths))
notes = torch.IntTensor(Note_Stack(notes))
durations = torch.IntTensor(Duration_Stack(durations))
note_lengths = torch.IntTensor(np.array(note_lengths))
singers = torch.IntTensor(Singer_Stack(singers))
languages = torch.IntTensor(Language_Stack(languages))
mel_lengths = torch.IntTensor(np.array(mel_lengths))

step = int(checkpoint_path.split('/')[-1][2:]) // 1000

os.makedirs('./Quality_Check', exist_ok= True)
os.makedirs(f'./Quality_Check/{step:04d}K.GT', exist_ok= True)
for index, (audio, language_label) in enumerate(zip(audio_gts, language_labels)):
    path = f'{index:05d}.{language_label}.wav'
    wavfile.write(os.path.join(f'./Quality_Check/{step:04d}K.GT', path), hp.Sound.Sample_Rate, (audio * 32767.5).astype(np.int16))

os.makedirs(f'./Quality_Check/{step:04d}K', exist_ok= True)

prediction_audios = []
for index in range(0, tokens.size(0), batch_size):
    prediction_audios.extend(inferencer.Inference_Step(
        tokens= tokens[index:index + batch_size],
        token_lengths= token_lengths[index:index + batch_size],
        token_on_note_lengths= token_on_note_lengths[index:index + batch_size],
        notes= notes[index:index + batch_size],
        durations= durations[index:index + batch_size],
        note_lengths= note_lengths[index:index + batch_size],
        singers= singers[index:index + batch_size],
        languages= languages[index:index + batch_size],
        mel_lengths= mel_lengths[index:index + batch_size],
        ))
for index, (audio, language_label) in enumerate(zip(prediction_audios, language_labels)):
    path = f'{index:05d}.{language_label}.wav'
    wavfile.write(os.path.join(f'./Quality_Check/{step:04d}K', path).replace('\\', '/'), hp.Sound.Sample_Rate, (audio * 32767.5).astype(np.int16))


def PESQ(ground_truth_path: str, prediction_path: str):
    pairs = []
    for ground_truth_file in tqdm(os.listdir(ground_truth_path), desc= 'PESQ'):
        if not os.path.exists(os.path.join(prediction_path, ground_truth_file)):
            continue
        prediction_file = os.path.join(prediction_path, ground_truth_file)
        ground_truth_file = os.path.join(ground_truth_path, ground_truth_file)
        pairs.append((ground_truth_file, prediction_file))

    scores = Pool(8).map(_Calc_PESQ, pairs)
    return np.nanmean(scores)

def _Calc_PESQ(pair: Tuple[str, str]):
    ground_truth_audio = (librosa.load(pair[0], sr= 16000)[0] * 32768).astype(np.int16)
    prediction_audio = (librosa.load(pair[1], sr= 16000)[0] * 32768).astype(np.int16)
    try:
        return pesq(
            fs= 16000,
            ref= ground_truth_audio,
            deg= prediction_audio,
            mode= 'nb'
            )
    except:
        return np.nan

asr= whisper.load_model('large')
asr_language_code_dict = {
    'Korean': 'ko',
    'English': 'en',
    'Chinese': 'zh',
    'Japanese': 'ja'
    }
def CWER(ground_truth_path: str, prediction_path: str):
    cers, wers = [], []

    for ground_truth_file in tqdm(sorted(os.listdir(ground_truth_path)), desc= 'CWER'):
        if not os.path.exists(os.path.join(prediction_path, ground_truth_file)):
            continue
        prediction_file = os.path.join(prediction_path, ground_truth_file)
        ground_truth_file = os.path.join(ground_truth_path, ground_truth_file)
        language = ground_truth_file.split('.')[-2]

        gt_text = asr.transcribe(ground_truth_file, language= asr_language_code_dict[language])['text']
        prediction_text = asr.transcribe(prediction_file, language= asr_language_code_dict[language])['text']

        try:
            cers.append(jiwer.cer(gt_text, prediction_text))
            wers.append(jiwer.wer(gt_text, prediction_text))
        except:
            print(ground_truth_file, gt_text)
            print(prediction_file, prediction_text)

    return np.nanmean(cers), np.nanmean(wers)


pesq_score = PESQ(
    ground_truth_path= f'./Quality_Check/{step:04d}K.GT',
    prediction_path= f'./Quality_Check/{step:04d}K',
    )
print(step, pesq_score)

cer_score, wer_score = CWER(
    ground_truth_path= f'./Quality_Check/{step:04d}K.GT',
    prediction_path= f'./Quality_Check/{step:04d}K',
    )
print(step, cer_score, wer_score)