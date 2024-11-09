import torch
import librosa, asyncio
import numpy as np
import pickle, os
from typing import Dict, List, Optional
import functools
from random import choice

from Pattern_Generator import G2P_Lyric, Convert_Feature_Based_Music, Expand_by_Duration

def Lyric_to_Token(lyric: List[str], token_dict: Dict[str, int]):
    return [
        token_dict[letter]
        for letter in list(lyric)
        ]

def Token_Stack(tokens: List[List[int]], token_dict: Dict[str, int], max_length: Optional[int]= None):
    max_token_length = max_length or max([len(token) for token in tokens])
    tokens = np.stack(
        [np.pad(token[:max_token_length], [0, max_token_length - len(token[:max_token_length])], constant_values= token_dict['<X>']) for token in tokens],
        axis= 0
        )
    return tokens

def Note_Stack(notes: List[List[int]], max_length: Optional[int]= None):
    max_note_length = max_length or max([len(note) for note in notes])
    notes = np.stack(
        [np.pad(note[:max_note_length], [0, max_note_length - len(note[:max_note_length])], constant_values= 0) for note in notes],
        axis= 0
        )
    return notes

def Duration_Stack(durations: List[List[int]], max_length: Optional[int]= None):
    max_duration_length = max_length or max([len(duration) for duration in durations])
    durations = np.stack(
        [np.pad(duration[:max_duration_length], [0, max_duration_length - len(duration[:max_duration_length])], constant_values= 0) for duration in durations],
        axis= 0
        )
    return durations

def Latent_Stack(latents: List[np.ndarray], max_length: Optional[int]= None):
    max_latent_length = max_length or max([latent.shape[1] for latent in latents])
    latents = np.stack(
        [np.pad(latent, [[0, 0], [0, max_latent_length - latent.shape[1]]], constant_values= 0) for latent in latents],
        axis= 0
        )
    return latents

class Dataset(torch.utils.data.Dataset):
    def __init__(
        self,
        token_dict: Dict[str, int],
        singer_dict: Dict[str, int],
        language_dict: Dict[str, int],
        pattern_path: str,
        metadata_file: str,
        accumulated_dataset_epoch: int= 1,
        augmentation_ratio: float= 0.0,
        use_pattern_cache: bool= False
        ):
        super().__init__()
        self.token_dict = token_dict
        self.singer_dict = singer_dict
        self.language_dict = language_dict
        self.pattern_path = pattern_path

        metadata_dict = pickle.load(open(
            os.path.join(pattern_path, metadata_file).replace('\\', '/'), 'rb'
            ))

        self.patterns = []
        max_pattern_by_singer = max([
            len(patterns)
            for patterns in metadata_dict['File_List_by_Singer_Dict'].values()
            ])
        for patterns in metadata_dict['File_List_by_Singer_Dict'].values():
            ratio = float(len(patterns)) / float(max_pattern_by_singer)
            if ratio < augmentation_ratio:
                patterns *= int(np.ceil(augmentation_ratio / ratio))
            self.patterns.extend(patterns)

        self.patterns = self.patterns * accumulated_dataset_epoch

        if use_pattern_cache:
            self.Pattern_LRU_Cache = functools.lru_cache(maxsize= None)(self.Pattern_LRU_Cache)

    def __getitem__(self, idx):
        path = os.path.join(self.pattern_path, self.patterns[idx]).replace('\\', '/')
        return self.Pattern_LRU_Cache(path)
    
    def Pattern_LRU_Cache(self, path: str):
        pattern_dict = pickle.load(open(path, 'rb'))
        
        token = Lyric_to_Token(
            lyric= pattern_dict['Lyric_Expand'],
            token_dict= self.token_dict
            )
        note = pattern_dict['Note_Expand']
        lyric_duration = pattern_dict['Lyric_Duration_Expand']
        note_duration = pattern_dict['Note_Duration_Expand']
        singer = self.singer_dict[pattern_dict['Singer']]
        language = self.language_dict[pattern_dict['Language']]
        latent_code = pattern_dict['Latent_Code']

        return token, note, lyric_duration, note_duration, singer, language, latent_code

    def __len__(self):
        return len(self.patterns)

class Inference_Dataset(torch.utils.data.Dataset):
    def __init__(
        self,
        token_dict: Dict[str, int],
        singer_dict: Dict[str, int],
        language_dict: Dict[str, int],
        lyrics: List[List[str]],
        notes: List[List[int]],
        durations: List[List[float]],
        singers: List[str],
        languages: List[str],
        sample_rate: int,
        hop_size: int
        ):
        super().__init__()
        self.token_dict = token_dict
        self.singer_dict = singer_dict
        self.language_dict = language_dict

        self.patterns = []
        for index, (lyric, note, duration, singer, language) in enumerate(zip(lyrics, notes, durations, singers, languages)):
            music = [x for x in zip(duration, lyric, note)]
            
            text = lyric
            
            if language == 'Korean':
                music = G2P_Lyric(music)
            else:
                raise NotImplementedError(f'Unsupported language: {language}')
            lyric, note, lyric_duration, note_duration = Convert_Feature_Based_Music(
                music= music,
                sample_rate= sample_rate,
                hop_size= hop_size,
                verbose= True
                )
            lyric_expand, note_expand, lyric_duration_expand, note_duration_expand = Expand_by_Duration(
                lyrics= lyric,
                notes= note,
                lyric_durations= lyric_duration,
                note_durations= note_duration
                )

            self.patterns.append((
                lyric_expand,
                note_expand,
                lyric_duration_expand,
                note_duration_expand,
                singer,
                language,
                text
                ))

    def __getitem__(self, idx):
        lyric, note, lyric_duration_expand, note_duration_expand, singer, language, text = self.patterns[idx]

        return \
            Lyric_to_Token(lyric, self.token_dict), note, lyric_duration_expand, note_duration_expand, \
            self.singer_dict[singer], self.language_dict[language], text

    def __len__(self):
        return len(self.patterns)


class Collater:
    def __init__(
        self,
        token_dict: Dict[str, int],
        pattern_length: int,
        hop_size: int
        ):
        self.token_dict = token_dict
        self.pattern_length = pattern_length
        self.hop_size = hop_size

    def __call__(self, batch):
        tokens, notes, lyric_durations, note_durations, singers, languages, latent_codes = zip(*batch)

        offsets = []
        for token in tokens:
            counts = 0
            while True:
                offset = np.random.randint(0, len(token) - self.pattern_length + 1)
                number_of_non_silence = sum([
                    x != self.token_dict['<X>']
                    for x in token[offset:offset+self.pattern_length]
                    ])
                if number_of_non_silence / self.pattern_length > 0.5 or counts > 10:
                    offsets.append(offset)
                    break
                counts += 1

        tokens = Token_Stack([
            token[offset:offset+self.pattern_length]
            for token, offset in zip(tokens, offsets)
            ], self.token_dict)
        notes = Note_Stack([
            note[offset:offset+self.pattern_length]
            for note, offset in zip(notes, offsets)
            ])
        lyric_durations = Duration_Stack([
            duration[offset:offset+self.pattern_length]
            for duration, offset in zip(lyric_durations, offsets)
            ])
        note_durations = Duration_Stack([
            duration[offset:offset+self.pattern_length]
            for duration, offset in zip(note_durations, offsets)
            ])
        singers = np.array(singers)
        languages = np.array(languages)
        latent_codes = Latent_Stack([
            latent_code[:, offset:offset+self.pattern_length]
            for latent_code, offset in zip(latent_codes, offsets)
            ])

        tokens = torch.IntTensor(tokens)   # [Batch, Token_t]
        notes = torch.IntTensor(notes) # [Batch, Token_t]
        lyric_durations = torch.IntTensor(lyric_durations) # [Batch, Token_t]
        note_durations = torch.IntTensor(note_durations) # [Batch, Token_t]
        lengths = torch.IntTensor([self.pattern_length] * len(batch))
        singers = torch.IntTensor(singers)  # [Batch]
        languages = torch.IntTensor(languages)  # [Batch]
        latent_codes = torch.LongTensor(latent_codes)  # [Batch, Latent_Code_n, Latent_t]
        
        return tokens, notes, lyric_durations, note_durations, lengths, singers, languages, latent_codes

class Inference_Collater:
    def __init__(self,
        token_dict: Dict[str, int]
        ):
        self.token_dict = token_dict
         
    def __call__(self, batch):
        tokens, notes, lyric_durations, note_durations, singers, languages, lyrics = zip(*batch)
        
        lengths = np.array([len(token) for token in tokens])
        max_length = max(lengths)
        
        tokens = Token_Stack(tokens, self.token_dict, max_length)
        notes = Note_Stack(notes, max_length)
        lyric_durations = Duration_Stack(lyric_durations, max_length)
        note_durations = Duration_Stack(note_durations, max_length)
        singers = np.array(singers)
        languages = np.array(languages)

        tokens = torch.IntTensor(tokens)   # [Batch, Time]
        notes = torch.IntTensor(notes)   # [Batch, Time]
        lyric_durations = torch.IntTensor(lyric_durations)   # [Batch, Time]
        note_durations = torch.IntTensor(note_durations)   # [Batch, Time]
        lengths = torch.IntTensor(lengths)   # [Batch]
        singers = torch.IntTensor(singers)  # [Batch]
        languages = torch.IntTensor(languages)  # [Batch]
        
        lyrics = [''.join([(x if x != '<X>' else ' ') for x in lyric]) for lyric in lyrics]

        return tokens, notes, lyric_durations, note_durations, lengths, singers, languages, lyrics