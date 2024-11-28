import torch
import numpy as np
import pickle, os
from typing import Dict, List, Optional
import functools
from random import choice, sample, shuffle
from typing import Union

from Pattern_Generator import Korean_G2P_Lyric, English_Phoneme_Split, Chinese_G2P_Lyric, Japanese_G2P_Lyric, Convert_Feature_Based_Duration

def Lyric_to_Token(lyric: List[str], token_dict: Dict[str, int]):
    return [
        token_dict[letter]
        for letter in list(lyric)
        ]

def Alignment(lengths: Union[List[int], np.ndarray]) -> np.ndarray:
    alignment = np.zeros(shape= (len(lengths), sum(lengths)), dtype= int)

    start = 0
    for row_index, length in enumerate(lengths):
        alignment[row_index, start:start + length] = 1

def Token_Stack(tokens: List[List[int]], token_dict: Dict[str, int], max_length: Optional[int]= None):
    max_token_length = max_length or max([len(token) for token in tokens])
    tokens = np.stack(
        [np.pad(token[:max_token_length], [0, max_token_length - len(token[:max_token_length])], constant_values= token_dict['<X>']) for token in tokens],
        axis= 0
        )
    return tokens

def Token_on_Note_Length_Stack(token_on_note_lengths: List[List[int]], max_length: Optional[int]= None):
    max_length = max_length or max([len(token_on_note_length) for token_on_note_length in token_on_note_lengths])
    token_on_note_lengths = np.stack(
        [
            np.pad(
                token_on_note_length[:max_length],
                [0, max_length - len(token_on_note_length[:max_length])],
                constant_values= 0
                )
            for token_on_note_length in token_on_note_lengths
            ],
        axis= 0
        )
    return token_on_note_lengths

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

def Singer_Stack(singers: List[List[int]], max_length: Optional[int]= None):
    max_length = max_length or max([len(singer) for singer in singers])
    singers = np.stack(
        [np.pad(singer[:max_length], [0, max_length - len(singer[:max_length])], constant_values= 0) for singer in singers],
        axis= 0
        )
    return singers

def Language_Stack(languages: List[List[int]], max_length: Optional[int]= None):
    max_length = max_length or max([len(note) for note in languages])
    languages = np.stack(
        [np.pad(language[:max_length], [0, max_length - len(language[:max_length])], constant_values= 0) for language in languages],
        axis= 0
        )
    return languages

def Mel_Stack(mels: List[np.ndarray], max_length: Optional[int]= None):
    max_mel_length = max_length or max([mel.shape[1] for mel in mels])
    mels = np.stack(
        [np.pad(mel, [[0, 0], [0, max_mel_length - mel.shape[1]]], constant_values= mel.min()) for mel in mels],
        axis= 0
        )
    return mels


class Dataset(torch.utils.data.Dataset):
    def __init__(
        self,
        token_dict: Dict[str, int],
        singer_dict: Dict[str, int],
        language_dict: Dict[str, int],
        pattern_path: str,
        metadata_file: str,
        num_combination: int,
        pattern_length_min: int,
        pattern_length_max: int,
        accumulated_dataset_epoch: int= 1,
        augmentation_ratio: float= 0.0,
        use_pattern_cache: bool= False
        ):
        super().__init__()
        assert pattern_length_min % num_combination == 0, f'pattern_length_min must be dividable by num_combiation: {pattern_length_min} % {num_combination} != 0'
        assert pattern_length_max % num_combination == 0, f'pattern_length_max must be dividable by num_combiation: {pattern_length_max} % {num_combination} != 0'
        
        self.token_dict = token_dict
        self.singer_dict = singer_dict
        self.language_dict = language_dict
        self.pattern_path = pattern_path
        self.num_combination = num_combination
        self.pattern_length_min = pattern_length_min
        self.pattern_length_max = pattern_length_max
        self.sample_pattern_length_min = pattern_length_min // num_combination
        self.sample_pattern_length_max = pattern_length_max // num_combination

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
        token_combination, token_on_note_length_combination, \
        note_combination, duration_combination, \
        singer_combination, language_combination, mel_combination = [], [], [], [], [], [], []

        for sample_pattern in [self.patterns[idx]] + list(sample(self.patterns, self.num_combination - 1)):
            token, token_on_note_length, note, duration, singer, language, mel = self.Pattern_LRU_Cache(
                os.path.join(self.pattern_path, sample_pattern).replace('\\', '/')
                )
            
            # note offset
            while True:
                note_offset_start = np.random.randint(0, len(note))
                note_offset_end = None
                for offset_end_candidate in range(note_offset_start + 1, len(note) + 1):    # as large as possible
                    if sum(duration[note_offset_start:offset_end_candidate]) < self.sample_pattern_length_min:
                        continue
                    elif sum(duration[note_offset_start:offset_end_candidate]) > self.sample_pattern_length_max:
                        break
                    note_offset_end = offset_end_candidate
                if not note_offset_end is None:
                    break
            
            # mel code offset
            mel_offset_start, mel_offset_end = sum(duration[:note_offset_start]), sum(duration[:note_offset_end])

            token = [x for token_on_note in token[note_offset_start:note_offset_end] for x in token_on_note]
            token_on_note_length = token_on_note_length[note_offset_start:note_offset_end]
            note = note[note_offset_start:note_offset_end]
            duration = duration[note_offset_start:note_offset_end]
            singer = [singer] * len(token)
            language = [language] * len(token)
            mel = mel[:, mel_offset_start:mel_offset_end]

            token_combination.extend(token)
            token_on_note_length_combination.extend(token_on_note_length)
            note_combination.extend(note)
            duration_combination.extend(duration)
            singer_combination.extend(singer)
            language_combination.extend(language)
            mel_combination.append(mel)

        mel_combination = np.concatenate(mel_combination, axis= 1)
        
        return \
            token_combination, token_on_note_length_combination, \
            note_combination, duration_combination, \
            singer_combination, language_combination, mel_combination
    
    def Pattern_LRU_Cache(self, path: str):
        pattern_dict = pickle.load(open(path, 'rb'))

        duration, pronunciation, note, _ = zip(*pattern_dict['Music'])
        token, token_on_note_length = zip(*[
            (Lyric_to_Token(x, self.token_dict), len(x))
            for x in pronunciation
            ])
        singer = self.singer_dict[pattern_dict['Singer']]
        language = self.language_dict[pattern_dict['Language']]
        mel = pattern_dict['Mel']
        
        return token, token_on_note_length, note, duration, singer, language, mel

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
            if language == 'Korean':
                music = Korean_G2P_Lyric(music)
            elif language == 'English':
                music = English_Phoneme_Split(music)
            elif language == 'Chinese':
                music = Chinese_G2P_Lyric(music)
            elif language == 'Japanese':
                music = Japanese_G2P_Lyric(music)
            else:
                raise NotImplementedError(f'Unsupported language: {language}')
            music = Convert_Feature_Based_Duration(
                music= music,
                sample_rate= sample_rate,
                hop_size= hop_size
                )

            self.patterns.append((
                music,
                singer,
                language
                ))

    def __getitem__(self, idx):
        music, singer, language = self.patterns[idx]

        duration, pronunciation, note, lyric = zip(*music)
        token, token_on_note_length = zip(*[
            (Lyric_to_Token(x, self.token_dict), len(x))
            for x in pronunciation
            ])
        token = [x for token_on_note in token for x in token_on_note]
        singer = [self.singer_dict[singer]] * len(token)
        language = [self.language_dict[language]] * len(token)

        return token, token_on_note_length, note, duration, singer, language, lyric

    def __len__(self):
        return len(self.patterns)


class Collater:
    def __init__(
        self,
        token_dict: Dict[str, int],
        ):
        self.token_dict = token_dict

    def __call__(self, batch):
        tokens, token_on_note_lengths, notes, durations, singers, languages, mels = zip(*batch)
        
        token_lengths = np.array([len(token) for token in tokens])
        note_lengths = np.array([len(note) for note in notes])
        mel_lengths = np.array([mel.shape[1] for mel in mels])

        tokens = Token_Stack(tokens, self.token_dict)
        token_on_note_lengths = Token_on_Note_Length_Stack(token_on_note_lengths)
        notes = Note_Stack(notes)
        durations = Duration_Stack(durations)
        singers = Singer_Stack(singers)
        languages = Language_Stack(languages)
        mels = Mel_Stack(mels)

        tokens = torch.IntTensor(tokens)   # [Batch, Token_t]
        token_lengths = torch.IntTensor(token_lengths)  # [Batch]
        token_on_note_lengths = torch.IntTensor(token_on_note_lengths) # [Batch, Token_t]

        notes = torch.IntTensor(notes) # [Batch, Note_t]
        durations = torch.IntTensor(durations) # [Batch, Note_t]
        note_lengths = torch.IntTensor(note_lengths)    # [Batch]

        singers = torch.IntTensor(singers)  # [Batch, Token_t]
        languages = torch.IntTensor(languages)  # [Batch, Token_t]

        mels = torch.FloatTensor(mels)  # [Batch, N_Mel, Mel_t]
        mel_lengths = torch.IntTensor(mel_lengths)    # [Batch]
        
        return \
            tokens, token_lengths, token_on_note_lengths, \
            notes, durations, note_lengths, \
            singers, languages, \
            mels, mel_lengths

class Inference_Collater:
    def __init__(self,
        token_dict: Dict[str, int]
        ):
        self.token_dict = token_dict
         
    def __call__(self, batch):
        tokens, token_on_note_lengths, notes, durations, singers, languages, lyrics = zip(*batch)

        token_lengths = np.array([len(token) for token in tokens])
        note_lengths = np.array([len(note) for note in notes])
        mel_lengths = np.array([sum(duration) for duration in durations])

        tokens = Token_Stack(tokens, self.token_dict)
        token_on_note_lengths = Token_on_Note_Length_Stack(token_on_note_lengths)
        notes = Note_Stack(notes)
        durations = Duration_Stack(durations)
        singers = Singer_Stack(singers)
        languages = Language_Stack(languages)
        
        tokens = torch.IntTensor(tokens)   # [Batch, Token_t]
        token_lengths = torch.IntTensor(token_lengths)  # [Batch]
        token_on_note_lengths = torch.IntTensor(token_on_note_lengths) # [Batch, Note_t], not Token_t

        notes = torch.IntTensor(notes) # [Batch, Note_t]
        durations = torch.IntTensor(durations) # [Batch, Note_t]
        note_lengths = torch.IntTensor(note_lengths)    # [Batch]

        singers = torch.IntTensor(singers)  # [Batch, Token_t]
        languages = torch.IntTensor(languages)  # [Batch, Token_t]

        mel_lengths = torch.IntTensor(mel_lengths)    # [Batch]

        return \
            tokens, token_lengths, token_on_note_lengths, \
            notes, durations, note_lengths, \
            singers, languages, \
            mel_lengths, lyrics