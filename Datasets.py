import torch
import numpy as np
import pickle, os
from typing import Dict, List, Optional
import functools
from random import choice, sample, shuffle
from typing import Union

from Pattern.Inference import Korean, English, Japanese, Chinese
from Pattern.Generator import Note, Dict_to_Note

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

def Tech_Stack(techs: List[np.ndarray], max_length: Optional[int]= None):
    max_tech_length = max_length or max([tech.shape[1] for tech in techs])
    techs = np.stack(
        [np.pad(tech, [[0, 0], [0, max_tech_length - tech.shape[1]]], constant_values= 0) for tech in techs],
        axis= 0
        )
    return techs

def Mel_Stack(mels: List[np.ndarray], max_length: Optional[int]= None):
    max_mel_length = max_length or max([mel.shape[1] for mel in mels])
    mels = np.stack(
        [np.pad(mel, [[0, 0], [0, max_mel_length - mel.shape[1]]], constant_values= mel.min()) for mel in mels],
        axis= 0
        )
    return mels

def F0_Stack(f0s: List[np.ndarray], max_length: Optional[int]= None):
    max_f0_length = max_length or max([f0.shape[0] for f0 in f0s])
    f0s = np.stack(
        [np.pad(f0, [0, max_f0_length - f0.shape[0]], constant_values= 0.0) for f0 in f0s],
        axis= 0
        )
    return f0s

class Dataset(torch.utils.data.Dataset):
    def __init__(
        self,
        token_dict: Dict[str, int],
        singer_dict: Dict[str, int],
        language_dict: Dict[str, int],
        metadata_path: str,
        pattern_length_min: int,
        pattern_length_max: int,
        accumulated_dataset_epoch: int= 1,
        augmentation_ratio: float= 0.0,
        use_pattern_cache: bool= False
        ):
        super().__init__()
        
        self.token_dict = token_dict
        self.singer_dict = singer_dict
        self.language_dict = language_dict
        self.pattern_length_min = pattern_length_min
        self.pattern_length_max = pattern_length_max
        self.dataset_path = os.path.dirname(metadata_path)

        metadata_dict = pickle.load(open(metadata_path, 'rb'))

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
        token, token_on_note_length, note, \
        duration, tech, singer, language, mel, f0 = self.Pattern_LRU_Cache(
            os.path.join(self.dataset_path, self.patterns[idx]).replace('\\', '/')
            )
        singer = [singer] * len(token)
        language = [language] * len(token)

        return \
            token, token_on_note_length, \
            note, duration, \
            singer, language, \
            tech, mel, f0
    
    def Pattern_LRU_Cache(self, path: str):
        pattern_dict = pickle.load(open(path, 'rb'))
        music: list[Note] = [Dict_to_Note(x) for x in pattern_dict['Music']]
        
        duration, pronunciation, pitch = zip(*[
            (note.Duration, note.Lyric, note.Pitch)
            for note in music
            ])
        tech = [
            note.Tech
            for note in music
            for _ in range(note.Duration)
            ]        
        token, token_on_note_length = zip(*[
            (Lyric_to_Token(x, self.token_dict), len(x))
            for x in pronunciation
            ])
        token = [x for token_on_note in token for x in token_on_note]
        
        tech = np.array(tech, dtype= np.int32).T
        singer = self.singer_dict[pattern_dict['Singer']]
        language = self.language_dict[pattern_dict['Language']]
        mel = pattern_dict['Mel']
        f0 = pattern_dict['F0']

        return token, token_on_note_length, pitch, duration, tech, singer, language, mel, f0

    def __len__(self):
        return len(self.patterns)

class Inference_Dataset(torch.utils.data.Dataset):
    def __init__(
        self,
        token_dict: Dict[str, int],
        singer_dict: Dict[str, int],
        language_dict: Dict[str, int],
        lyrics: list[list[str]],
        durations: list[list[float | list[float | list[float]]]],
        pitchs: list[list[int | list[int | list[int]]]],
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
        for index, (lyric, pitch, duration, singer, language) in enumerate(zip(lyrics, pitchs, durations, singers, languages)):
            if language == 'Korean':
                music = Korean(
                    lyric= lyric,
                    duration= duration,
                    pitch= pitch,
                    sample_rate= sample_rate,
                    hop_size= hop_size
                    )
            elif language == 'English':
                music = English(
                    lyric= lyric,
                    duration= duration,
                    pitch= pitch,
                    sample_rate= sample_rate,
                    hop_size= hop_size
                    )
            elif language == 'Chinese':
                music = Chinese(
                    lyric= lyric,
                    duration= duration,
                    pitch= pitch,
                    sample_rate= sample_rate,
                    hop_size= hop_size
                    )
            elif language == 'Japanese':
                music = Japanese(
                    lyric= lyric,
                    duration= duration,
                    pitch= pitch,
                    sample_rate= sample_rate,
                    hop_size= hop_size
                    )
            else:
                raise NotImplementedError(f'Unsupported language: {language}')

            self.patterns.append((
                music,
                singer,
                language
                ))

    def __getitem__(self, idx):
        music, singer, language = self.patterns[idx]

        duration, pronunciation, pitch, lyric = zip(*[
            (note.Duration, note.Lyric, note.Pitch, note.Text)
            for note in music
            ])
        token, token_on_note_length = zip(*[
            (Lyric_to_Token(x, self.token_dict), len(x))
            for x in pronunciation
            ])
        token = [x for token_on_note in token for x in token_on_note]
        singer = [self.singer_dict[singer]] * len(token)
        language = [self.language_dict[language]] * len(token)

        return token, token_on_note_length, pitch, duration, singer, language, lyric

    def __len__(self):
        return len(self.patterns)


class Collater:
    def __init__(
        self,
        token_dict: Dict[str, int],
        ):
        self.token_dict = token_dict

    def __call__(self, batch):
        tokens, token_on_note_lengths, \
        notes, durations, \
        singers, languages, \
        techs, mels, f0s = zip(*batch)
        
        token_lengths = np.array([len(token) for token in tokens])
        note_lengths = np.array([len(note) for note in notes])
        mel_lengths = np.array([mel.shape[1] for mel in mels])

        tokens = Token_Stack(tokens, self.token_dict)
        token_on_note_lengths = Token_on_Note_Length_Stack(token_on_note_lengths)
        notes = Note_Stack(notes)
        durations = Duration_Stack(durations)
        singers = Singer_Stack(singers)
        languages = Language_Stack(languages)
        techs = Tech_Stack(techs)
        mels = Mel_Stack(mels)
        f0s = F0_Stack(f0s)

        tokens = torch.IntTensor(tokens)   # [Batch, Token_t]
        token_lengths = torch.IntTensor(token_lengths)  # [Batch]
        token_on_note_lengths = torch.IntTensor(token_on_note_lengths) # [Batch, Token_t]

        notes = torch.IntTensor(notes) # [Batch, Note_t]
        durations = torch.IntTensor(durations) # [Batch, Note_t]
        note_lengths = torch.IntTensor(note_lengths)    # [Batch]

        singers = torch.IntTensor(singers)  # [Batch, Token_t]
        languages = torch.IntTensor(languages)  # [Batch, Token_t]

        techs = torch.IntTensor(techs)  # [Batch, N_Techs, Mel_t]
        mels = torch.FloatTensor(mels)  # [Batch, N_Mel, Mel_t]
        f0s = torch.FloatTensor(f0s)  # [Batch, N_Mel, Mel_t]
        mel_lengths = torch.IntTensor(mel_lengths)    # [Batch]
        
        return \
            tokens, token_lengths, token_on_note_lengths, \
            notes, durations, note_lengths, \
            singers, languages, \
            techs, mels, f0s, mel_lengths

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