import numpy as np
from typing import List, Dict, Union, Optional, Tuple

import re, logging
from phonemizer import logger as phonemizer_logger
from phonemizer.backend import BACKENDS
from phonemizer.punctuation import Punctuation
from phonemizer.separator import default_separator, Separator
from phonemizer.phonemize import _phonemize

from .. import Note, Convert_Duration_Second_to_Frame, Convert_Lyric_Syllable_to_Sequence

logger = phonemizer_logger.get_logger()
logger.setLevel(logging.CRITICAL)

# English
english_phonemizer = BACKENDS['espeak'](
    language= 'en-us',
    punctuation_marks= Punctuation.default_marks(),
    preserve_punctuation= True,
    with_stress= False,
    tie= True,
    language_switch= 'remove-flags',
    words_mismatch= 'ignore',
    logger= logger
    )
whitespace_re = re.compile(r'\s+')
english_abbreviations = [(re.compile('\\b%s\\.' % x[0], re.IGNORECASE), x[1]) for x in [
    ('mrs', 'misess'),
    ('mr', 'mister'),
    ('dr', 'doctor'),
    ('st', 'saint'),
    ('co', 'company'),
    ('jr', 'junior'),
    ('maj', 'major'),
    ('gen', 'general'),
    ('drs', 'doctors'),
    ('rev', 'reverend'),
    ('lt', 'lieutenant'),
    ('hon', 'honorable'),
    ('sgt', 'sergeant'),
    ('capt', 'captain'),
    ('esq', 'esquire'),
    ('ltd', 'limited'),
    ('col', 'colonel'),
    ('ft', 'fort'),
    ]]
def expand_abbreviations(text: str):
    for regex, replacement in english_abbreviations:
        text = re.sub(regex, replacement, text)

    return text

def _English_Phonemize(texts: Union[str, List[str]], chunk: int= 5000):
    tie_bar = '\u0361'
    
    if type(texts) == str:
        texts = [texts]

    texts = [text.lower() for text in texts]
    texts = [expand_abbreviations(text) for text in texts]

    pronunciations = []

    indices = range(0, len(texts), chunk)
    for index in indices:
        pronunciations_chunk = _phonemize(
            backend= english_phonemizer,
            text= texts[index:index + chunk],
            separator= Separator(phone= '|'), # default_separator,
            strip= True,
            njobs= 4,
            prepend_text= False,
            preserve_empty_lines= False
            )
        pronunciations.extend([
            re.sub(whitespace_re, ' ', pronunciation).replace('ː', '').replace(tie_bar, '')
            for pronunciation in pronunciations_chunk
            ])

    return pronunciations

def _English_Syllablize(
    pronunciation: str
    ) -> List[Tuple[List[str], str, List[str]]]:
    # /ʔ/ is not vowel.
    # However, in English, the next vowel of /ʔ/ is usually silence /ə/.
    # To rule make, I manage /ʔ/ like a vowel
    diphones = ['tʃ', 'hw', 'aʊ', 'eɪ', 'oʊ', 'dʒ', 'ɔɪ', 'aɪ']
    use_vowels = [
        'aʊ', 'eɪ', 'oʊ', 'ɔɪ', 'aɪ', 'a', 'e', 'i', 'o', 'u', 
        'æ', 'ɐ', 'ɑ', 'ɔ', 'ə', 'ɛ', 'ɜ', 'ɪ', 'ʊ', 'ʌ', 'ᵻ', 'ɚ', 'ʔ', 
        ]

    syllable_list: List[Tuple[List[str], str, List[str]]] = []

    onsets = []
    nucleus = None
    index = 0
    while index < len(pronunciation):
        is_diphone = False
        if pronunciation[index:index + 2] in diphones:
            phoneme = pronunciation[index:index + 2]
            is_diphone = True
        else:
            phoneme = pronunciation[index]
        
        if not phoneme in use_vowels:
            onsets.append(phoneme)
        else:   # vowel
            nucleus = phoneme
            syllable_list.append((onsets, nucleus, []))
            onsets = []
            nucleus = None
        index += 1 + is_diphone

    syllable_list[-1] = (syllable_list[-1][0], syllable_list[-1][1], onsets)

    index = 0
    while index < len(syllable_list) - 1:
        next_onset = syllable_list[index + 1][0]
        if len(next_onset) > 1:
            syllable_list[index] = (
                syllable_list[index][0],
                syllable_list[index][1],
                next_onset[:1]
                )
            syllable_list[index + 1] = (
                next_onset[1:],
                syllable_list[index + 1][1],
                syllable_list[index + 1][2],
                )
        index += 1

    return syllable_list

def Process(
    lyric: list[str],
    duration: list[float | list[float | list[float]]],
    pitch: list[int | list[int | list[int]]],
    sample_rate: int,
    hop_size: int
    ) -> list[Note]:
    '''
    duration and pitch note.
    single syllable - float: (syllable, duration, pitch)
    single syllable - list[float]: (syllable, duration[0], pitch[0]), ... (<SLUR>, duration[n], pitch[n])
    multi syllable - float: (syllable[0], duration / n_syllable, pitch)

    multi syllable - list[float]: (syllable[0], duration[0], pitch[0]), ... (syllable[n], duration[n], pitch[n])

    multi syllable - list[list[float]] (ex. glo--------------------ria):
        (syllable[0], duration[0][0], pitch[0][0]), (syllable[0], duration[0][1], pitch[0][1]), ...,
        (syllable[n], duration[n][0], pitch[n][0]), (syllable[n], duration[n][1], pitch[n][1]), ...
    '''
    text_align = []
    lyric_realign = []
    duration_realign = []
    pitch_realign = []
    for index, (lyric_note, duration_note, pitch_note) in enumerate(zip(lyric, duration, pitch)):
        if lyric_note  == '<X>':
            text_align.append(lyric_note)
            lyric_realign.append([lyric_note])
            duration_realign.append(duration_note)
            pitch_realign.append(pitch_note)
        else:
            ipa = _English_Syllablize(_English_Phonemize(texts= lyric_note)[0])
            if type(duration_note) == float and type(pitch_note) == int:    # single syllable - float, multi syllable - float
                text_align.extend([lyric_note] + ['<SLUR>'] * (len(ipa) - 1))
                lyric_realign.extend(ipa)
                duration_realign.extend([duration_note / len(ipa)] * len(ipa))
                pitch_realign.extend([pitch_note] * len(ipa))
            elif type(duration_note) == list and type(pitch_note) == list and len(ipa) == 1:  # single syllable - list[float].
                ipa = ipa[0]
                first_ipa = (ipa[0], ipa[1], [])
                middle_ipa = ([], ipa[1], [])
                last_ipa = ([], ipa[1], ipa[2])                
                text_align.extend([lyric_note] + ['<SLUR>'] * (len(duration_note) - 1))
                lyric_realign.extend([first_ipa] + [middle_ipa] * (len(duration_note) - 2) + [last_ipa])
                duration_realign.extend(duration_note)
                pitch_realign.extend(pitch_note)
            elif type(duration_note) == list and type(pitch_note) == list and len(ipa) == len(duration_note):
                text_length = 0
                for ipa_note, duration_slur_note, pitch_slur_note in zip(ipa, duration_note, pitch_note):
                    if type(duration_slur_note) == float:  # multi syllable - list[float]
                        text_length += 1
                        lyric_realign.append(ipa_note)
                        duration_realign.append(duration_slur_note)
                        pitch_realign.append(pitch_slur_note)
                    elif type(duration_slur_note) == list and len(duration_slur_note) == 1:  # multi syllable - list[float]
                        text_length += 1
                        lyric_realign.append(ipa_note)
                        duration_realign.append(duration_slur_note[0])
                        pitch_realign.append(pitch_slur_note[0])
                    elif type(duration_slur_note) == list:   # multi syllable - list[list[float]]
                        text_length += len(duration_slur_note)
                        first_ipa_note = (ipa_note[0], ipa_note[1], [])
                        middle_ipa_note = ([], ipa_note[1], [])
                        last_ipa_note = ([], ipa_note[1], ipa_note[2])
                        lyric_realign.extend([first_ipa_note] + [middle_ipa_note] * (len(duration_slur_note) - 2) + [last_ipa_note])
                        duration_realign.extend(duration_slur_note)
                        pitch_realign.extend(pitch_slur_note)
                text_align.extend([lyric_note] + ['<SLUR>'] * (text_length - 1))
            else:
                raise ValueError(f'Inconsist among ipa, duration and pitch index {index}: {ipa}, {duration_note}, {pitch_note}')
    
    music = [
        Note(
            Duration= duration,
            Lyric= ipa,
            Pitch= note,
            Text= text,
            Tech= None
            )
        for duration, ipa, note, text in zip(
            duration_realign, lyric_realign, pitch_realign, text_align
            )
        ]
    music = Convert_Duration_Second_to_Frame(
        music= music,
        sample_rate= sample_rate,
        hop_size= hop_size
        )

    music = Convert_Lyric_Syllable_to_Sequence(music)
    
    return music

