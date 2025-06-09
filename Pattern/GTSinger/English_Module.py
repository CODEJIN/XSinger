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
    diphones = ['tʃ', 'hw', 'aʊ', 'kw', 'eɪ', 'oʊ', 'dʒ', 'ɔɪ', 'aɪ']
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

def Process(metadata: dict, sample_rate: int, hop_size: int) -> tuple[list[Note], np.ndarray]:
    # word base info
    text_list = metadata['txt']

    # phoneme base info
    phoneme_duration_list = metadata['ph_durs']
    phoneme_note_list = metadata['ep_pitches']
    mix_tech_list = metadata['mix_tech']
    falsetto_tech_list = metadata['falsetto_tech']
    breathy_tech_list = metadata['breathy_tech']
    pharyngeal_tech_list = metadata['pharyngeal_tech']
    vibrato_tech_list = metadata['vibrato_tech']
    glissando_tech_list = metadata['glissando_tech']

    # phoneme to word
    phoneme_to_word_list = metadata['ph2words']

    text_list_0 = text_list
    note_list_0 = [[] for _ in text_list]
    duration_list_0 = [[] for _ in text_list]
    mix_tech_list_0 = [[] for _ in text_list]
    falsetto_tech_list_0 = [[] for _ in text_list]
    breathy_tech_list_0 = [[] for _ in text_list]
    pharyngeal_tech_list_0 = [[] for _ in text_list]
    vibrato_tech_list_0 = [[] for _ in text_list]
    glissando_tech_list_0 = [[] for _ in text_list]
    for (
        phoneme_note, phoneme_duration, phoneme_to_word,
        mix_tech, falsetto_tech, breathy_tech,
        pharyngeal_tech, vibrato_tech, glissando_tech 
        ) in zip(
        phoneme_note_list, phoneme_duration_list, phoneme_to_word_list,
        mix_tech_list, falsetto_tech_list, breathy_tech_list,
        pharyngeal_tech_list, vibrato_tech_list, glissando_tech_list 
        ):
        note_list_0[phoneme_to_word].append(phoneme_note)
        duration_list_0[phoneme_to_word].append(phoneme_duration)
        mix_tech_list_0[phoneme_to_word].append(mix_tech)
        falsetto_tech_list_0[phoneme_to_word].append(falsetto_tech)
        breathy_tech_list_0[phoneme_to_word].append(breathy_tech)
        pharyngeal_tech_list_0[phoneme_to_word].append(pharyngeal_tech)
        vibrato_tech_list_0[phoneme_to_word].append(vibrato_tech)
        glissando_tech_list_0[phoneme_to_word].append(glissando_tech)

    ipa_list_0 = []
    for text in text_list_0:
        if text in ['<SP>', '<AP>']:
            ipa_list_0.append(['<X>'])
        else:
            ipa = _English_Syllablize(_English_Phonemize(texts= text)[0])
            ipa_list_0.append(ipa)
    
    text_list_1 = []
    ipa_list_1 = []
    note_list_1 = []
    duration_list_1 = []
    mix_tech_list_1 = []
    falsetto_tech_list_1 = []
    breathy_tech_list_1 = []
    pharyngeal_tech_list_1 = []
    vibrato_tech_list_1 = []
    glissando_tech_list_1 = []
    for (
        text, ipa, phoneme_note, phoneme_duration,
        mix_tech, falsetto_tech, breathy_tech,
        pharyngeal_tech, vibrato_tech, glissando_tech 
        ) in zip(
        text_list_0, ipa_list_0, note_list_0, duration_list_0,
        mix_tech_list_0, falsetto_tech_list_0,
        breathy_tech_list_0, pharyngeal_tech_list_0,
        vibrato_tech_list_0, glissando_tech_list_0
        ):
        num_syllable = len(ipa) if ipa != ['<X>'] else 1

        num_phonemes = [
            len(onsets) + 1 + len(coda)
            for onsets, nucleus, coda in ipa
            ] if ipa != ['<X>'] else [1]

        if num_syllable == 1 and len(phoneme_note) > 1:
            text_list_1.append('<X>' if text in ['<SP>', '<AP>'] else text)
            ipa_list_1.extend(ipa)
            note_list_1.append(phoneme_note)
            duration_list_1.append(phoneme_duration)
            mix_tech_list_1.append(mix_tech)
            falsetto_tech_list_1.append(falsetto_tech)
            breathy_tech_list_1.append(breathy_tech)
            pharyngeal_tech_list_1.append(pharyngeal_tech)
            vibrato_tech_list_1.append(vibrato_tech)
            glissando_tech_list_1.append(glissando_tech)
        # elif sum(num_phonemes) <= len(phoneme_note):
        else:
            for index in range(len(phoneme_note) - sum(num_phonemes)):
                num_phonemes[index % len(num_phonemes)] += 1
            for index in range(sum(num_phonemes) - len(phoneme_note)):
                num_phonemes[-(index % len(num_phonemes)) - 1] -= 1
            
            text_list_1.append('<X>' if text in ['<SP>', '<AP>'] else text)
            text_list_1.extend(['<SLUR>'] * (num_syllable - 1))
            ipa_list_1.extend(ipa)

            start_index = 0
            for num_phoneme in num_phonemes:
                note_list_1.append(phoneme_note[start_index:start_index + num_phoneme])
                duration_list_1.append(phoneme_duration[start_index:start_index + num_phoneme])
                mix_tech_list_1.append(mix_tech[start_index:start_index + num_phoneme])
                falsetto_tech_list_1.append(falsetto_tech[start_index:start_index + num_phoneme])
                breathy_tech_list_1.append(breathy_tech[start_index:start_index + num_phoneme])
                pharyngeal_tech_list_1.append(pharyngeal_tech[start_index:start_index + num_phoneme])
                vibrato_tech_list_1.append(vibrato_tech[start_index:start_index + num_phoneme])
                glissando_tech_list_1.append(glissando_tech[start_index:start_index + num_phoneme])
                start_index += num_phoneme

    text_list = []
    ipa_list = []
    note_list = []
    duration_list = []
    mix_tech_list = []
    falsetto_tech_list = []
    breathy_tech_list = []
    pharyngeal_tech_list = []
    vibrato_tech_list = []
    glissando_tech_list = []
    for (
        text, ipa, phoneme_note, phoneme_duration,
        mix_tech, falsetto_tech, breathy_tech,
        pharyngeal_tech, vibrato_tech, glissando_tech 
        ) in zip(
        text_list_1, ipa_list_1, note_list_1, duration_list_1,
        mix_tech_list_1, falsetto_tech_list_1,
        breathy_tech_list_1, pharyngeal_tech_list_1,
        vibrato_tech_list_1, glissando_tech_list_1
        ):
        current_text = []
        current_ipa = []
        current_note = []
        current_duration = []
        current_mix_tech = []
        current_falsetto_tech = []
        current_breathy_tech = []
        current_pharyngeal_tech = []
        current_vibrato_tech = []
        current_glissando_tech = []
        for note_index, note in enumerate(phoneme_note):
            if len(current_note) > 0 and current_note[-1] == note:
                current_duration[-1] += phoneme_duration[note_index]
                current_mix_tech[-1] += mix_tech[note_index]
                current_falsetto_tech[-1] += falsetto_tech[note_index]
                current_breathy_tech[-1] += breathy_tech[note_index]
                current_pharyngeal_tech[-1] += pharyngeal_tech[note_index]
                current_vibrato_tech[-1] += vibrato_tech[note_index]
                current_glissando_tech[-1] += glissando_tech[note_index]
            else:
                if len(current_note) == 0:
                    current_text.append(text)
                    current_ipa.append(ipa)
                elif current_note[-1] != note:
                    current_text.append('<SLUR>')
                    ipa = current_ipa.pop()
                    current_ipa.extend([(ipa[0], ipa[1], []), ([], ipa[1], ipa[2])])
                else:
                    assert False    # ?                        
                current_note.append(note)
                current_duration.append(phoneme_duration[note_index])
                current_mix_tech.append(mix_tech[note_index])
                current_falsetto_tech.append(falsetto_tech[note_index])
                current_breathy_tech.append(breathy_tech[note_index])
                current_pharyngeal_tech.append(pharyngeal_tech[note_index])
                current_vibrato_tech.append(vibrato_tech[note_index])
                current_glissando_tech.append(glissando_tech[note_index])

        text_list.extend(current_text)
        ipa_list.extend(current_ipa)
        note_list.extend(current_note)
        duration_list.extend(current_duration)
        mix_tech_list.extend(current_mix_tech)
        falsetto_tech_list.extend(current_falsetto_tech)
        breathy_tech_list.extend(current_breathy_tech)
        pharyngeal_tech_list.extend(current_pharyngeal_tech)
        vibrato_tech_list.extend(current_vibrato_tech)
        glissando_tech_list.extend(current_glissando_tech)

    duration_list = [round(x, 5) for x in duration_list]
    mix_tech_list = [max(0, min(1, x)) for x in mix_tech_list]
    falsetto_tech_list = [max(0, min(1, x)) for x in falsetto_tech_list]
    breathy_tech_list = [max(0, min(1, x)) for x in breathy_tech_list]
    pharyngeal_tech_list = [max(0, min(1, x)) for x in pharyngeal_tech_list]
    vibrato_tech_list = [max(0, min(1, x)) for x in vibrato_tech_list]
    glissando_tech_list = [max(0, min(1, x)) for x in glissando_tech_list]
    
    music = [
        Note(
            Duration= duration,
            Lyric= ipa,
            Pitch= note,
            Text= text,
            Tech= None
            )
        for duration, ipa, note, text in zip(
            duration_list, ipa_list, note_list, text_list
            )
        ]
    music = Convert_Duration_Second_to_Frame(
        music= music,
        sample_rate= sample_rate,
        hop_size= hop_size
        )
    music = Convert_Lyric_Syllable_to_Sequence(music)
    
    # GTSinger has the tech info.
    tech = np.array([
        mix_tech_list,
        falsetto_tech_list,
        breathy_tech_list,
        pharyngeal_tech_list,
        vibrato_tech_list,
        glissando_tech_list 
        ])
    
    return music, tech

