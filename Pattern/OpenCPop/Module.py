import os, librosa
from argparse import Namespace
import numpy as np
import pandas as pd
from typing import List, Dict, Union, Optional, Tuple
from pypinyin import pinyin
from pinyin_to_ipa import pinyin_to_ipa

from .. import Note, Convert_Duration_Second_to_Frame, Convert_Lyric_Syllable_to_Sequence

chinese_diphthongs = ['ai̯','au̯','ou̯','ei̯']
chinese_monophthongs = ['ɛ','a','ʊ','i','ə','ɤ','o','ɔ','u','y','e']
chinese_syllablc_consonants = ['ɻ̩', 'ɹ̩', 'ɚ', 'n']  # 'for 嗯(n)'
chinese_onsets = sorted([
    't͡ɕʰ','t͡sʰ','t͡ʂʰ','ʈ͡ʂʰ','t͡ɕ','t͡s','t͡ʂ','ʈ͡ʂ',
    'kʰ','pʰ','tʰ','ʐ̩','ɕ','f','j','k','l','m','n',
    'p','ʐ','s','ʂ','t','w','x','ɥ','h','ʂ','ʐ','ɻ'
    ], key= lambda x: len(x), reverse= True)
chinese_codas = ['n','ŋ']

def _Chinese_Phonemize(
    texts: Union[str, List[str]],
    syllable_splitter: str= ' '
    ):
    if type(texts) == str:
        texts = [texts]

    return [
        f'{syllable_splitter}'.join([
            ''.join([
                x for x in pinyin_to_ipa(pinyin(character, v_to_u= True)[0][0])[0]
                ]).
                replace('˧', '').replace('˩', '').replace('˥', '').
                replace('tɕ', 't͡ɕ').replace('tɕʰ', 't͡ɕʰ').replace('ts', 't͡s').
                replace('tsʰ', 't͡sʰ').replace('tʂ', 't͡ʂ').replace('tʂʰ', 't͡ʂʰ').
                replace('ʈʂ', 'ʈ͡ʂ').replace('ʈʂʰ', 'ʈ͡ʂʰ')
            for character in text
            ])
        for text in texts        
        ]

def _Chinese_Syllablize(pronunciation: str):
    pronunciation = [
        _Chinese_Split_Phoneme(syllable= x)
        for x in pronunciation.split(' ')
        ] # type: ignore

    syllables = []
    for syllable in pronunciation:
        is_exists_vowel = False
        for index, phoneme in enumerate(syllable):
            if phoneme in chinese_diphthongs + chinese_monophthongs + chinese_syllablc_consonants:
                syllables.append((syllable[:index], phoneme, syllable[index + 1:]))
                is_exists_vowel = True
                break
        assert is_exists_vowel, (pronunciation, syllable)    # no vowel.

    return syllables

def _Chinese_Split_Phoneme(syllable: str):
    if len(syllable) == 0:
        return []

    for phoneme in chinese_diphthongs + chinese_monophthongs + chinese_syllablc_consonants + chinese_onsets + chinese_codas:
        if phoneme in syllable:
            phoneme_index = syllable.index(phoneme)
            return \
                _Chinese_Split_Phoneme(syllable[:phoneme_index]) + \
                [phoneme] + \
                _Chinese_Split_Phoneme(syllable[phoneme_index + len(phoneme):])

    return [syllable]

def Process(
    text: str,
    phoneme: str,
    note: str,
    duration: str,
    slur: str,
    music_label: str,   # for special case.
    sample_rate: int,
    hop_size: int
    ) -> list[Note]:
    phoneme = [x for x in phoneme.strip().split(' ')]
    note = [
        librosa.note_to_midi(x.split('/')[0]) if x != 'rest' else 0
        for x in note.strip().split(' ')
        ]
    duration = [float(x) for x in duration.strip().split(' ')]
    slur = [int(x) for x in slur.strip().split(' ')]
    ipa_phoneme = [
        _Chinese_Syllablize(_Chinese_Phonemize(chinese_character)[0])[0]
        for chinese_character in text
        ]
    
    # problem cases.
    if music_label == '2034001289':
        slur[10] = 1
    elif music_label == '2046001761':
        note[35] = 0
    elif music_label == '2054002025':
        note[13] = 0
    elif music_label == '2092003414':
        note[15] = 0
    elif music_label == '2099003690':
        slur[7] = 1

    text_index = 0
    note_index = 0
    text_realign = []
    ipa_realign = []
    note_realign = []
    duration_realign = []
    while note_index < len(note):
        if note[note_index] == 0:
            text_realign.append('<X>')
            ipa_realign.append(['<X>'])
            note_realign.append(note[note_index])
            duration_realign.append(duration[note_index])
            note_index += 1
        elif slur[note_index] == 1:
            slur_phoneme = ([], ipa_realign[-1][1], ipa_realign[-1][2])
            ipa_realign[-1] = (ipa_realign[-1][0], ipa_realign[-1][1], [])
            
            text_realign.append(text_realign[-1])
            ipa_realign.append(slur_phoneme)
            note_realign.append(note[note_index])
            duration_realign.append(duration[note_index])
            note_index += 1
        elif phoneme[note_index][0] in ['a', 'i', 'u', 'e', 'o', 'v']:   # v is ü
            text_realign.append(text[text_index])
            ipa_realign.append(ipa_phoneme[text_index])
            note_realign.append(note[note_index])
            duration_realign.append(duration[note_index])
            text_index += 1
            note_index += 1
        elif note[note_index] == note[note_index + 1]:
            text_realign.append(text[text_index])
            ipa_realign.append(ipa_phoneme[text_index])
            note_realign.append(note[note_index])
            duration_realign.append(duration[note_index])
            text_index += 1
            note_index += 2
        else:
            raise ValueError(f'note process error: {note[note_index]} != {note[note_index + 1]}')

    music = [
        Note(
            Duration= message_time,
            Lyric= lyric,
            Pitch= note,
            Text= text,
            Tech= None
            )
        for message_time, lyric, note, text in zip(
            duration_realign, ipa_realign, note_realign, text_realign
            )
        ]

    music = Convert_Duration_Second_to_Frame(
        music= music,
        sample_rate= sample_rate,
        hop_size= hop_size
        )
    music = Convert_Lyric_Syllable_to_Sequence(music)
    
    return music
