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
    duration: list[float],
    lyric: list[str],
    pitch: list[int],
    sample_rate: int,
    hop_size: int
    ) -> list[Note]:
    for chinese_character in lyric:
        print(chinese_character)
        print(_Chinese_Phonemize(chinese_character)[0])
    
    ipa_phoneme = [
        _Chinese_Syllablize(_Chinese_Phonemize(chinese_character)[0])[0]
        for chinese_character in lyric
        ]
    music = [x for x in zip(duration, ipa_phoneme, pitch, lyric)]

    music = [
        Note(
            Duration= duration,
            Lyric= lyric,
            Pitch= note,
            Text= text,
            Tech= None
            )
        for duration, lyric, note, text in music
        ]
    music = Convert_Duration_Second_to_Frame(
        music= music,
        sample_rate= sample_rate,
        hop_size= hop_size
        )
    music = Convert_Lyric_Syllable_to_Sequence(music)
    
    return music