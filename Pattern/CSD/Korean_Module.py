import os
from argparse import Namespace
import numpy as np
import pandas as pd
from typing import List, Dict, Union, Optional, Tuple
from ko_pron import romanise

from .. import Note, Convert_Duration_Second_to_Frame, Convert_Lyric_Syllable_to_Sequence

# Korean
def _Korean_Phonemize(texts: Union[str, List[str]]):
    if type(texts) == str:
        texts = [texts]

    pronunciations = [
        romanise(text, system_index= 'ipa')
        for text in texts
        ]
    
    pronunciations = [
        pronunciation if not '] ~ [' in pronunciation else pronunciation.split('] ~ [')[0]
        for pronunciation in pronunciations
        ]
    
    pronunciations = [
        pronunciation.replace('x', '')
        for pronunciation in pronunciations
        ]

    return pronunciations

korean_vowels = sorted(
    list({'u', 'o̞', 'we̞', 'jo', 'i', 'ɯ', 'ø̞', 'wʌ̹', 'jʌ̹', 'ja̠', 'je̞', 'ɥi', 'ju', 'wa̠', 'ʌ̹', 'ɰi', 'e̞', 'a̠'}),
    key= lambda x: len(x),
    reverse= True
    )
korean_boundary_dict = {
    't͡ɕ͈': ('', 't͡ɕ͈'), 'tʰ': ('', 'tʰ'), 'ç': ('', 'ç'), 't͈': ('', 't͈'), 's͈': ('', 's͈'), 'ɸʷ': ('', 'ɸʷ'), 
    't': ('', 't'), 'ɕʰ': ('', 'ɕʰ'), 'ɸ': ('', 'ɸ'), 'm': ('', 'm'), 'sʰ': ('', 'sʰ'), 'kʰ': ('', 'kʰ'), 
    'ʃʰ': ('', 'ʃʰ'), 'p͈': ('', 'p͈'), 'ɾ': ('', 'ɾ'), 'kç': ('', 'kç'), 'ɲ': ('', 'ɲ'), 't͡ɕʰ': ('', 't͡ɕʰ'), 
    'p': ('', 'p'), 't͡ɕ': ('', 't͡ɕ'), 'n': ('', 'n'), 'pʰ': ('', 'pʰ'), 'k͈': ('', 'k͈'), 'ɕ͈': ('', 'ɕ͈'), 
    'h': ('', 'h'), 'k': ('', 'k'), 'ɦ': ('', 'ɦ'), 'ɡ': ('', 'ɡ'), 'd': ('', 'd'), 'd͡ʑ': ('', 'd͡ʑ'),
    'ʎ': ('', 'ʎ'), 'b': ('', 'b'), 'ɣ': ('', 'ɣ'), 'β': ('', 'β'), 'ʝ': ('', 'ʝ'),
    'ɭ': ('ɭ', ''), 'k̚': ('k', ''), 'ŋ': ('ŋ', ''), 'p̚': ('p', ''), 't̚': ('t', ''),
    }

def _Korean_Split_Vowel(pronunciation: str):
    if len(pronunciation) == 0:
        return []

    for vowel in korean_vowels:
        if vowel in pronunciation:
            vowel_index = pronunciation.index(vowel)
            return \
                _Korean_Split_Vowel(pronunciation[:vowel_index]) + \
                [vowel] + \
                _Korean_Split_Vowel(pronunciation[vowel_index + len(vowel):])
    
    return [pronunciation]

def _Korean_Syllablize(pronunciation: str):
    pronunciation = _Korean_Split_Vowel(
        pronunciation= pronunciation
        ) # type: ignore
    
    syllables = []
    current_consonants = []
    if any(
        x in pronunciation and pronunciation[-1] != x
        for x in ['ɭ', 'k̚', 'p̚', 't̚']
        ):
        print(pronunciation)
        assert False

    for phoneme in pronunciation:
        if phoneme in korean_vowels:
            if \
                len(syllables) > 0 and \
                (   
                    len(current_consonants) > 1 or
                    (len(current_consonants) > 0 and current_consonants[0] in ['ɭ', 'ŋ'])
                    ):
                syllables[-1][2].append(current_consonants[0])
                current_consonants = current_consonants[1:]
            syllables.append((current_consonants, phoneme, []))
            current_consonants = []
        else:   # consonant
            if phoneme in korean_boundary_dict.keys():
                current_consonants.extend([x for x in korean_boundary_dict[phoneme] if x != ''])
            else:
                is_applied = False
                for index in range(len(phoneme)):
                    if phoneme[:index] in korean_boundary_dict.keys() and phoneme[index:] in korean_boundary_dict.keys():
                        current_consonants.extend([x for x in korean_boundary_dict[phoneme[:index]] if x != ''])
                        current_consonants.extend([x for x in korean_boundary_dict[phoneme[index:]] if x != ''])
                        is_applied = True
                        break
                assert is_applied, (pronunciation, phoneme)

    if len(current_consonants) > 0:
        syllables[-1][2].extend(current_consonants)

    return syllables

def Korean_G2P_Lyric(music):
    grapheme_string_list = []
    current_grapheme_string = []
    for _, lyric, _ in music:
        if lyric == '<X>':
            grapheme_string_list.append(''.join(current_grapheme_string))
            current_grapheme_string = []
        else:
            current_grapheme_string.append(lyric)
    grapheme_string_list.append(''.join(current_grapheme_string))

    pronunciations = _Korean_Phonemize(
        texts= grapheme_string_list
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
        music[index] = (
            music[index][0],
            ['<X>'] if lyric == '<X>' else lyric,
            music[index][2],
            music[index][1]
            )

    return music


def Process(
    midi_path: str,
    lyric_path: str,
    sample_rate: int,
    hop_size: int
    ) -> list[Note]:
    mid = pd.read_csv(midi_path)
    lyric = ''.join(open(lyric_path, encoding= 'utf-8-sig').readlines()).replace('\n', '').replace(' ', '')
    music = []
    for x, syllable in zip(mid.iloc, lyric):
        if len(music) == 0 and x.start > 0.0:   # initial silence
            music.append([0.0, x.start, '<X>', 0])
        elif len(music) > 0 and music[-1][1] < x.start:
            music.append([music[-1][1], x.start, '<X>', 0])
        elif len(music) > 0 and music[-1][1] > x.start:
            x.start = music[-1][1]

        if x.end <= x.start:
            print(x)
            assert False

        music.append([
            x.start,
            x.end,
            syllable,
            x.pitch
            ])
    music = [
        (round(end - start, 5), syllable, note)
        for start, end, syllable, note in music
        ]
    
    music = Korean_G2P_Lyric(music)
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