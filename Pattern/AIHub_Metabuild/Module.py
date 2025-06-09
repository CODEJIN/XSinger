import os, json, hgtk
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

nucleus_long_sound_convert_dict = {'ㅐ': 'ㅔ', 'ㅒ': 'ㅔ', 'ㅖ': 'ㅔ', 'ㅚ': 'ㅞ', 'ㅙ': 'ㅞ', 'ㅢ': 'ㅣ'}
def Process(
    json_path: str,
    sample_rate: int,
    hop_size: int
    ) -> list[Note]:
    silence_criterion = hop_size * 4 / sample_rate

    music_json = json.load(open(json_path, 'r', encoding= 'utf-8-sig'))
    music = []
    for x in music_json['notes']:
        start, end, lyric, note = float(x['start_time']), float(x['end_time']), x['lyric'], int(x['midi_num'])
        if len(music) == 0 and start > 0.0:
            music.append([0.0, start, '<X>', 0])
        elif len(music) > 0 and start > music[-1][1]:
            if start - music[-1][1] >= silence_criterion:
                music.append([music[-1][1], start, '<X>', 0])
            elif start - music[-1][1] < silence_criterion:
                music[-1][1] = start

        if lyric == '' and note == 96:
            lyric, note = '<X>', 0
        music.append([start, end, lyric, note])

    for music_index in range(len(music)):
        if music[music_index][2] == '' and music[music_index][3] != 0:
            for previous_music_index in reversed(range(music_index)):
                if music[previous_music_index][2] != '<X>':
                    break
            for next_music_index in range(music_index + 1, len(music)):
                if music[next_music_index][2] != '<X>':
                    break

            before_lyric = ''.join([
                x[2] if x[2] != '' else ' '
                for x in music[previous_music_index:next_music_index + 1]
                ])
            
            previous_onset, previous_nucleus, previous_coda = hgtk.letter.decompose(music[previous_music_index][2])
            next_onset, next_nucleus, next_coda = hgtk.letter.decompose(music[next_music_index][2])

            change_previous_coda = False
            if previous_coda == '': # 단순 장음
                onset = 'ㅇ'
                nucleus = nucleus_long_sound_convert_dict.get(previous_nucleus, previous_nucleus)
                coda = ''
            elif previous_coda != '' and next_onset == 'ㅇ' and previous_nucleus == next_nucleus: # 장음의 중간일 가능성이 높음, previous coda가 이쪽의 onset으로 옮겨져야 함
                onset = previous_coda
                nucleus = nucleus_long_sound_convert_dict.get(previous_nucleus, previous_nucleus)
                coda = ''
                change_previous_coda = True
            elif previous_coda != '' and next_onset == 'ㅇ' and previous_nucleus != next_nucleus: # 장음의 끝일 가능성이 높음, previous coda가 이쪽의 coda로 옮겨져야 함
                onset = 'ㅇ'
                nucleus = nucleus_long_sound_convert_dict.get(previous_nucleus, previous_nucleus)
                coda = previous_coda
                change_previous_coda = True
            elif previous_coda != '' and next_onset != 'ㅇ': # 세 음절 중 소실된 가운데 음절일 가능성이 높음, previous coda가 이쪽으로 옮겨져야 함
                onset = previous_coda
                nucleus = 'ㅡ'
                coda = ''
                change_previous_coda = True

            if change_previous_coda:
                music[previous_music_index][2] = hgtk.letter.compose(previous_onset, previous_nucleus, '')
            music[music_index][2] = hgtk.letter.compose(onset, nucleus, coda)

            after_lyric = ''.join([
                x[2] if x[2] != '' else ' '
                for x in music[previous_music_index:next_music_index + 1]
                ])
            print(f'{json_path}, {before_lyric} -> {after_lyric}')

    music_fix = []
    for music_index in range(len(music)):
        if len(music_fix) == 0 or music_fix[-1][2] != music[music_index][2] or music_fix[-1][3] != music[music_index][3]:
            music_fix.append(music[music_index])
        else:   # music_fix[-1][2] == music[music_index][2] and music_fix[-1][3] == music[music_index][3]:
            music_fix[-1][1] = music[music_index][1]
    music = music_fix

    music = [
        [end - start, syllable, note]
        for start, end, syllable, note in music
        ]
    
    music = Korean_G2P_Lyric(music)
    if music is None:
        return None
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