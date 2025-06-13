import os, mido, textgrid as tg
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
    'tɕʰ','tsʰ','tʂʰ','ʈʂʰ','tɕ','ts','tʂ','ʈʂ',
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
                ]).replace('ɹ̩', 'ɚ').replace('ɻ̩', 'ɚ').replace('ɔ', 'o').replace('ɛ', 'e').replace('ʊ', 'u')\
                .replace('˧', '').replace('˩', '').replace('˥', '')
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
    midi_path: str,
    tg_path: int,
    sample_rate: int,
    hop_size: int
    ) -> list[Note]:
    music = []
    note_states = {}
    last_notes = []
    mid = mido.MidiFile(midi_path)
    for message in list(mid):
        if message.type == 'note_on' and message.velocity != 0:
            if len(note_states) == 0:
                if message.time < 0.1:
                    music[-1][0] += message.time
                else:
                    if len(music) > 0 and music[-1][1] == 0:
                        music[-1][0] += message.time
                    else:
                        music.append([message.time, 0])
            else:
                note_states[last_notes[-1]]['Time'] += message.time
            note_states[message.note] = {
                'Time': 0.0
                }
            last_notes.append(message.note)
        elif message.type == 'note_off' or (message.type == 'note_on' and message.velocity == 0):
            note_states[message.note]['Time'] += message.time
            music.append([note_states[message.note]['Time'], message.note])
            del note_states[message.note]
            del last_notes[last_notes.index(message.note)]
        else:
            if len(note_states) == 0:
                if len(music) > 0 and music[-1][1] == 0:
                    music[-1][0] += message.time
                else:
                    music.append([message.time, 0])
            else:
                note_states[last_notes]['Time'] += message.time
    if len(note_states) > 0:
        print(f'{midi_path} is skipped.')
        return None
    music = [
        (round(duration, 5), note)
        for duration, note in music
        if duration > 0.0
        ]

    # all data [start_time, end_time, duration, ...]
    cumulated_music = []
    cumulated_time = 0
    for duration, note in music:
        # music: [duration, note]
        cumulated_music.append((cumulated_time, cumulated_time + duration, duration, note))
        cumulated_time += duration

    music = cumulated_music   # [start_time, end_time, duration, note]
    
    textgrids = []
    for textgrid_interval in tg.TextGrid.fromFile(tg_path)[0].intervals:
        chinese_character = textgrid_interval.mark
        if chinese_character in ['<AP>', '<SP>']:
            chinese_character = '<X>'
            
        ipa_phoneme = \
            [chinese_character] \
            if chinese_character == '<X>' else \
            _Chinese_Syllablize(_Chinese_Phonemize(chinese_character)[0])[0]

        textgrids.append((
            np.round(textgrid_interval.minTime, 3),
            np.round(textgrid_interval.maxTime, 3),
            np.round(textgrid_interval.duration(), 3),
            chinese_character,
            ipa_phoneme
            ))

    def Overlap_Ratio(
        a_range: Union[List[float], Tuple[float, float]],
        b_range: Union[List[float], Tuple[float, float]]
        ):
        '''
        A∩B / A∪B
        '''
        a_range = sorted(a_range)
        b_range = sorted(b_range)
        assert len(a_range) == 2 or len(b_range) == 2
        if a_range[1] < b_range[0] or a_range[0] > b_range[1]:
            return 0.0
        
        intersection = min(a_range[1], b_range[1]) - max(a_range[0], b_range[0])
        union = max(a_range[1], b_range[1]) - min(a_range[0], b_range[0])

        return intersection / union

    music_index = 0
    textgrid_index = 0
    music_textgrid_list = []            
    while music_index < len(music) and textgrid_index < len(textgrids):
        # music[0]: start_time
        # music[1]: end_time
        # music[2]: duration
        # music[3]: note

        # textgrid[0]: start_time
        # textgrid[1]: end_time
        # textgrid[2]: duration
        # textgrid[3]: chinese chracter
        # textgrid[4]: ipa phoneme
        current_music = music[music_index]
        current_textgrid = textgrids[textgrid_index]
        current_textgrid_and_current_music_overlap = Overlap_Ratio(
            (current_music[0], current_music[1]),
            (current_textgrid[0], current_textgrid[1]),
            )
        
        multi_textgrid_and_current_music_overlap = \
        current_textgrid_and_multi_music_overlap = \
            current_textgrid_and_current_music_overlap
        
        if music_index < len(music) - 1:
            current_textgrid_and_multi_music_overlap = Overlap_Ratio(
                (current_music[0], music[music_index + 1][1]),
                (current_textgrid[0], current_textgrid[1]),
                )
        if textgrid_index < len(textgrids) - 1:
            multi_textgrid_and_current_music_overlap = Overlap_Ratio(
                (current_music[0], current_music[1]),
                (current_textgrid[0], textgrids[textgrid_index + 1][1]),
                )

        if all([
            current_textgrid_and_current_music_overlap >= current_textgrid_and_multi_music_overlap,
            current_textgrid_and_current_music_overlap >= multi_textgrid_and_current_music_overlap,
            ]): # (1 textgrid, 1 music)
            music_textgrid_list.append(([current_music], [current_textgrid]))
            music_index += 1
            textgrid_index += 1
        elif current_textgrid_and_current_music_overlap < current_textgrid_and_multi_music_overlap: # (1 textgrid, n music)
            fixed_start_time = current_music[0]
            current_overlap = current_textgrid_and_current_music_overlap
            music_list = [current_music]
            while music_index + 1 < len(music):
                next_music = music[music_index + 1]
                next_overlap = Overlap_Ratio(
                    (fixed_start_time, next_music[1]),
                    (current_textgrid[0], current_textgrid[1])
                    )
                if next_overlap > current_overlap:
                    music_list.append(next_music)
                    current_music = next_music
                    current_overlap = next_overlap
                    music_index += 1
                else:
                    break

            textgrid_list = [current_textgrid]
            # last fix check, this managed very small silence
            if textgrid_index < len(textgrids) - 1:
                next_textgrid = textgrids[textgrid_index + 1]
                if next_textgrid[3] == '<X>':
                    fixed_overlap = Overlap_Ratio(
                        (fixed_start_time, current_music[1]),
                        (current_textgrid[0], next_textgrid[1])
                        )
                    if fixed_overlap > current_overlap:
                        textgrid_list.append(next_textgrid)
                        textgrid_index += 1

            music_textgrid_list.append((music_list, textgrid_list))
            music_index += 1
            textgrid_index += 1
        elif current_textgrid_and_current_music_overlap < multi_textgrid_and_current_music_overlap: # (n textgrid, 1 music)
            fixed_start_time = current_textgrid[0]
            current_overlap = current_textgrid_and_current_music_overlap
            textgrid_list = [current_textgrid]
            while textgrid_index + 1 < len(textgrids):
                next_textgrid = textgrids[textgrid_index + 1]
                next_overlap = Overlap_Ratio(
                    (current_music[0], current_music[1]),
                    (fixed_start_time, next_textgrid[1])
                    )
                if next_overlap > current_overlap:
                    textgrid_list.append(next_textgrid)
                    current_textgrid = next_textgrid
                    current_overlap = next_overlap
                    textgrid_index += 1
                else:
                    break


            music_list = [current_music]
            if music_index < len(music) - 1:
                # last fix check, this managed very small silence
                next_music = music[music_index + 1]
                if next_music[3] == 0:
                    fixed_overlap = Overlap_Ratio(
                        (current_music[0], next_music[1]),
                        (fixed_start_time, current_textgrid[1])
                        )
                    if fixed_overlap > current_overlap:
                        music_list.append(next_music)
                        music_index += 1

            music_textgrid_list.append((music_list, textgrid_list))
            music_index += 1
            textgrid_index += 1
        else:
            print(f'sync complexity: {midi_path}')
            return None
        
    if music_index < len(music):
        assert music[music_index][3] == 0, 'Not processed music info.'
        music_textgrid_list.append((music[music_index:], []))
    if textgrid_index < len(textgrids): # some midi data does not have last silence info.
        assert textgrids[textgrid_index][3] == '<X>', 'Not processed textgrid info.'
        music_textgrid_list.append(([], textgrids[textgrid_index:]))

    music = []
    for music_list, textgrid_list in music_textgrid_list:
        if len(music_list) > 1 and len(textgrid_list) > 1:    # special case, this is independent if
            clean_music_list = []
            for start_time, end_time, duration, note in music_list:
                if len(clean_music_list) > 0 and note == 0:
                    clean_music_list[-1] = (
                        clean_music_list[-1][0],
                        end_time,
                        clean_music_list[-1][2] + duration,
                        clean_music_list[-1][3],
                        )
                    continue
                clean_music_list.append((start_time, end_time, duration, note))
            music_list = clean_music_list


            clean_textgrid_list = []
            for start_time, end_time, duration, chinese_character, ipa_phoneme in textgrid_list:
                if len(clean_textgrid_list) > 0 and chinese_character == '<X>':
                    clean_textgrid_list[-1] = (
                        clean_textgrid_list[-1][0],
                        end_time,
                        clean_textgrid_list[-1][2] + duration,
                        clean_textgrid_list[-1][3],
                        clean_textgrid_list[-1][4],
                        )
                    continue
                clean_textgrid_list.append((start_time, end_time, duration, chinese_character, ipa_phoneme))
            textgrid_list = clean_textgrid_list

            # if both of music_list and textgrid_list are multiple after cleaning, it is critical.
            assert not (len(music_list) > 1 and len(textgrid_list) > 1), f'sync complexity: {midi_path}'

        if len(music_list) == 0 and len(textgrid_list) > 0: # special case
            for textgrid in textgrid_list:
                _, _, textgrid_duration, chinese_character, ipa_phoneme = textgrid
                if chinese_character == '<X>':
                    music.append((textgrid_duration, ipa_phoneme, 0, chinese_character))                
        elif len(music_list) == 1 and len(textgrid_list) == 1:
            _, _, music_duration, note = music_list[0]
            _, _, _, chinese_character, ipa_phoneme = textgrid_list[0]
            music.append((music_duration, ipa_phoneme, note, chinese_character))
        elif len(music_list) == 1 and len(textgrid_list) > 1:
            _, _, music_duration, note = music_list[0]
            for textgrid in textgrid_list:
                _, _, textgrid_duration, chinese_character, ipa_phoneme = textgrid
                if chinese_character == '<X>' and len(music) > 0:
                    music[-1] = ((music[-1][0] + textgrid_duration, music[-1][1], music[-1][2], music[-1][3]))
                else:
                    music.append((textgrid_duration, ipa_phoneme, note, chinese_character))
        elif len(music_list) > 1 and len(textgrid_list) == 1:
            _, _, _, chinese_character, (onset, nucleus, coda) = textgrid_list[0]
            
            if all([x[3] == 0 for x in music_list]) or all([x[3] != 0 for x in music_list]):    # everything is 0 or nothing is 0.
                pass
            else:
                # 일반 음표에 note 0이 섞인 케이스
                fixed_music_list = []
                for x in music_list:
                    if x[3] != 0:
                        fixed_music_list.append(x)
                    else:
                        fixed_music_list[-1] = (
                            fixed_music_list[-1][0],
                            fixed_music_list[-1][1] + x[1],
                            fixed_music_list[-1][2] + x[2],
                            fixed_music_list[-1][3],
                            )
                music_list = fixed_music_list
            
            for index, (_, _, music_duration, note) in enumerate(music_list):
                if index == 0: ipa_phoneme = (onset, nucleus, [])
                elif index == len(music) - 1: ipa_phoneme = ([], nucleus, coda)
                else: ipa_phoneme = ([], nucleus, [])   # middle
                music.append((music_duration, ipa_phoneme, note, chinese_character))

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
    music = [x for x in music if x.Duration > 0 or x.Text != '<X>']

    if any([x.Duration == 0 for x in music]):
        print(f'\'{midi_path}\' has an garbege note. Generator remove the note temporarily.')
        for x in music:
            print(('' if x.Duration > 0 else '***') + str(x))
        music = [x for x in music if x.Duration > 0]

    return music
