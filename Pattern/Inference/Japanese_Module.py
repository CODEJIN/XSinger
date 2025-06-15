import numpy as np
from typing import List, Dict, Union, Optional, Tuple
import music21, librosa

from .. import Note, Convert_Duration_Second_to_Frame, Convert_Lyric_Syllable_to_Sequence

def dakuten_fix(syllable):
    dakuten_map = {
        'う': 'ゔ',
        'か': 'が', 'き': 'ぎ', 'く': 'ぐ', 'け': 'げ', 'こ': 'ご',
        'さ': 'ざ', 'し': 'じ', 'す': 'ず', 'せ': 'ぜ', 'そ': 'ぞ',
        'た': 'だ', 'ち': 'ぢ', 'つ': 'づ', 'て': 'で', 'と': 'ど',
        'は': 'ば', 'ひ': 'び', 'ふ': 'ぶ', 'へ': 'べ', 'ほ': 'ぼ',        
        }
    
    return dakuten_map[syllable]

def get_duration_seconds(quater_length, current_tempo):
    return quater_length * (60 / current_tempo.number)

def remove_initial_consonant(syllable):
    # 히라가나 모음 매핑
    vowel_map = {
        'あ': 'あ', 'い': 'い', 'う': 'う', 'え': 'え', 'お': 'お',
        'か': 'あ', 'き': 'い', 'く': 'う', 'け': 'え', 'こ': 'お',
        'さ': 'あ', 'し': 'い', 'す': 'う', 'せ': 'え', 'そ': 'お',
        'た': 'あ', 'ち': 'い', 'つ': 'う', 'て': 'え', 'と': 'お',
        'な': 'あ', 'に': 'い', 'ぬ': 'う', 'ね': 'え', 'の': 'お',
        'は': 'あ', 'ひ': 'い', 'ふ': 'う', 'へ': 'え', 'ほ': 'お',
        'ま': 'あ', 'み': 'い', 'む': 'う', 'め': 'え', 'も': 'お',
        'や': 'あ', 'ゆ': 'う', 'よ': 'お',
        'ら': 'あ', 'り': 'い', 'る': 'う', 'れ': 'え', 'ろ': 'お',
        'わ': 'あ', 'を': 'お', 'ん': 'ん',
        'が': 'あ', 'ぎ': 'い', 'ぐ': 'う', 'げ': 'え', 'ご': 'お',
        'ざ': 'あ', 'じ': 'い', 'ず': 'う', 'ぜ': 'え', 'ぞ': 'お',
        'だ': 'あ', 'ぢ': 'い', 'づ': 'う', 'で': 'え', 'ど': 'お',
        'ば': 'あ', 'び': 'い', 'ぶ': 'う', 'べ': 'え', 'ぼ': 'お',
        'ぱ': 'あ', 'ぴ': 'い', 'ぷ': 'う', 'ぺ': 'え', 'ぽ': 'お',

        'ぁ': 'あ', 'ぃ': 'い', 'ぅ': 'う', 'ぇ': 'え', 'ぉ': 'お',
        'ゕ': 'あ', 'ゖ': 'え',
        'ゃ': 'あ', 'ゅ': 'う', 'ょ': 'お',
        'ゎ': 'あ',
        }
    
    if len(syllable) == 2 and syllable[1] in ['ん', 'っ']:
        return remove_initial_consonant(syllable[0]) + syllable[1]

    return vowel_map[syllable[-1]]

def Japanese_G2P_Lyric(music):
    ipa_music = []
    for music_index, (current_duration, current_lyric, current_note) in enumerate(music):
        if current_lyric == '<SLUR>':
            if music_index == 0:
                return None # impossible.
            previous_ipa_lyric = ipa_music[music_index - 1][1]
            ipa_lyric = ([], previous_ipa_lyric[1], previous_ipa_lyric[2])
            ipa_music[music_index - 1] = (
                ipa_music[music_index - 1][0],
                (previous_ipa_lyric[0], previous_ipa_lyric[1], []),
                ipa_music[music_index - 1][2],
                ipa_music[music_index - 1][3],
                )   # coda moving.
            ipa_music.append((
                current_duration,
                ipa_lyric,
                current_note,
                current_lyric
                ))
            continue
        elif current_lyric == '<X>':
            ipa_music.append((current_duration, [current_lyric], current_note, current_lyric))
            continue
        next_syllable = None
        if music_index + 1 < len(music) and music[music_index + 1][1] != '<X>':
            next_syllable = music[music_index + 1][1]
        elif music_index + 2 < len(music) and music[music_index + 1][0] < 0.5: # music[music_index + 1][1] == '<X>' but short
            next_syllable = music[music_index + 2][1]
        else:
            next_syllable = None

        ipa_lyric = _Japanese_Syllable_to_IPA(current_lyric, next_syllable)
        ipa_music.append((
            current_duration,
            [ipa_lyric] if ipa_lyric == '<X>' else ipa_lyric,
            current_note,
            current_lyric
            ))

    return ipa_music

def _Japanese_Syllable_to_IPA(
    syllable: str,
    next_syllable: Optional[str]= None  # for 'ん', 'っ'
    ):
    ipa_dict = {
        'あ'  : [[   ], 'a', []], 'い': [[    ], 'i', []],  'う'  : [[    ], 'ɯ', []],  'え': [[   ], 'e', []], 'お'  : [[    ], 'o', []],
        'か'  : [['k'], 'a', []], 'き': [['k' ], 'i', []],  'く'  : [['k' ], 'ɯ', []],  'け': [['k'], 'e', []], 'こ'  : [['k',], 'o', []],
        'さ'  : [['s'], 'a', []], 'し': [['ɕ' ], 'i', []],  'す'  : [['s' ], 'ɯ', []],  'せ': [['s'], 'e', []], 'そ'  : [['s',], 'o', []],
        'た'  : [['t'], 'a', []], 'ち': [['tɕ'], 'i', []],  'つ'  : [['ts'], 'ɯ', []],  'て': [['t'], 'e', []], 'と'  : [['t',], 'o', []],
        'な'  : [['n'], 'a', []], 'に': [['ɲ' ], 'i', []],  'ぬ'  : [['n' ], 'ɯ', []],  'ね': [['n'], 'e', []], 'の'  : [['n',], 'o', []],
        'は'  : [['h'], 'a', []], 'ひ': [['ç' ], 'i', []],  'ふ'  : [['ɸ' ], 'ɯ', []],  'へ': [['h'], 'e', []], 'ほ'  : [['h',], 'o', []],
        'ま'  : [['m'], 'a', []], 'み': [['m' ], 'i', []],  'む'  : [['m' ], 'ɯ', []],  'め': [['m'], 'e', []], 'も'  : [['m',], 'o', []],
        'や'  : [['j'], 'a', []],                           'ゆ'  : [['j' ], 'ɯ', []],                          'よ'  : [['j',], 'o', []],
        'ら'  : [['ɾ'], 'a', []], 'り': [['ɾ' ], 'i', []],  'る'  : [['ɾ' ], 'ɯ', []],  'れ': [['ɾ'], 'e', []], 'ろ'  : [['ɾ',], 'o', []],
        'わ'  : [['w'], 'a', []],                                                                               'を'  : [[    ], 'o', []],

                                                            'ゔ'  : [['v',], 'ɯ', []],
        'が'  : [['g' ], 'a', []], 'ぎ': [['g' ], 'i', []], 'ぐ'  : [['g',], 'ɯ', []], 'げ': [['g' ], 'e', []], 'ご'  : [['g',], 'o', []],
        'ざ'  : [['dz'], 'a', []], 'じ': [['dʑ'], 'i', []], 'ず'  : [['dz'], 'ɯ', []], 'ぜ': [['dz'], 'e', []], 'ぞ'  : [['dz'], 'o', []],
        'だ'  : [['d' ], 'a', []], 'ぢ': [['dʑ'], 'i', []], 'づ'  : [['dz'], 'ɯ', []], 'で': [['d' ], 'e', []], 'ど'  : [['d',], 'o', []],
        'ば'  : [['b' ], 'a', []], 'び': [['b' ], 'i', []], 'ぶ'  : [['b',], 'ɯ', []], 'べ': [['b' ], 'e', []], 'ぼ'  : [['b',], 'o', []],

        'ぱ'  : [['p' ], 'a', []], 'ぴ': [['p' ], 'i', []], 'ぷ'  : [['p',], 'ɯ', []], 'ぺ': [['p' ], 'e', []], 'ぽ'  : [['p',], 'o', []],

        'きゃ': [['kʲ'], 'a', []],                          'きゅ': [['kʲ'], 'ɯ', []],                          'きょ': [['kʲ'], 'o', []],
        'しゃ': [['ɕ' ], 'a', []],                          'しゅ': [['ɕ' ], 'ɯ', []],                          'しょ': [['ɕ' ], 'o', []],
        'ちゃ': [['tɕ'], 'a', []],                          'ちゅ': [['tɕ'], 'ɯ', []],                          'ちょ': [['tɕ'], 'o', []],
        'にゃ': [['ɲ' ], 'a', []],                          'にゅ': [['ɲ' ], 'ɯ', []],                          'にょ': [['ɲ' ], 'o', []],
        'ひゃ': [['ç' ], 'a', []],                          'ひゅ': [['ç' ], 'ɯ', []],                          'ひょ': [['ç' ], 'o', []],
        'みゃ': [['mʲ'], 'a', []],                          'みゅ': [['mʲ'], 'ɯ', []],                          'みょ': [['mʲ'], 'o', []],
        'りゃ': [['ɾʲ'], 'a', []],                          'りゅ': [['ɾʲ'], 'ɯ', []],                          'りょ': [['ɾʲ'], 'o', []],
        'ぎゃ': [['gʲ'], 'a', []],                          'ぎゅ': [['gʲ'], 'ɯ', []],                          'ぎょ': [['gʲ'], 'o', []],
        'じゃ': [['dʑ'], 'a', []],                          'じゅ': [['dʑ'], 'ɯ', []],                          'じょ': [['dʑ'], 'o', []],
        'びゃ': [['bʲ'], 'a', []],                          'びゅ': [['bʲ'], 'ɯ', []],                          'びょ': [['bʲ'], 'o', []],
        'ぴゃ': [['pʲ'], 'a', []],                          'ぴゅ': [['pʲ'], 'ɯ', []],                          'ぴょ': [['pʲ'], 'o', []],

        'ぁ'  : [[   ], 'a', []], 'ぃ': [[    ], 'i', []],  'ぅ'  : [[    ], 'ɯ', []],  'ぇ': [[   ], 'e', []], 'ぉ'  : [[    ], 'o', []],

        # Special case in Ofuton
        'てゃ': [['tʲ'], 'a', []],                          'てゅ': [['tʲ'], 'ɯ', []],                          'てょ': [['tʲ'], 'o', []],
        'でゃ': [['dʲ'], 'a', []],                          'でゅ': [['dʲ'], 'ɯ', []],                          'でょ': [['dʲ'], 'o', []],
        }
    
    syllable = syllable.replace('ヴ', 'ゔ').replace('シ', 'し').replace('んん', 'ん') # Ofuton uses some katakana.
    if not next_syllable is None:
        next_syllable = next_syllable.replace('ヴ', 'ゔ').replace('シ', 'し').replace('んん', 'ん') # Ofuton uses some katakana.

    if syllable in ipa_dict.keys():
        return ipa_dict[syllable]
    elif len(syllable) > 1 and syllable[-1] == 'っ':
        ipa = _Japanese_Syllable_to_IPA(syllable[:-1], 'っ')
        next_onset = ipa_dict[next_syllable[0]][0] if not next_syllable is None else []
        # if len(next_onset) == 0:
        #     next_onset.append('ʔ')
        ipa[2] = next_onset
        return ipa
    elif len(syllable) > 1 and syllable[-1] == 'ん':
        ipa = _Japanese_Syllable_to_IPA(syllable[:-1], 'ん')
        _, _, ipa[2] = _Japanese_Syllable_to_IPA(syllable[-1], next_syllable)
        return ipa
    elif len(syllable) == 2:
        ipa = ipa_dict[syllable[0]]
        if len(ipa[2]) != 0:
            raise NotImplementedError(f'unimplemented syllable: {syllable}')
        elif syllable[1] in ['ぁ', 'ぃ', 'ぅ', 'ぇ', 'ぉ']:
            stegana_ipa = ipa_dict[syllable[1]]
            ipa[0] = ipa[0] + stegana_ipa[0]
            ipa[1] = stegana_ipa[1]
        else:   # syllable[1] in ['ゕ', 'ゖ', 'ゎ', 'ゃ', 'ゅ', 'ょ']
            raise NotImplementedError(f'unimplemented syllable: {syllable}')
        return ipa
    elif len(syllable) == 3 and syllable[2] == 'ん':
        ipa = _Japanese_Syllable_to_IPA(syllable[:2], syllable[2])
        _, _, ipa[2] = _Japanese_Syllable_to_IPA(syllable[2], next_syllable)
        return ipa

    if syllable == 'ん':
        if next_syllable is None:
            return [[], '', ['ɴ']]
        elif next_syllable == 'っ':  # 'んっ' is impossible, but some songs in Ofuton have 'んっ'. 'っ' is ignored.
            return [[], '', ['ɴ']]
        elif next_syllable[0] in [
            'あ', 'い', 'う', 'え', 'お',
            'な', 'に', 'ぬ', 'ね', 'の',
            'ら', 'り', 'る', 'れ', 'ろ',
            'ざ', 'じ', 'ず', 'ぜ', 'ぞ',
            'だ', 'ぢ', 'づ', 'で', 'ど',
                        'ゔ',
            'や', 'ゆ', 'よ', 'わ', 'を'
        ]:
            return [[], '', ['n']]
        elif next_syllable[0] in [
            'は', 'ひ', 'ふ', 'へ', 'ほ',
            'ま', 'み', 'む', 'め', 'も',
            'ば', 'び', 'ぶ', 'べ', 'ぼ',
            'ぱ', 'ぴ', 'ぷ', 'ぺ', 'ぽ'
        ]:
            return [[], '', ['m']]
        elif next_syllable[0] in [
            'か', 'き', 'く', 'け', 'こ',
            'が', 'ぎ', 'ぐ', 'げ', 'ご'
        ]:
            return [[], '', ['ɴ']]
        elif next_syllable[0] in [
            'さ', 'し', 'す', 'せ', 'そ',
            'た', 'ち', 'つ', 'て', 'と'
        ]:
            return [[], '', ['n']]
        else:
            raise ValueError(f'Unsupported next syllable: {next_syllable}')

    raise NotImplementedError(f'unimplemented syllable: {syllable}, {next_syllable}')

def Process(
    duration: list[float],
    lyric: list[str],
    pitch: list[int],
    sample_rate: int,
    hop_size: int
    ) -> list[Note]:
    lyric_realign = []
    duration_realign = []
    pitch_realign = []
    for index, (lyric_note, duration_note, pitch_note) in enumerate(zip(lyric, duration, pitch)):
        if type(duration_note) == float and type(pitch_note) == int:
            lyric_realign.append(lyric_note)
            duration_realign.append(duration_note)
            pitch_realign.append(pitch_note)
            continue
        elif type(duration_note) == list and type(pitch_note) == list:
            if len(duration_note) != len(pitch_note):
                raise ValueError(f'Inconsist length of duration and pitch index {index}: {len(duration_note)}, {len(pitch_note)}')
            
            lyric_realign.extend([lyric_note] + ['<SLUR>'] * (len(duration_note) - 1))
            duration_realign.extend(duration_note)
            pitch_realign.extend(pitch_note)
        else:
            raise ValueError(f'Unsupported type of duration and pitch index {index}: {type(duration_note)}, {type(pitch_note)}')

    music = [x for x in zip(duration_realign, lyric_realign, pitch_realign)]
    # music = [x for x in zip(duration, lyric, pitch)]

    music = Japanese_G2P_Lyric(music)
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

    for note in music:
        while '' in note.Lyric:
            note.Lyric.remove('')

    
    return music

