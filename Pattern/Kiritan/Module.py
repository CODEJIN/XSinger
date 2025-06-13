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
        if current_lyric == '<X>':
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
    score_path: str,
    sample_rate: int,
    hop_size: int
    ) -> list[Note]:
    
    score = music21.converter.parse(score_path)
    current_tempo = music21.tempo.MetronomeMark(number= 120)

    music = []
    cumulative_time = 0.0

    for part in score.parts:
        previous_split = 1
        for element in part.flatten():
            if isinstance(element, music21.tempo.MetronomeMark):
                current_tempo = element
            elif isinstance(element, music21.note.Note):
                note = librosa.note_to_midi(element.nameWithOctave.replace('-', '♭'))
                if not element.lyric is None and '゛' in element.lyric:
                    if element.lyric[0] == '゛':
                        music[-1][1][2] = music[-1][1][:-1] + dakuten_fix(music[-1][1][-1])
                        element.lyric = element.lyric[:-1]
                    else:
                        hakuten_index = element.lyric.index('゛')
                        element.lyric = element.lyric[:max(0, hakuten_index - 2)] + dakuten_fix(element.lyric[hakuten_index - 1]) + element.lyric[hakuten_index + 1:]

                if element.lyric is None:                        
                    duration_seconds = get_duration_seconds(element.quarterLength, current_tempo)
                    cumulative_time += duration_seconds
                    if music[-1][2] == note:
                        for previous_split_index in range(1, previous_split + 1):
                            music[-previous_split_index] = (
                                music[-previous_split_index][0] + duration_seconds / previous_split,
                                music[-previous_split_index][1],
                                music[-previous_split_index][2]
                                )
                    else:   # music[-1][0] != note, 28의 'つう', 47 'ろ' 체크
                        if previous_split == 1:
                            music.append((duration_seconds, remove_initial_consonant(music[-1][1]), note))
                            if music[-2][1][-1] in ['ん', 'っ']:    # -2 is correct because of new score.
                                music[-2] = (music[-2][0], music[-2][1][:-1], music[-2][2])
                        elif previous_split == 2:
                            if music[-1][1][-1] in ['ん', 'っ']:
                                print(f'{score_path} is skipped.')
                                return None
                            music[-2] = (music[-2][0] + music[-1][0], music[-2][1], music[-2][2])
                            music[-1] = (duration_seconds, music[-1][1], note)
                        else:
                            print(f'{score_path} is skipped.')
                            return None
                elif len(element.lyric) == 1:
                    duration_seconds = get_duration_seconds(element.quarterLength, current_tempo)
                    cumulative_time += duration_seconds
                    lyric = element.lyric
                    if element.lyric == 'ー':
                        lyric = remove_initial_consonant(music[-1][1])
                    elif element.lyric in ['ん', 'っ']:
                        previous_syllable = [
                            previous_syllable
                            for _, previous_syllable, _ in music
                            if previous_syllable != '<X>'][-1]  # sometime staccato makes problem.
                        lyric = remove_initial_consonant(previous_syllable) + element.lyric                            
                    music.append((duration_seconds, lyric, note))
                    previous_split = 1
                else:   # len(element.lyric) > 1
                    lyrics = []
                    for syllable in element.lyric:
                        if not syllable in ['ぁ', 'ぃ', 'ぅ', 'ぇ', 'ぉ', 'ゕ', 'ゖ', 'っ', 'ゃ', 'ゅ', 'ょ', 'ゎ', 'ん'] or len(lyrics) == 0:
                            lyrics.append([syllable])
                        else:
                            lyrics[-1].append(syllable)
                    lyrics = [''.join(x) for x in lyrics]
                    notes = [note] * len(lyrics)
                    duration_seconds = [get_duration_seconds(element.quarterLength / len(lyrics), current_tempo)] * len(lyrics)
                    cumulative_time += sum(duration_seconds)
                    for note, duration, lyric in zip(notes, duration_seconds, lyrics):
                        music.append((duration, lyric, note))                        
                    previous_split = len(lyrics)
            elif isinstance(element, music21.note.Rest):
                duration_seconds = get_duration_seconds(element.quarterLength, current_tempo)
                cumulative_time += duration_seconds
                if len(music) > 0 and music[-1][2] == 0:
                    music[-1] = (music[-1][0] + duration_seconds, music[-1][1], music[-1][2])
                else:
                    music.append((duration_seconds, '<X>', 0))

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
    while True:
        if music[-1].Lyric[0] == '<X>':
            music = music[:-1]
        else:
            break

    for note in music:
        if '' in note.Lyric:
            print(score_path)
            for x in music:
                print(x)
            assert False

    return music

