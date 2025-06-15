import librosa
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
    lyric: list[str],
    duration: list[float | list[float]],
    pitch: list[int | list[int]],
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

    ipa_realign = [
        x if x in ['<X>', '<SLUR>'] else _Chinese_Syllablize(_Chinese_Phonemize(x)[0])[0]
        for x in lyric_realign        
        ]
    
    music = [
        Note(
            Duration= duration,
            Lyric= lyric,
            Pitch= note,
            Text= text,
            Tech= None
            )
        for duration, lyric, note, text in zip(duration_realign, ipa_realign, pitch_realign, lyric_realign)
        ]
    music = Convert_Duration_Second_to_Frame(
        music= music,
        sample_rate= sample_rate,
        hop_size= hop_size
        )
    
    for index, note in enumerate(music):
        if note.Lyric == '<X>':
            note.Lyric = [note.Lyric]
        elif note.Lyric == '<SLUR>':
            previous_lyric = music[index - 1].Lyric
            music[index - 1].Lyric = (previous_lyric[0], previous_lyric[1], [])
            note.Lyric = ([], previous_lyric[1], previous_lyric[2])

    music = Convert_Lyric_Syllable_to_Sequence(music)

    return music