import os
import pandas as pd

from .. import Note, Convert_Duration_Second_to_Frame, Convert_Lyric_Syllable_to_Sequence

def English_Phoneme_Split(
    music: list[tuple[float, str, int]]
    ) -> tuple[float, tuple[list[str], str, list[str]], int]:
    # /ʔ/ is not vowel.
    # However, in English, the next vowel of /ʔ/ is usually silence /ə/.
    # To rule make, I manage /ʔ/ like a vowel
    diphones = ['tʃ', 'hw', 'aʊ', 'kw', 'eɪ', 'oʊ', 'dʒ', 'ɔɪ', 'aɪ']
    use_vowels = [
        'aʊ', 'eɪ', 'oʊ', 'ɔɪ', 'aɪ', 'e', 'i', 'o', 'u', 'ɐ', 'a',
        'æ', 'ɑ', 'ɔ', 'ə', 'ɛ', 'ɜ', 'ɪ', 'ʊ', 'ʌ', 'ᵻ', 'ɚ', 'ʔ', 
        ]

    pronunciations = [x[1] for x in music]
    lyrics = []
    for pronunciation in pronunciations:
        if pronunciation == '<X>':
            lyrics.append(pronunciation)
            continue
        onsets = []
        nucleus = None
        codas = []
        index = 0
        while index < len(pronunciation):
            is_diphone = False
            if pronunciation[index:index + 2] in diphones:
                phoneme = pronunciation[index:index + 2]
                is_diphone = True
            else:
                phoneme = pronunciation[index]
            
            if not phoneme in use_vowels:
                if nucleus is None:
                    onsets.append(phoneme)
                else:   # coda
                    codas.append(phoneme)
            else:   # vowel
                nucleus = phoneme
                if len(codas) > 0:
                    assert False
            index += 1 + is_diphone

        if nucleus == 'ɐ': nucleus = 'ə'
        if nucleus == 'a': nucleus = 'æ'
        lyrics.append([onsets, nucleus, codas])

    # assert len(lyrics) == len(music)

    if len(lyrics) != len(music):
        return None
    music = [
        (music[0], lyric if lyric != '<X>' else ['<X>'], music[2], music[1])
        for music, lyric in zip(music, lyrics)
        ]

    return music

def Process(
    midi_path: str,
    lyric_path: str,
    sample_rate: int,
    hop_size: int
    ) -> list[Note]:
    midi_df = pd.read_csv(midi_path)
    lyrics = [
        line.strip().split('\t')[0]
        for line in open(lyric_path, 'r', encoding= 'utf-8-sig').readlines()
        ]   # deque is better, but I don't use.
    assert len(lyrics) == len(midi_df), f'There is a lyric data has a problem in: \'{lyric_path}\', lyric length: {len(lyrics)}, midi length: {len(midi_df)}'
    midi_df['syllable'] = lyrics

    music = []
    last_end_time = 0.0
    for _, row in midi_df.iterrows():
        if last_end_time < row['start']:
            music.append((row['start'] - last_end_time, '<X>', 0))
        elif last_end_time > row['start']:
            row['start'] = last_end_time

        music.append((round(row['end'] - row['start'], 5), row['syllable'], row['pitch']))
        last_end_time = row['end']
    music = English_Phoneme_Split(music)
    
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