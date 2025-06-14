import mido, hgtk
from typing import List, Union
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
        pronunciation.replace( '\u0361', '')
        for pronunciation in pronunciations
        ]
    
    return pronunciations

korean_vowels = sorted(
    # list({'u', 'o', 'wɛ', 'jo', 'i', 'ɯ', 'ø', 'wʌ', 'jʌ', 'ja', 'je', 'wi', 'ju', 'wa', 'ʌ', 'ɰi', 'e', 'a', 'we'}),
    list({'e̞', 'i', 'o̞', 'ju', 'a̠', 'ɯ', 'ʌ̹', 'wa̠', 'u', 'jo', 'ɰ', 'jʌ̹', 'je̞', 'ja̠', 'ɰi', 'ɥi', 'wʌ̹', 'we̞'}),
    key= lambda x: len(x),
    reverse= True
    )

korean_boundary_dict = {
    'b': ('', 'b'), 'ɕ͈': ('', 'ɕ͈'), 'd': ('', 'd'), 'pʰ': ('', 'pʰ'),
    'tɕʰ': ('', 'tɕʰ'), 'kʰ': ('', 'kʰ'), 'ɾ': ('', 'ɾ'), 'm': ('', 'm'), 'n': ('', 'n'), 
    'ç': ('', 'ç'), 'sʰ': ('', 'sʰ'), 'tʰ': ('', 'tʰ'), 'p͈': ('', 'p͈'), 'k': ('', 'k'), 
    's͈': ('', 's͈'), 'dʑ': ('', 'dʑ'), 'p': ('', 'p'), 'ɦ': ('', 'ɦ'), 'k͈': ('', 'k͈'), 
    'ɭ': ('ɭ', ''), 'ɡ': ('', 'ɡ'), 't': ('', 't'), 'h': ('', 'h'), 'ʎ': ('', 'ʎ'), 
    'tɕʰ': ('', 'tɕʰ'), 'ŋ': ('ŋ', ''), 'ɕʰ': ('', 'ɕʰ'), 'k̚': ('k', ''), 'tɕ͈': ('', 'tɕ͈'),
    't͈': ('', 't͈'), 'tɕ': ('', 'tɕ'), 'p̚': ('p', ''), 'ʝ': ('', 'ʝ'), 'ɲ': ('', 'ɲ'), 
    't̚': ('t', ''), 'ɸʷ': ('', 'ɸʷ'), 'β': ('', 'β'), 'kx': ('', 'kx'), 'ɣ': ('', 'ɣ'), 
    'x': ('', 'x'), 'ʃʰ': ('', 'ʃʰ'),  'ɸ': ('', 'ɸ'), 'kç': ('', 'kç'), 
    }

convert_phoneme_dict = {
    'a̠': 'a',
    'dʑ': 'tɕ',
    'e̞': 'e',
    'ja̠': 'ja',
    'je̞': 'je',
    'jo': 'jo',
    'ju': 'ju',
    'jʌ̹': 'jɔ', # 'jʌ' -> 'jɔ',
    'kx': 'kʰ',
    'kç': 'kʰ',
    'o̞': 'o',
    'sʰ': 's',
    'wa̠': 'wa',
    'we̞': 'we',
    'wʌ̹': 'wʌ',
    'x': 'h',
    'ç': 'h',
    'ɕʰ': 's',
    'ɕ͈': 's͈',
    'ɣ': 'ɡ',
    'ɥi': 'wi',
    'ɦ': 'h',
    'ɭ': 'ɾ',
    'ɲ': 'n',
    'ɸ': 'h',
    'ɸʷ': 'h',
    'ʃʰ': 's',
    'ʌ̹': 'ʌ',
    'ʝ': 'tɕ',
    'β': 'b',
    'ʎ': 'ɾ'
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
        for x in ['k̚', 'p̚', 't̚']
        ):
        print(pronunciation)
        assert False

    for phoneme in pronunciation:
        if phoneme in korean_vowels:
            if \
                len(syllables) > 0 and \
                (   
                    len(current_consonants) > 1 or
                    (len(current_consonants) > 0 and current_consonants[0] in ['ŋ'])
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
        if lyric != '<X>':
            lyric = (
                [convert_phoneme_dict.get(x, x) for x in lyric[0]],
                convert_phoneme_dict.get(lyric[1], lyric[1]),
                [convert_phoneme_dict.get(x, x) for x in lyric[2]],
                )
            if hgtk.letter.decompose(music[index][1])[1] == 'ㅙ':   # ko_pron does not consider 'wɛ'
                lyric = (lyric[0], 'wɛ', lyric[2])
            elif hgtk.letter.decompose(music[index][1])[1] == 'ㅐ':   # ko_pron does not consider 'ɛ'
                lyric = (lyric[0], 'ɛ', lyric[2])
            elif hgtk.letter.decompose(music[index][1])[1] == 'ㅒ':
                lyric = (lyric[0], 'jɛ', lyric[2])
        music[index] = (
            music[index][0],
            ['<X>'] if lyric == '<X>' else lyric,
            music[index][2],
            music[index][1]
            )

    return music


def Process(
    midi_path: str,
    sample_rate: int,
    hop_size: int
    ) -> list[Note]:
    mid = mido.MidiFile(midi_path, charset='CP949')

    music = []
    note_states = {}
    last_note = None

    # Note on 쉼표
    # From Lyric to message before note on: real note
    for message in list(mid):
        if message.type == 'note_on' and message.velocity != 0:                
            if len(note_states) == 0:
                if message.time < 0.1:
                    music[-1][0] += message.time
                else:
                    if len(music) > 0 and music[-1][1] == '<X>':
                        music[-1][0] += message.time
                    else:
                        music.append([message.time, '<X>', 0])
            else:
                note_states[last_note]['Time'] += message.time
            note_states[message.note] = {
                'Lyric': None,
                'Time': 0.0
                }
            last_note = message.note
        elif message.type == 'lyrics':
            if message.text == '\r' or last_note is None:    # If there is a bug in lyric
                continue
            note_states[last_note]['Lyric'] = message.text.strip()
            note_states[last_note]['Time'] += message.time
        elif message.type == 'note_off' or (message.type == 'note_on' and message.velocity == 0):
            note_states[message.note]['Time'] += message.time
            music.append([note_states[message.note]['Time'], note_states[message.note]['Lyric'], message.note])
            del note_states[message.note]
            last_note = None
        else:
            if len(note_states) == 0:
                if len(music) > 0 and music[-1][1] == '<X>':
                    music[-1][0] += message.time
                else:
                    music.append([message.time, '<X>', 0])
            else:
                note_states[last_note]['Time'] += message.time
    if len(note_states) > 0:
        return None
    music = [x for x in music if x[0] > 0.0]
    
    music = [
        [
            duration,
            '<X>' if lyric in ['J', 'H'] else lyric,
            0 if lyric in ['J', 'H'] else note
            ]
        for duration, lyric, note in music
        ]
    # for x in music:
    #     print(x)
    # assert False
    
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