import numpy as np
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

# Basically, Chinese does not consider onset, nucleus, coda. It uses intial + final.
# But, in multilingual condition, onset, nucleus, coda concept is usable for compatibility.
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

def Process(metadata: dict, sample_rate: int, hop_size: int) -> tuple[list[Note], np.ndarray]:
    # word base info
    text_list = metadata['txt']

    # phoneme base info
    phoneme_duration_list = metadata['ph_durs']
    phoneme_note_list = metadata['ep_pitches']
    mix_tech_list = metadata['mix_tech']
    falsetto_tech_list = metadata['falsetto_tech']
    breathy_tech_list = metadata['breathy_tech']
    pharyngeal_tech_list = metadata['pharyngeal_tech']
    vibrato_tech_list = metadata['vibrato_tech']
    glissando_tech_list = metadata['glissando_tech']

    # phoneme to word
    phoneme_to_word_list = metadata['ph2words']

    text_list_0 = text_list
    note_list_0 = [[] for _ in text_list]
    duration_list_0 = [[] for _ in text_list]
    mix_tech_list_0 = [[] for _ in text_list]
    falsetto_tech_list_0 = [[] for _ in text_list]
    breathy_tech_list_0 = [[] for _ in text_list]
    pharyngeal_tech_list_0 = [[] for _ in text_list]
    vibrato_tech_list_0 = [[] for _ in text_list]
    glissando_tech_list_0 = [[] for _ in text_list]
    for (
        phoneme_note, phoneme_duration, phoneme_to_word,
        mix_tech, falsetto_tech, breathy_tech,
        pharyngeal_tech, vibrato_tech, glissando_tech 
        ) in zip(
        phoneme_note_list, phoneme_duration_list, phoneme_to_word_list,
        mix_tech_list, falsetto_tech_list, breathy_tech_list,
        pharyngeal_tech_list, vibrato_tech_list, glissando_tech_list 
        ):
        note_list_0[phoneme_to_word].append(phoneme_note)
        duration_list_0[phoneme_to_word].append(phoneme_duration)
        mix_tech_list_0[phoneme_to_word].append(mix_tech)
        falsetto_tech_list_0[phoneme_to_word].append(falsetto_tech)
        breathy_tech_list_0[phoneme_to_word].append(breathy_tech)
        pharyngeal_tech_list_0[phoneme_to_word].append(pharyngeal_tech)
        vibrato_tech_list_0[phoneme_to_word].append(vibrato_tech)
        glissando_tech_list_0[phoneme_to_word].append(glissando_tech)

    ipa_list_0 = []
    for text in text_list_0:
        if text in ['<SP>', '<AP>']:
            ipa_list_0.append([('<X>',)])
        else:
            ipa = _Chinese_Syllablize(_Chinese_Phonemize(texts= text)[0])
            ipa_list_0.append(ipa)
    
    text_list_1 = []
    ipa_list_1 = []
    note_list_1 = []
    duration_list_1 = []
    mix_tech_list_1 = []
    falsetto_tech_list_1 = []
    breathy_tech_list_1 = []
    pharyngeal_tech_list_1 = []
    vibrato_tech_list_1 = []
    glissando_tech_list_1 = []
    for (
        text, ipa, phoneme_note, phoneme_duration,
        mix_tech, falsetto_tech, breathy_tech,
        pharyngeal_tech, vibrato_tech, glissando_tech 
        ) in zip(
        text_list_0, ipa_list_0, note_list_0, duration_list_0,
        mix_tech_list_0, falsetto_tech_list_0,
        breathy_tech_list_0, pharyngeal_tech_list_0,
        vibrato_tech_list_0, glissando_tech_list_0
        ):
        num_syllable = len(ipa) if ipa[0] != '<X>' else 1

        num_phonemes = [
            len(onsets) + 1 + len(coda)
            for onsets, nucleus, coda in ipa
            ] if ipa[0][0] != '<X>' else [1]

        if num_syllable == 1 and len(phoneme_note) > 1:
            text_list_1.append('<X>' if text in ['<SP>', '<AP>'] else text)
            ipa_list_1.extend(ipa)
            note_list_1.append(phoneme_note)
            duration_list_1.append(phoneme_duration)
            mix_tech_list_1.append(mix_tech)
            falsetto_tech_list_1.append(falsetto_tech)
            breathy_tech_list_1.append(breathy_tech)
            pharyngeal_tech_list_1.append(pharyngeal_tech)
            vibrato_tech_list_1.append(vibrato_tech)
            glissando_tech_list_1.append(glissando_tech)
        # elif sum(num_phonemes) <= len(phoneme_note):
        else:
            for index in range(len(phoneme_note) - sum(num_phonemes)):
                num_phonemes[index % len(num_phonemes)] += 1
            for index in range(sum(num_phonemes) - len(phoneme_note)):
                num_phonemes[-(index % len(num_phonemes)) - 1] -= 1
            
            text_list_1.append('<X>' if text in ['<SP>', '<AP>'] else text)
            text_list_1.extend(['<SLUR>'] * (num_syllable - 1))
            ipa_list_1.extend(ipa)

            start_index = 0
            for num_phoneme in num_phonemes:
                note_list_1.append(phoneme_note[start_index:start_index + num_phoneme])
                duration_list_1.append(phoneme_duration[start_index:start_index + num_phoneme])
                mix_tech_list_1.append(mix_tech[start_index:start_index + num_phoneme])
                falsetto_tech_list_1.append(falsetto_tech[start_index:start_index + num_phoneme])
                breathy_tech_list_1.append(breathy_tech[start_index:start_index + num_phoneme])
                pharyngeal_tech_list_1.append(pharyngeal_tech[start_index:start_index + num_phoneme])
                vibrato_tech_list_1.append(vibrato_tech[start_index:start_index + num_phoneme])
                glissando_tech_list_1.append(glissando_tech[start_index:start_index + num_phoneme])
                start_index += num_phoneme

    text_list = []
    ipa_list = []
    note_list = []
    duration_list = []
    mix_tech_list = []
    falsetto_tech_list = []
    breathy_tech_list = []
    pharyngeal_tech_list = []
    vibrato_tech_list = []
    glissando_tech_list = []
    for (
        text, ipa, phoneme_note, phoneme_duration,
        mix_tech, falsetto_tech, breathy_tech,
        pharyngeal_tech, vibrato_tech, glissando_tech 
        ) in zip(
        text_list_1, ipa_list_1, note_list_1, duration_list_1,
        mix_tech_list_1, falsetto_tech_list_1,
        breathy_tech_list_1, pharyngeal_tech_list_1,
        vibrato_tech_list_1, glissando_tech_list_1
        ):
        current_text = []
        current_ipa = []
        current_note = []
        current_duration = []
        current_mix_tech = []
        current_falsetto_tech = []
        current_breathy_tech = []
        current_pharyngeal_tech = []
        current_vibrato_tech = []
        current_glissando_tech = []
        for note_index, note in enumerate(phoneme_note):
            if len(current_note) > 0 and current_note[-1] == note:
                current_duration[-1] += phoneme_duration[note_index]
                current_mix_tech[-1] += mix_tech[note_index]
                current_falsetto_tech[-1] += falsetto_tech[note_index]
                current_breathy_tech[-1] += breathy_tech[note_index]
                current_pharyngeal_tech[-1] += pharyngeal_tech[note_index]
                current_vibrato_tech[-1] += vibrato_tech[note_index]
                current_glissando_tech[-1] += glissando_tech[note_index]
            else:
                if len(current_note) == 0:
                    current_text.append(text)
                    current_ipa.append(ipa)
                elif current_note[-1] != note:
                    current_text.append('<SLUR>')
                    ipa = current_ipa.pop()
                    current_ipa.extend([(ipa[0], ipa[1], []), ([], ipa[1], ipa[2])])
                else:
                    assert False    # ?
                current_note.append(note)
                current_duration.append(phoneme_duration[note_index])
                current_mix_tech.append(mix_tech[note_index])
                current_falsetto_tech.append(falsetto_tech[note_index])
                current_breathy_tech.append(breathy_tech[note_index])
                current_pharyngeal_tech.append(pharyngeal_tech[note_index])
                current_vibrato_tech.append(vibrato_tech[note_index])
                current_glissando_tech.append(glissando_tech[note_index])

        text_list.extend(current_text)
        ipa_list.extend(current_ipa)
        note_list.extend(current_note)
        duration_list.extend(current_duration)
        mix_tech_list.extend(current_mix_tech)
        falsetto_tech_list.extend(current_falsetto_tech)
        breathy_tech_list.extend(current_breathy_tech)
        pharyngeal_tech_list.extend(current_pharyngeal_tech)
        vibrato_tech_list.extend(current_vibrato_tech)
        glissando_tech_list.extend(current_glissando_tech)

    duration_list = [round(x, 5) for x in duration_list]
    mix_tech_list = [max(0, min(1, x)) for x in mix_tech_list]
    falsetto_tech_list = [max(0, min(1, x)) for x in falsetto_tech_list]
    breathy_tech_list = [max(0, min(1, x)) for x in breathy_tech_list]
    pharyngeal_tech_list = [max(0, min(1, x)) for x in pharyngeal_tech_list]
    vibrato_tech_list = [max(0, min(1, x)) for x in vibrato_tech_list]
    glissando_tech_list = [max(0, min(1, x)) for x in glissando_tech_list]
    
    music = [
        Note(
            Duration= duration,
            Lyric= ipa,
            Pitch= note,
            Text= text,
            Tech= None
            )
        for duration, ipa, note, text in zip(
            duration_list, ipa_list, note_list, text_list
            )
        ]
    music = Convert_Duration_Second_to_Frame(
        music= music,
        sample_rate= sample_rate,
        hop_size= hop_size
        )
    music = Convert_Lyric_Syllable_to_Sequence(music)
    
    # GTSinger has the tech info.
    tech = np.array([
        mix_tech_list,
        falsetto_tech_list,
        breathy_tech_list,
        pharyngeal_tech_list,
        vibrato_tech_list,
        glissando_tech_list 
        ])
    
    return music, tech

