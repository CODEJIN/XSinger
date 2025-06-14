import hgtk
import numpy as np
from typing import List, Dict, Union, Optional, Tuple

from .. import Note, Convert_Duration_Second_to_Frame, Convert_Lyric_Syllable_to_Sequence

def remove_diacritics_and_fix(ipa: str):
    return ipa.replace('_ko', '').replace('\u031A', '').replace('ː', '')

def _Syllablize(
    pronunciation: list[str]
    ) -> List[Tuple[List[str], List[str], List[str]]]:
    use_vowels = ['ɯ', 'e', 'a', 'i', 'ʌ', 'o', 'ɛ', 'u']
    
    convert_phoneme = {
        'dʑ': 'tɕ',
        'c': 'tɕ',
        'cʰ': 'tɕʰ',
        'c͈': 'tɕ͈',
        'ç': 'h',
        'ɦ': 'h',
        'ɸ': 'h',
        'ɭ': 'ɾ',
        'ɲ': 'n',
        'x': 'h',
        'ɨ': 'ɯ',
        'ɐ': 'a',
        'ɕʰ': 's',
        'ɕ͈': 's͈',
        'ɰ': 'ɯ',
        'sʰ': 's',
        }

    syllable_list: List[Tuple[List[str], str, List[str]]] = []

    onsets = []
    nucleus = []
    index = 0
    while index < len(pronunciation):
        phoneme = convert_phoneme.get(pronunciation[index], pronunciation[index])
        if not phoneme in use_vowels:
            onsets.append(phoneme)
        else:   # vowel
            while len(onsets) > 0 and onsets[-1] in ['w', 'j', 'ɥ']:
                nucleus.append(onsets.pop())
            nucleus.append(phoneme)
            syllable_list.append((onsets, nucleus, []))
            onsets = []
            nucleus = []
        index += 1

    syllable_list[-1] = (syllable_list[-1][0], syllable_list[-1][1], onsets)

    index = 0
    while index < len(syllable_list) - 1:
        next_onset = syllable_list[index + 1][0]
        if len(next_onset) > 1 or (len(next_onset) == 1 and next_onset[0] == 'ɴ'):
            syllable_list[index] = (
                syllable_list[index][0],
                syllable_list[index][1],
                next_onset[:1]
                )
            syllable_list[index + 1] = (
                next_onset[1:],
                syllable_list[index + 1][1],
                syllable_list[index + 1][2],
                )
        index += 1

    return syllable_list


def Process(metadata: dict, sample_rate: int, hop_size: int) -> tuple[list[Note], np.ndarray]:
    # word base info
    text_list = metadata['txt']

    # phoneme base info
    ipa_list = metadata['ph']
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
    ipa_list_0 = [[] for _ in text_list]
    note_list_0 = [[] for _ in text_list]
    duration_list_0 = [[] for _ in text_list]
    mix_tech_list_0 = [[] for _ in text_list]
    falsetto_tech_list_0 = [[] for _ in text_list]
    breathy_tech_list_0 = [[] for _ in text_list]
    pharyngeal_tech_list_0 = [[] for _ in text_list]
    vibrato_tech_list_0 = [[] for _ in text_list]
    glissando_tech_list_0 = [[] for _ in text_list]
    for (
        ipa, phoneme_note, phoneme_duration, phoneme_to_word,
        mix_tech, falsetto_tech, breathy_tech,
        pharyngeal_tech, vibrato_tech, glissando_tech 
        ) in zip(
        ipa_list, phoneme_note_list, phoneme_duration_list, phoneme_to_word_list,
        mix_tech_list, falsetto_tech_list, breathy_tech_list,
        pharyngeal_tech_list, vibrato_tech_list, glissando_tech_list 
        ):
        ipa = remove_diacritics_and_fix(ipa)        
        ipa_list_0[phoneme_to_word].append(remove_diacritics_and_fix(ipa))
        note_list_0[phoneme_to_word].append(phoneme_note)
        duration_list_0[phoneme_to_word].append(phoneme_duration)
        mix_tech_list_0[phoneme_to_word].append(mix_tech)
        falsetto_tech_list_0[phoneme_to_word].append(falsetto_tech)
        breathy_tech_list_0[phoneme_to_word].append(breathy_tech)
        pharyngeal_tech_list_0[phoneme_to_word].append(pharyngeal_tech)
        vibrato_tech_list_0[phoneme_to_word].append(vibrato_tech)
        glissando_tech_list_0[phoneme_to_word].append(glissando_tech)
        
        if ipa[-1] in ['\u02B2', '\u02B7']:
            ipa_list_0[phoneme_to_word].pop()
            ipa_list_0[phoneme_to_word].append(ipa[:-1])
            if ipa[-1] == '\u02B2':
                ipa_list_0[phoneme_to_word].append('j')
            elif ipa[-1] == '\u02B7':
                ipa_list_0[phoneme_to_word].append('w')            
            note_list_0[phoneme_to_word].append(note_list_0[phoneme_to_word][-1])
            duration_list_0[phoneme_to_word].append(0.0)
            mix_tech_list_0[phoneme_to_word].append(mix_tech_list_0[phoneme_to_word][-1])
            falsetto_tech_list_0[phoneme_to_word].append(falsetto_tech_list_0[phoneme_to_word][-1])
            breathy_tech_list_0[phoneme_to_word].append(breathy_tech_list_0[phoneme_to_word][-1])
            pharyngeal_tech_list_0[phoneme_to_word].append(pharyngeal_tech_list_0[phoneme_to_word][-1])
            vibrato_tech_list_0[phoneme_to_word].append(vibrato_tech_list_0[phoneme_to_word][-1])
            glissando_tech_list_0[phoneme_to_word].append(glissando_tech_list_0[phoneme_to_word][-1])

    note_list_0 = [
        [
            phoneme_note[note_index + 1] if phoneme_note[note_index] is None else phoneme_note[note_index]
            for note_index in range(len(phoneme_note))
            ]
        for phoneme_note in note_list_0
        ]   # the note of 'j', 'w' is changed from None to next vowel's note.

    try:
        ipa_list_0 = [
            [['<X>',]] if x[0] in ['<SP>', '<AP>'] else _Syllablize(x)
            for x in ipa_list_0
            ]
    except IndexError as e: # Some phoneme does not have nucleus in GTSinger.
        print('{} is skipped.'.format(metadata['item_name']))
        return None, None

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
    
    # solve some diacritic like pʷpʷ ('봐', (['p', 'w', 'p'], ['w', 'a'], []), 70, 0.268) -> (['p'], ['w', 'a'], [])
    for ipa in ipa_list:
        if 'j'  in ipa[0]:
            phoneme_index = ipa[0].index('j')
            if ipa[0][phoneme_index - 1] == ipa[0][phoneme_index + 1]:
                ipa[0].pop(phoneme_index)
                ipa[0].pop(phoneme_index)
        if 'w'  in ipa[0]:
            phoneme_index = ipa[0].index('w')
            if ipa[0][phoneme_index - 1] == ipa[0][phoneme_index + 1]:
                ipa[0].pop(phoneme_index)
                ipa[0].pop(phoneme_index)

    #diphone concat
    ipa_list =  [
        (ipa[0], ''.join(ipa[1]), ipa[2]) if len(ipa) == 3 else ipa
        for ipa in ipa_list
        ]

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
    
    for note in music:
        new_lyric = []
        for phoneme in note.Lyric:
            while 'jj' in phoneme:
                phoneme = phoneme.replace('jj', 'j')
            while 'ww' in phoneme:
                phoneme = phoneme.replace('ww', 'w')
            while 'ji' in phoneme:
                phoneme = phoneme.replace('ji', 'i')
            while 'jɛ' in phoneme:
                phoneme = phoneme.replace('jɛ', 'je')
            while 'wo' in phoneme:
                phoneme = phoneme.replace('wo', 'o')
            while 'wu' in phoneme:
                phoneme = phoneme.replace('wu', 'u')            
            new_lyric.append(phoneme)

        note.Lyric = new_lyric
            
    for note in music:
        while len(note.Lyric) > 1 and note.Lyric[0] == note.Lyric[1]:
            note.Lyric.pop(0)
        while len(note.Lyric) > 1 and note.Lyric[-1] == note.Lyric[-2]:
            note.Lyric.pop()

    note_index = 1
    while note_index < len(music):
        if note_index > 0 and music[note_index].Text == '<SLUR>' and music[note_index].Lyric[0] != music[note_index -1].Lyric[-1]:
            if len(music[note_index -1].Lyric[-1]) > 1 and music[note_index -1].Lyric[-1][1] == music[note_index].Lyric[0]:
                music[note_index].Lyric[0] = music[note_index -1].Lyric[-1]
            elif music[note_index -1].Lyric[-1] == 'ɯ' and music[note_index].Lyric[0] == 'i':
                music[note_index -1].Lyric[-1] = 'ɯi'
                music[note_index].Lyric[0] = 'ɯi'
            else:
                for x in music:
                    print(x)
                assert False
        if note_index > 0 and \
            music[note_index].Text == '<SLUR>' and \
            music[note_index].Lyric == music[note_index -1].Lyric and \
            music[note_index].Pitch == music[note_index -1].Pitch:
            music[note_index -1].Duration += music[note_index].Duration
            music.pop(note_index)
            note_index -= 1
        note_index += 1

    for note in music:
        if hgtk.checker.is_hangul(note.Text):
            if hgtk.letter.decompose(note.Text)[1] == 'ㅙ':   # ko_pron does not consider 'wɛ'
                note.Lyric[note.Lyric.index('we')] = 'wɛ'
            elif hgtk.letter.decompose(note.Text)[1] == 'ㅒ':
                note.Lyric[note.Lyric.index('je')] = 'jɛ'

        while 'ɯi' in note.Lyric:
            note.Lyric[note.Lyric.index('ɯi')] = 'ɰi'
        while 'jʌ' in note.Lyric:
            note.Lyric[note.Lyric.index('jʌ')] = 'jɔ'
        while 'ɥi' in note.Lyric:
            note.Lyric[note.Lyric.index('ɥi')] = 'wi'
        while 'ɥɥi' in note.Lyric:
            note.Lyric[note.Lyric.index('ɥɥi')] = 'wi'

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
