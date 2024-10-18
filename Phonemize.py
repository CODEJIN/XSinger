import logging, re, sys

from phonemizer import logger as phonemizer_logger
from phonemizer.backend import BACKENDS
from phonemizer.punctuation import Punctuation
from phonemizer.separator import default_separator, Separator
from phonemizer.phonemize import _phonemize

from typing import List, Dict, Union, Optional

from ko_pron import romanise
from pypinyin import pinyin
from pinyin_to_ipa import pinyin_to_ipa

from enum import Enum

logger = phonemizer_logger.get_logger()
logger.setLevel(logging.CRITICAL)

# English
english_phonemizer = BACKENDS['espeak'](
    language= 'en-us',
    punctuation_marks= Punctuation.default_marks(),
    preserve_punctuation= True,
    with_stress= False,
    tie= True,
    language_switch= 'remove-flags',
    words_mismatch= 'ignore',
    logger= logger
    )
whitespace_re = re.compile(r'\s+')
english_abbreviations = [(re.compile('\\b%s\\.' % x[0], re.IGNORECASE), x[1]) for x in [
    ('mrs', 'misess'),
    ('mr', 'mister'),
    ('dr', 'doctor'),
    ('st', 'saint'),
    ('co', 'company'),
    ('jr', 'junior'),
    ('maj', 'major'),
    ('gen', 'general'),
    ('drs', 'doctors'),
    ('rev', 'reverend'),
    ('lt', 'lieutenant'),
    ('hon', 'honorable'),
    ('sgt', 'sergeant'),
    ('capt', 'captain'),
    ('esq', 'esquire'),
    ('ltd', 'limited'),
    ('col', 'colonel'),
    ('ft', 'fort'),
    ]]
def expand_abbreviations(text: str):
    for regex, replacement in english_abbreviations:
        text = re.sub(regex, replacement, text)

    return text

def _English_Phonemize(texts: Union[str, List[str]], chunk: int= 5000):
    if type(texts) == str:
        texts = [texts]

    texts = [text.lower() for text in texts]
    texts = [expand_abbreviations(text) for text in texts]

    pronunciations = []

    indices = range(0, len(texts), chunk)
    for index in indices:
        pronunciations_chunk = _phonemize(
            backend= english_phonemizer,
            text= texts[index:index + chunk],
            separator= Separator(phone= '|'), # default_separator,
            strip= True,
            njobs= 4,
            prepend_text= False,
            preserve_empty_lines= False
            )
        pronunciations.extend([
            re.sub(whitespace_re, ' ', pronunciation).replace('ː', '')
            for pronunciation in pronunciations_chunk
            ])

    return pronunciations





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
    'ns͈': ('n', 's͈'),
    'p̚t͡ɕʰ': ('p̚', 't͡ɕʰ'),
    'h': ('', 'h'),
    'ɭsʰ': ('ɭ', 'sʰ'),
    't͡ɕ͈': ('', 't͡ɕ͈'),
    't̚t͡ɕ͈': ('t̚', 't͡ɕ͈'),
    'p̚': ('', 'p̚'),
    'ŋd͡ʑ': ('ŋ', 'd͡ʑ'),
    'mt͡ɕ͈': ('m', 't͡ɕ͈'),
    'nn': ('n', 'n'),
    'nɡ': ('n', 'ɡ'),
    'ŋp͈': ('ŋ', 'p͈'),
    'ɭt͡ɕʰ': ('ɭ', 't͡ɕʰ'),
    'mk͈': ('m', 'k͈'),
    't̚t͈': ('t̚', 't͈'),
    't͈': ('', 't͈'),
    'ɭkʰ': ('ɭ', 'kʰ'),
    'ŋd': ('ŋ', 'd'),
    'ŋn': ('ŋ', 'n'),
    't̚k͈': ('t̚', 'k͈'),
    'ɭb': ('ɭ', 'b'),
    'ŋb': ('ŋ', 'b'),
    'ɭs͈': ('ɭ', 's͈'),
    'p̚pʰ': ('p̚', 'pʰ'),
    'ɡ': ('', 'ɡ'),
    'np͈': ('n', 'p͈'),
    'k̚p͈': ('k̚', 'p͈'),
    't̚p͈': ('t̚', 'p͈'),
    'nb': ('n', 'b'),
    't': ('', 't'),
    'k͈': ('', 'k͈'),
    'k̚pʰ': ('k̚', 'pʰ'),
    'md': ('m', 'd'),
    'ŋt͡ɕ͈': ('ŋ', 't͡ɕ͈'),
    'ɦ': ('', 'ɦ'),
    'kʰ': ('', 'kʰ'),
    'nk͈': ('n', 'k͈'),
    'nm': ('n', 'm'),
    'm': ('', 'm'),
    'ɭɭ': ('ɭ', 'ɭ'),
    'k': ('', 'k'),
    'ŋɦ': ('ŋ', 'ɦ'),
    'b': ('', 'b'),
    't̚pʰ': ('t̚', 'pʰ'),
    't̚t͡ɕʰ': ('t̚', 't͡ɕʰ'),
    'p͈': ('', 'p͈'),
    'sʰ': ('', 'sʰ'),
    'k̚t͈': ('k̚', 't͈'),
    'nɦ': ('n', 'ɦ'),
    't̚': ('', 't̚'),
    'ŋt͈': ('ŋ', 't͈'),
    'ŋm': ('ŋ', 'm'),
    'k̚': ('', 'k̚'),
    'p̚p͈': ('p̚', 'p͈'),
    'ŋkʰ': ('ŋ', 'kʰ'),
    'k̚tʰ': ('k̚', 'tʰ'),
    'ms͈': ('m', 's͈'),
    'ntʰ': ('n', 'tʰ'),
    'ŋtʰ': ('ŋ', 'tʰ'),
    'ɾ': ('', 'ɾ'),
    'ŋt͡ɕʰ': ('ŋ', 't͡ɕʰ'),
    'nd͡ʑ': ('n', 'd͡ʑ'),
    'mtʰ': ('m', 'tʰ'),
    'ɭpʰ': ('ɭ', 'pʰ'),
    'ɭk͈': ('ɭ', 'k͈'),
    'ɭɡ': ('ɭ', 'ɡ'),
    'mɡ': ('m', 'ɡ'),
    't͡ɕ': ('', 't͡ɕ'),
    'p̚kʰ': ('p̚', 'kʰ'),
    'ɭ': ('ɭ', ''),
    'p̚t͈': ('p̚', 't͈'),
    'd͡ʑ': ('', 'd͡ʑ'),
    't͡ɕʰ': ('', 't͡ɕʰ'),
    'ks͈': ('k', 's͈'),
    'ɾɦ': ('ɾ', 'ɦ'),
    'p': ('', 'p'),
    'mt͈': ('m', 't͈'),
    'md͡ʑ': ('m', 'd͡ʑ'),
    'p̚k͈': ('p̚', 'k͈'),
    'k̚t͡ɕ͈': ('k̚', 't͡ɕ͈'),
    'mm': ('m', 'm'),
    'nt͡ɕʰ': ('n', 't͡ɕʰ'),
    'mɦ': ('m', 'ɦ'),
    'mb': ('m', 'b'),
    'k̚t͡ɕʰ': ('k̚', 't͡ɕʰ'),
    'ɭn': ('ɭ', 'n'),
    'mp͈': ('m', 'p͈'),
    'ŋ': ('ŋ', ''),
    'nt͡ɕ͈': ('n', 't͡ɕ͈'),
    'p̚t͡ɕ͈': ('p̚', 't͡ɕ͈'),
    'nsʰ': ('n', 'sʰ'),
    'ŋɡ': ('ŋ', 'ɡ'),
    'ɭp͈': ('ɭ', 'p͈'),
    't̚tʰ': ('t̚', 'tʰ'),
    'ŋpʰ': ('ŋ', 'pʰ'),
    'ɭt͡ɕ͈': ('ɭ', 't͡ɕ͈'),
    'nd': ('n', 'd'),
    'ŋsʰ': ('ŋ', 'sʰ'),
    'ɭt͈': ('ɭ', 't͈'),
    'p̚tʰ': ('p̚', 'tʰ'),
    'ŋk͈': ('ŋ', 'k͈'),
    'mpʰ': ('m', 'pʰ'),
    'ɭm': ('ɭ', 'm'),
    'npʰ': ('n', 'pʰ'),
    's͈': ('', 's͈'),
    'pʰ': ('', 'pʰ'),
    'k̚kʰ': ('k̚', 'kʰ'),
    'mn': ('m', 'n'),
    'nt͈': ('n', 't͈'),
    'mt͡ɕʰ': ('m', 't͡ɕʰ'),
    't̚s͈': ('t̚', 's͈'),
    'k̚k͈': ('k̚', 'k͈'),
    'ps͈': ('p', 's͈'),
    'ɭtʰ': ('ɭ', 'tʰ'),
    'mkʰ': ('m', 'kʰ'),
    'ŋs͈': ('ŋ', 's͈'),
    't̚kʰ': ('t̚', 'kʰ'),
    'ɭd': ('ɭ', 'd'),
    'tʰ': ('', 'tʰ'),
    'nkʰ': ('n', 'kʰ'),
    'n': ('', 'n'),
    'ɭd͡ʑ': ('ɭ', 'd͡ʑ'),
    'msʰ': ('m', 'sʰ'),
    'd': ('', 'd'),
    # custom by dataset
    'ɭʎ': ('ɭ', 'ʎ'),
    'ɾɣ': ('ɾ', 'ɣ'),
    'mɕʰ': ('m', 'ɕʰ'),
    'ŋɕʰ': ('ŋ', 'ɕʰ'),
    'ŋɣ': ('ŋ', 'ɣ'),
    'ŋɲ': ('ŋ', 'ɲ'),
    'ŋβ': ('ŋ', 'β'),
    'ɭɕʰ': ('ɭ', 'ɕʰ'),
    'ɭɕ͈': ('ɭ', 'ɕ͈'),
    'kç': ('k', 'ç'),
    'kɕ͈': ('k', 'ɕ͈'),
    'nɕʰ': ('n', 'nɕʰ'),
    'nɣ': ('n', 'ɣ'),
    'nɲ': ('n', 'ɲ'),
    'nʃʰ': ('n', 'ʃʰ'),
    'nʝ': ('n', 'ʝ'),
    'nβ': ('n', 'β'),
    'pɕ͈': ('p', 'ɕ͈'),
    'mɲ': ('m', 'ɲ'),
    'nɕʰ': ('n', 'ɕʰ'),
    'p̚k': ('p̚', 'k')
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
    for phoneme in pronunciation:
        if phoneme in korean_vowels:
            if len(syllables) > 0 and len(current_consonants) > 1 and current_consonants[0] != 'ŋ':
                syllables[-1][2].append(current_consonants[0])
                current_consonants = current_consonants[1:]            
            syllables.append((current_consonants, phoneme, []))
            current_consonants = []
        else:   # consonant
            if phoneme in korean_boundary_dict.keys():
                current_consonants.extend([x for x in korean_boundary_dict[phoneme] if x != ''])
            else:
                current_consonants.append(phoneme)

    if len(current_consonants) > 0:
        syllables[-1][2].extend(current_consonants)

    return syllables


# Chinese
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
                ]).
                replace('˧', '').replace('˩', '').replace('˥', '').
                replace('tɕ', 't͡ɕ').replace('tɕʰ', 't͡ɕʰ').replace('ts', 't͡s').
                replace('tsʰ', 't͡sʰ').replace('tʂ', 't͡ʂ').replace('tʂʰ', 't͡ʂʰ').
                replace('ʈʂ', 'ʈ͡ʂ').replace('ʈʂʰ', 'ʈ͡ʂʰ')
            for character in text
            ])
        for text in texts        
        ]

chinese_diphthongs = ['ai̯','au̯','ou̯','ei̯']
chinese_monophthongs = ['ɛ','a','ʊ','i','ə','ɤ','o','ɔ','u','y','e']
chinese_syllablc_consonants = ['ɻ̩', 'ɹ̩', 'ɚ', 'n']  # 'for 嗯(n)'
chinese_onsets = sorted([
    't͡ɕʰ','t͡sʰ','t͡ʂʰ','ʈ͡ʂʰ','t͡ɕ','t͡s','t͡ʂ','ʈ͡ʂ',
    'kʰ','pʰ','tʰ','ʐ̩','ɕ','f','j','k','l','m','n',
    'p','ʐ','s','ʂ','t','w','x','ɥ','h','ʂ','ʐ','ɻ'
    ], key= lambda x: len(x), reverse= True)
chinese_codas = ['n','ŋ']

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



# Japanese
def remove_initial_consonant(syllable):
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
        'ぱ': 'あ', 'ぴ': 'い', 'ぷ': 'う', 'ぺ': 'え', 'ぽ': 'お'
    }
    
    return vowel_map.get(syllable, syllable)

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
        'わ'  : [['ɰ'], 'a', []],                                                                               'を'  : [[    ], 'o', []],

                                                            'ゔ'  : [['v',], 'u', []], 
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
        }
    
    if syllable in ipa_dict.keys():
        return ipa_dict[syllable]
    elif len(syllable) > 1 and syllable[-1] == 'っ':
        ipa = _Japanese_Syllable_to_IPA(syllable[:-1], 'っ')
        next_onset = ipa_dict[next_syllable[0]][0] if not next_syllable is None else []
        if len(next_onset) == 0:
            next_onset.append('ʔ')
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
        else: # syllable[1] in ['ゕ', 'ゖ', 'ゎ', 'ゃ', 'ゅ', 'ょ']
            raise NotImplementedError(f'unimplemented syllable: {syllable}')        
        return ipa    
    elif len(syllable) == 3 and syllable[2] == 'ん':
        ipa = _Japanese_Syllable_to_IPA(syllable[:2], syllable[2])
        _, _, ipa[2] = _Japanese_Syllable_to_IPA(syllable[2], next_syllable)
        return ipa
    
    if syllable == 'ん':    # single 'ん'
        if next_syllable is None:
            return [[], '', ['ɴ']]
        elif next_syllable[0] in [
            'あ', 'い', 'う', 'え', 'お',
            'な', 'に', 'ぬ', 'ね', 'の',
            'ら', 'り', 'る', 'れ', 'ろ',
            'ざ', 'じ', 'ず', 'ぜ', 'ぞ',
            'だ', 'ぢ', 'づ', 'で', 'ど'
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
            return [[], '', ['ŋ']]
        elif next_syllable[0] in [
            'や', 'ゆ', 'よ', 'わ', 'を'
            ]:
            return [[], '', ['ɰ̃']]
        elif next_syllable[0] in [
            'さ', 'し', 'す', 'せ', 'そ',
            'た', 'ち', 'つ', 'て', 'と'
            ]:
            return [[], '', ['n̪']]
        else:
            raise ValueError(f'Unsupported next syllable: {next_syllable}')

    raise NotImplementedError(f'unimplemented syllable: {syllable}, {next_syllable}')

def _Japanese_Phonemize(texts: Union[str, List[str]]):
    if type(texts) == str:
        texts = [texts]

    pronunciations = []

    indices = range(0, len(texts), chunk)
    for index in indices:
        
        pronunciations.extend(pronunciations_chunk)

    return pronunciations

def _Japanese_Split_Phoneme(syllable: str):
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

def _Japanese_Syllablize(pronunciation: str):
    pronunciation = [
        _Japanese_Split_Phoneme(syllable= x)
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





class Language(Enum):
    Korean= 0
    English= 1
    Chinese= 2
    Japanese= 3

def Phonemize(texts: Union[str, List[str]], language: Language):
    if language == Language.Korean:
        return _Korean_Phonemize(texts)
    elif language == Language.English:
        return _English_Phonemize(texts)
    elif language == Language.Chinese:
        return _Chinese_Phonemize(texts)
    elif language == Language.Japanese:
        return _Japanese_Phonemize(texts)
    else:
        raise NotImplementedError(f'Unsupported language: {language}')