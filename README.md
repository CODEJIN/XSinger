# X-Singer: Code-Mixed Singing Voice Synthesis via Cross-Lingual Learning

* This repository is an unofficial implementation of X-Singer.
* It is based on the following paper:
```
Hwang, J. S., Noh, H., Hong, Y., Oh, I., & Center, N. A. (2024). X-Singer: Code-Mixed Singing Voice Synthesis via Cross-Lingual Learning. In Proc. Interspeech 2024 (pp. 1885-1889).
```

# Structure
* The structure is derived from X-Singer, with several modifications:
* The original paper uses relative multi-head attention. In this implementation, positional encoding has been replaced with RoPE (Rotary Position Embedding).
* The guided attention loss for the cross-attention mechanism has been modified:
    * The original guided attention is calculated using the total lyric and mel length.
    * In Singing Voice Synthesis (SVS), some notes have long durations, which can mislead the guided attention mechanism.
    * In this repository, guided attention is calculated based on note duration.
* A token predictor is added to improve pronunciation.
    * Note: The effectiveness of this addition has not been verified.

# Supported dataset
* Korean
    * [Multi-speaker Singing Data in AIHub](https://aihub.or.kr/aihubdata/data/view.do?currMenu=115&topMenu=100&dataSetSn=465)
    * [Diverse Timbres Guided Vocal data in AIHub](https://aihub.or.kr/aihubdata/data/view.do?currMenu=115&topMenu=100&dataSetSn=473)
* Japanese
    * [Tohoku Kiritan singing database](https://zunko.jp/kiridev/login.php)
    * [Ofuton P Utagoe DB](https://sites.google.com/view/oftn-utagoedb/)
* Chinese
    * [M4Singer](https://github.com/M4Singer/M4Singer)

# Generate pattern

```
python Pattern_Generate.py [parameters]
```
## Parameters
* `amz`: Path to Multi-Speaker Singing Data in AIHub.
* `amb`: Path to Diverse Timbres Guided Vocal Data in AIHub.
* `m4`: Path to M4Singer.
* `kiritan`: Path to Tohoku Kiritan Singing Database.
* `ofuton`: Path to Ofuton P Utagoe DB.
* `hp`: Path to hyperparameter file.
* `verbose`: Enable detailed logging if set.

## About phonemizer
* This repository uses the Phonemizer library to generate phoneme strings.
    * Refer to the [installation guide](https://bootphon.github.io/phonemizer/install.html) for Phonemizer and its backends.
* For Windows users, additional setup is required:
    * Refer to this issue for details.
    * In a Conda environment, the following commands might be helpful:
        * Please refer [here](https://github.com/bootphon/phonemizer/issues/44)    
        ```bash
        conda env config vars set PHONEMIZER_ESPEAK_PATH='C:\Program Files\eSpeak NG'
        conda env config vars set PHONEMIZER_ESPEAK_LIBRARY='C:\Program Files\eSpeak NG\libespeak-ng.dll'
        ```

# Vocoder
* To simplify usage, NVIDIA's BigVGAN has been converted into a TorchScript module.
* Download the TorchScript model [here](https://drive.google.com/file/d/1j89dmAcMyld40pn747yHcfLEZY8zI2iD/view?usp=sharing)

# Running the code

## Train
* Run the following command to train the model:
```
accelerate launch Train.py -hp Hyper_Parameters.yaml
```