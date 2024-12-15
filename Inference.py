import torch
import logging, yaml, sys, math
import matplotlib.pyplot as plt
from tqdm import tqdm
from accelerate import Accelerator
from typing import List


from Modules.Modules import RectifiedFlowSVS
from Datasets import Inference_Dataset as Dataset, Inference_Collater as Collater
from meldataset import spectral_de_normalize_torch
from Arg_Parser import Recursive_Parse

import matplotlib as mpl
# 유니코드 깨짐현상 해결
mpl.rcParams['axes.unicode_minus'] = False
# 나눔고딕 폰트 적용
plt.rcParams["font.family"] = 'NanumGothic'

logging.basicConfig(
    level=logging.INFO, stream=sys.stdout,
    format= '%(asctime)s (%(module)s:%(lineno)d) %(levelname)s: %(message)s'
    )

class Inferencer:
    def __init__(
        self,
        hp_path: str,
        checkpoint_path: str,
        batch_size= 1
        ):
        self.device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')
        
        self.hp = Recursive_Parse(yaml.load(
            open(hp_path, encoding='utf-8'),
            Loader=yaml.Loader
            ))

        self.accelerator = Accelerator(
            split_batches= True,
            mixed_precision= 'fp16' if self.hp.Use_Mixed_Precision else 'no',   # no, fp16, bf16, fp8
            gradient_accumulation_steps= self.hp.Train.Accumulated_Gradient_Step
            )

        self.model = torch.optim.swa_utils.AveragedModel(RectifiedFlowSVS(self.hp).to(self.device))
        self.model.Inference = self.model.module.Inference
        self.accelerator.prepare(self.model)

        self.vocoder = torch.jit.load('BigVGAN_24K_100band_256x.pts', map_location= 'cpu').to(self.device)

        self.Load_Checkpoint(checkpoint_path)
        self.batch_size = batch_size

    def Dataset_Generate(
        self,
        lyrics: List[List[str]],
        notes: List[List[int]],
        durations: List[List[float]],
        singers: List[str],
        languages: List[str],
        ):
        token_dict = yaml.load(open(self.hp.Token_Path, encoding= 'utf-8-sig'), Loader=yaml.Loader)
        singer_dict = yaml.load(open(self.hp.Singer_Info_Path, 'r', encoding= 'utf-8-sig'), Loader=yaml.Loader)        
        language_dict = yaml.load(open(self.hp.Language_Info_Path, encoding= 'utf-8-sig'), Loader=yaml.Loader)

        dataset = Dataset(
            token_dict= token_dict,
            singer_dict = singer_dict,
            language_dict= language_dict,
            lyrics= lyrics,
            notes= notes,
            durations= durations,
            singers= singers,
            languages= languages,
            sample_rate= self.hp.Sound.Sample_Rate,
            hop_size= self.hp.Sound.Hop_Size,
            )
        collater = Collater(
            token_dict= token_dict
            )

        return torch.utils.data.DataLoader(
            dataset= dataset,
            sampler= torch.utils.data.SequentialSampler(dataset),
            collate_fn= collater,
            batch_size= self.batch_size,
            num_workers= self.hp.Train.Num_Workers,
            pin_memory= True
            )

    def Load_Checkpoint(self, path):
        state_dict = torch.load(f'{path}/pytorch_model_1.bin', map_location= 'cpu')
        self.model.load_state_dict(state_dict)
        
        logging.info(f'Checkpoint loaded from \'{path}\'.')

    @torch.inference_mode()
    def Inference_Step(
        self,
        tokens: torch.IntTensor,
        token_lengths: torch.IntTensor,
        token_on_note_lengths: torch.IntTensor,
        notes: torch.IntTensor,
        durations: torch.IntTensor,
        note_lengths: torch.IntTensor,
        singers: torch.IntTensor,
        languages: torch.IntTensor,
        mel_lengths: torch.IntTensor
        ):
        prediction_mels, _, _ = self.model.Inference(
            tokens= tokens,
            languages= languages,
            token_lengths= token_lengths,
            token_on_note_lengths= token_on_note_lengths,
            notes= notes,
            durations= durations,
            note_lengths= note_lengths,
            singers= singers,
            mel_lengths= mel_lengths
            )
        
        audio_lengths = [
            length * self.hp.Sound.Hop_Size
            for length in mel_lengths
            ]
        prediction_audios = [
            audio[:length]
            for audio, length in zip(self.vocoder(prediction_mels).cpu().numpy(), audio_lengths)
            ]

        return prediction_audios

    def Inference_Epoch(
        self,
        lyrics: List[List[str]],
        notes: List[List[int]],
        durations: List[List[float]],
        singers: List[str],
        languages: List[str],
        use_tqdm= True
        ):
        dataloader = self.Dataset_Generate(
            lyrics= lyrics,
            notes= notes,
            durations= durations,
            singers= singers,
            languages= languages,
            )
        if use_tqdm:
            dataloader = tqdm(
                dataloader,
                desc='[Inference]',
                total= math.ceil(len(dataloader.dataset) / self.batch_size)
                )
        audios = []
        for tokens, token_lengths, token_on_note_lengths, \
            notes, durations, note_lengths, \
            singers, languages, \
            mel_lengths, lyrics in dataloader:
            audios.extend(self.Inference_Step(
                tokens= tokens,
                token_lengths= token_lengths,
                token_on_note_lengths= token_on_note_lengths,
                notes= notes, 
                durations= durations,
                note_lengths= note_lengths,
                singers= singers,
                languages= languages,
                mel_lengths= mel_lengths,
                ))
        
        return audios
    
if __name__ == '__main__':
    inferencer = Inferencer(
        hp_path= './results/Checkpoint/Hyper_Parameters.yaml',
        checkpoint_path= './results/Checkpoint/S_166424',
        batch_size= 4
        )
    
    audios = inferencer.Inference_Epoch(
        lyrics= [
            ['마','음','울','적','한','날','에','<X>','거','리','를','걸','어','보','고','향','기','로','운','칵','테','일','에','취','해','도','보','고','한','편','의','시','가','있','는','<X>','전','시','회','장','도','가','고','밤','새','도','<X>','록','그','리','움','에','편','질','쓰','고','파',],
            ],
        notes= [
            [68,68,68,75,73,72,70,0,72,72,72,73,72,67,67,65,65,65,68,68,66,65,63,65,68,67,68,70,68,68,68,75,73,72,70,0,72,72,72,73,72,67,67,65,65,65,67,68,68,65,63,63,65,68,67,70,68],
            ],
        durations= [
            [0.33,0.16,0.33,0.49,0.33,0.16,0.81,0.33,0.16,0.16,0.33,0.16,0.49,0.16,0.82,0.33,0.16,0.33,0.49,0.33,0.16,0.33,0.49,0.33,0.33,0.16,0.33,1.47,0.33,0.16,0.33,0.49,0.33,0.16,0.81,0.33,0.16,0.16,0.33,0.16,0.49,0.16,0.82,0.33,0.16,0.33,0.16,0.33,0.49,0.16,0.33,0.33,0.33,0.33,0.16,0.33,0.82],
            ],
        singers= [
            'CSD',
            ],
        languages= [
            'Korean',
            ],
        )
    
    from scipy.io import wavfile
    for index, audio in enumerate(audios):
        wavfile.write(
            f'{index}.wav', inferencer.hp.Sound.Sample_Rate, audio
            )