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

        self.Load_Checkpoint(checkpoint_path)
        self.batch_size = batch_size

    def Dataset_Generate(
        self,
        texts: List[str],
        languages: List[str],
        ):
        token_dict = yaml.load(open(self.hp.Token_Path, encoding= 'utf-8-sig'), Loader=yaml.Loader)
        language_dict = yaml.load(open(self.hp.Language_Info_Path, encoding= 'utf-8-sig'), Loader=yaml.Loader)

        dataset = Dataset(
            token_dict= token_dict,
            language_dict= language_dict,
            use_between_padding= self.hp.Use_Between_Padding,
            texts= texts,
            languages= languages,
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
        languages: torch.IntTensor
        ):
        prediction_audios, _, prediction_alignments = self.model.Inference(
            tokens= tokens.to(self.device),
            token_lengths= token_lengths.to(self.device),
            languages= languages.to(self.device),
            )
        
        latent_code_lengths = [
            alignment[:token_length, :].sum().long().item()
            for token_length, alignment in zip(token_lengths, prediction_alignments)
            ]
        audio_lengths = [
            length * self.hp.Sound.Hop_Size
            for length in latent_code_lengths
            ]

        prediction_audios = [
            audio[:length]
            for audio, length in zip(prediction_audios.clamp(-1.0, 1.0).cpu().numpy(), audio_lengths)
            ]

        return prediction_audios

    def Inference_Epoch(
        self,
        texts: List[str],
        languages: List[str],
        use_tqdm= True
        ):
        dataloader = self.Dataset_Generate(
            texts= texts,
            languages= languages
            )
        if use_tqdm:
            dataloader = tqdm(
                dataloader,
                desc='[Inference]',
                total= math.ceil(len(dataloader.dataset) / self.batch_size)
                )
        audios = []
        for tokens, token_lengths, languages, _, _ in dataloader:
            audios.extend(self.Inference_Step(tokens, token_lengths, languages))
        
        return audios
    
if __name__ == '__main__':
    inferencer = Inferencer(
        hp_path= './results/Checkpoint/Hyper_Parameters.yaml',
        checkpoint_path= './results/Checkpoint/S_166424',
        batch_size= 4
        )
    
    audios = inferencer.Inference_Epoch(
        texts= [
            'Printing, in the only sense with which we are at present concerned, differs from most if not from all the arts and crafts represented in the Exhibition',
            'Do not kill the goose that lays the golden eggs.',
            'A good medicine tastes bitter.',
            'Do not count your chickens before they hatch.',
            'If you laugh, blessings will come your way.'
            ],
        languages= [
            'English',
            'English',
            'English',
            'English',
            'English'
            ]
        )
    
    from scipy.io import wavfile
    for index, audio in enumerate(audios):
        wavfile.write(
            f'{index}.wav', inferencer.hp.Sound.Sample_Rate, audio
            )