import os
os.environ['FOR_DISABLE_CONSOLE_CTRL_HANDLER'] = 'T'    # This is ot prevent to be called Fortran Ctrl+C crash in Windows.
import torch
import numpy as np
import logging, yaml, os, sys, argparse, math, wandb, warnings, traceback
from functools import partial
from tqdm import tqdm
from collections import defaultdict
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
from scipy.io import wavfile
from accelerate import Accelerator
from Logger import Logger
from typing import List

from Modules.Modules import RectifiedFlowSVS, Mask_Generate
from Datasets import Dataset, Inference_Dataset, Collater, Inference_Collater
from hificodec.vqvae import VQVAE

from meldataset import mel_spectrogram
from Arg_Parser import Recursive_Parse, To_Non_Recursive_Dict

import matplotlib as mpl
# 유니코드 깨짐현상 해결
mpl.rcParams['axes.unicode_minus'] = False
# 나눔고딕 폰트 적용
plt.rcParams["font.family"] = 'NanumGothic'

logging.basicConfig(
    level=logging.INFO, stream=sys.stdout,
    format= '%(asctime)s (%(module)s:%(lineno)d) %(levelname)s: %(message)s'
    )
warnings.filterwarnings('ignore')

def log_uncaught_exceptions(exc_type, exc_value, exc_traceback):
    with open("error_log.txt", "a") as log_file:
        log_file.write("Uncaught exception:\n")
        traceback.print_exception(exc_type, exc_value, exc_traceback, file=log_file)

# sys.excepthook = log_uncaught_exceptions

# torch.autograd.set_detect_anomaly(True)

class Trainer:
    def __init__(self, hp_path, steps= 0):
        self.hp_path = hp_path
        self.num_gpus = int(os.getenv("WORLD_SIZE", '1'))
        
        self.hp = Recursive_Parse(yaml.load(
            open(self.hp_path, encoding='utf-8'),
            Loader=yaml.Loader
            ))
        self.hificodec = VQVAE(
            config_path= './hificodec/config_24k_320d.json',
            ckpt_path= './hificodec/HiFi-Codec-24k-320d',
            with_encoder= True
            )

        self.accelerator = Accelerator(
            split_batches= True,
            mixed_precision= 'fp16' if self.hp.Use_Mixed_Precision else 'no',   # no, fp16, bf16, fp8
            gradient_accumulation_steps= self.hp.Train.Accumulated_Gradient_Step
            )

        if not torch.cuda.is_available():
            self.device = torch.device('cpu')
        else:
            self.device = self.accelerator.device
            torch.backends.cudnn.enabled = True
            torch.backends.cudnn.benchmark = False
        
        self.steps = steps

        self.Dataset_Generate()
        self.Model_Generate()
        
        self.dataloader_dict['Train'], self.dataloader_dict['Eval'], self.dataloader_dict['Inference'], \
        self.model_dict['RectifiedFlowSVS'], self.model_dict['RectifiedFlowSVS_EMA'], \
        self.optimizer_dict['RectifiedFlowSVS'], self.scheduler_dict['RectifiedFlowSVS'] = self.accelerator.prepare(
            self.dataloader_dict['Train'],
            self.dataloader_dict['Eval'],
            self.dataloader_dict['Inference'],
            self.model_dict['RectifiedFlowSVS'],
            self.model_dict['RectifiedFlowSVS_EMA'],
            self.optimizer_dict['RectifiedFlowSVS'],
            self.scheduler_dict['RectifiedFlowSVS'],
            )
        
        self.Load_Checkpoint()

        self.scalar_dict = {
            'Train': defaultdict(float),
            'Evaluation': defaultdict(float),
            }

        if self.accelerator.is_main_process:
            self.writer_dict = {
                'Train': Logger(os.path.join(self.hp.Log_Path, 'Train')),
                'Evaluation': Logger(os.path.join(self.hp.Log_Path, 'Evaluation')),
                }
            
            if self.hp.Weights_and_Biases.Use:
                wandb.init(
                    project= self.hp.Weights_and_Biases.Project,
                    entity= self.hp.Weights_and_Biases.Entity,
                    name= self.hp.Weights_and_Biases.Name,
                    config= To_Non_Recursive_Dict(self.hp)
                    )
                wandb.watch(self.model_dict['RectifiedFlowSVS'])

    def Dataset_Generate(self):
        token_dict = yaml.load(open(self.hp.Token_Path, encoding= 'utf-8-sig'), Loader=yaml.Loader)
        singer_dict = yaml.load(open(self.hp.Singer_Info_Path, 'r', encoding= 'utf-8-sig'), Loader=yaml.Loader)
        language_dict = yaml.load(open(self.hp.Language_Info_Path, 'r', encoding= 'utf-8-sig'), Loader=yaml.Loader)
        
        train_dataset = Dataset(
            token_dict= token_dict,
            singer_dict = singer_dict,
            language_dict= language_dict,
            pattern_path= self.hp.Train.Train_Pattern.Path,
            metadata_file= self.hp.Train.Train_Pattern.Metadata_File,
            accumulated_dataset_epoch= self.hp.Train.Train_Pattern.Accumulated_Dataset_Epoch,
            augmentation_ratio= self.hp.Train.Train_Pattern.Augmentation_Ratio,
            use_pattern_cache= self.hp.Train.Pattern_Cache
            )
        eval_dataset = Dataset(
            token_dict= token_dict,
            singer_dict = singer_dict,
            language_dict= language_dict,
            pattern_path= self.hp.Train.Eval_Pattern.Path,
            metadata_file= self.hp.Train.Eval_Pattern.Metadata_File,
            use_pattern_cache= self.hp.Train.Pattern_Cache,
            accumulated_dataset_epoch= self.hp.Train.Eval_Pattern.Accumulated_Dataset_Epoch,
            )
        inference_dataset = Inference_Dataset(
            token_dict= token_dict,
            singer_dict = singer_dict,
            language_dict= language_dict,
            lyrics= self.hp.Train.Inference_in_Train.Lyric,
            notes= self.hp.Train.Inference_in_Train.Note,
            durations= self.hp.Train.Inference_in_Train.Duration,
            singers= self.hp.Train.Inference_in_Train.Singer,
            languages= self.hp.Train.Inference_in_Train.Language,
            sample_rate= self.hp.Sound.Sample_Rate,
            hop_size= self.hp.Sound.Hop_Size,
            )

        if self.accelerator.is_main_process:
            logging.info('The number of train patterns = {}.'.format(len(train_dataset) // self.hp.Train.Train_Pattern.Accumulated_Dataset_Epoch))
            logging.info('The number of development patterns = {}.'.format(len(eval_dataset)))
            logging.info('The number of inference patterns = {}.'.format(len(inference_dataset)))

        collater = Collater(
            token_dict= token_dict,
            pattern_length= self.hp.Train.Pattern_Length,
            hop_size= self.hp.Sound.Hop_Size
            )
        inference_collater = Inference_Collater(
            token_dict= token_dict
            )

        self.dataloader_dict = {}
        self.dataloader_dict['Train'] = torch.utils.data.DataLoader(
            dataset= train_dataset,
            sampler= torch.utils.data.RandomSampler(train_dataset),
            collate_fn= collater,
            batch_size= self.hp.Train.Batch_Size,
            num_workers= self.hp.Train.Num_Workers,
            pin_memory= True
            )
        self.dataloader_dict['Eval'] = torch.utils.data.DataLoader(
            dataset= eval_dataset,
            sampler= torch.utils.data.RandomSampler(eval_dataset),
            collate_fn= collater,
            batch_size= self.hp.Train.Batch_Size,
            num_workers= self.hp.Train.Num_Workers,
            pin_memory= True
            )
        self.dataloader_dict['Inference'] = torch.utils.data.DataLoader(
            dataset= inference_dataset,
            sampler= torch.utils.data.SequentialSampler(inference_dataset),
            collate_fn= inference_collater,
            batch_size= self.hp.Inference_Batch_Size or self.hp.Train.Batch_Size,
            num_workers= self.hp.Train.Num_Workers,
            pin_memory= True
            )

    def Model_Generate(self):
        self.model_dict = {
            'RectifiedFlowSVS': RectifiedFlowSVS(self.hp, self.hificodec).to(self.device)
            }
        self.model_dict['RectifiedFlowSVS_EMA'] = torch.optim.swa_utils.AveragedModel(
            self.model_dict['RectifiedFlowSVS'],
            multi_avg_fn= torch.optim.swa_utils.get_ema_multi_avg_fn(0.9999)
            )
        self.model_dict['RectifiedFlowSVS_EMA'].Inference = self.model_dict['RectifiedFlowSVS_EMA'].module.Inference

        self.criterion_dict = {
            'MSE': torch.nn.MSELoss().to(self.device),
            'CE': torch.nn.CrossEntropyLoss().to(self.device)
            }

        self.optimizer_dict = {
            'RectifiedFlowSVS': torch.optim.NAdam(
                params= self.model_dict['RectifiedFlowSVS'].parameters(),
                lr= self.hp.Train.Learning_Rate.Initial,
                betas=(self.hp.Train.ADAM.Beta1, self.hp.Train.ADAM.Beta2),
                eps= self.hp.Train.ADAM.Epsilon,
                weight_decay= self.hp.Train.Weight_Decay
                )
            }
        self.scheduler_dict = {
            'RectifiedFlowSVS': torch.optim.lr_scheduler.ExponentialLR(
                optimizer= self.optimizer_dict['RectifiedFlowSVS'],
                gamma= self.hp.Train.Learning_Rate.Decay,
                last_epoch= -1
                )
            }
        
        self.mel_func = partial(
            mel_spectrogram,
            n_fft= 2048,
            num_mels= 80,
            sampling_rate= self.hp.Sound.Sample_Rate,
            hop_size= self.hp.Sound.Hop_Size,
            win_size= self.hp.Sound.Hop_Size * 4,
            fmin= 0,
            fmax= None
            )

        # if self.accelerator.is_main_process:
        #     logging.info(self.model_dict['RectifiedFlowSVS'])

    def Train_Step(
        self,
        tokens: torch.IntTensor,
        notes: torch.IntTensor,
        lyric_durations: torch.IntTensor,
        note_durations: torch.IntTensor,
        lengths: torch.IntTensor,
        singers: torch.IntTensor,
        languages: torch.IntTensor,
        f0s: torch.FloatTensor,
        latent_codes: torch.IntTensor
        ):
        loss_dict = {}

        with self.accelerator.accumulate(self.model_dict['RectifiedFlowSVS']):
            flows, prediction_flows, prediction_f0s, prediction_singers = self.model_dict['RectifiedFlowSVS'](
                tokens= tokens,
                notes= notes,
                lyric_durations= lyric_durations,
                note_durations= note_durations,
                lengths= lengths,
                singers= singers,
                languages= languages,
                f0s= f0s,
                latent_codes= latent_codes
                )
            
            loss_dict['Diffusion'] = self.criterion_dict['MSE'](
                prediction_flows,
                flows,
                )
            loss_dict['F0'] = self.criterion_dict['MSE'](
                (prediction_f0s + 1).log(),
                (f0s + 1).log(),
                )
            loss_dict['Singer'] = self.criterion_dict['CE'](
                input= prediction_singers,
                target= singers.long(),
                )

            self.optimizer_dict['RectifiedFlowSVS'].zero_grad()
            self.accelerator.backward(
                loss_dict['Diffusion'] +
                loss_dict['F0'] # +
                # loss_dict['Singer']
                )

            if self.hp.Train.Gradient_Norm > 0.0:
                self.accelerator.clip_grad_norm_(
                    parameters= self.model_dict['RectifiedFlowSVS'].parameters(),
                    max_norm= self.hp.Train.Gradient_Norm
                    )
                
            self.optimizer_dict['RectifiedFlowSVS'].step()            
            self.model_dict['RectifiedFlowSVS_EMA'].update_parameters(self.model_dict['RectifiedFlowSVS'])

            self.steps += 1
            self.tqdm.update(1)

        for tag, loss in loss_dict.items():
            self.scalar_dict['Train']['Loss/{}'.format(tag)] += loss.item()

    def Train_Epoch(self):
        for tokens, notes, lyric_durations, note_durations, lengths, singers, languages, f0s, latent_codes in self.dataloader_dict['Train']:
            self.Train_Step(
                tokens= tokens,
                notes= notes,
                lyric_durations= lyric_durations,
                note_durations= note_durations,
                lengths= lengths,
                singers= singers,
                languages= languages,
                f0s= f0s,
                latent_codes= latent_codes
                )

            if self.steps % (math.ceil(len(self.dataloader_dict['Train'].dataset) / self.hp.Train.Train_Pattern.Accumulated_Dataset_Epoch / self.hp.Train.Batch_Size) * self.hp.Train.Learning_Rate.Decay_Epoch) == 0:
                self.scheduler_dict['RectifiedFlowSVS'].step()

            if self.steps % self.hp.Train.Checkpoint_Save_Interval == 0:
                self.Save_Checkpoint()

            if self.steps % self.hp.Train.Logging_Interval == 0 and self.accelerator.is_main_process:
                self.scalar_dict['Train'] = {
                    tag: loss / self.hp.Train.Logging_Interval
                    for tag, loss in self.scalar_dict['Train'].items()
                    }
                self.scalar_dict['Train']['Learning_Rate'] = self.scheduler_dict['RectifiedFlowSVS'].get_last_lr()[0]
                self.writer_dict['Train'].add_scalar_dict(self.scalar_dict['Train'], self.steps)
                if self.hp.Weights_and_Biases.Use:
                    wandb.log(
                        data= {
                            f'Train.{key}': value
                            for key, value in self.scalar_dict['Train'].items()
                            },
                        step= self.steps,
                        commit= self.steps % self.hp.Train.Evaluation_Interval != 0
                        )
                self.scalar_dict['Train'] = defaultdict(float)

            if self.steps % self.hp.Train.Evaluation_Interval == 0:
                self.Evaluation_Epoch()

            if self.steps % self.hp.Train.Inference_Interval == 0:
                self.Inference_Epoch()
            
            if self.steps >= self.hp.Train.Max_Step:
                return

    @torch.no_grad()
    def Evaluation_Step(
        self,
        tokens: torch.IntTensor,
        notes: torch.IntTensor,
        lyric_durations: torch.IntTensor,
        note_durations: torch.IntTensor,
        lengths: torch.IntTensor,
        singers: torch.IntTensor,
        languages: torch.IntTensor,
        f0s: torch.FloatTensor,
        latent_codes: torch.IntTensor
        ):
        loss_dict = {}

        flows, prediction_flows, prediction_f0s, prediction_singers = self.model_dict['RectifiedFlowSVS_EMA'](
            tokens= tokens,
            notes= notes,
            lyric_durations= lyric_durations,
            note_durations= note_durations,
            lengths= lengths,
            singers= singers,
            languages= languages,
            f0s= f0s,
            latent_codes= latent_codes
            )
        
        loss_dict['Diffusion'] = self.criterion_dict['MSE'](
            prediction_flows,
            flows,
            )
        loss_dict['F0'] = self.criterion_dict['MSE'](
            (prediction_f0s + 1).log(),
            (f0s + 1).log(),
            )
        loss_dict['Singer'] = self.criterion_dict['CE'](
            input= prediction_singers,
            target= singers.long(),
            )

        for tag, loss in loss_dict.items():
            self.scalar_dict['Evaluation']['Loss/{}'.format(tag)] += loss.item()

    def Evaluation_Epoch(self):
        logging.info('(Steps: {}) Start evaluation.'.format(self.steps))

        for model in self.model_dict.values():
            model.eval()

        for step, (tokens, notes, lyric_durations, note_durations, lengths, singers, languages, f0s, latent_codes) in tqdm(
            enumerate(self.dataloader_dict['Eval'], 1),
            desc='[Evaluation]',
            total= math.ceil(len(self.dataloader_dict['Eval'].dataset) / self.hp.Train.Batch_Size / self.num_gpus)
            ):
            alignments = self.Evaluation_Step(
                tokens= tokens,
                notes= notes,
                lyric_durations= lyric_durations,
                note_durations= note_durations,
                lengths= lengths,
                singers= singers,
                languages= languages,
                f0s= f0s,
                latent_codes= latent_codes
                )

        if self.accelerator.is_main_process:
            self.scalar_dict['Evaluation'] = {
                tag: loss / step
                for tag, loss in self.scalar_dict['Evaluation'].items()
                }
            self.writer_dict['Evaluation'].add_scalar_dict(self.scalar_dict['Evaluation'], self.steps)
            # self.writer_dict['Evaluation'].add_histogram_model(self.model_dict['RectifiedFlowSVS'], 'RectifiedFlowSVS', self.steps, delete_keywords=[])
        
            index = np.random.randint(0, tokens.size(0))

            with torch.inference_mode():
                if self.num_gpus > 1:
                    target_audios = self.model_dict['RectifiedFlowSVS_EMA'].module.module.hificodec(latent_codes[index, None].mT)[:, 0]
                    inference_func = self.model_dict['RectifiedFlowSVS_EMA'].module.module.Inference
                else:
                    target_audios = self.model_dict['RectifiedFlowSVS_EMA'].module.hificodec(latent_codes[index, None].mT)[:, 0]
                    inference_func = self.model_dict['RectifiedFlowSVS_EMA'].Inference

                prediction_audios, prediction_f0s = inference_func(
                    tokens= tokens[index, None],
                    notes= notes[index, None],
                    lyric_durations= lyric_durations[index, None],
                    note_durations= note_durations[index, None],                    
                    lengths= lengths[index, None],
                    singers= singers[index, None],
                    languages= languages[index, None],
                    )

            length = lengths[index].item()
            audio_length = length * self.hp.Sound.Hop_Size
            
            target_audio = target_audios[0, :audio_length].float().clamp(-1.0, 1.0)
            prediction_audio = prediction_audios[0, :audio_length].float().clamp(-1.0, 1.0)

            target_mel = self.mel_func(target_audio[None])[0].cpu().numpy()
            prediction_mel = self.mel_func(prediction_audio[None])[0].cpu().numpy()

            target_audio = target_audio.cpu().numpy()
            prediction_audio = prediction_audio.cpu().numpy()

            target_f0 = f0s[index, :length].cpu().numpy() 
            prediction_f0 = prediction_f0s[0, :length].cpu().numpy()

            image_dict = {
                'Mel/Target': (target_mel, None, 'auto', None, None, None),
                'Mel/Prediction': (prediction_mel, None, 'auto', None, None, None),
                'F0/Target': (target_f0, None, 'auto', None, None, None),
                'F0/Prediction': (prediction_f0, None, 'auto', None, None, None),
                }
            audio_dict = {
                'Audio/Target': (target_audio, self.hp.Sound.Sample_Rate),
                'Audio/Linear': (prediction_audio, self.hp.Sound.Sample_Rate),
                }

            self.writer_dict['Evaluation'].add_image_dict(image_dict, self.steps)
            self.writer_dict['Evaluation'].add_audio_dict(audio_dict, self.steps)

            if self.hp.Weights_and_Biases.Use:
                wandb.log(
                    data= {
                        f'Evaluation.{key}': value
                        for key, value in self.scalar_dict['Evaluation'].items()
                        },
                    step= self.steps,
                    commit= False
                    )
                wandb.log(
                    data= {
                        'Evaluation.Mel.Target': wandb.Image(target_mel),
                        'Evaluation.Mel.Prediction': wandb.Image(prediction_mel),
                        'Evaluation.Audio.Target': wandb.Audio(
                            target_audio,
                            sample_rate= self.hp.Sound.Sample_Rate,
                            caption= 'Target_Audio'
                            ),
                        'Evaluation.Audio.Prediction': wandb.Audio(
                            prediction_audio,
                            sample_rate= self.hp.Sound.Sample_Rate,
                            caption= 'Prediction_Audio'
                            ),
                        },
                    step= self.steps,
                    commit= True
                    )

        self.scalar_dict['Evaluation'] = defaultdict(float)

        for model in self.model_dict.values():
            model.train()


    @torch.inference_mode()
    def Inference_Step(
        self,
        tokens: torch.IntTensor,
        notes: torch.IntTensor,
        lyric_durations: torch.IntTensor,
        note_durations: torch.IntTensor,
        lengths: torch.IntTensor,
        singers: torch.IntTensor,
        languages: torch.IntTensor,
        lyrics: List[str],
        start_index= 0,
        tag_step= False
        ):
        if self.num_gpus > 1:
            inference_func = self.model_dict['RectifiedFlowSVS_EMA'].module.module.Inference
        else:
            inference_func = self.model_dict['RectifiedFlowSVS_EMA'].Inference

        prediction_audios, prediction_f0s = inference_func(
            tokens= tokens,
            notes= notes,
            lyric_durations= lyric_durations,
            note_durations= note_durations,
            lengths= lengths,
            singers= singers,
            languages= languages,
            )
        audio_lengths = [
            length * self.hp.Sound.Hop_Size
            for length in lengths
            ]

        prediction_mels = [
            mel[:, :length]
            for mel, length in zip(self.mel_func(prediction_audios).cpu().numpy(), lengths)
            ]
        prediction_audios = [
            audio[:length]
            for audio, length in zip(prediction_audios.cpu().numpy(), audio_lengths)
            ]
        
        prediction_f0s = [
            f0[:length]
            for f0, length in zip(prediction_f0s.cpu().numpy(), lengths)
            ]

        files = []
        for index in range(tokens.size(0)):
            tags = []
            if tag_step: tags.append('Step-{}'.format(self.steps))
            tags.append('IDX_{}'.format(index + start_index))
            files.append('.'.join(tags))

        os.makedirs(os.path.join(self.hp.Inference_Path, 'Step-{}'.format(self.steps), 'PNG').replace('\\', '/'), exist_ok= True)
        os.makedirs(os.path.join(self.hp.Inference_Path, 'Step-{}'.format(self.steps), 'WAV').replace('\\', '/'), exist_ok= True)
        for index, (
            mel,
            audio,
            f0,
            lyric,
            file
            ) in enumerate(zip(
            prediction_mels,
            prediction_audios,
            prediction_f0s,
            lyrics,
            files
            )):
            title = 'Lyric: {}'.format(lyric if len(lyric) < 90 else lyric[:90])
            new_figure = plt.figure(figsize=(20, 5 * 3), dpi=100)

            ax = plt.subplot2grid((3, 1), (0, 0))
            plt.imshow(mel, aspect='auto', origin='lower')
            plt.title(f'Prediction Mel  {title}')
            plt.colorbar(ax= ax)
            ax = plt.subplot2grid((3, 1), (1, 0))
            plt.plot(f0)
            plt.title('Prediction F0    {}'.format(title))
            plt.margins(x= 0)
            plt.colorbar(ax= ax)
            ax = plt.subplot2grid((3, 1), (2, 0))
            plt.plot(audio)
            plt.title('Prediction Audio    {}'.format(title))
            plt.margins(x= 0)
            plt.colorbar(ax= ax)
            plt.tight_layout()
            plt.savefig(os.path.join(self.hp.Inference_Path, 'Step-{}'.format(self.steps), 'PNG', '{}.png'.format(file)).replace('\\', '/'))
            plt.close(new_figure)
            
            wavfile.write(
                os.path.join(self.hp.Inference_Path, 'Step-{}'.format(self.steps), 'WAV', '{}.wav'.format(file)).replace('\\', '/'),
                self.hp.Sound.Sample_Rate,
                audio
                )
            
    def Inference_Epoch(self):
        if not self.accelerator.is_main_process:
            return
            
        logging.info('(Steps: {}) Start inference.'.format(self.steps))

        for model in self.model_dict.values():
            model.eval()

        batch_size = self.hp.Inference_Batch_Size or self.hp.Train.Batch_Size
        for step, (tokens, notes, lyric_durations, note_durations, lengths, singers, languages, lyrics) in tqdm(
            enumerate(self.dataloader_dict['Inference']),
            desc='[Inference]',
            total= math.ceil(len(self.dataloader_dict['Inference'].dataset) / batch_size)
            ):
            self.Inference_Step(
                tokens= tokens,
                notes= notes,
                lyric_durations= lyric_durations,
                note_durations= note_durations,
                lengths= lengths,
                singers= singers,
                languages= languages,
                lyrics= lyrics, 
                start_index= step * batch_size
                )

        for model in self.model_dict.values():
            model.train()

    def Load_Checkpoint(self):
        if not os.path.exists(self.hp.Checkpoint_Path):
            return
        elif self.steps == 0:
            paths = [
                os.path.join(self.hp.Checkpoint_Path, path).replace('\\', '/')
                for path in os.listdir(self.hp.Checkpoint_Path)
                if path.startswith('S_')
                ]
            if len(paths) > 0:
                path = max(paths, key = os.path.getctime)
                self.steps = int(path.split('_')[1])
            else:
                return  # Initial training
        else:
            path = os.path.join(self.hp.Checkpoint_Path, 'S_{}'.format(self.steps).replace('\\', '/'))

        self.accelerator.load_state(path)

        if self.accelerator.is_main_process:
            logging.info('Checkpoint loaded at {} steps.'.format(self.steps))

    def Save_Checkpoint(self):        
        if not self.accelerator.is_main_process:
            return
        
        os.makedirs(self.hp.Checkpoint_Path, exist_ok= True)
        checkpoint_path = os.path.join(self.hp.Checkpoint_Path, 'S_{}'.format(self.steps).replace('\\', '/'))

        self.accelerator.wait_for_everyone()
        self.accelerator.save_state(checkpoint_path, safe_serialization= False) # sefetensor cannot use because of shared tensor problem
        # self.accelerator.save_state(checkpoint_path)

        logging.info('Checkpoint saved at {} steps.'.format(self.steps))

        if all([
            self.hp.Weights_and_Biases.Use,
            self.hp.Weights_and_Biases.Save_Checkpoint.Use,
            self.steps % self.hp.Weights_and_Biases.Save_Checkpoint.Interval == 0
            ]):
            wandb.save(checkpoint_path)

    def Train(self):
        hp_path = os.path.join(self.hp.Checkpoint_Path, 'Hyper_Parameters.yaml').replace('\\', '/')
        if not os.path.exists(hp_path):
            from shutil import copyfile
            os.makedirs(self.hp.Checkpoint_Path, exist_ok= True)
            copyfile(self.hp_path, hp_path)
        
        if self.steps == 0:
            self.Evaluation_Epoch()

        if self.hp.Train.Initial_Inference:
            self.Inference_Epoch()

        self.tqdm = tqdm(
            initial= self.steps,
            total= self.hp.Train.Max_Step,
            desc='[Training]'
            )

        while self.steps < self.hp.Train.Max_Step:
            try:
                self.Train_Epoch()
            except KeyboardInterrupt:
                self.Save_Checkpoint()
                exit(1)
            
        self.tqdm.close()
        logging.info('Finished training.')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-hp', '--hyper_parameters', required= True, type= str)
    parser.add_argument('-s', '--steps', default= 0, type= int)    
    parser.add_argument('-r', '--local_rank', default= 0, type= int)
    args = parser.parse_args()
    
    hp = Recursive_Parse(yaml.load(
        open(args.hyper_parameters, encoding='utf-8'),
        Loader=yaml.Loader
        ))
    os.environ['CUDA_VISIBLE_DEVICES'] = hp.Device
    
    trainer = Trainer(hp_path= args.hyper_parameters, steps= args.steps)
    trainer.Train()

# accelerate launch Train.py -hp Hyper_Parameters.yaml