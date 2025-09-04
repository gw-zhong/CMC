import numpy as np
from torch.utils.data.dataset import Dataset
import pickle
import os
import torch


############################################################################################
# This file provides basic processing script for the multimodal datasets we use. For other
# datasets, small modifications may be needed (depending on the type of the data, etc.)
############################################################################################


class MMDataset(Dataset):
    def __init__(self, args, mode='train', split_mode=''):
        self.mode = mode
        self.split_mode = split_mode
        self.args = args
        DATA_MAP = {
            'SIMS': self.__init_sims,
            'SIMS-v2': self.__init_simsv2,
            'MOSI': self.__init_mosi,
            'MOSEI': self.__init_mosei
        }
        DATA_MAP[args.dataset]()

    def __init_sims(self):
        if self.args.dataset == 'SIMS':
            if self.split_mode != '':
                path = os.path.join(self.args.data_path, 'CH-' + self.args.dataset, 'Processed', f'[{self.mode}]{self.split_mode}.pkl')
            else:
                path = os.path.join(self.args.data_path, 'CH-' + self.args.dataset, 'Processed', 'unaligned.pkl')
        elif self.args.dataset == 'SIMS-v2':
            if self.split_mode != '':
                path = os.path.join(self.args.data_path, 'CH-' + self.args.dataset, 'CH-SIMS-v2(s)', 'Processed', f'[{self.mode}]{self.split_mode}.pkl')
            else:
                path = os.path.join(self.args.data_path, 'CH-' + self.args.dataset, 'CH-SIMS-v2(s)', 'Processed', 'unaligned.pkl')
        elif self.args.dataset == 'MOSI' or self.args.dataset == 'MOSEI':
            path = os.path.join(self.args.data_path, 'CMU-' + self.args.dataset, 'Processed', 'unaligned_50.pkl')
        with open(path, 'rb') as f:
            data = pickle.load(f)
        self.text = data[self.mode]['text_bert'].astype(np.float32)
        self.vision = data[self.mode]['vision'].astype(np.float32)
        self.audio = data[self.mode]['audio'].astype(np.float32)
        self.rawText = data[self.mode]['raw_text']
        self.ids = data[self.mode]['id']

        if 'SIMS' in self.args.dataset:
            classification_labels = {
                'M': np.where(data[self.mode]['regression_labels'] < 0, 0, np.where(data[self.mode]['regression_labels'] == 0, 1, 2)),
                'T': np.where(data[self.mode]['regression_labels_T'] < 0, 0, np.where(data[self.mode]['regression_labels_T'] == 0, 1, 2)),
                'A': np.where(data[self.mode]['regression_labels_A'] < 0, 0, np.where(data[self.mode]['regression_labels_A'] == 0, 1, 2)),
                'V': np.where(data[self.mode]['regression_labels_V'] < 0, 0, np.where(data[self.mode]['regression_labels_V'] == 0, 1, 2)),
            }
        else:  # MOSI and MOSEI don't have unimodal labels
            classification_labels = {
                'M': np.where(data[self.mode]['regression_labels'] < 0, 0, np.where(data[self.mode]['regression_labels'] == 0, 1, 2)),
                'T': np.where(data[self.mode]['regression_labels'] < 0, 0, np.where(data[self.mode]['regression_labels'] == 0, 1, 2)),
                'A': np.where(data[self.mode]['regression_labels'] < 0, 0, np.where(data[self.mode]['regression_labels'] == 0, 1, 2)),
                'V': np.where(data[self.mode]['regression_labels'] < 0, 0, np.where(data[self.mode]['regression_labels'] == 0, 1, 2)),
            }

        self.labels = {
            'M': classification_labels['M'].astype(np.float32),
            'T': classification_labels['T'].astype(np.float32),
            'A': classification_labels['A'].astype(np.float32),
            'V': classification_labels['V'].astype(np.float32)
        }

        self.text_lengths = np.sum(self.text[:, 1], axis=1).astype(np.int16).tolist()
        self.audio_lengths = data[self.mode]['audio_lengths']
        self.vision_lengths = data[self.mode]['vision_lengths']
        self.audio[self.audio == -np.inf] = 0

    def __init_simsv2(self):
        return self.__init_sims()

    def __init_mosi(self):
        return self.__init_sims()

    def __init_mosei(self):
        return self.__init_sims()

    def __len__(self):
        return len(self.labels['M'])

    def get_seq_len(self):
        return (self.text.shape[2], self.audio.shape[1], self.vision.shape[1])

    def get_feature_dim(self):
        return (768, self.audio.shape[2], self.vision.shape[2])

    def __getitem__(self, index):
        sample = {
            'raw_text': self.rawText[index],
            'text': torch.Tensor(self.text[index]),
            'text_lengths': self.text_lengths[index],
            'audio': torch.Tensor(self.audio[index]),
            'audio_lengths': self.audio_lengths[index],
            'vision': torch.Tensor(self.vision[index]),
            'vision_lengths': self.vision_lengths[index],
            'index': index,
            'id': self.ids[index],
            'labels': {k: torch.Tensor(v[index].reshape(-1)) for k, v in self.labels.items()}
        }
        return sample
