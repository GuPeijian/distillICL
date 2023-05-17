"""
process dataset:
    1 sample labeled data and unlabeled data
    2 prepare icl examples and locate each label location for loss calculation
"""
import json
import codecs
import random
from pathlib import Path
from typing import Any, Callable, Dict, List, NewType, Optional, Tuple, Union
import os
import math
from tqdm import tqdm

import copy
from paddle.io import Dataset

class BASEDataset(Dataset):
    def __init__(
        self,
        data_dir,
        mode,
        k=100
    ):
        """data key: sentence, label[0/1]"""
        super().__init__()
        # if mode == 'dev':
        #     mode = 'dev_subsample'
        self.data_dir=data_dir
        data_file = os.path.join(data_dir, mode + '.jsonl')
        self.data = []
        with open(data_file, 'r') as f:
            read_lines = f.readlines()
            for line in read_lines:
                instance = json.loads(line.strip())
                self.data.append(instance)

        self.sampled_data=[]
        # customize your own label map in inheritance
        self.dataset_name=''
        self.label2id = {'negative': 0, 'positive': 1}
        self.label2verb = {'0': 'negative', '1': 'positive'}
        self.id2verb = ['negative', 'positive']

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        return self.data[index]

    def sample_ids(self,ids):
        sampled_data=[]
        for id in ids:
            if id < len(self.label_data):
                sampled_data.append(self.label_data[id])
            else:
                sampled_data.append(self.unlabel_data[id-len(self.label_data)])
        self.sampled_data=sampled_data

    def subsamplebyshot(self, n_shot, N, seed, exclude=None):
        # exclude
        if exclude is not None:
            for ins in exclude:
                self.data.remove(ins)
        # aggregate data by each category
        random.seed(seed)
        data_by_cls = {}
        for i in range(self.__len__()):
            if self.label2id[self.data[i]['label']] not in data_by_cls:
                data_by_cls[self.label2id[self.data[i]['label']]] = []
            data_by_cls[self.label2id[self.data[i]['label']]].append(self.data[i])
        # evenly sample n examples from each category
        data_subsample = []
        #get n-shot subset
        n_shot_subsample_list=[]
        n_shot_subsample=[]
        left_subsample=[]

        for cls in data_by_cls.keys():
            data_subsampled_by_cls = random.sample(data_by_cls[cls], min(N, len(data_by_cls[cls])))
            data_subsample.extend(data_subsampled_by_cls)
            n_shot_subsample_list.append(data_subsampled_by_cls[:n_shot])
            n_shot_subsample.extend(data_subsampled_by_cls[:n_shot])
            left_subsample.extend(data_subsampled_by_cls[n_shot:])
        
        self.n_shot=n_shot

        self.data=data_subsample
        self.label_data_list = n_shot_subsample_list
        self.label_data=n_shot_subsample
        self.unlabel_data=left_subsample


class SST2Dataset(BASEDataset):
    def __init__(
        self,
        data_dir,
        mode,
        k=100
    ):
        """data key: sentence, label[0/1]"""
        super().__init__(data_dir, mode)
        self.dataset_name='sst2'
        self.label2id = {'0': 0, '1': 1}
        self.label2verb = {'0': 'negative', '1': 'positive'}
        self.id2verb = ['negative', 'positive']
    def get_texts(self):
        text_list=[]
        for i in range(len(self.label_data)):
            text_item=[]
            item=self.label_data[i]
            text_item.append(item["sentence"])
            text_list.append(text_item)
        return text_list


class SUBJDataset(BASEDataset):
    def __init__(
        self,
        data_dir,
        mode,
        k=100
    ):
        """data key: sentence, label[0/1]"""
        super().__init__(data_dir, mode)
        # subj only has test set
        self.dataset_name='subj'
        self.label2id = {'0': 0, '1': 1}
        self.label2verb = {'0': 'subjective', '1': 'objective'}
        self.id2verb = ['subjective', 'objective']

    def get_texts(self):
        text_list=[]
        for i in range(len(self.label_data)):
            text_item=[]
            item=self.label_data[i]
            text_item.append(item["sentence"])
            text_list.append(text_item)
        return text_list

class AGNEWSDataset(BASEDataset):
    def __init__(
        self,
        data_dir,
        mode,
        k=100
    ):
        """data key: sentence, label[0/1]"""
        super().__init__(data_dir, mode)
        self.dataset_name='agnews'
        self.label2id = {'1': 0, '2': 1, '3': 2, '4': 3}
        self.label2verb = {'1': 'world', '2': 'sports', '3': 'business', '4': 'technology'}
        self.id2verb = ['world', 'sports', 'business', 'technology']


class CBDataset(BASEDataset):
    def __init__(
        self,
        data_dir,
        mode,
        k=100
    ):
        """data key: sentence, label[0/1]"""
        super().__init__(data_dir, mode)
        self.dataset_name='cb'
        self.label2id = {'contradiction': 0, 'entailment': 1, 'neutral': 2}
        self.label2verb = {'contradiction': 'false', 'entailment': 'true', 'neutral': 'neither'}
        self.id2verb = ['false', 'true', 'neither']


class CRDataset(BASEDataset):
    def __init__(
        self,
        data_dir,
        mode,
        k=100
    ):
        """data key: sentence, label[0/1]"""
        super().__init__(data_dir, mode)
        self.dataset_name='cr'
        self.label2id = {'0': 0, '1': 1}
        self.label2verb = {'0': 'negative', '1': 'positive'}
        self.id2verb = ['negative', 'positive']


class DBPEDIADataset(BASEDataset):
    def __init__(
        self,
        data_dir,
        mode,
        k=100
    ):
        """data key: sentence, label[0/1]"""
        if mode == 'dev':
            mode = 'dev_subsample'
        else:
            mode = 'train_subset'  # this is an exception case
        super().__init__(data_dir, mode)
        self.dataset_name='dbpedia'
        self.label2id = {'1': 0, '2': 1, '3': 2, '4': 3, '5': 4,
                         '6': 5, '7': 6, '8': 7, '9': 8, '10': 9,
                         '11': 10, '12': 11, '13': 12, '14': 13}
        self.label2verb = {'1': 'company', '2': 'school', '3': 'artist', '4': 'athlete', '5': 'politics',
                           '6': 'transportation', '7': 'building', '8': 'nature', '9': 'village', '10': 'animal',
                           '11': 'plant', '12': 'album', '13': 'film', '14': 'book'}
        self.id2verb = ['company', 'school', 'artist', 'athlete', 'politics',
                        'transportation', 'building', 'nature', 'village', 'animal',
                        'plant', 'album', 'film', 'book']


class MPQADataset(BASEDataset):
    def __init__(
        self,
        data_dir,
        mode,
        k=100
    ):
        """data key: sentence, label[0/1]"""
        super().__init__(data_dir, mode)
        self.dataset_name='mpqa'
        self.label2id = {'0': 0, '1': 1}
        self.label2verb = {'0': 'negative', '1': 'positive'}
        self.id2verb = ['negative', 'positive']


class MRDataset(BASEDataset):
    def __init__(
        self,
        data_dir,
        mode,
        k=100
    ):
        """data key: sentence, label[0/1]"""
        super().__init__(data_dir, mode)
        self.dataset_name='mr'
        self.label2id = {'0': 0, '1': 1}
        self.label2verb = {'0': 'negative', '1': 'positive'}
        self.id2verb = ['negative', 'positive']


class RTEDataset(BASEDataset):
    def __init__(
        self,
        data_dir,
        mode,
        k=100
    ):
        """data key: sentence, label[0/1]"""
        super().__init__(data_dir, mode)
        self.dataset_name='rte'
        self.label2id = {'not_entailment': 0, 'entailment': 1}
        self.label2verb = {'not_entailment': 'false', 'entailment': 'true'}
        self.id2verb = ['false', 'true']


class SST5Dataset(BASEDataset):
    def __init__(
        self,
        data_dir,
        mode,
        k=100
    ):
        """data key: sentence, label[0/1]"""
        super().__init__(data_dir, mode)
        self.dataset_name='sst5'
        self.label2id = {'0': 0, '1': 1, '2': 2, '3': 3, '4': 4}
        self.label2verb = {'0': 'terrible', '1': 'bad', '2': 'okay', '3': 'good', '4': 'great'}
        self.id2verb = ['terrible', 'bad', 'okay', 'good', 'great']


class TRECDataset(BASEDataset):
    def __init__(
        self,
        data_dir,
        mode,
        k=100
    ):
        """data key: sentence, label[0/1]"""
        super().__init__(data_dir, mode)
        self.dataset_name='trec'
        self.label2id = {'0': 0, '1': 1, '2': 2, '3': 3, '4': 4, '5': 5}
        self.label2verb = {'0': 'description', '1': 'entity', '2': 'expression', '3': 'human','4': 'location', '5': 'number'}
        self.id2verb = ['description', 'entity', 'expression', 'human', 'location', 'number']