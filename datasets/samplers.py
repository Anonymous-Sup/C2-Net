import os
import torch
import math
import numpy as np
from copy import deepcopy
from torch.utils.data import Sampler

from collections import defaultdict
import copy
import random



class FewshotBatchSampler(Sampler):
    def __init__(self, data_source, way, shot, query_shot, trial=250):
        
        self.data_source = data_source
        self.way = way
        self.shot = shot
        self.query_shot = query_shot
        self.trial = trial

        self.index_dic = defaultdict(list)
        # Create a dictionary mapping each class (pid) to its image indices
        for index, (_, pid, _, _) in enumerate(self.data_source):
            self.index_dic[pid].append(index)
        self.pids = list(self.index_dic.keys())  # List of unique class IDs (pids)

    def __iter__(self):
        index_dic = deepcopy(self.index_dic)
        for _ in range(self.trial):
            
            episode_idxs = []  # Collect indices for this episode
            
            # Randomly select `num_classes` (N) from available pids (with replacement allowed)
            selected_pids = random.sample(self.pids, self.way)
    
            for pid in selected_pids:
                np.random.shuffle(index_dic[pid])

            for pid in selected_pids:
                episode_idxs.extend(index_dic[pid][:self.shot])
            for pid in selected_pids:
                episode_idxs.extend(index_dic[pid][self.shot: self.shot + self.query_shot])

            yield episode_idxs


class ValSampler(Sampler):
    def __init__(self, data_source, way, shot, query_shot, trial=2000):
        
        self.data_source = data_source
        self.way = way
        self.shot = shot
        self.query_shot = query_shot
        self.trial = trial

        self.index_dic = defaultdict(list)
        # Create a dictionary mapping each class (pid) to its image indices
        for index, (_, pid, _, _) in enumerate(self.data_source):
            self.index_dic[pid].append(index)
        self.pids = list(self.index_dic.keys())  # List of unique class IDs (pids)

    def __iter__(self):
        
        index_dic = deepcopy(self.index_dic)
        for _ in range(self.trial):
            
            episode_idxs = []  # Collect indices for this episode
            
            # Randomly select `num_classes` (N) from available pids (with replacement allowed)
            selected_pids = random.sample(self.pids, self.way)
    
            for pid in selected_pids:
                np.random.shuffle(index_dic[pid])
                assert len(index_dic[pid]) >= (self.shot + self.query_shot), 'Not enough samples for the episode, got {} but need {}'.format(len(index_dic[pid]), self.shot + self.query_shot)

            for pid in selected_pids:
                episode_idxs.extend(index_dic[pid][:self.shot])

            for pid in selected_pids:
                episode_idxs.extend(index_dic[pid][self.shot: self.shot + self.query_shot])

            yield episode_idxs


class RandomSampler(Sampler):
    def __init__(self, data_source_query, datasource_gallery, way, shot, query_shot, trial=2000):
        
        self.data_source_query = data_source_query
        self.datasource_gallery = datasource_gallery
        self.way = way
        self.shot = shot
        self.query_shot = query_shot
        self.trial = trial

        self.index_dic = defaultdict(list)
        self.index_dic_gallery = defaultdict(list)
        # Create a dictionary mapping each class (pid) to its image indices
        for index, (_, pid, _, _) in enumerate(self.data_source_query):
            self.index_dic[pid].append(index)
        for index, (_, pid, _, _) in enumerate(self.datasource_gallery):
            self.index_dic_gallery[pid].append(index)

        self.pids = list(self.index_dic.keys())  # List of unique class IDs (pids)

    def __iter__(self):
        
        index_dic = deepcopy(self.index_dic)
        index_dic_gallery = deepcopy(self.index_dic_gallery)

        for _ in range(self.trial):
            
            episode_idxs = []  # Collect indices for this episode
            
            # Randomly select `num_classes` (N) from available pids (with replacement allowed)
            selected_pids = random.sample(self.pids, self.way)
    
            for pid in selected_pids:
                np.random.shuffle(index_dic[pid])
                np.random.shuffle(index_dic_gallery[pid])

            for pid in selected_pids:
                episode_idxs.extend(index_dic[pid])

            for pid in selected_pids:
                episode_idxs.extend(index_dic_gallery[pid][:self.query_shot])

            yield episode_idxs

        
# meta-training
class meta_batchsampler(Sampler):
    def __init__(self, data_source, way, shots, trial=250):

        class2id = {}
        for i, (image_path, class_id) in enumerate(data_source.imgs):
            if class_id not in class2id:
                class2id[class_id] = []
            class2id[class_id].append(i)

        self.class2id = class2id
        self.way = way
        self.shot = shots[0]
        self.trial = trial
        self.query_shot = shots[1]


    def __iter__(self):

        way = self.way
        shot = self.shot
        trial = self.trial
        query_shot = self.query_shot

        class2id = deepcopy(self.class2id)
        list_class_id = list(class2id.keys())

        for i in range(trial):

            id_list = []

            np.random.shuffle(list_class_id)
            picked_class = list_class_id[:way]

            for cat in picked_class:
                np.random.shuffle(class2id[cat])

            for cat in picked_class:
                id_list.extend(class2id[cat][:shot])
            for cat in picked_class:
                id_list.extend(class2id[cat][shot:(shot + query_shot)])

            yield id_list



# meta-testing
class random_sampler(Sampler):

    def __init__(self, data_source, way, shot, query_shot=16, trial=2000):

        class2id = {}

        for i, (image_path, class_id) in enumerate(data_source.imgs):
            if class_id not in class2id:
                class2id[class_id] = []
            class2id[class_id].append(i)

        self.class2id = class2id
        self.way = way
        self.shot = shot
        self.trial = trial
        self.query_shot = 16

    def __iter__(self):

        way = self.way
        shot = self.shot
        trial = self.trial
        query_shot = self.query_shot

        class2id = deepcopy(self.class2id)
        list_class_id = list(class2id.keys())

        for i in range(trial):

            id_list = []

            np.random.shuffle(list_class_id)
            picked_class = list_class_id[:way]

            for cat in picked_class:
                np.random.shuffle(class2id[cat])

            for cat in picked_class:
                id_list.extend(class2id[cat][:shot])
            for cat in picked_class:
                id_list.extend(class2id[cat][shot:(shot + query_shot)])

            yield id_list