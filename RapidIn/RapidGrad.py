from torch.multiprocessing import Queue, Value, Lock, Barrier, Manager, Array
import torch.multiprocessing as mp
from ctypes import c_bool, c_int
from RapidIn.data_loader import get_model_tokenizer, TrainDataset, TestDataset, get_tokenizer, get_model
from RapidIn.calc_inner import grad_z
from RapidIn.utils import save_json, display_progress, load_json
from torch.utils.data import default_collate
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from tqdm import tqdm
import os
import numpy as np
from copy import copy
from pathlib import Path
import torch
import time
import math
import pickle

MAX_DATASET_SIZE = int(1e8)


class RapidGrad():
    def __init__(self, config, map_location, seed=42):
        self.is_init = False
        self.D = None
        self.K = None
        self.random_mat = None
        self.M = 1
        self.shuffle_lambda = config.influence.RapidGrad.shuffle_lambda*2
        self.perm_mat_list = []
        self.perm_dim_list = []
        self.config = config
        self.map_location = map_location
        self.seed = seed

    def __call__(self, vec, K):
        if self.is_init == False:
            print("Creating random and shuffling matrices. It may take a few minutes.")
            D = len(vec)
            self.init(D)
        for i, (dim, perm_mat) in enumerate(zip(self.perm_dim_list, self.perm_mat_list)):
            if i%2 == 0:
                vec = vec.reshape((dim, -1))
                vec = vec[perm_mat, :]
            else:
                vec = vec.reshape((-1, dim))
                vec = vec[:, perm_mat]
        vec = vec.reshape((-1))
        vec = vec*self.random_mat

        if isinstance(K, list):
            vec_list = []
            for k in K:
                step = self.D//k
                vec_list.append(torch.sum(vec.reshape((-1, step)), axis=1))
            return vec_list
        else:
            step = self.D//K
            vec = torch.sum(vec.reshape((-1, step)), axis=1)
            return vec

    def init(self, D):
        self.is_init = True
        np.random.seed(self.seed)
        self.D = D
        self.file_name = os.path.join(
            self.config.influence.grads_path,
            f"RapidGrad_D{self.D}_n{self.shuffle_lambda}.obj"
        )
        if not self.load():
            self.create_random_mat(D)
            self.create_perm_mat(D)
            self.save()
        self.random_mat = torch.from_numpy(self.random_mat).to(dtype=torch.float16).to(self.map_location)

    def create_random_mat(self, D):
        self.random_mat = np.random.randint(0, 2, (D,), dtype=np.int8)
        self.random_mat[self.random_mat < 1e-8] = -1

    def create_perm_mat(self, D):
        lt = []
        while D != 1:
            for i in range(2, int(D + 1)):
                if D % i == 0:
                    lt.append(i)
                    D = D / i
                    break
        for _ in tqdm(range(self.shuffle_lambda)):
            x = np.random.randint(len(lt)//4, len(lt)//2 + 1)
            np.random.shuffle(lt)
            dim = np.prod(lt[:x], dtype=np.longlong)
            self.perm_dim_list.append(dim)
            self.perm_mat_list.append(np.random.permutation(dim))

    def save(self):
        if os.path.exists(self.file_name):
            return
        with open(self.file_name, 'wb') as f:
            pickle.dump(self, f);

    def load(self):
        if not os.path.exists(self.file_name):
            return False
        with open(self.file_name, 'rb') as f:
            new_obj = pickle.load(f)
        map_location = self.map_location
        self.__dict__ = copy(new_obj.__dict__)
        self.map_location = map_location
        return True

