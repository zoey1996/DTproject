import torch
from torch.utils.data import Dataset
from PyBioMed.PyProtein import CTD
import numpy as np
import re


class SmilesDataset(Dataset):

    def __init__(self,data,content,stoi,itos,block_size,aug_prob=0.5,pro=None):

        chars = sorted(list(content))
        data_size,vocab_size = len(data),len(chars)
        print('data has %d smiles, %d unique characters.' % (data_size, vocab_size))

        self.stoi = stoi
        self.itos = itos
        self.max_len = block_size
        self.vocab_size = vocab_size
        self.data = data
        self.pro = pro
        self.aug_prob = aug_prob
        
    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        selfies, pro = self.data[idx],self.pro[idx]  #Get selfies & protein sequence

        dix = [self.stoi[s] for s in selfies]
        if len(dix) > self.max_len:
            dix = dix[:self.max_len]

        pro_dix = list(CTD.CalculateCTD(pro).values())



        x = torch.tensor(dix[:-1], dtype=torch.long)
        y = torch.tensor(dix[1:], dtype = torch.long)
        pro = torch.tensor(pro_dix, dtype = torch.float)

        return x, y, pro
