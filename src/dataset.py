"""
@Filename       : dataset.py
@Create Time    : 6/18/2020 7:37 PM
@Author         : Rylynn
@Description    : 

"""
from torch.utils.data import Dataset

import logging

class DBLPDataset(Dataset):
    def __init__(self):
        logging.info("Loading data from dataset...")

        # TODO: load data

        logging.info("Loading data finished, total {} record.".format(0))

    def __getitem__(self, item):
        pass

    def __len__(self):
        pass


with open("../data/DBLP/author_label.dat") as f:
    for l in f.readlines():
        print(l)
