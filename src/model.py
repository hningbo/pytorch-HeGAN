"""
@Filename       : model.py
@Create Time    : 6/18/2020 3:43 PM
@Author         : Rylynn
@Description    : 

"""

import logging
import torch
import torch.nn as nn

from torch.utils.data import DataLoader
from dataset import *
from components import *

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(filename)s[line:%(lineno)d] - %(levelname)s: %(message)s')


class HeGAN(object):
    def __init__(self, args):
        self.args = args
        logging.info("Building Generator...")
        self.generator = Generator(args)
        logging.info("Building Discriminator...")
        self.discriminator = Discriminator(args)

    def show_config(self):

        print(self.args)

    def train(self):
        dblp_dataset = DBLPDataset()
        data_loader = DataLoader(dblp_dataset, shuffle=True, batch_size=self.args.batch_size)

        discriminator_optimizer = torch.optim.Adam(self.discriminator.parameters(), self.args.dis_lr)
        generator_optimizer = torch.optim.Adam(self.generator.parameters(), self.args.gen_lr)

        for (batch_idx, data) in enumerate(data_loader):
            pass
            node_idx = None
            relation_idx = None
            # TODO:split train data into 3 parts

            # Training discriminator
            fake_nodes_embed = self.generator.generate_fake_nodes(node_idx, relation_idx)
            discriminator_loss = self.discriminator.forward(node_idx, relation_idx, fake_nodes_embed)
            discriminator_optimizer.zero_grad()
            discriminator_loss.backward()
            discriminator_optimizer.step()

            # Training generator
            temp = self.discriminator.multify(node_idx, relation_idx)
            generator_loss = self.generator.forward(node_idx, relation_idx, dis_temp=temp)

            l2_regularization = torch.tensor([0], dtype=torch.float32)
            for param in self.generator.parameters():
                l2_regularization += torch.norm(param, 2)

            generator_loss = generator_loss + l2_regularization

            generator_optimizer.zero_grad()
            generator_loss.backward()
            generator_optimizer.step()

dblp_dataset = DBLPDataset()