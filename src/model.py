"""
@Filename       : model.py
@Create Time    : 6/18/2020 3:43 PM
@Author         : Rylynn
@Description    : 

"""
import argparse
import datetime
import logging
import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import torch.nn as nn

from torch.utils.data import DataLoader
from args import args
from dataset import *
from components import *

logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(filename)s[line:%(lineno)d] - %(levelname)s: %(message)s')


class HeGAN(object):
    def __init__(self, args):
        self.args = args
        logging.info("Loading pretrain embedding file...")
        pretrain_emb = self.read_pretrain_embed('../pretrain/dblp_pre_train.emb', node_size=args.node_size, embed_size=args.node_embed_size)
        self.args.pretrain_embed = pretrain_emb
        logging.info("Pretrain embedding file loaded.")

        logging.info("Building Generator...")
        generator = Generator(self.args)
        self.generator = generator.cuda()

        logging.info("Building Discriminator...")
        discriminator = Discriminator(self.args)
        self.discriminator = discriminator.cuda()

        self.name = "HeGAN-" + datetime.datetime.now().strftime('%Y-%m-%d_%H%M%S')
        if args.name:
            self.name = args.name + datetime.datetime.now().strftime('%Y-%m-%d_%H%M%S')


    def read_pretrain_embed(self, pretrain_file, node_size, embed_size):
        embedding_matrix = np.random.rand(node_size, embed_size)
        i = -1
        with open(pretrain_file) as infile:
            for line in infile.readlines()[1:]:
                i += 1
                emd = line.strip().split()
                embedding_matrix[int(emd[0]), :] = self.str_list_to_float(emd[1:])
        return embedding_matrix

    def str_list_to_float(self, str_list):
        return [float(item) for item in str_list]

    def train(self):
        writer = SummaryWriter("./log/" + self.name)

        dblp_dataset = DBLPDataset()
        gen_data_loader = DataLoader(dblp_dataset.generator_dataset, shuffle=True, batch_size=self.args.batch_size,
                                     num_workers=8, pin_memory=True)
        dis_data_loader = DataLoader(dblp_dataset.discriminator_dataset, shuffle=True, batch_size=self.args.batch_size,
                                     num_workers=8, pin_memory=True)

        discriminator_optimizer = torch.optim.Adam(self.discriminator.parameters(), self.args.dis_lr)
        generator_optimizer = torch.optim.Adam(self.generator.parameters(), self.args.gen_lr)

        logging.info("Training Begin...")
        for total_idx in range(self.args.epoch):
            # Training discriminator
            for dis_idx in range(self.args.epoch_g):
                dis_batch_loss = 0
                for (batch_idx, data) in tqdm(enumerate(dis_data_loader)):
                    pos_node_idx = data[0].cuda()
                    pos_relation_idx = data[1].cuda()
                    pos_node_neighbor_idx = data[2].cuda()

                    neg_node_idx = data[3].cuda()
                    neg_relation_idx = data[4].cuda()
                    neg_node_neighbor_idx = data[5].cuda()

                    fake_nodes_embed = self.generator.generate_fake_nodes(pos_node_idx, pos_relation_idx)

                    prob_pos = self.discriminator(pos_node_idx, pos_relation_idx, pos_node_neighbor_idx)
                    prob_neg = self.discriminator(neg_node_idx, neg_relation_idx, neg_node_neighbor_idx)
                    prob_fake = self.discriminator.forward_fake(pos_node_idx, pos_relation_idx, fake_nodes_embed)

                    discriminator_loss_pos = torch.sum(torch.log(prob_pos))
                    discriminator_loss_neg = torch.sum(torch.log(-prob_neg))
                    discriminator_loss_fake = torch.sum(torch.log(-prob_fake))

                    discriminator_loss = discriminator_loss_pos + discriminator_loss_neg + discriminator_loss_fake
                    dis_batch_loss += discriminator_loss.item()

                    discriminator_optimizer.zero_grad()
                    discriminator_loss.backward()
                    discriminator_optimizer.step()

                logging.info("Total epoch: {}, Discriminator epoch: {}, loss: {}.".
                             format(total_idx, dis_idx, dis_batch_loss / len(dis_data_loader)))
                writer.add_scalar("dis_loss", dis_batch_loss / len(dis_data_loader))

            # Training generator
            for gen_idx in range(self.args.epoch_d):
                gen_batch_loss = 0
                for (batch_idx, data) in tqdm(enumerate(gen_data_loader)):
                    node_idx = data[0].cuda()
                    relation_idx = data[1].cuda()

                    temp = self.discriminator.multify(node_idx, relation_idx)
                    prob = self.generator(node_idx, relation_idx, dis_temp=temp)
                    generator_loss = self.generator.loss(prob)

                    l2_regularization = torch.tensor([0], dtype=torch.float32)
                    for param in self.generator.parameters():
                        l2_regularization += torch.norm(param, 2)

                    generator_loss = generator_loss + l2_regularization
                    gen_batch_loss += generator_loss.item()

                    generator_optimizer.zero_grad()
                    generator_loss.backward()
                    generator_optimizer.step()

                logging.info("Total epoch: {}, Discriminator epoch: {}, loss: {}.".
                             format(total_idx, gen_idx, gen_batch_loss / len(gen_data_loader)))
                writer.add_scalar("gen_loss", gen_batch_loss / len(gen_data_loader))

        writer.close()




def main():
    he_gan = HeGAN(args=args)
    he_gan.train()

def test():
    dblp_dataset = DBLPDataset()
    dl = DataLoader(dblp_dataset.discriminator_dataset, shuffle=True, batch_size=args.batch_size)
    for (idx, a) in enumerate(dl):
        print(a)


if __name__ == '__main__':
    # test()
    main()
