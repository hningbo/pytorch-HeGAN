"""
@Filename       : components.py
@Create Time    : 6/18/2020 3:43 PM
@Author         : Rylynn
@Description    : 

"""

import torch
import torch.nn as nn


class Discriminator(nn.Module):
    def __init__(self, args):
        super(Discriminator, self).__init__()
        self.args = args

        self.node_embed = nn.Embedding(args.node_size, args.node_embed_size)
        self.relation_embed = nn.Embedding(args.relation_size, args.node_embed_size *args.node_embed_size)
        self.node_embed.from_pretrained(args.node_embed)
        nn.init.xavier_uniform(self.relation_embed)

        self.sigmoid = nn.Sigmoid()

    def forward(self, node_idx, relation_idx, fake_node_embed):
        temp = self.multify(node_idx, relation_idx)
        score = torch.matmul(temp, fake_node_embed)
        prob = self.sigmoid(score)
        return prob

    def multify(self, node_idx, relation_idx):
        """
        get e_u^D * M_r^b
        :param node_idx:
        :param relation_idx:
        :return:
        """
        node_embed = self.node_embed(node_idx)
        relation_embed = self.relation_embed(relation_idx)
        relation_embed.reshape((-1, self.args.node_embed_size, self.args.node_embed_size))
        temp = torch.matmul(node_embed, relation_embed)
        return temp

class Generator(nn.Module):
    def __init__(self, args):
        super(Generator, self).__init__()

        node_embed_size = args.node_embed_size

        self.node_embed = nn.Embedding(args.node_size, args.node_embed_size)
        self.relation_embed = nn.Embedding(args.relation_size, args.node_embed_size * args.node_embed_size)

        nn.init.xavier_uniform(self.node_embed)
        nn.init.xavier_uniform(self.relation_embed)

        self.fc = nn.Sequential()
        self.fc.add_module("w_1", nn.Linear(node_embed_size, node_embed_size))
        self.fc.add_module("a_1", nn.LeakyReLU())

        self.fc.add_module("w_2", nn.Linear(node_embed_size, node_embed_size))
        self.fc.add_module("a_2", nn.LeakyReLU())

        nn.init.xavier_uniform(self.fc['w_1'])
        nn.init.xavier_uniform(self.fc['w_2'])

        self.sigmoid = nn.Sigmoid()

    def forward(self, node_idx, relation_idx, dis_temp):
        fake_nodes = self.generate_fake_nodes(node_idx, relation_idx)
        score = torch.matmul(dis_temp, fake_nodes)
        prob = self.sigmoid(score)

        loss = torch.sum(torch.log(- prob))
        return loss

    def generate_fake_nodes(self, node_idx, relation_idx):
        node_embed = self.node_embed(node_idx)
        relation_embed = self.relation_embed(relation_idx)
        relation_embed.reshape((-1, self.args.node_embed_size, self.args.node_embed_size))
        temp = torch.matmul(node_embed, relation_embed)
        temp = temp + torch.rand(temp.shape)
        output = self.fc(temp)

        return output

