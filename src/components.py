"""
@Filename       : components.py
@Create Time    : 6/18/2020 3:43 PM
@Author         : Rylynn
@Description    : 

"""
from collections import OrderedDict

import torch
import torch.nn as nn


class Discriminator(nn.Module):
    def __init__(self, args):
        super(Discriminator, self).__init__()
        self.args = args

        self.node_embed = nn.Embedding.from_pretrained(torch.from_numpy(args.pretrain_embed)).float()
        self.relation_embed = nn.Embedding(args.relation_size, args.node_embed_size * args.node_embed_size)

        nn.init.xavier_uniform(self.relation_embed.weight)

        self.sigmoid = nn.Sigmoid()

    def forward_fake(self, node_idx, relation_idx, fake_node_embed):
        node_embed = self.node_embed(node_idx)
        node_embed = node_embed.reshape((-1, 1, self.args.node_embed_size))
        relation_embed = self.relation_embed(relation_idx)
        relation_embed = relation_embed.reshape((-1, self.args.node_embed_size, self.args.node_embed_size))
        temp = torch.matmul(node_embed, relation_embed)

        score = torch.sum(torch.mul(temp, fake_node_embed), 0)
        prob = self.sigmoid(score)
        return prob

    def forward(self, node_idx, relation_idx, node_neighbor_idx):
        node_embed = self.node_embed(node_idx)
        node_embed = node_embed.reshape((-1, 1, self.args.node_embed_size))
        relation_embed = self.relation_embed(relation_idx)
        relation_embed = relation_embed.reshape((-1, self.args.node_embed_size, self.args.node_embed_size))
        temp = torch.matmul(node_embed, relation_embed)

        score = torch.sum(torch.mul(temp, self.node_embed(node_neighbor_idx)), 0)

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
        relation_embed = relation_embed.reshape((-1, self.args.node_embed_size, self.args.node_embed_size))
        temp = torch.matmul(node_embed, relation_embed)
        return temp


class Generator(nn.Module):
    def __init__(self, args):
        super(Generator, self).__init__()
        self.args = args

        node_embed_size = args.node_embed_size

        self.node_embed = nn.Embedding.from_pretrained(torch.from_numpy(args.pretrain_embed)).float()
        self.relation_embed = nn.Embedding(args.relation_size, args.node_embed_size * args.node_embed_size)

        nn.init.xavier_uniform(self.relation_embed.weight)

        self.fc = nn.Sequential(
            OrderedDict([
                ("w_1", nn.Linear(node_embed_size, node_embed_size)),
                ("a_1", nn.LeakyReLU()),
                ("w_2", nn.Linear(node_embed_size, node_embed_size)),
                ("a_2", nn.LeakyReLU())
            ])
        )

        nn.init.xavier_uniform(self.fc[0].weight)
        nn.init.xavier_uniform(self.fc[2].weight)

        self.sigmoid = nn.Sigmoid()

    def forward(self, node_idx, relation_idx, dis_temp):
        fake_nodes = self.generate_fake_nodes(node_idx, relation_idx)
        score = torch.matmul(dis_temp, fake_nodes)
        prob = self.sigmoid(score)
        return prob

    def loss(self, prob):
        loss = torch.sum(torch.log(- prob))
        return loss

    def generate_fake_nodes(self, node_idx, relation_idx):
        node_embed = self.node_embed(node_idx)
        node_embed = node_embed.reshape((-1, 1, self.args.node_embed_size))
        relation_embed = self.relation_embed(relation_idx)
        relation_embed = relation_embed.reshape((-1, self.args.node_embed_size, self.args.node_embed_size))
        temp = torch.matmul(node_embed, relation_embed)

        # add noise
        temp = temp + torch.randn(temp.shape, requires_grad=False).cuda()
        output = self.fc(temp)

        return output
