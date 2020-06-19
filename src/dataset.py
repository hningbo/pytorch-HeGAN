"""
@Filename       : dataset.py
@Create Time    : 6/18/2020 7:37 PM
@Author         : Rylynn
@Description    : 

"""
import random

from torch.utils.data import Dataset

import logging
import numpy as np
from args import args


class DBLPDataset():
    def __init__(self):
        logging.info("Loading data from dataset...")
        self.node_size, self.relation_size, self.graph = read_graph('../data/DBLP/dblp_triple.dat')
        logging.info("Loading data finished, total {} nodes, {} relations.".format(self.node_size, self.relation_size))
        self.generator_dataset = self.GeneratorDataset(self.graph)
        self.discriminator_dataset = self.DiscriminatorDataset(self.graph)

    class GeneratorDataset(Dataset):
        def __init__(self, graph):
            self.graph = graph
            self.node_list = list(graph.keys())
            self.node_idx = []
            self.relation_idx = []

            self.sample()

        def __getitem__(self, index):
            return self.node_idx[index], self.relation_idx[index]

        def __len__(self):
            return len(self.node_idx)

        def sample(self):
            logging.info("Sampling data for Generator")

            for node_id in self.node_list:
                for i in range(args.sample_num):
                    relations = list(self.graph[node_id].keys())
                    relation_id = random.sample(relations, 1)[0]

                    self.node_idx.append(node_id)
                    self.relation_idx.append(relation_id)

    class DiscriminatorDataset(Dataset):
        def __init__(self, graph):
            self.graph = graph
            self.node_list = list(graph.keys())

            # real node and real relation
            self.pos_node_idx = []
            self.pos_relation_idx = []
            self.pos_node_neighbor_idx = []

            # real node and wrong relation
            self.neg_node_idx = []
            self.neg_relation_idx = []
            self.neg_node_neighbor_idx = []

            self.sample()

        def __getitem__(self, index):
            return self.pos_node_idx[index], self.pos_relation_idx[index], self.pos_node_neighbor_idx[index], \
                   self.neg_node_idx[index], self.neg_relation_idx[index], self.neg_node_neighbor_idx[index]

        def __len__(self):
            return len(self.pos_node_idx)

        def sample(self):
            logging.info("Sampling data for Discriminator")
            for node_id in self.node_list:
                for i in range(args.sample_num):
                    # sample real node and true relation
                    relations = list(self.graph[node_id].keys())
                    relation_id = random.sample(relations, 1)[0]
                    neighbors = self.graph[node_id][relation_id]
                    node_neighbor_id = neighbors[np.random.randint(0, len(neighbors))]

                    self.pos_node_idx.append(node_id)
                    self.pos_relation_idx.append(relation_id)
                    self.pos_node_neighbor_idx.append(node_neighbor_id)

                    # sample real node and wrong relation
                    self.neg_node_idx.append(node_id)
                    self.neg_node_neighbor_idx.append(node_neighbor_id)
                    neg_relation_id = np.random.randint(0, args.relation_size)
                    while neg_relation_id == relation_id:
                        neg_relation_id = np.random.randint(0, args.relation_size)
                    self.neg_relation_idx.append(neg_relation_id)


def read_graph(graph_filename):
    # p -> a : 0
    # a -> p : 1
    # p -> c : 2
    # c -> p : 3
    # p -> t : 4
    # t -> p : 5
    # graph_filename = '../data/dblp/dblp_triple.dat'

    relations = set()
    nodes = set()
    graph = {}

    with open(graph_filename) as infile:
        for line in infile.readlines():
            source_node, target_node, relation = line.strip().split(' ')
            source_node = int(source_node)
            target_node = int(target_node)
            relation = int(relation)

            nodes.add(source_node)
            nodes.add(target_node)
            relations.add(relation)

            if source_node not in graph:
                graph[source_node] = {}

            if relation not in graph[source_node]:
                graph[source_node][relation] = []

            graph[source_node][relation].append(target_node)

    n_node = len(nodes)
    return n_node, len(relations), graph
