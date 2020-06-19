"""
@Filename       : args.py
@Create Time    : 6/18/2020 8:27 PM
@Author         : Rylynn
@Description    : 

"""
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--name', type=str, help='Model name')
parser.add_argument('--batch_size', type=int, default=128, help='Batch size')
parser.add_argument('--epoch', type=int, default=20, help='Training epoch')
parser.add_argument('--epoch_g', type=int, default=10, help='Generator epoch in each training epoch')
parser.add_argument('--epoch_d', type=int, default=10, help='Discriminator epoch in each training epoch')
parser.add_argument('--sample_num', type=int, default=20, help='Number of sampling in one node')
parser.add_argument('--gen_lr', type=float, default=0.001, help='Learning rate of Generator')
parser.add_argument('--dis_lr', type=float, default=0.001, help='Learning rate of Discriminator')
parser.add_argument('--node_embed_size', type=int, default=64, help='Node Embedding size')
parser.add_argument('--node_size', type=int, default=37791, help='Node Embedding size')
parser.add_argument('--relation_size', type=int, default=6, help='Relation size')
args = parser.parse_args()