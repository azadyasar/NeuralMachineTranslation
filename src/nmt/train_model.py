import os
import argparse


def train_nmt_model(args: argparse.Namespace):
  print("Training...")

def add_subparser(subparsers: argparse._SubParsersAction):
  parser = subparsers.add_parser('train', help='Train NMT model')
  
  group = parser.add_argument_group('Corpus and vocabulary')
  group.add_argument('--train_corpu', required=True,
                     help='training corpus file path')
  
  
  parser.set_defaults(func=train_nmt_model)