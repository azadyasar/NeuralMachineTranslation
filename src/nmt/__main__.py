import argparse
from nmt import (train_model)

if __name__ == '__main__':
  parser = argparse.ArgumentParser(
    prog='NMT',
    description='PyTorch implementation of GRU+Attention and the Transformer NMT models')
  subparsers = parser.add_subparsers(dest='subcommands')
  
  train_model.add_subparser(subparsers)
  
  args = parser.parse_args()
  argparse.func(args)