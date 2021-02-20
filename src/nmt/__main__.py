import argparse
from nmt import (train_model,
                 translate)

import logging
logging.basicConfig(level=logging.INFO)

if __name__ == '__main__':
  parser = argparse.ArgumentParser(
    prog='nmt',
    description='PyTorch implementation of GRU+Attention and the Transformer NMT models')
  subparsers = parser.add_subparsers(dest='subcommands')
  
  train_model.add_subparser(subparsers)
  translate.add_subparser(subparsers)
  
  args = parser.parse_args()
  args.func(args)