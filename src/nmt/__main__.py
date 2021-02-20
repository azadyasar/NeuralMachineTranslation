import argparse
from nmt import train_model

def encode_with_trsp(x):
  pass

def encode_with_ensp(x):
  pass

if __name__ == '__main__':
  parser = argparse.ArgumentParser(
    prog='nmt',
    description='PyTorch implementation of GRU+Attention and the Transformer NMT models')
  subparsers = parser.add_subparsers(dest='subcommands')
  
  train_model.add_subparser(subparsers)
  
  args = parser.parse_args()
  args.func(args)