import os
import argparse
from nmt.data import Vocabulary, Dataset
from nmt.training import Trainer, TransformerModelConfig
from nmt.util import get_device

import logging
logger = logging.getLogger("Trainer")

def train_nmt_model(args: argparse.Namespace):
  source_vocab = Vocabulary(args.src_vocab)
  target_vocab = Vocabulary(args.trg_vocab)
  device = get_device()
  config = TransformerModelConfig(input_dim=len(source_vocab),
                                  output_dim=len(target_vocab),
                                  hid_dim=args.hid_dims,
                                  enc_layers=args.enc_layers,
                                  dec_layers=args.dec_layers,
                                  enc_heads=args.enc_heads,
                                  dec_heads=args.dec_heads,
                                  enc_pf_dim=args.enc_pf_dim,
                                  dec_pf_dim=args.dec_pf_dim,
                                  enc_dropout=args.enc_dropout,
                                  dec_dropout=args.dec_dropout,
                                  device=device,
                                  src_vocab=source_vocab,
                                  trg_vocab=target_vocab,
                                  max_length=args.max_len,
                                  batch_sz=args.batch_sz,
                                  save_model_path=args.save_model_path)
  
  trainer = Trainer(config=config,
                    learning_rate=args.lr,
                    weight_decay=args.wd_rate,
                    n_epochs=args.n_epochs,
                    clip=args.clip)
  
  train_dataset = Dataset(path=args.train_dataset,
                          src_vocab=source_vocab,
                          trg_vocab=target_vocab,
                          device=device)
  eval_dataset = Dataset(path=args.eval_dataset,
                          src_vocab=source_vocab,
                          trg_vocab=target_vocab,
                          device=device)
  test_dataset = Dataset(path=args.test_dataset,
                          src_vocab=source_vocab,
                          trg_vocab=target_vocab,
                          device=device)
  
  train_dataset.read_and_index()
  eval_dataset.read_and_index()
  test_dataset.read_and_index()
  
  trainer.train(train_dataset=train_dataset, valid_dataset=eval_dataset)
  
  logger.info("\n\nTraining completed. Evaluating model on the test dataset.")
  trainer.evaluate_(eval_dataset=test_dataset)

def add_subparser(subparsers: argparse._SubParsersAction):
  parser = subparsers.add_parser('train', help='Train NMT model')
  
  group = parser.add_argument_group('Dataset and vocabulary')
  group.add_argument('--train_dataset', required=True,
                     help='training dataset file path')
  group.add_argument('--eval_dataset', required=True,
                     help='evaluation dataset file path')
  group.add_argument('--test_dataset', required=True,
                     help='test dataset file path')
  group.add_argument('--src_vocab', required=True,
                     help='source BPE model file path')
  group.add_argument('--trg_vocab', required=True,
                     help='target BPE model file path')
  
  group = parser.add_argument_group('Transformer model configurations')
  group.add_argument('--hid_dims', default=256, type=int,
                       help='hidden vector dimensions')
  group.add_argument('--enc_layers', default=8, type=int,
                       help='number of encoder layers')
  group.add_argument('--dec_layers', default=8, type=int,
                       help='number of decoder layers')
  group.add_argument('--enc_heads', default=4, type=int,
                       help='number of encoder attention heads')
  group.add_argument('--dec_heads', default=4, type=int,
                       help='number of decoder attention heads')
  group.add_argument('--enc_pf_dim', default=256*4, type=int,
                       help='encoder position-wise feed forward dimension. hid_dims * 4 is suggested.')
  group.add_argument('--dec_pf_dim', default=256*4, type=int,
                       help='decoder position-wise feed forward dimension. hid_dims * 4 is suggested.')
  group.add_argument('--max_len', default=172, type=int,
                       help='maximum number of tokens')

  group = parser.add_argument_group('Training specs')
  group.add_argument('--lr', default=0.0005, type=float,
                       help='encoder dropout rate')
  group.add_argument('--enc_dropout', default=0.25, type=float,
                       help='encoder dropout rate')
  group.add_argument('--dec_dropout', default=0.25, type=int,
                       help='decoder dropout rate')
  group.add_argument('--wd_rate', default=1e-4, type=float,
                       help='weight decay rate')
  group.add_argument('--clip', default=1., type=float,
                       help='gradient clipping')
  group.add_argument('--n_epochs', default=20, type=int,
                       help="number of epochs")
  group.add_argument('--batch_sz', default=128, type=int,
                       help="batch size")
  group.add_argument('--save_model_path', default='transformer_nmt.pt',
                       help='save trained model to the file')
  
  parser.set_defaults(func=train_nmt_model)