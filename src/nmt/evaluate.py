import argparse
from nmt.data import Vocabulary, Dataset
from nmt.evaluation import TransformerModelConfig, Evaluator
from nmt.util import get_device

import logging
logger = logging.getLogger("Evaluator")

def evaluate_model(args: argparse.Namespace):
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
                                  model_path=args.model_path)
  
  evaluator = Evaluator(config)
  
  test_dataset = Dataset(path=args.test_dataset,
                         src_vocab=source_vocab,
                         trg_vocab=target_vocab,
                         device=device)
  test_dataset.read_and_index()
  
  logger.info("Calculating model loss..")
  evaluator.test(test_dataset)
  logger.info("Calculating model BLEU score..")
  evaluator.calculate_bleu_score(test_dataset)


def add_subparser(subparsers: argparse._SubParsersAction):
  parser = subparsers.add_parser('evaluate', help='Evaluate a trained NMT model')
  
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
  group.add_argument('--enc_dropout', default=0.25, type=float,
                      help='encoder dropout rate')
  group.add_argument('--dec_dropout', default=0.25, type=int,
                       help='decoder dropout rate')
  group.add_argument('--batch_sz', default=128, type=int,
                      help='batch size')
  
  group = parser.add_argument_group('Vocabulary, dataset, and model paths')
  group.add_argument('--test_dataset', required=True,
                     help='path to the test dataset')
  group.add_argument('--src_vocab', required=True,
                     help='source BPE model file path')
  group.add_argument('--trg_vocab', required=True,
                     help='target BPE model file path')
  group.add_argument('--model_path', default='transformer_nmt.pt',
                      help='trained model path')
  
  
  parser.set_defaults(func=evaluate_model)