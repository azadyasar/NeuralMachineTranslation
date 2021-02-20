import argparse
from nmt.data import Vocabulary
from nmt.translation.configuration import TransformerModelConfig
from nmt.util import get_device
from nmt.translation import Translator

import logging
logger = logging.getLogger("Translator")

def translate_with_nmt_model(args: argparse.Namespace):
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
                                  model_path=args.model_path)
  translator = Translator(config=config)
  
  while True:
    input_sentence = input(">>")
    translation, translation_tokens, attention, _ = translator.translate(input_sentence.lower())
    print("==>" + translation + "\t\tAttention map is being saved to " + input_sentence[:10] + ".png")
    translator.display_attention(input_sentence,
                                 translation_tokens,
                                 attention,
                                 n_cols=4,
                                 figure_path='_'.join(input_sentence[:10].split()) + ".png")

def add_subparser(subparsers: argparse._SubParsersAction):
  parser = subparsers.add_parser('translate', help='Translate with a trained NMT model')
  
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
  
  group = parser.add_argument_group('Vocabulary and model paths')
  group.add_argument('--src_vocab', required=True,
                     help='source BPE model file path')
  group.add_argument('--trg_vocab', required=True,
                     help='target BPE model file path')
  group.add_argument('--model_path', default='transformer_nmt.pt',
                      help='trained model path')
  parser.set_defaults(func=translate_with_nmt_model)