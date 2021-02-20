from nmt.modeling import (Transformer,
                          Encoder as TransformerEncoder,
                          Decoder as TransformerDecoder)
from nmt.util import count_parameters
from nmt.data import Vocabulary
import torch

import logging
logger = logging.getLogger("Translator")


class TransformerModelConfig(object):
  def __init__(self,
               input_dim: int,
               output_dim: int,
               hid_dim: int,
               enc_layers: int,
               dec_layers: int,
               enc_heads: int,
               dec_heads: int,
               enc_pf_dim: int,
               dec_pf_dim: int,
               enc_dropout: float,
               dec_dropout: float,
               src_pad_idx: int,
               trg_pad_idx: int,
               device: str,
               src_vocab: Vocabulary,
               trg_vocab: Vocabulary,
               max_length: int,
               batch_sz: int,
               model_path: str = 'transformer_nmt.pt'):
    self.input_dim = input_dim
    self.output_dim = output_dim
    self.hid_dim = hid_dim
    self.enc_layers = enc_layers
    self.dec_layers = dec_layers
    self.enc_heads = enc_heads
    self.dec_heads = dec_heads
    self.enc_pf_dim = enc_pf_dim
    self.dec_pf_dim = dec_pf_dim
    self.enc_dropout = enc_dropout
    self.dec_dropout = dec_dropout
    self.device = device
    self.src_vocab = src_vocab
    self.trg_vocab = trg_vocab
    self.src_pad_idx = self.src_vocab.pad_idx
    self.trg_pad_idx = self.trg_vocab.pad_idx
    self.max_length = max_length
    self.model_path = model_path
    
  def load_model(self):
    enc = TransformerEncoder(self.input_dim,
                             self.hid_dim,
                             self.enc_layers,
                             self.enc_heads,
                             self.enc_pf_dim,
                             self.enc_dropout,
                             self.device,
                             max_length=164)
    dec = TransformerDecoder(self.output_dim,
                             self.hid_dim,
                             self.dec_layers,
                             self.dec_heads,
                             self.dec_pf_dim,
                             self.dec_dropout,
                             self.device,
                             self.max_length)

    model = Transformer(encoder=enc,
                        decoder=dec,
                        src_pad_idx=self.src_pad_idx,
                        trg_pad_idx=self.trg_pad_idx,
                        device=self.device).to(self.device)

    logger.info(f"Constructed Transformer model with {count_parameters(model):,} trainable parameters.")
    model.load_state_dict(torch.load(self.model_path), map_location=self.device)
    logger.info(f"Loaded parameters.")
    
    return model

    