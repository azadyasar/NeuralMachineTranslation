import torch
import torch.nn as nn
from nmt.modeling.transformer import (Encoder, Decoder)

class Transformer(nn.Module):
  def __init__(self, encoder: Encoder, decoder: Decoder, src_pad_idx, trg_pad_idx, device):
    super().__init__()

    self.encoder = encoder
    self.decoder = decoder
    self.src_pad_idx = src_pad_idx
    self.trg_pad_idx = trg_pad_idx
    self.device = device

  def make_src_mask(self, src):
    # src = [batch_sz, src_len]
    src_mask = (src != self.src_pad_idx).unsqueeze(1).unsqueeze(2)
    # src_mask = [batch_sz, 1, 1, src_len]
    return src_mask

  def make_trg_mask(self, trg):
    # trg = [batch_sz, trg_len]

    trg_pad_mask = (trg != self.trg_pad_idx).unsqueeze(1).unsqueeze(2)
    # trg_pad_mask = [batch_sz, 1, 1, trg_len]

    trg_len = trg.shape[1]
    trg_sub_mask = torch.tril(torch.ones((trg_len, trg_len), device=self.device)).bool()
    # trg_sub_mask = [trg_len, trg_len]

    trg_mask = trg_pad_mask & trg_sub_mask
    # trg_mask = [batch_sz, 1, trg_len, trg_len]

    return trg_mask

  def forward(self, src, trg):
    # src = [batch_sz, src_len]
    # trg = [batch_sz, trg_len]

    src_mask = self.make_src_mask(src)
    trg_mask = self.make_trg_mask(trg)

    # src_mask = [batch_sz, 1, 1, src_len]
    # trg_mask = [batch_sz, 1, trg_len, trg_len]

    enc_src = self.encoder(src, src_mask)
    # enc_src = [batch_sz, src_len, hid_dim]

    output, attention = self.decoder(trg, enc_src, trg_mask, src_mask)

    # output = [batch_sz, trg_len, output_dim]
    # attention = [batch_sz, n_heads, trg_len, src_len]

    return output, attention