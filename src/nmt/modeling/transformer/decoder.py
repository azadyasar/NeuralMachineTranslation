import torch
import torch.nn as nn
from nmt.modeling.transformer import MultiHeadAttentionLayer, PositionwiseFeedforwardLayer

class DecoderLayer(nn.Module):
  def __init__(self, hid_dim, n_heads, pf_dim, dropout, device):
    super().__init__()

    self.self_attn_layer_norm = nn.LayerNorm(hid_dim)
    self.enc_attn_layer_norm = nn.LayerNorm(hid_dim)
    self.ff_layer_norm = nn.LayerNorm(hid_dim)
    self.self_attention = MultiHeadAttentionLayer(hid_dim, n_heads, dropout, device)
    self.encoder_attention = MultiHeadAttentionLayer(hid_dim, n_heads, dropout, device)
    self.positionwise_feedforward = PositionwiseFeedforwardLayer(hid_dim, pf_dim, dropout)

    self.dropout = nn.Dropout(dropout)

  def forward(self, trg, enc_src, trg_mask, src_mask):
    # trg = [batch_sz, trg_len, hid_dim]
    # enc_src = [batch_sz, src_len, hid_dim]
    # trg_mask = [batch_sz, 1, trg_len, trg_len]
    # src_mask = [batch_sz, 1, 1, src_len]

    # self attention
    _trg, _ = self.self_attention(trg, trg, trg, trg_mask)

    # dropout, residual connection and layer norm
    trg = self.self_attn_layer_norm(trg + self.dropout(_trg))
    # trg = [batch_sz, trg_len, hid_dim]

    # encoder attention
    _trg, attention = self.encoder_attention(trg, enc_src, enc_src, src_mask)

    # dropout, residual connection and layer norm
    trg = self.enc_attn_layer_norm(trg + self.dropout(_trg))
    # trg = [batch_sz, trg_len, hid_dim]

    # positionwise feedforward
    _trg = self.positionwise_feedforward(trg)

    # dropout, residual and layer norm
    trg = self.ff_layer_norm(trg + self.dropout(_trg))
    # trg = [batch_sz, trg_len, hid_dim]
    # attention = [batch_sz, n_heads, trg_len, src_len]

    return trg, attention
  
class Decoder(nn.Module):
  def __init__(self, output_dim, hid_dim, n_layers, n_heads, pf_dim, dropout, device, max_length):
    super().__init__()

    self.device = device

    self.tok_embedding = nn.Embedding(output_dim, hid_dim)
    self.pos_embedding = nn.Embedding(max_length, hid_dim)

    self.layers = nn.ModuleList([DecoderLayer(hid_dim,
                                              n_heads,
                                              pf_dim,
                                              dropout,
                                              device) 
                                for _ in range(n_layers)])
    self.fc_out = nn.Linear(hid_dim, output_dim)
    self.dropout = nn.Dropout(dropout)
    self.scale = torch.sqrt(torch.FloatTensor([hid_dim])).to(device)

  def forward(self, trg, enc_src, trg_mask, src_mask):
    # trg = [batch_sz, trg_len]
    # enc_src = [batch_sz, src_len, hid_dim]
    # trg_mask = [batch_sz, 1, trg_len, trg_len]
    # src_mask = [batch_sz, 1, 1, src_len]

    batch_sz = trg.shape[0]
    trg_len = trg.shape[1]

    pos = torch.arange(0, trg_len).unsqueeze(0).repeat(batch_sz, 1).to(self.device)
    # pos = [batch_sz, trg_len]

    trg = self.dropout((self.tok_embedding(trg) * self.scale) + self.pos_embedding(pos))
    # trg = [batch_sz, trg_len, hid_dim]

    for layer in self.layers:
      trg, attention = layer(trg, enc_src, trg_mask, src_mask)
    # trg = [batch_sz, trg_len, hid_dim]
    # attention = [batch_sz, n_heads, trg_len, src_len]

    output = self.fc_out(trg)
    # output = [batch_sz, trg_len, output_dim]

    return output, attention