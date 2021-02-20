import torch
import torch.nn as nn
from nmt.modeling.transformer import (MultiHeadAttentionLayer, PositionwiseFeedforwardLayer)

class EncoderLayer(nn.Module):
  def __init__(self, hid_dim, n_heads, pf_dim, dropout, device):
    super().__init__()

    self.self_attn_layer_norm = nn.LayerNorm(hid_dim)
    self.ff_layer_norm = nn.LayerNorm(hid_dim)
    self.self_attention = MultiHeadAttentionLayer(hid_dim, n_heads, dropout, device)
    self.positionwise_feedforward = PositionwiseFeedforwardLayer(hid_dim, pf_dim, dropout)

    self.dropout = nn.Dropout(dropout)

  def forward(self, src, src_mask):
    # src = [batch_sz, src_len, hid_dim]
    # src_mask = [batch_sz, 1, 1, src_len]

    # self attention
    _src, self_attn = self.self_attention(src, src, src, src_mask)

    # dropout, residual connection and layer norm
    src = self.self_attn_layer_norm(src + self.dropout(_src))
    #src = [batch_sz, src_len, hid_dim]

    _src = self.positionwise_feedforward(src)

    # dropout, residual and layer norm
    src = self.ff_layer_norm(src + self.dropout(_src))
    # src = [batch_sz, src_len, hid_dim]

    return src

  def forward_w_attn(self, src, src_mask):
    # src = [batch_sz, src_len, hid_dim]
    # src_mask = [batch_sz, 1, 1, src_len]

    # self attention
    _src, self_attn = self.self_attention(src, src, src, src_mask)

    # dropout, residual connection and layer norm
    src = self.self_attn_layer_norm(src + self.dropout(_src))
    #src = [batch_sz, src_len, hid_dim]

    _src = self.positionwise_feedforward(src)

    # dropout, residual and layer norm
    src = self.ff_layer_norm(src + self.dropout(_src))
    # src = [batch_sz, src_len, hid_dim]
    
    return src, self_attn
  
class Encoder(nn.Module):
  def __init__(self, input_dim, hid_dim, n_layers, n_heads, pf_dim, dropout, device, max_length):
    super().__init__()

    self.device = device
    
    self.tok_embedding = nn.Embedding(input_dim, hid_dim)
    self.pos_embedding = nn.Embedding(max_length, hid_dim)

    self.layers = nn.ModuleList([EncoderLayer(hid_dim, n_heads, pf_dim, dropout, device) for _ in range(n_layers)])
    self.dropout = nn.Dropout(dropout)
    self.scale = torch.sqrt(torch.FloatTensor([hid_dim])).to(device)

  def forward(self, src, src_mask):
    # src = [batch_sz, src_len]
    # src_mask = [batch_sz, 1, 1, src_len]
    batch_sz = src.shape[0]
    src_len = src.shape[1]

    pos = torch.arange(0, src_len).unsqueeze(0).repeat(batch_sz, 1).to(self.device)
    # pos = [batch_sz, src_len]

    src = self.dropout((self.tok_embedding(src) * self.scale) + self.pos_embedding(pos))
    # src = [batch_sz, src_len, hid_dim]

    for layer in self.layers:
      src = layer(src, src_mask)
    # src = [batch_sz, src_len, hid_dim]
    
    return src
  
  def forward_w_attn(self, src, src_mask):
    # src = [batch_sz, src_len]
    # src_mask = [batch_sz, 1, 1, src_len]
    batch_sz = src.shape[0]
    src_len = src.shape[1]

    pos = torch.arange(0, src_len).unsqueeze(0).repeat(batch_sz, 1).to(self.device)
    # pos = [batch_sz, src_len]

    src = self.dropout((self.tok_embedding(src) * self.scale) + self.pos_embedding(pos))
    # src = [batch_sz, src_len, hid_dim]

    for layer in self.layers:
      src, attn = layer.forward_w_attn(src, src_mask)
    # src = [batch_sz, src_len, hid_dim]
    
    return src, attn