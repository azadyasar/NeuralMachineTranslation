import torch
import torch.nn as nn

class PositionwiseFeedforwardLayer(nn.Module):
  def __init__(self, hid_dim, pf_dim, dropout):
    super().__init__()

    self.fc_1 = nn.Linear(hid_dim, pf_dim)
    self.fc_2 = nn.Linear(pf_dim, hid_dim)

    self.dropout = nn.Dropout(dropout)

  def forward(self, x):
    # x = [batch_sz, seq_len, hid_dim]
    x = self.dropout(torch.relu(self.fc_1(x)))
    # x = [batch_sz, seq_len, pf_dim]

    x = self.fc_2(x)
    # x = [batch_sz, seq_len, hid_dim]

    return x