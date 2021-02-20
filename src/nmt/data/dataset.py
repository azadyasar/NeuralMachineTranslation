import torch
from nmt.data import Vocabulary
from tqdm import tqdm
from typing import List, Tuple

import logging
logger = logging.getLogger("DataReader")

class Batch(object):
  def __init__(self, src, trg, device):
    self.src = src.to(device)
    self.trg = trg.to(device)

class Dataset(object):
  def __init__(self,
               path: str,
               src_vocab: Vocabulary,
               trg_vocab: Vocabulary,
               device: str):
    self.dataset_path = path
    self.src_vocab, self.trg_vocab = src_vocab, trg_vocab
    self.current_idx = -1
    self.device = device
    
  def __len__(self):
    if self.indexed_source_seqs:
      return len(self.indexed_source_seqs)
    return 0
    
  def read_and_index(self):
    self.indexed_source_seqs = []
    self.indexed_target_seqs = []
    with open(self.dataset_path, 'r', encoding='utf-8') as ifi:
      logger.info(f"Reading and indexing the dataset {self.dataset_path}")
      lines = ifi.readlines()
      del lines[0]
      
      for line in tqdm(lines):
        en_sent, tr_sent = line.split('\t')
        en_idx = torch.tensor(self.trg_vocab.encode_and_pack(en_sent))
        tr_idx = torch.tensor(self.src_vocab.encode_and_pack(tr_sent))
        
        self.indexed_source_seqs.append(tr_idx)
        self.indexed_target_seqs.append(en_idx)
     
  def generate(self, batch_sz: int) -> Batch:
    for i in range(0, len(self.indexed_source_seqs), batch_sz):
      src_tensor = self.pad_tensor(self.indexed_source_seqs[i:i+batch_sz], self.src_vocab.pad_idx)
      trg_tensor = self.pad_tensor(self.indexed_target_seqs[i:i+batch_sz], self.trg_vocab.pad_idx)
      batch = Batch(src=src_tensor, trg=trg_tensor, device=self.device)
      
      yield batch
     
  def pad_tensor(self,
                 sequences: List[torch.Tensor],
                 pad_idx: int) -> torch.Tensor:
    num = len(sequences)
    max_len = max([s.size(0) for s in sequences])
    out_dims = (num, max_len)
    out_tensor = sequences[0].data.new(*out_dims).fill_(pad_idx)
    for i, tensor in enumerate(sequences):
        length = tensor.size(0)
        out_tensor[i, :length] = tensor
    return out_tensor
    