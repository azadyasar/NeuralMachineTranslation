from torchtext.data import Field
import sentencepiece as spm
from nmt.util import load_vocab
from typing import List

class Vocabulary(object):
  def __init__(self, tokenizer_path: str):
    self.tokenizer = spm.SentencePieceProcessor(tokenizer_path)
    
  def encode(self, sentence: str) -> List[str]:
    return self.tokenizer.Encode(sentence)
    
  def encode_and_pack(self, sentence: str, max_len: int = None) -> List[str]:
    result =  [self.bos_idx] + self.encode(sentence) + [self.eos_idx]
    if max_len is not None:
      result += [self.pad_idx] * (max_len - len(result))
    
    return result
  
  def decode(self, tokens: List[int]) -> str:
    return self.tokenizer.Decode(tokens)
  
  
  def id_to_piece(self, id: int) -> str:
    return self.tokenizer.IdToPiece(id)
  
  def piece_to_id(self, piece: str) -> int:
    return self.tokenizer.PieceToId(piece)
  
  @property
  def bos_idx(self) -> int:
    return self.tokenizer.bos_id()
  
  @property
  def eos_idx(self) -> int:
    return self.tokenizer.eos_id()
  
  @property
  def unk_idx(self) -> int:
    return self.tokenizer.unk_id()
  
  @property
  def pad_idx(self) -> int:
    return self.tokenizer.pad_id()
  
  def __len__(self) -> int:
    return self.tokenizer.vocab_size()
  