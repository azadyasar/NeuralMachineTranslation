from torchtext.data import Field
import sentencepiece as spm
from nmt.util import load_vocab
from typing import List

class Vocabulary(object):
  def __init__(self, tokenizer_path: str, field_path: str):
    self.tokenizer = spm.SentencePieceProcessor(tokenizer_path)
    self.field = load_vocab(field_path)
    
  def encode(self, sentence: str) -> List[str]:
    return self.tokenizer.Encode(sentence)
    
  def encode_and_pack(self, sentence: str) -> List[str]:
    return ['<start>'] + self.encode(sentence) + ['<end>']
  
  def decode(self, tokens: List[int]) -> str:
    return self.tokenizer.Decode(tokens)
  
  def get_pad_idx(self) -> int:
    return self.field.vocab.stoi[self.field.pad_token]
  
  @property
  def vocab(self) -> Field:
    return self.field