from nmt.evaluation import TransformerModelConfig
from nmt.data import Dataset
import torch
import torch.nn as nn
from torchtext.data.metrics import bleu_score
from tqdm import tqdm
import time
import numpy as np

import logging
logger = logging.getLogger("Evaluator")

class Evaluator(object):
  def __init__(self,
               config: TransformerModelConfig):
    self.config = config
    self.model = config.load_model()
    
    self.criterion = nn.CrossEntropyLoss(ignore_index=self.config.trg_pad_idx)
    
  def test(self,
            test_dataset: Dataset):
    self.model.eval()
    epoch_loss = 0
    
    with torch.no_grad():
        for batch in tqdm(test_dataset.generate(self.config.batch_sz)):
            src = batch.src
            trg = batch.trg

            output, _ = self.model(src, trg[:,:-1])
            # output = [batch_sz, trg_len - 1, output_dim]
            # trg = [batch_sz, trg_len]

            output_dim = output.shape[-1]

            output = output.contiguous().view(-1, output_dim)
            trg = trg[:, 1:].contiguous().view(-1)
            # output = [batch_sz * trg_len - 1, output_dim]
            # trg = [batch_sz * trg_len - 1]

            loss = self.criterion(output, trg)
            epoch_loss += loss.item()
        
    test_loss = epoch_loss / len(test_dataset)
    logger.info(f'| Test Loss: {test_loss:.3f} | Test PPL: {np.exp(test_loss):7.3f} |')
    return test_loss
  
  def translate_sentence_vectorized(self, src_tensor, max_len=128):
    self.model.eval()
    
    assert isinstance(src_tensor, torch.Tensor)
    src_mask = self.model.make_src_mask(src_tensor)

    with torch.no_grad():
      enc_src = self.model.encoder(src_tensor, src_mask)
      # enc_src = [batch_sz, src_len, hid_dim]

    trg_indexes = [[self.config.trg_vocab.bos_idx] for _ in range(len(src_tensor))]
    # Even though some examples might have been completed by producing a <eos> token
    # we still need to feed them through the model because others are not yet finished
    # and all examples act as a batch. Once every single sentence prediction encounters
    # <eos> token, then we can stop predicting.
    translations_done = [0] * len(src_tensor)
    for i in range(max_len):
      trg_tensor = torch.LongTensor(trg_indexes).to(self.config.device)
      trg_mask = self.model.make_trg_mask(trg_tensor)
      with torch.no_grad():
        output, attention = self.model.decoder(trg_tensor, enc_src, trg_mask, src_mask)
      pred_tokens = output.argmax(2)[:,-1]
      for i, pred_token_i in enumerate(pred_tokens):
        trg_indexes[i].append(pred_token_i)
        if pred_token_i == self.config.trg_vocab.eos_idx:
          translations_done[i] = 1
      if all(translations_done):
        break

    # Iterate through each predicted example one by one;
    # Cut-off the portion including the after the <eos> token
    pred_sentences = []
    for trg_sentence in trg_indexes:
      pred_sentence = []
      for i in range(1, len(trg_sentence)):
        if trg_sentence[i] == self.config.trg_vocab.eos_idx:
          break
        pred_sentence.append(self.config.trg_vocab.id_to_piece(trg_sentence[i].item()))
      pred_sentences.append(pred_sentence)

    return pred_sentences, attention

  def calculate_bleu_score(self, test_dataset: Dataset, max_len = 128) -> float:
    trgs = []
    pred_trgs = []
    
    with torch.no_grad():
      for batch in tqdm(test_dataset.generate(self.config.batch_sz)):
        src = batch.src
        trg = batch.trg
        _trgs = []
        for sentence in trg:
          tmp = []
          for i in sentence[1:]:
            if i == self.config.trg_vocab.eos_idx or\
              i == self.config.trg_vocab.pad_idx:
              break
            tmp.append(self.config.trg_vocab.id_to_piece(i.item()))
          _trgs.append([tmp])
        trgs += _trgs
        pred_trg, _ = self.translate_sentence_vectorized(src, max_len=max_len)
        pred_trgs += pred_trg
    
    logger.info(f'BLEU score = {bleu_score*100:.2f}')
    return pred_trgs, trgs, bleu_score(pred_trgs, trgs)