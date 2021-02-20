from nmt.translation import TransformerModelConfig
from nmt.data import Dataset
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

import logging
logger = logging.getLogger("Translator")

class Translator(object):
  def __init__(self,
               config: TransformerModelConfig):
    self.config = config
    self.model = config.load_model()
    
    self.criterion = nn.CrossEntropyLoss(ignore_index=self.config.trg_pad_idx)
    
  def translate(self,
                sentence: str,
                max_len: int = 100) -> str:
    
    self.model.eval()
    trg_field = self.config.trg_vocab.vocab

    src_tokens = self.config.src_vocab.encode_and_pack(sentence.lower())
    src_indexes = [self.config.src_vocab.vocab.stoi[token] for token in src_tokens]

    src_tensor = torch.LongTensor(src_indexes).unsqueeze(0).to(self.config.device)
    src_mask = self.model.make_src_mask(src_tensor)

    with torch.no_grad():
      enc_src, self_attn = self.model.encoder.forward_w_attn(src_tensor, src_mask)
    
    trg_indexes = [trg_field.stoi[trg_field.init_token]]

    for i in range(max_len):
      trg_tensor = torch.LongTensor(trg_indexes).unsqueeze(0).to(self.config.device)
      trg_mask = self.model.make_trg_mask(trg_tensor)
      with torch.no_grad():
        output, attention = self.model.decoder(trg_tensor, enc_src, trg_mask, src_mask)
      pred_token = output.argmax(2)[:,-1].item()
      trg_indexes.append(pred_token)

      if pred_token == trg_field.vocab.stoi[trg_field.eos_token]:
        break
    
    trg_tokens = [trg_field.vocab.itos[i] for i in trg_indexes]
    return trg_tokens[1:], attention, self_attn
  
  def display_attention(self, sentence, translation, attention, n_cols=4, figure_path = 'attention_figure.png'):
    # assert n_rows * n_cols == n_heads
    n_heads = self.config.dec_heads
    n_rows = n_heads // n_cols
    
    if isinstance(sentence, str):
      sentence = self.config.src_vocab.encode_and_pack(sentence)

    fig = plt.figure(figsize=(16,16))
    plt.axis('off')
    
    for i in range(n_heads):
      ax = fig.add_subplot(n_rows, n_cols, i+1)

      _attention = attention.squeeze(0)[i].cpu().detach().numpy()
      cax = ax.matshow(_attention, cmap='bone')

      ax.tick_params(labelsize=12)
      ax.set_xticklabels([''] + [t.lower() for t in sentence],
                        rotation=45)
      ax.set_yticklabels([''] + translation)

      ax.xaxis.set_major_locator(ticker.MultipleLocator(1))
      ax.yaxis.set_major_locator(ticker.MultipleLocator(1))
    
    plt.tight_layout()
    fig.subplots_adjust(hspace=0.25, wspace=0.25)
    plt.savefig(figure_path)
    
    
    