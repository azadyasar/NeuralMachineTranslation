from nmt.training import TransformerModelConfig
from nmt.util import get_lr, epoch_time
import torch
import torch.nn as nn
import torch.optim as optim
from torchtext.data import BucketIterator
from tqdm import tqdm
import time
import datetime
import numpy as np

import logging
logger = logging.getLogger("Trainer")

class Trainer(object):
  def __init__(self,
               config: TransformerModelConfig,
               learning_rate: float,
               weight_decay: float,
               n_epochs: int,
               clip: float = 1.):
    self.config = config
    self.model = config.create_model()
    self.n_epochs = n_epochs
    self.clip = clip
    
    self.optimizer = optim.Adam(self.model.parameters(), lr = learning_rate, weight_decay=weight_decay)
    self.scheduler = optim.lr_scheduler.StepLR(self.optimizer, step_size=2, gamma=0.7)
    self.criterion = nn.CrossEntropyLoss(ignore_index=self.config.trg_pad_idx)
    
    
  def train(self,
            train_iterator: BucketIterator,
            valid_iterator: BucketIterator):
    best_valid_loss = float('inf')
    train_losses, valid_losses = [], []
    
    for epoch in range(self.n_epochs):
        start_time = time.time()
        
        train_loss = self.train_(train_iterator)
        torch.cuda.empty_cache()
        valid_loss = self.evaluate_(valid_iterator)
        lr = get_lr(self.optimizer)
        self.scheduler.step()

        end_time = time.time()
        epoch_mins, epoch_secs = epoch_time(start_time, end_time)
        
        if valid_loss < best_valid_loss:
            best_valid_loss = valid_loss
            torch.save(self.model.state_dict(), self.config.save_model_path)

        train_losses.append(train_loss)
        valid_losses.append(valid_loss)
        logger.info(datetime.datetime.utcnow())
        logger.info(f'Epoch: {epoch+1:02} | Time: {epoch_mins}m {epoch_secs}s. Learning rate = {lr:1.8f}')
        logger.info(f'\tTrain Loss: {train_loss:.4f} | Train PPL: {np.exp(train_loss):7.3f}')
        logger.info(f'\t Val. Loss: {valid_loss:.4f} |  Val. PPL: {np.exp(valid_loss):7.3f}')
        logger.info("---" * 50)
    
  def train_(self, iterator: BucketIterator):
    self.model.train()
    epoch_loss = 0
    
    for i, batch in tqdm(enumerate(iterator)):  
        src = batch.src
        trg = batch.trg
        
        self.optimizer.zero_grad()        
        output, _ = self.model(src, trg[:, :-1])
        # trg = [batch_sz, trg_len]
        # output = [batch_sz, trg_len - 1, output_dim]

        output_dim = output.shape[-1]

        output = output.contiguous().view(-1, output_dim)
        trg = trg[:,1:].contiguous().view(-1)
        # output = [batch_sz * trg_len - 1, output_dim]
        # trg = [batch_sz * trg_len - 1]
      
        loss = self.criterion(output, trg)
        loss.backward()

        torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.clip)
        self.optimizer.step()
        
        epoch_loss += loss.item()   

    return epoch_loss / len(iterator)
  
  def evaluate_(self, iterator: BucketIterator):
    self.model.eval()
    epoch_loss = 0
    
    with torch.no_grad():
        for batch in tqdm(iterator):
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
        
    return epoch_loss / len(iterator)