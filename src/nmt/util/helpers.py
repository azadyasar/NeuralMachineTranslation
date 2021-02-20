import pickle
import torch
import torch.nn as nn

def load_vocab(path):
  with open(path, 'rb') as inFile:
    return pickle.load(inFile)
  
def save_vocab(vocab, path):
    with open(path, 'wb') as output:
      pickle.dump(vocab, output)
  
def count_parameters(model: nn.Module):
  return sum(p.numel() for p in model.parameters() if p.requires_grad)

def init_weights(model: nn.Module):
  if hasattr(model, 'weight') and model.weight.dim() > 1:
    nn.init.xavier_uniform_(model.weight.data)
    
def get_device() -> str:
  return torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def epoch_time(start_time, end_time):
    elapsed_time = end_time - start_time
    elapsed_mins = int(elapsed_time / 60)
    elapsed_secs = int(elapsed_time - (elapsed_mins * 60))
    return elapsed_mins, elapsed_secs
  
def get_lr(optimizer: torch.optim):
    for param_group in optimizer.param_groups:
        return param_group['lr']