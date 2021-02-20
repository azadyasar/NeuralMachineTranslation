from torchtext.data import BucketIterator, TabularDataset
from nmt.data import Vocabulary

class Dataset(object):
  def __init__(self,
               train_path: str,
               eval_path: str,
               test_path: str,
               src_vocab: Vocabulary,
               trg_vocab: Vocabulary):
    data_fields = [('trg', trg_vocab), ('src', src_vocab)]
    self.train_data, self.val_data, self.test_data = TabularDataset.splits(path='./',
                                                                                train=train_path,
                                                                                validation=eval_path,
                                                                                test=test_path,
                                                                                format='tsv',
                                                                                fields=data_fields,
                                                                                skip_header=True)
    
  def create_iterators(self, device: str = 'cpu', batch_sz: int = 128):
    train_iterator, valid_iterator, test_iterator = BucketIterator.splits(
        (self.train_data, self.val_data, self.test_data),
        batch_size=batch_sz,
        sort_within_batch=False,
        sort=False,
        device=device)
    return train_iterator, valid_iterator, test_iterator
    