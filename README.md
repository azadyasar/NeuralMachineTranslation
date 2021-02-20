# Neural Machine Translation




### Train custom BPE tokenizers
In order to train a custom BPE model, you can run the *train_bpe* script by issuing the following command. sentences.tsv contain a tsv file with *en - tr* sentence pairs.
```
python data/train_bpe.py --file_path sentences.tsv
```