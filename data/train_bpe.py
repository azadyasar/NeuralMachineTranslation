import io
import sentencepiece as spm
import argparse

if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument('--file_path', required=True, type=str, help='path to the tsv file with format en - tr')
  args = parser.parse_args()
  
  lines = io.open(args.file_path, encoding='UTF-8').read().strip().split('\n')
  sentence_pairs = [[s for s in l.split('\t')] for l in lines]
  
  # Remove tsv headers
  del sentence_pairs[0]
  print(f"Nbr of sentence pairs: {len(sentence_pairs)}, Ex: {sentence_pairs[0]}")
  
  with open('tr_corpus.txt', 'w') as outTr:
    with open('en_corpus.txt', 'w') as outEn:
      for sent_pair in sentence_pairs:
        outEn.write(sent_pair[0].lower() + "\n")
        outTr.write(sent_pair[1].lower() + "\n")
        
  print("Training the source tr tokenizer")
  spm.SentencePieceTrainer.train(input="tr_corpus.txt", model_prefix="new_tr_sp", vocab_size=25000, character_coverage=1.,
                                 pad_id=3)
  print("Training the target en tokenizer")
  spm.SentencePieceTrainer.train(input="en_corpus.txt", model_prefix="new_en_sp", vocab_size=15000, character_coverage=1.,
                                 pad_id=3) 
  