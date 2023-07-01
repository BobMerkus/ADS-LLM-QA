import re
import hashlib
import regex as re
from collections import Counter
import string
from transformers import AutoTokenizer

PATTERNS = {
    'phone_number' : r'^\+?[1-9]\d{1,14}$',
    'url' : r'^http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+',
    'email' : r'^[a-zA-Z0-9]+(?:\.[a-zA-Z0-9]+)*@[a-zA-Z0-9]+(?:\.[a-zA-Z0-9]+)*$'
}

def split_chunks(text:str, chunk_size=1000, overlap=100) -> list[str]:
    """Split a string into equal chunks of size `chunk_size`, with characters overlapping by `overlap`. Returns a list of strings.

    Args:
        text (str): The text to split into chunks.
        chunk_size (int, optional): Defaults to 1000.
        overlap (int, optional):  Defaults to 100.

    Returns:
        list[str]: The list of chunks is returned.
    """
    chunks = [text[i-overlap:i+chunk_size] if i>overlap else text[i:i+chunk_size] for i in range(0, len(text), chunk_size)]
    return chunks

def hash_md5(text:str) -> str:
  return hashlib.md5(text.encode()).hexdigest()

def match(text:str|list[str], patterns:dict[str:str]=PATTERNS) -> bool:
  "Use regex matching to find specific things"
  if not isinstance(text, str or isinstance(text, list)):
      raise TypeError('Text must be a string or list.')
  text = ' '.join(text) if isinstance(text, list) else text
  matches = {key:re.compile(pattern).findall(text) for key, pattern in patterns.items()}
  return {k:v for k, v in matches.items() if v}

def normalize_answer(s):
  """Lower text and remove punctuation, articles and extra whitespace. From: Official evaluation script for SQuAD version 2.0. 
  This implementation removes trailing .0 before removing punctuation, so numerical values are compared fairly (e.g. 1.0 -> 1 and 32.00 -> 32).)
  Source: 
  - https://github.com/white127/SQUAD-2.0-bidaf/blob/master/evaluate-v2.0.py
  - https://qa.fastforwardlabs.com/no%20answer/null%20threshold/bert/distilbert/exact%20match/f1/robust%20predictions/2020/06/09/Evaluating_BERT_on_SQuAD.html#F1.
  """
  def remove_articles(text):
    regex = re.compile(r'\b(a|an|the)\b', re.UNICODE)
    return re.sub(regex, ' ', text)
  def white_space_fix(text):
    return ' '.join(text.split())
  def remove_punc(text):
    exclude = set(string.punctuation)
    return ''.join(ch for ch in text if ch not in exclude)
  def lower(text):
    return text.lower()
  def remove_trailing_zeros(text):
    regex_trailing_zero = re.compile(r'^(\d+(?:\.\d*?[1-9](?=0|\b))?)\.?0*$')
    return regex_trailing_zero.sub(r'\1', text)
  return white_space_fix(remove_articles(remove_punc(lower(remove_trailing_zeros(s)))))

def f1_score(text:str, reference_text:str, tokenizer:AutoTokenizer=None) -> float:
  """Computes the F1 score of two strings. 
  From: Official evaluation script for SQuAD version 2.0. 
  Source: https://github.com/white127/SQUAD-2.0-bidaf/blob/master/evaluate-v2.0.py"""
  text = normalize_answer(text)
  reference_text = normalize_answer(reference_text)
  if tokenizer:
    text = tokenizer(text, return_tensors='pt').input_ids.squeeze(0).tolist()
    reference_text = tokenizer(reference_text, return_tensors='pt').input_ids.squeeze(0).tolist()
  else:
    text = text.split()
    reference_text = reference_text.split()
  common = Counter(text) & Counter(reference_text)
  num_same = sum(common.values())
  if num_same == 0:
      return 0
  precision = 1.0 * num_same / len(text)
  recall = 1.0 * num_same / len(reference_text)
  f1 = (2 * precision * recall) / (precision + recall)
  assert 0 <= f1 <= 1
  return f1

if __name__=="__main__":
  tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
  f1_score_nlp('1.0', '1', tokenizer)
  f1_score_nlp('Hello my name is John the third', 'Hello my name is John', tokenizer)
  f1_score_nlp('Hello my name is John the third', 'Hello my name is John')