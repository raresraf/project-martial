from collections import Counter
from math import log

def generate_ngrams(text, n):
  """Generates n-grams from a given text.

  Args:
    text: The input text as a string.
    n: The length of the n-grams to generate.

  Returns:
    A list of n-grams.
  """
  text = [bytes([i]) for i in text]
  ngrams = []
  for i in range(len(text) - n + 1):
    ngrams.append(b''.join(text[i:i + n]))
  return ngrams

def calculate_tf_idf(ngrams, all_ngrams):
  """Calculates the TF-IDF weights for a set of n-grams.

  Args:
    ngrams: The list of n-grams for a single document.
    all_ngrams: A list of all n-grams from all documents.

  Returns:
    A dictionary of n-grams and their TF-IDF weights.
  """
  tf_idf = {}
  ngram_counts = Counter(ngrams)
  num_docs = len(all_ngrams)
  for ngram in ngram_counts:
    tf = ngram_counts[ngram] / len(ngrams)
    idf = log(num_docs / sum(1 for doc_ngrams in all_ngrams if ngram in doc_ngrams))
    tf_idf[ngram] = tf * idf
  return tf_idf

def calculate_cosine_similarity(tfidf1, tfidf2):
  """Calculates the cosine similarity between two TF-IDF vectors.

  Args:
    tfidf1: The first TF-IDF vector.
    tfidf2: The second TF-IDF vector.

  Returns:
    The cosine similarity score as a float between 0 and 1.
  """
  dot_product = sum(tfidf1.get(ngram, 0) * tfidf2.get(ngram, 0) for ngram in set(tfidf1) | set(tfidf2))
  magnitude1 = sum(value ** 2 for value in tfidf1.values()) ** 0.5
  magnitude2 = sum(value ** 2 for value in tfidf2.values()) ** 0.5
  return dot_product / (magnitude1 * magnitude2) if magnitude1 and magnitude2 else 0

def compare_files_tfidf(file1_path, file2_path, all_file_path):
  """Compares two files using TF-IDF and cosine similarity for 2-grams, 3-grams, and 4-grams.

  Args:
    file1_path: The path to the first file.
    file2_path: The path to the second file.
  """
  with open(file1_path, 'rb') as file1, open(file2_path, 'rb') as file2, open(all_file_path, 'rb') as fileall:
    text1 = file1.read()
    text2 = file2.read()
    textall = fileall.read()

  for n in range(2, 5):
    ngrams1 = generate_ngrams(text1, n)
    ngrams2 = generate_ngrams(text2, n)
    all_ngrams = generate_ngrams(textall, n)

    tfidf1 = calculate_tf_idf(ngrams1, all_ngrams)
    tfidf2 = calculate_tf_idf(ngrams2, all_ngrams)

    similarity = calculate_cosine_similarity(tfidf1, tfidf2)
    print(f"{n}-gram TF-IDF similarity: {similarity:.4f}")

if __name__ == "__main__":
  file1_path = '/Users/raresraf/code/project-martial/experimental/network/mysql_8_0/scenarios/scenario_1/dump_1.bin'
  file2_path = '/Users/raresraf/code/project-martial/experimental/network/mysql_8_0_gcp/scenarios/scenario_1/dump_2.bin'
  all_file_path = '/Users/raresraf/code/project-martial/experimental/network/combined.txt'
  compare_files_tfidf(file1_path, file2_path, all_file_path)

