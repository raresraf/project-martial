from collections import Counter
from math import log
import re
import os

class NetworkTrafficAnalysis:
    def __init__(self, corpus_dir=None):
        self.file_contents = {}
        self.all_ngrams_documents = {n: [] for n in range(2, 5)}
        self.all_unique_ngrams_in_corpus = {n: set() for n in range(2, 5)}
        self.remove_alphanumeric = False

        if corpus_dir:
            self._load_corpus(corpus_dir)
        else:
            default_corpus_dir = os.path.join(os.path.dirname(__file__), '..', 'dataset', 'packets', 'network')
            self._load_corpus(default_corpus_dir)

    def _load_corpus(self, corpus_dir):
        if not os.path.isdir(corpus_dir):
            print(f"Warning: Corpus directory not found at {corpus_dir}. TF-IDF might not work correctly.")
            return

        print(f"Loading N-gram corpus from directory: {corpus_dir}")
        for root, _, files in os.walk(corpus_dir):
            for file_name in files:
                if file_name == 'combined.bin':
                    file_path = os.path.join(root, file_name)
                    try:
                        with open(file_path, 'rb') as f:
                            text_content = f.read()
                        if self.remove_alphanumeric:
                            text_content = re.sub(b'[a-zA-Z0-9]', b'', text_content)

                        for n in range(2, 5):
                            ngrams_in_doc = self._generate_ngrams(text_content, n)
                            if ngrams_in_doc:
                                self.all_ngrams_documents[n].append(ngrams_in_doc)
                                self.all_unique_ngrams_in_corpus[n].update(ngrams_in_doc)
                        print(f"  - Loaded {file_path}")
                    except Exception as e:
                        print(f"Error loading corpus file {file_path}: {e}")
        print("N-gram corpus loading complete.")

    def load_file_content(self, file_id, content_string):
        self.file_contents[file_id] = content_string.encode('latin-1')

    def _generate_ngrams(self, text_bytes, n):
        ngrams = []
        if len(text_bytes) < n:
            return ngrams
        for i in range(len(text_bytes) - n + 1):
            ngrams.append(text_bytes[i:i + n])
        return ngrams

    def _calculate_tf_idf(self, ngrams_in_doc, n_val):
        tf_idf = {}
        if not ngrams_in_doc:
            return tf_idf

        ngram_counts = Counter(ngrams_in_doc)
        
        num_docs_in_corpus = len(self.all_ngrams_documents[n_val])
        if num_docs_in_corpus == 0:
            for ngram in ngram_counts:
                tf = ngram_counts[ngram] / len(ngrams_in_doc)
                tf_idf[ngram] = tf * 1
            return tf_idf

        for ngram in ngram_counts:
            tf = ngram_counts[ngram] / len(ngrams_in_doc)
            
            docs_containing_ngram = sum(1 for doc_ngrams_list in self.all_ngrams_documents[n_val] if ngram in doc_ngrams_list)
            
            idf = log(num_docs_in_corpus / (docs_containing_ngram + 1e-9))
            
            tf_idf[ngram] = tf * idf
        return tf_idf

    def _calculate_cosine_similarity(self, tfidf1, tfidf2):
        dot_product = sum(tfidf1.get(ngram, 0) * tfidf2.get(ngram, 0) for ngram in set(tfidf1) | set(tfidf2))
        magnitude1 = sum(value ** 2 for value in tfidf1.values()) ** 0.5
        magnitude2 = sum(value ** 2 for value in tfidf2.values()) ** 0.5
        return dot_product / (magnitude1 * magnitude2) if magnitude1 and magnitude2 else 0

    def compare_network_traffic(self, file1_id, file2_id, n_gram_selection=4):
        text1_bytes = self.file_contents.get(file1_id)
        text2_bytes = self.file_contents.get(file2_id)

        if not text1_bytes or not text2_bytes:
            return {"error": "Missing file content for comparison."}

        if self.remove_alphanumeric:
            text1_bytes = re.sub(b'[a-zA-Z0-9]', b'', text1_bytes)
            text2_bytes = re.sub(b'[a-zA-Z0-9]', b'', text2_bytes)

        results = {}
        for n in range(2, 5):
            ngrams1 = self._generate_ngrams(text1_bytes, n)
            ngrams2 = self._generate_ngrams(text2_bytes, n)

            tfidf1 = self._calculate_tf_idf(ngrams1, n)
            tfidf2 = self._calculate_tf_idf(ngrams2, n)

            similarity = self._calculate_cosine_similarity(tfidf1, tfidf2)
            results[f"{n}-gram_similarity"] = similarity
        
        results["similarity"] = results.get(f"{n_gram_selection}-gram_similarity", 0)
        results["selected_ngram"] = n_gram_selection
        
        results["identical"] = [] 
        results["complexity"] = []

        return results

