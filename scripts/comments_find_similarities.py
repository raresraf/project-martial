import datetime
from rapidfuzz import fuzz
import tempfile
import en_core_web_lg
from sklearn.metrics.pairwise import cosine_similarity
from simple_elmo import ElmoModel
from spacy.tokens import Doc
import numpy as np
import tensorflow as tf
import tensorflow_hub as hub
from sentence_transformers import SentenceTransformer, util
import json
from tqdm import tqdm

model = "elmo" # can be "levenshtein", "word2vec", "elmo", "roberta", "use"
threshold = 0.98

with open('/Users/raresraf/code/project-martial/dataset/comments-6-kubernetes.txt', 'r') as f:
    data = json.load(f)

if model == "word2vec":    
    spacy_core_web = en_core_web_lg.load()
if model == "elmo":    
    tf.compat.v1.reset_default_graph()
    elmo = ElmoModel()
    elmo.load("/Users/raresraf/code/project-martial/209")
if model == "roberta":
    roberta = SentenceTransformer('stsb-roberta-large')
if model == "use":
    use = hub.load("https://tfhub.dev/google/universal-sentence-encoder/4")

total_len = len(data)
with tqdm(total=total_len, desc="Generating embeddings") as pbar:
    for k, v in data.items():
        if model == "levenshtein":
            continue
        if model == "word2vec":
            embd = spacy_core_web(v["comment"])
        if model == "elmo":
            long_comm_tensor = elmo.get_elmo_vectors(v["comment"])
            long_comm_tensor_avged = np.average(long_comm_tensor, axis=0)
            embd = long_comm_tensor_avged
        if model == "roberta":
            embd = roberta.encode([v["comment"]], convert_to_tensor=True)
        if model == "use":
            embd = use([v["comment"]])
        v["embd"] = embd
        pbar.update(1)        
          
with tqdm(total=total_len, desc="Processing embeddings") as pbar:        
    for k1, v1 in data.items():
        pbar.update(1)
        for k2, v2 in data.items():
            if model == "levenshtein":
                similarity = fuzz.ratio(v1["comment"], v2["comment"])
            if model == "word2vec":
                similarity = v1["embd"].similarity(v2["embd"])
            else:
                embd1 = v1["embd"]
                embd2 = v2["embd"]
                similarity = cosine_similarity(embd1, embd2)
            if similarity > threshold:
                v1["similar_with"].append(int(k2))
                v2["similar_with"].append(int(k1))
            
              
for k, v in data.items():
    if "embd" in v:
        del v["embd"]
    v["similar_with"] = list(set(v["similar_with"]))
      
with open(f'/Users/raresraf/code/project-martial/dataset/{model}-comments-6-kubernetes.txt', 'w') as f:
    json.dump(data, f, indent=4, sort_keys=True)
