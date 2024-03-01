import datetime
from rapidfuzz import fuzz
import tempfile
from sklearn.metrics.pairwise import cosine_similarity
from simple_elmo import ElmoModel
from spacy.tokens import Doc
import numpy as np
import tensorflow as tf
import tensorflow_hub as hub
from sentence_transformers import SentenceTransformer, util
import json
from tqdm import tqdm


with open('/Users/raresraf/code/project-martial/dataset/comments-6-kubernetes.txt', 'r') as f:
    data = json.load(f)
    
use = hub.load("https://tfhub.dev/google/universal-sentence-encoder/4")

total_len = len(data)
with tqdm(total=total_len, desc="Generating embeddings") as pbar:
    for k, v in data.items():
        embd = use([v["comment"]])
        v["embd"] = embd
        pbar.update(1)        
        
        
with tqdm(total=total_len, desc="Processing embeddings") as pbar:        
    for k1, v1 in data.items():
        pbar.update(1)
        if not "embd" in v1:
           continue 
        for k2, v2 in data.items():
            if not "embd" in v2:
                continue 
            embd1 = v1["embd"]
            embd2 = v2["embd"]
            similarity = cosine_similarity(embd1, embd2)
            if similarity > 0.9:
                v1["similar_with"].append(int(k2))
                v2["similar_with"].append(int(k1))
            
            
      
for k, v in data.items():
    if "embd" in v:
        del v["embd"]
    v["similar_with"] = list(set(v["similar_with"]))
      
with open('/Users/raresraf/code/project-martial/dataset/use-comments-6-kubernetes.txt', 'w') as f:
    json.dump(data, f, indent=4, sort_keys=True)
