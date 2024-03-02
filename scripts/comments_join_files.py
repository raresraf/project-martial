import json

with open('/Users/raresraf/code/project-martial/dataset/comments-6-kubernetes.txt', 'r') as f:
    data_1 = json.load(f)
    
with open('/Users/raresraf/code/project-martial/dataset/levenshtein-comments-6-kubernetes.txt', 'r') as f:
    data_2 = json.load(f)

data = {}
total_len = len(data_1)
for k, v in data_1.items():
    v["similar_with"].extend(data_2[k]["similar_with"])
    v["similar_with"].sort()
    data[k] = {"comment": v["comment"], "similar_with": list(set(v["similar_with"]))}
    
with open('/Users/raresraf/code/project-martial/dataset/ground-truth-comments-6-kubernetes.txt', 'w') as f:
    json.dump(data, f, indent=4, sort_keys=True)