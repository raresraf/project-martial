
import json

with open('/Users/raresraf/code/project-martial/dataset/ground-truth-comments-6-kubernetes.txt', 'r') as f:
    data = json.load(f)

with open('/Users/raresraf/code/project-martial/dataset/annotated-ground-truth-comments-6-kubernetes.txt', 'w') as f:
    for i in range(1, len(data)+1):
        f.write(f"{i}:{data[str(i)]['comment']}\n")
        for j in data[str(i)]["similar_with"]:
            f.write(f"    {j}: {data[str(j)]['comment']}\n")
