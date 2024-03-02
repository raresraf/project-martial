import json
from tqdm import tqdm
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report

DEBUG = True


with open('/Users/raresraf/code/project-martial/dataset/use-comments-6-kubernetes.txt', 'r') as f:
    data_actual = json.load(f)
    
with open('/Users/raresraf/code/project-martial/dataset/elmo-comments-6-kubernetes.txt', 'r') as f:
    data_pred = json.load(f)

pred = []
actual = []

total_len = len(data_actual)
for k, v in data_actual.items():
    v_pred = data_pred[k]
    for i in range(1, total_len+1):
        if i in v["similar_with"]:
            a = 1
        else:
            a = 0
        actual.append(a)
        if i in v_pred["similar_with"]:
            p = 1
        else:
            p = 0
        pred.append(p)
        
        c1 = v["comment"]
        c2 = data_actual[str(i)]["comment"]
        if DEBUG and a != p and (not c1 in c2) and (not c2 in c1):
            print("* * * * *")
            print(c1)
            print(c2)
            print(f"actual: {a}, pred: {p}")
            
        
print(confusion_matrix(actual, pred, labels=[1, 0]))
print(classification_report(actual, pred, target_names=['not similar', 'similar']))