import json
from tqdm import tqdm
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report

"""
* * * NEW MODEL: levenshtein * * *
[[   76456    22690]
 [      16 12817674]]
              precision    recall  f1-score   support

 not similar       1.00      1.00      1.00  12817690
     similar       1.00      0.77      0.87     99146

    accuracy                           1.00  12916836
   macro avg       1.00      0.89      0.93  12916836
weighted avg       1.00      1.00      1.00  12916836

* * * NEW MODEL: word2vec * * *
[[   98620      526]
 [    9920 12807770]]
              precision    recall  f1-score   support

 not similar       1.00      1.00      1.00  12817690
     similar       0.91      0.99      0.95     99146

    accuracy                           1.00  12916836
   macro avg       0.95      1.00      0.97  12916836
weighted avg       1.00      1.00      1.00  12916836

* * * NEW MODEL: elmo * * *
[[  99130      16]
 [6881546 5936144]]
              precision    recall  f1-score   support

 not similar       1.00      0.46      0.63  12817690
     similar       0.01      1.00      0.03     99146

    accuracy                           0.47  12916836
   macro avg       0.51      0.73      0.33  12916836
weighted avg       0.99      0.47      0.63  12916836

* * * NEW MODEL: roberta * * *
[[   77628    21518]
 [   16476 12801214]]
              precision    recall  f1-score   support

 not similar       1.00      1.00      1.00  12817690
     similar       0.82      0.78      0.80     99146

    accuracy                           1.00  12916836
   macro avg       0.91      0.89      0.90  12916836
weighted avg       1.00      1.00      1.00  12916836

* * * NEW MODEL: use * * *
[[   97324     1822]
 [     140 12817550]]
              precision    recall  f1-score   support

 not similar       1.00      1.00      1.00  12817690
     similar       1.00      0.98      0.99     99146

    accuracy                           1.00  12916836
   macro avg       1.00      0.99      0.99  12916836
weighted avg       1.00      1.00      1.00  12916836
"""

DEBUG = False


with open('/Users/raresraf/code/project-martial/dataset/ground-truth-comments-6-kubernetes.txt', 'r') as f:
    data_actual = json.load(f)



for model in ["levenshtein", "word2vec", "elmo", "roberta", "use"]:
    print(f"* * * NEW MODEL: {model} * * *")
    with open(f'/Users/raresraf/code/project-martial/dataset/{model}-comments-6-kubernetes.txt', 'r') as f:
        data_pred = json.load(f)

    pred = []
    actual = []

    total_len = len(data_actual)
    for k, v in data_actual.items():
        v_pred = data_pred[k]
        for i in range(1, total_len+1):
            if not str(i) in data_actual:
                continue 
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