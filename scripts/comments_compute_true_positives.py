import json
from tqdm import tqdm
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report

"""
* * * NEW MODEL: levenshtein * * *
[[   76456    22690]
 [      16 12817674]]
              precision    recall  f1-score   support

 not similar    0.99823   1.00000   0.99912  12817690
     similar    0.99979   0.77115   0.87071     99146

    accuracy                        0.99824  12916836
   macro avg    0.99901   0.88557   0.93491  12916836
weighted avg    0.99824   0.99824   0.99813  12916836

* * * NEW MODEL: word2vec * * *
[[   98620      526]
 [    9920 12807770]]
              precision    recall  f1-score   support

 not similar    0.99996   0.99923   0.99959  12817690
     similar    0.90861   0.99469   0.94970     99146

    accuracy                        0.99919  12916836
   macro avg    0.95428   0.99696   0.97465  12916836
weighted avg    0.99926   0.99919   0.99921  12916836

* * * NEW MODEL: elmo * * *
[[  99130      16]
 [6881546 5936144]]
              precision    recall  f1-score   support

 not similar    1.00000   0.46312   0.63306  12817690
     similar    0.01420   0.99984   0.02800     99146

    accuracy                        0.46724  12916836
   macro avg    0.50710   0.73148   0.33053  12916836
weighted avg    0.99243   0.46724   0.62841  12916836

* * * NEW MODEL: roberta * * *
[[   77628    21518]
 [   16476 12801214]]
              precision    recall  f1-score   support

 not similar    0.99832   0.99871   0.99852  12817690
     similar    0.82492   0.78297   0.80339     99146

    accuracy                        0.99706  12916836
   macro avg    0.91162   0.89084   0.90096  12916836
weighted avg    0.99699   0.99706   0.99702  12916836

* * * NEW MODEL: use * * *
[[   97324     1822]
 [     140 12817550]]
              precision    recall  f1-score   support

 not similar    0.99986   0.99999   0.99992  12817690
     similar    0.99856   0.98162   0.99002     99146

    accuracy                        0.99985  12916836
   macro avg    0.99921   0.99081   0.99497  12916836
weighted avg    0.99985   0.99985   0.99985  12916836
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
    print(classification_report(actual, pred, target_names=['not similar', 'similar'], digits=5))
