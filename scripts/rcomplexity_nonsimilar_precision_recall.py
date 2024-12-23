import matplotlib.pyplot as plt
import numpy as np

threshold = [0.4, 0.42, 0.44, 0.46, 0.48, 0.5, 0.52, 0.54, 0.56, 0.58, 0.6]
precision_of_identifying_not_similar = [
0.802, 
0.793, 
0.784, 
0.773, 
0.765, 
0.754, 
0.743, 
0.735, 
0.726, 
0.711, 
0.706, 
]
recall_of_identifying_not_similar = [
0.72,
0.751,
0.782,
0.804,
0.825,
0.833,
0.855,
0.873,
0.891,
0.902,
0.911,
]
  


plt.plot(threshold, precision_of_identifying_not_similar, marker="o", label="Precision score of the model for identifying not-similar code")
plt.plot(threshold, recall_of_identifying_not_similar, marker="x", label="Recall score of the model for identifying not-similar code")
plt.xlabel("Threshold")
plt.ylabel("Metric")
plt.title("Performance of the model for identifying not-similar code")

plt.xticks(np.arange(0.4, 0.61, 0.02))
plt.yticks(np.arange(0.6, 0.9, 0.025))
plt.legend()

plt.grid(True)
plt.show()
