import matplotlib.pyplot as plt
import numpy as np

threshold = [0.4, 0.42, 0.44, 0.46, 0.48, 0.5, 0.52, 0.54, 0.56, 0.58, 0.6]
f1 = [0.77, 0.771, 0.78, 0.781, 0.783, 0.788, 0.773, 0.773, 0.768, 0.757, 0.751]

plt.plot(threshold, f1, marker='o')
plt.xlabel('Threshold')
plt.ylabel('F1 score')
plt.title('F1 score of the model with respect to the similarity detection threshold')

plt.xticks(np.arange(0.4, 0.61, 0.02))
plt.yticks(np.arange(0.7, 0.85, 0.025))
plt.grid(True)
plt.show()