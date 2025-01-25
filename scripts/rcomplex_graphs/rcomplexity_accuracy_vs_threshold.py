import matplotlib.pyplot as plt
import numpy as np

threshold = [0.4, 0.42, 0.44, 0.46, 0.48, 0.5, 0.52, 0.54, 0.56, 0.58, 0.6]
accuracy = [0.77, 0.772, 0.78, 0.781, 0.792, 0.799, 0.784, 0.776, 0.77, 0.761, 0.76]

plt.plot(threshold, accuracy, marker='o')
plt.xlabel('Threshold')
plt.ylabel('Accuracy')
plt.title('Accuracy of the model with respect to the similarity detection threshold')

plt.xticks(np.arange(0.4, 0.61, 0.02))
plt.yticks(np.arange(0.7, 0.85, 0.025))
plt.grid(True)
plt.show()