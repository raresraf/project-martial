import matplotlib.pyplot as plt
import numpy as np

threshold = [0.4, 0.42, 0.44, 0.46, 0.48, 0.5, 0.52, 0.54, 0.56, 0.58, 0.6]
precision_of_identifying_similar = [
    0.751,
    0.762,
    0.781,
    0.794,
    0.803,
    0.811,
    0.832,
    0.846,
    0.856,
    0.865,
    0.871,
]
recall_of_identifying_similar = [
    0.821,
    0.806,
    0.785,
    0.764,
    0.747,
    0.728,
    0.701,
    0.683,
    0.653,
    0.631,
    0.618,
]


plt.plot(
    threshold,
    precision_of_identifying_similar,
    marker="o",
    label="Precision score of the model for identifying similar code",
)
plt.plot(
    threshold,
    recall_of_identifying_similar,
    marker="x",
    label="Recall score of the model for identifying similar code",
)
plt.xlabel("Threshold")
plt.ylabel("Metric")
plt.title("Performance of the model for identifying similar code")

plt.xticks(np.arange(0.4, 0.61, 0.02))
plt.yticks(np.arange(0.6, 0.9, 0.025))
plt.legend()

plt.grid(True)
plt.show()
