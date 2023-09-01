import numpy as np
from sklearn.metrics import pairwise_distances
from scipy.spatial.distance import cosine
from numpy import genfromtxt
import numpy as np
import matplotlib.pyplot as plt


start_id = [42, 1]
datafiles = [f'/Users/raresraf/code/project-martial/dataset/eslint-after-1-{i}.csv' for i in start_id]

fig, axs = plt.subplots(1,2, figsize=(12,6))

for ax in axs:
    # get data
    A = genfromtxt(datafiles.pop(), delimiter=',')
    # get rid of text column
    A = A[:,1:]
    # get rid of text line
    A = A[1:,:]

    data = 1-pairwise_distances(A, metric="cosine")

    cax = ax.imshow(data, interpolation='nearest')
    fig.colorbar(cax)

    start = start_id.pop()
    labels = [f'1.{i}.0' for i in range(start,79)]
    xaxis = np.arange(len(labels))
    ax.set_xticks(xaxis)
    ax.set_yticks(xaxis)
    ax.set_xticklabels(labels, rotation = 50)
    ax.set_yticklabels(labels)
    ax.set_xlabel("VScode release")
    ax.set_ylabel("VScode release")
    
    counter = start
    dist = int((79 - start) / 10)
    for label in ax.get_xaxis().get_ticklabels()[::1]:
        if counter % dist != start % dist:
    	    label.set_visible(False)
        counter = counter + 1
    counter = start
    for label in ax.get_yaxis().get_ticklabels()[::1]:
        if counter % dist != start % dist:
            label.set_visible(False)
        counter = counter + 1

fig.suptitle("Cosine-based similarity based on ESlint\n issues in various VScode releases",fontsize='medium')
plt.show()
