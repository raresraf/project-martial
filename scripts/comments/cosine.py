import numpy as np
from sklearn.metrics import pairwise_distances
from scipy.spatial.distance import cosine
from numpy import genfromtxt
import numpy as np
import matplotlib.pyplot as plt
import matplotlib

start_id = [42, 1]
datafiles = [f'/Users/raresraf/code/project-martial/dataset/eslint-after-1-{i}.csv' for i in start_id]

fig, axs = plt.subplots(1,2, figsize=(12,6))

matplotlib.rcParams.update({'font.size': 16})

one_y_label_only = True
for ax in axs:
    # get data
    A = genfromtxt(datafiles.pop(), delimiter=',')
    # get rid of text column
    A = A[:,1:]
    # get rid of text line
    A = A[1:,:]

    data = 1-pairwise_distances(A, metric="cosine")
    print(data)

    cax = ax.imshow(data, interpolation='nearest')
    fig.colorbar(cax)

    start = start_id.pop()
    labels = [f'1.{i}.0' for i in range(start,79)]
    xaxis = np.arange(len(labels))
    ax.set_xticks(xaxis)
    ax.set_yticks(xaxis)
    ax.set_xticklabels(labels, rotation = 50)
    ax.set_yticklabels(labels)
    ax.set_xlabel("VSCode release")
    if one_y_label_only:
        ax.set_ylabel("VSCode release")
        one_y_label_only = False

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

    for item in (ax.get_xticklabels() + ax.get_yticklabels()):
        item.set_fontsize(14)


    for item in ([ax.title, ax.xaxis.label, ax.yaxis.label]):
        item.set_fontsize(16)

fig.suptitle("Cosine-based similarity based on ESlint\n issues in various VSCode releases",fontsize='medium')
plt.show()
