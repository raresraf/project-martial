import numpy as np
import matplotlib.pyplot as plt 


x = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
y_generate = [1.22, 2.08, 3.04, 4.1, 5.23, 6.66, 7.43, 8.33, 9.32, 10.01]
y_process = [254, 795, 1866, 3920, 5123, 7344, 9457, 11654, 14902, 17778]
y_embs = [788, 1556, 2304, 3032, 3740, 4428, 5096, 5744, 6372, 6980]


new_x = np.linspace(min(x), max(x), num=1000)
coefs_generate = np.polyfit(x,y_generate,1)
new_line_generate = np.polyval(coefs_generate, new_x)

coefs_process = np.polyfit(x,y_process,2)
new_line_process = np.polyval(coefs_process, new_x)

coefs_embs = np.polyfit(x,y_embs,1)
new_line_embs = np.polyval(coefs_embs, new_x)

fig, axes = plt.subplots(1, 3, figsize=(15, 5))

axes[0].scatter(new_x,new_line_generate,c='r', marker='^', s=0.2)
axes[0].scatter(x,y_generate, marker='X', s=48)
axes[0].set_xlim(0,max(x)+1)
axes[0].set_ylim(0,max(y_generate) * 110 / 100)
axes[0].set_xlabel('r', fontsize=12)
axes[0].set_ylabel('Time for generating USE embeddings (seconds)', fontsize=12)
axes[0].minorticks_on()
axes[0].xaxis.grid(True, which='major', linestyle='--')
axes[0].yaxis.grid(True, which='major', linestyle='--')


axes[1].scatter(x,y_process)
axes[1].scatter(new_x,new_line_process,c='r', marker='^', s=0.2)
axes[1].set_xlim(0,max(x)+1)
axes[1].set_ylim(0,max(y_process) * 110 / 100)
axes[1].set_xlabel('r', fontsize=12)
axes[1].set_ylabel('Time for processing USE embeddings (seconds)', fontsize=12)
axes[1].minorticks_on()
axes[1].xaxis.grid(True, which='major', linestyle='--')
axes[1].yaxis.grid(True, which='major', linestyle='--')

axes[2].scatter(x,y_embs)
axes[2].scatter(new_x,new_line_embs,c='r', marker='^', s=0.2)
axes[2].set_xlim(0,max(x)+1)
axes[2].set_ylim(0,max(y_embs) * 110 / 100)
axes[2].set_xlabel('r', fontsize=12)
axes[2].set_ylabel('Number of embeddings generated', fontsize=12)
axes[2].minorticks_on()
axes[2].xaxis.grid(True, which='major', linestyle='--')
axes[2].yaxis.grid(True, which='major', linestyle='--')

plt.tight_layout()
plt.show()