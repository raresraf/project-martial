import numpy as np
import matplotlib.pyplot as plt 
import matplotlib.gridspec as gridspec

fig = plt.figure(figsize=(15, 10))

gs = gridspec.GridSpec(2, 4)
gs.update(wspace=0.5)

x = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
y_generate = [1.22, 2.08, 3.04, 4.1, 5.23, 6.66, 7.43, 8.33, 9.32, 10.01]
y_process = [254, 795, 1866, 3920, 5123, 7344, 9457, 11654, 14902, 17778]
y_embs = [.788, 1.556, 2.304, 3.032, 3.740, 4.428, 5.096, 5.744, 6.372, 6.980]


new_x = np.linspace(min(x), max(x), num=1000)
coefs_generate = np.polyfit(x,y_generate,1)
new_line_generate = np.polyval(coefs_generate, new_x)

coefs_process = np.polyfit(x,y_process,2)
new_line_process = np.polyval(coefs_process, new_x)

coefs_embs = np.polyfit(x,y_embs,1)
new_line_embs = np.polyval(coefs_embs, new_x)

ax1 = fig.add_subplot(gs[0, 1:3])
ax2 = fig.add_subplot(gs[1, :2])
ax3 = fig.add_subplot(gs[1, 2:])

ax1.scatter(new_x,new_line_embs,c='r', marker='^', s=0.2)
ax1.scatter(x,y_embs, marker='X', s=48)
ax1.set_xlim(0,max(x)+1)
ax1.set_ylim(0,max(y_embs) * 110 / 100)
ax1.set_xlabel('r', fontsize=12)
ax1.set_ylabel('Number of embeddings generated\n(thousands)', fontsize=12)
ax1.minorticks_on()
ax1.xaxis.grid(True, which='major', linestyle='--')
ax1.yaxis.grid(True, which='major', linestyle='--')


ax2.scatter(new_x,new_line_generate,c='r', marker='^', s=0.2)
ax2.scatter(x,y_generate, marker='X', s=48)
ax2.set_xlim(0,max(x)+1)
ax2.set_ylim(0,max(y_generate) * 110 / 100)
ax2.set_xlabel('r', fontsize=12)
ax2.set_ylabel(r'Time for $\mathbf{generating}$ the USE embeddings' '\n(seconds)', fontsize=12)
ax2.minorticks_on()
ax2.xaxis.grid(True, which='major', linestyle='--')
ax2.yaxis.grid(True, which='major', linestyle='--')


ax3.scatter(new_x,new_line_process,c='r', marker='^', s=0.2)
ax3.scatter(x,y_process, marker='X', s=48)
ax3.set_xlim(0,max(x)+1)
ax3.set_ylim(0,max(y_process) * 110 / 100)
ax3.set_xlabel('r', fontsize=12)
ax3.set_ylabel(r'Time for $\mathbf{processing}$ the USE embeddings' '\n(seconds)', fontsize=12)
ax3.minorticks_on()
ax3.xaxis.grid(True, which='major', linestyle='--')
ax3.yaxis.grid(True, which='major', linestyle='--')
ax3.yaxis.set_label_position("right")
ax3.yaxis.tick_right()

fig.tight_layout()
plt.show()