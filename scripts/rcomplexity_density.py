import numpy as np
import matplotlib.pyplot as plt

data = [0.5061,0.3848,0.0619,0.0,0.3222,0.1076,1.0,0.6837,0.3761,0.4154,0.0,0.0,0.4206,0.6619,0.027,0.0,0.0325,0.2204,0.0195,0.0,0.0,0.1413,0.2458,0.2237,0.3058,0.3331,0.0,0.0,0.0465,0.0631,0.0,0.0,0.1697,0.1808,0.0,0.0]

data_array = np.array(data).reshape(9, 4)

plt.figure(figsize=(3, 6)) 

plt.imshow(data_array, cmap='viridis', interpolation='nearest')
plt.colorbar()

plt.title("Parameters of the model\nused for detecting code similarity,\nusing complexity-based birthmarks")

plt.xlabel("$c_{i}$") 
plt.ylabel("$f_{j}$")

plt.xticks(range(data_array.shape[1]), range(1, data_array.shape[1] + 1))
plt.yticks(range(data_array.shape[0]), range(1, data_array.shape[0] + 1))


plt.show()