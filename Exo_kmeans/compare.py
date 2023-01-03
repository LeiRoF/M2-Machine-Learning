from config import *
import numpy as np
import matplotlib.pyplot as plt

data = np.loadtxt(DATA, delimiter=delimiter)
results = np.loadtxt(RESULTS)

x = np.arange(data.shape[0])
data = data[:, -1]
results = results[:, -1]

plt.scatter(x, data, c=results)
plt.show()
