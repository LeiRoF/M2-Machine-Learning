import numpy as np
import matplotlib.pyplot as plt

data = np.loadtxt("iris.txt")

# Plot the data
plt.figure(figsize=(18,12))
plt.subplot(231)
plt.scatter(data[:,0], data[:,1], c=data[:,-1], cmap=plt.cm.Paired)
plt.xlabel("1")
plt.ylabel("2")
plt.colorbar()

plt.subplot(232)
plt.scatter(data[:,0], data[:,2], c=data[:,-1], cmap=plt.cm.Paired)
plt.xlabel("1")
plt.ylabel("3")
plt.colorbar()

plt.subplot(233)
plt.scatter(data[:,0], data[:,3], c=data[:,-1], cmap=plt.cm.Paired)
plt.xlabel("1")
plt.ylabel("4")
plt.colorbar()

plt.subplot(234)
plt.scatter(data[:,1], data[:,2], c=data[:,-1], cmap=plt.cm.Paired)
plt.xlabel("2")
plt.ylabel("3")
plt.colorbar()

plt.subplot(235)
plt.title("2 with 4")
plt.scatter(data[:,1], data[:,3], c=data[:,-1], cmap=plt.cm.Paired)
plt.xlabel("2")
plt.ylabel("4")
plt.colorbar()

plt.subplot(236)
plt.scatter(data[:,2], data[:,3], c=data[:,-1], cmap=plt.cm.Paired)
plt.xlabel("3")
plt.ylabel("4")
plt.colorbar()

plt.show()