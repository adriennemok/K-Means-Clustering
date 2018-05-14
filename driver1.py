import numpy as np
import matplotlib.pyplot as plt
import kmeans

data = np.genfromtxt('iris.data', delimiter=',', dtype=np.float32)
attrs = data[:, 0:4]
attr_names = {0: 'sepal length', 1: 'sepal width', 2: 'petal length', 3: 'petal width'}

labels, it, centers, min_dists = kmeans.kmeans(attrs, 3, 'fast')

attr1 = 0
attr2 = 1
plt.scatter(attrs[:, attr1], attrs[:, attr2], c=labels)
plt.xlabel(attr_names[attr1])
plt.ylabel(attr_names[attr2])
plt.savefig('iris.png')
