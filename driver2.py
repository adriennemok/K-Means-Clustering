import numpy as np
import h5py
import kmeans
import random
import os
import shutil

f = h5py.File('sift_all.h5')
sift_all = f['/sift_all'][:]

labels, it, centers, min_dists = kmeans.kmeans(sift_all, 1024, 'fast')

with open('patches.txt') as f:
	patches = f.read().split()

selected = set()

count = 0
while count < 10:
	label = random.randint(0, 1023)
	if label in selected:
		continue
	subset = np.where(labels == label)[0]
	if len(subset) == 0:
		continue
	selected.add(label)
	count += 1

for label in selected:
	try:
		os.makedirs(str(label))
	except OSError:
		pass
	subset = np.where(labels == label)[0]
	min_dist = min_dists[subset]
	idx = min_dist.argsort()
	subset = subset[idx]
	for j in range(min(len(subset), 100)):
		shutil.copyfile(patches[subset[j]], os.path.join(str(label), str(j) + '.bmp'))
