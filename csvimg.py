import numpy as np
from scipy import misc
import csv
from matplotlib import pyplot as plt
import os

_dir = ".\database\ALL_MIAS"
for s,r,f in os.walk(_dir):
	for file in f:
		name, ext = os.path.splitext(file)
		if not ("pgm" in ext): continue
		image_path = os.path.join(s, file)
		print(image_path)
		img = misc.imread(image_path, mode='L', flatten=True)
		img = np.array(img)
		np.savetxt('%s\%s.txt'%(_dir, name), img, fmt='%f', delimiter=',')
		