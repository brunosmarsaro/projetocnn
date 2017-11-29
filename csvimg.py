import numpy as np
from scipy import misc
import csv
from matplotlib import pyplot as plt

image_path = "./MIAS/malignant/mdb141.pgm"
img = misc.imread(image_path, mode='L', flatten=True)
#plt.imshow(img)
#plt.show()
img = np.array(img)

np.savetxt('img_array.out', img, fmt='%f', delimiter=',')
'''
for line in img:
	print(line)
'''
