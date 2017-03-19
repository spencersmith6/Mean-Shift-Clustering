from PIL import Image
import numpy as np
from sklearn.cluster import MeanShift, estimate_bandwidth, DBSCAN
from time import time
import matplotlib.pyplot as plt
import numpy as np
import scipy as sp
from skimage.morphology import closing


im = Image.open('baboon.jpg', 'r')
im = np.array(im)

plt.imshow(im)


length, height = len(im), len(im[0])
flat_image=np.reshape(im, [-1, 3])


bandwidth = estimate_bandwidth(flat_image, quantile=.2, n_samples=100)
print bandwidth
ms = MeanShift(bandwidth=50, bin_seeding=True, min_bin_freq = 100)

start =time()
ms.fit(flat_image)
print time()- start
labels = ms.labels_

cluster_centers = ms.cluster_centers_

labels_unique = np.unique(labels)
n_clusters_ = len(labels_unique)
print("number of estimated clusters : %d" % n_clusters_)

imgarray = np.reshape(labels, [length, height])

#######
layer = 2

region =  (layer==imgarray)
nregion = ~ region

shift = -1
edgex1 = (region ^ np.roll(nregion,shift=shift,axis=0))
edgey1 = (region ^ np.roll(nregion,shift=shift,axis=1))

plt.imshow(imgarray)


plt.contour(edgex1,2,colors='y',lw=2.)
plt.contour(edgey1,2,colors='y',lw=2.)
plt.imshow(imgarray)



#CLUMP_SIZE = 20
#
# count = []
# current= imgarray[0][0]
# for col in range(len(imgarray)):
#     for el in range(len(col)):
#         if el == current:
#             count +=1
#             current = el

# # Plot image vs segmented image
# plt.figure(2)
# plt.subplot(2, 1, 1)
# plt.imshow(im)
# plt.axis('off')
# plt.subplot(2, 1, 2)
# plt.imshow(np.reshape(labels, [851, 1280]))
# plt.axis('off')
