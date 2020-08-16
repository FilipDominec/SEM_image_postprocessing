#!/usr/bin/python3
#-*- coding: utf-8 -*-
# Author: Filip Dominec; inspired by an example from Robert Layton, Olivier Grisel, Mathieu Blondel
#
# License: BSD 3 clause

# todos optional: 
# * try im_med = ndimage.median_filter(im_noise, 3)

print(__doc__)
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.metrics import pairwise_distances_argmin
from sklearn.datasets import load_sample_image
from sklearn.utils import shuffle
from time import time

from numpy.random import random
import imageio, sys
from scipy.ndimage.filters import gaussian_filter


# Load the input images
#input_layers = load_sample_image("input_layers.jpg")
#imnames = ('~/p/200608_LED/M2/edxmap_new/map2_AlK.png', '~/p/200608_LED/M2/edxmap_new/map2_CuK.png', 
       #'~/p/200608_LED/M2/edxmap_new/map2_OK.png', '~/p/200608_LED/M2/edxmap_new/map2_GaK.png', '~/p/200608_LED/M2/edxmap_new/map2_OK.png')

def load_Siemens_BMP(fname):
    """ Experimental loading of BMPs from Siemens microscopes (atypical - that cannot be loaded by imageio)
    See https://ide.kaitai.io/ for more information on BMP header. 
    """
    with open(fname, mode='rb') as file: # first analyze the header
        fileContent = file.read()
        ofs, w, h, bpp, compr = [int.from_bytes(fileContent[s:e], byteorder='little', signed=False) for s,e in 
                ((0x0a,0x0e),(0x12,0x16),(0x16,0x1a),(0x1c,0x1e),(0x1e,0x22))]
    assert bpp == 8, f'monochrome/LUT image assumed (8 bit per pixel); {fname} has {bpp}bpp'
    assert compr == 0, 'no decompression algorithm implemented'
    return np.fromfile(fname, dtype=np.uint8)[ofs:ofs+w*h].reshape(h,w)[::-1,:] # BMP is "upside down" - flip vertically

n_colors = int(sys.argv[1])
imnames  = sys.argv[2:]
#for imname in imnames: print('LOADING', imname, imageio.imread(imname).shape)

print(imnames)
input_layers = []
for imname in imnames:
    try: 
        input_layer = imageio.imread(imname)
    except:
        input_layer = load_Siemens_BMP(imname)
    print(input_layer,input_layer.shape)
    #input_layer = input_layer * float(len(np.unique(input_layer))-1) / np.max(input_layer) ## normalize - the more unique levels found, the more EDX signal there was
    input_layer = input_layer * float(len(np.unique(input_layer))-1)**.5 / np.max(input_layer) ## normalize - the more unique levels found, the more EDX signal there was
    #input_layer = input_layer  / np.max(input_layer) ## normalize - the more unique levels found, the more EDX signal there was
    #input_layers.append(gaussian_filter(255-np.sum(imageio.imread(imname),axis=0), sigma=2)) ## slightly smear
    input_layers.append(gaussian_filter(input_layer, sigma=1.)) ## slightly smear
input_layers = np.dstack(input_layers)


# Convert to floats instead of the default 8 bits integer coding. Dividing by
# 255 is important so that plt.imshow works properly  XXX
#input_layers = np.array(input_layers, dtype=np.float64) / 255

# Load Image and transform to a 2D numpy array.
w, h, d = original_shape = tuple(input_layers.shape)
image_array = np.reshape(input_layers, (w * h, d))

print("Fitting model on a small sub-sample of the data")
t0 = time()
image_array_sample = shuffle(image_array, random_state=0)[:1000]
kmeans = KMeans(n_clusters=n_colors, random_state=0).fit(image_array_sample)
print("done in %0.3fs." % (time() - t0))

# Get labels for all points
print("Predicting color indices on the full image (k-means)")
t0 = time()
labels = kmeans.predict(image_array)
print("done in %0.3fs." % (time() - t0))


#codebook_random = shuffle(image_array, random_state=0)[:n_colors]
#print("Predicting color indices on the full image (random)")
#t0 = time()
#labels_random = pairwise_distances_argmin(codebook_random,
                                          #image_array,
                                          #axis=0)
#print("done in %0.3fs." % (time() - t0))


def recreate_image(codebook, labels, w, h):
    """Recreate the (compressed) image from the code book & labels"""
    d = codebook.shape[1]
    image = np.zeros((w, h, d))
    label_idx = 0
    for i in range(w):
        for j in range(h):
            image[i][j] = codebook[labels[label_idx]]
            label_idx += 1
    return image

# Display all results, alongside original image
#plt.figure(1)
#plt.clf()
#plt.axis('off')
#plt.title('Original image (96,615 colors)')
#plt.imshow(input_layers)

fig = plt.figure()
#fig, (ax1, ax) = plt.subplots(nrows=1, ncols=2)
ax1 = fig.add_subplot(121)

#plt.clf()
plt.axis('off')
ax1.set_title('Quantized image ({:d} colors, K-Means)'.format(n_colors))
#ax1.imshow(recreate_image(kmeans.cluster_centers_, labels, w, h))

print(kmeans.cluster_centers_, )
idx = np.arange(len(kmeans.cluster_centers_), dtype=int)
print(kmeans.cluster_centers_[idx])
for n in range(1000):
    newidx = shuffle(idx, random_state=n)
    newmetric = np.sum((kmeans.cluster_centers_[newidx][:-1]-kmeans.cluster_centers_[newidx][1:])**2)
    if 'bestmetric' not in locals() or newmetric < bestmetric:
        bestidx = newidx
        bestmetric = newmetric
kmeans.cluster_centers_ = kmeans.cluster_centers_[bestidx]
label_dict = dict(zip(bestidx, idx))
print('label_dict', label_dict,zip(idx, bestidx))
labels = [label_dict[x] for x in labels]

import matplotlib.cm
my_palette = matplotlib.cm.gist_rainbow(np.linspace(0, 1, n_colors+1)[:-1])
ax1.imshow(recreate_image(my_palette, labels, w, h))

for color, clustcent in zip(my_palette, kmeans.cluster_centers_):
    l = ''
    for cc,imn in zip(clustcent,imnames):
        if cc>.2:  l+='{:s}$_{{ {:.2f} }}$'.format(imn[:-5].split('_')[-1], cc)
    for cc,imn in zip(clustcent,imnames):
        if cc>.05 and cc<=.2:  l+='({:s}$_{{ {:.2f} }}$) '.format(imn[:-5].split('_')[-1], cc)
    #for cc in clustcent:
        #if cc>.2: l += '('+ elem_name[-6:-3] +') '
    ax1.plot([0,0],[0,0], label = l, lw=10, c=color)
plt.legend()

## Spider plot (a visual legend)
df = kmeans.cluster_centers_
categories = imnames
df = np.hstack((df, df[:,:1])) 

 
# What will be the angle of each axis in the plot? (we divide the plot / number of variable)
#angles = [n / float(len(imnames)) * 360 for n in range(len(imnames))]
#angles += angles[:1]
angles = np.hstack((np.linspace(0, 2*np.pi, len(imnames)), [0]))


ax = fig.add_subplot(122, polar=True)
fig.set_facecolor('grey')
 
# Draw one axe per variable + add labels labels yet
ax.set_xticks(angles[:-1], categories) #, text_size=12
 
# Draw ylabels
ax.set_rlabel_position(0)
ax.set_yticks([10,20,30], ["10","20","30"]) #, color="grey", size=12
#ax.set_ylim(0,40)
 
# Plot data
#print('angles, df',angles, df)
#print (imnames)
ax.set_thetagrids(angles*180/np.pi, [name.split('_')[-1].split('.')[0] for name in imnames])
for color, d, label in zip(my_palette, df, imnames):
    ax.plot(angles, d, linewidth=3, linestyle='solid', label='label', color=color) 
    ax.fill(angles, d,  alpha=0.2, color=color)
#fig.savefig('aa.png')

plt.show()
