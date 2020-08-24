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
import matplotlib.cm
from sklearn.cluster import KMeans
from sklearn.metrics import pairwise_distances_argmin
from sklearn.datasets import load_sample_image
from sklearn.utils import shuffle
from time import time

from numpy.random import random
import imageio, sys
from scipy.ndimage.filters import gaussian_filter


# User input and settings
n_colors = int(sys.argv[1])
imnames  = sys.argv[2:]


# Load the input images
def load_Siemens_BMP(fname):
    """ 
    Experimental loading of BMPs from Siemens microscopes (atypical - that cannot be loaded by imageio)
    See https://ide.kaitai.io/ for more information on BMP header. 
    """
    with open(fname, mode='rb') as file: # first analyze the header
        fileContent = file.read()
        ofs, w, h, bpp, compr = [int.from_bytes(fileContent[s:e], byteorder='little', signed=False) for s,e in 
                ((0x0a,0x0e),(0x12,0x16),(0x16,0x1a),(0x1c,0x1e),(0x1e,0x22))]
    assert bpp == 8, f'monochrome/LUT image assumed (8 bit per pixel); {fname} has {bpp}bpp'
    assert compr == 0, 'no decompression algorithm implemented'
    return np.fromfile(fname, dtype=np.uint8)[ofs:ofs+w*h].reshape(h,w)[::-1,:] # BMP is "upside down" - flip vertically

input_layers = []
for imname in imnames:
    try: input_layer = imageio.imread(imname)
    except: input_layer = load_Siemens_BMP(imname)

    ## partial de-normalization: the more unique levels found in each image, the more EDX signal there was 
    input_layer = input_layer * float(len(np.unique(input_layer))-1)**.5 / np.max(input_layer) 
    ## slightly smear
    input_layers.append(gaussian_filter(input_layer, sigma=1.5)) 
input_layers = np.dstack(input_layers)


# Convert to floats instead of the default 8 bits integer coding. Dividing by
# 255 is important so that plt.imshow works properly  XXX
#input_layers = np.array(input_layers, dtype=np.float64) / 255

# Load Image and transform to a 2D numpy array.
w, h, d = input_layers.shape
image_array = np.reshape(input_layers, (w * h, d))

## Guess the optimum number of clusters (todo: should another algorithm be chosen?)
scores = []
for n_colors in range(3,len(imnames)+1): # subjectively proposed range of colour count to test
    t0 = time()
    image_array_sample = shuffle(image_array, random_state=0)[:100]
    kmeans = KMeans(n_clusters=n_colors, random_state=0).fit(image_array_sample)
    #print('n_colors, kmeans.inertia, c*i',n_colors, kmeans.inertia_, n_colors, kmeans.inertia_*n_colors**1.2)
    scores.append((kmeans.inertia_*n_colors**(1+1./len(imnames)), n_colors))
    #print("done in %0.3fs." % (time() - t0))

    # Get labels for all points
    t0 = time()
    labels = kmeans.predict(image_array)
    #print("done in %0.3fs." % (time() - t0))
scores.sort()
_, n_colors = scores[0]
n_colors = 12
print(scores,n_colors)


## Actual clustering on larger sample
image_array_sample = shuffle(image_array, random_state=0)[:1000]
kmeans = KMeans(n_clusters=n_colors, random_state=0).fit(image_array_sample)
t0 = time()
labels = kmeans.predict(image_array)

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

## Reorder the cluster and label arrays so that 
idx = np.arange(len(kmeans.cluster_centers_), dtype=int)
for n in range(10000):
    newidx = shuffle(idx, random_state=n)
    newmetric = np.sum((kmeans.cluster_centers_[newidx][:-1]-kmeans.cluster_centers_[newidx][1:])**2)
    if 'bestmetric' not in locals() or newmetric < bestmetric:
        bestidx = newidx
        bestmetric = newmetric
kmeans.cluster_centers_ = kmeans.cluster_centers_[bestidx]
label_dict = dict(zip(bestidx, idx))
labels = [label_dict[x] for x in labels]


## Generate color coding
my_palette = matplotlib.cm.gist_rainbow(np.linspace(0, 1, n_colors+1)[:-1])

def recreate_image(codebook, labels, w, h): ## todo: obsolete?
    """Recreate the (compressed) image from the code book & labels"""
    d = codebook.shape[1]
    image = np.zeros((w, h, d))
    label_idx = 0
    for i in range(w):
        for j in range(h):
            image[i][j] = codebook[labels[label_idx]]
            label_idx += 1
    return image

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
