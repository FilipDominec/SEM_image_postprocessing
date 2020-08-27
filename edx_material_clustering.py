#!/usr/bin/python3
#-*- coding: utf-8 -*-
# Author: Filip Dominec; inspired by an example from Robert Layton, Olivier Grisel, Mathieu Blondel
# Installation on Linux (systemwide - run as root):
#   sudo pip3 install -U scikit-learn
#
# License: BSD 3 clause

# todos optional: 
# * try im_med = ndimage.median_filter(im_noise, 3)

print(__doc__)
import numpy as np
from sklearn.cluster import KMeans
#from sklearn.metrics import pairwise_distances_argmin
#from sklearn.datasets import load_sample_image
from sklearn.utils import shuffle ## TODO get rid of
from time import time

from numpy.random import random
import imageio, sys
from scipy.ndimage.filters import gaussian_filter

import pure_numpy_image_processing as pnip


# STATIC SETTINGS
SMOOTHING_PX = .7      # higher value -> less jagged material regions, but worse resolution

DENORM_EXP   =  .5      # partial de-normalization: EDX saves images as normalized. The more 
                        # unique levels we count in each image, 
                        # the more EDX signal there was. Select DENORM_EXP = 0 to disable this.
                        # Select DENORM_EXP = 1 for full proportionality, but this seems "too much".




# User input
#n_colors = int(sys.argv[1])
#imnames  = sys.argv[2:]
imnames  = sys.argv[1:]



input_layers = []
for imname in imnames:
    #try: input_layer = imageio.imread(imname)
    #except: input_layer = load_Siemens_BMP(imname)
    input_layer = pnip.safe_imload(imname)
    #xxx assert input_layer.shape[2] == 1, 'did not expect RGB images from an EDX channel'

    ## partial de-normalization:
    input_layer = input_layer * float(len(np.unique(input_layer))-1)**DENORM_EXP / np.max(input_layer) 
    ## slightly smear
    input_layers.append(gaussian_filter(input_layer, sigma=SMOOTHING_PX)) 
input_layers = np.dstack(input_layers)



# TODO try https://scikit-learn.org/stable/auto_examples/mixture/plot_gmm_covariances.html#sphx-glr-auto-examples-mixture-plot-gmm-covariances-py

# == Clustering (using KMeans algorithm here) ==
# Load Image and transform to a 2D numpy array.
w, h, d = input_layers.shape
pixel_array = np.reshape(input_layers, (w * h, d))

## Guess the optimum number of clusters (todo: should another algorithm be chosen?)
scores = []
for n_colors in range(3,len(imnames)+1): # subjectively proposed range of colour count to test
    t0 = time()
    pixel_array_sample = shuffle(pixel_array, random_state=0)[:100]
    kmeans = KMeans(n_clusters=n_colors, random_state=0).fit(pixel_array_sample)
    #print('n_colors, kmeans.inertia, c*i',n_colors, kmeans.inertia_, n_colors, kmeans.inertia_*n_colors**1.2)
    scores.append((kmeans.inertia_*n_colors**(1+1./len(imnames)), n_colors))
    #print("done in %0.3fs." % (time() - t0))

    # Get labels for all points
    t0 = time()
    labels = kmeans.predict(pixel_array)
    #print("done in %0.3fs." % (time() - t0))
scores.sort()
#_, n_colors = scores[0]
n_colors = 8 #XXX
#print(scores,n_colors)


## Actual clustering on larger sample
pixel_array_sample = shuffle(pixel_array, random_state=0)[:1000]
kmeans = KMeans(n_clusters=n_colors, random_state=0).fit(pixel_array_sample)
labels = kmeans.predict(pixel_array)




## == Output to a numpy array == 
palette = np.array([pnip.hsv_to_rgb(i,1,1) for i in np.linspace(0,1-1/n_colors,n_colors)])
print(palette)
labels_remapped = pnip.rgb_palette(n=n_colors)[labels]
im_reshaped = labels_remapped.reshape([w,h,3]) # / (np.max(labels)+1)
imageio.imsave('edx_raw.png', im_reshaped)

## Reorder the cluster and label arrays so that similar materials have similar index (and thus, colour)
idx = np.arange(len(kmeans.cluster_centers_), dtype=int)
for n in range(30000):
    newidx = shuffle(idx, random_state=n)
    newmetric = np.sum((kmeans.cluster_centers_[newidx][:-1]-kmeans.cluster_centers_[newidx][1:])**2)
    if 'bestmetric' not in locals() or newmetric < bestmetric:
        bestidx = newidx
        bestmetric = newmetric
    #print(bestmetric, newmetric)
kmeans.cluster_centers_ = kmeans.cluster_centers_[bestidx]
label_dict = dict(zip(bestidx, idx))
labels = [label_dict[x] for x in labels]

labels_remapped = palette[labels]
im_reshaped = labels_remapped.reshape([w,h,3]) # / (np.max(labels)+1)

bgim = pnip.safe_imload('~/SEM/LED_reports/LED_reports_2020-06-00/M2/emap200x/I30S.TIF') ## FIXME
imageio.imsave('edx_wb.png', bgim)

import scipy.ndimage
im_resc = np.dstack([scipy.ndimage.zoom(im_reshaped[:,:,ch], [bgim.shape[i]/im_reshaped.shape[i] for i in range(2)], order=1) for ch in range(3)])
imageio.imsave('edx_raw_remap_resc.png', im_resc)

im_resc = np.dstack([bgim**.5*scipy.ndimage.zoom(3+im_reshaped[:,:,ch], [bgim.shape[i]/im_reshaped.shape[i] for i in range(2)], order=1) for ch in range(3)])
imageio.imsave('edx_target.png', im_resc)

#im_resc = np.dstack([np.pad(bgim,[(0,5),(0,0),(0,0)])*np.pad(im_rescbgim,[(0,5),(0,0),(0,0)]))
#imageio.imsave('edx_target.png', im_resc)

quit() # XXX













## == Output using matplotlib (TODO: remove this dep) ===
# Diagnostics: Display all results, alongside original image
fig = plt.figure()
ax1 = fig.add_subplot(121)

#plt.clf()
plt.axis('off')
ax1.set_title('Quantized image ({:d} colors, K-Means)'.format(n_colors))
#ax1.imshow(recreate_image(kmeans.cluster_centers_, labels, w, h))

## Reorder the cluster and label arrays so that similar materials have similar index (and thus, colour)
idx = np.arange(len(kmeans.cluster_centers_), dtype=int)
for n in range(1000):
    newidx = shuffle(idx, random_state=n)
    newmetric = np.sum((kmeans.cluster_centers_[newidx][:-1]-kmeans.cluster_centers_[newidx][1:])**2)
    if 'bestmetric' not in locals() or newmetric < bestmetric:
        bestidx = newidx
        bestmetric = newmetric
    #print(bestmetric, newmetric)
kmeans.cluster_centers_ = kmeans.cluster_centers_[bestidx]
label_dict = dict(zip(bestidx, idx))
labels = [label_dict[x] for x in labels]


## Generate color coding
import matplotlib.pyplot as plt ## TODO get rid of
import matplotlib.cm ## TODO get rid of
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
df = np.hstack((df, df[:,:1])) 
angles = np.hstack((np.linspace(0, 2*np.pi, len(imnames)), [0]))
ax = fig.add_subplot(122, polar=True)
fig.set_facecolor('grey')
ax.set_xticks(angles[:-1], imnames) #, text_size=12
ax.set_rlabel_position(0)
ax.set_yticks([10,20,30], ["10","20","30"]) #, color="grey", size=12
ax.set_thetagrids(angles*180/np.pi, [name.split('_')[-1].split('.')[0] for name in imnames])
for color, d, label in zip(my_palette, df, imnames):
    ax.plot(angles, d, linewidth=3, linestyle='solid', label='label', color=color) 
    ax.fill(angles, d,  alpha=0.2, color=color)
plt.show()
