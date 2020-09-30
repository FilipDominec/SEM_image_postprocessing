#!/usr/bin/python3
#-*- coding: utf-8 -*-

"""
Author: Filip Dominec; inspired by an example from Robert Layton, Olivier Grisel, Mathieu Blondel

Installation of dependencies

  pip3 install -U scikit-learn

License: BSD 3 clause

"""

BG_GAMMA_CURVE = 0.5        ## use 1 for linear colour scaling; use cca 0.5 to enhance color hue visibility in the shadows
FG_DESATURATE  = 1          ## use 0 for full saturation; use e.g. 3 for better visibility of the underlying SEM image

SEM2EDX_ZOOM_CORR = 1.04    ## , the areas scanned by SEM and consequent EDX mapping are not the same
MAX_SHIFT_LAB2SEM = MAX_SHIFT_LAB2SEM      ## 

print(__doc__)
import numpy as np
from sklearn.cluster import KMeans
#from sklearn.metrics import pairwise_distances_argmin
#from sklearn.datasets import load_sample_image
from sklearn.utils import shuffle ## TODO get rid of

import imageio, sys, pathlib
from scipy.ndimage.filters import gaussian_filter

import pure_numpy_image_processing as pnip
import annotate_image


# STATIC SETTINGS
SMOOTHING_PX = .7       # higher value -> less jagged material regions, but 
                        # worse resolution 

DENORM_EXP   =  .5      # partial de-normalization: EDX saves images as 
                        # normalized. The more unique levels we count in each 
                        # image, the more EDX signal there was. Select 
                        # DENORM_EXP = 1 for full proportionality, but  
                        # DENORM_EXP = 0.5 seems to give better results.



# User input
#n_colors = int(sys.argv[1])
#imnames  = sys.argv[2:]
imnames  = sys.argv[1:]



input_layers = []
for imname in imnames:
    if '.TIF' in imname:
        im_SEM_name = imname
    elif 'Lab' in imname:
        lab_name = imname
    else:
        #try: input_layer = imageio.imread(imname)
        #except: input_layer = load_Siemens_BMP(imname)
        input_layer = pnip.safe_imload(imname)
        assert len(input_layer.shape) == 2, 'did not expect RGB images from an EDX channel'

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
#scores = []
#for n_colors in range(3,len(imnames)+1): # subjectively proposed range of colour count to test
    #pixel_array_sample = shuffle(pixel_array, random_state=0)[:100]
    #kmeans = KMeans(n_clusters=n_colors, random_state=0).fit(pixel_array_sample)
    #scores.append((kmeans.inertia_*n_colors**(1+1./len(imnames)), n_colors))
    #labels = kmeans.predict(pixel_array)
#scores.sort()
#_, n_colors = scores[0]
#print(scores,n_colors)

n_colors = 8 #XXX


## Actual clustering on larger sample
pixel_array_sample = shuffle(pixel_array, random_state=0)[:1000]
kmeans = KMeans(n_clusters=n_colors, random_state=0).fit(pixel_array_sample)
labels = kmeans.predict(pixel_array)




## == Output to a numpy array == 
palette = np.array([pnip.hsv_to_rgb(i,1,1) for i in np.linspace(0,1-1/n_colors,n_colors)])
print(palette)
labels_remapped = pnip.rgb_palette(n_colors=n_colors)[labels]
im_reshaped = labels_remapped.reshape([w,h,3]) # / (np.max(labels)+1)
#imageio.imsave('edx_raw.png', im_reshaped)

## Reorder the cluster and label arrays so that similar materials have similar index (and thus, colour)
idx = np.arange(len(kmeans.cluster_centers_), dtype=int)
for n in range(3000):
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
EDX_coloring = labels_remapped.reshape([w,h,3]) # / (np.max(labels)+1)


if 'im_SEM_name' in locals()  and  'lab_name' in locals():
    import scipy.ndimage

    im_SEM = pnip.safe_imload(im_SEM_name) 
    im_LAB = pnip.safe_imload(lab_name)
    im_LAB_resize2SEM = scipy.ndimage.zoom(im_LAB, [SEM2EDX_ZOOM_CORR * im_SEM.shape[i]/EDX_coloring.shape[i] for i in range(2)], order=1) #todo use pnip
    imageio.imsave('edx_im_LAB_resize2SEM.png', im_LAB_resize2SEM)

    # Find the shift of high quality SEM image against "Lab1" image (i.e. SEM image taken during EDX map)
    shift, _ = pnip.find_affine_and_shift(
            im_LAB_resize2SEM[:,:], 
            im_SEM[MAX_SHIFT_LAB2SEM:-MAX_SHIFT_LAB2SEM,MAX_SHIFT_LAB2SEM:-MAX_SHIFT_LAB2SEM], 
            max_shift=0.15, 
            decim=1, 
            use_affine_transform=False)
    im_SEM3 = 0 * np.dstack(np.pad(im_SEM, MAX_SHIFT_LAB2SEM, mode='constant') for ch in range(3))[:,:,:]
    pnip.paste_overlay(im_SEM3, im_SEM, shift, np.array([1,1,1])) # , normalize=np.max(newimg_crop)

    imageio.imsave('edx_im_SEM3.png', im_SEM3)

    #im_resc = np.dstack([scipy.ndimage.zoom(EDX_coloring[:,:,ch], [im_SEM.shape[i]/EDX_coloring.shape[i] for i in range(2)], order=1) for ch in range(3)]) #todo use pnip
    #imageio.imsave('edx_raw_remap_resc.png', im_resc)


    EDX_zoomed = np.dstack([scipy.ndimage.zoom(FG_DESATURATE+channel, [im_SEM.shape[i]/channel.shape[i] for i in range(2)], order=1) for channel in EDX_coloring])
    print("DEBUG: EDX_zoomed = ", EDX_zoomed.shape)
    EDX_padded = np.dstack([np.pad(channel, MAX_SHIFT_LAB2SEM, mode='constant') for channel in EDX_zoomed])
    print("DEBUG: EDX_padded = ", EDX_padded.shape)

    composite = im_SEM3**BG_GAMMA_CURVE*EDX_padded
    imageio.imsave('edx_target.png', composite)

    ## TODO bar test
    im_SEM_header = annotate_image.analyze_header_XL30(im_SEM_name)
    composite_annot = annotate_image.add_databar_XL30(composite, sys.argv[1], im_SEM_header, 
                appendix_lines= [[]],
                appendix_bars = [[{'style':'bar','xwidth':MAX_SHIFT_LAB2SEM, 'xpitch':60, 'color':0.6}, {'style':'bar','xwidth':30, 'xpitch':60, 'color':[.2,.5,.9]}]] # TODO
                )
    imageio.imsave(str(pathlib.Path(sys.argv[1]).parent / 'target_annot.png'), composite_annot)


#Note this scipt replaces my original approach:
    #mkdir trash; mv *IPR *txt trash; echo "gimp: filter-batch-Batch process [add files]... output-format:PNG [start] [quit] alt-F4 " 
    #gimp; mv *bmp trash; mogrify -colorspace GRAY -normalize -negate -fx 'u*.95' *png;    
    #~/p/multichannel_image_overlay/annotate_image.py *TIF; mkdir orig; mv *TIF orig/;   
    #convert *NiK.png  *AgL.png *CuK.png -combine -set colorspace sRGB -negate  NiAgCu.png
    #convert *OK.png  *NK.png *SiK.png -combine -set colorspace sRGB -negate  ONSi.png
    #convert *AlK.png  *GaK.png *CK.png -combine -set colorspace sRGB -negate  AlGaC.png
