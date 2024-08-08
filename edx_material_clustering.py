#!/usr/bin/python3
#-*- coding: utf-8 -*-

"""
Author: Filip Dominec; inspired by an example from Robert Layton, Olivier Grisel, Mathieu Blondel

Installation of dependencies

  pip3 install -U scikit-learn

License: BSD 3 clause


"""

#TODO: try to get along without 'sklearn' dep, using
        #https://docs.scipy.org/doc/scipy/reference/cluster.html#module-scipy.cluster
        #np.random
# TODO: check out totally different approach 
 #https://scikit-image.org/docs/stable/auto_examples/segmentation/plot_trainable_segmentation.html#sphx-glr-auto-examples-segmentation-plot-trainable-segmentation-py
# TODO : fix the scalebar, it should not depend on the black borders width !


# STATIC SETTINGS
n_colors = 6

SMOOTHING_PX = 2.       # higher value -> less jagged material regions, but 
                        # worse accuracy of EDX regions 

DENORM_EXP   =  .2      # partial de-normalization: Philips EDX saves images as 
                        # normalized. The more unique levels we count in each 
                        # image, the more EDX signal there was. Select 
                        # DENORM_EXP = 1 for full proportionality, but  
                        # DENORM_EXP = 0.5 seems to give better results.

BG_GAMMA_CURVE = 0.8    # use 1 for linear colour scaling for SEM underlying layer;
                        # use cca 0.5 to enhance color hue visibility in the shadows

FG_DESATURATE  = 1      # use 0 for full saturation of the resulting composite image; 
                        # use e.g. 3 for better visibility of the underlying SEM image

SEM2EDX_ZOOM_CORR = 1.04    ## , the areas scanned by SEM and consequent EDX mapping are not the same
MAX_SHIFT_LAB2SEM = 45      ## 

FORCE_SHIFT = None      # if not set, it is auto-determined by image matching
#FORCE_SHIFT = [0,0]    # positive values = SE underlayer moves towards top left corner
#FORCE_SHIFT = [30,40]  # positive values = SE underlayer moves towards top left corner

#note = "_2" # arbitrary output file note

#print(__doc__)
import numpy as np
from sklearn.cluster import KMeans
#from sklearn.metrics import pairwise_distances_argmin
#from sklearn.datasets import load_sample_image
from sklearn.utils import shuffle ## TODO get rid of

import imageio, sys, pathlib

import scipy.ndimage
from scipy.ndimage.filters import gaussian_filter

import pure_numpy_image_processing as pnip
import annotate_image


# TODO anisotropy

# User input
#n_colors = int(sys.argv[1])
#imnames  = sys.argv[2:]
imnames  = sys.argv[1:]



input_layers, element_names = [], []
for imname in imnames:
    if '.TIF' in imname:
        im_SEM_name = imname
    elif 'Lab' in imname:
        lab_name = imname
    else:
        #try: input_layer = imageio.imread(imname)
        #except: input_layer = load_Philips30XL_BMP(imname)
        input_layer = pnip.safe_imload(imname)
        element_names.append(imname.rsplit('_')[-1].rsplit('.')[0])
        assert len(input_layer.shape) == 2, 'did not expect RGB images from an EDX channel'

        ## partial de-normalization:
        input_layer = input_layer * float(len(np.unique(input_layer))-1)**DENORM_EXP / np.max(input_layer) 
        ## slightly smear
        input_layers.append(gaussian_filter(input_layer, sigma=SMOOTHING_PX)) 
input_layers = np.dstack(input_layers)
assert 'im_SEM_name' in locals(), 'SEM image missing: specify one *.TIF file in arguments'
assert 'lab_name' in locals(), '"LAB" (i.e. simultaneous SEM image taken along with EDX) missing: specify one *Lab*.* file in arguments'




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



## Actual clustering on larger sample
pixel_array_sample = shuffle(pixel_array, random_state=0)[:10000]
kmeans = KMeans(n_clusters=n_colors, random_state=0).fit(pixel_array_sample)
labels = kmeans.predict(pixel_array)




## == Output to a numpy array == 
palette = np.array([pnip.hsv_to_rgb(i,1,1) for i in np.linspace(0,1-1/n_colors,n_colors)])
#print(palette)
labels_remapped = pnip.rgb_palette(n_colors=n_colors)[labels]
im_reshaped = labels_remapped.reshape([w,h,3]) # / (np.max(labels)+1)
#imageio.imsave('edx_raw.png', im_reshaped)

## Reorder the cluster and label arrays so that similar materials have similar index (and thus, colour)
idx = np.arange(len(kmeans.cluster_centers_), dtype=int)
for n in range(30000):
    newidx = shuffle(idx, random_state=n)
    newmetric = np.sum((kmeans.cluster_centers_[newidx][:-1]**.5-kmeans.cluster_centers_[newidx][1:]**.5)**2)
    if 'bestmetric' not in locals() or newmetric < bestmetric:
        bestidx = newidx
        bestmetric = newmetric
    #print(bestmetric, newmetric)
kmeans.cluster_centers_ = kmeans.cluster_centers_[bestidx]
label_dict = dict(zip(bestidx, idx))
labels = [label_dict[x] for x in labels]

labels_remapped = palette[labels]
EDX_coloring = labels_remapped.reshape([w,h,3])
#imageio.imsave('edx_raw_remapped.png', im_reshaped)


im_SEM = pnip.safe_imload(im_SEM_name) 
im_SEM -= np.min(im_SEM) # TODO
im_LAB = pnip.safe_imload(lab_name)
im_LAB_resize2SEM = scipy.ndimage.zoom(im_LAB, [SEM2EDX_ZOOM_CORR * im_SEM.shape[i]/EDX_coloring.shape[i] for i in range(2)], order=1) #todo use pnip
#imageio.imsave('edx_im_LAB_resize2SEM.png', im_LAB_resize2SEM)

# Find the shift of high quality SEM image against "Lab1" image (i.e. SEM image taken during EDX map)
if FORCE_SHIFT:
    shift = np.array(FORCE_SHIFT)
else:
    shift, _ = pnip.find_affine_and_shift(
            im_LAB_resize2SEM[:,:], 
            im_SEM[MAX_SHIFT_LAB2SEM:-MAX_SHIFT_LAB2SEM,MAX_SHIFT_LAB2SEM:-MAX_SHIFT_LAB2SEM], 
            max_shift=0.05, 
            decim=1, 
            detect_edges=True,
            use_affine_transform=False)
im_SEM3 = 0 * np.dstack(np.pad(im_SEM, MAX_SHIFT_LAB2SEM, mode='constant') for ch in range(3))[:,:,:]
pnip.paste_overlay(im_SEM3, im_SEM, shift, np.array([1,1,1])) # , normalize=np.max(newimg_crop)

#imageio.imsave('edx_im_SEM3.png', im_SEM3)

EDX_zoomed = scipy.ndimage.zoom(FG_DESATURATE+EDX_coloring, [im_SEM.shape[i]/EDX_coloring.shape[i] for i in range(2)] + [1], order=1)
EDX_padded = np.dstack([np.pad(EDX_zoomed[:,:,ch], MAX_SHIFT_LAB2SEM, mode='constant') for ch in range(3)])

composite = im_SEM3**BG_GAMMA_CURVE*EDX_padded
composite = pnip.auto_crop_black_borders(composite, return_indices_only=False)
#imageio.imsave('edx_target2021.png', composite)

## TODO bar test
im_SEM_header = annotate_image.analyze_header_XL30(im_SEM_name)
charwidth = 9
colwidth = (composite.shape[1]-169)//(len(element_names)*charwidth)
appendix_line = [[pnip.white, f'{en[:-1]:{colwidth}}'] for en in element_names]
appendix_bars = []
for cc,co in zip(kmeans.cluster_centers_, palette):
    bar = []
    for ele in cc:
        bar.append({'style':'bar', 'xwidth':int(ele/np.max(kmeans.cluster_centers_)*(colwidth*charwidth-2)+.5), 'xpitch':colwidth*charwidth, 'color':co})
    appendix_bars.append(bar)
composite_annot = annotate_image.add_databar_XL30(composite, sys.argv[1], im_SEM_header, 
            appendix_lines= [appendix_line],
            appendix_bars = appendix_bars # TODO
            )
imageio.imsave(str(pathlib.Path(sys.argv[1]).parent / f'edx_composite_{n_colors}colors{note}.png'), composite_annot)


#Note this scipt replaces my original approach:
    #mkdir trash; mv *IPR *txt trash; echo "gimp: filter-batch-Batch process [add files]... output-format:PNG [start] [quit] alt-F4 " 
    #gimp; mv *bmp trash; mogrify -colorspace GRAY -normalize -negate -fx 'u*.95' *png;    
    #~/p/multichannel_image_overlay/annotate_image.py *TIF; mkdir orig; mv *TIF orig/;   
    #convert *NiK.png  *AgL.png *CuK.png -combine -set colorspace sRGB -negate  NiAgCu.png
    #convert *OK.png  *NK.png *SiK.png -combine -set colorspace sRGB -negate  ONSi.png
    #convert *AlK.png  *GaK.png *CK.png -combine -set colorspace sRGB -negate  AlGaC.png
