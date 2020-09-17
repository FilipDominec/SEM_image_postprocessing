#!/usr/bin/python3  
#-*- coding: utf-8 -*-
"""
Multichannel overlay 

Most important features:
    * Automatically aligns images so that their edge features are matched. (Technically, it maximizes abs value of Laplacians.)
    * Optionally, affine transform can be used for better matching images that are rotated or slightly distorted. (This happens
      typically when e.g. electron acceleration voltage change between images or when sample gradually 
      accumulates charge.)
    * Exports the coloured composite image, along with each channel separately. 
    * Additionally, "extra" images can be aligned and exported in grayscale, but they do not affect alignment of further channels. 
      This can be used, e.g., to align SEM images to a series of spectrally resolved cathodoluminescence mappings.

Additional features
    * Images can be sharpened and contrast-enhanced
    * Since electron-microscope images can contain a databar, the bottom portion of the images is ignored during alignment

TODOs: 
    * join the export with annotate.py
    * don't forget to anisotropically scale at the load time
    * (too aggressive clipping, or clipping channels too early?) colors are still not as crisp as they used to be 
    * is there some colour mismatch between affine-tr and normal operation?
    * interactive GUI D&D application
    * test on windows
"""

## User settings

# Settings for correlation of images:
DISABLE_TRANSFORM = False   ## if set to true, the images will just be put atop of each other (no shift, no affine tr.)
USE_AFFINE_TRANSFORM = 0    ## enables scaling, tilting and rotating the images; otherwise they are just shifted
REL_MAX_SHIFT=.10          ## pixels cropped from the second image determine the maximum shift to be detected (higher number results in slower computation)
DECIM=2                     ## decimation of images prior to correlation (value of 2-5 speeds up processing, but does not affect the results much)

# Fine tuning
#DATABAR_PCT = (61./484)     ## relative height of databar at the images' bottom - these must be ignored when searching for correlation
DATABAR_PCT =  0.01            ##     (when no databar present, e.g. thanks to retouching)
CONSECUTIVE_ALIGNMENT = True ## if disabled, images are aligned always to the first one
FORCE_DOWNSCALE = 0         ## TODO
TRMATRIX_FACTOR = 0.5       ## tuning parameter, theoretically this should be 1.0; 

EXTRA_IMG_IDENT = 'S'   # each image containing this in its name is treated as extra  ## TODO identify extra by analyzing headers!
EXTRA_IMG_LABEL = '+'   # each image name preceded by this is treated as extra (and this symbol is removed prior to loading)
def is_extra(imname): return (imname[0] == EXTRA_IMG_LABEL or (EXTRA_IMG_IDENT in Path(imname).stem.upper())) ## TODO this should be better defined...

# Image post-processing settings:
SATURATION_ENHANCE = .15
UNSHARP_WEIGHT = 0.0 # #2
UNSHARP_RADIUS = 6.0


try:
    from IPython import get_ipython
    get_ipython().magic('reset -sf')
except:
    pass

## Import common moduli
import sys, os, time, collections, imageio
from pathlib import Path 
import numpy as np
from scipy.signal import convolve2d
from scipy.ndimage.filters import gaussian_filter
np.warnings.filterwarnings('ignore')

import pure_numpy_image_processing as pnip
import annotate_image




image_names = sys.argv[1:]

#colors = matplotlib.cm.gist_rainbow_r(np.linspace(0.25, 1, len([s for s in image_names if not is_extra(s)])))   ## Generate a nice rainbow scale for all non-extra images
#colors = [c*np.array([.8, .7, .9, 1]) for c in colors[::-1]] ## suppress green channel
#colors = pnip.rgb_palette(len([s for s in image_names if not is_extra(s)])
n_color_channels = len([s for s in image_names if not is_extra(s)])
colors = [pnip.hsv_to_rgb(h=h) for  h in np.linspace(1+1/6 if n_color_channels==2 else 1, 1+2/3, n_color_channels)] 
colors2 = colors[::-1]
WHITE = [1,1,1]
channel_outputs, extra_outputs = [], []
shiftvec_sum, shiftvec_new, trmatrix_sum, trmatrix_new = np.zeros(2), np.zeros(2), np.eye(2), np.eye(2)   ## Initialize affine transform to identity, and image shift to zero
for image_name in image_names:
    print('loading', image_name, 'detected as "extra image"' if is_extra(image_name) else ''); 
    newimg = pnip.safe_imload(str(Path(image_name).parent / Path(image_name).name.lstrip(EXTRA_IMG_LABEL)), retouch=True)

    #import scipy.ndimage
    #newimg = scipy.ndimage.median_filter(newimg, 2)

    color_tint = WHITE if is_extra(image_name) else colors.pop()
    max_shift = int(REL_MAX_SHIFT*newimg.shape[0])
    if 'image_padding' not in locals(): image_padding = max_shift*len(image_names) ## temporary very wide black padding for image alignment
    newimg_crop = gaussian_filter(newimg, sigma=DECIM*.5)[max_shift:-max_shift-int(newimg.shape[0]*DATABAR_PCT):DECIM, max_shift:-max_shift:DECIM]*1.0

    if 'refimg' in locals() and not DISABLE_TRANSFORM: ## the first image will be simply put to centre (nothing to align against)
        shiftvec_new, trmatrix_new = pnip.find_affine_and_shift(
                refimg_crop, newimg_crop, max_shift=max_shift, decim=DECIM, use_affine_transform=USE_AFFINE_TRANSFORM)
        shiftvec_sum += shiftvec_new 
        trmatrix_sum += (trmatrix_new - np.eye(2))*TRMATRIX_FACTOR
        print('... is shifted by {:}px against its reference image and by {:}px against the first one'.format(
            shiftvec_new*DECIM, shiftvec_sum))

    newimg_processed = pnip.my_affine_tr(        ## Process the new image: sharpening, affine transform, and padding...
            np.pad(pnip.unsharp_mask(newimg, weight=(0 if is_extra(image_name) else UNSHARP_WEIGHT), radius=UNSHARP_RADIUS), 
                    pad_width=max_shift, mode='constant'), trmatrix_sum) 
    
    if not is_extra(image_name):            ## ... then shifting and adding the image to the composite canvas
        if 'composite_output' not in locals(): 
            composite_output = np.zeros([newimg.shape[0]+2*image_padding, newimg.shape[1]+2*image_padding, 3])
        pnip.paste_overlay(composite_output, newimg_processed, shiftvec_sum, color_tint, normalize=np.mean(newimg_crop)*6)

    ## Export the image individually (either as colored channel, or as an extra image)
    single_output = np.zeros([newimg.shape[0]+2*image_padding, newimg.shape[1]+2*image_padding, 3])
    pnip.paste_overlay(single_output, newimg_processed, shiftvec_sum, color_tint, normalize=np.mean(newimg_crop)*6) 
    target_output = extra_outputs if is_extra(image_name) else channel_outputs
    target_output.append({'im':single_output, 'imname':image_name, 'header':annotate_image.analyze_header_XL30(image_name)})

    if not CONSECUTIVE_ALIGNMENT:   ## optionally, search alignment against the very first image
        shiftvec_sum, trmatrix_sum = np.zeros(2), np.eye(2)
    elif is_extra(image_name):      ## undo alignment changes (since extra imgs never affect alignment of further images)
        shiftvec_sum -= shiftvec_new
        trmatrix_sum -= (trmatrix_new - np.eye(2))*TRMATRIX_FACTOR

    if 'refimg' not in locals() or (CONSECUTIVE_ALIGNMENT and not is_extra(image_name)): ## store the image as a reference
        refimg, refimg_crop = newimg, newimg[:-int(newimg.shape[0]*DATABAR_PCT):DECIM, ::DECIM]*1.0

## Generate 5th line in the databar: color coding explanation
param_key, param_values = annotate_image.extract_dictkey_that_differs([co['header'] for co in channel_outputs], key_filter=['flAccV']) # 'Magnification', 'lDetName', 
if not param_values: 
    param_key, param_values = u'λ(nm)', extract_stringpart_that_differs([co['imname'] for co in channel_outputs])
assert param_values, 'aligned images, but could not extract a scanned parameter from their header nor names'


## Will crop all images identically - according to the extent of unused black margins in the composite image
crop_vert, crop_horiz = pnip.auto_crop_black_borders(composite_output, return_indices_only=True)


## Export individual channels, 
for n, ch_dict, color, param_value in zip(range(len(channel_outputs)), channel_outputs, colors2, param_values): 
    appendix_line = [[.6, 'Single channel for '], [WHITE, param_key+' = '], [color, param_value]]
    ch_dict['im'] = annotate_image.add_databar_XL30(ch_dict['im'][crop_vert,crop_horiz,:], ch_dict['imname'], ch_dict['header'], appendix_lines=[appendix_line]) # -> "Single channel for λ(nm) = 123"
    imageio.imsave(str(Path(ch_dict['imname']).parent / ('channel{:02d}_'.format(n) + Path(ch_dict['imname']).stem +'_ANNOT2.png')), ch_dict['im'])

for n, ch_dict in enumerate(extra_outputs): 
    im = annotate_image.add_databar_XL30(ch_dict['im'][crop_vert,crop_horiz,:], ch_dict['imname'], ch_dict['header'], appendix_lines=[[]])
    imageio.imsave(str(Path(ch_dict['imname']).parent / ('extra{:02d}_'.format(n) + Path(ch_dict['imname']).stem.lstrip('+')+ '.png')), ch_dict['im'])


## Generate 5th line in the databar for all-channel composite images
summary_ih = channel_outputs[0]['header']     # (take the header of the first file, assuming other have their headers identical)
dbar_appendix = [[[0.6, 'Composite channels by '], [WHITE, param_key+': ' ] ]]
for color, param_value in zip(colors2, param_values): dbar_appendix[0].append([color,' '+param_value]) ## append to 0th line of the appending

composite_output /= np.max(composite_output) # normalize all channels
imageio.imsave(str(Path(channel_outputs[0]['imname']).parent / ('composite_saturate_MedianFilter0px.png')), 
        annotate_image.add_databar_XL30(pnip.saturate(composite_output, saturation_enhance=SATURATION_ENHANCE)[crop_vert,crop_horiz,:], channel_outputs[0]['imname'], 
            summary_ih, appendix_lines=dbar_appendix, 
            ))
imageio.imsave(str(Path(channel_outputs[0]['imname']).parent / 'composite.png'), 
        annotate_image.add_databar_XL30(composite_output[crop_vert,crop_horiz,:], channel_outputs[0]['imname'],
            summary_ih, appendix_lines=dbar_appendix))
