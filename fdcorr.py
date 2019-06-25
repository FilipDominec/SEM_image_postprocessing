#!/usr/bin/python3  
#-*- coding: utf-8 -*-
"""

todo: test on '/home/dominecf/LIMBA/SEM-dle_data_obsolentni/2019/06cerven/190613_319B_254A-FH/319B-FH-190613'
and '/home/dominecf/LIMBA/SEM-dle_cisla_vzorku/2019/283C_FH-190317_190611/283C_190611'

I use scipy.ndimage for image processing.
"""

## User settings

max_shift=60        ## pixels cropped from the second image determine the maximum shift to be detected (higher number results in slower computation)
decim=2        ## decimation of images prior to correlation (does not affect the results much)
databarh = 60   ## height of databar at the images' bottom - will be discarded first
SE_kernel_smoothing = False #  50/decim # optional smoothing of the SE kernel


## Image post-processing
saturation_enhance = .2
unsharp_weight = 2
unsharp_radius = 10

## Import common moduli
import matplotlib, sys, os, time, collections, imageio
import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import correlate2d, convolve2d
from scipy.ndimage.filters import laplace
np.warnings.filterwarnings('ignore')

def find_shift(im1, im2, smoothing=0, plot_correlation_name=None):
    """
    shifts im2 against im1 so that a best correlation is found, returns a tuple of the pixel shift
    """
    ## compute the image correlation, and its smoothed Laplacian
    corr = correlate2d(im1, im2, mode='valid')
    cr=1  # post-laplace cropping, there were some edge artifacts
    if smoothing:
        smoothing_kernel = np.outer(2**-(np.linspace(-2,2,smoothing)**2), 2**-(np.linspace(-2,2,smoothing)**2))
        smoothing_kernel /= np.sum(smoothing_kernel)
    else:
        smoothing_kernel = [[1]]
    laplcorr = convolve2d(-np.abs(laplace(corr)), smoothing_kernel, mode='valid')[cr:-2-cr,cr:-cr] # simple smoothing and removing spurious last 2lines
    vsize, hsize  = laplcorr.shape

    ## find optimum translation for the best image match
    raw_shifts = (np.unravel_index(np.argmin(laplcorr), laplcorr.shape)) # x,y coords of the minimum in the correlation map
    vshift, hshift = int((vsize/2 - raw_shifts[0] + 0.5)*decim), int((hsize/2 - raw_shifts[1] - 0.5)*decim) # linear transform against image centre
    print('second image is vertically and horizontally shifted by ({:},{:}) px against the previous one'.format(vshift,hshift))

    if plot_correlation_name:
        fig, ax = matplotlib.pyplot.subplots(nrows=1, ncols=1, figsize=(15,15))
        im = ax.imshow(laplcorr) # fig.colorbar(im, ax=ax, shrink=0.5)
        def plot_cross(h,v,c):
            ax.plot([h/2-5-.5,h/2+5-.5],[v/2+.5,v/2+.5],c=c,lw=.5)
            ax.plot([h/2-.5,h/2-.5],[v/2-5+.5,v/2+5+.5],c=c,lw=.5)
        plot_cross(hsize, vsize, 'k')
        plot_cross(hsize-hshift, vsize-vshift, 'w')
        fig.savefig(plot_correlation_name, bbox_inches='tight')

    return [vshift, hshift]

## Generate a nice rainbow scale
color_pre_map = np.linspace(0.30, 1, len(sys.argv)-1)
colors = matplotlib.cm.gist_rainbow_r(color_pre_map*.5 + np.sin(color_pre_map*np.pi/2)**2*.5) 

def paste_overlay(bgimage, fgimage, vs, hs, color, normalize=np.inf):
    for channel in range(3):
        bgimage[image_padding-vs:image_padding+fgimage.shape[0]-vs, 
                image_padding-hs:image_padding+fgimage.shape[1]-hs, channel] += np.clip(fgimage*float(color[channel])/normalize, 0, 1)

image_padding = max_shift*len(sys.argv[1:]) ## safe very wide black padding, could be optimized!

#def extend_canvas():

def safe_imload(imname):
    im = imageio.imread(imname.lstrip('+')) # plus sign has a special meaning of an 'extra' image
    if len(im.shape) > 2: im = im[:,:,0] # using monochrome images only; strip other channels than the first
    return im, (imname[0] == '+' or ('SC' in imname))

channel_outputs = []
extra_outputs = []
for image_name, color in zip(sys.argv[1:], colors):
    ## Load an image
    im, is_extra = safe_imload(image_name)
    im2crop = im[max_shift:-max_shift-databarh:decim, max_shift:-max_shift:decim]*1.0

    if 'prev_image' in locals():      ## Find shift from previous image
        im1crop = prev_image[:-databarh:decim, ::decim]*1.0
        ## Find the best correlation of both (cropped) images
        vshift, hshift = find_shift(im1crop, im2crop, plot_correlation_name=image_name.rsplit('.')[0] + '_correlation.png')

        #increment the shifts
        cumvshift += vshift; cumhshift += hshift 
    else:
        vshift, hshift, cumvshift, cumhshift = 0, 0, 0, 0     ## Initialize position to centre

    if is_extra:  # TODO
        extra_output = np.zeros([im.shape[0]+2*image_padding, im.shape[1]+2*image_padding, 3])
        paste_overlay(extra_output, im, cumvshift, cumhshift, [1,1,1,1], normalize=np.max(im2crop))
        extra_outputs.append(extra_output)
        cumvshift -= vshift; cumhshift -= hshift  # revert the temporary shift for extra image
    else:
        ## Prepare the composite canvas with the first centered image
        if 'composite_output' not in locals():
            # TODO extend(composite_output, im.shape[0]+cumvshift, im.shape[1]+cumhshift)
            composite_output = np.zeros([im.shape[0]+2*image_padding, im.shape[1]+2*image_padding, 3])
        paste_overlay(composite_output, im, cumvshift, cumhshift, color, normalize=np.max(im2crop))

        channel_output = np.zeros([im.shape[0]+2*image_padding, im.shape[1]+2*image_padding, 3])
        paste_overlay(channel_output, im, cumvshift, cumhshift, color, normalize=np.max(im2crop))
        channel_outputs.append(channel_output)

        prev_image = im

    # ----

#    if im1 == None:
#        im1 = im2
# 
#    if is_extra: 
#        # export white image (from SEM signal) displaced by cumvshift,cumhshift
#        color = [255,255,255,255]
#        extra_output = [im1.shape[0]+2*image_padding, im1.shape[1]+2*image_padding, 3])*16
#        paste_overlay(extra_output, im1, 0, 0, color, normalize=np.max(im1crop))
#        extra_outputs.append(extra_output)
#        im1 = im2
#    else
#        #im1 = np.clip(im1, 0, np.max(im1crop)) should prevent color bleaching in the databar?
#        #im2 = np.clip(im2, 0, np.max(im1crop))
# 
#        ## Find the best correlation of both (cropped) images
#        vshift, hshift = find_shift(im1crop, im2crop, plot_correlation_name=imname2.rsplit('.')[0] + 'alignedto' + imname1.rsplit('.')[0]+'_correlation.png')
# 
#        if not is_extra: 
#            cumvshift += vshift; cumhshift += hshift
#        paste_overlay(composite_output, im2, cumvshift, cumhshift, color, normalize=np.max(im2crop))
#    else:
#        pass
# 
#    prev_image = image

imageio.imsave('composite_' + '.png', composite_output)
for n,i in enumerate(channel_outputs): imageio.imsave('channel_' + str(n) + '.png', i)
for n,i in enumerate(extra_outputs): imageio.imsave('extra_' + str(n) + '.png', i)

    #np.dstack([im2*256/np.max(im2),im1*256/np.max(im1),im2*256/np.max(im2)])


unsharp_kernel = np.outer(2**-(np.linspace(-2,2,unsharp_radius)**2), 2**-(np.linspace(-2,2,unsharp_radius)**2))
unsharp_kernel /= np.sum(unsharp_kernel)
unsharp = np.dstack([
    convolve2d(composite_output[:,:,0], unsharp_kernel, mode='same'),
    convolve2d(composite_output[:,:,1], unsharp_kernel, mode='same'),
    convolve2d(composite_output[:,:,2], unsharp_kernel, mode='same')
    ])
composite_output = np.clip(composite_output*(1+unsharp_weight) - unsharp*unsharp_weight, 0, np.max(composite_output))
#clipped_sample = composite_output[composite_output.shape[0]//2-100:composite_output.shape[0]//2+100, composite_output.shape[1]//2-100:composite_output.shape[1]//2+100, :]
#composite_output = (composite_output - np.min(clipped_sample)) * np.max(composite_output) / np.max(clipped_sample)  
#imageio.imsave('clipped_sample' + '.png', clipped_sample)
imageio.imsave('composite_sharp' + '.png', composite_output)

monochr = np.dstack([np.sum(composite_output, axis=2)]*3)
composite_output = np.clip(composite_output*(1.5+saturation_enhance) - monochr*saturation_enhance, 0, np.max(composite_output))
imageio.imsave('composite_satu_sharp' + '.png', composite_output)

#    ## Composing output image
#    from PIL import Image
#    a = np.array(Image.open(sys.argv[1].rstrip('_')))
#    b = np.array(Image.open(sys.argv[2].rstrip('_')))
#    #b = convolve2d(b, SE_kernel, mode='valid')
#    #b = convolve2d(b, np.outer(2**-(np.linspace(-2,2,20)**2), 2**-(np.linspace(-2,2,20)**2)), mode='valid')
#    b -= np.min(b[:-databarh,:])
# 
#    ixs, iys = a.shape   # images' sizeSE_kernel, 
#    cxs, cys = ixs+max_shift*2, iys+max_shift*2 # canvas size 
#    aa = Image.new('L', (cxs,cys), (0))
#    aa.paste(Image.fromarray(np.round(np.array(a)*(255/np.max(a)))), (int(cxs/2-ixs/2), int(cys/2-iys/2)))
#    bb = Image.new('L', (cxs,cys), (0))
#    bb.paste(Image.fromarray(np.round(np.array(b)*(255/np.max(b[:-databarh,:])))), (int(cxs/2-ixs/2-hshift), int(cys/2-iys/2-vshift)))
# 
#    #background = Image.new('RGBA', (1440, 900), (255, 255, 255, 255))
#    c = Image.merge('RGB', (bb, bb, bb))
#    #a[:,:,0] *=0
#    #a[:,:,1] *=0
#    #Image.fromarray(a)
#    c.save('aligned'+sys.argv[2].rstrip('_'))
# 
#    #background = Image.new('RGBA', (1440, 900), (255, 255, 255, 255))
#    #bg_w, bg_h = background.size
#    #background.save('out.png')
# 
#    #bottom.save("over.png")
# 
#    #r, g, b, a = top.split()
#    #top = Image.merge("RGB", (top.split()[0], bottom.split()[0], bottom.split()[0]*0))
#    #mask = Image.merge("L", (bottom.split()[0],))
#    #bottom.paste(top, (0, 0), mask)
#    #bottom.save("over.png")



#  ## optional approximate conversion from "height map" to "SEM signal" ; possible todo: separate into identity+emboss+laplacian
#  SE_kernel =     [[ 0, -1, -2, -1,  0],
#                   [-1, -3,  1, -3, -1],
#                   [-2,  1, 30,  1, -2],
#                   [-1, -3,  1, -4, -1],
#                   [ 0, -2, -5, -2,  0]]
#  if SE_kernel_smoothing: 
#      SE_kernel = convolve2d(np.outer(2**-(np.linspace(-2,2,SE_kernel_smoothing)**2), 2**-(np.linspace(-2,2,SE_kernel_smoothing)**2)), SE_kernel,  mode='valid')
#  if sys.argv[1][-1] == '_': im1 = convolve2d(im1, SE_kernel, mode='valid')
#  if sys.argv[2][-1] == '_': im2 = convolve2d(im2, SE_kernel, mode='valid')
