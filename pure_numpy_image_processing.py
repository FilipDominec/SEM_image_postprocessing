#!/usr/bin/python3  
#-*- coding: utf-8 -*-
"""
Pure Numpy Image Editing:

Sometimes it is easier to re-implement procedures with basic tools than to ensure advanced dependencies (like CV2) are met.

An image is represented by a simple numpy array, always having 3 dimensions. These are: 
            width =  image width
            height = image height
            depth =  1 for monochrome image, 3 for R-G-B colour images

Note this module relies also on Scipy functions.

FIXME: sometimes monochrome images are returned as just 2-D arrays
TODO: check if there is no reasonable alternative, then put all functions from this module into a class
TODO: make this compatible with https://scipy-lectures.org/advanced/image_processing/ and propose partial merge

"""

import imageio, pathlib
import numpy as np

from scipy.ndimage import laplace, gaussian_filter
from scipy.ndimage import affine_transform, zoom
from scipy.signal import correlate2d, convolve2d
from scipy.optimize import differential_evolution


# Load the input images

def load_Philips30XL_BMP(fname):
    """ 

    Loading of BMPs from Philips30XL microscopes EDX mapping software (an atypical format which cannot be loaded by imageio)
    See https://ide.kaitai.io/ for more information on BMP header. 
    """
    with open(fname, mode='rb') as file: # first analyze the header
        fileContent = file.read()
        ofs, w, h, bpp, compr = [int.from_bytes(fileContent[s:e], byteorder='little', signed=False) for s,e in 
                ((0x0a,0x0e),(0x12,0x16),(0x16,0x1a),(0x1c,0x1e),(0x1e,0x22))]
    assert bpp == 8, f'monochrome/LUT image assumed (8 bit per pixel); {fname} has {bpp}bpp'
    assert compr == 0, 'no decompression algorithm implemented'
    return np.fromfile(fname, dtype=np.uint8)[ofs:ofs+w*h].reshape(h,w)[::-1,:] # BMP is "upside down" - flip vertically

def safe_imload(imname, retouch_databar=False, return_whitemask=False):
    """
    Loads an image as 1-channel (that is, either grayscale, or a fixed palette such as those from Philips30XL EDX)

    Returns:
        single-channel 2D numpy array (width x height) with values between 0.0 and 1.0

    """
    try: im = imageio.imread(str(imname)) 
    except: im = load_Philips30XL_BMP(imname)
    im = im*1.0/255 if np.max(im)<256 else im*1.0/65535   ## 16-bit depth images should have at least one pixel over 255
    if len(im.shape) > 2: im = im[:,:,0] # always using monochrome images only; strip other channels than the first

    white_mask = (im==255 ) # np.max(im))

    if retouch_databar:
        for shift,axis in ((1,0),(-1,0),(1,1),(-1,1),(2,0)): # add  (-2,1),(2,1),  for 2px-wide retouch
            mask = (im==np.max(im))
            im[mask] = np.roll(im, shift, axis)[mask]

    if return_whitemask:
        return im, white_mask
    else:
        return im


## Blurring, denoising and sharpening

def twopixel_despike(im):
    """ Removes "salt" noise of near white pixels in the image. This is to be applied before 
    any other blurring. """
    def twopixel_despike_channel(ch):
            #despiked = np.min(np.dstack([ch[:-1,:], ch[1:,:]]), axis=2)
            spike_threshold = 1.2
            despike_mask = ch[:-1,:]   >   (ch[1:,:] * spike_threshold)
            despiked = ch[:-1,:]
            np.putmask(despiked, despike_mask, ch[1:,:])
            #despiked[despike_mask] = ch[1:,:]

            last_row = ch[-1,:]
            res = np.vstack([despiked, last_row]) # keep shape
            #print(ch.shape,  res.shape)
            return res # vertical pixel-wise de-spiking
    if len(np.shape(im)) == 3:      # handle channels of colour image separately
        return np.dstack([twopixel_despike_channel(channel) for channel in im])
    else:
        return twopixel_despike_channel(im)

def guess_blur_radius_from_spotsize_XL30(image_header):
    """ High-resolution images with high spotsize value are inherently blurred by electron beam size.
    Blur the image accordingly to reduce pixel noise, keeping useful information.
    (Specific for the Philips XL30 microscope.)
    Note this blurring should be done after filtering the salt-and-pepper noise. """
    rad = float(image_header['Magnification'])/5000   *  2**(float(image_header['flSpot']) * .5 - 2.5)
    print(f"For magnif {float(image_header['Magnification'])} and spot {float(image_header['flSpot'])} guessing blur radius {rad} px")
    return rad

def blur(im, radius, twopixel_despike=False):
    def blurring_filter(ch, **kwargs):
        return gaussian_filter(ch, **kwargs)

    if len(np.shape(im)) == 3:      # handle channels of colour image separately
        return np.dstack([blurring_filter(channel, sigma=radius) for channel in im])
    else:
        return blurring_filter(im, sigma=radius)

def unsharp_mask(im, weight, radius, clip_to_max=True):
    if weight:
        unsharp = blur(im, radius)
        im = np.clip(im*(1+weight) - unsharp*weight, 0, np.max(im) if clip_to_max else np.inf)
        #im = np.clip(im*(1+weight) - unsharp*weight, 0, np.sum(im)*8/im.size ) ## TODO fix color clipping?
    return im


## Geometry adjustments

def my_affine_tr(im, trmatrix, shiftvec=np.zeros(2)): ## convenience function around scipy's implementation
    troffset = np.dot(np.eye(2)-trmatrix, np.array(im.shape)/2) ## transform around centre, not corner 
    if np.all(np.isclose(trmatrix, np.eye(2))): return im
    return affine_transform(im, trmatrix, offset=shiftvec+troffset, output_shape=None, output=None, order=3, mode='constant', cval=0.0, prefilter=True)

def find_affine_and_shift(im1, im2, max_shift, decim, use_affine_transform=True, detect_edges=False, rel_smoothing=2.5E-3):
    def find_shift(im1, im2, max_shift, decim):
        """
        shifts im2 against im1 so that a best correlation is found, returns a tuple of the pixel shift

        This approach is fast, but asserts the images are not deformed (shifted only).
        """

        #rel_smoothing = .0025         ## smoothing of the correlation map (not the output), relative to image width
        #rel_smoothing = False      ## no smoothing of the correlation map
        #plot_correlation  = False    ## diagnostics

        if detect_edges: ## search for best overlap of edges (~ Laplacian of the image correlation)
            corr = correlate2d(laplace(im1), im2, mode='valid')     
            lc = np.abs(gaussian_filter(corr, sigma=rel_smoothing*im1.shape[1])) 
        else: ## search for best overlap of brightness
            corr = correlate2d(im1-np.mean(im1), im2-np.mean(im2), mode='valid')     
            corr = gaussian_filter(corr, sigma=rel_smoothing*im1.shape[1])
            lc = np.abs(corr) 
        #cr=1  # post-laplace cropping, there were some edge artifacts

        raw_shifts = (np.unravel_index(np.argmax(np.abs(lc)), lc.shape)) # x,y coords of the optimum in the correlation map
        vshift_rel, hshift_rel = int((lc.shape[0]/2 - raw_shifts[0] - 0.5)), int((lc.shape[1]/2 - raw_shifts[1] - 0.5)) # centre image

        #import time, matplotlib.pyplot as plt ## Optional debugging
        #fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(15,15)); im = ax.imshow(lc)  # 4 lines for diagnostics only:
        #def plot_cross(h,v,c): ax.plot([h/2-5-.5,h/2+5-.5],[v/2+.5,v/2+.5],c=c,lw=.5); ax.plot([h/2-.5,h/2-.5],[v/2-5+.5,v/2+5+.5],c=c,lw=.5)
        #plot_cross(lc.shape[1], lc.shape[0], 'k'); plot_cross(lc.shape[1]-hshift_rel*2, lc.shape[0]-vshift_rel*2, 'w')
        #fig.savefig(f'correlation{time.time()}.png', bbox_inches='tight') ## needs basename path fixing     +image_name.replace('TIF','PNG')

        return np.array([vshift_rel, hshift_rel])*decim, np.eye(2) # shift vector (plus identity affine transform matrix)


    def find_affine(im1, im2, max_shift, decim, max_tr=.1, verbose=False):
        """
        optimizes the image overlap through finding not only translation, but also skew/rotation/stretch of the second image 

        im2 should always be smaller than im1, so that displacement still guarantees 100% overlap

        output: 
            translation vector of the image centres (2)
            affine transform matrix (2x2)
        """
        crop_up     = int(im1.shape[0]/2-im2.shape[0]/2)
        crop_bottom = int(im1.shape[0]/2-im2.shape[0]/2+.5)
        crop_left   = int(im1.shape[1]/2-im2.shape[1]/2)
        crop_right  = int(im1.shape[1]/2-im2.shape[1]/2+.5)
        if detect_edges:
            test_im = laplace(im1[crop_up:-crop_bottom,crop_left:-crop_right]) ## TODO apply gaussian_filter if smoothing
        else:
            test_im = im1[crop_up:-crop_bottom,crop_left:-crop_right]
            test_im -= np.mean(test_im)
        im2_subtr = im2 - np.mean(im2)

        def fitf(p): 
            return -np.abs(np.sum(test_im*my_affine_tr(im2_subtr, p[2:].reshape(2,2), shiftvec=p[:2])))

        bounds = [(-max_shift,max_shift), (-max_shift,max_shift), (1-max_tr, 1+max_tr), (0-max_tr, 0+max_tr),(0-max_tr, 0+max_tr),(1-max_tr, 1+max_tr)]
        result = differential_evolution(fitf, bounds=bounds)
        return np.array(result.x[:2]*decim*.999), np.array(result.x[2:].reshape(2,2))

    if use_affine_transform:    return find_affine(im1, im2, max_shift, decim)    ## Find the optimum affine transform of both images (by fitting 2x2 matrix)
    else:                       return find_shift(im1, im2, max_shift, decim)     ## Find the best correlation of both images by brute-force search

def anisotropic_prescale(im, pixel_anisotropy=1.0): 
    """
    Simple correction of images - some microscopes save them with non-square pixels (e.g. our Philips30XL SEM).

    """
    return zoom(im, [1./pixel_anisotropy] + [1]*(len(im.shape)-1), order=1)

def downscaletwice(im):
    """
    A convenience function to reduce image sizes when pixels are far smaller than SEM beam resolution. 
    Its settings were tuned to reduce visual noise without affecting sharpness of detail. 
    """
    return zoom(
            convolve2d(im, [[.25,.25],[.25,.25]], mode='valid'), 
            [0.5, 0.5] + [1]*(len(im.shape)-2), 
            order=1) 

def auto_crop_black_borders(im, return_indices_only=False):
    """
    Removes all columns and rows that are zero (i.e. black)

    Note that np.ix_(..) serves to adjust t0,t1 dimensions for *rectangular* indexing (instead of *diagonal* one)
    """
    nonzero = np.sum(im,axis=2)!=0   if len(im.shape) > 2   else   im!=0
    t1, t0 = [np.any(nonzero, axis=axis) for axis in range(2)]
    return np.ix_(t0,t1) if return_indices_only else im[np.ix_(t0,t1)] 



## Colors

def auto_contrast_SEM(image, ignore_bottom_part=0.2):
    # Image contrast auto-enhancement (except CL, where intensity is to be preserved)
    im = image - np.min(image) 
    return np.clip(im * 1. / np.max(im[:int(im.shape[0]*(1-ignore_bottom_part)),:]), 0, 1)

def saturate(im, saturation_enhance):
    orig_max = np.max(im)
    monochr = np.dstack([np.sum(im, axis=2)]*3)
    satu = np.clip(im*(1.+saturation_enhance) - monochr*saturation_enhance, 0, orig_max) # prevent negative values / overshoots
    return satu / np.max(satu) * orig_max ## keep peak brightness if too dim

white = [1,1,1]

def pale(color): # average R,G,B values with white - for better readability of colored text
    return np.mean([color, white], axis=0)

def match_wb_and_color(im1, im2): 
    ''' Converts grayscale image 'im2' into a colourful one, and vice versa (the color mode of 'im1' being followed) '''
    if len(im2.shape) > len(im1.shape): im2 = im2[:,:,len(im1.shape)] ## reduce channel depth if needed
    if len(im2.shape) < len(im1.shape): im2 = np.dstack([im2]*len(im1.shape))
    return im2

def hsv_to_rgb(h, s=1, v=1, red_norm=.9, green_norm=.8): ## adapted from https://docs.python.org/2/library/colorsys.html with perceptual coeffs for red and green
    if s == 0.0: return v*red_norm, v*green_norm, v
    i = int(h*6.0) # XXX assume int() truncates!
    f = (h*6.0) - i
    p = v*(1.0 - s)
    q = v*(1.0 - s*f)
    t = v*(1.0 - s*(1.0-f))
    i = i%6
    if i == 0: return np.array([v*red_norm, t*green_norm, p])
    if i == 1: return np.array([q*red_norm, v*green_norm, p])
    if i == 2: return np.array([p*red_norm, v*green_norm, t])
    if i == 3: return np.array([p*red_norm, q*green_norm, v])
    if i == 4: return np.array([t*red_norm, p*green_norm, v])
    if i == 5: return np.array([v*red_norm, p*green_norm, q])

def rgb_palette(n_colors, red_norm=.7, green_norm=.5): # todo stretch hue around orange-yellow-green a bit?
    return np.array([hsv_to_rgb(i,1,1,red_norm, green_norm) for i in np.linspace(0,1-1/n_colors,n_colors)])

def paste_overlay_inplace(bgimage, fgimage, shiftvec, color_tint, normalize=1, channel_exponent=1.):
    """ 
    Adds monochrome `fgimage` to the RGB-coloured background image with specified RGB color_tint

    For efficiency, this function modifies the bgimage in place; if this is not desired, provide this function with 
    a bgimage.copy() instead.
    """
    print(bgimage, fgimage)
    for channel in range(3):
        vs, hs = shiftvec.astype(int)
        vc = int(bgimage.shape[0]/2 - fgimage.shape[0]/2) # vertical ..
        hc = int(bgimage.shape[1]/2 - fgimage.shape[1]/2) # .. and horizontal centers

        sub_image = bgimage[vc-vs:vc+fgimage.shape[0]-vs, 
                hc-hs:hc+fgimage.shape[1]-hs, 
                channel]
        sub_image += np.clip(fgimage**channel_exponent*float(color_tint[channel])/normalize, 0, 1)


## Text/image/drawing overlay routines

def inmydir(fn): return pathlib.Path(__file__).resolve().parent/fn # finds the basename in the script's dir

def text_initialize(typecase_rel_path='typecase.png'):
    typecase_str = ''.join([chr(c) for c in list(range(32,127))+list(range(0x391,0x3a2))+list(range(0x3a3, 0x3aa))+\
        list(range(0x3b1,0x3c2))+list(range(0x3c3,0x3ca))+[0xd7]]) # basic ASCII table + greek 

    try: 
        typecase_img = safe_imload(str(inmydir(typecase_rel_path))) 
    except FileNotFoundError:
        print('No type set found. To generate one: \n\t0. (optionally) turn on moderate pixel hinting, but disable ' +\
                '"RGB sub-pixel hinting" \n\t1. make a screenshot of the line below, \n\t2. convert it to grayscale, '+\
                '\n\t3. crop the 144 glyphs between delimiting blocks and \n\t 4. save it in this folder as typecase.png')
        print(chr(0x2588)+typecase_str.replace('\\',chr(0x29f5))+chr(0x2588)) # (prevents backslash escaping)
    ch, cw = typecase_img.shape[0], round(typecase_img.shape[1]/len(typecase_str)) ## character height and width
    typecase_dict = dict([(c, typecase_img[:,cw*n:cw*n+cw]) for n,c in enumerate(typecase_str)]) ## lookup dict for glyphs
    typecase_dict['ch'], typecase_dict['cw'] = ch, cw
    return typecase_dict, ch, cw

def put_text(im, text, x, y, typecase_dict, color=1, scale=1):
    ch, cw  = typecase_dict['ch'], typecase_dict['cw']
    for n,c in enumerate(text): 
        if x+cw*(n+1)*scale > im.shape[1]: print('Warning, text on image clipped to',text[:n]); break
        else: 
            glyph = typecase_dict.get(c, typecase_dict['?'])
            if scale > 1:
                glyph = np.repeat(glyph, max(1,int(scale)), axis=0)
                glyph = np.repeat(glyph, max(1,int(scale)), axis=1)
                for gx in range(1, glyph.shape[0]-1):
                    for gy in range(1, glyph.shape[1]-1):

                        glyph[gx,gy] = np.modus([glyph[gx,gy]]*1 + [glyph[gx+dx,gy+dy] for dx in [-1,0,1] for dy in  [-1,0,1]]  )
            #print (im[y:y+ch*scale, x+cw*n*scale:x+cw*(n+1)*scale].shape , glyph.shape)
            im[y:y+ch*scale, x+cw*n*scale:x+cw*(n+1)*scale] = match_wb_and_color(im, glyph) * color
    return im

def put_image(im, inserted_img, x, y, color=1):
    if type(inserted_img) is str: inserted_img = safe_imload(inserted_img)
    inserted_img = match_wb_and_color(im, inserted_img[:im.shape[0]-y, :im.shape[1]-x]) ## adjust colors and clip if needed
    im[y:y+inserted_img.shape[0], x:x+inserted_img.shape[1]] = inserted_img * color
    return im

def put_scale(im, x, y, h, xw, color=None):
    """
    Accepts both monochrome (2D) and RGB images (3D array)
    """
    if color is None: color = 1. if len(im.shape) == 2 else np.ones(im.shape[2])*1.
    im[y+2:y+h-2,                     x-1:x+1,     ] = color
    im[y+2:y+h-2,                     x-1+xw:x+1+xw] = color
    im[y+int(h/2)-1:y+int(h/2)+1,   x-1:x+1+xw] = color
    return im

def put_bar(im, x, y, h, xw, color=None):
    if color is None: color = 1. if len(im.shape) == 2 else np.ones(im.shape[2])*1.
    im[y+2:y+h-2, x-1:x+xw] = color
    return im


