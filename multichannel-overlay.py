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
    * sort channels by parametric value, not by file list order
    * fix test5 in fzu - one SEM + one CL fails to extract param that differs
    * fix test6 in fzu - fails to detect params (flAccV etc.)  in edited images, skip this function!
    * join the export with annotate.py
    * don't forget to anisotropically scale at the load time
    * (too aggressive clipping, or clipping channels too early?) colors are still not as crisp as they used to be 
    * is there some colour mismatch between affine-tr and normal operation?
    * interactive GUI D&D application
    * test on windows
"""


try:
    from IPython import get_ipython
    get_ipython().magic('reset -sf')
except:
    pass

## Import common moduli
import sys, os, time, collections, imageio
from pathlib import Path 
import numpy as np
from scipy.ndimage.filters import gaussian_filter
np.warnings.filterwarnings('ignore')

import pure_numpy_image_processing as pnip
import annotate_image



## If no config file found, copy the default one (from the script's own directory)
import pathlib, sys
config_file_name = 'config.txt'
default_config_file_name = 'default_config.txt'
if not pathlib.Path(config_file_name).is_file():
    print(f'Creating {config_file_name} as a copy from {default_config_file_name}')
    with open(config_file_name, 'w') as config_file:
        config_file.write('input_files = ' + ' '.join(sys.argv[1:]))
        for line in (pathlib.Path(__file__).resolve().parent/default_config_file_name).read_text():
            config_file.write(line)

class MyConfig(object):  ## configparser is lame & ConfigIt/localconfig are on pypi
    def __init__(self, config_file, splitter='=', commenter='#'): 
        from ast import literal_eval
        for n, line in enumerate(pathlib.Path(config_file).read_text().split('\n')):
            line = line.split(commenter)[0]
            if line.strip() == "": continue ## skip whitespace or comment-only lines

            try:
                key, value = line.split(splitter, maxsplit=1) 
                try: self.__dict__[key.strip()] = literal_eval(value.strip())
                except SyntaxError: self.__dict__[key.strip()] = value.strip()
            except ValueError:
                print(f"Error parsing config file {config_file}, line {n}: '{line}'")


config = MyConfig(config_file=config_file_name)

def is_extra(imname): 
    return (imname[0] == config.extra_img_label 
            or (config.extra_img_ident.upper() in Path(imname).stem.upper())) 

#for a in dir(config): print(a, getattr(config,a), type(getattr(config,a)))

image_names = sys.argv[1:]  or  getattr(config, 'input_files', '').split()


#colors = matplotlib.cm.gist_rainbow_r(np.linspace(0.25, 1, len([s for s in image_names if not is_extra(s)])))   ## Generate a nice rainbow scale for all non-extra images
#colors = [c*np.array([.8, .7, .9, 1]) for c in colors[::-1]] ## suppress green channel
#colors = pnip.rgb_palette(len([s for s in image_names if not is_extra(s)])
n_color_channels = len([s for s in image_names if not is_extra(s)])
gn = getattr(config, 'green_channel_factor', 0.8) if n_color_channels>2 else 0.8
colors = [pnip.hsv_to_rgb(h=h, green_norm=gn) for  h in 
        np.linspace(1+1/6 if n_color_channels==2 else 1, 1+2/3, n_color_channels)] 
used_colors = []
channel_outputs, extra_outputs = [], []
shiftvec_sum, shiftvec_new, trmatrix_sum, trmatrix_new = np.zeros(2), np.zeros(2), np.eye(2), np.eye(2)   ## Initialize affine transform to identity, and image shift to zero
for image_name in image_names:
    if image_name.lower() == "dummy": colors.pop(); continue
    print('loading', image_name, 'detected as "extra image"' if is_extra(image_name) else ''); 
    newimg = pnip.safe_imload(Path(image_name) / '..' / Path(image_name).name.lstrip(config.extra_img_label), 
            retouch=config.retouch_databar)
    newimg = pnip.anisotropic_prescale(newimg, pixel_anisotropy= getattr(config, 'pixel_anisotropy', 0.91))


    image_header = annotate_image.analyze_header_XL30(image_name)
    #if 'M05' in image_name: image_header={'flAccV':'5000','lDetName':'2','Magnification':'5000','flSpot':'3', 'flWD':'8.3'} ## Manual fix

    # High-resolution images with high-spotsize are inherently blurred by electrn beam size.
    # Blur the image accordingly to reduce pixel noise, keeping useful information.
    # (Specific for the Philips XL30 microscope.)
    radius = float(image_header['Magnification'])/5000   *  2**(float(image_header['flSpot']) * .5 - 2)
    print("RADI", radius)
    if radius > 1: newimg = pnip.blur(newimg, radius=radius)

    if getattr(config, 'force_downsample', 1.0) or \
            ((newimg.shape[1] > getattr(config, 'downsample_size_threshold', 1000)) and 
            (float(image_header['Magnification']) >= getattr(config, 'downsample_magn_threshold', 10000))):
        newimg = pnip.downscaletwice(newimg)

    if getattr(config, 'subtract_min_brightness', False):
        newimg -= np.min(newimg)

    if is_extra(image_name): 
        color_tint = pnip.white 
    else:
        color_tint = colors.pop()
        used_colors.append(color_tint) 

    max_shift = int(config.rel_max_shift*newimg.shape[0])
    if 'image_padding' not in locals(): image_padding = max_shift*len(image_names) ## temporary very wide black padding for image alignment
    newimg_crop = newimg[max_shift:-max_shift-int(newimg.shape[0]*config.databar_pct):config.decim, max_shift:-max_shift:config.decim]*1.0

    if 'refimg' in locals() and not config.disable_transform: ## the first image will be simply put to centre (nothing to align against)
        shiftvec_new, trmatrix_new = pnip.find_affine_and_shift(
                refimg_crop, newimg_crop, max_shift=max_shift, decim=config.decim, 
                use_affine_transform=config.use_affine_transform, 
                detect_edges=config.detect_edges,
                rel_smoothing=config.rel_smoothing)
        shiftvec_sum += shiftvec_new 
        trmatrix_sum += (trmatrix_new - np.eye(2))*config.trmatrix_factor
        print('... is shifted by {:}px against its reference image and by {:}px against the first one'.format(
            shiftvec_new, shiftvec_sum))

    newimg_processed = pnip.my_affine_tr(        ## Process the new image: sharpening, affine transform, and padding...
            np.pad(pnip.unsharp_mask(newimg, weight=(0 if is_extra(image_name) else config.unsharp_weight), radius=config.unsharp_radius), 
                    pad_width=max_shift, mode='constant'), trmatrix_sum) 
    
            
    img_norm =  np.max(newimg_crop)**(1-config.max_brightness_norm) * (np.mean(newimg_crop)*6)**config.mean_brightness_norm
    if not is_extra(image_name):            ## ... then shifting and adding the image to the composite canvas
        if 'composite_output' not in locals(): 
            composite_output = np.zeros([newimg.shape[0]+2*image_padding, newimg.shape[1]+2*image_padding, 3])
        pnip.paste_overlay(composite_output, newimg_processed, shiftvec_sum, color_tint, normalize=img_norm)

    ## Export the image individually (either as colored channel, or as an extra image)
    single_output = np.zeros([newimg.shape[0]+2*image_padding, newimg.shape[1]+2*image_padding, 3])
    pnip.paste_overlay(single_output, newimg_processed, shiftvec_sum, color_tint, normalize=img_norm) 
    target_output = extra_outputs if is_extra(image_name) else channel_outputs
    target_output.append({'im':single_output, 'imname':image_name, 'header':image_header})

    if not config.consecutive_alignment:   ## optionally, search alignment against the very first image
        shiftvec_sum, trmatrix_sum = np.zeros(2), np.eye(2)
    elif is_extra(image_name):      ## undo alignment changes (since extra imgs never affect alignment of further images)
        shiftvec_sum -= shiftvec_new
        trmatrix_sum -= (trmatrix_new - np.eye(2))*config.trmatrix_factor

    if 'refimg' not in locals() or (config.consecutive_alignment and not is_extra(image_name)): ## store the image as a reference
        refimg, refimg_crop = newimg, newimg[:-int(newimg.shape[0]*config.databar_pct):config.decim, ::config.decim]*1.0

## Generate 5th line in the databar: color coding explanation
param_key, param_values = annotate_image.extract_dictkey_that_differs([co['header'] for co in channel_outputs], key_filter=['flAccV']) # 'Magnification', 'lDetName', 
print(param_key, param_values)
print(getattr(config, 'force_label_by_filename', False), config)
if not param_values or getattr(config, 'force_label_by_filename', False):
    param_key, param_values = config.param_in_filename, annotate_image.extract_stringpart_that_differs([co['imname'] for co in channel_outputs])
assert param_values, 'aligned images, but could not extract a scanned parameter from their header nor names'


## Will crop all images identically - according to the extent of unused black margins in the composite image
crop_vert, crop_horiz = pnip.auto_crop_black_borders(composite_output, return_indices_only=True)


## Export individual channels, 
igamma = 1 / getattr(config, 'gamma', 1.0) 

for n, ch_dict, color, param_value in zip(range(len(channel_outputs)), channel_outputs, used_colors, param_values): 
    appendix_line = [[.6, 'Single channel for '], [pnip.white, param_key+' = '], [color, param_value]]
    ch_dict['im'] = annotate_image.add_databar_XL30(ch_dict['im'][crop_vert,crop_horiz,:]**igamma, 
            ch_dict['imname'], 
            ch_dict['header'], 
            appendix_lines=[appendix_line],
            #downscaletwice=getattr(config, 'force_downsample', False)
            ) # -> "Single channel for λ(nm) = 123"
    imageio.imsave(str(Path(ch_dict['imname']).parent / ('channel{:02d}_'.format(n) + Path(ch_dict['imname']).stem +'.png')), ch_dict['im'])

for n, ch_dict in enumerate(extra_outputs): 
    ch_dict['im'] = annotate_image.add_databar_XL30(
            ch_dict['im'][crop_vert,crop_horiz,:]**igamma, ch_dict['imname'], ch_dict['header'], 
            appendix_lines=[[]],
            #downscaletwice=getattr(config, 'force_downsample', False)
            )
    imageio.imsave(str(Path(ch_dict['imname']).parent / ('extra{:02d}_'.format(n) + Path(ch_dict['imname']).stem.lstrip('+')+ '.png')), ch_dict['im'])


## Generate 5th line in the databar for all-channel composite images
summary_ih = channel_outputs[0]['header']     # (take the header of the first file, assuming other have their headers identical)
dbar_appendix = [[[0.6, 'Color by '], [pnip.white, param_key+': ' ] ]]
for color, param_value in zip(used_colors, param_values): dbar_appendix[0].append([color,' '+param_value]) ## append to 0th line of the appending

composite_output /= np.max(composite_output) # normalize all channels
imageio.imsave(
        str(Path(channel_outputs[0]['imname']).parent / ('composite_saturate__.png')), 
        annotate_image.add_databar_XL30(
            pnip.saturate(
                composite_output, 
                saturation_enhance=config.saturation_enhance)[crop_vert,crop_horiz,:]**igamma, 
            channel_outputs[0]['imname'], 
            summary_ih, appendix_lines=dbar_appendix, 
            )
        )
imageio.imsave(
        str(Path(channel_outputs[0]['imname']).parent / 'composite__.png'), 
        annotate_image.add_databar_XL30(
            composite_output[crop_vert,crop_horiz,:]**igamma, 
            channel_outputs[0]['imname'],
            summary_ih, appendix_lines=dbar_appendix)
        )
