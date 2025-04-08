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
    
    * should also analyze covariance of elements (c.f. p/170816-2dimghistogram_covariance) 
"""


try: # first, clean up after previous run if started in IPython console
    from IPython import get_ipython
    get_ipython().magic('reset -sf') 
except:
    pass

## Import common moduli
import sys
import os
import time
import collections
import pathlib
import sys
import traceback
sys.excepthook = lambda t,v,tb: input(''.join(traceback.format_exception(t, v, tb)) + 
        '\nPress Enter to continue. Please consider reporting this to the developers.')

import imageio
import numpy as np
from scipy.ndimage.filters import gaussian_filter
import tkinter
from tkinter import filedialog
np.warnings.filterwarnings('ignore')

import pure_numpy_image_processing as pnip
import annotate_image


class MyConfig(object):  ## configparser is lame & ConfigIt/localconfig are on pypi
    # TODO switch to the GUIConfig class being developed
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


def is_extra(imname, config):  
    """ extra files are typically SEM images to be aligned & exported, but not to be added in the cathodoluminescence stack """
    return (pathlib.Path(imname).stem[0] == config.extra_img_label 
            or (config.extra_img_ident.upper() in pathlib.Path(imname).stem.upper())) 


def process_images():
    ## If no config file found, copy the default one (from the script's own directory) 
    config_file_name = 'config.txt' # config in current working directory (for each project separately)
            # might rather point to pathlib.Path(image_names[0]).resolve().parent/
    default_config_file_path = pathlib.Path(__file__).resolve().parent/'default_config.txt' # defaults in script's dir

    ## TODO TEST THE LOGIC OF OPENING FILES / CONFIG

    # Get at least one input file name - either by command line arguments, or by graphical dialog
    # This/these may be images or a previous config file
    if len(sys.argv[1:]) > 1:
        input_files = sys.argv[1:]
        print('Input file names given', input_files)
    else:
        # Running Tkinter file dialog without main app is a bit tricky because of dialog window losing focus 
        # TODO check if this makes no trouble with Windows - the background window wouldn't close...
        # https://stackoverflow.com/questions/30678508/how-to-use-tkinter-filedialog-without-a-window 
        print('No input arguments provided, will ask for input image files (or for a config.txt containing file names from previous run)')
        root = tkinter.Tk() 
        #root.withdraw() 
        root.iconify() 
        input_files = filedialog.askopenfilenames(filetypes=[("Images from Philips XL30 ESEM", "*.TIF"), ("Previous processing configuration", "*.txt"), ("All files", "*.*"),])
        root.iconify()
        root.destroy()

    if pathlib.Path(input_files[0]).name == 'config.txt':
        # TODO load the files config_file_path
        config_file_path = pathlib.Path(input_files[0])
        input_files = input_files[1:]
    else:
        # if config.txt not provided explicitly, get it or generate a new one in the directory of first image file
        config_file_path = pathlib.Path(input_files[0]).parent / config_file_name
        if not config_file_path.is_file(): # make a fresh config file
            print(f'Creating {config_file_name} as a copy from {default_config_file_path}; storing current input files choice:', input_files)
            with open(config_file_path, 'w') as config_file:
                config_file.write('input_files = ' + ' '.join(input_files))
                for line in default_config_file_path.read_text('utf-8'): 
                    config_file.write(line)

    config = MyConfig(config_file=config_file_path)

    print('Using configuration options:')
    for a in dir(config): print(a, getattr(config,a), type(getattr(config,a)))

    if input_files:
        print('Following image files were selected by user', input_files)
        image_names = input_files  
    else:  
        image_names = getattr(config, 'input_files', '').split()
        print('No image files were selected by user, using the existing ones in', config_file_path, ': \n\n', image_names)

    #colors = matplotlib.cm.gist_rainbow_r(np.linspace(0.25, 1, len([s for s in image_names if not is_extra(s)])))   ## Generate a nice rainbow scale for all non-extra images
    #colors = [c*np.array([.8, .7, .9, 1]) for c in colors[::-1]] ## suppress green channel
    #colors = pnip.rgb_palette(len([s for s in image_names if not is_extra(s)])
    n_color_channels = len([s for s in image_names if not is_extra(s, config)])
    gn = getattr(config, 'green_channel_factor', 0.8) if n_color_channels>2 else 0.8
    colors = [pnip.hsv_to_rgb(h=h, green_norm=gn) for  h in 
            np.linspace(1+1/6 if n_color_channels==2 else 1, 1+2/3, n_color_channels)] 
    used_colors = []
    channel_outputs, extra_outputs = [], []
    shiftvec_sum, shiftvec_new, trmatrix_sum, trmatrix_new = np.zeros(2), np.zeros(2), np.eye(2), np.eye(2)   ## Initialize affine transform to identity, and image shift to zero
    for image_name in image_names:
        if image_name.lower() == "dummy": colors.pop(); continue
        print('Loading', image_name, 'detected as "extra image"' if is_extra(image_name, config) else ''); 
        newimg = pnip.safe_imload(pathlib.Path(image_name).parent / pathlib.Path(image_name).name.lstrip(config.extra_img_label), 
                retouch_databar=config.retouch_databar)

        image_header = annotate_image.analyze_header_XL30(image_name)
        ## Manual fix: 
        #if 'M05' in image_name: image_header={'flAccV':'5000','lDetName':'2','Magnification':'5000','flSpot':'3', 'flWD':'8.3'} 


        ## Image pre-processing - TODO should be unified between annotate_image.py and multichannel-overlay.py

        # High-resolution images with high spot size are inherently blurred by electron beam size.
        # Blur the image accordingly to reduce pixel noise, keeping useful information.
        # (Specific for the Philips XL30 microscope.)
        newimg = pnip.twopixel_despike(newimg)

        radius = pnip.guess_blur_radius_from_spotsize_XL30(image_header)
        if radius > 1: 
            print("De-noising with Gaussian blur with radius", radius, "px, estimated for SEM resolution (at this magnification & spotsize)")
            newimg = pnip.blur(newimg, radius=radius)

        newimg = pnip.anisotropic_prescale(newimg, pixel_anisotropy= getattr(config, 'pixel_anisotropy', 0.91))

        if getattr(config, 'force_downsample', 1.0) or \
                ((newimg.shape[1] > getattr(config, 'downsample_size_threshold', 1000)) and 
                (float(image_header['Magnification']) >= getattr(config, 'downsample_magn_threshold', 10000))):
            newimg = pnip.downscaletwice(newimg)

        if getattr(config, 'subtract_min_brightness', False):
            newimg -= np.min(newimg)



        ## Geometrical transform - co-aligning images
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
            print('... detected as shifted by {:}px against its reference image and by {:}px against the first one'.format(
                shiftvec_new, shiftvec_sum))

        newimg_processed = pnip.my_affine_tr(        ## Process the new image: sharpening, affine transform, and padding...
                np.pad(pnip.unsharp_mask(newimg, weight=(0 if is_extra(image_name, config) else config.unsharp_weight), radius=config.unsharp_radius), 
                        pad_width=max_shift, mode='constant'), trmatrix_sum) 
        
                
        ## Color & brightness adjustment
        if is_extra(image_name, config): 
            color_tint = pnip.white 
        else:
            color_tint = colors.pop()
            used_colors.append(color_tint) 

        img_norm =  np.max(newimg_crop)**(1-config.max_brightness_norm) * (np.mean(newimg_crop)*6)**config.mean_brightness_norm
        if not is_extra(image_name, config):            ## ... then shifting and adding the image to the composite canvas
            if 'composite_output' not in locals(): 
                composite_output = np.zeros([newimg.shape[0]+2*image_padding, newimg.shape[1]+2*image_padding, 3])
            pnip.paste_overlay(composite_output, newimg_processed, shiftvec_sum, color_tint, normalize=img_norm)

        ## Export the image individually (either as colored channel, or as an extra image)
        single_output = np.zeros([newimg.shape[0]+2*image_padding, newimg.shape[1]+2*image_padding, 3])
        pnip.paste_overlay(single_output, newimg_processed, shiftvec_sum, color_tint, normalize=img_norm) 
        target_output = extra_outputs if is_extra(image_name, config) else channel_outputs
        target_output.append({'im':single_output, 'imname':image_name, 'header':image_header})

        if not config.consecutive_alignment:   ## optionally, search alignment against the very first image
            shiftvec_sum, trmatrix_sum = np.zeros(2), np.eye(2)
        elif is_extra(image_name, config):      ## undo alignment changes (since extra imgs never affect alignment of further images)
            shiftvec_sum -= shiftvec_new
            trmatrix_sum -= (trmatrix_new - np.eye(2))*config.trmatrix_factor

        if 'refimg' not in locals() or (config.consecutive_alignment and not is_extra(image_name, config)): ## store the image as a reference
            refimg, refimg_crop = newimg, newimg[:-int(newimg.shape[0]*config.databar_pct):config.decim, ::config.decim]*1.0

    ## Auto-generate naming of the individual channels
    param_key, param_values = annotate_image.extract_dictkey_that_differs([co['header'] for co in channel_outputs], key_filter=['flAccV']) # 'Magnification', 'lDetName', 
    #print(param_key, param_values)
    #print(getattr(config, 'force_label_by_filename', False), config)
    if not param_values or getattr(config, 'force_label_by_filename', False):
        param_key, param_values = config.param_in_filename, annotate_image.extract_stringpart_that_differs([co['imname'] for co in channel_outputs])
    assert param_values, 'aligned images, but could not extract a scanned parameter from their header nor names'


    ## Will crop all images identically - according to the extent of unused black margins in the composite image
    crop_vert, crop_horiz = pnip.auto_crop_black_borders(composite_output, return_indices_only=True)


    ## Export individual channels, finally cropping them, applying gamma contrast - and providing each with its own databar.
    for n, ch_dict, color, param_value in zip(range(len(channel_outputs)), channel_outputs, used_colors, param_values): 
        appendix_line = [[.6, 'Single channel for '], [pnip.white, param_key+' = '], [pnip.pale(color), param_value]]
        ch_dict['im'] = annotate_image.add_databar_XL30(ch_dict['im'][crop_vert,crop_horiz,:]**(1/getattr(config, 'gamma', 1.0)), 
                ch_dict['imname'], 
                ch_dict['header'], 
                appendix_lines=[appendix_line],
                auto_label_CL_images=False
                #downscaletwice=getattr(config, 'force_downsample', False)
                ) 
        imageio.imsave(str(pathlib.Path(ch_dict['imname']).parent / ('channel{:02d}_'.format(n) + pathlib.Path(ch_dict['imname']).stem +'.png')), ch_dict['im'])

    for n, ch_dict in enumerate(extra_outputs): 
        ch_dict['im'] = annotate_image.add_databar_XL30(
                ch_dict['im'][crop_vert,crop_horiz,:]**(1/getattr(config, 'gamma', 1.0)), ch_dict['imname'], ch_dict['header'], 
                appendix_lines=[[]],
                #downscaletwice=getattr(config, 'force_downsample', False)
                )
        imageio.imsave(str(pathlib.Path(ch_dict['imname']).parent / ('extra{:02d}_'.format(n) + pathlib.Path(ch_dict['imname']).stem.lstrip('+')+ '.png')), ch_dict['im'])


    ## Prepare colored 5th line in the databar for all-channel composite images
    dbar_appendix = [[[0.6, 'Color by '], [pnip.white, param_key+': ' ] ]]
    for color, param_value in zip(used_colors, param_values): 
        dbar_appendix[0].append([pnip.pale(color),' '+param_value]) ## append to 0th line of the appending

    composite_output /= np.max(composite_output) # normalize all channels
    composite_output_sat = pnip.saturate(composite_output, saturation_enhance=config.saturation_enhance)
    for output_image, output_imname in [(composite_output, 'composite'), (composite_output_sat, 'composite_saturate')]:
        imageio.imsave(
                str(pathlib.Path(channel_outputs[0]['imname']).parent / (output_imname + "_" + pathlib.Path(channel_outputs[0]['imname']).stem + ".png")), 
                annotate_image.add_databar_XL30(
                    output_image[crop_vert,crop_horiz,:]**(1/getattr(config, 'gamma', 1.0)), 
                    channel_outputs[0]['imname'], 
                    channel_outputs[0]['header'],  # take the header of the first file, assuming other have their headers identical
                    appendix_lines=dbar_appendix, 
                    auto_label_CL_images=False
                    )
                )

if __name__ == '__main__':
        
    process_images()
