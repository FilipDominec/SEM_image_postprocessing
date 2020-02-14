#!/usr/bin/python3
#-*- coding: utf-8 -*-



import numpy as np
import sys, os, time, collections, imageio, warnings, pathlib, scipy.ndimage
warnings.filterwarnings("ignore")

## (microscope-dependent settings)
detectors = {'0': 'SE', '2':'Si diode', '3':'CL'}

## General image-manipulation routines
def match_wb_and_color(im1, im2): 
    ''' Converts grayscale image 'im2' into a colourful one, and vice versa (the color mode of 'im1' being followed) '''
    if len(im2.shape) > len(im1.shape): im2 = im2[:,:,0] ## reduce channel depth if needed
    if len(im2.shape) < len(im1.shape): im2 = np.dstack([im2]*3)
    return im2

## Font overlay routines
def inmydir(fn): return pathlib.Path(__file__).resolve().parent/fn # finds the basename in the script's dir
typecase_str = ''.join([chr(c) for c in list(range(32,127))+list(range(0x391,0x3a2))+list(range(0x3a3, 0x3aa))+\
    list(range(0x3b1,0x3c2))+list(range(0x3c3,0x3ca))+[0xd7]]) # basic ASCII table + greek 

def im_print(im, text, x, y, color=1):
    for n,c in enumerate(text): 
        if x+cw+cw*n>im.shape[1]: print('Warning, text on image clipped to',text[:n]); break
        else: 
            im[y:y+ch, x+cw*n:x+cw*(n+1)] = match_wb_and_color(im, typecase_dict.get(c, typecase_dict['?'])) * color
    return im

## Image overlay
def insert_image(im, inserted_img, x, y, color=1):
    if type(inserted_img) is str: inserted_img = imageio.imread(inserted_img)
    inserted_img = match_wb_and_color(im, inserted_img[:im.shape[0]-y, :im.shape[1]-x]) ## adjust colors and clip if needed
    im[y:y+inserted_img.shape[0], x:x+inserted_img.shape[1]] = inserted_img * color
    return im

def round125(n):
    expo = 10**np.floor(np.log10(n))
    mant = n/expo
    if mant > 5: return 5*expo
    if mant > 2: return 2*expo
    return 1*expo
def putscale(im, x, y, h, xw):
    white = 255 if len(im.shape) == 2 else np.ones(im.shape[2])*255
    im[y+2:y+h-2,                     x-1:x+1,     ] = white
    im[y+2:y+h-2,                     x-1+xw:x+1+xw] = white
    im[y+int(h/2)-1:y+int(h/2)+1,   x-1:x+1+xw] = white
    return im

def annotate_initialize():
    logo_im = imageio.imread(str(inmydir('logo.png'))) 

    try: 
        typecase_img = imageio.imread(str(inmydir('typecase.png'))) # note: imageio.imread did not accept pathlib.Path objects!
    except FileNotFoundError:
        print('No type set found. To generate one: \n\t0. (optionally) turn on moderate pixel hinting, but disable ' +\
                '"RGB sub-pixel hinting" \n\t1. make a screenshot of the line below, \n\t2. convert it to grayscale, '+\
                '\n\t3. crop the text between delimiting blocks and \n\t 4. save it in this folder as typecase.png')
        print(chr(0x2588)+typecase_str.replace('\\',chr(0x29f5))+chr(0x2588)) # (prevents backslash escaping)
    ch, cw = typecase_img.shape[0], round(typecase_img.shape[1]/len(typecase_str)) ## character height and width
    typecase_dict = dict([(c, typecase_img[:,cw*n:cw*n+cw]) for n,c in enumerate(typecase_str)]) ## lookup dict for glyphs
    return logo_im, typecase_dict, ch, cw


downsample_size_threshold = 1000 #[px], smaller image will not be downsampled
downsample_magn_threshold = 5000 #[×], lower magnifications will not be downsampled

# Load images
#for imname in sys.argv[1:]:
    #try:
        #im = imageio.imread(imname)

## Load images
def annotate_process(imnames):
    for imname in imnames:
        im = imageio.imread(imname)

        ## Analyze the TIFF image header specific for Philips/FEI 30XL
        try:
            with open(imname, encoding = "ISO-8859-1") as of: 
                # TODO seek for [DatabarData] first, then count the 194 lines!
                ih = dict(l.strip().split(' = ') for l in of.read().split('\n')[:194] if '=' in l)
        except:
            print('Image {:} does not contain readable SEM metadata, skipping it...'.format(imname))
            continue

        try:
            ## Preprocess the parameters
            anisotropy = .91
            size_x = 117500. / float(ih['Magnification']) /1.03
            size_y = size_x / im.shape[1] * im.shape[0]  / anisotropy
            if size_x > 1000: size_str = '{:<4f}'.format(size_x/1000)[:4] + '×{:<4f}'.format(size_y/1000)[:4] + ' mm'
            elif size_x < 1:  size_str = '{:<4f}'.format(size_x*1000)[:4] + '×{:<4f}'.format(size_y*1000)[:4] + ' nm'
            else:             size_str = '{:<4f}'.format(size_x)[:4]      + '×{:<4f}'.format(size_y)[:4]      + ' μm'

            try: sample_name, author_name = os.path.basename(os.path.dirname(os.path.abspath(imname))).replace('-','_').split('_')[:2]
            except ValueError: sample_name, author_name = '', ''
            if not sample_name:
                try: sample_name, author_name = os.path.basename(os.path.dirname(os.path.dirname(os.path.abspath(imname)))).replace('-','_').split('_')[:2]
                except ValueError: sample_name, author_name = '', ''

            ## Prepare the scale bar
            scale_bar = round125(size_x/4.8) # in μm
            if scale_bar > 1000:     scale_num, scale_unit = scale_bar / 1000, 'mm' 
            elif scale_bar < 1:      scale_num, scale_unit = scale_bar * 1000, 'nm' 
            else:                    scale_num, scale_unit = scale_bar,        'μm' 



            ## Rescale image and, if in SE-mode, normalize it
            im = scipy.ndimage.zoom(im, [1./anisotropy] + [1]*(len(im.shape)-1))

            if detectors.get(ih['lDetName'],'')  not in ('CL',):
                im -= np.min(im[:int(im.shape[0]*.8),:])
                im = np.clip(im * 256. / np.max(im[:int(im.shape[0]*.8),:]),0, 255)


            ## Put the logo & web on the image
            im = np.pad(im, [(0,ch*4)]+[(0,0)]*(len(im.shape)-1), mode='constant')
            im = insert_image(im, logo_im, x=0, y=int(im.shape[0]-int(ch*4/2)-logo_im.shape[0]/2))
            xpos = logo_im.shape[1]+10 if im.shape[1]>logo_im.shape[1]+cw*55 else 0
            if xpos > 0: im = im_print(im, 'www.fzu.cz/~movpe', x=8, y=im.shape[0]-ch, color=.6)

            ## Print the first couple of rows
            im = im_print(im, '{:<6} {:<6} {:<6} {:<6} {:<13} {:<8}'.format(
                'AccV', 'Spot', 'WDist', 'Magnif', 'DimXY', 'Scale:'), x=xpos, y=im.shape[0]-ch*4, color=.6)
            im = im_print(im, '{:<.0f} {:}'.format(
                scale_num, scale_unit), x=xpos+cw*49, y=im.shape[0]-ch*4, color=1)
            im = putscale(im, xpos+cw*42, im.shape[0]-ch*3, ch, int(scale_bar/size_x*im.shape[1]))

            ## Print the second couple of rows
            im = im_print(im, '{:<6.0f} {:<6.1f} {:<6.2f} {:<6} {:<13}'.format(
                float(ih['flAccV']), float(ih['flSpot']), float(ih['flWD']), 
                '{:<.0f}'.format(float(ih['Magnification']))+'×', 
                size_str), x=xpos, y=im.shape[0]-ch*3, color=1)
            im = im_print(im, '{:<13} {:<13} {:<13}'.format(
                'Detector', 'Made', 'Sample name'), x=xpos, y=im.shape[0]-ch*2, color=.6)
            im = im_print(im, '{:<13} {:<13} {:<13}'.format(
                detectors.get(ih['lDetName'],''), 
                author_name+(' ' if author_name else '')+time.strftime('%Y-%m-%d', time.gmtime(os.path.getmtime(imname))), 
                sample_name), x=xpos, y=im.shape[0]-ch, color=1)

            ## TODO: coloured indication of wavelength
                
            ## Export image
            outname = os.path.splitext(imname)[0]+'.png'
            if not os.path.isfile(outname): imageio.imsave(outname, im)
        except Exception as e: 
            import traceback
            print('Error: image {:} skipped: \n\n'.format(imname), e,traceback.print_exc() ), traceback.print_exc()
            

#<<<<<<< HEAD
        # Prepare the scale bar
        #scale_bar = round125(size_x/4.8) # in μm
        #if scale_bar > 1000:     scale_num, scale_unit = scale_bar / 1000, 'mm' 
        #elif scale_bar < 1:      scale_num, scale_unit = scale_bar * 1000, 'nm' 
        #else:                    scale_num, scale_unit = scale_bar,        'μm' 
#
#
#
        # Rescale image and, if in SE-mode, normalize it
#
        #print( (im.shape[1] > downsample_size_threshold) and (float(ih['Magnification']) >= downsample_magn_threshold))
        #print( (im.shape[1] > downsample_size_threshold) , (float(ih['Magnification']) >= downsample_magn_threshold))
        #print( (im.shape[1] , downsample_size_threshold) , (float(ih['Magnification']) , downsample_magn_threshold))
#
        #print(im.shape)
        #if (im.shape[1] > downsample_size_threshold) and (float(ih['Magnification']) >= downsample_magn_threshold):
            #im = scipy.ndimage.zoom(im, [1./anisotropy/2] + [0.5] + [1]*(len(im.shape)-2), order=1)
        #else:
            #im = scipy.ndimage.zoom(im, [1./anisotropy] + [1]*(len(im.shape)-1))
        #print(im.shape)
#
        #if detectors.get(ih['lDetName'],'')  not in ('CL',):
            #im -= np.min(im[:int(im.shape[0]*.8),:])
            #im = np.clip(im * 256. / np.max(im[:int(im.shape[0]*.8),:]),0, 255)
            #im = (im+1)%256
#
#
        # Put the logo & web on the image
        #print(im[-1,:10])
        #im = np.pad(im, [(0,ch*4)]+[(0,0)]*(len(im.shape)-1), mode='constant')
        #print(im[-1,:10])
        #im = im_logo(im, logo_im, x=0, y=int(im.shape[0]-int(ch*4/2)-logo_im.shape[0]/2))
        #xpos = logo_im.shape[1]+10 if im.shape[1]>logo_im.shape[1]+cw*55 else 0
        #if xpos > 0: im = im_print(im, 'www.fzu.cz/~movpe', x=8, y=im.shape[0]-ch, color=.6)
#
        # Print the first couple of rows
        #im = im_print(im, '{:<6} {:<6} {:<6} {:<6} {:<13} {:<8}'.format(
            #'AccV', 'Spot', 'WDist', 'Magnif', 'DimXY', 'Scale:'), x=xpos, y=im.shape[0]-ch*4, color=.6)
        #im = im_print(im, '{:<.0f} {:}'.format(
            #scale_num, scale_unit), x=xpos+cw*49, y=im.shape[0]-ch*4, color=1)
        #im = putscale(im, xpos+cw*42, im.shape[0]-ch*3, ch, int(scale_bar/size_x*im.shape[1]))
#=======
#>>>>>>> 3bf477264aef13a9523b5f0081fbdd6730d9fa94

if __name__ == '__main__':
    logo_im, typecase_dict, ch, cw = annotate_initialize()
    annotate_process(imnames = sys.argv[1:])


