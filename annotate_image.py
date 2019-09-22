#!/usr/bin/python3
#-*- coding: utf-8 -*-

"""
    * could also interpret the tiff image metadata from our SEM:  
                from PIL import Image
                with Image.open('image.tif') as img:
                    img.tag[34680][0].split('\r\n')   # or use direct header loading...
Note: could generate databar like: 
"""

import numpy as np
import sys, os, time, collections, imageio
import scipy.ndimage

## Prepare the font
typecase_str = ''.join([chr(c) for c in list(range(32,127))+list(range(0x391,0x3a2))+list(range(0x3a3, 0x3aa))+\
    list(range(0x3b1,0x3c2))+list(range(0x3c3,0x3ca))]) # basic ASCII table + greek 
try: typecase_img = imageio.imread('typecase.png')
except FileNotFoundError:
    print('No type set found. To generate one: \n\t1. make a screenshot, \n\t2. convert it to grayscale, '+\
            '\n\t3. crop the text between delimiting blocks and \n\t 4. save it in this folder as typecase.png')
    print(chr(0x2588)+typecase_str.replace('\\',chr(0x29f5))+chr(0x2588)) # (prevents backslash escaping)
ch, cw = typecase_img.shape[0], int(typecase_img.shape[1]/len(typecase_str)) ## character height and width
typecase_dict = dict([(c, typecase_img[:,cw*n:cw*n+cw]) for n,c in enumerate(typecase_str)]) ## lookup dict for glyphs
def match_wb_and_color(im1, im2): 
    ''' Allows to put a grayscale image in a colourful one, and vice versa (in which case it is converted to grays) '''
    if len(im2.shape) > len(im1.shape): im2 = im2[:,:,0] ## reduce channel depth if needed
    if len(im2.shape) < len(im1.shape): im2 = np.dstack([im2]*3)
    return im2
def im_print(im, text, x, y, color=1):
    for n,c in enumerate(text): 
        if x+cw+cw*n>im.shape[1]: print('Warning, text on image clipped to',text[:n]); break
        else: 
            im[y:y+ch, x+cw*n:x+cw*(n+1)] = match_wb_and_color(im, typecase_dict.get(c, typecase_dict['?'])) * color
            #im[y:y+ch, x+cw*n:x+cw*(n+1),:] = np.outer(typecase_dict.get(c, typecase_dict['?']), color*np.ones(3))
    return im

## Prepare the logo
def im_logo(im, logo_img, x, y, color=1):
    if type(logo_img) is str: logo_img = imageio.imread(logo_img)
    logo_img = match_wb_and_color(im, logo_img[:im.shape[0]-y, :im.shape[1]-x]) ## adjust colors and clip if needed
    im[y:y+logo_img.shape[0], x:x+logo_img.shape[1]] = logo_img * color
    return im


#imname = 'test283/channel0_10L415A.png'
#imname = 'test283/10L415A.TIF'
#imname = '/home/dominecf/LIMBA/SEM-dle_cisla_vzorku/2019/319B-FH-190613_190625/10M460C.TIF'
imname = '/home/dominecf/LIMBA/SEM-dle_cisla_vzorku/2019/319B-FH-190613_190625/10NSA.TIF'

## Analyze the TIFF image header specific for Philips/FEI 30XL
with open(imname, encoding = "ISO-8859-1") as of: 
    ih = dict(l.strip().split(' = ') for l in of.read().split('\n')[:194] if '=' in l)
#for par in ('flAccV', 'flSpot', 'lDetName', 'flWD', 'Magnification', 'SizeX', 'SizeY'):
    #print(par, '=', ih[par])



im = imageio.imread(imname)


#logo_im = imageio.imread('logo2.png') # test
logo_im = imageio.imread('logo_rgb.png') # test

size_x = 117500. / float(ih['Magnification']) * .6
size_y = size_x / 1424*968  / .91 
if size_x > 1000: 
    size_str = '{:<4f}'.format(size_x/1000)[:4] + 'x' + '{:<4f}'.format(size_y/1000)[:4] + ' mm'
elif size_x < .1:                                                                      
    size_str = '{:<4f}'.format(size_x*1000)[:4] + 'x' + '{:<4f}'.format(size_y*1000)[:4]  + ' nm'
else:                                                                                        
    size_str = '{:<4f}'.format(size_x)[:4]      + 'x' + '{:<4f}'.format(size_y)[:4]            + ' μm'

def putscale(im, x,y,h, xw):
    white = 255 if len(im.shape) == 2 else np.ones(im.shape[2])*255
    im[y+2:y+h-2,                     x-1:x+1,     ] = white
    im[y+2:y+h-2,                     x-1+xw:x+1+xw] = white
    im[y+int(h/2)-1:y+int(h/2)+1,   x-1:x+1+xw] = white
    return im

def round125(n):
    expo = 10**np.trunc(np.log10(n))
    mant = n/expo
    if mant > 5: return 5
    if mant > 2: return 2
    return 1
scale_bar = round125(size_x/5) # in μm




#print(im.shape)
im = scipy.ndimage.zoom(im, [1./.91] + [1]*(len(im.shape)-1))
im -= np.min(im)
print(np.max(im[:int(im.shape[0]*.8),:]))
im = np.clip(im * 256. / np.max(im[:int(im.shape[0]*.8),:]),0, 255)

print(np.max(im[:int(im.shape[0]*.8),:]))
xpos = logo_im.shape[1]+10 if im.shape[1]>1200 else 0

sample_name = '' # TODO ...
author_name = '' # TODO ...

im = np.pad(im, [(0,ch*4)]+[(0,0)]*(len(im.shape)-1), mode='constant')
im = im_logo(im, logo_im, x=0, y=int(im.shape[0]-int(ch*4/2)-logo_im.shape[0]/2))
im = im_print(im, '{:<6} {:<6} {:<6} {:<6} {:<13} {:<8}'.format(
    'AccV', 'Spot', 'WDist', 'Magnif', 'DimXY', 'Scale: {:<.0f} μm'.format(scale_bar)), x=xpos, y=im.shape[0]-ch*4, color=.6)
im = im_print(im, '{:<6.0f} {:<6.1f} {:<6.2f} {:<6} {:<13}'.format(
    float(ih['flAccV']), float(ih['flSpot']), float(ih['flWD']), 
    '{:<.0f}'.format(float(ih['Magnification']))+'x', 
    size_str), x=xpos, y=im.shape[0]-ch*3, color=1)
im = im_print(im, '{:<13} {:<13} {:<13}'.format('Detector', 'Made', 'Sample name'), x=xpos, y=im.shape[0]-ch*2, color=.6)
im = im_print(im, '{:<13} {:<13} {:<13}'.format(ih['lDetName'], time.strftime('%Y-%m-%d', time.gmtime(os.path.getmtime(imname))), sample_name), x=xpos, y=im.shape[0]-ch, color=1)
if im.shape[1]> 1200: im = im_print(im, 'www.fzu.cz/~movpe', x=98, y=im.shape[0]-ch, color=.6)
im = putscale(im, xpos+465, im.shape[0]-ch*3, ch, int(scale_bar/size_x*im.shape[1]))
    

imageio.imsave('out.png', im)
