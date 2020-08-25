#!/usr/bin/python3
#-*- coding: utf-8 -*-

## TODO change os.path to pathlib
# TODO: but note that imageio.imread did not accept pathlib.Path objects -> convert it to string first!
# Static user settings
OVERWRITE_ALLOWED = True
downsample_size_threshold = 1000   # [px]: smaller image will not be downsampled
downsample_magn_threshold = 15000  # [×]: lower magnifications will not be downsampled

PIXEL_ANISOTROPY = .91
UNITY_MAGNIF_XDIM = 117500./1.03




import numpy as np
import sys, os, time, collections, imageio, warnings, pathlib
warnings.filterwarnings("ignore")

import pure_numpy_image_processing as pnip

## (microscope-dependent settings)
detectors = {'0': 'SE', '2':'Si diode', '3':'CL'}


def analyze_header_XL30(imname, allow_underscore_alias=True):
        """ 
        Analyze the TIFF image header specific for Philips/FEI 30XL Microscope Control software (running under WinNT early 2000s)

        Accepts:
            imname 
                path to be analyzed
            boolean allow_underscore_alias
                if set to True and image has no compatible header, try loading it from "_filename" in 
                the same directory (the image file was perhaps edited)
        
        Returns a dict of all 194 key/value pairs found in the ascii header
        """
        try:
            with open(imname, encoding = "ISO-8859-1") as of: 
                # TODO seek for [DatabarData] first, then count the 194 lines!
                ih = dict(l.strip().split(' = ') for l in of.read().split('\n')[:194] if '=' in l)
        except:
            if not allow_underscore_alias: 
                print('Error: image {:} does not contain readable SEM metadata, skipping it...'.format(imname))
                
            print('Trying to load metadata from ', pathlib.Path(imname).parent / ('_'+pathlib.Path(imname).name))
            with open(str(pathlib.Path(imname).parent / ('_'+pathlib.Path(imname).name)), encoding = "ISO-8859-1") as of: 
                ih = dict(l.strip().split(' = ') for l in of.read().split('\n')[:194] if '=' in l)
                #ih['lDetName'] = '3' ## XXX FIXME: hack for detector override
        return ih

def add_databar_XL30(im, ih):
    """
    """



## Load images
def annotate_process(imnames):
    for imname in imnames:
        im = imageio.imread(imname)

        ih = analyze_header_XL30(imname)
        #im = add_databar_XL30(im TODO

        try:
            add_databar_XL30(im, ih)
            ## Preprocess the parameters
            size_x = UNITY_MAGNIF_XDIM / float(ih['Magnification'])  / 10. ## XXX
            #size_x = UNITY_MAGNIF_XDIM / float(ih['Magnification'])  
            size_y = size_x / im.shape[1] * im.shape[0]  / PIXEL_ANISOTROPY
            if size_x > 1000: size_str = '{:<4.4f}'.format(size_x/1000)[:4] + '×{:<4.4f}'.format(size_y/1000)[:4] + ' mm'
            elif size_x < 1:  size_str = '{:<4.4f}'.format(size_x*1000)[:4] + '×{:<4.4f}'.format(size_y*1000)[:4] + ' nm'
            else:             size_str = '{:<4.4f}'.format(size_x)[:4]      + '×{:<4.4f}'.format(size_y)[:4]      + ' μm'

            try: sample_name, author_name = os.path.basename(os.path.dirname(os.path.abspath(imname))).split('_')[:2]
            except ValueError: sample_name, author_name = '', ''
            if not sample_name or len(author_name)>2:
                try: sample_name, author_name = os.path.basename(os.path.dirname(os.path.dirname(os.path.abspath(imname)))).split('_')[:2]
                except ValueError: sample_name, author_name = '', ''

            ## Prepare the scale bar
            def round125(n):
                expo = 10**np.floor(np.log10(n))
                mant = n/expo
                if mant > 5: return 5*expo
                if mant > 2: return 2*expo
                return 1*expo
            scale_bar = round125(size_x/4.8) # in μm
            if scale_bar > 1000:     scale_num, scale_unit = scale_bar / 1000, 'mm' 
            elif scale_bar < 1:      scale_num, scale_unit = scale_bar * 1000, 'nm' 
            else:                    scale_num, scale_unit = scale_bar,        'μm' 


            ## Rescale image (and down-scale, if it is high-res and high-magnif)
            im = pnip.anisotropic_prescale(im, pixel_anisotropy=PIXEL_ANISOTROPY, 
                    downscaletwice = (im.shape[1] > downsample_size_threshold) and (float(ih['Magnification']) >= downsample_magn_threshold))

            if detectors.get(ih['lDetName'],'')  not in ('CL',):  
                im = pnip.auto_contrast_SEM(im)


            ## Put the logo & web on the image
            im = np.pad(im, [(0,ch*4)]+[(0,0)]*(len(im.shape)-1), mode='constant')
            im = pnip.put_image(im, logo_im, x=0, y=int(im.shape[0]-int(ch*4/2)-logo_im.shape[0]/2))
            xpos = logo_im.shape[1]+10 if im.shape[1]>logo_im.shape[1]+cw*55 else 0
            if xpos > 0: im = pnip.put_text(im, 'www.fzu.cz/~movpe', x=8, y=im.shape[0]-ch, cw=cw, ch=ch, typecase_dict=typecase_dict, color=.6)

            ## Print the first couple of rows in the databar
            im = pnip.put_text(im, '{:<6} {:<6} {:<6} {:<6} {:<13} {:<8}'.format(
                'AccV', 'Spot', 'WDist', 'Magnif', 'DimXY', 'Scale:'), x=xpos, y=im.shape[0]-ch*4, cw=cw, ch=ch, typecase_dict=typecase_dict, color=.6)
            im = pnip.put_text(im, '{:<.0f} {:}'.format(
                scale_num, scale_unit), x=xpos+cw*49, y=im.shape[0]-ch*4, cw=cw, ch=ch, typecase_dict=typecase_dict, color=1)
            im = pnip.put_scale(im, xpos+cw*42, im.shape[0]-ch*3, ch, int(scale_bar/size_x*im.shape[1]))

            ## Print the second couple of rows in the databar
            im = pnip.put_text(im, '{:<6.0f} {:<6.1f} {:<6.2f} {:<6} {:<13}'.format(
                float(ih['flAccV']), float(ih['flSpot']), float(ih['flWD']), 
                '{:<.0f}'.format(float(ih['Magnification']))+'×', 
                size_str), x=xpos, y=im.shape[0]-ch*3, cw=cw, ch=ch, typecase_dict=typecase_dict, color=1)
            im = pnip.put_text(im, '{:<13} {:<13} {:<13}'.format(
                'Detector', 'Made', 'Sample name'), x=xpos, y=im.shape[0]-ch*2, cw=cw, ch=ch, typecase_dict=typecase_dict, color=.6)
            im = pnip.put_text(im, '{:<13} {:<13} {:<13}'.format(
                detectors.get(ih['lDetName'],''), 
                author_name+(' ' if author_name else '')+time.strftime('%Y-%m-%d', time.gmtime(os.path.getmtime(imname))), 
                sample_name), x=xpos, y=im.shape[0]-ch, cw=cw, ch=ch, typecase_dict=typecase_dict, color=1)

            ## TODO: coloured indication of wavelength
                
            ## Export image
            outname = os.path.splitext(imname)[0]+'.png'
            im = im/200+10
            if not os.path.isfile(outname) or OVERWRITE_ALLOWED: imageio.imsave(outname, im)
        except Exception as e: 
            import traceback
            print('Error: image {:} skipped: \n\n'.format(imname), e,traceback.print_exc() ), traceback.print_exc()
            
if __name__ == '__main__':
    logo_im = imageio.imread(str(pnip.inmydir('logo.png'))) 
    typecase_dict, ch, cw = pnip.text_initialize()
    annotate_process(imnames = sys.argv[1:])


