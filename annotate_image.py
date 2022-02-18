#!/usr/bin/python3
#-*- coding: utf-8 -*-

# Note: visually, any de-noising technique seems to obstruct the scientific
#   value so no median filters etc. applied. Downscaling makes sense for 
#   high-magnif imgs, though.
# Static user settings
OVERWRITE_ALLOWED = True
downsample_size_threshold = 1000
downsample_magn_threshold = 15000
PIXEL_ANISOTROPY = .91
UNITY_MAGNIF_XDIM = 117500./1.03




import numpy as np
import sys, time, imageio, warnings, pathlib
warnings.filterwarnings("ignore")
import pure_numpy_image_processing as pnip

## (microscope-dependent settings)
detectors = {'0': 'SE', '2':'Aux', '3':'CL'}


def analyze_header_XL30(imname, allow_underscore_alias=True):
        """ 
        Analyze the TIFF image header, which is "specific" for Philips/FEI 30XL 
        microscope control software (running under WinNT from early 2000s)

        Accepts:
            imname 
                path to be analyzed
            boolean allow_underscore_alias
                if set to True and image has no compatible header, try loading
                it from "_filename" in the same directory (the image file was 
                perhaps edited)
        
        Returns a dict of all 194 key/value pairs found in the ascii header,
        or {} if no header is found.
        
        """
        try:
            with open(imname, encoding = "ISO-8859-1") as of: 
                # TODO seek for [DatabarData] first, then count the 194 lines!
                ih = dict(l.strip().split(' = ') for l in of.read().split('\n')[:194] if '=' in l)
        except:
            print('Warning: image {:} does not contain readable SEM metadata'.format(imname))
            if not allow_underscore_alias: 
                 print('    skipping it...')
            else:
                print('Trying to load metadata from ', pathlib.Path(imname).parent / ('_'+pathlib.Path(imname).name))
                try: 
                    with open(str(pathlib.Path(imname).parent / ('_'+pathlib.Path(imname).name)), encoding = "ISO-8859-1") as of: 
                        ih = dict(l.strip().split(' = ') for l in of.read().split('\n')[:194] if '=' in l)
                        #ih['lDetName'] = '3' ## XXX FIXME: hack for detector override
                except FileNotFoundError: 
                    return {} ## empty dict 
        return ih



def extract_dictkey_that_differs(dict_list, key_filter=None):
    """
    Similar to extract_stringpart_that_differs. Searches for the (first) key that leads to a difference among supplied dicts.

    >>> extract_dictkey_that_differs([{'a':10, 'b':5}, {'a':10, 'b':5}, {'a':10, 'b':6}])
    ('b', [5, 5, 6])
    """
    dict_list = list(dict_list)
    #print("DEBUG: dict_list = ", dict_list)
    if isinstance(dict_list[0], dict) and dict_list[0]:
        for key in (key_filter if key_filter else dict_list[0].keys()):
            for dic in dict_list[1:]:
                if dic[key] != dict_list[0][key]:
                    return key, [d[key] for d in dict_list]
    return None, None

def extract_stringpart_that_differs(str_list):
    """
    Recognizes alpha- and numeric- parts of a string. Getting a list of such similar strings, finds the part that differs.

    >>> extract_stringpart_that_differs(['10.3K380.TIF', '10.3K400.TIF', '10.3K420.TIF',])
    ('λ(nm)', ('380', '400', '420'))
    """
    def split_string_alpha_numeric(name):
        """
        Splits a string into minimum number of chunks, so that each chunk either
        1) contains number-like characters [ASCII number less than ord("A")], or,
        2) contains letter-like characters [ASCII number equal or more than ord("A")].
        Additionally, underscore _ is always split, serving as a forced separator.
        Last dot is split, too, as it usually separates file name extension.
        Number- and letter-like chunks are returned in a list of strings (no conversion).
        >>> split_string_alpha_numeric('10.3K380.TIF')
        ['10.3', 'K', '380', 'TIF']
        >>> split_string_alpha_numeric('10.3K3_80.TIF')
        ['10.3', 'K', '3', '80', 'TIF']
        """
        return ''.join((l+' ' if (ord(r)-63)*(ord(l)-63)<0 else l) for l,r in zip(name,name[1:]+'_'))[::-1].replace('.',' ',1)[::-1].split()
    str_list = list(str_list)
    assert len(str_list)>1
    assert isinstance(str_list[0], str)
    for column in zip(*[split_string_alpha_numeric(name) for name in str_list]):
        for field in column[1:]:
            if field != column[0]:
                return column
    return None # i.e. all strings are the same?



def add_databar_XL30(im, imname, ih, extra_color_list=None, appendix_lines=[], appendix_bars=[],):
    """
    Input:
        * image (as a 2D numpy array), 
        * its file name, and 
        * image header from the Siemens XL30 SEM (as a dict)
            * if it is a list/tuple, additional line sums up the difference
            * if there is no difference in the headers, it is extracted from the filenames
        * downsample_size_threshold [px]: smaller image will not be downsampled
        * downsample_magn_threshold: lower SEM magnifications will not be downsampled
    Note: 
        * the 'lDetName' parameter changes the behaviour:
            * auto-scaling colour if "SE"
            * multi-colouring if 'lDetName'[0] == "CL" and 
    Note2: 

    Output: 
        * the same image with a meaningful databar in the image bottom
    """

    def extract_sample_author_name(filepath, max_depth=3):
        """ 
        This function assumes that directories are named as SS*_AA_YYMMDD/ 
        where SSSSS is the sample name, AA author and YYMMDD is the date.
        >>> extract_sample_author_name('./123_JD_200725/')  # to test, make the dir first
        ('123', 'JD')
        """
        sample_name_, author_name_ = '', ''
        for parent in list(pathlib.Path(filepath).resolve().parents)[:max_depth]:
            try: 
                sample_name_, author_name_ = parent.name.split('_')[:2]
            except ValueError: pass 
            if len(author_name_)==2 and author_name_.isupper():
                return sample_name_, author_name_
        return '', ''
    sample_name, author_name = extract_sample_author_name(imname)


    ## Prepare the scale bar
    def log_floor(n, mantissa_thresholds=(1, 2, 5)):
        """
        Rounds a number downwards with regards to a logarithmic scale. By 
        default, the returned values are one from the following set:
            ... 0.5, 1.0, 2, 5, 10, 20, 50, 100, ...
        
        Rounding to closest power of 10 is obtained simply with 
            mantissa_thresholds=(1,)

        In electrical engineering, standard resistor values are given by
            mantissa_thresholds=(1, 1.5, 2.2, 3.3, 4.7, 6.8)
        >>> log_floor(0.15)
        0.10000000000000001
        >>> log_floor(7e23)
        4.9999999999999999e+23
        """
        expo = 10**np.floor(np.log10(n))
        mant = n/expo
        for mant_thr in np.sort(mantissa_thresholds)[::-1]: 
            if mant>=mant_thr: 
                return mant_thr * expo



    ## Initialize raster graphics to be put in the image
    logo_im = pnip.safe_imload(pnip.inmydir('logo.png'))
    typecase_dict, ch, cw = pnip.text_initialize()

    if not ih or detectors.get(ih['lDetName'],'')  not in ('CL',):  
        im = pnip.auto_contrast_SEM(im)

    ## Put the logo & web on the image
    dbartop = im.shape[0] #+ch*(4+len(appendix_lines))
    im = np.pad(im, [(0,ch*(4+len(appendix_lines))+int(ch/2)*len(appendix_bars))]+[(0,0)]*(len(im.shape)-1), mode='constant')
    im = pnip.put_image(im, logo_im, x=0, y=int(dbartop+ch*1))
    xpos = logo_im.shape[1]+10 if im.shape[1]>logo_im.shape[1]+cw*55 else 0
    if xpos > 0: im = pnip.put_text(im, 'www.fzu.cz/~movpe', x=8, y=dbartop+ch*3, cw=cw, ch=ch, typecase_dict=typecase_dict, color=.6)

    if ih: 
        ## Preprocess the parameters
        size_x = UNITY_MAGNIF_XDIM / float(ih['Magnification'])  
        size_y = size_x / im.shape[1] * im.shape[0]  / PIXEL_ANISOTROPY
        if size_x > 1000: size_str = '{:<4f}'.format(size_x/1000)[:4] + '×{:<4f}'.format(size_y/1000)[:4] + ' mm'
        elif size_x < 1:  size_str = '{:<4f}'.format(size_x*1000)[:4] + '×{:<4f}'.format(size_y*1000)[:4] + ' nm'
        else:             size_str = '{:<4f}'.format(size_x)[:4]      + '×{:<4f}'.format(size_y)[:4]      + ' μm'

        ## Print the first couple of rows in the databar
        scale_bar = log_floor(size_x/4.8) # in μm
        if scale_bar > 1000:     scale_num, scale_unit = scale_bar / 1000, 'mm' 
        elif scale_bar < 1:      scale_num, scale_unit = scale_bar * 1000, 'nm' 
        else:                    scale_num, scale_unit = scale_bar,        'μm' 
        im = pnip.put_text(im, '{:<6} {:<6} {:<6} {:<6} {:<13} {:<8}'.format(
            'AccV', 'Spot', 'WDist', 'Magnif', 'DimXY', 'Scale:'), x=xpos, y=dbartop+ch*0, cw=cw, ch=ch, typecase_dict=typecase_dict, color=.6)
        im = pnip.put_text(im, '{:<.0f} {:}'.format(
            scale_num, scale_unit), x=xpos+cw*49, y=dbartop+ch*0, cw=cw, ch=ch, typecase_dict=typecase_dict, color=1)
        im = pnip.put_scale(im, xpos+cw*42, dbartop+ch*1, ch, int(scale_bar/size_x*im.shape[1]))
        im = pnip.put_text(im, '{:<6.0f} {:<6.1f} {:<6.2f} {:<6} {:<13}'.format(
            float(ih['flAccV']), float(ih['flSpot']), float(ih['flWD']), 
            '{:<.0f}'.format(float(ih['Magnification']))+'×', 
            size_str), x=xpos, y=dbartop+ch*1, cw=cw, ch=ch, typecase_dict=typecase_dict, color=1)

        ## Print the second couple of rows in the databar
        im = pnip.put_text(im, '{:<13} {:<13} {:<13}'.format(
            'Detector', 'Made', 'Sample name'), x=xpos, y=dbartop+ch*2, cw=cw, ch=ch, typecase_dict=typecase_dict, color=.6)
        im = pnip.put_text(im, '{:<13} {:<13} {:<13}'.format(
            detectors.get(ih['lDetName'],''), 
            author_name+(' ' if author_name else '')+time.strftime('%Y-%m-%d', time.gmtime(pathlib.Path(imname).stat().st_ctime)), 
            sample_name), x=xpos, y=dbartop+ch*3, cw=cw, ch=ch, typecase_dict=typecase_dict, color=1)

    for nline, line in enumerate(appendix_lines):
        xcaret = xpos
        for color, content in line:
            #if type(content) == str:
            im = pnip.put_text(im, content, x=xcaret, y=dbartop+ch*(4+nline), cw=cw, ch=ch, typecase_dict=typecase_dict, color=color)
            xcaret += cw*len(content)
    for nline, barline in enumerate(appendix_bars):
        xcaret = xpos
        for bar in barline:
            # TODO test - draw a bar --> pnip.
            def put_bar(im, x, y, h, xw, color=None):
                if color is None: color = 1. if len(im.shape) == 2 else np.ones(im.shape[2])*1.
                im[y+2:y+h-2, x-1:x+xw] = color
                return im
            if isinstance(bar, dict) and bar.get('style') == 'bar':
                print('BL',barline)
                im = put_bar(im, x=xcaret, y=dbartop+ch*(4+len(appendix_lines))+int(ch/200)*nline, h=int(ch/2)-2, xw=bar.get('xwidth'), color=bar.get('color',1))
                #im = put_bar(im, x=xcaret, y=dbartop+ch*(4+len(appendix_lines))+int(ch/2)*nline, h=int(ch/2)-1, xw=content.get('xwidth'))
                xcaret += bar.get('xpitch')

    return im



## Load images
def annotate_individually(imnames):
    for imname in imnames:
        print(f'Annotating {imname}')
        im = pnip.safe_imload(imname, retouch=True)

        ih = analyze_header_XL30(imname)

        ## Rescale image to make pixels isotropic 
        ## Down-scale it if pixel size is far smaller than SEM resolution
        im = pnip.anisotropic_prescale(
                im, 
                pixel_anisotropy=PIXEL_ANISOTROPY, 
                )
        if ((im.shape[1] > downsample_size_threshold) and (float(ih['Magnification']) >= downsample_magn_threshold)):
            im = pnip.downscaletwice(im)

        im = add_databar_XL30(im, imname, ih)

        try: ## Export image
            outname = pathlib.Path(imname).parent / (pathlib.Path(imname).stem + '.png')
            if not pathlib.Path(outname).is_file() or OVERWRITE_ALLOWED: 
                imageio.imsave(str(outname), im)
                # try metadata with PIL? https://stackoverflow.com/questions/58399070/how-do-i-save-custom-information-to-a-png-image-file-in-python
                print(f"OK: Processed {imname} and exported to {outname}.")
            else: print(f"Warning: file {imname} exists, and overwriting was not allowed. Not saving.")
        except Exception as e: 
            import traceback
            print('Error: image {:} skipped: \n\n'.format(outname), e,traceback.print_exc() ), traceback.print_exc()
            
if __name__ == '__main__':
    imnames = sys.argv[1:]
    if imnames: annotate_individually()
    else: print("Please specify one or more TIF files to be processed individually")


