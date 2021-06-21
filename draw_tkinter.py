#!/usr/bin/python3
#-*- coding: utf-8 -*-
#https://python-forum.io/Thread-Tkinter-createing-a-tkinter-photoimage-from-array-in-python3

import time
import numpy as np
import tkinter as tk
from scipy.ndimage.filters import laplace

def data2rgbarray(data,
            zoom = [1,1],
            cmap = ('viridis', 'inferno', 'rainbow', 'redblue', 'grey', 'sky')[5],
            emboss = 4,
            vmin = None,
            vmax = None,
            subtract_lines_median = False,
            subtract_lines_mean = False):
    ## Preprocessing useful e.g. for scanning-probe microscopes
    if subtract_lines_median: data -= np.median(data, axis=1, keepdims=True)
    if subtract_lines_mean: data -= np.mean(data, axis=1, keepdims=True)

    ## Auto-scaling
    if vmin is None: vmin = np.min(data)
    if vmax is None: vmax = np.max(data)
    data = (data-vmin)/(vmax-vmin) 

    ## Converting numeric data to [coloured] image
    rgb_lut = { 'grey':   255*np.array([[0, 1], [0, 1], [0, 1]]),
        'viridis': 255*np.array([[.26,.22,.12,.36,.99], [.00,.32,.56,.78,.90], [.32,.54,.55,.38,.14]]),
        'inferno': 255*np.array([[.00,.33,.72,.97,.98], [.04,.06,.21,.55,.99], [.01,.42,.33,.03,.64]]),
        'rainbow': 255*np.array([[0,.0,.0,.9,1,.7,0], [0,.7,.8,.6,0,.0,0], [1,.7,.0,.0,0,.7,1]]),
        'redblue': 255*np.array([[.0,.5,1.,1.,.5], [.0,.8,1.,.8,.0], [.5,1.,1.,.5,.0]]), 
        'sky': 255*np.array([[.00,.15,.29,.79,.99,.99], [.00,.16,.39,.48,.83,.99], [.00,.40,.47,.24,.33,.99]])}
    lut_ticks = np.linspace(0, 1, len(rgb_lut[cmap][0])) # 1st value for negatives
    image = np.dstack([np.interp(data, lut_ticks, rgb_lut[cmap][channel]) for channel in range(3)])

    if emboss:
        for channel in range(3):
            image[:-1,:-1,channel] += np.nan_to_num((data[1:,1:] - data[:-1,:-1])*emboss*255)
        image = np.clip(image, 0, 255)

    ## Zooming (replicating pixels)
    image = np.repeat(image, max(1,int(zoom[0])), axis=0)
    image = np.repeat(image, max(1,int(zoom[1])), axis=1)
    ## Unzooming (decimating pixels)
    image = image[::max(1,int(1/zoom[0]+.5)), ::max(1,int(1/zoom[1]+.5))]
    return image

def rgbarray2tkcanvas(image, canvas, image_id=1):
    #global TKPimage
    PPMimage = f'P6 {image.shape[1]} {image.shape[0]} 255 '.encode() + np.array(image, dtype=np.int8).tobytes()
    TKPimage = tk.PhotoImage(width=image.shape[1], height=image.shape[0], data=PPMimage, format='PPM')
    if hasattr(canvas, 'dummy_image_reference'):
        canvas.itemconfig(image_id, image=TKPimage)
    else:
        #canvas.create_image(canvas.winfo_width(), canvas.winfo_height()//3, image=TKPimage, anchor=tk.NW)
        canvas.create_image(canvas.winfo_width(), canvas.winfo_height()//3, image=TKPimage, anchor=tk.NW)
    #print(canvas.winfo_width(), canvas.winfo_height())
    #canvas.create_image(canvas.winfo_width(), canvas.winfo_height(), image=TKPimage, anchor=tk.NW)
    canvas.dummy_image_reference = TKPimage # prevents garbage collecting of the PhotoImage object

times = 0
timestart = time.time()
image = None
def update():
        global times
        global timestart

        global data
        global data2

        times+=1
        if times%100==0:
            print("%.02f FPS"%(times/(time.time()-timestart)))
            times = 0
            timestart = time.time()

        #scharr = np.array([[ -3-3j, 0-10j,  +3 -3j],
                           #[-10+0j, 0+ 0j, +10 +0j],
                           #[ -3+3j, 0+10j,  +3 +3j]]) # Gx + j*Gy
        #from scipy import signal
        #lap_ker = np.array([[ 1,  2, 1],
                           #[ 2,-12, 2],
                           #[ 1,  2, 1]])
        #lapl = signal.convolve2d(data, lap_ker, boundary='symm', mode='same') 
        lapl = laplace(data) 

        data2 += lapl
        data = data + data2*.01  #  + lapl**3*1e3
        #data[100:] -= data2[100:] *data[100:] *1e-1 
        #data = data + data2*2e-3 - data**5*1e1
        #data = data  - data**5*1e1

        if not times%100:
            data[20:150,20:50] = 0
            data2[20:150,20:50] = 0

        image = data2rgbarray(data, zoom=[3,3], vmax=vmax, vmin=-vmax, cmap='redblue', emboss=0)
        rgbarray2tkcanvas(image=image, canvas=canvas)

        #image2 = data2rgbarray(data2, zoom=[3,3], vmax=vmax, vmin=0, cmap='grey', emboss=0)
        #rgbarray2tkcanvas(image=(image*[1/26,1/26,0])**2+(image2*[0,0,1/16])**2, canvas=canvas)

        root.after(1, update) # scheduling tkinter to run next update after 1 ms





np.random.seed(seed=42)
data = np.random.random((300, 300))
import scipy.ndimage.filters
data = scipy.ndimage.filters.gaussian_filter(data, sigma=12) + scipy.ndimage.filters.gaussian_filter(data, sigma=3)*.3
data -= np.mean(data)
data2 = np.zeros_like(data)
vmax = np.max(data)




root = tk.Tk()
root.geometry("600x600+300+300")
fr = tk.Frame()

fr.master.title("numpy array as tkinter image")
fr.pack(fill=tk.BOTH, expand=True)
canvas = tk.Canvas(fr)
canvas.pack(fill=tk.BOTH, expand=True, anchor=tk.CENTER)

update()
canvas.create_line(250, 200,   250, 250,   200, 250,   250, 200)
root.mainloop()
