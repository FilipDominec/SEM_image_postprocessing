#!/usr/bin/python3
#-*- coding: utf-8 -*-
#https://python-forum.io/Thread-Tkinter-createing-a-tkinter-photoimage-from-array-in-python3

import time
import numpy as np
import tkinter as tk

X_SIZE, Y_SIZE = 200, 200
X_ZOOM, Y_ZOOM = 3, 3


class ArrayPlottingCanvas(tk.Canvas):
    pass

    rgb_lut = { 'grey':   255*np.array([[0, 1], [0, 1], [0, 1]]),
        'viridis': 255*np.array([[.26,.22,.12,.36,.99], [.00,.32,.56,.78,.90], [.32,.54,.55,.38,.14]]),
        'inferno': 255*np.array([[.00,.33,.72,.97,.98], [.04,.06,.21,.55,.99], [.01,.42,.33,.03,.64]]),
        'rainbow': 255*np.array([[0,.0,.0,.9,1,.7,0], [0,.7,.8,.6,0,.0,0], [1,.7,.0,.0,0,.7,1]]),
        'redblue': 255*np.array([[.0,.5,1.,1.,.5], [.0,.8,1.,.8,.0], [.5,1.,1.,.5,.0]]), 
        'sky': 255*np.array([[.00,.15,.29,.79,.99,.99], [.00,.16,.39,.48,.83,.99], [.00,.40,.47,.24,.33,.99]])}

    def plot_2d_array(self, 
                data,
                zoom = [1,1],
                cmap = ('viridis', 'inferno', 'rainbow', 'redblue', 'grey', 'sky')[5],
                shading = 4,
                vmin = None,
                vmax = None,
                subtract_lines_median = False,
                subtract_lines_mean = False):

        ## Line-noise preprocessing useful e.g. for scanning-probe microscopes
        if subtract_lines_median: data -= np.median(data, axis=1, keepdims=True)
        if subtract_lines_mean: data -= np.mean(data, axis=1, keepdims=True)

        ## Auto-scaling
        if vmin is None: vmin = np.min(data)
        if vmax is None: vmax = np.max(data)
        data = (data-vmin)/(vmax-vmin) 

        ## Converting numeric data to [coloured] image, using look-up table and shading
        lut_ticks = np.linspace(0, 1, len(self.rgb_lut[cmap][0])) # 1st value for negatives
        image = np.dstack([np.interp(data, lut_ticks, self.rgb_lut[cmap][channel]) for channel in range(3)])

        if shading:
            for channel in range(3):
                image[:-1,:-1,channel] += np.nan_to_num((data[1:,1:] - data[:-1,:-1])*shading*255)
            image = np.clip(image, 0, 255)

        ## Zooming (replicating pixels)
        image = np.repeat(image, max(1,int(zoom[0])), axis=0)
        image = np.repeat(image, max(1,int(zoom[1])), axis=1)
        ## Unzooming (decimating pixels)
        image = image[::max(1,int(1/zoom[0]+.5)), ::max(1,int(1/zoom[1]+.5))]

        # The actual image generation & plotting
        PPMimage = f'P6 {image.shape[1]} {image.shape[0]} 255 '.encode() + np.array(image, dtype=np.int8).tobytes()
        TKPimage = tk.PhotoImage(width=image.shape[1], height=image.shape[0], data=PPMimage, format='PPM')
        if hasattr(self, 'dummy_image_reference'):  # 
            self.itemconfig(1, image=TKPimage)
        else: 
            self.create_image(self.winfo_width(), self.winfo_height()//3, image=TKPimage, anchor=tk.NW)
        self.dummy_image_reference = TKPimage # crucial: prevents garbage collecting of the PhotoImage object

def my_laplace2d(data): # if scipy is not present
    return 4* data[1:-1, 1:-1] - data[2:, 1:-1] - data[:-2, 1:-1] - data[1:-1, 2:] - data[1:-1, :-2]

times = 0
time_ref = time.time()
image = None
def my_update_routine():  # simulating a ripple tank, updating the plot as fast as possible
        #TODO (times, time_ref, data, velocity_data) = state  
        global times
        global time_ref

        global data
        global velocity_data

        times+=1
        if times%100==0:
            frame.master.title(f"numpy->tkinter @ {times/(time.time()-time_ref):.1f} FPS")
            times, time_ref = 0, time.time()

        ## Update wave equation, perhaps the preferred way if scipy present...
        #from scipy.ndimage.filters import laplace
        #velocity_data -= laplace(data)

        ## Update wave equation, requiring only numpy
        lapl = my_laplace2d(data) 
        velocity_data[1:-1, 1:-1] -= my_laplace2d(data)
        data += velocity_data*0.4       # (more than 0.4 is unstable)
        data[20:150,120:150] -= velocity_data[20:150,120:150]*0.35 # Here the wave is much slower (..e. shallow water?)

        ## Slight artificial smoothing to dampen pixel-wise noise (avoid numeric dispersion near Brillouin-zone edge)
        data[1:-1, 1:-1] -= my_laplace2d(data)/3e3



        # Any "real-time" interaction possible
        #if not times%1: # ... region of smooth wave damping
            #velocity_data[20:150,20:50] *= .8 

        #data = np.roll(data, 1, axis=1); velocity_data = np.roll(velocity_data, 1, axis=1) # ... moving medium 

        data[165:170,125:130] += .1    # ... stirring the pond, causing shock wave if medium moves



        ## Update image
        image = canvas.plot_2d_array(data, zoom=[X_ZOOM,Y_ZOOM], vmax=vmax, vmin=-vmax, cmap='viridis', shading=3.)

        root.after(1, my_update_routine) # scheduling tkinter to run next update after 1 ms


## Prepare some initial data (smoothed noise)
np.random.seed(seed=42)
data = np.random.random((X_SIZE, Y_SIZE)) * 1
data -= np.mean(data)
data = np.pad(data, 1)

#import scipy.ndimage.filters      # (the preferred way, if scipy present)
#data = scipy.ndimage.filters.gaussian_filter(data, sigma=12) +\
        #scipy.ndimage.filters.gaussian_filter(data, sigma=3)*.3
for x in range(100):
    data[1:-1, 1:-1] = data[1:-1, 1:-1] - my_laplace2d(data)/8
    #data = np.pad(data,[[1,1],[1,1]])

velocity_data = np.zeros_like(data)
vmax = max(np.max(data), -np.min(data)) # fixed symmetric color range
#vmax = 2

#TODO state = [times, time_ref, data, velocity_data]


## Build the GUI
root = tk.Tk()
root.geometry(f"{X_SIZE*X_ZOOM+5}x{Y_SIZE*Y_ZOOM+5}")
frame = tk.Frame()
frame.pack(fill=tk.BOTH, expand=True)

canvas = ArrayPlottingCanvas(frame)
canvas.pack(fill=tk.BOTH, expand=True, anchor=tk.CENTER)

#print (time.time(), flush=1)
my_update_routine()
#canvas.create_line(250, 200,   250, 250,   200, 250,   250, 200) # drawings stay atop
root.mainloop()
