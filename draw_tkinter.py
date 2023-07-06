#!/usr/bin/python3
#-*- coding: utf-8 -*-
#https://python-forum.io/Thread-Tkinter-createing-a-tkinter-photoimage-from-array-in-python3

import time
import numpy as np
import tkinter as tk

X_SIZE, Y_SIZE = 500, 300
X_ZOOM, Y_ZOOM = 1, 1


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

class RippleTankDemo():

    def __init__(self, canvas, parent_window=None):
        self.times = 0
        self.time_ref = time.time()
        self.image = None
        self.canvas = canvas
        self.parent_window = parent_window

        ## Prepare some initial self.data (smoothed noise)
        np.random.seed(seed=42)
        self.data = np.random.random((Y_SIZE, X_SIZE)) * 1
        self.data -= np.mean(self.data)
        self.data = np.pad(self.data, 1)
        for x in range(100): ## initial smoothing of noise
            self.data[1:-1, 1:-1] -= self.my_laplace2d(self.data)/8
            #self.data = np.pad(self.data,[[1,1],[1,1]])
        
        self.velocity_data = np.zeros_like(self.data)
        self.vmax = max(np.max(self.data), -np.min(self.data)) # fixed symmetric color range
        #vmax = 2


    def my_laplace2d(self, data): # if scipy is not present
        return 4* self.data[1:-1, 1:-1] - self.data[2:, 1:-1] - self.data[:-2, 1:-1] - self.data[1:-1, 2:] - self.data[1:-1, :-2]


    def my_update_routine(self):  # simulating a ripple tank, updating the plot as fast as possible

        self.times+=1
        if self.times%100==0:
            if self.parent_window:
                self.parent_window.title(f"numpy->tkinter @ {self.times/(time.time()-self.time_ref):.1f} FPS")
            self.times, self.time_ref = 0, time.time()

        ## Update wave equation, if scipy present this is perhaps the preferred way ...
        #from scipy.ndimage.filters import laplace
        #self.velocity_data -= laplace(self.data)

        ## Update wave equation
        lapl = self.my_laplace2d(self.data) 
        self.velocity_data[1:-1, 1:-1] -= lapl
        self.data += self.velocity_data*0.4       # (more than 0.4 is unstable)
        self.data[1:-1, 1:-1] -= lapl/1e3   # gently dampen highest spatial frequencies (avoid pixel-wise noise)

        # Any "real-time" interaction possible, examples:
        self.data[20:150,120:150] -= self.velocity_data[20:150,120:150]*0.35 # region of shallow water (the wave is much slower)
        if not self.times%1: self.velocity_data[20:150,20:50] *= .8      # region of smooth wave damping
        if not self.times%100: 
            self.data[170:180,125:145] += .1    # ... stirring the pond, causing shock wave if medium moves
        #self.data = np.roll(self.data, 1, axis=1); self.velocity_data = np.roll(self.velocity_data, 1, axis=1) # ... moving medium 


        ## Update image
        self.canvas.plot_2d_array(
                self.data, 
                zoom=[Y_ZOOM,X_ZOOM], 
                vmax=self.vmax, 
                vmin=-self.vmax, 
                cmap='sky', 
                shading=3.)

        root.after(1, self.my_update_routine) # scheduling tkinter to run next update after 1 ms


## Build the GUI
root = tk.Tk()
root.geometry(f"{X_SIZE*X_ZOOM+5}x{Y_SIZE*Y_ZOOM+5}")
frame = tk.Frame()
frame.pack(fill=tk.BOTH, expand=True)

canvas = ArrayPlottingCanvas(frame)
canvas.pack(fill=tk.BOTH, expand=True, anchor=tk.CENTER)

rippletank = RippleTankDemo(canvas=canvas, parent_window=frame.master)
rippletank.my_update_routine() # start the simulation

canvas.create_line(250, 200,   250, 250,   200, 250,   250, 200) # drawings stay atop
root.mainloop()
