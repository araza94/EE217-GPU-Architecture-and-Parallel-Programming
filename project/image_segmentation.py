#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sun Feb 24 15:38:07 2019

@author: araza
"""

import numpy as np
from PIL import Image
import pycuda.driver as cuda
import pycuda.autoinit
from pycuda.compiler import SourceModule
import matplotlib.pyplot as plt

h_in = np.array(Image.open("lenna.png"),dtype=np.uint8)
h_out = np.empty(h_in.shape[:-1],dtype=np.uint8)

d_in = cuda.mem_alloc(h_in.nbytes)
d_out = cuda.mem_alloc(h_out.nbytes)

cuda.memcpy_htod(d_in, h_in)
            
mod = SourceModule(
    """
    __constant__ unsigned char centroids[48];
    __global__ void rgb2grey(const unsigned char *in, unsigned char *out, int height, int width, int k){
        int y = threadIdx.y+ blockIdx.y* blockDim.y;
        int x = threadIdx.x+ blockIdx.x* blockDim.x;
        
        if (x < width && y < height){
            int idx = (width*y+x)*3;
            //out[width*y+x] = centroids[3];//(unsigned char)(0.299f*in[idx]+ 0.587f*in[idx+1] + 0.114f*in[idx+2]);
            
            
            float d[16];            
            for (int h=0; h<k; h++){
                d[h]=0;
                for (int i=0; i<3; i++){
                    float in_f = in[idx+i];
                    float c_f = centroids[3*h+i];            
                    d[h] += (in_f-c_f)*(in_f-c_f);
                 }            
            }   
            //if (d[0]<d[1]) out[width*y+x] = 0;
            //else out[width*y+x] = 1;            
            float min = d[0];
            int loc = 0;
            for (int c=1; c<k; c++){
                if (d[c] < min){
                    min = d[c];
                    loc = c;
                }                    
            }
            out[width*y+x] = loc;            
        }
    }        
    """)
k = np.int32(3)
BLOCK_SIZE = 16
height = np.int32(h_in.shape[0])
width = np.int32(h_in.shape[1])
means = np.random.randint(low=0,high=256, size=(k,3),dtype = np.uint8)

func = mod.get_function("rgb2grey")
cen = mod.get_global("centroids")[0]

for i in range(100):
    cuda.memcpy_htod(cen, means)
    func(d_in, d_out, height, width,k,
         grid=((height-1)/BLOCK_SIZE+1,(width-1)/BLOCK_SIZE+1,1),
              block=(BLOCK_SIZE,BLOCK_SIZE,1))
    cuda.memcpy_dtoh(h_out, d_out)
    #new_means = np.zeros(size=means.shape, dtype=np.float)
    for j in range(k):
        loc = np.array(np.where(h_out==j)).T
        means[j,:] = np.mean(h_in[loc[:,0],loc[:,1],:],axis=0)
                
    
    
im_grey = Image.fromarray(h_out,'L')
plt.imshow(im_grey, cmap='gray')
#im_grey.save('lenna_grey.png')
#im_grey.show()