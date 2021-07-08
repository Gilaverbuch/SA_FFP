import matplotlib.pyplot as plt
import numpy as np 
import math
import cmath
from math import exp,pi,sin,cos,radians,sqrt,acos,degrees
import scipy
from scipy import signal
from scipy import interpolate
# from scikits.umfpack import spsolve, splu
import sys
# from netCDF4 import Dataset, date2num, num2date
import os
import scipy.linalg.lapack as la
import time
import numba
from numba import jit

def go_to_time(G,layers,wavenumbers,num_of_freq,freq,frequency1,time,df, dt, earth_interface, ocean_interface, Rec_alt, dz):
    
    levels=len(Rec_alt)
    l=np.linspace(0,num_of_freq-1,num_of_freq,dtype=np.int32)
    P_time=np.zeros([levels,wavenumbers,num_of_freq],dtype=np.float64)
    
    # pos_0= earth_interface + ocean_interface
    # pos= np.int64(pos_0) + np.int64(Rec_alt/dz)
    # print('pos',pos)


    #Ricker
    fc=frequency1
    t0=3
    R=(1-2*(pi*fc*(time-t0))**2)*np.exp(-(pi*fc*(time-t0))**2)
    R=np.concatenate([R,R*0])
    R_f=np.fft.fft(R)
    

    n=len(R)
    S=np.zeros(n,dtype=np.complex128)
    S=np.fft.fft(R)
    freq_S = np.fft.fftfreq(n, dt)
    
    
    temp3=np.exp(-1j*time[0]*df*l[:])
    temp2=np.concatenate([temp3,temp3[:]*0])
    for i in range(0,levels):
        temp_pos=i
        for j in range(0,wavenumbers):
            
            temp1=np.concatenate([G[temp_pos,j,:],(G[temp_pos,j,::-1])*0])
            
            pp=np.fft.ifft(temp1*temp2*S)*num_of_freq*2*2*pi
            P_time[i,j,:]=np.real(pp[num_of_freq:])
            P_time[i,j,:]=P_time[i,j,::-1]


    return P_time




