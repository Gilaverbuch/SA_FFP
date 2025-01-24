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



def go_to_range(G,cylin_spred,r,dK,wavenumbers,K,layers,kmin,kmax,rho,omega,eps):


    l=np.linspace(0,wavenumbers-1,wavenumbers,dtype=np.int64)
    temp1=np.zeros(wavenumbers*2,dtype=np.complex64)
    temp=(dK)*np.exp((eps+1j*np.real(K[0]))*r[:])    
    P=np.zeros([layers,wavenumbers],dtype=np.complex64)
    cylin_spred=np.exp(-1j*pi/4)/np.sqrt(2*pi*r[:])
    aa=np.zeros(wavenumbers*2)
    pos1=0
    while np.real(K[pos1])<=kmin:
            pos1=pos1+1;                
                
    pos2=wavenumbers-1
    while np.real(K[pos2])>=kmax:
            pos2=pos2-1;
    wind = signal.windows.tukey(pos2-pos1)
    aa[pos1:pos2]=wind

    temp3=np.sqrt(K[:])*np.exp(1j*r[0]*dK*l[:])
    temp2=np.concatenate([temp3,temp3[:]*0])

    for i in range(0,layers):
        temp1=np.concatenate([G[i,:],(G[i,:])*0])
        # temp3=np.sqrt(K[:])*np.exp(1j*r[0]*dK*l[:])
        # temp2=np.concatenate([temp3,temp3[:]*0])
        pp=np.fft.ifft(temp1*temp2*aa)*wavenumbers*2*2*pi
        P[i,:]=rho[i]*(omega**2)*pp[:wavenumbers]*2*cylin_spred*temp


    return P,aa

def go_to_range_acoustic(G_p,G_w,cylin_spred,r,dK,wavenumbers,K,layers,kmin,kmax,rho,omega,eps):


    l=np.linspace(0,wavenumbers-1,wavenumbers,dtype=np.int32)
    temp1=np.zeros(wavenumbers*2,dtype=np.complex64)
    temp=(dK)*np.exp((eps+1j*np.real(K[0]))*r[:])    
    P=np.zeros([layers,wavenumbers],dtype=np.complex64)
    W=np.zeros([layers,wavenumbers],dtype=np.complex64)
    cylin_spred=np.exp(-1j*pi/4)/np.sqrt(2*pi*r[:])
    aa=np.zeros(wavenumbers*2)
    pos1=0
    while np.real(K[pos1])<=kmin:
            pos1=pos1+1;                
                
    pos2=wavenumbers-1
    while np.real(K[pos2])>=kmax:
            pos2=pos2-1;
    wind = signal.windows.tukey(pos2-pos1)
    aa[pos1:pos2]=wind

    temp3=np.sqrt(K[:])*np.exp(1j*r[0]*dK*l[:])
    temp2=np.concatenate([temp3,temp3[:]*0])


    for i in range(0,layers):
        temp1=np.concatenate([G_p[i,:],(G_p[i,::-1])*0])
        # temp3=np.sqrt(K[:])*np.exp(1j*r[0]*dK*l[:])
        # temp2=np.concatenate([temp3,temp3[:]*0])
        pp=np.fft.ifft(temp1*temp2*aa)*wavenumbers*2*2*pi
        P[i,:]=rho[i]*(omega**2)*pp[:wavenumbers]*2*cylin_spred*temp

    for i in range(0,layers):
        temp1=np.concatenate([G_w[i,:],(G_w[i,::-1])*0])
        # temp3=np.sqrt(K[:])*np.exp(1j*r[0]*dK*l[:])
        # temp2=np.concatenate([temp3,temp3[:]*0])
        pp=np.fft.ifft(temp1*temp2*aa)*wavenumbers*2*2*pi
        W[i,:]=pp[:wavenumbers]*2*cylin_spred*temp


    return P,W,aa
