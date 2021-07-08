import matplotlib.pyplot as plt
import numpy as np 
from math import pi
import sys
# from netCDF4 import Dataset, date2num, num2date
import os
import scipy.linalg.lapack as la
import numba
from numba import jit


#--------------------------------------------------------------------------------------------------
#--------------------------------------------------------------------------------------------------

# @jit(nopython=True)
@jit('i4,c8,f4[:],f4[:],c8[:],c8[:],f4,i4[:],i4,c8[:],i4,i4,f4[:],i4, c8[:,:], c8[:], f4[:], i4, i4, i4, i4', nopython=True,fastmath=True)
def coefficients_Ab_acoustic_parallel(layers,kr,Vp,Vs,lamda,mu,omega,z,dz,Force_1D,BCtop,BCbottom,rho,earth_interface, A, 
                                        alpha_1D, Kmp, n, kl, ku, mat_size):
    

    c_s=np.zeros((layers*4,4),dtype=np.complex128)
    c1_s=np.zeros((layers*4,4),dtype=np.complex128)

    c_f=np.zeros((layers*2,2),dtype=np.complex128)
    c1_f=np.zeros((layers*2,2),dtype=np.complex128)



    #fluid layers coeff.
    for i in range(0,layers):

        Kpz=np.sqrt(Kmp[i]**2 - kr**2)
        alpha_1D[i]=1j*Kpz


        #bottom interface
        c_f[i*2,0]=alpha_1D[i]*np.exp(alpha_1D[i]*dz)
        c_f[i*2,1]=-alpha_1D[i]*np.exp(alpha_1D[i]*0)
        # c_f[i*2+1,0]=rho[i]*(omega**2)*np.exp(alpha_1D[i]*dz)
        # c_f[i*2+1,1]=rho[i]*(omega**2)*np.exp(alpha_1D[i]*0)
        c_f[i*2+1,0]=-rho[i]*(omega**2)*np.exp(alpha_1D[i]*dz)
        c_f[i*2+1,1]=-rho[i]*(omega**2)*np.exp(alpha_1D[i]*0)
                
        #top interface
        c1_f[i*2,0]=alpha_1D[i]*np.exp(alpha_1D[i]*0)
        c1_f[i*2,1]=-alpha_1D[i]*np.exp(alpha_1D[i]*dz)
        # c1_f[i*2+1,0]=rho[i]*(omega**2)*np.exp(alpha_1D[i]*0)
        # c1_f[i*2+1,1]=rho[i]*(omega**2)*np.exp(alpha_1D[i]*dz)
        c1_f[i*2+1,0]=-rho[i]*(omega**2)*np.exp(alpha_1D[i]*0)
        c1_f[i*2+1,1]=-rho[i]*(omega**2)*np.exp(alpha_1D[i]*dz)





    # mat_size=layers*2
    # C=np.zeros([mat_size, mat_size] , dtype=np.complex128)

    # n, kl, ku = len(Force), 2, 2
    # A = np.zeros((2*kl+ku+1,n),dtype=np.complex128)

    # A[kl + ku + 1 + ii - jj - 1, jj - 1] = C_Ab[ii-1, jj-1]
    # |
    # A[kl + ku + 1 + (posx + 1) - (posy + 1) - 1, (posy + 1) - 1] = C_Ab[posx, posy]
    # |
    # A[kl + ku + (posx) -(posy), (posy)] = C_Ab[posx, posy]

    #SETTING TOP BOUNDARY CONDITIONS FOR ELASTIC LAYER
    if BCtop==1:
        A[kl + ku , 0] =rho[0]*(omega**2)
        A[kl + ku -1, 1] =rho[0]*(omega**2)*np.exp(alpha_1D[0]*dz)
    elif BCtop==2:
        A[kl + ku , 0] =alpha_1D[0]*1
        A[kl + ku -1, 1] =-alpha_1D[0]*np.exp(alpha_1D[0]*dz)
    elif BCtop==3:
        A[kl + ku , 0] =2
        A[kl + ku -1, 1] =0





    for i in range(0,(layers-1)*2,2):

        A[kl + ku + 1, i] =c_f[i,0]
        A[kl + ku , i+1] =c_f[i,1]
        A[kl + ku + 2, i] =c_f[i+1,0]
        A[kl + ku + 1, i+1] =c_f[i+1,1]

        A[kl + ku -1, i+2] =-c1_f[i+2,0]
        A[kl + ku -2, i+3] =-c1_f[i+2,1]
        A[kl + ku , i+2] =-c1_f[i+3,0]
        A[kl + ku -1, i+3] =-c1_f[i+3,1]




    #SETTING BOTTOM BOUNDARY CONDITIONS FOR FLUID LAYER

    if BCbottom==1:
        A[kl + ku + 1, mat_size-2] =rho[layers-1]*(omega**2)*np.exp(alpha_1D[layers-1]*dz)
        A[kl + ku , mat_size-1] =rho[layers-1]*(omega**2)
    elif BCbottom==2:
        A[kl + ku + 1, mat_size-2] =np.exp(alpha_1D[layers-1]*dz)
        A[kl + ku , mat_size-1] =-1
    elif BCbottom==3:
        A[kl + ku + 1, mat_size-2] =0
        A[kl + ku , mat_size-1] =2


    return A,Force_1D,alpha_1D 


#--------------------------------------------------------------------------------------------------

# @jit(nopython=True)
@jit(' c8[:],i4,i4,i4,i4,i4[:],c8,f4,i4,f4[:],f4[:],c8[:],c8[:],f4[:],f4,i4,i4,i4,i4 ',nopython=True,fastmath=True)
def force_vec_acoustic_parallel(Force_1D, layers,S_depth,S_medium,S_type,z,kr,omega,dz,Vp,Vs,lamda,mu,rho,dK,earth_interface,Earth_depth,Ocean_depth,Atm_depth):

    #the force for each interface should be according to F_stress_(n+1)-F_stress_n
    #and F_disp_(n+1)-F_disp_n



    if S_medium==2:
        S_layer=int((Ocean_depth - S_depth)/dz)
        S_depth=Ocean_depth - S_depth
        pos=(S_layer+1)*2 -1


    elif S_medium==3:
        S_depth=Atm_depth-S_depth
        S_layer=int((Ocean_depth + Atm_depth - S_depth)/dz)
        S_depth=Ocean_depth + Atm_depth - S_depth
        pos=(S_layer +1)*2 -1


    Kmp=float(omega)/Vp[S_layer]
    Kpz=np.sqrt(Kmp**2-kr**2)
    alpha=1j*Kpz


    Force_1D[pos-2]=-1/(4*pi)*np.exp(alpha*np.abs(z[S_layer]-S_depth))
    Force_1D[pos-1]=-rho[S_layer+1]*(omega**2)/(4*pi*alpha)*np.exp(alpha*np.abs(z[S_layer]-S_depth))

    Force_1D[pos]=-1/(4*pi)*np.exp(alpha*np.abs(z[S_layer+1]-S_depth))
    Force_1D[pos+1]=rho[S_layer+1]*(omega**2)/(4*pi*alpha)*np.exp(alpha*np.abs(z[S_layer+1]-S_depth))


    return Force_1D/(rho[S_layer+1]*omega**2)

#--------------------------------------------------------------------------------------------------
#--------------------------------------------------------------------------------------------------