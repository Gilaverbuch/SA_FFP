import matplotlib.pyplot as plt
import numpy as np 
from math import pi
# import scipy
# from scipy import signal
# from scipy import interpolate
# from scikits.umfpack import spsolve, splu
import sys
from netCDF4 import Dataset, date2num, num2date
import os
import scipy.linalg.lapack as la
import numba
from numba import jit


#--------------------------------------------------------------------------------------------------
#--------------------------------------------------------------------------------------------------
def coefficients_Ab_acoustic(layers,kr,Vp,Vs,lamda,mu,omega,z,dz,Force,BCtop,BCbottom,rho,earth_interface, A, alpha_1D, Kmp):
    

    c_s=np.zeros([layers*4,4],dtype=complex)
    c1_s=np.zeros([layers*4,4],dtype=complex)

    c_f=np.zeros([layers*2,2],dtype=complex)
    c1_f=np.zeros([layers*2,2],dtype=complex)



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





    mat_size=layers*2
    C=np.zeros([mat_size, mat_size] , dtype=complex)

    n, kl, ku = len(Force), 2, 2
    A = np.zeros((2*kl+ku+1,n),dtype=complex)

    # A[kl + ku + 1 + ii - jj - 1, jj - 1] = C_Ab[ii-1, jj-1]
    # |
    # A[kl + ku + 1 + (posx + 1) - (posy + 1) - 1, (posy + 1) - 1] = C_Ab[posx, posy]
    # |
    # A[kl + ku + (posx) -(posy), (posy)] = C_Ab[posx, posy]

    #SETTING TOP BOUNDARY CONDITIONS FOR ELASTIC LAYER
    if BCtop=='free':
        A[kl + ku , 0] =C[0,0]=rho[0]*(omega**2)
        A[kl + ku -1, 1] =C[0,1]=rho[0]*(omega**2)*np.exp(alpha_1D[0]*dz)
    elif BCtop=='rigid':
        A[kl + ku , 0] =C[0,0]=alpha_1D[0]*1
        A[kl + ku -1, 1] =C[0,1]=-alpha_1D[0]*np.exp(alpha_1D[0]*dz)
    elif BCtop=='radiation':
        A[kl + ku , 0] =C[0,0]=2
        A[kl + ku -1, 1] =C[0,1]=0
    else:
        print ('wrong boundary conditions')
        sys.exit()




    for i in range(0,(layers-1)*2,2):

        A[kl + ku + 1, i] =C[i+1,i]=c_f[i,0]
        A[kl + ku , i+1] =C[i+1,i+1]=c_f[i,1]
        A[kl + ku + 2, i] =C[i+2,i]=c_f[i+1,0]
        A[kl + ku + 1, i+1] =C[i+2,i+1]=c_f[i+1,1]

        A[kl + ku -1, i+2] =C[i+1,i+2]=-c1_f[i+2,0]
        A[kl + ku -2, i+3] =C[i+1,i+1+2]=-c1_f[i+2,1]
        A[kl + ku , i+2] =C[i+2,i+2]=-c1_f[i+3,0]
        A[kl + ku -1, i+3] =C[i+2,i+1+2]=-c1_f[i+3,1]




    #SETTING BOTTOM BOUNDARY CONDITIONS FOR FLUID LAYER

    if BCbottom=='free':
        A[kl + ku + 1, mat_size-2] =C[mat_size-1,mat_size-2]=rho[layers-1]*(omega**2)*np.exp(alpha_1D[layers-1]*dz)
        A[kl + ku , mat_size-1] =C[mat_size-1,mat_size-1]=rho[layers-1]*(omega**2)
    elif BCbottom=='rigid':
        A[kl + ku + 1, mat_size-2] =C[mat_size-1,mat_size-2]=np.exp(alpha_1D[layers-1]*dz)
        A[kl + ku , mat_size-1] =C[mat_size-1,mat_size-1]=-1
    elif BCbottom=='radiation':
        A[kl + ku + 1, mat_size-2] =C[mat_size-1,mat_size-2]=0
        A[kl + ku , mat_size-1] =C[mat_size-1,mat_size-1]=2
    else:
        print ('wrong boundary conditions')
        sys.exit()

    return A,Force,alpha_1D #,beta,C1,C2,C3,C4


#--------------------------------------------------------------------------------------------------


def force_vec_acoustic(Force_1D, layers,S_depth,S_medium,S_type,z,kr,omega,dz,Vp,Vs,lamda,mu,rho,dK,earth_interface,Earth_depth,Ocean_depth,Atm_depth):

    #the force for each interface should be according to F_stress_(n+1)-F_stress_n
    #and F_disp_(n+1)-F_disp_n
    # mat_size=layers*2 
    # Force=np.zeros(mat_size,dtype=complex)


    if S_medium=='ocean':
        S_layer=int((Ocean_depth - S_depth)/dz)
        S_depth=Ocean_depth - S_depth
        pos=(S_layer+1)*2 -1


    elif S_medium=='atm':
        S_depth=Atm_depth-S_depth
        S_layer=int((Ocean_depth + Atm_depth - S_depth)/dz)
        S_depth=Ocean_depth + Atm_depth - S_depth
        pos=(S_layer +1)*2 -1

        # S_depth=Ocean_depth+S_depth
        # S_layer=int((S_depth)/dz)
        # pos=S_layer*2

    else:
        print ('wrong parameters for source!')
        sys.exit()

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