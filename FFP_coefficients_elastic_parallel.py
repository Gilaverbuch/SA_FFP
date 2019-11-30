import matplotlib.pyplot as plt
import numpy as np 
from math import pi
import sys
from netCDF4 import Dataset, date2num, num2date
import os
import scipy.linalg.lapack as la
import numba
from numba import jit


#--------------------------------------------------------------------------------------------------
#--------------------------------------------------------------------------------------------------
# @jit(nopython=True,fastmath=True)
@jit('i4, c8, f4[:], f4[:], c8[:], c8[:], f4, i4[:], i4, i4, i4, f4[:], i4, c8[:,:], c8[:], c8[:], c8[:],'
        'c8[:], c8[:], c8[:], c8[:], c8[:], c8[:], i4, i4, i4, i4', nopython=True,fastmath=True)
def coefficients_Ab_elastic_parallel(layers,kr,Vp,Vs,lamda,mu,omega,z,dz,BCtop,BCbottom,rho,earth_interface,
                            A,Force_1D,alpha_1D,beta_1D,C1_1D,C2_1D,C3_1D,C4_1D, Kmp, Kms, n, kl, ku, mat_size):





    c_s=np.zeros((layers*4,4),dtype=np.complex64)
    c1_s=np.zeros((layers*4,4),dtype=np.complex64)

    


    #elastic layer coeff.
    for i in range(0,earth_interface):

        Kpz=np.sqrt(Kmp[i]**2 - kr**2)
        Ksz=np.sqrt(Kms[i]**2 - kr**2)

        alpha_1D[i]=1j*Kpz
        beta_1D[i]=1j*Ksz
        C1_1D[i]=lamda[i]*alpha_1D[i]**2 - lamda[i]*kr**2 + 2*mu[i]*alpha_1D[i]**2
        C2_1D[i]=2*mu[i]*kr*beta_1D[i]
        C3_1D[i]=-2*kr*alpha_1D[i]
        C4_1D[i]=kr**2 + beta_1D[i]**2


  
        #bottom interface
        c_s[i*4,0]=alpha_1D[i]*np.exp(alpha_1D[i]*dz)
        c_s[i*4,1]=kr*np.exp(beta_1D[i]*dz)
        c_s[i*4,2]=-alpha_1D[i]
        c_s[i*4,3]=kr

        c_s[i*4+1,0]=-kr*np.exp(alpha_1D[i]*dz)
        c_s[i*4+1,1]=-beta_1D[i]*np.exp(beta_1D[i]*dz)
        c_s[i*4+1,2]=-kr
        c_s[i*4+1,3]=beta_1D[i]

        c_s[i*4+2,0]=C1_1D[i]*np.exp(alpha_1D[i]*dz)
        c_s[i*4+2,1]=C2_1D[i]*np.exp(beta_1D[i]*dz)
        c_s[i*4+2,2]=C1_1D[i]
        c_s[i*4+2,3]=-C2_1D[i]

        c_s[i*4+3,0]=mu[i]*C3_1D[i]*np.exp(alpha_1D[i]*dz)
        c_s[i*4+3,1]=-mu[i]*C4_1D[i]*np.exp(beta_1D[i]*dz)
        c_s[i*4+3,2]=-mu[i]*C3_1D[i]
        c_s[i*4+3,3]=-mu[i]*C4_1D[i]



                
        #top interface
        c1_s[i*4,0]=alpha_1D[i]
        c1_s[i*4,1]=kr
        c1_s[i*4,2]=-alpha_1D[i]*np.exp(alpha_1D[i]*dz)
        c1_s[i*4,3]=kr*np.exp(beta_1D[i]*dz)

        c1_s[i*4+1,0]=-kr
        c1_s[i*4+1,1]=-beta_1D[i]
        c1_s[i*4+1,2]=-kr*np.exp(alpha_1D[i]*dz)
        c1_s[i*4+1,3]=beta_1D[i]*np.exp(beta_1D[i]*dz)

        c1_s[i*4+2,0]=C1_1D[i]
        c1_s[i*4+2,1]=C2_1D[i]
        c1_s[i*4+2,2]=C1_1D[i]*np.exp(alpha_1D[i]*dz)
        c1_s[i*4+2,3]=-C2_1D[i]*np.exp(beta_1D[i]*dz)

        c1_s[i*4+3,0]=mu[i]*C3_1D[i]
        c1_s[i*4+3,1]=-mu[i]*C4_1D[i]
        c1_s[i*4+3,2]=-mu[i]*C3_1D[i]*np.exp(alpha_1D[i]*dz)
        c1_s[i*4+3,3]=-mu[i]*C4_1D[i]*np.exp(beta_1D[i]*dz)




    # mat_size=earth_interface*4 
    # C=np.zeros([mat_size, mat_size] , dtype=complex)

    # n, kl, ku = len(Force), 5, 5
    # A = np.zeros((2*kl+ku+1,n),dtype=complex)

    # A[kl + ku + 1 + ii - jj - 1, jj - 1] = C_Ab[ii-1, jj-1]
    # |
    # A[kl + ku + 1 + (posx + 1) - (posy + 1) - 1, (posy + 1) - 1] = C_Ab[posx, posy]
    # |
    # A[kl + ku + (posx) -(posy), (posy)] = C_Ab[posx, posy]


    #SETTING TOP BOUNDARY CONDITIONS FOR ELASTIC LAYER
    if BCtop==1:

        #Vetical displacement w
        A[kl + ku   , 0] =alpha_1D[0]
        A[kl + ku - 1,1] =kr
        A[kl + ku -2, 2] =-alpha_1D[0]*np.exp(alpha_1D[0]*dz)
        A[kl + ku -3, 3] =kr*np.exp(beta_1D[0]*dz)

        #Horizontal displacement u
        A[kl + ku +1 , 0] =-kr
        A[kl + ku     ,1] =-beta_1D[0]
        A[kl + ku -1 , 2] =-kr*np.exp(alpha_1D[0]*dz)
        A[kl + ku -2  ,3] =beta_1D[0]*np.exp(beta_1D[0]*dz)

    elif BCtop==2:
        #Vetical stress sigma_zz
        A[kl + ku   , 0] =C1_1D[0]
        A[kl + ku - 1,1] =C2_1D[0]
        A[kl + ku -2, 2] =C1_1D[0]*np.exp(alpha_1D[0]*dz)
        A[kl + ku -3, 3] =-C2_1D[0]*np.exp(beta_1D[0]*dz)

        #Horizontal stress sigma_rz
        A[kl + ku +1 , 0] =C3_1D[0]
        A[kl + ku     ,1] =-C4_1D[0]
        A[kl + ku -1 , 2] =-C3_1D[0]*np.exp(alpha_1D[0]*dz)
        A[kl + ku -2  ,3] =-C4_1D[0]*np.exp(beta_1D[0]*dz)

    elif BCtop==3:


        #Phi
        A[kl + ku   , 0] =2
        A[kl + ku - 1,1] =0
        A[kl + ku -2, 2] =0
        A[kl + ku -3, 3] =0

        #Psi
        A[kl + ku +1 , 0] =0
        A[kl + ku     ,1] =2
        A[kl + ku -1 , 2] =0
        A[kl + ku -2  ,3] =0



    #create C matrix with 2 for loops. 1st for the solid and 2nd for the fluid.
    for i in range(0,(earth_interface-1)*4,4):
        for j in range(0,4):

            A[kl + ku + j+2, i] =c_s[i+j,0]
            A[kl + ku + j+1, i+1] =c_s[i+j,1]
            A[kl + ku + j, i+2] =c_s[i+j,2]
            A[kl + ku + j-1, i+3] =c_s[i+j,3]

            A[kl + ku + j-2, i+4] =-c1_s[i+4+j,0]
            A[kl + ku + j-3, i+5] =-c1_s[i+4+j,1]
            A[kl + ku + j-4, i+6] =-c1_s[i+4+j,2]
            A[kl + ku + j-5, i+7] =-c1_s[i+4+j,3]




    #SETTING BOTTOM BOUNDARY CONDITIONS FOR FLUID LAYER

    if BCbottom==1:

        #Vetical stress sigma_zz
        A[kl + ku + 3, mat_size-4] = C1_1D[layers-1]*np.exp(alpha_1D[layers-1]*dz)
        A[kl + ku + 2, mat_size-3] = C2_1D[layers-1]*np.exp(beta_1D[layers-1]*dz)
        A[kl + ku + 1, mat_size-2] = C1_1D[layers-1]
        A[kl + ku    , mat_size-1] = -C2_1D[layers-1]

        #Horizontal stress sigma_rz
        A[kl + ku + 2, mat_size-4] = C3_1D[layers-1]*np.exp(alpha_1D[layers-1]*dz)
        A[kl + ku + 1, mat_size-3] = -C4_1D[layers-1]*np.exp(beta_1D[layers-1]*dz)
        A[kl + ku    , mat_size-2] = -C3_1D[layers-1]
        A[kl + ku - 1, mat_size-1] = -C4_1D[layers-1]

    elif BCbottom==2:
        #Vetical displacement w
        A[kl + ku + 3, mat_size-4] = alpha_1D[layers-1]*np.exp(alpha_1D[layers-1]*dz)
        A[kl + ku + 2, mat_size-3] = kr*np.exp(beta_1D[layers-1]*dz)
        A[kl + ku + 1, mat_size-2] = -alpha_1D[layers-1]
        A[kl + ku    , mat_size-1] = kr

        #Horizontal displacement u
        A[kl + ku + 2, mat_size-4] = -kr*np.exp(alpha_1D[layers-1]*dz)
        A[kl + ku + 1, mat_size-3] = -beta_1D[layers-1]*np.exp(beta_1D[layers-1]*dz)
        A[kl + ku    , mat_size-2] = -kr
        A[kl + ku - 1, mat_size-1]  = beta_1D[layers-1]

    elif BCbottom==3:
        #phi
        A[kl + ku + 3, mat_size-4] = 0
        A[kl + ku + 2, mat_size-3] = 0
        A[kl + ku + 1, mat_size-2] = 2
        A[kl + ku    , mat_size-1] = 0

        #Psi
        A[kl + ku + 2, mat_size-4] = 0
        A[kl + ku + 1, mat_size-3] = 0
        A[kl + ku    , mat_size-2] = 0
        A[kl + ku - 1, mat_size-1] = 2




    # A[kl + ku + (posx) -(posy), (posy)] = C_Ab[posx, posy]

    #############################
    ##SCALING
    #############################
    # for i in range(0,mat_size-1):
    #     scaler=max(np.abs(C[i,:]))
    #     C[i,:]=C[i,:]/scaler
    #     Force[i]=Force[i]/scaler

    # return A,Force_1D,alpha_1D,beta_1D,C1_1D,C2_1D,C3_1D,C4_1D

# @jit(nopython=True,fastmath=True)
@jit('c8[:], i4, i4, i4, i4, i4[:], c8, f4, i4, f4[:], f4[:], c8[:], c8[:], f4[:] , f4, i4, i4, i4, i4, f4[:], f4[:]', nopython=True,fastmath=True)
def force_vec_elastic_parallel(Force_1D, layers,S_depth,S_medium,S_type,z,kr,omega,dz,Vp,Vs,lamda,mu,rho,dK,earth_interface,
                                Earth_depth,Ocean_depth,Atm_depth,delta_Kp,delta_Ks):

    #need to fix the force for each wavenumber according to the integral equations for the forcing
    #terms for the displacement and the stress.
    #the force for each interface should be according to F_stress_(n+1)-F_stress_n
    #and F_disp_(n+1)-F_disp_n



    S_layer=earth_interface - int(S_depth/dz)-1
    S_depth=(earth_interface)*dz - S_depth
    pos=(S_layer-1)*4+2 
    Kmp=float(omega)/Vp[S_layer]  + 1j*delta_Kp[S_layer]
    Kms=float(omega)/Vs[S_layer]  + 1j*delta_Ks[S_layer]
    Kpz=np.sqrt(Kmp**2-kr**2)
    Ksz=np.sqrt(Kms**2-kr**2)
    alpha=1j*Kpz
    beta=1j*Ksz
    if S_type==1:
        #Forcing terma in the following order:w, u, sigma_zz, sigma_rz
        C1=(lamda[S_layer]*alpha**2 - lamda[S_layer]*kr**2 +2*mu[S_layer]*alpha**2)
        z_bottom=np.abs(z[S_layer]-S_depth)
        Force_1D[pos-4]=-(1/(4*pi))*np.exp(alpha*z_bottom)
        Force_1D[pos-3]=-(kr/(4*pi*alpha))*np.exp(alpha*z_bottom) 
        Force_1D[pos-2]=(1/(4*pi*alpha))*C1*np.exp(alpha*z_bottom)
        Force_1D[pos-1]=mu[S_layer]*(kr/(2*pi))*np.exp(alpha*z_bottom)

        z_top=np.abs(z[S_layer+1]-S_depth)
        Force_1D[pos]=-(1/(4*pi))*np.exp(alpha*z_top) 
        Force_1D[pos+1]=(kr/(4*pi*alpha))*np.exp(alpha*z_top) 
        Force_1D[pos+2]=-(1/(4*pi*alpha))*C1*np.exp(alpha*z_top)
        Force_1D[pos+3]=mu[S_layer]*(kr/(2*pi))*np.exp(alpha*z_top)




    elif S_type==2:
        C1=(lamda[S_layer]*alpha**2 - lamda[S_layer]*kr**2 +2*mu[S_layer]*alpha**2)
        z_bottom=np.abs(z[S_layer]-S_depth)

        Force_1D[pos-4]=-(1/(4*pi))*(alpha*np.exp(alpha*z_bottom) +  (kr**2)*(1/beta)*np.exp(beta*z_bottom))
        Force_1D[pos-3]=-(kr/(4*pi))*(np.exp(alpha*z_bottom) + np.exp(beta*z_bottom)) 
        Force_1D[pos-2]=(1/(4*pi))*(C1*np.exp(alpha*z_bottom) + 2*mu[S_layer]*(kr**2)*np.exp(beta*z_bottom))
        Force_1D[pos-1]=(mu[S_layer]/(4*pi))*(2*kr*alpha*np.exp(alpha*z_bottom) + (beta**2 + kr**3)*(1/beta)*np.exp(beta*z_bottom))

        z_top=np.abs(z[S_layer+1]-S_depth)
        Force_1D[pos]=(1/(4*pi))*(alpha*np.exp(alpha*z_top) +  (kr**2)*(1/beta)*np.exp(beta*z_top))
        Force_1D[pos+1]=-(kr/(4*pi))*(np.exp(alpha*z_top) + np.exp(beta*z_top)) 
        Force_1D[pos+2]=(1/(4*pi))*(C1*np.exp(alpha*z_top) + 2*mu[S_layer]*(kr**2)*np.exp(beta*z_top))
        Force_1D[pos+3]=-(mu[S_layer]/(4*pi))*(2*kr*alpha*np.exp(alpha*z_top) + (beta**2 + kr**3)*(1/beta)*np.exp(beta*z_top))




    return Force_1D/(rho[S_layer+1]*omega**2)

#--------------------------------------------------------------------------------------------------
#--------------------------------------------------------------------------------------------------