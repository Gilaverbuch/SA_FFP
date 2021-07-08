import matplotlib.pyplot as plt
import numpy as np 
from math import pi
# import scipy
# from scipy import signal
# from scipy import interpolate
# from scikits.umfpack import spsolve, splu
import sys
# from netCDF4 import Dataset, date2num, num2date
import os
import scipy.linalg.lapack as la
import numba
from numba import jit


from FFP_coefficients import coefficients_Ab, force_vec
from FFP_coefficients_acoustic import coefficients_Ab_acoustic, force_vec_acoustic
from FFP_coefficients_elastic import coefficients_Ab_elastic, force_vec_elastic

def get_greens(K,Vp,Vs,lamda,mu,layers,wavenumbers,omega,dz,S_medium,S_depth,S_type,z,BCtop,BCbottom,rho,kmin,kmax,dK,earth_interface,Earth_depth,Ocean_depth,Atm_depth,name):


    mat_size=earth_interface*4 + (layers-earth_interface-2)*2 
    n, kl, ku = mat_size, 5, 5

    G=np.zeros((layers,wavenumbers),dtype=complex)
    Force=np.zeros((mat_size,wavenumbers),dtype=complex)
    A_sol=np.zeros((mat_size,wavenumbers),dtype=complex)
    alpha=np.zeros((layers,wavenumbers),dtype=complex)
    beta=np.zeros((layers,wavenumbers),dtype=complex)
    C1=np.zeros((layers,wavenumbers),dtype=complex)
    C2=np.zeros((layers,wavenumbers),dtype=complex)
    C3=np.zeros((layers,wavenumbers),dtype=complex)
    C4=np.zeros((layers,wavenumbers),dtype=complex)
    C_Ab=np.zeros((2*kl+ku+1,n,wavenumbers),dtype=complex)

    Kmp=float(omega)/Vp[:]
    Kms=0*Kmp
    Kms=float(omega)/Vs[:earth_interface]

    pos_i=min(range(len(K)), key=lambda i: abs(np.real(K[i]) - kmin))
    pos_f=min(range(len(K)), key=lambda i: abs(np.real(K[i]) - kmax))


    for i in range(pos_i, pos_f):
        
       
        # print (np.real(K[i]))

        ######################
        #creating force vector
        ######################
        Force[:,i]=force_vec(Force[:,i], layers,S_depth,S_medium,S_type,z,K[i],omega,dz,Vp,Vs,lamda,mu,rho,dK,earth_interface,Earth_depth,Ocean_depth,Atm_depth)

        # C_Ab[:,:,i],Force[:,i],alpha[:,i],beta[:,i],C1[:,i],C2[:,i],C3[:,i],C4[:,i]=coefficients_Ab(layers,K[i],Vp,Vs,lamda,mu,omega,z,dz,BCtop,BCbottom,rho,earth_interface,
        #                                                                                             C_Ab[:,:,i],Force[:,i],alpha[:,i],beta[:,i],C1[:,i],C2[:,i],C3[:,i],C4[:,i], 
        #                                                                                             Kmp, Kms, n, kl, ku)

        coefficients_Ab(layers,K[i],Vp,Vs,lamda,mu,omega,z,dz,BCtop,BCbottom,rho,earth_interface,C_Ab[:,:,i],
                        Force[:,i],alpha[:,i],beta[:,i],C1[:,i],C2[:,i],C3[:,i],C4[:,i], Kmp, Kms, n, kl, ku)

    for i in range(pos_i, pos_f):

        lub, piv, A_sol[:,i], mmm = la.flapack.zgbsv(kl, ku, C_Ab[:,:,i], Force[:,i])



        if name=="exact":
            for j in range(0,earth_interface):

                #Solving for vertical displacement
                G[j,i]=alpha[j,i]*A_sol[j*4,i] + K[i]*A_sol[j*4+1,i] - alpha[j,i]*A_sol[j*4+2,i]*np.exp(alpha[j,i]*dz) + K[i]*A_sol[j*4+3,i]*np.exp(beta[j,i]*dz)

        else:
            for j in range(0,earth_interface):

                G[j,i]=C1[j,i]*alpha[j,i]*A_sol[j*4,i] + C2[j,i]*A_sol[j*4+1,i] + C1[j,i]*alpha[j,i,i]*A_sol[j*4+2,i]*np.exp(alpha[j,i]*dz) - C2[j,i]*A_sol[j*4+3,i]*np.exp(beta[j,i]*dz)

                #only S
                # G[j,i]= K[i]*A_sol[j*4+1]  + K[i]*A_sol[j*4+3]*np.exp(beta[j,i]*dz)

                #only P
                # G[j,i]=alpha[j,i]*A_sol[j*4]  - alpha[j,i]*A_sol[j*4+2]*np.exp(alpha[j,i]*dz)

        last_pos=j*4+3   
        for j in range(earth_interface,layers-2):
            sol_pos=last_pos + 1 + (j - earth_interface)*2 
            G[j,i]=A_sol[sol_pos,i] + A_sol[sol_pos+1,i]*np.exp(alpha[j,i]*dz)

    return G


#--------------------------------------------------------------------------------------------------
#--------------------------------------------------------------------------------------------------


def get_greens_acoustic(K,Vp,Vs,lamda,mu,layers,wavenumbers,omega,dz,S_medium,S_depth,S_type,z,BCtop,BCbottom,rho,kmin,kmax,dK,earth_interface,Earth_depth,Ocean_depth,Atm_depth,name):
    

    mat_size=layers*2
    n, kl, ku = mat_size, 2, 2

    G_p=np.zeros((layers,wavenumbers),dtype=complex)
    G_w=np.zeros((layers,wavenumbers),dtype=complex)
    Force=np.zeros((mat_size,wavenumbers),dtype=complex)
    alpha=np.zeros((layers,wavenumbers),dtype=complex)
    C_Ab=np.zeros((2*kl+ku+1,n,wavenumbers),dtype=complex)
    A_sol=np.zeros((mat_size,wavenumbers),dtype=complex)

    Kmp=float(omega)/Vp[:]

    pos_i=min(range(len(K)), key=lambda i: abs(np.real(K[i]) - kmin))
    pos_f=min(range(len(K)), key=lambda i: abs(np.real(K[i]) - kmax))


    for i in range(pos_i, pos_f):
        # print (np.real(K[i]))

        ######################
        #creating force vector
        ######################
        Force[:,i]=force_vec_acoustic(Force[:,i],layers,S_depth,S_medium,S_type,z,K[i],omega,dz,Vp,Vs,lamda,mu,rho,dK,earth_interface,Earth_depth,Ocean_depth,Atm_depth)


        ########################################
        ##Solving with LAPACK

        C_Ab[:,:,i],Force[:,i],alpha[:,i]=coefficients_Ab_acoustic(layers,K[i],Vp,Vs,lamda,mu,omega,z,dz,Force[:,i],BCtop,BCbottom,rho,earth_interface,
                                                                    C_Ab[:,:,i], alpha[:,i], Kmp)
        
    for i in range(pos_i, pos_f):
        lub, piv, A_sol[:,i], mmm = la.flapack.zgbsv(kl, ku, C_Ab[:,:,i], Force[:,i])



        for j in range(0,layers-1):
            sol_pos=j*2
            G_p[j,i]=A_sol[sol_pos,i] + A_sol[sol_pos+1,i]*np.exp(alpha[j,i]*dz)
            G_w[j,i]=alpha[j,i]*A_sol[sol_pos,i] - alpha[j,i]*A_sol[sol_pos+1,i]*np.exp(alpha[j,i]*dz)

    return G_p,G_w

#--------------------------------------------------------------------------------------------------
#--------------------------------------------------------------------------------------------------

def get_greens_elastic(K,Vp,Vs,lamda,mu,layers,wavenumbers,omega,dz,S_medium,S_depth,S_type,z,BCtop,BCbottom,rho,kmin,kmax,dK,earth_interface,Earth_depth,Ocean_depth,Atm_depth,name):
    mat_size=earth_interface*4 
    n, kl, ku = mat_size, 5, 5

    G=np.zeros((layers,wavenumbers),dtype=complex)
    Force=np.zeros((mat_size,wavenumbers),dtype=complex)
    A_sol=np.zeros((mat_size,wavenumbers),dtype=complex)
    alpha=np.zeros((layers,wavenumbers),dtype=complex)
    beta=np.zeros((layers,wavenumbers),dtype=complex)
    C1=np.zeros((layers,wavenumbers),dtype=complex)
    C2=np.zeros((layers,wavenumbers),dtype=complex)
    C3=np.zeros((layers,wavenumbers),dtype=complex)
    C4=np.zeros((layers,wavenumbers),dtype=complex)
    C_Ab=np.zeros((2*kl+ku+1,n,wavenumbers),dtype=complex)

    Kmp=float(omega)/Vp[:]
    Kms=0*Kmp
    Kms=float(omega)/Vs[:earth_interface]

    pos_i=min(range(len(K)), key=lambda i: abs(np.real(K[i]) - kmin))
    pos_f=min(range(len(K)), key=lambda i: abs(np.real(K[i]) - kmax))


    for i in range(pos_i, pos_f):
        # print (np.real(K[i]))

        ######################
        #creating force vector
        ######################
        Force[:,i]=force_vec_elastic(Force[:,i],layers,S_depth,S_medium,S_type,z,K[i],omega,dz,Vp,Vs,lamda,mu,rho,dK,earth_interface,Earth_depth,Ocean_depth,Atm_depth)


        ########################################
        ##Solving with LAPACK

        coefficients_Ab_elastic(layers,K[i],Vp,Vs,lamda,mu,omega,z,dz,BCtop,BCbottom,rho,earth_interface,
                                C_Ab[:,:,i],Force[:,i],alpha[:,i],beta[:,i],C1[:,i],C2[:,i],C3[:,i],C4[:,i], 
                                Kmp, Kms, n, kl, ku, mat_size)


    for i in range(pos_i, pos_f):
        lub, piv, A_sol[:,i], mmm = la.flapack.zgbsv(kl, ku, C_Ab[:,:,i], Force[:,i])




        if name=="exact":
            for j in range(0,earth_interface):

                #Solving for vertical displacement
                G[j,i]=alpha[j,i]*A_sol[j*4,i] + K[i]*A_sol[j*4+1,i] - alpha[j,i]*A_sol[j*4+2,i]*np.exp(alpha[j,i]*dz) + K[i]*A_sol[j*4+3,i]*np.exp(beta[j,i]*dz)

        else:
            for j in range(0,earth_interface):
                G[j,i]=alpha[j,i]*A_sol[j*4,i] + K[i]*A_sol[j*4+1,i] - alpha[j,i]*A_sol[j*4+2,i]*np.exp(alpha[j,i]*dz) + K[i]*A_sol[j*4+3,i]*np.exp(beta[j,i]*dz)

                # G[j,i]=C1[j]*alpha[j,i]*A_sol[j*4] + C2[j]*A_sol[j*4+1] + C1[j]*alpha[j,i]*A_sol[j*4+2]*np.exp(alpha[j,i]*dz) - C2[j]*A_sol[j*4+3]*np.exp(beta[j,i]*dz)

                #only S
                # G[j,i]= K[i]*A_sol[j*4+1]  + K[i]*A_sol[j*4+3]*np.exp(beta[j,i]*dz)

                #only P
                # G[j,i]=alpha[j,i]*A_sol[j*4]  - alpha[j,i]*A_sol[j*4+2]*np.exp(alpha[j,i]*dz)

    return G
