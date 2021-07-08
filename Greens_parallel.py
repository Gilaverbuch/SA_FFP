import matplotlib.pyplot as plt
import numpy as np 
from math import pi
import sys
# from netCDF4 import Dataset, date2num, num2date
import os
import scipy.linalg.lapack as la
import numba
from numba import jit, prange
import time


from FFP_coefficients_parallel import coefficients_Ab_parallel, force_vec_parallel, force_vec_parallel_linesource
from FFP_coefficients_acoustic_parallel import coefficients_Ab_acoustic_parallel, force_vec_acoustic_parallel
from FFP_coefficients_elastic_parallel import coefficients_Ab_elastic_parallel, force_vec_elastic_parallel

#--------------------------------------------------------------------------------------------------
#--------------------------------------------------------------------------------------------------
def get_greens_parallel(K,Vp,Vs,lamda,mu,layers,wavenumbers,omega,dz,S_medium,S_depth,S_type,z,BCtop,BCbottom,rho,kmin,kmax,dK,earth_interface,Earth_depth,Ocean_depth,Atm_depth,name,delta_Kp,delta_Ks):


    mat_size=earth_interface*4 + (layers-earth_interface-2)*2 
    n, kl, ku = mat_size, 5, 5

    G=np.zeros((layers,wavenumbers),dtype=np.complex64)
    Force=np.zeros((mat_size,wavenumbers),dtype=np.complex64)
    A_sol=np.zeros((mat_size,wavenumbers),dtype=np.complex64)
    alpha=np.zeros((layers,wavenumbers),dtype=np.complex64)
    beta=np.zeros((layers,wavenumbers),dtype=np.complex64)
    C1=np.zeros((layers,wavenumbers),dtype=np.complex64)
    C2=np.zeros((layers,wavenumbers),dtype=np.complex64)
    C3=np.zeros((layers,wavenumbers),dtype=np.complex64)
    C4=np.zeros((layers,wavenumbers),dtype=np.complex64)
    C_Ab=np.zeros((2*kl+ku+1,n,wavenumbers),dtype=np.complex64)

    # Kmp=np.zeros(layers,dtype=np.complex64)
    # Kms=np.zeros(layers,dtype=np.complex64)

    # Kmp=np.float32(omega)/Vp[:] + 1j*delta_Kp[:]
    # Kms[:earth_interface]=np.float32(omega)/Vs[:earth_interface] + 1j*delta_Ks[:earth_interface]

    Kmp=np.zeros(layers,dtype=np.float32)
    Kms=np.zeros(layers,dtype=np.float32)

    Kmp=np.float32(omega)/Vp[:] 
    Kms[:earth_interface]=np.float32(omega)/Vs[:earth_interface] 



    

    pos_i=min(range(len(K)), key=lambda i: abs(np.real(K[i]) - kmin))
    pos_f=min(range(len(K)), key=lambda i: abs(np.real(K[i]) - kmax))

    if S_medium=="earth":
        S_medium=1
    elif S_medium=="ocean":
        S_medium=2
    elif S_medium=="atm":
        S_medium=3

    if S_type=='source':
        S_type=1
    elif S_type=='force':
        S_type=2

    if BCtop=="free":
        BCtop=1
    elif BCtop=="rigid":
        BCtop=2
    elif BCtop=="radiation":
        BCtop=3

    if BCbottom=="free":
        BCbottom=1
    elif BCbottom=="rigid":
        BCbottom=2
    elif BCbottom=="radiation":
        BCbottom=3

    if name=="exact":
        name=1
    else:
        name=0

    S_medium=np.int32(S_medium)
    S_type=np.int32(S_type)
    BCtop=np.int32(BCtop)
    BCbottom=np.int32(BCbottom)


    set_C_F(K,Vp,Vs,lamda,mu,layers,wavenumbers,omega,dz,S_medium,S_depth,S_type, z,BCtop,BCbottom,rho,kmin,kmax,
                                     dK,earth_interface,Earth_depth,Ocean_depth,Atm_depth, C_Ab, Force, alpha, beta,
                                      pos_i, pos_f, C1,C2,C3,C4, n, kl, ku, Kmp, Kms, delta_Kp, delta_Ks)



    linear_solver(A_sol, C_Ab, Force, kl, ku, pos_i, pos_f)

    # a=np.argwhere(np.isinf(Force))
    # print('fORCE',a)



    construct_greens(K, A_sol, G, alpha, beta, C1, C2, earth_interface, layers, dz, pos_i, pos_f, name)

    # a=np.argwhere(np.isnan(G))
    # print('G',a)



    return G






# @jit('c8[:], f4[:], f4[:], c8[:], c8[:], i4, i4, f4, i4, i4, i4, i4, i4[:], i4,  i4, f4[:], f4 , f4, f4,'
#             'i4, i4, i4, i4, c8[:,:,:], c8[:,:], c8[:,:], c8[:,:], i4, i4, c8[:,:], c8[:,:], c8[:,:], c8[:,:], i4, i4, i4,' 
#             'c8[:], c8[:], f4[:], f4[:]' ,parallel=True, nopython=True, nogil=True, fastmath=True)

@jit('c8[:], f4[:], f4[:], c8[:], c8[:], i4, i4, f4, i4, i4, i4, i4, i4[:], i4,  i4, f4[:], f4 , f4, f4,'
            'i4, i4, i4, i4, c8[:,:,:], c8[:,:], c8[:,:], c8[:,:], i4, i4, c8[:,:], c8[:,:], c8[:,:], c8[:,:], i4, i4, i4,' 
            'f4[:], f4[:], f4[:], f4[:]' ,parallel=True, nopython=True, nogil=True, fastmath=True)
def set_C_F(K,Vp,Vs,lamda,mu,layers,wavenumbers,omega,dz,S_medium,S_depth,S_type,z,BCtop,BCbottom,rho,kmin,kmax,dK,
            earth_interface,Earth_depth,Ocean_depth,Atm_depth, C_Ab, Force, alpha, beta, pos_i, pos_f, C1,C2,C3,C4, n, kl, ku, 
            Kmp, Kms, delta_Kp, delta_Ks):


    for i in prange(pos_i, pos_f):
        ######################
        #creating force vector
        # ######################
        
        Force[:,i]=force_vec_parallel(Force[:,i],layers,S_depth,S_medium,S_type,z,K[i],omega,dz,Vp,Vs,lamda,mu,rho,dK
                                        ,earth_interface,Earth_depth,Ocean_depth,Atm_depth,delta_Kp, delta_Ks)

        # Force[:,i]=force_vec_parallel_linesource(Force[:,i],layers,S_depth,S_medium,S_type,z,K[i],omega,dz,Vp,Vs,lamda,mu,rho,dK
        #                                 ,earth_interface,Earth_depth,Ocean_depth,Atm_depth,delta_Kp, delta_Ks)


    

        coefficients_Ab_parallel(layers,K[i],Vp,Vs,lamda,mu,omega,z,dz,BCtop,BCbottom,rho,earth_interface,
                    C_Ab[:,:,i],Force[:,i],alpha[:,i],beta[:,i],C1[:,i],C2[:,i],C3[:,i],C4[:,i], Kmp, Kms, n, kl, ku)



@jit('c8[:], c8[:,:], c8[:,:], c8[:,:], c8[:,:], c8[:,:], c8[:,:], i4, i4, i4, i4, i4, i4',  parallel=True, nopython=True, nogil=True, fastmath=True)
def construct_greens(K, A_sol, G, alpha, beta, C1, C2, earth_interface, layers, dz, pos_i, pos_f, name):

    for i in prange(pos_i, pos_f):
        if name==1:
            for j in range(0,earth_interface):

                #Solving for vertical displacement
                G[j,i]=alpha[j,i]*A_sol[j*4,i] + K[i]*A_sol[j*4+1,i] - alpha[j,i]*A_sol[j*4+2,i]*np.exp(alpha[j,i]*dz) + K[i]*A_sol[j*4+3,i]*np.exp(beta[j,i]*dz)

        else:
            for j in range(0,earth_interface):

                G[j,i]=C1[j,i]*alpha[j,i]*A_sol[j*4,i] + C2[j,i]*A_sol[j*4+1,i] + C1[j,i]*alpha[j,i]*A_sol[j*4+2,i]*np.exp(alpha[j,i]*dz) - C2[j,i]*A_sol[j*4+3,i]*np.exp(beta[j,i]*dz)

                #only S
                # G[j,i]= K[i]*A_sol[j*4+1]  + K[i]*A_sol[j*4+3]*np.exp(beta[j,i]*dz)

                #only P
                # G[j,i]=C1[j,i]*alpha[j,i]*A_sol[j*4,i] + C1[j,i]*alpha[j,i]*A_sol[j*4+2,i]*np.exp(alpha[j,i]*dz) 

        last_pos=j*4+3   
        for j in range(earth_interface,layers-2):
            sol_pos=last_pos + 1 + (j - earth_interface)*2 
            G[j,i]=A_sol[sol_pos,i] + A_sol[sol_pos+1,i]*np.exp(alpha[j,i]*dz)

#--------------------------------------------------------------------------------------------------
#--------------------------------------------------------------------------------------------------


def get_greens_acoustic_parallel(K,Vp,Vs,lamda,mu,layers,wavenumbers,omega,dz,S_medium,S_depth,S_type,z,BCtop,BCbottom,rho,kmin,kmax,dK,earth_interface,Earth_depth,Ocean_depth,Atm_depth,name):
    

    mat_size=layers*2
    n, kl, ku = np.int32((mat_size, 2, 2))

    G_p=np.zeros((layers,wavenumbers),dtype=np.complex64)
    G_w=np.zeros((layers,wavenumbers),dtype=np.complex64)
    Force=np.zeros((mat_size,wavenumbers),dtype=np.complex64)
    alpha=np.zeros((layers,wavenumbers),dtype=np.complex64)
    C_Ab=np.zeros((2*kl+ku+1,n,wavenumbers),dtype=np.complex64)
    A_sol=np.zeros((mat_size,wavenumbers),dtype=np.complex64)


    Kmp=np.float32(omega)/Vp[:]

    pos_i=np.int32(min(range(len(K)), key=lambda i: abs(np.real(K[i]) - kmin)))
    pos_f=np.int32(min(range(len(K)), key=lambda i: abs(np.real(K[i]) - kmax)))



    if S_medium=="ocean":
        S_medium=2
    elif S_medium=="atm":
        S_medium=3

    if S_type=='source':
        S_type=1
    elif S_type=='force':
        S_type=2

    if BCtop=="free":
        BCtop=1
    elif BCtop=="rigid":
        BCtop=2
    elif BCtop=="radiation":
        BCtop=3

    if BCbottom=="free":
        BCbottom=1
    elif BCbottom=="rigid":
        BCbottom=2
    elif BCbottom=="radiation":
        BCbottom=3

    if name=="exact":
        name=1
    else:
        name=0

    S_medium=np.int32(S_medium)
    S_type=np.int32(S_type)
    BCtop=np.int32(BCtop)
    BCbottom=np.int32(BCbottom)



    set_C_F_acoustic(K,Vp,Vs,lamda,mu,layers,wavenumbers,omega,dz,S_medium,S_depth,S_type, z,BCtop,BCbottom,rho,kmin,kmax,
                                     dK,earth_interface,Earth_depth,Ocean_depth,Atm_depth, C_Ab, Force, alpha,
                                      pos_i, pos_f, n, kl, ku, mat_size, Kmp)



    linear_solver(A_sol, C_Ab, Force, kl, ku, pos_i, pos_f)


    construct_greens_acoustic(A_sol, G_p, G_w, alpha, layers, dz, pos_i, pos_f)


    return G_p,G_w





# @jit(parallel=True, nopython=True, nogil=True)
@jit('c8[:],f4[:],f4[:],c8[:],c8[:],i4,i4,f4,i4,i4,i4,i4,i4[:],i4,i4,f4[:],f4,f4,f4,i4,i4,i4,i4, c8[:,:,:], c8[:,:], c8[:,:], i4, i4, i4, i4, i4, i4, f4[:]'
                    ,parallel=True, nopython=True, nogil=True, fastmath=True)
def set_C_F_acoustic(K,Vp,Vs,lamda,mu,layers,wavenumbers,omega,dz,S_medium,S_depth,S_type,z,BCtop,BCbottom,rho,kmin,kmax,dK,
                    earth_interface,Earth_depth,Ocean_depth,Atm_depth, C_Ab, Force, alpha, pos_i, pos_f, n, kl, ku, mat_size, Kmp):


    for i in prange(pos_i, pos_f):
            # print (np.real(K[i]))

            ######################
            #creating force vector
            ######################
            Force[:,i]=force_vec_acoustic_parallel(Force[:,i],layers,S_depth,S_medium,S_type,z,K[i],omega,dz,Vp,Vs,
                                                lamda,mu,rho,dK,earth_interface,Earth_depth,Ocean_depth,Atm_depth)


            ########################################
            ##Solving with LAPACK

            C_Ab[:,:,i],Force[:,i],alpha[:,i]=coefficients_Ab_acoustic_parallel(layers,K[i],Vp,Vs,lamda,mu,omega,z,dz,Force[:,i],BCtop,BCbottom,rho,
                                                earth_interface, C_Ab[:,:,i], alpha[:,i], Kmp, n, kl, ku, mat_size)





@jit('c8[:,:], c8[:,:], c8[:,:], c8[:,:], i4, i4, i4, i4', parallel=True, nopython=True, nogil=True, fastmath=True)
def construct_greens_acoustic(A_sol, G_p, G_w, alpha, layers, dz, pos_i, pos_f):

    for i in prange(pos_i, pos_f):
        # lub, piv, A_sol[:,i], mmm = la.flapack.zgbsv(kl, ku, C_Ab[:,:,i], Force[:,i])



        for j in range(0,layers-1):
            sol_pos=j*2
            G_p[j,i]=A_sol[sol_pos,i] + A_sol[sol_pos+1,i]*np.exp(alpha[j,i]*dz)
            G_w[j,i]=alpha[j,i]*A_sol[sol_pos,i] - alpha[j,i]*A_sol[sol_pos+1,i]*np.exp(alpha[j,i]*dz)




#--------------------------------------------------------------------------------------------------
#--------------------------------------------------------------------------------------------------

def get_greens_elastic_parallel(K,Vp,Vs,lamda,mu,layers,wavenumbers,omega,dz,S_medium,S_depth,S_type,z,BCtop,BCbottom,rho,kmin,kmax,dK,
                                earth_interface,Earth_depth,Ocean_depth,Atm_depth,name,delta_Kp,delta_Ks):
    mat_size=earth_interface*4 
    n, kl, ku = mat_size, 5, 5

    G=np.zeros((layers,wavenumbers),dtype=np.complex64)
    Force=np.zeros((mat_size,wavenumbers),dtype=np.complex64)
    A_sol=np.zeros((mat_size,wavenumbers),dtype=np.complex64)
    alpha=np.zeros((layers,wavenumbers),dtype=np.complex64)
    beta=np.zeros((layers,wavenumbers),dtype=np.complex64)
    C1=np.zeros((layers,wavenumbers),dtype=np.complex64)
    C2=np.zeros((layers,wavenumbers),dtype=np.complex64)
    C3=np.zeros((layers,wavenumbers),dtype=np.complex64)
    C4=np.zeros((layers,wavenumbers),dtype=np.complex64)
    C_Ab=np.zeros((2*kl+ku+1,n,wavenumbers),dtype=np.complex64)

    Kmp=np.zeros(layers,dtype=np.complex64)
    Kms=np.zeros(layers,dtype=np.complex64)

    Kmp=float(omega)/Vp[:] + 1j*delta_Kp[:]
    Kms=float(omega)/Vs[:] + 1j*delta_Ks[:]

    pos_i=min(range(len(K)), key=lambda i: abs(np.real(K[i]) - kmin))
    pos_f=min(range(len(K)), key=lambda i: abs(np.real(K[i]) - kmax))

    if S_medium=="earth":
        S_medium=1
    elif S_medium=="ocean":
        S_medium=2
    elif S_medium=="atm":
        S_medium=3

    if S_type=='source':
        S_type=1
    elif S_type=='force':
        S_type=2

    if BCtop=="free":
        BCtop=1
    elif BCtop=="rigid":
        BCtop=2
    elif BCtop=="radiation":
        BCtop=3

    if BCbottom=="free":
        BCbottom=1
    elif BCbottom=="rigid":
        BCbottom=2
    elif BCbottom=="radiation":
        BCbottom=3

    if name=="exact":
        name=1
    else:
        name=0

    S_medium=np.int32(S_medium)
    S_type=np.int32(S_type)
    BCtop=np.int32(BCtop)
    BCbottom=np.int32(BCbottom)




    set_C_F_elastic(K,Vp,Vs,lamda,mu,layers,wavenumbers,omega,dz,S_medium,S_depth,S_type, z,BCtop,BCbottom,rho,kmin,kmax,
                                     dK,earth_interface,Earth_depth,Ocean_depth,Atm_depth, C_Ab, Force, alpha, beta,
                                      pos_i, pos_f, C1,C2,C3,C4, n, kl, ku, Kmp, Kms, mat_size,delta_Kp,delta_Ks)



    linear_solver(A_sol, C_Ab, Force, kl, ku, pos_i, pos_f)



    construct_greens_elastic(K, A_sol, G, alpha, beta, C1, C2, earth_interface, layers, dz, pos_i, pos_f, name)



    return G


@jit('c8[:], f4[:], f4[:], c8[:], c8[:], i4, i4, f4, i4, i4, i4, i4, i4[:], i4,  i4, f4[:], f4 , f4, f4,'
            'i4, i4, i4, i4, c8[:,:,:], c8[:,:], c8[:,:], c8[:,:], i4, i4, c8[:,:], c8[:,:], c8[:,:], c8[:,:], i4, i4, i4,' 
            'c8[:], c8[:], i4, f4[:], f4[:]' ,parallel=True, nopython=True, nogil=True, fastmath=True)
def set_C_F_elastic(K,Vp,Vs,lamda,mu,layers,wavenumbers,omega,dz,S_medium,S_depth,S_type,z,BCtop,BCbottom,rho,kmin,kmax,dK,
            earth_interface,Earth_depth,Ocean_depth,Atm_depth, C_Ab, Force, alpha, beta, pos_i, pos_f, C1,C2,C3,C4, n, kl, ku, 
            Kmp, Kms, mat_size,delta_Kp,delta_Ks):


    for i in prange(pos_i, pos_f):
        ######################
        #creating force vector
        ######################
        Force[:,i]=force_vec_elastic_parallel(Force[:,i],layers,S_depth,S_medium,S_type,z,K[i],omega,dz,Vp,Vs,lamda,mu,rho,dK,
                                                earth_interface,Earth_depth,Ocean_depth,Atm_depth,delta_Kp,delta_Ks)


        ########################################
        ##Solving with LAPACK

        coefficients_Ab_elastic_parallel(layers,K[i],Vp,Vs,lamda,mu,omega,z,dz,BCtop,BCbottom,rho,earth_interface,
                                C_Ab[:,:,i],Force[:,i],alpha[:,i],beta[:,i],C1[:,i],C2[:,i],C3[:,i],C4[:,i], 
                                Kmp, Kms, n, kl, ku, mat_size)



@jit('c8[:], c8[:,:], c8[:,:], c8[:,:], c8[:,:], c8[:,:], c8[:,:], i4, i4, i4, i4, i4, i4',  parallel=True, nopython=True, nogil=True, fastmath=True)
def construct_greens_elastic(K, A_sol, G, alpha, beta, C1, C2, earth_interface, layers, dz, pos_i, pos_f, name):

    for i in prange(pos_i, pos_f):
        if name==1:
            for j in range(0,earth_interface):

                #Solving for vertical displacement
                G[j,i]=alpha[j,i]*A_sol[j*4,i] + K[i]*A_sol[j*4+1,i] - alpha[j,i]*A_sol[j*4+2,i]*np.exp(alpha[j,i]*dz) + K[i]*A_sol[j*4+3,i]*np.exp(beta[j,i]*dz)

        else:
            for j in range(0,earth_interface):

                G[j,i]=C1[j,i]*alpha[j,i]*A_sol[j*4,i] + C2[j,i]*A_sol[j*4+1,i] + C1[j,i]*alpha[j,i]*A_sol[j*4+2,i]*np.exp(alpha[j,i]*dz) - C2[j,i]*A_sol[j*4+3,i]*np.exp(beta[j,i]*dz)

                #only S
                # G[j,i]= K[i]*A_sol[j*4+1]  + K[i]*A_sol[j*4+3]*np.exp(beta[j,i]*dz)

                #only P
                # G[j,i]=alpha[j,i]*A_sol[j*4]  - alpha[j,i]*A_sol[j*4+2]*np.exp(alpha[j,i]*dz)

        last_pos=j*4+3   
        for j in range(earth_interface,layers-2):
            sol_pos=last_pos + 1 + (j - earth_interface)*2 
            G[j,i]=A_sol[sol_pos,i] + A_sol[sol_pos+1,i]*np.exp(alpha[j,i]*dz)


#--------------------------------------------------------------------------------------------------
#--------------------------------------------------------------------------------------------------
def linear_solver(A_sol, C_Ab, Force, kl, ku, pos_i, pos_f):

    for i in range(pos_i, pos_f):
        lub, piv, A_sol[:,i], mmm = la.flapack.zgbsv(kl, ku, C_Ab[:,:,i], Force[:,i])




#--------------------------------------------------------------------------------------------------
#--------------------------------------------------------------------------------------------------






