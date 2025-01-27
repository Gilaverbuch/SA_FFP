import matplotlib.pyplot as plt
import numpy as np 
from math import pi
from scipy import signal
import sys
import time

from input_ import Parameters 
from Greens import get_greens, get_greens_acoustic, get_greens_elastic
from Greens_parallel import get_greens_parallel, get_greens_acoustic_parallel, get_greens_elastic_parallel
from Go_to_range import go_to_range, go_to_range_acoustic
#--------------------------------------------------------------------------------------------------
class Exact_solver:
    """
    This class contains all the different exact solutions.
    Input is the type of solution 
    Variables:
        filename: input file with initial parameters

    """

    def __init__(self,solution):
        self.solution_type=solution
        print(self.solution_type)


    def solve(self):

        if self.solution_type=='acoustic':

            t0 = time.time()
            print('Exact solution. Homogeneous half-space with rigid BC Vs FFP with solid-fluid interface.')
            acoustic_solver_rigid()
            t1 = time.time()
            print (t1-t0, 'seconds')


            t0 = time.time()
            print('Exact solution. Homogeneous half-space with free BC Vs acoustic FFP.')
            acoustic_solver_free()
            t1 = time.time()
            print (t1-t0, 'seconds')

            t0 = time.time()
            print('Exact solution. N2')
            acoustic_solver_N2()
            t1 = time.time()
            print (t1-t0, 'seconds')

            t0 = time.time()
            print('Exact solution. Toy profile')
            acoustic_solver_toy()
            t1 = time.time()
            print (t1-t0, 'seconds')

        elif self.solution_type=='elastic':

            t0 = time.time()
            print('Exact solution. Homogeneous half-space with free BC Vs elastic FFP.')
            elastic_solver_free()
            t1 = time.time()
            print (t1-t0, 'seconds')
            
#--------------------------------------------------------------------------------------------------
#--------------------------------------------------------------------------------------------------

def elastic_solver_free():

    ###############################################
    #EXACT SOUTION ELASTIC
    ###############################################

    Initial_parameters=Parameters("./Exact-elastic/input-version-exact-elastic-1")
    Initial_parameters.calc_parameters()
    Initial_parameters.wave_num_and_range()
    Initial_parameters.velo_prof()
    # Initial_parameters.print_velocity_profile()

    print ("Get Green's")
    # A1=get_greens(Initial_parameters.K,Initial_parameters.Vp,Initial_parameters.Vs,Initial_parameters.lamda,Initial_parameters.mu,Initial_parameters.layers,
    #             Initial_parameters.wavenumbers,Initial_parameters.omega,Initial_parameters.dz,
    #                 Initial_parameters.S_medium,Initial_parameters.S_depth,Initial_parameters.S_type,
    #                 Initial_parameters.z,Initial_parameters.BCtop,Initial_parameters.BCbottom,Initial_parameters.rho
    #                 ,Initial_parameters.Kmin,Initial_parameters.Kmax,Initial_parameters.dK,Initial_parameters.earth_interface,
    #                 Initial_parameters.Earth_depth,Initial_parameters.Ocean_depth,Initial_parameters.Atm_depth,Initial_parameters.Fname)

    A1=get_greens_parallel(Initial_parameters.K,Initial_parameters.Vp,Initial_parameters.Vs,Initial_parameters.lamda,Initial_parameters.mu,Initial_parameters.layers,
                Initial_parameters.wavenumbers,Initial_parameters.omega,Initial_parameters.dz,
                    Initial_parameters.S_medium,Initial_parameters.S_depth,Initial_parameters.S_type,
                    Initial_parameters.z,Initial_parameters.BCtop,Initial_parameters.BCbottom,Initial_parameters.rho
                    ,Initial_parameters.Kmin,Initial_parameters.Kmax,Initial_parameters.dK,Initial_parameters.earth_interface,
                    Initial_parameters.Earth_depth,Initial_parameters.Ocean_depth,Initial_parameters.Atm_depth,Initial_parameters.Fname,
                    Initial_parameters.delta_Kp, Initial_parameters.delta_Ks, Initial_parameters.atm_atten_profile)


    print ("Go to range")
    P1,smooth_window=go_to_range(A1,Initial_parameters.cylin_spred,Initial_parameters.r,Initial_parameters.dK,
                Initial_parameters.wavenumbers,Initial_parameters.K,Initial_parameters.layers,Initial_parameters.Kmin,Initial_parameters.Kmax,
                Initial_parameters.rho,Initial_parameters.omega,Initial_parameters.eps)




    Initial_parameters=Parameters("./Exact-elastic/input-version-exact-elastic-2")
    Initial_parameters.calc_parameters()
    Initial_parameters.wave_num_and_range()
    Initial_parameters.velo_prof()
    # Initial_parameters.print_velocity_profile()

    print ("Get Green's")

    # A2=get_greens_elastic(Initial_parameters.K,Initial_parameters.Vp,Initial_parameters.Vs,Initial_parameters.lamda,Initial_parameters.mu,Initial_parameters.layers,
    #         Initial_parameters.wavenumbers,Initial_parameters.omega,Initial_parameters.dz,
    #             Initial_parameters.S_medium,Initial_parameters.S_depth,Initial_parameters.S_type,
    #             Initial_parameters.z,Initial_parameters.BCtop,Initial_parameters.BCbottom,Initial_parameters.rho
    #             ,Initial_parameters.Kmin,Initial_parameters.Kmax,Initial_parameters.dK,Initial_parameters.earth_interface,
    #             Initial_parameters.Earth_depth,Initial_parameters.Ocean_depth,Initial_parameters.Atm_depth,Initial_parameters.Fname)

    A2=get_greens_elastic_parallel(Initial_parameters.K,Initial_parameters.Vp,Initial_parameters.Vs,Initial_parameters.lamda,Initial_parameters.mu,Initial_parameters.layers,
            Initial_parameters.wavenumbers,Initial_parameters.omega,Initial_parameters.dz,
                Initial_parameters.S_medium,Initial_parameters.S_depth,Initial_parameters.S_type,
                Initial_parameters.z,Initial_parameters.BCtop,Initial_parameters.BCbottom,Initial_parameters.rho
                ,Initial_parameters.Kmin,Initial_parameters.Kmax,Initial_parameters.dK,Initial_parameters.earth_interface,
                Initial_parameters.Earth_depth,Initial_parameters.Ocean_depth,Initial_parameters.Atm_depth,Initial_parameters.Fname,
                Initial_parameters.delta_Kp, Initial_parameters.delta_Ks)

    print ("Go to range")

    P2,smooth_window=go_to_range(A2,Initial_parameters.cylin_spred,Initial_parameters.r,Initial_parameters.dK,
                Initial_parameters.wavenumbers,Initial_parameters.K,Initial_parameters.layers,Initial_parameters.Kmin,Initial_parameters.Kmax,
                Initial_parameters.rho,Initial_parameters.omega,Initial_parameters.eps)

    print ("Get exact Green's")
    A_exact,P_exact=get_exact_greens_elastic_free(Initial_parameters.K,Initial_parameters.Vp,Initial_parameters.Vs,Initial_parameters.layers, Initial_parameters.earth_interface,
                    Initial_parameters.wavenumbers,Initial_parameters.omega,Initial_parameters.dz,
                    Initial_parameters.S_medium,Initial_parameters.S_depth,Initial_parameters.z,
                    Initial_parameters.BCtop,Initial_parameters.BCbottom,Initial_parameters.rho,Initial_parameters.Kmin,
                    Initial_parameters.Kmax,Initial_parameters.eps,Initial_parameters.dK,Initial_parameters.r,
                    Initial_parameters.lamda[0],Initial_parameters.mu[0])



    pos=35
    plt.figure(figsize=(20,10))
    plt.plot(Initial_parameters.r/1000,20*np.log10(np.abs(P1[Initial_parameters.earth_interface-pos,:])/(4*pi)),'r',linewidth=5,label='FFP-SA')
    plt.plot(Initial_parameters.r/1000,20*np.log10(np.abs(P2[Initial_parameters.earth_interface-pos,:])/(4*pi)),'g',linewidth=3,label='FFP-elastic')
    plt.plot(Initial_parameters.r/1000,20*np.log10(np.abs(P_exact[pos,:])/(4*pi)),'b',linewidth=1,label='Exact')
    plt.xlim([0,100])
    plt.title('Elastic free surface')
    plt.legend()
    plt.show()

#--------------------------------------------------------------------------------------------------
#--------------------------------------------------------------------------------------------------

def acoustic_solver_toy():

    ###############################################
    #EXACT SOUTION toy PROFILE
    ###############################################
    Initial_parameters=Parameters("./Toy/input-version-exact-toy")
    Initial_parameters.calc_parameters()
    Initial_parameters.wave_num_and_range()
    Initial_parameters.velo_prof()
    # Initial_parameters.print_velocity_profile()

    # print ("Get greens serial")
    # t0 = time.time()
    # A_p,A_w=get_greens_acoustic(Initial_parameters.K, Initial_parameters.Vp, Initial_parameters.Vs, Initial_parameters.lamda,
    #                         Initial_parameters.mu, Initial_parameters.layers, Initial_parameters.wavenumbers, Initial_parameters.omega,
    #                         Initial_parameters.dz, Initial_parameters.S_medium, Initial_parameters.S_depth, Initial_parameters.S_type,
    #                         Initial_parameters.z, Initial_parameters.BCtop, Initial_parameters.BCbottom, Initial_parameters.rho,
    #                         Initial_parameters.Kmin, Initial_parameters.Kmax, Initial_parameters.dK, Initial_parameters.earth_interface,
    #                         Initial_parameters.Earth_depth, Initial_parameters.Ocean_depth, Initial_parameters.Atm_depth, Initial_parameters.Fname)
    # t1 = time.time()
    # print (t1-t0)

    print ("Get Green's")
    t0 = time.time()
    A_p,A_w=get_greens_acoustic_parallel(Initial_parameters.K, Initial_parameters.Vp, Initial_parameters.Vs, Initial_parameters.lamda,
                            Initial_parameters.mu, Initial_parameters.layers, Initial_parameters.wavenumbers, Initial_parameters.omega,
                            Initial_parameters.dz, Initial_parameters.S_medium, Initial_parameters.S_depth, Initial_parameters.S_type,
                            Initial_parameters.z, Initial_parameters.BCtop, Initial_parameters.BCbottom, Initial_parameters.rho,
                            Initial_parameters.Kmin, Initial_parameters.Kmax, Initial_parameters.dK, Initial_parameters.earth_interface,
                            Initial_parameters.Earth_depth, Initial_parameters.Ocean_depth, Initial_parameters.Atm_depth, Initial_parameters.Fname,
                            Initial_parameters.atm_atten_profile)
    t1 = time.time()
    print (t1-t0)


    print ("Go to range")
    P,W,smooth_window=go_to_range_acoustic(A_p,A_w,Initial_parameters.cylin_spred,Initial_parameters.r,Initial_parameters.dK,
                Initial_parameters.wavenumbers,Initial_parameters.K,Initial_parameters.layers,Initial_parameters.Kmin,Initial_parameters.Kmax,
                Initial_parameters.rho,Initial_parameters.omega,Initial_parameters.eps)


    parameters=np.loadtxt('./Toy/tloss_1d.effc.ll.nm')
    l=len(parameters)
    Jelle_tl=np.ndarray(l,dtype=np.float32)
    Jelle_range=np.ndarray(l,dtype=np.float32)
    for i in range(0,l):
        Jelle_range[i]=parameters[i,0]
        Jelle_tl[i]=parameters[i,1]

    pos=0
    plt.figure(figsize=(20,5))
    plt.plot(Initial_parameters.r/1000,20*np.log10(np.abs(P[pos,:])/(1)),'r',linewidth=2,label='FFP')
    plt.plot(Jelle_range,Jelle_tl,'b',linewidth=1,label='NM')
    plt.xlim([0,1000])
    plt.title('Acoustic toy profile')
    plt.legend()
    plt.show()


#--------------------------------------------------------------------------------------------------
#--------------------------------------------------------------------------------------------------

def acoustic_solver_N2():

    ###############################################
    #EXACT SOUTION N^2 PROFILE
    ###############################################
    Initial_parameters=Parameters("./Exact-N2/input-version-exact-N2")
    Initial_parameters.calc_parameters()
    Initial_parameters.wave_num_and_range()
    Initial_parameters.velo_prof()
    # Initial_parameters.print_velocity_profile()

    print ("Get Green's")
    # A_p,A_w=get_greens_acoustic(Initial_parameters.K, Initial_parameters.Vp, Initial_parameters.Vs, Initial_parameters.lamda,
    #                         Initial_parameters.mu, Initial_parameters.layers, Initial_parameters.wavenumbers, Initial_parameters.omega,
    #                         Initial_parameters.dz, Initial_parameters.S_medium, Initial_parameters.S_depth, Initial_parameters.S_type,
    #                         Initial_parameters.z, Initial_parameters.BCtop, Initial_parameters.BCbottom, Initial_parameters.rho,
    #                         Initial_parameters.Kmin, Initial_parameters.Kmax, Initial_parameters.dK, Initial_parameters.earth_interface,
    #                         Initial_parameters.Earth_depth, Initial_parameters.Ocean_depth, Initial_parameters.Atm_depth, Initial_parameters.Fname)

    A_p,A_w=get_greens_acoustic_parallel(Initial_parameters.K, Initial_parameters.Vp, Initial_parameters.Vs, Initial_parameters.lamda,
                            Initial_parameters.mu, Initial_parameters.layers, Initial_parameters.wavenumbers, Initial_parameters.omega,
                            Initial_parameters.dz, Initial_parameters.S_medium, Initial_parameters.S_depth, Initial_parameters.S_type,
                            Initial_parameters.z, Initial_parameters.BCtop, Initial_parameters.BCbottom, Initial_parameters.rho,
                            Initial_parameters.Kmin, Initial_parameters.Kmax, Initial_parameters.dK, Initial_parameters.earth_interface,
                            Initial_parameters.Earth_depth, Initial_parameters.Ocean_depth, Initial_parameters.Atm_depth, Initial_parameters.Fname,
                            Initial_parameters.atm_atten_profile)



    print ("Go to range")
    P,W,smooth_window=go_to_range_acoustic(A_p,A_w,Initial_parameters.cylin_spred,Initial_parameters.r,Initial_parameters.dK,
                Initial_parameters.wavenumbers,Initial_parameters.K,Initial_parameters.layers,Initial_parameters.Kmin,Initial_parameters.Kmax,
                Initial_parameters.rho,Initial_parameters.omega,Initial_parameters.eps)


    n2_tl=np.loadtxt('./Exact-N2/N2_TL.txt')
    n2_range=np.loadtxt('./Exact-N2/N2_range.txt')

    pos=0
    plt.figure(figsize=(20,10))
    plt.plot(Initial_parameters.r/1000,20*np.log10(np.abs(P[pos,:])/(1)),'r',linewidth=2,label='FFP')
    plt.plot(n2_range,n2_tl,'b',linewidth=1,label='Exact')
    plt.xlim([0,1000])
    plt.title('Acoustic N^2')
    plt.legend()
    plt.show()


#--------------------------------------------------------------------------------------------------
#--------------------------------------------------------------------------------------------------
def acoustic_solver_rigid():
    Initial_parameters=Parameters("./Exact-acoustic-rigid/input-version-exact-acoustic-rigid")
    Initial_parameters.calc_parameters()
    Initial_parameters.wave_num_and_range()
    Initial_parameters.velo_prof()
    # Initial_parameters.print_velocity_profile()

    print ("Get Green's")
    # A=get_greens(Initial_parameters.K,Initial_parameters.Vp,Initial_parameters.Vs,Initial_parameters.lamda,Initial_parameters.mu,Initial_parameters.layers,
    #             Initial_parameters.wavenumbers,Initial_parameters.omega,Initial_parameters.dz,
    #                 Initial_parameters.S_medium,Initial_parameters.S_depth,Initial_parameters.S_type,
    #                 Initial_parameters.z,Initial_parameters.BCtop,Initial_parameters.BCbottom,Initial_parameters.rho
    #                 ,Initial_parameters.Kmin,Initial_parameters.Kmax,Initial_parameters.dK,Initial_parameters.earth_interface,
    #                 Initial_parameters.Earth_depth,Initial_parameters.Ocean_depth,Initial_parameters.Atm_depth,Initial_parameters.Fname)

    A=get_greens_parallel(Initial_parameters.K,Initial_parameters.Vp,Initial_parameters.Vs,Initial_parameters.lamda,Initial_parameters.mu,Initial_parameters.layers,
                Initial_parameters.wavenumbers,Initial_parameters.omega,Initial_parameters.dz,
                    Initial_parameters.S_medium,Initial_parameters.S_depth,Initial_parameters.S_type,
                    Initial_parameters.z,Initial_parameters.BCtop,Initial_parameters.BCbottom,Initial_parameters.rho
                    ,Initial_parameters.Kmin,Initial_parameters.Kmax,Initial_parameters.dK,Initial_parameters.earth_interface,
                    Initial_parameters.Earth_depth,Initial_parameters.Ocean_depth,Initial_parameters.Atm_depth,Initial_parameters.Fname,
                    Initial_parameters.delta_Kp, Initial_parameters.delta_Ks,Initial_parameters.atm_atten_profile)


    print ("Go to range")
    P,smooth_window=go_to_range(A,Initial_parameters.cylin_spred,Initial_parameters.r,Initial_parameters.dK,
                    Initial_parameters.wavenumbers,Initial_parameters.K,Initial_parameters.layers,Initial_parameters.Kmin,Initial_parameters.Kmax,
                    Initial_parameters.rho,Initial_parameters.omega,Initial_parameters.eps)




    print ("Get exact Green's")
    A_exact,P_exact=get_exact_greens_acoustic_rigid(Initial_parameters.K,Initial_parameters.Vp,Initial_parameters.layers, Initial_parameters.earth_interface,
            Initial_parameters.wavenumbers,Initial_parameters.omega,Initial_parameters.dz,
            Initial_parameters.S_medium,Initial_parameters.S_depth,Initial_parameters.z,
            Initial_parameters.BCtop,Initial_parameters.BCbottom,Initial_parameters.rho,Initial_parameters.Kmin,
            Initial_parameters.Kmax,Initial_parameters.eps,Initial_parameters.dK,Initial_parameters.r)



    pos=30
    plt.figure(figsize=(20,10))
    plt.plot(Initial_parameters.r/1000,20*np.log10(np.abs(P[pos+Initial_parameters.earth_interface,:])/(4*pi)),'r',linewidth=2,label='FFP')
    plt.plot(Initial_parameters.r/1000,20*np.log10(np.abs(P_exact[pos,:])/(4*pi)),'b',linewidth=1,label='Exact')
    plt.xlim([0,20])
    plt.title('Acoustic rigid surface')
    plt.legend()
    plt.show()


#--------------------------------------------------------------------------------------------------
#--------------------------------------------------------------------------------------------------


def acoustic_solver_free():
    Initial_parameters=Parameters("./Exact-acoustic-free/input-version-exact-acoustic-free")
    Initial_parameters.calc_parameters()
    Initial_parameters.wave_num_and_range()
    Initial_parameters.velo_prof()
    # Initial_parameters.print_velocity_profile()

    print ("Get Green's")
    # A_p,A_w=get_greens_acoustic(Initial_parameters.K, Initial_parameters.Vp, Initial_parameters.Vs, Initial_parameters.lamda,
    #                         Initial_parameters.mu, Initial_parameters.layers, Initial_parameters.wavenumbers, Initial_parameters.omega,
    #                         Initial_parameters.dz, Initial_parameters.S_medium, Initial_parameters.S_depth, Initial_parameters.S_type,
    #                         Initial_parameters.z, Initial_parameters.BCtop, Initial_parameters.BCbottom, Initial_parameters.rho,
    #                         Initial_parameters.Kmin, Initial_parameters.Kmax, Initial_parameters.dK, Initial_parameters.earth_interface,
    #                         Initial_parameters.Earth_depth, Initial_parameters.Ocean_depth, Initial_parameters.Atm_depth, Initial_parameters.Fname)

    A_p,A_w=get_greens_acoustic_parallel(Initial_parameters.K, Initial_parameters.Vp, Initial_parameters.Vs, Initial_parameters.lamda,
                            Initial_parameters.mu, Initial_parameters.layers, Initial_parameters.wavenumbers, Initial_parameters.omega,
                            Initial_parameters.dz, Initial_parameters.S_medium, Initial_parameters.S_depth, Initial_parameters.S_type,
                            Initial_parameters.z, Initial_parameters.BCtop, Initial_parameters.BCbottom, Initial_parameters.rho,
                            Initial_parameters.Kmin, Initial_parameters.Kmax, Initial_parameters.dK, Initial_parameters.earth_interface,
                            Initial_parameters.Earth_depth, Initial_parameters.Ocean_depth, Initial_parameters.Atm_depth, Initial_parameters.Fname,
                            Initial_parameters.atm_atten_profile)



    print ("go to range")
    P,W,smooth_window=go_to_range_acoustic(A_p,A_w,Initial_parameters.cylin_spred,Initial_parameters.r,Initial_parameters.dK,
                Initial_parameters.wavenumbers,Initial_parameters.K,Initial_parameters.layers,Initial_parameters.Kmin,Initial_parameters.Kmax,
                Initial_parameters.rho,Initial_parameters.omega,Initial_parameters.eps)




    print ("Get exact Green's")
    A_exact,P_exact=get_exact_greens_acoustic_free(Initial_parameters.K,Initial_parameters.Vp,Initial_parameters.layers, Initial_parameters.earth_interface,
            Initial_parameters.wavenumbers,Initial_parameters.omega,Initial_parameters.dz,
            Initial_parameters.S_medium,Initial_parameters.S_depth,Initial_parameters.z,
            Initial_parameters.BCtop,Initial_parameters.BCbottom,Initial_parameters.rho,Initial_parameters.Kmin,
            Initial_parameters.Kmax,Initial_parameters.eps,Initial_parameters.dK,Initial_parameters.r)



    pos=30
    plt.figure(figsize=(20,10))
    plt.plot(Initial_parameters.r/1000,20*np.log10(np.abs(P[pos,:])/(4*pi)),'r',linewidth=2,label='FFP')
    plt.plot(Initial_parameters.r/1000,20*np.log10(np.abs(P_exact[pos,:])/(4*pi)),'b',linewidth=1,label='Exact')
    plt.xlim([0,20])
    plt.title('Acoustic free surface')
    plt.legend()
    plt.show()








def get_exact_greens_acoustic_rigid(K,vel,layers,earth_interface,wavenumbers,omega,dz,S_layer,S_depth,z,BCtop,BCbottom,rho,kmin,kmax,eps,dK,r):

    layers=layers-earth_interface
    Km=float(omega)/vel[earth_interface:]
    G=np.zeros([layers,wavenumbers],dtype=np.complex64);
    nu_exact=np.zeros([layers,wavenumbers],dtype=np.complex64);


    for i in range(0,len(K)):
        if np.real(K[i])>=float(kmin) and np.real(K[i])<=float(kmax) :
            for j in range(0,layers):

                #----------------------------------------------------
                #EXACT SOLUTION FOR HOMOGENEOUS MEDIUM RIGID SURFACE:
                #----------------------------------------------------
                nu=1j*np.sqrt(Km[j]**2-K[i]**2)
                nu_exact[j,i]=nu

                #Free field (both boundaries radiating)
                # G[j,i]=(1/(4*pi*nu))*(4*pi/(rho[S_layer+1]*omega**2))*(np.exp(nu*np.abs(z[j]-S_depth))) #-np.exp(nu*(z[j]+S_depth)))

                #Rigid surface
                G[j,i]=(1/(4*pi*nu))*(np.exp(nu*np.abs(z[j]-S_depth))+np.sign(z[j]-S_depth)*np.exp(nu*(z[j]+S_depth)))
                # ################################################################################################

    ################################################################################################
    #GOING TO RANGE            
    ################################################################################################
    l=np.linspace(0,wavenumbers-1,wavenumbers,dtype=np.int32)
    temp1=np.ndarray(wavenumbers*2,dtype=np.complex64)
    temp=(dK)*np.exp((eps+1j*np.real(K[0]))*r[:])    
    P=np.ndarray([layers,wavenumbers],dtype=np.complex64)
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
    
    for i in range(0,layers):
        temp1=np.concatenate([G[i,:],(G[i,::-1])*0])
        temp3=np.sqrt(K[:])*np.exp(1j*r[0]*dK*l[:])
        temp2=np.concatenate([temp3,temp3[:]*0])
        pp=np.fft.ifft(temp1*temp2*aa)*wavenumbers*2*2*pi
        P[i,:]=pp[:wavenumbers]*2*cylin_spred*temp

    return G,P


def get_exact_greens_acoustic_free(K,vel,layers,earth_interface,wavenumbers,omega,dz,S_layer,S_depth,z,BCtop,BCbottom,rho,kmin,kmax,eps,dK,r):

    layers=layers-earth_interface
    Km=float(omega)/vel[earth_interface:]
    G=np.zeros([layers,wavenumbers],dtype=np.complex64);
    nu_exact=np.zeros([layers,wavenumbers],dtype=np.complex64);

    S_layer=int((z[layers-1] - S_depth)/dz)
    for i in range(0,len(K)):
        if np.real(K[i])>=float(kmin) and np.real(K[i])<=float(kmax) :
            for j in range(0,layers):

                #---------------------------------------------------
                #EXACT SOLUTION FOR HOMOGENEOUS MEDIUM FREE SURFACE:
                #---------------------------------------------------
                nu=1j*np.sqrt(Km[j]**2-K[i]**2)
                nu_exact[j,i]=nu

                #Free surface
                # G[j,i]=(1/(4*pi*nu))*(4*pi/(rho[S_layer+1]*omega**2))*(np.exp(nu*np.abs(z[j]-S_depth))-np.exp(nu*(z[j]+S_depth)))
                G[j,i]=(1/(4*pi*nu))*(np.exp(nu*np.abs(z[j]-S_depth))-np.exp(nu*(z[j]+S_depth)))

                #Rigid surface
    ################################################################################################
    #GOING TO RANGE            
    ################################################################################################
    l=np.linspace(0,wavenumbers-1,wavenumbers,dtype=np.int32)
    temp1=np.ndarray(wavenumbers*2,dtype=np.complex64)
    temp=(dK)*np.exp((eps+1j*np.real(K[0]))*r[:])    
    P=np.ndarray([layers,wavenumbers],dtype=np.complex64)
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
    
    for i in range(0,layers):
        temp1=np.concatenate([G[i,:],(G[i,::-1])*0])
        temp3=np.sqrt(K[:])*np.exp(1j*r[0]*dK*l[:])
        temp2=np.concatenate([temp3,temp3[:]*0])
        pp=np.fft.ifft(temp1*temp2*aa)*wavenumbers*2*2*pi
        P[i,:]=pp[:wavenumbers]*2*cylin_spred*temp

    return G,P


def get_exact_greens_elastic_free(K,Vp,Vs,layers,earth_interface,wavenumbers,omega,dz,S_layer,S_depth,z,BCtop,BCbottom,rho,kmin,kmax,eps,dK,r,lamda,mu):

    S_depth=S_depth+dz
    layers=earth_interface
    Kmp=float(omega)/Vp[0]
    Kms=float(omega)/Vs[0]
    C1=np.zeros(wavenumbers,dtype=np.complex64)
    C2=np.zeros(wavenumbers,dtype=np.complex64)
    C3=np.zeros(wavenumbers,dtype=np.complex64)
    C4=np.zeros(wavenumbers,dtype=np.complex64)
    A_plus=np.zeros(wavenumbers,dtype=np.complex64)
    B_plus=np.zeros(wavenumbers,dtype=np.complex64)
       

    G=np.zeros([layers,wavenumbers],dtype=np.complex64); 
    stress_h=np.zeros(wavenumbers,dtype=np.complex64); 
    stress_v=np.zeros(wavenumbers,dtype=np.complex64); 
    plt.figure()
    for i in range(0,len(K)):
        if np.real(K[i])>=float(kmin) and np.real(K[i])<=float(kmax) :
            kr=(K[i])
            Kpz=np.sqrt(Kmp**2-kr**2)
            Ksz=np.sqrt(Kms**2-kr**2)
            alpha=1j*Kpz
            beta=1j*Ksz
            C1[i]=lamda*alpha**2 - lamda*kr**2 + 2*mu*alpha**2
            C2[i]=2*mu*kr*beta
            C3[i]=-2*kr*alpha
            C4[i]=kr**2 + beta**2

            temp=C3[i]*C2[i]/C1[i] + C4[i]

            
            
            B_plus[i]=(1/temp)*(kr/(2*pi)  - C3[i]/(4*pi*alpha))*np.exp(alpha*S_depth)

            A_plus[i]=(-1/(4*pi*alpha))*np.exp(alpha*S_depth) - (C2[i]/C1[i])*B_plus[i]

            for j in range(0,layers):
                #################################################################################################
                # #EXACT SOLUTION FOR HOMOGENEOUS MEDIUM:

                G[j,i]=(1/(4*pi))*np.sign(z[j]-S_depth)*np.exp(alpha*np.abs(z[j]-S_depth)) + alpha*A_plus[i]*np.exp(alpha*z[j]) + kr*B_plus[i]*np.exp(beta*z[j])
                # G_w[j,i]=(1/(4*pi))*np.sign(z[j]-S_depth)*np.exp(alpha*np.abs(z[j]-S_depth)) + alpha*A_plus[i]*np.exp(alpha*z[j]) 
                # G_w[j,i]=kr*B_plus[i]*np.exp(beta*z[j])
                #################################################################################################

    ################################################################################################
    #GOING TO RANGE            
    ################################################################################################
    l=np.linspace(0,wavenumbers-1,wavenumbers,dtype=np.int32)
    temp1=np.ndarray(wavenumbers*2,dtype=np.complex64)
    temp=(dK)*np.exp((eps+1j*np.real(K[0]))*r[:])    
    P=np.ndarray((layers,wavenumbers),dtype=np.complex64)
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
    
    for i in range(0,layers):
        temp1=np.concatenate([G[i,:],(G[i,::-1])*0])
        temp3=np.sqrt(K[:])*np.exp(1j*r[0]*dK*l[:])
        temp2=np.concatenate([temp3,temp3[:]*0])
        pp=np.fft.ifft(temp1*temp2*aa)*wavenumbers*2*2*pi
        P[i,:]=pp[:wavenumbers]*2*cylin_spred*temp

    return G,P


