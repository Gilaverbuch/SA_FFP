import matplotlib.pyplot as plt
import numpy as np 
from math import pi
import sys
from netCDF4 import Dataset, date2num, num2date
import os
import scipy.linalg.lapack as la
import time
from numba import jit, prange
import csv

#my library
from input_ import Parameters 
from output_ import save_results, save_time
from exact_solutions import Exact_solver
from Greens_parallel import get_greens_parallel, get_greens_acoustic_parallel, get_greens_elastic_parallel
from Go_to_range import go_to_range, go_to_range_acoustic
from Go_to_time import go_to_time



def solve_broadband(Initial_parameters):

    Initial_parameters.wave_num_and_range_broadband()
    Initial_parameters.velo_prof()
    Initial_parameters.print_velocity_profile()
    levels=len(Initial_parameters.Rec_alt)
    print ('levels to save', Initial_parameters.Rec_alt)
    # A_p_f=np.zeros([Initial_parameters.layers,Initial_parameters.wavenumbers,Initial_parameters.num_of_freq],dtype=np.complex64);
    # A_w_f=np.zeros([Initial_parameters.layers,Initial_parameters.wavenumbers,Initial_parameters.num_of_freq],dtype=np.complex64);
    # P_f=np.zeros([Initial_parameters.layers,Initial_parameters.wavenumbers,Initial_parameters.num_of_freq],dtype=np.complex64);
    # W_f=np.zeros([Initial_parameters.layers,Initial_parameters.wavenumbers,Initial_parameters.num_of_freq],dtype=np.complex64);


    A_p_f=np.zeros([levels,Initial_parameters.wavenumbers,Initial_parameters.num_of_freq],dtype=np.complex64);
    A_w_f=np.zeros([levels,Initial_parameters.wavenumbers,Initial_parameters.num_of_freq],dtype=np.complex64);
    P_f=np.zeros([levels,Initial_parameters.wavenumbers,Initial_parameters.num_of_freq],dtype=np.complex64);
    W_f=np.zeros([levels,Initial_parameters.wavenumbers,Initial_parameters.num_of_freq],dtype=np.complex64);


    P_t=np.zeros([levels,Initial_parameters.wavenumbers,Initial_parameters.num_of_freq],dtype=np.float32);
    W_t=np.zeros([levels,Initial_parameters.wavenumbers,Initial_parameters.num_of_freq],dtype=np.float32);


    if Initial_parameters.Ocean=='non' and Initial_parameters.Atm=='non':
        print('SOLVING ONLY FOR ELASTIC MEDIUM')






    elif Initial_parameters.Earth=='non':
        print('SOLVING ONLY FOR ACOUSTIC MEDIUM')
        pos_0= Initial_parameters.earth_interface + Initial_parameters.ocean_interface
        pos= np.int64(pos_0) + np.int64(Initial_parameters.Rec_alt/Initial_parameters.dz)
        print ('level number', pos)


        f_pos_max=min(range(len(Initial_parameters.freq)),key=lambda i: abs(Initial_parameters.freq[i]-Initial_parameters.max_frequency))
        f_pos_max=np.int64(f_pos_max/1000)*1000
        t0 = time.time()
        f=1
        while Initial_parameters.freq[f]<=Initial_parameters.max_frequency:

            if ((f*100)%f_pos_max)==0:
                t_now = time.time()
                print ('progress %d perc. in %f seconds' % (np.int64(f*100/f_pos_max), t_now-t0))



            # A_p_f[:,:,f],A_w_f[:,:,f]=get_greens_acoustic_parallel(Initial_parameters.K, Initial_parameters.Vp, Initial_parameters.Vs, Initial_parameters.lamda,
            #                     Initial_parameters.mu, Initial_parameters.layers, Initial_parameters.wavenumbers, Initial_parameters.omega[f],
            #                     Initial_parameters.dz, Initial_parameters.S_medium, Initial_parameters.S_depth, Initial_parameters.S_type,
            #                     Initial_parameters.z, Initial_parameters.BCtop, Initial_parameters.BCbottom, Initial_parameters.rho,
            #                     Initial_parameters.Kmin, Initial_parameters.Kmax, Initial_parameters.dK, Initial_parameters.earth_interface,
            #                     Initial_parameters.Earth_depth, Initial_parameters.Ocean_depth, Initial_parameters.Atm_depth, Initial_parameters.Fname)
        


            # P_f[:,:,f],W_f[:,:,f],smooth_window=go_to_range_acoustic(A_p_f[:,:,f],A_w_f[:,:,f],Initial_parameters.cylin_spred,Initial_parameters.r,Initial_parameters.dK,
            #             Initial_parameters.wavenumbers,Initial_parameters.K,Initial_parameters.layers,Initial_parameters.Kmin,Initial_parameters.Kmax,
            #             Initial_parameters.rho,Initial_parameters.omega[f],Initial_parameters.eps)

            A_p_f_temp,A_w_f_temp=get_greens_acoustic_parallel(Initial_parameters.K, Initial_parameters.Vp, Initial_parameters.Vs, Initial_parameters.lamda,
                                Initial_parameters.mu, Initial_parameters.layers, Initial_parameters.wavenumbers, Initial_parameters.omega[f],
                                Initial_parameters.dz, Initial_parameters.S_medium, Initial_parameters.S_depth, Initial_parameters.S_type,
                                Initial_parameters.z, Initial_parameters.BCtop, Initial_parameters.BCbottom, Initial_parameters.rho,
                                Initial_parameters.Kmin, Initial_parameters.Kmax, Initial_parameters.dK, Initial_parameters.earth_interface,
                                Initial_parameters.Earth_depth, Initial_parameters.Ocean_depth, Initial_parameters.Atm_depth, Initial_parameters.Fname)
        


            P_f_temp,W_f_temp,smooth_window=go_to_range_acoustic(A_p_f_temp,A_w_f_temp,Initial_parameters.cylin_spred,Initial_parameters.r,Initial_parameters.dK,
                        Initial_parameters.wavenumbers,Initial_parameters.K,Initial_parameters.layers,Initial_parameters.Kmin,Initial_parameters.Kmax,
                        Initial_parameters.rho,Initial_parameters.omega[f],Initial_parameters.eps)

            for l in range(0,levels):

                P_f[l,:,f]= P_f_temp[pos[l],:]
                A_p_f[l,:,f]= A_p_f_temp[pos[l],:]*smooth_window[:Initial_parameters.wavenumbers]

                # p1=min(range(len(Initial_parameters.K)),key=lambda i: abs(Initial_parameters.omega[f]/np.real(Initial_parameters.K[i]-400)))
                # p2=min(range(len(Initial_parameters.K)),key=lambda i: abs(Initial_parameters.omega[f]/np.real(Initial_parameters.K[i]-300)))    

                # A_p_f[l,:,f]= A_p_f[l,:,f]/(np.mean(A_p_f[l,p1:p2,f]))


            f+=1

        t1 = time.time()
        print (t1-t0)


        P_t=go_to_time(P_f,Initial_parameters.layers,Initial_parameters.wavenumbers,Initial_parameters.num_of_freq,
              Initial_parameters.freq,Initial_parameters.frequency1,Initial_parameters.time,Initial_parameters.df,
              Initial_parameters.earth_interface, Initial_parameters.ocean_interface, Initial_parameters.dt, 
              Initial_parameters.Rec_alt, Initial_parameters.dz)




        save_time(P_t, P_f, A_p_f,  Initial_parameters.time, Initial_parameters.freq, Initial_parameters.r, Initial_parameters.Rec_alt,  np.real(Initial_parameters.K), 
                    Initial_parameters.Fname)

        pos=min(range(len(Initial_parameters.r)),key=lambda i: abs(Initial_parameters.r[i]/1000-50))

        plt.figure(figsize=(20,5))
        plt.plot(Initial_parameters.time,P_t[0,pos,:],'r')
        plt.xlabel('Time [sec]')
        plt.ylabel('Pressure [?]')
        plt.xlim([250,350])
        plt.show()

        R_max=np.int32(Initial_parameters.r[-1]/1000)
        fin_t=1500
        for l in range(0,levels):
            plt.figure(figsize=(20,10))
            for i in range(0,Initial_parameters.wavenumbers,20):
                plt.plot(P_t[l,i,:]*1e3+Initial_parameters.r[i]/1000 , Initial_parameters.time,'black')
            # plt.xlim([0,Initial_parameters.r[i]/1000])
            plt.xlim([5,R_max])
            plt.ylim([0,fin_t])
            plt.yticks(np.arange(0, fin_t, 100))
            plt.title(r'$\sigma_{zz}$')
            plt.xlabel('Range [km]')
            plt.ylabel('Time [sec]')
            plt.show()


        pos=min(range(len(Initial_parameters.freq)),key=lambda i: abs(Initial_parameters.freq[i]-0.1))
        print(Initial_parameters.omega[pos]/(2*pi))

        plt.figure(figsize=(10,10))
        for i in range(pos,1000, 10):    
            plt.plot(np.real(Initial_parameters.K), np.abs(A_p_f[0,:,i])+Initial_parameters.omega[i])

        plt.xlim([0,0.04])
        plt.show()


        # plt.figure(figsize=(10,10))
        # plt.pcolormesh(np.real(Initial_parameters.K) , Initial_parameters.omega[pos:], np.transpose(np.abs(A_p_f[0,:,pos:])), cmap=plt.cm.Greys)
        # plt.clim([5,30])
        # plt.ylim([Initial_parameters.omega[pos],3])
        # plt.xlim([0,0.02])
        # plt.show()

        # np.savetxt('modes_array.out', np.abs(A_p_f[0,:,:])) 
        # np.savetxt('k_array.out', np.real(Initial_parameters.K)) 
        # np.savetxt('omega_array.out', Initial_parameters.omega) 






    else:
        print('SOLVING  SEISMO-ACOUSTIC')

        pos_0= Initial_parameters.earth_interface + Initial_parameters.ocean_interface
        pos= np.int64(pos_0) + np.int64(Initial_parameters.Rec_alt/Initial_parameters.dz)
        print ('level number', pos)


        f_pos_max=min(range(len(Initial_parameters.freq)),key=lambda i: abs(Initial_parameters.freq[i]-Initial_parameters.max_frequency))
        f_pos_max=np.int64(f_pos_max/1000)*1000
        t0 = time.time()


        f=min(range(len(Initial_parameters.freq)),key=lambda i: abs(Initial_parameters.freq[i]-0.005))

        KKmax=np.zeros(Initial_parameters.num_of_freq, dtype=np.float32)
        KKmax[:]=Initial_parameters.Kmax
        KKmax[:f]= Initial_parameters.Kmax/2
        
        f=10
        print (Initial_parameters.freq[f])
        while Initial_parameters.freq[f]<=Initial_parameters.max_frequency:

            if ((f*100)%f_pos_max)==0:
                t_now = time.time()
                print ('progress %d perc. in %f seconds' % (np.int64(f*100/f_pos_max), t_now-t0))



            # A_p_f_temp=get_greens_parallel(Initial_parameters.K,Initial_parameters.Vp,Initial_parameters.Vs,Initial_parameters.lamda,Initial_parameters.mu,Initial_parameters.layers,
            #                         Initial_parameters.wavenumbers,Initial_parameters.omega[f],Initial_parameters.dz,
            #                             Initial_parameters.S_medium,Initial_parameters.S_depth,Initial_parameters.S_type,
            #                             Initial_parameters.z,Initial_parameters.BCtop,Initial_parameters.BCbottom,Initial_parameters.rho
            #                             ,Initial_parameters.Kmin,Initial_parameters.Kmax,Initial_parameters.dK,Initial_parameters.earth_interface,
            #                             Initial_parameters.Earth_depth,Initial_parameters.Ocean_depth,Initial_parameters.Atm_depth,Initial_parameters.Fname,
            #                             Initial_parameters.delta_Kp, Initial_parameters.delta_Ks)

                
            # P_f_temp,smooth_window=go_to_range(A_p_f_temp,Initial_parameters.cylin_spred,Initial_parameters.r,Initial_parameters.dK,
            #                         Initial_parameters.wavenumbers,Initial_parameters.K,Initial_parameters.layers,Initial_parameters.Kmin,Initial_parameters.Kmax,
            #                         Initial_parameters.rho,Initial_parameters.omega[f],Initial_parameters.eps)

            A_p_f_temp=get_greens_parallel(Initial_parameters.K,Initial_parameters.Vp,Initial_parameters.Vs,Initial_parameters.lamda,Initial_parameters.mu,Initial_parameters.layers,
                                    Initial_parameters.wavenumbers,Initial_parameters.omega[f],Initial_parameters.dz,
                                        Initial_parameters.S_medium,Initial_parameters.S_depth,Initial_parameters.S_type,
                                        Initial_parameters.z,Initial_parameters.BCtop,Initial_parameters.BCbottom,Initial_parameters.rho
                                        ,Initial_parameters.Kmin,KKmax[f],Initial_parameters.dK,Initial_parameters.earth_interface,
                                        Initial_parameters.Earth_depth,Initial_parameters.Ocean_depth,Initial_parameters.Atm_depth,Initial_parameters.Fname,
                                        Initial_parameters.delta_Kp, Initial_parameters.delta_Ks)

                
            P_f_temp,smooth_window=go_to_range(A_p_f_temp,Initial_parameters.cylin_spred,Initial_parameters.r,Initial_parameters.dK,
                                    Initial_parameters.wavenumbers,Initial_parameters.K,Initial_parameters.layers,Initial_parameters.Kmin,KKmax[f],
                                    Initial_parameters.rho,Initial_parameters.omega[f],Initial_parameters.eps)



            for l in range(0,levels):

                P_f[l,:,f]= P_f_temp[pos[l],:]
                A_p_f[l,:,f]= A_p_f_temp[pos[l],:]*smooth_window[:Initial_parameters.wavenumbers]

                # p1=min(range(len(Initial_parameters.K)),key=lambda i: abs(Initial_parameters.omega[f]/np.real(Initial_parameters.K[i]-400)))
                # p2=min(range(len(Initial_parameters.K)),key=lambda i: abs(Initial_parameters.omega[f]/np.real(Initial_parameters.K[i]-300)))    

                # A_p_f[l,:,f]= A_p_f[l,:,f]/(np.mean(A_p_f[l,p1:p2,f]))


            f+=1

        t1 = time.time()
        print (t1-t0)


        P_t=go_to_time(P_f,Initial_parameters.layers,Initial_parameters.wavenumbers,Initial_parameters.num_of_freq,
              Initial_parameters.freq,Initial_parameters.frequency1,Initial_parameters.time,Initial_parameters.df,
              Initial_parameters.earth_interface, Initial_parameters.ocean_interface, Initial_parameters.dt, 
              Initial_parameters.Rec_alt, Initial_parameters.dz)




        save_time(P_t, P_f, A_p_f,  Initial_parameters.time, Initial_parameters.freq, Initial_parameters.r, Initial_parameters.Rec_alt,  np.real(Initial_parameters.K), 
                    Initial_parameters.Fname)

        pos=min(range(len(Initial_parameters.r)),key=lambda i: abs(Initial_parameters.r[i]/1000-50))

        plt.figure(figsize=(20,5))
        plt.plot(Initial_parameters.time,P_t[0,pos,:],'r')
        plt.xlabel('Time [sec]')
        plt.ylabel('Pressure [?]')
        plt.xlim([250,350])
        plt.show()

        R_max=np.int32(Initial_parameters.r[-1]/1000)
        fin_t=1500
        for l in range(0,levels):
            plt.figure(figsize=(20,10))
            for i in range(0,Initial_parameters.wavenumbers,20):
                plt.plot(P_t[l,i,:]*1e3+Initial_parameters.r[i]/1000 , Initial_parameters.time,'black')
            # plt.xlim([0,Initial_parameters.r[i]/1000])
            plt.xlim([5,R_max])
            plt.ylim([0,fin_t])
            plt.yticks(np.arange(0, fin_t, 100))
            plt.title(r'$\sigma_{zz}$')
            plt.xlabel('Range [km]')
            plt.ylabel('Time [sec]')
            plt.show()


        pos=min(range(len(Initial_parameters.freq)),key=lambda i: abs(Initial_parameters.freq[i]-0.1))
        print(Initial_parameters.omega[pos]/(2*pi))

        plt.figure(figsize=(10,10))
        for i in range(pos,1000, 10):    
            plt.plot(np.real(Initial_parameters.K), np.abs(A_p_f[0,:,i])+Initial_parameters.omega[i])

        plt.xlim([0,0.04])
        plt.show()


        # plt.figure(figsize=(10,10))
        # plt.pcolormesh(np.real(Initial_parameters.K) , Initial_parameters.omega[pos:], np.transpose(np.abs(A_p_f[0,:,pos:])), cmap=plt.cm.Greys)
        # plt.clim([5,30])
        # plt.ylim([Initial_parameters.omega[pos],3])
        # plt.xlim([0,0.02])
        # plt.show()

        # np.savetxt('modes_array.out', np.abs(A_p_f[0,:,:])) 
        # np.savetxt('k_array.out', np.real(Initial_parameters.K)) 
        # np.savetxt('omega_array.out', Initial_parameters.omega) 





















