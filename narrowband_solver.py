import matplotlib.pyplot as plt
import numpy as np 
from math import pi
import sys
# from netCDF4 import Dataset, date2num, num2date
import os
import scipy.linalg.lapack as la
import time
from numba import jit, prange

#my library
from input_ import Parameters 
from output_ import save_results
from exact_solutions import Exact_solver
from Greens_parallel import get_greens_parallel, get_greens_acoustic_parallel, get_greens_elastic_parallel
from Go_to_range import go_to_range, go_to_range_acoustic



def solve_narrowband(Initial_parameters):

    Initial_parameters.wave_num_and_range()
    Initial_parameters.velo_prof()
    Initial_parameters.print_velocity_profile()

    if Initial_parameters.Ocean=='non' and Initial_parameters.Atm=='non':
        print('SOLVING ONLY FOR ELASTIC MEDIUM')

        print ("Get Green's")
        t0 = time.time()
        A=get_greens_elastic_parallel(Initial_parameters.K,Initial_parameters.Vp,Initial_parameters.Vs,Initial_parameters.lamda,Initial_parameters.mu,Initial_parameters.layers,
                Initial_parameters.wavenumbers,Initial_parameters.omega,Initial_parameters.dz,
                    Initial_parameters.S_medium,Initial_parameters.S_depth,Initial_parameters.S_type,
                    Initial_parameters.z,Initial_parameters.BCtop,Initial_parameters.BCbottom,Initial_parameters.rho
                    ,Initial_parameters.Kmin,Initial_parameters.Kmax,Initial_parameters.dK,Initial_parameters.earth_interface,
                    Initial_parameters.Earth_depth,Initial_parameters.Ocean_depth,Initial_parameters.Atm_depth,Initial_parameters.Fname,
                    Initial_parameters.delta_Kp, Initial_parameters.delta_Ks)

        print ("Go to range")
        P,smooth_window=go_to_range(A,Initial_parameters.cylin_spred,Initial_parameters.r,Initial_parameters.dK,
                    Initial_parameters.wavenumbers,Initial_parameters.K,Initial_parameters.layers,Initial_parameters.Kmin,Initial_parameters.Kmax,
                    Initial_parameters.rho,Initial_parameters.omega,Initial_parameters.eps)

        t1 = time.time()
        print (t1-t0)

        save_results(A, P, Initial_parameters.z, Initial_parameters.r, Initial_parameters.omega, 
                    Initial_parameters.K, Initial_parameters.earth_interface, Initial_parameters.Earth_depth, Initial_parameters.ocean_interface, 
                    Initial_parameters.Ocean_depth, smooth_window, Initial_parameters.wavenumbers,
                    Initial_parameters.layers, Initial_parameters.dz, Initial_parameters.Fname, Initial_parameters.Rec_alt)




    elif Initial_parameters.Earth=='non':
        print('SOLVING ONLY FOR ACOUSTIC MEDIUM')

        print ("Get Green's")
        t0 = time.time()
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


        t1 = time.time()
        print (t1-t0)

        save_results(A_p, P, Initial_parameters.z, Initial_parameters.r, Initial_parameters.omega, Initial_parameters.Vp,
                    Initial_parameters.K, Initial_parameters.earth_interface, Initial_parameters.Earth_depth, Initial_parameters.ocean_interface, 
                    Initial_parameters.Ocean_depth, smooth_window, Initial_parameters.wavenumbers,
                    Initial_parameters.layers, Initial_parameters.dz, Initial_parameters.Fname, Initial_parameters.Rec_alt, 
                    Initial_parameters.rho, Initial_parameters.S_medium, Initial_parameters.S_depth)





    else:
        print('SOLVING  SEISMO-ACOUSTIC')
        print ("Get Green's")
        t0 = time.time()
        A=get_greens_parallel(Initial_parameters.K,Initial_parameters.Vp,Initial_parameters.Vs,Initial_parameters.lamda,Initial_parameters.mu,Initial_parameters.layers,
                Initial_parameters.wavenumbers,Initial_parameters.omega,Initial_parameters.dz,
                    Initial_parameters.S_medium,Initial_parameters.S_depth,Initial_parameters.S_type,
                    Initial_parameters.z,Initial_parameters.BCtop,Initial_parameters.BCbottom,Initial_parameters.rho
                    ,Initial_parameters.Kmin,Initial_parameters.Kmax,Initial_parameters.dK,Initial_parameters.earth_interface,
                    Initial_parameters.Earth_depth,Initial_parameters.Ocean_depth,Initial_parameters.Atm_depth,Initial_parameters.Fname,
                    Initial_parameters.delta_Kp, Initial_parameters.delta_Ks, Initial_parameters.atm_atten_profile)


        print ("Go to range")
        P,smooth_window=go_to_range(A,Initial_parameters.cylin_spred,Initial_parameters.r,Initial_parameters.dK,
                        Initial_parameters.wavenumbers,Initial_parameters.K,Initial_parameters.layers,Initial_parameters.Kmin,Initial_parameters.Kmax,
                        Initial_parameters.rho,Initial_parameters.omega,Initial_parameters.eps)
    
    
        t1 = time.time()
        print (t1-t0)

        save_results(A, P, Initial_parameters.z, Initial_parameters.r, Initial_parameters.omega, Initial_parameters.Vp,
                    Initial_parameters.K, Initial_parameters.earth_interface, Initial_parameters.Earth_depth, Initial_parameters.ocean_interface, 
                    Initial_parameters.Ocean_depth, smooth_window, Initial_parameters.wavenumbers,
                    Initial_parameters.layers, Initial_parameters.dz, Initial_parameters.Fname, Initial_parameters.Rec_alt,
                    Initial_parameters.rho, Initial_parameters.S_medium, Initial_parameters.S_depth)

