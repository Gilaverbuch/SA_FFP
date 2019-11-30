import numba
numba.config.NUMBA_NUM_THREADS=4

import os 
os.environ['MKL_NUM_THREADS']='1'
os.environ['NUMEXPR_NUM_THREADS']='1'
os.environ['OMP_NUM_THREADS']='1'


import matplotlib.pyplot as plt
import numpy as np 
from math import pi
import sys
from netCDF4 import Dataset, date2num, num2date
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
from narrowband_solver import solve_narrowband
from broadband_solver import solve_broadband

#--------------------------------------------------------------------------------------------------
######
#MAIN#
######
def FFP(inFile):
    Initial_parameters=Parameters(inFile)
    Initial_parameters.calc_parameters()


    
    #################
    #Exact solutions#
    #################
    if Initial_parameters.Fname=='exact':
        E_S_=Exact_solver(Initial_parameters.exact)
        E_S_.solve()
        
        


    ################
    #Any input file#
    ################
    else:
        if Initial_parameters.simulation_type=='narrowband':
            print('NARROWBAND SIMULATION')
            solve_narrowband(Initial_parameters)


        elif Initial_parameters.simulation_type=='broadband':
            print('BROADBAND SIMULATION')
            solve_broadband(Initial_parameters)


if __name__ == "__main__":
    import sys
    FFP(sys.argv[1])















            # Initial_parameters.wave_num_and_range()
            # Initial_parameters.velo_prof()
            # Initial_parameters.print_velocity_profile()

            # if Initial_parameters.Ocean=='non' and Initial_parameters.Atm=='non':
            #     print('SOLVING ONLY FOR ELASTIC MEDIUM')

            #     print ("Get Green's")
            #     t0 = time.time()
            #     A=get_greens_elastic_parallel(Initial_parameters.K,Initial_parameters.Vp,Initial_parameters.Vs,Initial_parameters.lamda,Initial_parameters.mu,Initial_parameters.layers,
            #             Initial_parameters.wavenumbers,Initial_parameters.omega,Initial_parameters.dz,
            #                 Initial_parameters.S_medium,Initial_parameters.S_depth,Initial_parameters.S_type,
            #                 Initial_parameters.z,Initial_parameters.BCtop,Initial_parameters.BCbottom,Initial_parameters.rho
            #                 ,Initial_parameters.Kmin,Initial_parameters.Kmax,Initial_parameters.dK,Initial_parameters.earth_interface,
            #                 Initial_parameters.Earth_depth,Initial_parameters.Ocean_depth,Initial_parameters.Atm_depth,Initial_parameters.Fname,
            #                 Initial_parameters.delta_Kp, Initial_parameters.delta_Ks)

            #     print ("Go to range")
            #     P,smooth_window=go_to_range(A,Initial_parameters.cylin_spred,Initial_parameters.r,Initial_parameters.dK,
            #                 Initial_parameters.wavenumbers,Initial_parameters.K,Initial_parameters.layers,Initial_parameters.Kmin,Initial_parameters.Kmax,
            #                 Initial_parameters.rho,Initial_parameters.omega,Initial_parameters.eps)

            #     t1 = time.time()
            #     print (t1-t0)

            #     save_results(A, P, Initial_parameters.z, Initial_parameters.r, Initial_parameters.omega, 
            #                 Initial_parameters.K, Initial_parameters.earth_interface, Initial_parameters.Earth_depth, Initial_parameters.ocean_interface, 
            #                 Initial_parameters.Ocean_depth, smooth_window, Initial_parameters.wavenumbers,
            #                 Initial_parameters.layers, Initial_parameters.dz, Initial_parameters.Fname, Initial_parameters.Rec_alt)




            # elif Initial_parameters.Earth=='non':
            #     print('SOLVING ONLY FOR ACOUSTIC MEDIUM')

            #     print ("Get Green's")
            #     t0 = time.time()
            #     A_p,A_w=get_greens_acoustic_parallel(Initial_parameters.K, Initial_parameters.Vp, Initial_parameters.Vs, Initial_parameters.lamda,
            #                             Initial_parameters.mu, Initial_parameters.layers, Initial_parameters.wavenumbers, Initial_parameters.omega,
            #                             Initial_parameters.dz, Initial_parameters.S_medium, Initial_parameters.S_depth, Initial_parameters.S_type,
            #                             Initial_parameters.z, Initial_parameters.BCtop, Initial_parameters.BCbottom, Initial_parameters.rho,
            #                             Initial_parameters.Kmin, Initial_parameters.Kmax, Initial_parameters.dK, Initial_parameters.earth_interface,
            #                             Initial_parameters.Earth_depth, Initial_parameters.Ocean_depth, Initial_parameters.Atm_depth, Initial_parameters.Fname)
                


            #     print ("Go to range")
            #     P,W,smooth_window=go_to_range_acoustic(A_p,A_w,Initial_parameters.cylin_spred,Initial_parameters.r,Initial_parameters.dK,
            #                 Initial_parameters.wavenumbers,Initial_parameters.K,Initial_parameters.layers,Initial_parameters.Kmin,Initial_parameters.Kmax,
            #                 Initial_parameters.rho,Initial_parameters.omega,Initial_parameters.eps)


            #     t1 = time.time()
            #     print (t1-t0)

            #     save_results(A, P, Initial_parameters.z, Initial_parameters.r, Initial_parameters.omega, 
            #                 Initial_parameters.K, Initial_parameters.earth_interface, Initial_parameters.Earth_depth, Initial_parameters.ocean_interface, 
            #                 Initial_parameters.Ocean_depth, smooth_window, Initial_parameters.wavenumbers,
            #                 Initial_parameters.layers, Initial_parameters.dz, Initial_parameters.Fname, Initial_parameters.Rec_alt)

            # else:
            #     print('SOLVING  SEISMO-ACOUSTIC')
            #     print ("Get Green's")
            #     t0 = time.time()
            #     A=get_greens_parallel(Initial_parameters.K,Initial_parameters.Vp,Initial_parameters.Vs,Initial_parameters.lamda,Initial_parameters.mu,Initial_parameters.layers,
            #             Initial_parameters.wavenumbers,Initial_parameters.omega,Initial_parameters.dz,
            #                 Initial_parameters.S_medium,Initial_parameters.S_depth,Initial_parameters.S_type,
            #                 Initial_parameters.z,Initial_parameters.BCtop,Initial_parameters.BCbottom,Initial_parameters.rho
            #                 ,Initial_parameters.Kmin,Initial_parameters.Kmax,Initial_parameters.dK,Initial_parameters.earth_interface,
            #                 Initial_parameters.Earth_depth,Initial_parameters.Ocean_depth,Initial_parameters.Atm_depth,Initial_parameters.Fname,
            #                 Initial_parameters.delta_Kp, Initial_parameters.delta_Ks)


            #     print ("Go to range")
            #     P,smooth_window=go_to_range(A,Initial_parameters.cylin_spred,Initial_parameters.r,Initial_parameters.dK,
            #                     Initial_parameters.wavenumbers,Initial_parameters.K,Initial_parameters.layers,Initial_parameters.Kmin,Initial_parameters.Kmax,
            #                     Initial_parameters.rho,Initial_parameters.omega,Initial_parameters.eps)
            
            
            #     t1 = time.time()
            #     print (t1-t0)

            #     save_results(A, P, Initial_parameters.z, Initial_parameters.r, Initial_parameters.omega, 
            #                 Initial_parameters.K, Initial_parameters.earth_interface, Initial_parameters.Earth_depth, Initial_parameters.ocean_interface, 
            #                 Initial_parameters.Ocean_depth, smooth_window, Initial_parameters.wavenumbers,
            #                 Initial_parameters.layers, Initial_parameters.dz, Initial_parameters.Fname, Initial_parameters.Rec_alt)



