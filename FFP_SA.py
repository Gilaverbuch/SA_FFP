
import os 

os.environ['MKL_NUM_THREADS']= "1" 
os.environ['NUMEXPR_NUM_THREADS']= "1"
os.environ['OMP_NUM_THREADS']= "1"


import matplotlib.pyplot as plt
import numpy as np 
from math import pi
import sys
# from netCDF4 import Dataset, date2num, num2date
import os
import scipy.linalg.lapack as la
import time
import numba
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








