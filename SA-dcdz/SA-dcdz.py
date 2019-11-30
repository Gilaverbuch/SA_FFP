#!/usr/bin/env python

import matplotlib.pyplot as plt
import numpy as np 
import math
import cmath
from math import exp,pi,sin,cos,radians,sqrt,acos,degrees
import scipy
from scipy import interpolate
from scikits.umfpack import spsolve, splu
import sys
from netCDF4 import Dataset, date2num, num2date
import os

##############################################################################################################################################################
#general functions
##############################################################################################################################################################

#VELOCITY PROFILE FUNCTIONS
#########################################################################
def read_atm_profile(profile_name,atm_depth,dz,direction):
    #reading atmopsher conditions
    layers=int(atm_depth/dz)
    z=np.zeros(layers,dtype=int)
    for i in range(0,layers):
        z[i]=i*dz

    parameters=np.loadtxt(profile_name)
    l=len(parameters)
    alt=np.ndarray(l,dtype=float)
    wind_zonal=np.ndarray(l,dtype=float) #west-east direction
    wind_merid=np.ndarray(l,dtype=float) #north-south direct
    temp=np.ndarray(l,dtype=float)
    vel_adia=np.ndarray(l,dtype=float)
    vel_effec=np.ndarray(l,dtype=float)
    vel_interp=np.ndarray(layers,dtype=float)
    rho=np.ndarray(l,dtype=float)
    rho_interp=np.ndarray(layers,dtype=float)

    

    for i in range(0,l):
        alt[i]=parameters[i,0]
        wind_zonal[i]=parameters[i,1]
        wind_merid[i]=parameters[i,2]
        vel_adia[i]=331.3+0.66*(parameters[i,4]-272.15)#331.3*np.sqrt(1+(parameters[i,4]-273.15)/273.15)
        rho[i]=parameters[i,5]*1000

    # for i in range(0,l):
    #     alt[i]=parameters[i,0]
    #     wind_zonal[i]=0#parameters[i,1]
    #     wind_merid[i]=0#parameters[i,2]
    #     vel_adia[i]=parameters[i,1]#331.3+0.66*(parameters[i,4]-272.15)#331.3*np.sqrt(1+(parameters[i,4]-273.15)/273.15)
    #     rho[i]=1#parameters[i,5]*1000



    alt[:]=alt[:] - alt[0]

    phi=radians(direction)
    vel_effec=np.sqrt((vel_adia*sin(phi)+wind_zonal)**2 + (vel_adia*cos(phi)+wind_zonal)**2)


    a=scipy.interpolate.PchipInterpolator(alt,vel_effec, extrapolate=True)
    vel_interp=a(z/float(1000))
    a=scipy.interpolate.PchipInterpolator(alt,rho, extrapolate=True)
    rho_interp=a(z/float(1000))


    # print (vel_interp)
    plt.figure()
    plt.plot(vel_interp,z/1000,'r',linewidth=3.0)
    plt.plot(vel_adia,alt,'g')
    plt.show()

    




    return vel_interp,rho_interp
#########################################################################
def make_munk(ocean_depth,nz):
    # # Munk profile
    c0=float(1500)
    eps=float(0.00737)
    z_water= np.linspace(0,ocean_depth,nz) # grid coordinates
    x=np.linspace(0,ocean_depth,nz)
    temp=np.linspace(0,ocean_depth,nz)
    x[:]=2*(z_water[:]-1300)/1300;  
    temp=c0*(1+eps*(x[:]-1+np.exp(-x[:])))

    return temp[::-1]


#########################################################################
# INITIALIZING COEFFICIENTS MATRIX
#########################################################################
def coefficients(layers,kr,Vp,Vs,lamda,mu,omega,z,dz,Force,BCtop,BCbottom,rho,earth_interface):
    
    # Km and Kpz and nu need to be calculated for each frequency. kr is te only global veriable

    Kmp=float(omega)/Vp[:]
    Kms=0*Kmp
    Kms=float(omega)/Vs[:earth_interface]
    c_s=np.zeros([layers*4,4],dtype=complex)
    c1_s=np.zeros([layers*4,4],dtype=complex)

    c_f=np.zeros([layers*2,2],dtype=complex)
    c1_f=np.zeros([layers*2,2],dtype=complex)

    alpha=np.zeros([layers],dtype=complex)
    beta=np.zeros([layers],dtype=complex)
    C1=np.zeros([layers],dtype=complex)
    C2=np.zeros([layers],dtype=complex)
    C3=np.zeros([layers],dtype=complex)
    C4=np.zeros([layers],dtype=complex)


    #elastic layer coeff.
    for i in range(0,earth_interface):

        Kpz=np.sqrt(Kmp[i]**2 - kr**2)
        Ksz=np.sqrt(Kms[i]**2 - kr**2)

        alpha[i]=1j*Kpz
        beta[i]=1j*Ksz
        C1[i]=lamda[i]*alpha[i]**2 - lamda[i]*kr**2 + 2*mu[i]*alpha[i]**2
        C2[i]=2*mu[i]*kr*beta[i]
        C3[i]=-2*kr*alpha[i]
        C4[i]=kr**2 + beta[i]**2


  
        #bottom interface
        c_s[i*4,0]=alpha[i]*np.exp(alpha[i]*dz)
        c_s[i*4,1]=kr*np.exp(beta[i]*dz)
        c_s[i*4,2]=-alpha[i]
        c_s[i*4,3]=kr

        c_s[i*4+1,0]=-kr*np.exp(alpha[i]*dz)
        c_s[i*4+1,1]=-beta[i]*np.exp(beta[i]*dz)
        c_s[i*4+1,2]=-kr
        c_s[i*4+1,3]=beta[i]

        c_s[i*4+2,0]=C1[i]*np.exp(alpha[i]*dz)
        c_s[i*4+2,1]=C2[i]*np.exp(beta[i]*dz)
        c_s[i*4+2,2]=C1[i]
        c_s[i*4+2,3]=-C2[i]

        c_s[i*4+3,0]=mu[i]*C3[i]*np.exp(alpha[i]*dz)
        c_s[i*4+3,1]=-mu[i]*C4[i]*np.exp(beta[i]*dz)
        c_s[i*4+3,2]=-mu[i]*C3[i]
        c_s[i*4+3,3]=-mu[i]*C4[i]



                
        #top interface
        c1_s[i*4,0]=alpha[i]
        c1_s[i*4,1]=kr
        c1_s[i*4,2]=-alpha[i]*np.exp(alpha[i]*dz)
        c1_s[i*4,3]=kr*np.exp(beta[i]*dz)

        c1_s[i*4+1,0]=-kr
        c1_s[i*4+1,1]=-beta[i]
        c1_s[i*4+1,2]=-kr*np.exp(alpha[i]*dz)
        c1_s[i*4+1,3]=beta[i]*np.exp(beta[i]*dz)

        c1_s[i*4+2,0]=C1[i]
        c1_s[i*4+2,1]=C2[i]
        c1_s[i*4+2,2]=C1[i]*np.exp(alpha[i]*dz)
        c1_s[i*4+2,3]=-C2[i]*np.exp(beta[i]*dz)

        c1_s[i*4+3,0]=mu[i]*C3[i]
        c1_s[i*4+3,1]=-mu[i]*C4[i]
        c1_s[i*4+3,2]=-mu[i]*C3[i]*np.exp(alpha[i]*dz)
        c1_s[i*4+3,3]=-mu[i]*C4[i]*np.exp(beta[i]*dz)

       # #bottom interface
        # c_s[i*4,0]=i+1
        # c_s[i*4,1]=i+1
        # c_s[i*4,2]=i+1
        # c_s[i*4,3]=i+1

        # c_s[i*4+1,0]=i+1
        # c_s[i*4+1,1]=i+1
        # c_s[i*4+1,2]=i+1
        # c_s[i*4+1,3]=i+1

        # c_s[i*4+2,0]=i+1
        # c_s[i*4+2,1]=i+1
        # c_s[i*4+2,2]=i+1
        # c_s[i*4+2,3]=i+1

        # c_s[i*4+3,0]=i+1
        # c_s[i*4+3,1]=i+1
        # c_s[i*4+3,2]=i+1
        # c_s[i*4+3,3]=i+1



                
        # #top interface
        # c1_s[i*4,0]=i+1
        # c1_s[i*4,1]=i+1
        # c1_s[i*4,2]=i+1
        # c1_s[i*4,3]=i+1

        # c1_s[i*4+1,0]=i+1
        # c1_s[i*4+1,1]=i+1
        # c1_s[i*4+1,2]=i+1
        # c1_s[i*4+1,3]=i+1

        # c1_s[i*4+2,0]=i+1
        # c1_s[i*4+2,1]=i+1
        # c1_s[i*4+2,2]=i+1
        # c1_s[i*4+2,3]=i+1

        # c1_s[i*4+3,0]=i+1
        # c1_s[i*4+3,1]=i+1
        # c1_s[i*4+3,2]=i+1
        # c1_s[i*4+3,3]=i+1

    #fluid layers coeff.
    for i in range(earth_interface,layers):

        Kpz=np.sqrt(Kmp[i]**2 - kr**2)
        alpha[i]=1j*Kpz


        #bottom interface
        c_f[i*2,0]=alpha[i]*np.exp(alpha[i]*dz)
        c_f[i*2,1]=-alpha[i]*np.exp(alpha[i]*0)
        c_f[i*2+1,0]=rho[i]*(omega**2)*np.exp(alpha[i]*dz)
        c_f[i*2+1,1]=rho[i]*(omega**2)*np.exp(alpha[i]*0)
                
        #top interface
        c1_f[i*2,0]=alpha[i]*np.exp(alpha[i]*0)
        c1_f[i*2,1]=-alpha[i]*np.exp(alpha[i]*dz)
        c1_f[i*2+1,0]=rho[i]*(omega**2)*np.exp(alpha[i]*0)
        c1_f[i*2+1,1]=rho[i]*(omega**2)*np.exp(alpha[i]*dz)

        # #bottom interface
        # c_f[i*2,0]=i+1
        # c_f[i*2,1]=i+1
        # c_f[i*2+1,0]=i+1
        # c_f[i*2+1,1]=i+1
                
        # #top interface
        # c1_f[i*2,0]=i+1
        # c1_f[i*2,1]=i+1
        # c1_f[i*2+1,0]=i+1
        # c1_f[i*2+1,1]=i+1



    mat_size=earth_interface*4 + (layers-earth_interface-2)*2 
    C=np.zeros([mat_size, mat_size] , dtype=complex)


    #SETTING TOP BOUNDARY CONDITIONS FOR ELASTIC LAYER
    if BCtop=='rigid':
        #Vetical displacement w
        C[0,0]=alpha[0]
        C[0,1]=kr
        C[0,2]=-alpha[0]*np.exp(alpha[0]*dz)
        C[0,3]=kr*np.exp(beta[0]*dz)

        #Horizontal displacement u
        C[1,0]=-kr
        C[1,1]=-beta[0]
        C[1,2]=-kr*np.exp(alpha[0]*dz)
        C[1,3]=beta[0]*np.exp(beta[0]*dz)

    elif BCtop=='free':
        #Vetical stress sigma_zz
        C[0,0]=C1[0]
        C[0,1]=C2[0]
        C[0,2]=C1[0]*np.exp(alpha[0]*dz)
        C[0,3]=-C2[0]*np.exp(beta[0]*dz)

        #Horizontal stress sigma_rz
        C[1,0]=C3[0]
        C[1,1]=-C4[0]
        C[1,2]=-C3[0]*np.exp(alpha[0]*dz)
        C[1,3]=-C4[0]*np.exp(beta[0]*dz)

    elif BCtop=='radiation':


        #Phi
        C[0,0]=2
        C[0,1]=0
        C[0,2]=0
        C[0,3]=0

        #Psi
        C[1,0]=0
        C[1,1]=2
        C[1,2]=0
        C[1,3]=0
    else:
        print ('wrong boundary conditions')
        sys.exit()


    #create C matrix with 2 for loops. 1st for the solid and 2nd for the fluid.
    for i in range(0,(earth_interface-1)*4,4):
        for j in range(0,4):

            C[i+2+j,i]=c_s[i+j,0]
            C[i+2+j,i+1]=c_s[i+j,1]
            C[i+2+j,i+2]=c_s[i+j,2]
            C[i+2+j,i+3]=c_s[i+j,3]

            C[i+2+j,i+4]=-c1_s[i+4+j,0]
            C[i+2+j,i+4+1]=-c1_s[i+4+j,1]
            C[i+2+j,i+4+2]=-c1_s[i+4+j,2]
            C[i+2+j,i+4+3]=-c1_s[i+4+j,3]

    last_pos=i+2+j
    #NOW 3 ROWS OF B.C BETWEEN SOLID AND FLUID
    #BC 1 horizontal stress vanishes
    bc_pos=i+4
    bc_f_pos=int((last_pos-1)/2+2)

    C[last_pos+1,i+4]=c_s[bc_pos+3,0]
    C[last_pos+1,i+4+1]=c_s[bc_pos+3,1]
    C[last_pos+1,i+4+2]=c_s[bc_pos+3,2]
    C[last_pos+1,i+4+3]=c_s[bc_pos+3,3]

    #BC 2 continuity of vertical stress
    #solid
    C[last_pos+2,i+4]=c_s[bc_pos+2,0]
    C[last_pos+2,i+4+1]=c_s[bc_pos+2,1]
    C[last_pos+2,i+4+2]=c_s[bc_pos+2,2]
    C[last_pos+2,i+4+3]=c_s[bc_pos+2,3]

    #fluid
    C[last_pos+2,i+4+4]=c1_f[bc_f_pos+1,0]
    C[last_pos+2,i+4+5]=c1_f[bc_f_pos+1,1]

    #BC 3 continuity of vertical displacement
    #solid
    C[last_pos+3,i+4]=c_s[bc_pos,0]
    C[last_pos+3,i+4+1]=c_s[bc_pos,1]
    C[last_pos+3,i+4+2]=c_s[bc_pos,2]
    C[last_pos+3,i+4+3]=c_s[bc_pos,3]

    #fluid
    C[last_pos+3,i+4+4]=-c1_f[bc_f_pos,0]
    C[last_pos+3,i+4+5]=-c1_f[bc_f_pos,1]

    initial_pos=int((last_pos+4)/2)
    for i in range(last_pos+4,mat_size-2,2):

        fluid_pos=initial_pos + i-(last_pos+4)
        C[i,i-1]=c_f[fluid_pos,0]
        C[i,i]=c_f[fluid_pos,1]
        C[i+1,i-1]=c_f[fluid_pos+1,0]
        C[i+1,i]=c_f[fluid_pos+1,1]

        C[i,i+1]=-c1_f[fluid_pos+2,0]
        C[i,i+2]=-c1_f[fluid_pos+2,1]
        C[i+1,i+1]=-c1_f[fluid_pos+3,0]
        C[i+1,i+2]=-c1_f[fluid_pos+3,1] 




    #SETTING BOTTOM BOUNDARY CONDITIONS FOR FLUID LAYER

    if BCbottom=='free':
        C[mat_size-1,mat_size-2]=rho[layers-1]*(omega**2)*np.exp(alpha[layers-1]*dz)
        C[mat_size-1,mat_size-1]=rho[layers-1]*(omega**2)
    elif BCbottom=='rigid':
        C[mat_size-1,mat_size-2]=np.exp(alpha[layers-1]*dz)
        C[mat_size-1,mat_size-1]=-1
    elif BCbottom=='radiation':
        C[mat_size-1,mat_size-2]=0
        C[mat_size-1,mat_size-1]=2
    else:
        print ('wrong boundary conditions')
        sys.exit()


    #############################
    ##SCALING
    #############################
    for i in range(0,mat_size-1):
        scaler=max(np.abs(C[i,:]))
        C[i,:]=C[i,:]/scaler
        Force[i]=Force[i]/scaler

    return C,Force,alpha,beta


##############################################################################################################################################################
def force_vec(layers,S_depth,S_medium,S_type,z,kr,omega,dz,Vp,Vs,lamda,mu,rho,dK,earth_interface,Earth_depth,Ocean_depth,Atm_depth):

    #need to fix the force for each wavenumber according to the integral equations for the forcing
    #terms for the displacement and the stress.
    #the force for each interface should be according to F_stress_(n+1)-F_stress_n
    #and F_disp_(n+1)-F_disp_n
    mat_size=earth_interface*4 + (layers-earth_interface-2)*2 
    Force=np.zeros(mat_size,dtype=complex)

    if S_medium=='earth':

        S_layer=earth_interface - int(S_depth/dz)-1
        S_depth=(earth_interface)*dz - S_depth
        pos=(S_layer-1)*4+2 
        Kmp=float(omega)/Vp[S_layer]
        Kms=float(omega)/Vs[S_layer]
        Kpz=np.sqrt(Kmp**2-kr**2)
        Ksz=np.sqrt(Kms**2-kr**2)
        alpha=1j*Kpz
        beta=1j*Ksz
        if S_type=='source':
            #Forcing terma in the following order:w, u, sigma_zz, sigma_rz
            C1=(lamda[S_layer]*alpha**2 - lamda[S_layer]*kr**2 +2*mu[S_layer]*alpha**2)
            z_bottom=np.abs(z[S_layer]-S_depth)
            Force[pos-4]=-(1/(4*pi))*np.exp(alpha*z_bottom)
            Force[pos-3]=-(kr/(4*pi*alpha))*np.exp(alpha*z_bottom) 
            Force[pos-2]=(1/(4*pi*alpha))*C1*np.exp(alpha*z_bottom)
            Force[pos-1]=mu[S_layer]*(kr/(2*pi))*np.exp(alpha*z_bottom)

            z_top=np.abs(z[S_layer+1]-S_depth)
            Force[pos]=-(1/(4*pi))*np.exp(alpha*z_top) 
            Force[pos+1]=(kr/(4*pi*alpha))*np.exp(alpha*z_top) 
            Force[pos+2]=-(1/(4*pi*alpha))*C1*np.exp(alpha*z_top)
            Force[pos+3]=mu[S_layer]*(kr/(2*pi))*np.exp(alpha*z_top)
        elif S_type=='force':
            print ('point force')
            sys.exit()





        else:
            print ('wrong parameters for source!')
            sys.exit()

    elif S_medium=='ocean':
        S_layer=earth_interface + int((Ocean_depth - S_depth)/dz)
        S_depth=Earth_depth + Ocean_depth - S_depth
        pos=earth_interface*4 + (S_layer - earth_interface)*2 -1

        Kmp=float(omega)/Vp[S_layer]
        Kpz=np.sqrt(Kmp**2-kr**2)
        alpha=1j*Kpz

        Force[pos-2]=-1/(4*pi)*np.exp(alpha*np.abs(z[S_layer]-S_depth))
        Force[pos-1]=rho[S_layer+1]*(omega**2)/(4*pi*alpha)*np.exp(alpha*np.abs(z[S_layer]-S_depth))

        Force[pos]=-1/(4*pi)*np.exp(alpha*np.abs(z[S_layer+1]-S_depth))
        Force[pos+1]=-rho[S_layer+1]*(omega**2)/(4*pi*alpha)*np.exp(alpha*np.abs(z[S_layer+1]-S_depth))

    elif S_medium=='atm':
        S_depth=Atm_depth-S_depth
        S_layer=earth_interface + int((Ocean_depth + Atm_depth - S_depth)/dz)
        S_depth=Earth_depth + Ocean_depth + Atm_depth - S_depth
        pos=earth_interface*4 + (S_layer +1 - earth_interface)*2 -1

        # print (pos)
        Kmp=float(omega)/Vp[S_layer]
        Kpz=np.sqrt(Kmp**2-kr**2)
        alpha=1j*Kpz
        # print(z[S_layer],S_depth)

        Force[pos-2]=-1/(4*pi)*np.exp(alpha*np.abs(z[S_layer]-S_depth))
        Force[pos-1]=rho[S_layer+1]*(omega**2)/(4*pi*alpha)*np.exp(alpha*np.abs(z[S_layer]-S_depth))

        Force[pos]=-1/(4*pi)*np.exp(alpha*np.abs(z[S_layer+1]-S_depth))
        Force[pos+1]=-rho[S_layer+1]*(omega**2)/(4*pi*alpha)*np.exp(alpha*np.abs(z[S_layer+1]-S_depth))

        print(Force[pos-2:pos+1])
        sys.exit()


    else:
        print ('wrong parameters for source!')
        sys.exit()






    return Force*4*pi/(rho[S_layer+1]*omega**2)
##############################################################################################################################################################


def get_greens(K,Vp,Vs,lamda,mu,layers,wavenumbers,omega,dz,S_medium,S_depth,S_type,z,BCtop,BCbottom,rho,kmin,kmax,dK,earth_interface,Earth_depth,Ocean_depth,Atm_depth):

    G=np.zeros([layers,wavenumbers],dtype=complex);
    alpha=np.zeros([layers,wavenumbers],dtype=complex)
    beta=np.zeros([layers,wavenumbers],dtype=complex)


    for i in range(0,len(K)):
        
        if np.real(K[i])>=float(kmin) and np.real(K[i])<=float(kmax) :
            # print (np.real(K[i]))

            ######################
            #creating force vector
            ######################
            Force=force_vec(layers,S_depth,S_medium,S_type,z,K[i],omega,dz,Vp,Vs,lamda,mu,rho,dK,earth_interface,Earth_depth,Ocean_depth,Atm_depth)

            ############################
            #creating coefficents matrix 
            ############################
            C,Force,alpha[:,i],beta[:,i]=coefficients(layers,K[i],Vp,Vs,lamda,mu,omega,z,dz,Force,BCtop,BCbottom,rho,earth_interface)
            CC=scipy.sparse.csr_matrix(C)
            A_sol=spsolve(CC,Force[:]) #this is the fast solver


            for j in range(0,earth_interface):

                G[j,i]=alpha[j,i]*A_sol[j*4] + K[i]*A_sol[j*4+1] - alpha[j,i]*A_sol[j*4+2]*np.exp(alpha[j,i]*dz) + K[i]*A_sol[j*4+3]*np.exp(beta[j,i]*dz)

                #only S
                # G[j,i]= K[i]*A_sol[j*4+1]  + K[i]*A_sol[j*4+3]*np.exp(beta[j,i]*dz)

                #only P
                # G[j,i]=alpha[j,i]*A_sol[j*4]  - alpha[j,i]*A_sol[j*4+2]*np.exp(alpha[j,i]*dz)

            last_pos=j*4+3   
            for j in range(earth_interface,layers-2):
                sol_pos=last_pos + 1 + (j - earth_interface)*2 
                G[j,i]=A_sol[sol_pos] + A_sol[sol_pos+1]*np.exp(alpha[j,i]*dz)

    return G




def go_to_range(G,cylin_spred,r,dK,wavenumbers,K,layers,kmin,kmax,rho,omega,eps):


    l=np.linspace(0,wavenumbers-1,wavenumbers,dtype=int)
    temp1=np.zeros(wavenumbers*2,dtype=complex)
    temp=(dK)*np.exp((eps+1j*np.real(K[0]))*r[:])    
    P=np.zeros([layers,wavenumbers],dtype=complex)
    cylin_spred=np.exp(-1j*pi/4)/np.sqrt(2*pi*r[:])
    aa=np.zeros(Initial_parameters.wavenumbers*2)
    pos1=0
    while np.real(K[pos1])<=kmin:
            pos1=pos1+1;                
                
    pos2=wavenumbers-1
    while np.real(K[pos2])>=kmax:
            pos2=pos2-1;
    wind=np.hanning(pos2-pos1)
    aa[pos1:pos2]=wind
    for i in range(0,layers):
        temp1=np.concatenate([G[i,:],(G[i,::-1])*0])
        temp3=np.sqrt(K[:])*np.exp(1j*r[0]*dK*l[:])
        temp2=np.concatenate([temp3,temp3[:]*0])
        pp=np.fft.ifft(temp1*temp2*aa)*wavenumbers*2*2*pi
        P[i,:]=rho[i]*(omega**2)*pp[:wavenumbers]*2*cylin_spred*temp


    return P,aa

##############################################################################################################################################################
##############################################################################################################################################################
class Parameters:
    """
    Parameters
    Read, calculate and save all the parameters needed for modeling the acoustic wavefield in stratified medium
    Variables:
        filename: input file with initial parameters

    """

    def __init__(self,filename):
        self.filename=filename





    def calc_parameters(self):
        """
        Reading input file, saves input parameters and calculating number of layers, depth vector and source layer  

        """

        print ('reading parameters...')
        with open(self.filename) as data:
                lines = [line.split() for line in data]


        i=0
        while lines[i]:
            if lines[i][0]=='Frequency':
                if 'f=' in lines[i][1]:
                    self.frequency1=float((lines[i][1].replace('f=','')))
                    
            elif lines[i][0]=='dz':
                if 'dz=' in lines[i][1]:
                    self.dz=int((lines[i][1].replace('dz=','')))
                    
            elif lines[i][0]=='Wavenumbers':
                if 'n=' in lines[i][1]:
                    self.wavenumbers=int((lines[i][1].replace('n=','')))
                    
            elif lines[i][0]=='Source':
                l=len(lines[i])
                for j in range(1,l):
                    if 'S_medium=' in lines[i][j]:
                        self.S_medium=((lines[i][j].replace('S_medium=','')))
                    elif 'S_depth=' in lines[i][j]: 
                        self.S_depth=int((lines[i][j].replace('S_depth=','')))
                    elif 'S_type=' in lines[i][j]: 
                        self.S_type=((lines[i][j].replace('S_type=','')))
                        
            elif lines[i][0]=='BC-top':
                if 'bc=' in lines[i][1]:
                    self.BCtop=((lines[i][1].replace('bc=','')))
                    
            elif lines[i][0]=='BC-bottom':
                if 'bc=' in lines[i][1]:
                    self.BCbottom=((lines[i][1].replace('bc=','')))
                    
            elif lines[i][0]=='File-name':
                if 'name=' in lines[i][1]:
                    self.Fname=((lines[i][1].replace('name=','')))
                    
            elif lines[i][0]=='C':
                l=len(lines[i])
                for j in range(1,l):
                    if 'Cmin=' in lines[i][j]:
                        self.Cmin=float((lines[i][j].replace('Cmin=','')))
                    elif 'Cmax=' in lines[i][j]: 
                        self.Cmax=float((lines[i][j].replace('Cmax=','')))
                        
            elif lines[i][0]=='Earth':
                l=len(lines[i])
                for j in range(1,l):
                    if 'type=' in lines[i][j]:
                        self.Earth=((lines[i][j].replace('type=','')))
                    elif 'depth=' in lines[i][j]: 
                        self.Earth_depth=int((lines[i][j].replace('depth=','')))
                        
            elif lines[i][0]=='Ocean':
                l=len(lines[i])
                for j in range(1,l):
                    if 'type=' in lines[i][j]:
                        self.Ocean=((lines[i][j].replace('type=','')))
                    elif 'depth=' in lines[i][j]: 
                        self.Ocean_depth=int((lines[i][j].replace('depth=','')))
                if self.Ocean=='non':
                    self.Ocean_depth=0
                    
            elif lines[i][0]=='Atm':
                l=len(lines[i])
                for j in range(1,l):
                    if 'type=' in lines[i][j]:
                        self.Atm=((lines[i][j].replace('type=','')))
                    elif 'depth=' in lines[i][j]: 
                        self.Atm_depth=int((lines[i][j].replace('depth=','')))
                    elif 'direction=' in lines[i][j]: 
                        self.direction=float((lines[i][j].replace('direction=','')))
                    elif 'dcdz=' in lines[i][j]: 
                        self.dcdz=float((lines[i][j].replace('dcdz=','')))
                if self.Atm=='non':
                    self.Atm_depth=0
                if not 'dcdz' in dir(self) and self.Atm=='linear':
                    print('Linear profile must get dc/dz')
                    sys.exit()
                if not 'direction' in dir(self) and self.Atm!='linear' and self.Atm!='homogeneous' and self.Atm!='non':
                    print('Input file must get direction of propagation')
                    sys.exit()
            
            i=i+1
        if self.Atm=='non' and self.Ocean=='non':
                print('One of the acoustic mediums must exist')
                sys.exit()


             
        # self.frequency1=    float(lines[0][1])
        # self.dz=            int(lines[1][1])
        # self.wavenumbers=int(lines[2][1])
        # self.S_medium=   (lines[3][1])
        # self.S_depth=   int(lines[3][2])
        # self.S_type=   (lines[3][3])
        # self.BCtop=     lines[4][1]
        # self.BCbottom=  lines[5][1]
        # self.Fname=     lines[6][1]
        # self.Cmin=      float(lines[7][1])
        # self.Cmax=       float(lines[7][2])
        # self.Earth=           lines[8][1]
        # self.Earth_depth=     int(lines[8][2]) 
        # self.Ocean=           lines[9][1]
        # self.Ocean_depth=     int(lines[9][2])
        # self.Atm=           lines[10][1]
        # self.Atm_depth=     int(lines[10][2]) 
        # if self.Atm=='linear':
        #     self.dcdz=  float(lines[10][3])
        # elif self.Atm!='non' and self.Atm!='linear' and self.Atm!='homogeneous':
        #     self.direction=     float(lines[10][3])



        


        self.depth=self.Earth_depth + self.Ocean_depth + self.Atm_depth

        #calculating other parameters
        self.omega=2*pi*self.frequency1
        self.layers=int(self.depth/self.dz)
        self.z=np.zeros(self.layers,dtype=int)
        for i in range(0,self.layers):
            self.z[i]=i*self.dz
        
        # if self.S_medium=='earth':
        #     self.S_depth=self.Earth_depth - self.S_depth
        #     self.S_layer=(np.argmin(np.abs(self.z[:] - self.S_depth)))
        #     print (self.S_depth,self.S_layer)
        

    def wave_num_and_range(self):
        """
        Calculating Kmin,Kmax,dK,K,Rmax,r,cylin_spred,wavenumbers,dr
        """

        self.Kmin=2*pi*self.frequency1/(self.Cmax)
        self.Kmax=2*pi*self.frequency1/(self.Cmin)
        self.dK=(self.Kmax-self.Kmin)/self.wavenumbers

        while self.wavenumbers*self.dK<=self.Kmin:
            print ('increasing wavenumbers...')
            self.wavenumbers=self.wavenumbers*2
        while self.wavenumbers*self.dK<=self.Kmax:
            print ('increasing wavenumbers...')
            self.wavenumbers=self.wavenumbers*2
        self.dr=pi/(self.wavenumbers*self.dK)


        self.K=np.zeros(self.wavenumbers,dtype=complex)
        self.r=np.zeros(self.wavenumbers,dtype=float)
        for i in range(0,self.wavenumbers):
            self.K[i]=i*self.dK
            self.r[i]=self.dr +i*self.dr

        # print ('max range',np.max(self.r)/1000)
        # plt.figure()
        # plt.plot(self.K,self.K)
        # plt.plot(self.Kmin,0,'.',markersize=10)
        # plt.plot(self.Kmax,0,'.',markersize=10)
        # plt.show()
        
        self.eps=self.dK
        self.K=self.K-1j*self.eps
        self.cylin_spred=np.ndarray(self.wavenumbers,dtype=complex)
        self.cylin_spred=np.exp(-1j*pi/4)/np.sqrt(2*pi*self.r[:])


    def velo_prof(self):
        """
        Create the model velocity profile. The function can get for the atmosphere: "file_name", "homogeneous"=330 m/s.
        Ocean profile can get : "file_name", "homogeneous"=1500 m/s or "Munk" for 5km Munk profile.
        Earth profile can  get "file_name" or name of rock type for homogeneous "Granite" "Sandstone" "Wet-sand" add more... 
        """
        self.Vp=np.zeros(self.layers,dtype=float)
        self.Vs=np.zeros(self.layers,dtype=float)
        self.mu=np.zeros(self.layers,dtype=float)
        self.lamda=np.zeros(self.layers,dtype=float)
        self.rho=np.zeros(self.layers,dtype=float)
        calc_lam= lambda vp,vs,ro: ro*(vp**2-2*vs**2) 
        calc_mu= lambda vs,ro: ro*vs**2  

        #earth-ocean-atmosphere:
        if self.Earth!='non' and self.Ocean!='non' and self.Atm!='non':
            self.earth_interface=int(self.Earth_depth/self.dz)
            self.ocean_interface=self.earth_interface + int(self.Ocean_depth/self.dz)

            if self.Earth=="Granite":
                print ("Granite")
                self.rho[:self.earth_interface]=2700
                self.Vp[:self.earth_interface]=5000
                self.Vs[:self.earth_interface]=2000
                self.lamda[:]=calc_lam(self.Vp,self.Vs,self.rho)
                self.mu[:]=calc_mu(self.Vs,self.rho)

            elif self.Earth=="Wet-sand":
                print ("Wet sand")
                self.rho[:self.earth_interface]=1900; 
                self.Vp[:self.earth_interface]=1500; 
                self.Vs[:self.earth_interface]=400; 
                self.lamda[:]=calc_lam(self.Vp,self.Vs,self.rho)
                self.mu[:]=calc_mu(self.Vs,self.rho)
            

            if self.Ocean=="homogeneous":
                self.Vp[self.earth_interface:self.ocean_interface]=1500
                self.rho[self.earth_interface:self.ocean_interface]=1000

            elif self.Ocean=="munk":
                print ("Ocean Munk profile")
                nz=self.ocean_interface-self.earth_interface
                self.Vp[self.earth_interface:self.ocean_interface]=make_munk(self.Ocean_depth,nz)
                self.rho[self.earth_interface:self.ocean_interface]=1000


            if self.Atm=="homogeneous":
                self.Vp[self.ocean_interface:]=330
                self.rho[self.ocean_interface:]=1

            elif self.Atm=="linear":
                self.Vp[self.ocean_interface:]=330 + self.dcdz*(self.z[self.ocean_interface:]-self.z[self.ocean_interface])
                self.rho[self.ocean_interface:]=1

            else:
                self.Vp[self.ocean_interface:],self.rho[self.ocean_interface:]=read_atm_profile(self.Atm, self.Atm_depth, self.dz, self.direction)

        elif self.Earth!='non' and self.Ocean!='non' and self.Atm=='non':
            self.earth_interface=int(self.Earth_depth/self.dz)

            if self.Earth=="Granite":
                print ("Granite")
                self.rho[:self.earth_interface]=2700
                self.Vp[:self.earth_interface]=5000
                self.Vs[:self.earth_interface]=2000
                self.lamda[:]=calc_lam(self.Vp,self.Vs,self.rho)
                self.mu[:]=calc_mu(self.Vs,self.rho)

            elif self.Earth=="Wet-sand":
                print ("Wet sand")
                self.rho[:self.earth_interface]=1900
                self.Vp[:self.earth_interface]=1500
                self.Vs[:self.earth_interface]=400
                self.lamda[:]=calc_lam(self.Vp,self.Vs,self.rho)
                self.mu[:]=calc_mu(self.Vs,self.rho)
            

            if self.Ocean=="homogeneous":
                self.Vp[self.earth_interface:]=1500
                self.rho[self.earth_interface:]=1000

            elif self.Ocean=="munk":
                print ("Ocean Munk profile")
                nz=self.layers-self.earth_interface
                self.Vp[self.earth_interface:]=make_munk(self.Ocean_depth,nz)
                self.rho[self.earth_interface:]=1000

        elif self.Earth!='non' and self.Ocean=='non' and self.Atm!='non':
            self.earth_interface=int(self.Earth_depth/self.dz)

            if self.Earth=="Granite":
                print ("Granite")
                self.rho[:self.earth_interface]=2700
                self.Vp[:self.earth_interface]=5000
                self.Vs[:self.earth_interface]=20
                # self.rho[:self.earth_interface]=3700
                # self.Vp[:self.earth_interface]=7000
                # self.Vs[:self.earth_interface]=5000
                self.lamda[:]=calc_lam(self.Vp,self.Vs,self.rho)
                self.mu[:]=calc_mu(self.Vs,self.rho)

            elif self.Earth=="Wet-sand":
                print ("Wet sand")
                self.rho[:self.earth_interface]=1900;
                self.Vp[:self.earth_interface]=1500
                self.Vs[:self.earth_interface]=400
                self.lamda[:]=calc_lam(self.Vp,self.Vs,self.rho)
                self.mu[:]=calc_mu(self.Vs,self.rho)
            

            if self.Atm=="homogeneous":
                self.Vp[self.earth_interface:]=330
                self.rho[self.earth_interface:]=1

            elif self.Atm=="linear":
                self.Vp[self.earth_interface:]=330 + self.dcdz*(self.z[self.earth_interface:]-self.z[self.earth_interface])
                self.rho[self.earth_interface:]=1

            else:
                self.Vp[self.earth_interface:],self.rho[self.earth_interface:]=read_atm_profile(self.Atm, self.Atm_depth, self.dz, self.direction)


                


    def print_velocity_profile(self):
        """
        Prints the velocity profile. If Ocean is included, the depth will start at -5Km
        """
        axis_font = {'fontname':'Arial', 'size':'25'}
        title_font = {'fontname':'Arial', 'size':'20'}
        fontsize=10;

        plt.figure()
        plt.plot(self.Vp,self.z/1000.00,label='Vp')
        plt.plot(self.Vs,self.z/1000.00,label='Vs')
        plt.tick_params(axis='y', labelsize=fontsize)
        plt.tick_params(axis='x', labelsize=fontsize)
        plt.xlabel('Velocity [m/s]',**axis_font);
        plt.ylabel('Depth [Km]',**axis_font);
        plt.title('Velocity profile', **title_font)
        plt.xlim([min(self.Vs)-100,max(self.Vp)+100])
        plt.legend()
        plt.show()



##########################################################################################################################



##########################################################################################################################
#####
#MAIN
#####
Initial_parameters=Parameters("input")
Initial_parameters.calc_parameters()
Initial_parameters.wave_num_and_range()
Initial_parameters.velo_prof()
# Initial_parameters.print_velocity_profile()


print ("Get greens")
A=get_greens(Initial_parameters.K,Initial_parameters.Vp,Initial_parameters.Vs,Initial_parameters.lamda,Initial_parameters.mu,Initial_parameters.layers,
        Initial_parameters.wavenumbers,Initial_parameters.omega,Initial_parameters.dz,
            Initial_parameters.S_medium,Initial_parameters.S_depth,Initial_parameters.S_type,
            Initial_parameters.z,Initial_parameters.BCtop,Initial_parameters.BCbottom,Initial_parameters.rho
            ,Initial_parameters.Kmin,Initial_parameters.Kmax,Initial_parameters.dK,Initial_parameters.earth_interface,
            Initial_parameters.Earth_depth,Initial_parameters.Ocean_depth,Initial_parameters.Atm_depth)


print ("go to range")
P,smooth_window=go_to_range(A,Initial_parameters.cylin_spred,Initial_parameters.r,Initial_parameters.dK,
            Initial_parameters.wavenumbers,Initial_parameters.K,Initial_parameters.layers,Initial_parameters.Kmin,Initial_parameters.Kmax,
            Initial_parameters.rho,Initial_parameters.omega,Initial_parameters.eps)



# plt.figure()
# plt.plot(np.real(Initial_parameters.K),FFP,'r',label='FFP')
# plt.plot(np.real(Initial_parameters.K),A_exact[50,:],'g',label='Exact')
# plt.xlim([Initial_parameters.Kmin-0.005,Initial_parameters.Kmax])
# plt.legend()
# plt.show()

# plt.figure()
# plt.title('C1')
# plt.plot(C1_F,'r',label='FFP')
# plt.plot(C1_E,'g',label='Exact')
# plt.legend()
# plt.show()

# plt.figure()
# plt.title('C2')
# plt.plot(C2_F,'r',label='FFP')
# plt.plot(C2_E,'g',label='Exact')
# plt.legend()
# plt.show()

# plt.figure()
# plt.title('C3')
# plt.plot(C3_F,'r',label='FFP')
# plt.plot(C3_E,'g',label='Exact')
# plt.legend()
# plt.show()

# plt.figure()
# plt.title('C4')
# plt.plot(C4_F,'r',label='FFP')
# plt.plot(C4_E,'g',label='Exact')
# plt.legend()
# plt.show()


####################################################################################################
#SAVING TO FILES
###################################################################################################
# np.savetxt(Initial_parameters.Fname+'-depth',Initial_parameters.z,fmt='%.4e')
# np.savetxt(Initial_parameters.Fname+'-Vp',Initial_parameters.Vp,fmt='%.4e')
# np.savetxt(Initial_parameters.Fname+'-Vs',Initial_parameters.Vs,fmt='%.4e')
# np.savetxt(Initial_parameters.Fname+'-range',Initial_parameters.r,fmt='%.4e')
# os.remove('Uzz.nc')
# ncfile = Dataset('Uzz.nc', 'w', format='NETCDF4_CLASSIC')

# Uzz_atts = {'units': 'diss', 'long_name':   'Transmission Loss'}
# z_atts = {'units': 'km', 'long_name':   'Altitude', 'positive': 'up', 'axis': 'Z'}
# r_atts = {'units': 'km', 'long_name':   'Range'}

# # (Initial_parameters.layers, Initial_parameters.wavenumbers) = data['tl'].shape

# ncfile.createDimension('z', Initial_parameters.layers)
# ncfile.createDimension('r', Initial_parameters.wavenumbers)

# print (ncfile.dimensions['z'])

# r_var = ncfile.createVariable('r', np.float32, ('r',))
# z_var = ncfile.createVariable('z', np.float32, ('z',))
# Uzz_var = ncfile.createVariable('diss', np.float64, ('z','r', ))

# print (ncfile.variables['diss'])
# r_var.setncatts(r_atts)
# z_var.setncatts(z_atts)
# Uzz_var.setncatts(Uzz_atts)

# print (Uzz_var.units)

# r_var[:] = Initial_parameters.r/1000
# z_var[:] = Initial_parameters.z/1000
# Uzz_var[:] = np.log10(np.abs(P[:,:]))
# ncfile.close()




