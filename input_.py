import matplotlib.pyplot as plt
import numpy as np 
import math
from math import pi, sin, cos, radians, sqrt, acos, degrees
import scipy
from scipy import signal, interpolate
import sys

#--------------------------------------------------------------------------------------------------
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

            if lines[i][0]=='Simulation':
                if 'type=' in lines[i][1]:
                    self.simulation_type=(lines[i][1].replace('type=',''))

            if lines[i][0]=='Frequency':
                if 'f=' in lines[i][1]:
                    self.frequency1=np.float32((lines[i][1].replace('f=','')))
                    
            elif lines[i][0]=='dz':
                if 'dz=' in lines[i][1]:
                    self.dz=np.int32((lines[i][1].replace('dz=','')))
                    
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

                if self.Fname=='exact':
                    self.exact=((lines[i][2].replace('type=','')))
                    
            elif lines[i][0]=='C':
                l=len(lines[i])
                for j in range(1,l):
                    if 'Cmin=' in lines[i][j]:
                        self.Cmin=np.float32((lines[i][j].replace('Cmin=','')))
                    elif 'Cmax=' in lines[i][j]: 
                        self.Cmax=np.float32((lines[i][j].replace('Cmax=','')))
                        
            elif lines[i][0]=='Earth':
                l=len(lines[i])
                for j in range(1,l):
                    if 'type=' in lines[i][j]:
                        self.Earth=((lines[i][j].replace('type=','')))
                    elif 'depth=' in lines[i][j]: 
                        self.Earth_depth=np.int32((lines[i][j].replace('depth=','')))
                    elif 'Attenuation=' in lines[i][j]: 
                        self.Earth_attenuation=(lines[i][j].replace('Attenuation=',''))
                if self.Earth=='non':
                    self.Earth_depth=np.int32(0)

                        
            elif lines[i][0]=='Ocean':
                l=len(lines[i])
                for j in range(1,l):
                    if 'type=' in lines[i][j]:
                        self.Ocean=((lines[i][j].replace('type=','')))
                    elif 'depth=' in lines[i][j]: 
                        self.Ocean_depth=np.int32((lines[i][j].replace('depth=','')))
                if self.Ocean=='non':
                    self.Ocean_depth=np.int32(0)
                    
            elif lines[i][0]=='Atm':
                l=len(lines[i])
                for j in range(1,l):
                    if 'ATM_type=' in lines[i][j]:
                        self.Atm=((lines[i][j].replace('ATM_type=','')))
                    elif 'depth=' in lines[i][j]: 
                        self.Atm_depth=np.int32((lines[i][j].replace('depth=','')))
                    elif 'direction=' in lines[i][j]: 
                        self.direction=float((lines[i][j].replace('direction=','')))
                    elif 'dcdz=' in lines[i][j]: 
                        self.dcdz=np.float32((lines[i][j].replace('dcdz=','')))
                if self.Atm=='non':
                    self.Atm_depth=np.int32(0)
                if not 'dcdz' in dir(self) and self.Atm=='linear':
                    print('Linear profile must get dc/dz')
                    sys.exit()
                if not 'direction' in dir(self) and self.Atm!='linear' and self.Atm!='homogeneous' and self.Atm!='non':
                    print('Input file must get direction of propagation')
                    sys.exit()

            elif lines[i][0]=='Receiver':
                l=len(lines[i])
                for j in range(1,l):
                    if 'Rec_alt=' in lines[i][j]:
                        self.Rec_alt=((lines[i][j].replace('Rec_alt=','')))
                        self.Rec_alt=np.fromstring(self.Rec_alt, dtype=np.int32, sep=',')


            
            i=i+1
        if self.Atm=='non' and self.Ocean=='non' and self.Earth=='non':
                print('One of the acoustic mediums must exist')
                sys.exit()


            



        


        self.depth=self.Earth_depth + self.Ocean_depth + self.Atm_depth

        #calculating other parameters
        self.omega=np.float32(2*pi*self.frequency1)
        self.layers=np.int32(self.depth/self.dz)
        self.z=np.zeros(self.layers,dtype=np.int32)
        for i in range(0,self.layers):
            self.z[i]=i*self.dz
        
        

    def wave_num_and_range(self):
        """
        Calculating Kmin,Kmax,dK,K,Rmax,r,cylin_spred,wavenumbers,dr
        """

        self.Kmin=np.float32(2*pi*self.frequency1/(self.Cmax))
        self.Kmax=np.float32(2*pi*self.frequency1/(self.Cmin))
        self.dK=np.float32((self.Kmax-self.Kmin)/self.wavenumbers)

        while self.wavenumbers*self.dK<=self.Kmin:
            self.wavenumbers=self.wavenumbers*2
        while self.wavenumbers*self.dK<=self.Kmax:
            self.wavenumbers=self.wavenumbers*2
        self.dr=np.float32(pi/(self.wavenumbers*self.dK))

        self.wavenumbers=np.int32(self.wavenumbers)

        self.K=np.zeros(self.wavenumbers,dtype=np.complex64)
        self.r=np.zeros(self.wavenumbers,dtype=np.float32)
        for i in range(0,self.wavenumbers):
            self.K[i]=i*self.dK
            self.r[i]=self.dr +i*self.dr

        # print ('max range',np.max(self.r)/1000)
        # plt.figure()
        # plt.plot(self.K,self.K)
        # plt.plot(self.Kmin,0,'.',markersize=10)
        # plt.plot(self.Kmax,0,'.',markersize=10)
        # plt.show()
        
        self.eps=self.dK*4
        self.K=self.K-1j*self.eps
        self.cylin_spred=np.ndarray(self.wavenumbers,dtype=np.complex64)
        self.cylin_spred=np.exp(-1j*pi/4)/np.sqrt(2*pi*self.r[:])


    def wave_num_and_range_broadband(self):
        """
        Calculating Kmin,Kmax,dK,K,Rmax,r,cylin_spred,wavenumbers,dr
        """
        self.max_frequency=self.frequency1*10
        self.Kmin=np.float32(0.0/(self.Cmax))
        self.Kmax=np.float32(2*pi*self.max_frequency/(self.Cmin))
        self.dK=np.float32((self.Kmax-self.Kmin)/self.wavenumbers)

        while self.wavenumbers*self.dK<=self.Kmin:
            self.wavenumbers=self.wavenumbers*2
        while self.wavenumbers*self.dK<=self.Kmax:
            self.wavenumbers=self.wavenumbers*2
        self.dr=np.float32(pi/(self.wavenumbers*self.dK))

        self.wavenumbers=np.int32(self.wavenumbers)

        self.K=np.zeros(self.wavenumbers,dtype=np.complex64)
        self.r=np.zeros(self.wavenumbers,dtype=np.float32)
        for i in range(0,self.wavenumbers):
            self.K[i]=i*self.dK
            self.r[i]=self.dr +i*self.dr

        print('max range is ', self.r[-1]/1000 )


        self.num_of_freq=np.int32(self.wavenumbers)

        self.df=(self.max_frequency)/self.num_of_freq

        while self.num_of_freq*self.df<=0:
            print ('increasing wavenumbers...')
            self.num_of_freq=self.num_of_freq*2
        while self.num_of_freq*self.df<=self.max_frequency:
            print ('increasing wavenumbers...')
            self.num_of_freq=self.num_of_freq*2

        self.num_of_freq=self.num_of_freq*2
        self.dt=1/(self.num_of_freq*self.df*2)

        print ('dt is', self.dt)

        self.freq=np.zeros(self.num_of_freq,dtype=np.float32)
        self.time=np.zeros(self.num_of_freq,dtype=np.float32)
        for i in range(0,self.num_of_freq):
            self.freq[i]=i*self.df
            self.time[i]=self.dt +i*self.dt

        print('max time is ', self.time[-1] )

        self.omega=2*pi*self.freq 
        
        self.eps=self.dK*4
        self.K=self.K-1j*self.eps
        self.cylin_spred=np.ndarray(self.wavenumbers,dtype=np.complex64)
        self.cylin_spred=np.exp(-1j*pi/4)/np.sqrt(2*pi*self.r[:])


    def velo_prof(self):
        """
        Create the model velocity profile. The function can get for the atmosphere: "file_name", "homogeneous"=330 m/s.
        Ocean profile can get : "file_name", "homogeneous"=1500 m/s or "Munk" for 5km Munk profile.
        Earth profile can  get "file_name" or name of rock type for homogeneous "Granite" "Sandstone" "Wet-sand" add more... 
        """
        self.Vp=np.zeros(self.layers,dtype=np.float32)
        self.Vs=np.zeros(self.layers,dtype=np.float32)
        self.mu=np.zeros(self.layers,dtype=np.complex64)
        self.lamda=np.zeros(self.layers,dtype=np.complex64)
        self.rho=np.zeros(self.layers,dtype=np.float32)
        calc_lam= lambda vp,vs,ro: ro*(vp**2-2*vs**2) 
        calc_mu= lambda vs,ro: ro*vs**2  

        #earth-ocean-atmosphere:
        if self.Earth!='non' and self.Ocean!='non' and self.Atm!='non':
            print('earth-ocean-atmosphere')
            self.earth_interface=np.int32(self.Earth_depth/self.dz)
            self.ocean_interface=self.earth_interface + int(self.Ocean_depth/self.dz)

            if self.Earth=="Granite":
                self.rho[:self.earth_interface]=2700
                self.Vp[:self.earth_interface]=5000
                self.Vs[:self.earth_interface]=2000
                if self.Earth_attenuation=="off": 
                    self.lamda[:]=calc_lam(self.Vp,self.Vs,self.rho)
                    self.mu[:]=calc_mu(self.Vs,self.rho)
                elif self.Earth_attenuation=="on":
                    self.lamda[:]=calc_lam(self.Vp,self.Vs,self.rho)
                    self.mu[:]=calc_mu(self.Vs,self.rho)
                    self.Q_mu, self.Q_lamda, self.delta_Kp, self.delta_Ks= calc_attenuation_parameters(self.Vs, self.mu, self.lamda, self.earth_interface , self.layers)
                    self.mu[:]= self.mu[:] - 1j*self.Q_mu[:]
                    self.lamda[:]= self.lamda[:] - 1j*self.Q_lamda[:]

            elif self.Earth=="Wet-sand":
                self.rho[:self.earth_interface]=1900; 
                self.Vp[:self.earth_interface]=1500; 
                self.Vs[:self.earth_interface]=600; 
                if self.Earth_attenuation=="off": 
                    self.lamda[:]=calc_lam(self.Vp,self.Vs,self.rho)
                    self.mu[:]=calc_mu(self.Vs,self.rho)
                elif self.Earth_attenuation=="on":
                    self.lamda[:]=calc_lam(self.Vp,self.Vs,self.rho)
                    self.mu[:]=calc_mu(self.Vs,self.rho)
                    self.Q_mu, self.Q_lamda, self.delta_Kp, self.delta_Ks= calc_attenuation_parameters(self.Vs, self.mu, self.lamda, self.earth_interface , self.layers)
                    self.mu[:]= self.mu[:] - 1j*self.Q_mu[:]
                    self.lamda[:]= self.lamda[:] - 1j*self.Q_lamda[:]
            

            if self.Ocean=="homogeneous":
                self.Vp[self.earth_interface:self.ocean_interface]=1500
                self.rho[self.earth_interface:self.ocean_interface]=1000

            elif self.Ocean=="munk":
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


        #earth-ocean
        elif self.Earth!='non' and self.Ocean!='non' and self.Atm=='non':
            print('earth-ocean')
            self.earth_interface=int(self.Earth_depth/self.dz)
            self.ocean_interface=self.earth_interface + int(self.Ocean_depth/self.dz)

            if self.Earth=="Granite":
                self.rho[:self.earth_interface]=2700
                self.Vp[:self.earth_interface]=5000
                self.Vs[:self.earth_interface]=2000
                if self.Earth_attenuation=="off": 
                    self.lamda[:]=calc_lam(self.Vp,self.Vs,self.rho)
                    self.mu[:]=calc_mu(self.Vs,self.rho)
                elif self.Earth_attenuation=="on":
                    self.lamda[:]=calc_lam(self.Vp,self.Vs,self.rho)
                    self.mu[:]=calc_mu(self.Vs,self.rho)
                    self.Q_mu, self.Q_lamda, self.delta_Kp, self.delta_Ks= calc_attenuation_parameters(self.Vs, self.mu, self.lamda, self.earth_interface , self.layers)
                    self.mu[:]= self.mu[:] - 1j*self.Q_mu[:]
                    self.lamda[:]= self.lamda[:] - 1j*self.Q_lamda[:]

            elif self.Earth=="Wet-sand":
                self.rho[:self.earth_interface]=1900
                self.Vp[:self.earth_interface]=1500
                self.Vs[:self.earth_interface]=600
                if self.Earth_attenuation=="off": 
                    self.lamda[:]=calc_lam(self.Vp,self.Vs,self.rho)
                    self.mu[:]=calc_mu(self.Vs,self.rho)
                elif self.Earth_attenuation=="on":
                    self.lamda[:]=calc_lam(self.Vp,self.Vs,self.rho)
                    self.mu[:]=calc_mu(self.Vs,self.rho)
                    self.Q_mu, self.Q_lamda, self.delta_Kp, self.delta_Ks= calc_attenuation_parameters(self.Vs, self.mu, self.lamda, self.earth_interface , self.layers)
                    self.mu[:]= self.mu[:] - 1j*self.Q_mu[:]
                    self.lamda[:]= self.lamda[:] - 1j*self.Q_lamda[:]
            

            if self.Ocean=="homogeneous":
                self.Vp[self.earth_interface:]=1500
                self.rho[self.earth_interface:]=1000

            elif self.Ocean=="munk":
                nz=self.layers-self.earth_interface
                self.Vp[self.earth_interface:]=make_munk(self.Ocean_depth,nz)
                self.rho[self.earth_interface:]=1000

        #earth-atmosphere
        elif self.Earth!='non' and self.Ocean=='non' and self.Atm!='non':
            print('earth-atmosphere')
            self.earth_interface=np.int32(self.Earth_depth/self.dz)
            self.ocean_interface=np.int32(0)

            if self.Earth=="Granite":
                self.rho[:self.earth_interface]=2700
                self.Vp[:self.earth_interface]=5000
                self.Vs[:self.earth_interface]=2000
                if self.Earth_attenuation=="off": 
                    self.lamda[:]=calc_lam(self.Vp,self.Vs,self.rho)
                    self.mu[:]=calc_mu(self.Vs,self.rho)
                elif self.Earth_attenuation=="on":
                    self.lamda[:]=calc_lam(self.Vp,self.Vs,self.rho)
                    self.mu[:]=calc_mu(self.Vs,self.rho)
                    self.Q_mu, self.Q_lamda, self.delta_Kp, self.delta_Ks= calc_attenuation_parameters(self.Vs, self.mu, self.lamda, self.earth_interface , self.layers)
                    self.mu[:]= self.mu[:] - 1j*self.Q_mu[:]
                    self.lamda[:]= self.lamda[:] - 1j*self.Q_lamda[:]

            elif self.Earth=="Wet-sand":
                self.rho[:self.earth_interface]=1900;
                self.Vp[:self.earth_interface]=1500
                self.Vs[:self.earth_interface]=600
                if self.Earth_attenuation=="off": 
                    self.lamda[:]=calc_lam(self.Vp,self.Vs,self.rho)
                    self.mu[:]=calc_mu(self.Vs,self.rho)
                elif self.Earth_attenuation=="on":
                    self.lamda[:]=calc_lam(self.Vp,self.Vs,self.rho)
                    self.mu[:]=calc_mu(self.Vs,self.rho)
                    self.Q_mu, self.Q_lamda, self.delta_Kp, self.delta_Ks= calc_attenuation_parameters(self.Vs, self.mu, self.lamda, self.earth_interface , self.layers)
                    self.mu[:]= self.mu[:] - 1j*self.Q_mu[:]
                    self.lamda[:]= self.lamda[:] - 1j*self.Q_lamda[:]
            
            else:
                self.rho[:self.earth_interface], self.Vp[:self.earth_interface], self.Vs[:self.earth_interface] = read_elastic_profile(self.Earth, self.Earth_depth, self.dz)
                if self.Earth_attenuation=="off": 
                    self.lamda[:]=calc_lam(self.Vp,self.Vs,self.rho)
                    self.mu[:]=calc_mu(self.Vs,self.rho)
                elif self.Earth_attenuation=="on":
                    self.lamda[:]=calc_lam(self.Vp,self.Vs,self.rho)
                    self.mu[:]=calc_mu(self.Vs,self.rho)
                    self.Q_mu, self.Q_lamda, self.delta_Kp, self.delta_Ks= calc_attenuation_parameters(self.Vs, self.mu, self.lamda, self.earth_interface , self.layers)
                    self.mu[:]= self.mu[:] - 1j*self.Q_mu[:]
                    self.lamda[:]= self.lamda[:] - 1j*self.Q_lamda[:]

            if self.Atm=="homogeneous":
                self.Vp[self.earth_interface:]=330
                self.rho[self.earth_interface:]=1

            elif self.Atm=="linear":
                self.Vp[self.earth_interface:]=330 + self.dcdz*(self.z[self.earth_interface:]-self.z[self.earth_interface])
                self.rho[self.earth_interface:]=1

            elif self.Atm=="N2":
                a=3.5e-11
                b=4e-6
                self.rho[self.earth_interface:]=1
                temp=self.z[self.earth_interface:]-self.z[self.earth_interface]
                temp=temp[::-1]
                self.Vp[self.earth_interface:]=np.sqrt(1/(a*temp + b))

            else:
                self.Vp[self.earth_interface:],self.rho[self.earth_interface:]=read_atm_profile(self.Atm, self.Atm_depth, self.dz, self.direction)



        #ocean-atmosphere
        elif self.Earth=='non' and self.Ocean!='non' and self.Atm!='non':
            print('ocean-atmosphere')
            self.earth_interface=np.int32(0)
            self.ocean_interface=np.int32(self.Ocean_depth/self.dz)

            if self.Ocean=="homogeneous":
                self.Vp[:self.ocean_interface]=1500
                self.rho[:self.ocean_interface]=1000

            elif self.Ocean=="munk":
                nz=self.ocean_interface
                self.Vp[:self.ocean_interface]=make_munk(self.Ocean_depth,nz)
                self.rho[:self.ocean_interface]=1000
            

            if self.Atm=="homogeneous":
                self.Vp[self.ocean_interface:]=330
                self.rho[self.ocean_interface:]=1

            elif self.Atm=="linear":
                self.Vp[self.ocean_interface:]=330 + self.dcdz*(self.z[self.earth_interface:]-self.z[self.earth_interface])
                self.rho[self.ocean_interface:]=1

            elif self.Atm=="N2":
                a=3.5e-11
                b=4e-6
                self.rho[self.ocean_interface:]=1
                temp=self.z[self.ocean_interface:]-self.z[self.ocean_interface]
                temp=temp[::-1]
                self.Vp[self.ocean_interface:]=np.sqrt(1/(a*temp + b))

            else:
                self.Vp[self.ocean_interface:],self.rho[self.ocean_interface:]=read_atm_profile(self.Atm, self.Atm_depth, self.dz, self.direction)


        #atmosphere
        elif self.Earth=='non' and self.Ocean=='non' and self.Atm!='non':
            print('only atmosphere')

            self.earth_interface=np.int32(0)
            self.ocean_interface=np.int32(0)

            if self.Atm=="homogeneous":
                self.Vp[:]=330
                self.rho[:]=1

            elif self.Atm=="linear":
                self.Vp[:]=330 + self.dcdz*(self.z[self.earth_interface:]-self.z[self.earth_interface])
                self.rho[:]=1

            elif self.Atm=="N2":
                a=3.5e-11
                b=4e-6
                self.rho[:]=1
                temp=self.z[:]-self.z[self.earth_interface]
                temp=temp[::-1]
                self.Vp[:]=np.sqrt(1/(a*temp + b))

            else:
                self.Vp[:],self.rho[self.earth_interface:]=read_atm_profile(self.Atm, self.Atm_depth, self.dz, self.direction)

        #earth
        elif self.Earth!='non' and self.Ocean=='non' and self.Atm=='non':
            print('earth-atmosphere')
            self.earth_interface=np.int32(self.Earth_depth/self.dz)

            if self.Earth=="Granite":
                self.rho[:self.earth_interface]=2700
                self.Vp[:self.earth_interface]=5000
                self.Vs[:self.earth_interface]=2000
                if self.Earth_attenuation=="off": 
                    self.lamda[:]=calc_lam(self.Vp,self.Vs,self.rho)
                    self.mu[:]=calc_mu(self.Vs,self.rho)
                elif self.Earth_attenuation=="on":
                    self.lamda[:]=calc_lam(self.Vp,self.Vs,self.rho)
                    self.mu[:]=calc_mu(self.Vs,self.rho)
                    self.Q_mu, self.Q_lamda, self.delta_Kp, self.delta_Ks= calc_attenuation_parameters(self.Vs, self.mu, self.lamda, self.earth_interface , self.layers)
                    self.mu[:]= self.mu[:] - 1j*self.Q_mu[:]
                    self.lamda[:]= self.lamda[:] - 1j*self.Q_lamda[:]

            elif self.Earth=="Wet-sand":
                self.rho[:self.earth_interface]=1900;
                self.Vp[:self.earth_interface]=1500
                self.Vs[:self.earth_interface]=600
                if self.Earth_attenuation=="off": 
                    self.lamda[:]=calc_lam(self.Vp,self.Vs,self.rho)
                    self.mu[:]=calc_mu(self.Vs,self.rho)
                elif self.Earth_attenuation=="on":
                    self.lamda[:]=calc_lam(self.Vp,self.Vs,self.rho)
                    self.mu[:]=calc_mu(self.Vs,self.rho)
                    self.Q_mu, self.Q_lamda, self.delta_Kp, self.delta_Ks= calc_attenuation_parameters(self.Vs, self.mu, self.lamda, self.earth_interface , self.layers)
                    self.mu[:]= self.mu[:] - 1j*self.Q_mu[:]
                    self.lamda[:]= self.lamda[:] - 1j*self.Q_lamda[:]
                
        if self.Earth_attenuation=="off":
            self.delta_Kp=np.zeros(self.layers,dtype=np.float32)
            self.delta_Ks=np.zeros(self.layers,dtype=np.float32)

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







#VELOCITY PROFILE FUNCTIONS
#########################################################################
def read_atm_profile(profile_name,atm_depth,dz,direction):
    #reading atmopsher conditions
    layers=int(atm_depth/dz)
    z=np.zeros(layers,dtype=np.int32)
    for i in range(0,layers):
        z[i]=i*dz

    parameters=np.loadtxt(profile_name)
    l=len(parameters)
    alt=np.ndarray(l,dtype=np.float32)
    wind_zonal=np.ndarray(l,dtype=np.float32) #west-east direction
    wind_merid=np.ndarray(l,dtype=np.float32) #north-south direct
    temp=np.ndarray(l,dtype=np.float32)
    vel_adia=np.ndarray(l,dtype=np.float32)
    vel_effec=np.ndarray(l,dtype=np.float32)
    vel_interp=np.ndarray(layers,dtype=np.float32)
    rho=np.ndarray(l,dtype=np.float32)
    rho_interp=np.ndarray(layers,dtype=np.float32)

    

    for i in range(0,l):
        alt[i]=parameters[i,0]
        wind_zonal[i]=parameters[i,1]
        wind_merid[i]=parameters[i,2]
        vel_adia[i]=20.05 * np.sqrt(parameters[i,4]) 
        rho[i]=parameters[i,5]*1000


    alt[:]=alt[:] - alt[0]

    phi=radians(direction)
    vel_effec=vel_adia + sin(phi)*wind_zonal + cos(phi)*wind_merid


    a=scipy.interpolate.PchipInterpolator(alt,vel_effec, extrapolate=True)
    vel_interp=a(z/float(1000))
    a=scipy.interpolate.PchipInterpolator(alt,rho, extrapolate=True)
    rho_interp=a(z/float(1000))


    plt.figure()
    plt.plot(vel_interp,z/1000,'r',linewidth=3.0)
    plt.plot(vel_adia,alt,'g')
    plt.show()

    




    return vel_interp,rho_interp

#--------------------------------------------------------------------------------------------------
#--------------------------------------------------------------------------------------------------
def make_munk(ocean_depth,nz):
    # # Munk profile
    c0=np.float32(1500)
    eps=np.float32(0.00737)
    z_water= np.linspace(0,ocean_depth,nz) # grid coordinates
    x=np.linspace(0,ocean_depth,nz)
    temp=np.linspace(0,ocean_depth,nz)
    x[:]=2*(z_water[:]-1300)/1300;  
    temp=c0*(1+eps*(x[:]-1+np.exp(-x[:])))

    return temp[::-1]

#--------------------------------------------------------------------------------------------------
#--------------------------------------------------------------------------------------------------

def calc_attenuation_parameters(Vs, mu, lamda, earth_interface , layers):

    Q_p=np.zeros(layers,dtype=np.float32)
    Q_s=np.zeros(layers,dtype=np.float32)
    Q_mu=np.zeros(layers,dtype=np.float32)
    Q_lamda=np.zeros(layers,dtype=np.float32)
    delta_Kp=np.zeros(layers,dtype=np.float32)
    delta_Ks=np.zeros(layers,dtype=np.float32)

    for i in range(0,earth_interface):
        Q_s[i], Q_p[i]= get_qp_qs(Vs[i]/1000)

        delta_Kp[i]= 1/(2*Q_p[i])
        delta_Ks[i]= 1/(2*Q_s[i])

        Q_mu[i]= np.real(mu[i])/Q_s[i]
        Q_lamda[i]= (np.real(lamda[i])+2*np.real(mu[i]))/Q_p[i] - 2*Q_mu[i]


    return Q_mu, Q_lamda, delta_Kp, delta_Ks

def get_qp_qs(vs):
    """
    Calculate Shear-wave Quality Factor based on Brocher (2008).
    .. note:: If Shear-wave velocity is less-than 0.3 km/s, Shear-wave
              Quality Factor is set to 13.
    Parameters
    ----------
    vs : float or sequence
        Shear-wave velocity in km/s.
    Returns
    -------
    float or sequence
        Shear-wave Quality Factor.
    """
    qs = (-16
          + 104.13 * vs
          - 25.225 * vs**2
          + 8.2184 * vs**3)
    try:
        qs[vs < 0.3] = 13
    except TypeError:
        if vs < 0.3:
            qs = 13
    qp=2*qs

    return qs,qp

#-------------------------------------------------------------------------------------------------------------

def get_rho(vp):
    """
    Calculate :math:`\\rho` (density) based on Brocher (2008).

    Parameters
    ----------
    vp : float or sequence
        Pressure-wave velocity in km/s.

    Returns
    -------
    float or sequence
        :math:`\\rho` (density) in gr/cm^3.
    """
    rho = (1.6612 * vp
           - 0.4721 * vp**2
           + 0.0671 * vp**3
           - 0.0043 * vp**4
           + 0.000106 * vp**5)
    return rho


def read_elastic_profile(prof, depth, dz):

    
    parameters=np.loadtxt(prof)
    
    Depth= parameters[:,0] * 1e3
    P= parameters[:,1]
    S= parameters[:,2]
    
    Z= np.arange(0,Depth[-1]+dz, dz, dtype=np.int32)
    Rho=np.zeros(Z.size, dtype=np.float32)
    Vp=np.zeros(Z.size, dtype=np.float32)
    Vs=np.zeros(Z.size, dtype=np.float32)
    
    pos1=0
    pos2=0
    for i in range(0,len(Depth)-1):
        
        pos2=int(Depth[i+1]/dz)
        
        Vp[pos1:pos2]= np.repeat(P[i], pos2-pos1)
        Vs[pos1:pos2]= np.repeat(S[i], pos2-pos1)
        
        
        pos1=pos2
        
    Vp[pos1:]= np.repeat(P[i+1], Z.size- pos1)
    Vs[pos1:]= np.repeat(S[i+1], Z.size- pos1)
    Rho= get_rho(Vp) * 1000
    
    num_layers= int(depth/dz)
    
    Rho_out=np.zeros(num_layers, dtype=np.float32)
    Vp_out=np.zeros(num_layers, dtype=np.float32)
    Vs_out=np.zeros(num_layers, dtype=np.float32)
    
    Rho_out = Rho[:num_layers]
    Vp_out  = Vp[:num_layers] 
    Vs_out  = Vs[:num_layers]
    
    
    # plt.figure(figsize=(5, 10))
    # plt.plot(parameters[:,1], parameters[:,0],'b.', markersize=12)
    # plt.plot(Vp, Z/1000,'r.', markersize=5 )
    # plt.plot(Vp_out, Z[:num_layers]/1000,'g.', markersize=2 )
    # plt.gca().invert_yaxis()  
    # plt.xlim([5,10])
    # plt.show()

    # plt.figure(figsize=(5, 10))
    # plt.plot(parameters[:,2], parameters[:,0],'b.', markersize=12)
    # plt.plot(Vs, Z/1000,'r.', markersize=5 )
    # plt.plot(Vs_out, Z[:num_layers]/1000,'g.', markersize=2 )
    # plt.gca().invert_yaxis()  
    # plt.xlim([2,5])
    # plt.show()

    # plt.figure(figsize=(5, 10))
    # plt.plot(Rho, Z/1000,'r.', markersize=5 )
    # plt.plot(Rho_out, Z[:num_layers]/1000,'g.', markersize=2 )
    # plt.gca().invert_yaxis()  
    # # plt.xlim([5,10])
    # plt.show()
    

    return Rho_out[::-1] , Vp_out[::-1]*1e3, Vs_out[::-1]*1e3

