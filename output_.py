import matplotlib.pyplot as plt
import numpy as np
from math import pi
import sys
import csv
# from netCDF4 import Dataset, date2num, num2date
import os, shutil, glob
import obspy
from obspy import Trace, Stream, UTCDateTime






def save_time(P_t, P_f, A_p_f,  time, freq, r, Rec_alt,  K,  Fname):


    loc_i='./'
    loc_f='./Results/'+Fname

    if os.path.isdir(loc_f)==True:
        print('directory exist. removing and recreating')
        shutil.rmtree(loc_f)
        os.mkdir(loc_f)
    if os.path.isdir(loc_f)==False:
        print('creating directory')
        os.mkdir(loc_f)

    loc_f=loc_f+'/'
    shutil.copy(loc_i+'/input-nb', loc_f+'/input-nb')



    now_date=UTCDateTime()
    dt=time[1]-time[0]

    t=Trace(P_t[0,0,:])
    t.stats["starttime"]=now_date
    t.stats["delta"]=dt
    t.stats["npts"]=len(time)
    # t.filter("lowpass", freq=0.5)
    t.stats.sac = obspy.core.AttribDict()
    t.stats.STLA=0
    t.stats.sac.STLA=0

    # t.stats.sac = obspy.core.AttribDict()
    # t.stats.stlat=0
    st=Stream(t)
    for i in range(1,len(r)):
        t=Trace(P_t[0,i,:])
        t.stats["starttime"]=now_date
        t.stats["delta"]=dt
        t.stats["npts"]=len(time)
        # t.filter("lowpass", freq=0.5)
        t.stats.sac = obspy.core.AttribDict()
        t.stats.stlat=0
        t.stats.sac.STLA=0

        st.append(t)
    i=0
    for tr in st:
        tr.stats.sac.dist = r[i]/1000
        i+=1


        # tr.stats.stlon = r[i]
        # tr.stats.sac.STLO = r[i]
        # tr.stats.dist = r[i]/1000

        tr.write("waveforms{0:0>7}.sac".format(i), format='sac')

    #st.write("waveforms", format='sac')

    loc_i='./'
    loc_f='./Results/'+Fname

    for file in glob.glob(loc_i+'waveforms*'):
        shutil.move(file, loc_f)

    # num_f=len(r)
    # name=[0]*num_f
    # for i in range(1,num_f+1):
    #     if i<10:
    #         name[i-1]='waveforms0'+str(i)
    #     else:
    #         name[i-1]='waveforms'+str(i)

    # with open('waveformss_list', 'w') as f:
    #         writer = csv.writer(f, delimiter='\t')
    #         writer.writerows(zip(name, r/1000, r*0))


    U_name='omega-k.nc'

    # os.remove(U_name)

    ncfile = Dataset(U_name, 'w', format='NETCDF4_CLASSIC')

    DR_atts = {'units': 'amplitude', 'long_name':   'Dispersion relation'}
    f_atts = {'units': 'Hz', 'long_name':   'frequency'}
    k_atts = {'units': 'm', 'long_name':   'wavenumber'}

    # (Initial_parameters.layers, Initial_parameters.wavenumbers) = data['tl'].shape

    ncfile.createDimension('f', freq.shape[0])
    ncfile.createDimension('K', K.shape[0])


    f_var = ncfile.createVariable('f', np.float64, ('f',))
    k_var = ncfile.createVariable('K', np.float64, ('K',))
    DR_var = ncfile.createVariable('amplitude', np.float64, ('f', 'K', ))

    f_var.setncatts(f_atts)
    k_var.setncatts(k_atts)
    DR_var.setncatts(DR_atts)




    f_var[:] = freq
    k_var[:] = K
    DR_var[:] = np.transpose(np.abs(A_p_f[0,:,:]))

    ncfile.close()

    os.rename(loc_i+U_name, loc_f+'/'+U_name)


    time_name='time.nc'

    ncfile = Dataset(time_name, 'w', format='NETCDF4_CLASSIC')

    DR_atts = {'units': 'amplitude', 'long_name':   'Dispersion relation'}
    t_atts = {'units': 'sec', 'long_name':   'time'}
    r_atts = {'units': 'm', 'long_name':   'range'}

    # (Initial_parameters.layers, Initial_parameters.wavenumbers) = data['tl'].shape

    ncfile.createDimension('t', time.shape[0])
    ncfile.createDimension('r', r.shape[0])


    t_var = ncfile.createVariable('t', np.float64, ('t',))
    r_var = ncfile.createVariable('r', np.float64, ('r',))
    DR_var = ncfile.createVariable('amplitude', np.float64, ('t', 'r', ))


    f_var.setncatts(f_atts)
    k_var.setncatts(k_atts)
    DR_var.setncatts(DR_atts)




    t_var[:] = time
    r_var[:] = r/1000
    DR_var[:] = np.transpose(P_t[0,:,:])

    ncfile.close()

    os.rename(loc_i+time_name, loc_f+'/'+time_name)







def save_results(A, P, z, r, omega, Vp, K, earth_interface, Earth_depth, ocean_interface, Ocean_depth,
                    smooth_window, wavenumbers, layers, dz, Fname, Rec_alt, rho, S_medium, S_depth):
    #this function gets the Green's functions and pressure, and saves the P, TL and modes for a given altitude alt
    #P and TL are as function of range. Modes are function of phase velocity.

    # loc_i='/Users/gil/Dropbox/study/FFP/seismo-acoustic/parallel/'
    # loc_f='/Users/gil/Dropbox/study/FFP/seismo-acoustic/parallel/'+Fname

    loc_i='./'
    loc_f='./Results/'+Fname

    if os.path.isdir(loc_f)==True:
        print('directory exist. removing and recreating')
        shutil.rmtree(loc_f)
        os.mkdir(loc_f)
    if os.path.isdir(loc_f)==False:
        print('creating directory')
        os.mkdir(loc_f)

    loc_f=loc_f+'/'
    shutil.copy(loc_i+'/input-parameters', loc_f+'/input-parameters')

    pos_0= earth_interface + ocean_interface

    if S_medium=='earth':
        S_layer=earth_interface - int(S_depth/dz)-1

        P_null= rho[S_layer]*(omega**2)*np.exp(1j * omega/Vp[S_layer])/(4*pi)

    elif S_medium=='ocean':
        S_layer=earth_interface + int((Ocean_depth - S_depth)/dz)

        P_null= 1*(omega**2)*np.exp(1j * omega/Vp[S_layer])/(4*pi)

    elif S_medium=='atm':
        S_layer=earth_interface + ocean_interface + int(S_depth/dz)

        P_null= -rho[S_layer]*(omega**2)*np.exp(1j * omega/Vp[S_layer])/(4*pi)

    phases=np.zeros(wavenumbers, dtype=np.float64)
    for i in range(1,wavenumbers):
        phases[i]=omega/np.real(K[i])



    for alt in Rec_alt:


        pos= pos_0 + np.int64(alt/dz)
        alt=np.int64(alt/dz) * dz

        TL_name='TL_alt_'+str(alt)+'.csv'
        P_name='abs-pressure_alt_'+str(alt)+'.csv'
        Modes_name='modes_alt_'+str(alt)+'.csv'
        U_name1='TL_intensity_ref_source.nc'
        U_name2='Red_Pressure_ref_source.nc'
        U_name3='TL_intensity_ref_reciever.nc'
        U_name4='Abs_pressure.nc'


        with open(TL_name, 'w') as f:
            writer = csv.writer(f, delimiter='\t')
            a1 =  np.abs(P[pos,:])/np.sqrt(rho[pos]*Vp[pos])
            a2 =  np.abs(P_null)/np.sqrt(rho[S_layer]*Vp[S_layer])

            writer.writerows(zip(r/1000,20*np.log10(a1/a2)))





        plt.figure(figsize=[15,5])
        plt.plot(r/1000,20*np.log10(a1/a2))
        plt.xlim([0,1000])
        plt.show()


        with open(P_name, 'w') as f:
            writer = csv.writer(f, delimiter='\t')
            writer.writerows(zip(r/1000,np.abs(P[pos,:])))



        with open(Modes_name, 'w') as f:
            writer = csv.writer(f, delimiter='\t')
            writer.writerows(zip(phases,np.abs(A[pos,:])*smooth_window[:wavenumbers]))

        plt.figure(figsize=[15,5])
        plt.plot(phases,np.abs(A[pos,:])*smooth_window[:wavenumbers])
        plt.xlim([300,500])
        plt.show()



        os.rename(loc_i+TL_name, loc_f+TL_name)
        os.rename(loc_i+P_name, loc_f+P_name)
        os.rename(loc_i+Modes_name, loc_f+Modes_name)

    vel_name='Vp.csv'
    with open(vel_name, 'w') as f:
            writer = csv.writer(f, delimiter='\t')
            writer.writerows(zip(z/1000, Vp))

    os.rename(loc_i+vel_name, loc_f+vel_name)

    dens_name='rho.csv'
    with open(dens_name, 'w') as f:
            writer = csv.writer(f, delimiter='\t')
            writer.writerows(zip(z/1000, rho))

    os.rename(loc_i+dens_name, loc_f+dens_name)

    # ----------------------------------------------------------------------------------------
    # TL wave intensity ref source
    # ----------------------------------------------------------------------------------------
    ncfile = Dataset(U_name1, 'w', format='NETCDF4_CLASSIC')

    Uzz_atts = {'units': 'diss', 'long_name':   'Transmission Loss'}
    z_atts = {'units': 'km', 'long_name':   'Altitude', 'positive': 'up', 'axis': 'Z'}
    r_atts = {'units': 'km', 'long_name':   'Range'}

    # (Initial_parameters.layers, Initial_parameters.wavenumbers) = data['tl'].shape

    ncfile.createDimension('z', layers)
    ncfile.createDimension('r', wavenumbers)


    r_var = ncfile.createVariable('r', np.float64, ('r',))
    z_var = ncfile.createVariable('z', np.float64, ('z',))
    Uzz_var = ncfile.createVariable('diss', np.float64, ('z','r', ))

    r_var.setncatts(r_atts)
    z_var.setncatts(z_atts)
    Uzz_var.setncatts(Uzz_atts)


    displacement = P.copy()

    a2 = np.abs(P_null)/np.sqrt(rho[S_layer]*Vp[S_layer])
    for l in range(0, layers):

        a1 =  np.abs(P[l,:])/np.sqrt(rho[l]*Vp[l])

        displacement[l,:]=20*np.log10(a1/a2)


    displacement=np.real(displacement)

    displacement[np.isposinf(displacement)] = -2000
    displacement[np.isneginf(displacement)] = -2000
    displacement[np.isnan(displacement)] = -2000



    r_var[:] = r/1000
    z_var[:] = z/1000 - (Earth_depth + Ocean_depth)/1000
    Uzz_var[:] = displacement

    ncfile.close()


    os.rename(loc_i+U_name1, loc_f+U_name1)
    # ----------------------------------------------------------------------------------------


    # ----------------------------------------------------------------------------------------
    # TL reduced pressure ref source
    # ----------------------------------------------------------------------------------------
    # ncfile = Dataset(U_name2, 'w', format='NETCDF4_CLASSIC')

    # Uzz_atts = {'units': 'diss', 'long_name':   'Transmission Loss'}
    # z_atts = {'units': 'km', 'long_name':   'Altitude', 'positive': 'up', 'axis': 'Z'}
    # r_atts = {'units': 'km', 'long_name':   'Range'}

    # # (Initial_parameters.layers, Initial_parameters.wavenumbers) = data['tl'].shape

    # ncfile.createDimension('z', layers)
    # ncfile.createDimension('r', wavenumbers)


    # r_var = ncfile.createVariable('r', np.float64, ('r',))
    # z_var = ncfile.createVariable('z', np.float64, ('z',))
    # Uzz_var = ncfile.createVariable('diss', np.float64, ('z','r', ))

    # r_var.setncatts(r_atts)
    # z_var.setncatts(z_atts)
    # Uzz_var.setncatts(Uzz_atts)


    # displacement = P.copy()

    # a2 =  np.abs(P_null)/(np.sqrt(rho[S_layer]))
    # for l in range(0, layers):
    #     a1 =  np.abs(P[l,:])/np.sqrt(rho[l])
    #     displacement[l,:]=np.log10(a1/a2)


    # displacement=np.real(displacement)

    # displacement[np.isposinf(displacement)] = -2000
    # displacement[np.isneginf(displacement)] = -2000
    # displacement[np.isnan(displacement)] = -2000



    # r_var[:] = r/1000
    # z_var[:] = z/1000 - (Earth_depth + Ocean_depth)/1000
    # Uzz_var[:] = displacement

    # ncfile.close()


    # os.rename(loc_i+U_name2, loc_f+U_name2)
    # ----------------------------------------------------------------------------------------


    # ----------------------------------------------------------------------------------------
    # TL wave intensity ref reciever
    # ----------------------------------------------------------------------------------------
    # ncfile = Dataset(U_name3, 'w', format='NETCDF4_CLASSIC')

    # Uzz_atts = {'units': 'diss', 'long_name':   'Transmission Loss'}
    # z_atts = {'units': 'km', 'long_name':   'Altitude', 'positive': 'up', 'axis': 'Z'}
    # r_atts = {'units': 'km', 'long_name':   'Range'}

    # # (Initial_parameters.layers, Initial_parameters.wavenumbers) = data['tl'].shape

    # ncfile.createDimension('z', layers)
    # ncfile.createDimension('r', wavenumbers)


    # r_var = ncfile.createVariable('r', np.float64, ('r',))
    # z_var = ncfile.createVariable('z', np.float64, ('z',))
    # Uzz_var = ncfile.createVariable('diss', np.float64, ('z','r', ))

    # r_var.setncatts(r_atts)
    # z_var.setncatts(z_atts)
    # Uzz_var.setncatts(Uzz_atts)


    # displacement = P.copy()

    # a2 = np.abs(P[pos,0])/np.sqrt(rho[pos]*Vp[pos])
    # print(a2, pos, P.min(), P.max(), P.mean())
    # for l in range(0, layers):

    #     a1 =  np.abs(P[l,:])/np.sqrt(rho[l]*Vp[l])

    #     displacement[l,:]=20*np.log10(a1/a2)


    # displacement=np.real(displacement)

    # displacement[np.isposinf(displacement)] = -2000
    # displacement[np.isneginf(displacement)] = -2000
    # displacement[np.isnan(displacement)] = -2000



    # r_var[:] = r/1000
    # z_var[:] = z/1000 - (Earth_depth + Ocean_depth)/1000
    # Uzz_var[:] = displacement

    # ncfile.close()


    # os.rename(loc_i+U_name3, loc_f+U_name3)
    # ----------------------------------------------------------------------------------------


    # ----------------------------------------------------------------------------------------
    # Absolute pressure
    # ----------------------------------------------------------------------------------------
    ncfile = Dataset(U_name4, 'w', format='NETCDF4_CLASSIC')

    Uzz_atts = {'units': 'Pa', 'long_name':   'Absolute pressure'}
    z_atts = {'units': 'km', 'long_name':   'Altitude', 'positive': 'up', 'axis': 'Z'}
    r_atts = {'units': 'km', 'long_name':   'Range'}

    # (Initial_parameters.layers, Initial_parameters.wavenumbers) = data['tl'].shape

    ncfile.createDimension('z', layers)
    ncfile.createDimension('r', wavenumbers)


    r_var = ncfile.createVariable('r', np.float64, ('r',))
    z_var = ncfile.createVariable('z', np.float64, ('z',))
    Uzz_var = ncfile.createVariable('P_abs', np.float64, ('z','r', ))

    r_var.setncatts(r_atts)
    z_var.setncatts(z_atts)
    Uzz_var.setncatts(Uzz_atts)


    displacement = P.copy()

    for l in range(0, layers):

        a1 =  np.abs(P[l,:])#/np.sqrt(rho[l]*Vp[l])

        displacement[l,:]=a1


    displacement=np.real(displacement)

    displacement[np.isposinf(displacement)] = -2000
    displacement[np.isneginf(displacement)] = -2000
    displacement[np.isnan(displacement)] = -2000



    r_var[:] = r/1000
    z_var[:] = z/1000 - (Earth_depth + Ocean_depth)/1000
    Uzz_var[:] = displacement

    ncfile.close()


    os.rename(loc_i+U_name4, loc_f+U_name4)
    # ----------------------------------------------------------------------------------------





def save_results_inversion(A, P, z, r, omega, K, earth_interface, Earth_depth, ocean_interface, Ocean_depth, smooth_window,
                            wavenumbers, layers, dz, Fname, Rec_alt, Zs, fre, direction):
    #this function gets the Green's functions and pressure, and saves the P, TL and modes for a given altitude alt
    #P and TL are as function of range. Modes are function of phase velocity.


    loc_i='/Users/gil/Dropbox/study/FFP/seismo-acoustic/estimating-source-depth/Toy_profile/'
    loc_f='/Users/gil/Dropbox/study/FFP/seismo-acoustic/estimating-source-depth/Toy_profile/TL_data/'

    pos_0= earth_interface + ocean_interface
    print ('pos 0 ', pos_0)
    phases=np.zeros(wavenumbers, dtype=np.float64)
    for i in range(1,wavenumbers):
        phases[i]=omega/np.real(K[i])


    for alt in Rec_alt:


        pos= pos_0 + np.int64(alt/dz)
        alt=np.int64(alt/dz) * dz

        print('layer to save', pos)

        # TL_name=Fname+'_TL_alt_'+str(alt)+'.csv'
        # P_name=Fname+'_abs-pressure_alt_'+str(alt)+'.csv'
        # Modes_name=Fname+'_modes_alt_'+str(alt)+'.csv'

        TL_name=Fname+'_TL_Zs_'+'{0:0>5}'.format(Zs)+'_f_'+'{0:0>5.1f}'.format(fre)+'.csv'
        P_name=Fname+'_P_Zs_'+'{0:0>5}'.format(Zs)+'_f_'+'{0:0>5.1f}'.format(fre)+'.csv'
        Modes_name=Fname+'_modes_Z_s_'+'{0:0>5}'.format(Zs)+'_f_'+'{0:0>5.1f}'.format(fre)+'.csv'

        # with open(TL_name, 'w') as f:
        #     writer = csv.writer(f, delimiter='\t')
        #     writer.writerows(zip(r/1000,20*np.log10(np.abs(P[pos,:])/(4*pi))))

        # with open(P_name, 'w') as f:
        #     writer = csv.writer(f, delimiter='\t')
        #     writer.writerows(zip(r/1000,np.abs(P[pos,:])))

        # with open(Modes_name, 'w') as f:
        #     writer = csv.writer(f, delimiter='\t')
        #     writer.writerows(zip(phases,np.abs(A[pos,:])*smooth_window[:wavenumbers]))

        # os.rename(loc_i+TL_name, loc_f+TL_name)
        # os.rename(loc_i+P_name, loc_f+P_name)
        # os.rename(loc_i+Modes_name, loc_f+Modes_name)


        # P_name=Fname+'_P_Zs_'+'{0:0>5}'.format(Zs)+'_f_'+'{0:0>5.1f}'.format(fre)+'.csv'
        P_name='TL_az'+str(int(direction))+'_s'+str(Zs)+'_f'+str(fre)+'_.csv'
        print(P_name)
        with open(P_name, 'w') as f:
            writer = csv.writer(f, delimiter='\t')
            writer.writerows(zip(r/1000,np.abs(P[pos,:])))

        os.rename(loc_i+P_name, loc_f+P_name)



    # plt.figure(figsize=[15,5])
    # plt.plot(r/1000,20*np.log10(np.abs(P[pos,:])/(4*pi)))
    # plt.xlim([0,1000])
    # plt.show()
