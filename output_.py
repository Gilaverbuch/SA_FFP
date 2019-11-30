import matplotlib.pyplot as plt
import numpy as np 
from math import pi
import sys
import csv
from netCDF4 import Dataset, date2num, num2date
import os, shutil, glob
import obspy
from obspy import Trace, Stream, UTCDateTime




def save_time(P_t, P_f, A_p_f,  time, freq, r, Rec_alt,  K,  Fname):


    loc_i='./'
    loc_f='./'+Fname
    
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
    loc_f='./'+Fname

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







def save_results(A, P, z, r, omega, Vp, K, earth_interface, Earth_depth, ocean_interface, Ocean_depth, smooth_window, wavenumbers, layers, dz, Fname, Rec_alt):
    #this function gets the Green's functions and pressure, and saves the P, TL and modes for a given altitude alt
    #P and TL are as function of range. Modes are function of phase velocity.

    # loc_i='/Users/gil/Dropbox/study/FFP/seismo-acoustic/parallel/'
    # loc_f='/Users/gil/Dropbox/study/FFP/seismo-acoustic/parallel/'+Fname

    loc_i='./'
    loc_f='./'+Fname
    
    if os.path.isdir(loc_f)==True:
        print('directory exist. removing and recreating')
        shutil.rmtree(loc_f)
        os.mkdir(loc_f)
    if os.path.isdir(loc_f)==False:
        print('creating directory')
        os.mkdir(loc_f)

    loc_f=loc_f+'/'
    shutil.copy(loc_i+'/input-nb', loc_f+'/input-nb')

    pos_0= earth_interface + ocean_interface
    phases=np.zeros(wavenumbers, dtype=np.float64)
    for i in range(1,wavenumbers):
        phases[i]=omega/np.real(K[i])

    P_null= 1*(omega**2)*np.exp(1j * omega/330)/(4*pi)

    for alt in Rec_alt:


        pos= pos_0 + np.int64(alt/dz)
        alt=np.int64(alt/dz) * dz

        TL_name='TL_alt_'+str(alt)+'.csv'
        P_name='abs-pressure_alt_'+str(alt)+'.csv'
        Modes_name='modes_alt_'+str(alt)+'.csv'
        U_name='Uzz.nc'

        
        with open(TL_name, 'w') as f:
            writer = csv.writer(f, delimiter='\t')
            writer.writerows(zip(r/1000,20*np.log10(np.abs(P[pos,:])/(np.abs(P[pos,0])))))

        with open(P_name, 'w') as f:
            writer = csv.writer(f, delimiter='\t')
            writer.writerows(zip(r/1000,np.abs(P[pos,:])))

        with open(Modes_name, 'w') as f:
            writer = csv.writer(f, delimiter='\t')
            writer.writerows(zip(phases,np.abs(A[pos,:])*smooth_window[:wavenumbers]))



        os.rename(loc_i+TL_name, loc_f+TL_name)
        os.rename(loc_i+P_name, loc_f+P_name)
        os.rename(loc_i+Modes_name, loc_f+Modes_name)

    vel_name='Vp.csv'
    with open(vel_name, 'w') as f:
            writer = csv.writer(f, delimiter='\t')
            writer.writerows(zip(z/1000, Vp))

    os.rename(loc_i+vel_name, loc_f+vel_name)

    plt.figure(figsize=[15,5])
    plt.plot(r/1000,20*np.log10(np.abs(P[pos,:])/(np.abs(P[pos,0]))))
    plt.xlim([0,1000])
    plt.show()

    ncfile = Dataset(U_name, 'w', format='NETCDF4_CLASSIC')

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

    # Uzz_var[:] = 20*np.log10(np.abs(P[:,:])/(4*pi))

    # print (Uzz_var[:])
    displacement=20*np.log10(np.abs(P[:,:])/(np.abs(P[pos,0])))
    displacement=np.real(displacement)
    # displacement[displacement==-np.inf] = -2000
    # displacement[displacement==np.inf] = -2000
    # displacement[displacement==np.nan] = -2000
    displacement[np.isposinf(displacement)] = -2000
    displacement[np.isneginf(displacement)] = -2000
    displacement[np.isnan(displacement)] = -2000
    # for i in range(0,layers):
    #     for j in range(0,wavenumbers):
    #         if displacement[i,j] in [-np.inf, np.inf]:
    #             displacement[i,j]=-1000


    r_var[:] = r/1000
    z_var[:] = z/1000 - (Earth_depth + Ocean_depth)/1000
    Uzz_var[:] = displacement

    ncfile.close()

    

    
    os.rename(loc_i+U_name, loc_f+U_name)






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
