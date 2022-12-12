

import matplotlib.pyplot as plt
plt.rcParams['font.size'] = '16'
plt.rcParams['figure.dpi'] = 125
plt.rcParams['figure.facecolor'] = 'white'
plt.rcParams['pcolor.shading'] = 'auto'
import numpy as np 
import xarray as xr



tl = xr.open_dataset('Results/test_munk/TL_intensity_ref_source.nc')
print (tl)



r = tl['Range'].values
alt = tl['Altitude'].values
TL = tl['TL'].values
tl.close()



# getting the original colormap using cm.get_cmap() function
orig_map=plt.cm.get_cmap('hot')
  
# reversing the original colormap using reversed() function
reversed_map = orig_map.reversed()

plt.figure(figsize=(15,5))
plt.pcolormesh(r, alt, TL,  cmap=reversed_map)
plt.ylim([-5,0])
plt.xlim([0,1000])
plt.xlabel('Range [km]')
plt.ylabel('Depth [km]')
plt.colorbar(label='TL [dB]')
plt.clim([-180,-90])
plt.show()


