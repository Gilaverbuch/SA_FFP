# SA_FFP
seismo-acoustic Fast Field Program for coupled elastic acoustic wave propagation

Remove new directories before commiting and pushing. The results directories are too large for github.


Specifications for input file:

1. MAKE SURE THAT THERE IS AN EMPTY LINE AFTER THE LAST INPUT PARAMETERS!

2. Simulation type can get narrowband/broadband

3. .Frequency gets any value

4. dz indicates the size of the layers

5. Wavenumbers defines the number of wavenumbers within [Kmin,Kmax]. If the propagation range is not enough -> increase wavenumbers

6. Source gets medium=earth/ocean/atmosphere, depth/depth/altitude= dsorce depth or altitude in case of atmosphere and type=source/force (point force only in solid)

7. BC-bottom/top gets bc=free/rigid/radiation. top is at z=0. bottom is at z=layers.

8. File-name indicates the name of the folder that contains all the simulation results. If "exact" type=acoustic/elastic/N2

9. Earth gets type=Granite/Sandstone/Wet-sand. depth=the depth of the solid part. If type=non -> depth=0. If type=linear direction->dcdz=slope nad Vp0=? and Vs0=? starting vel at the top layer 

10. Ocean gets type=munk/homogeneous/non. depth=the depth of the ocean part. If type=non -> depth=0. For homogeneous c=1500 m/s

111. Atm gets type=homogeneous/linear/file_name/non/N2/. If type=non -> depth=0. for homogeneous c=330. if type=file name -> direction indicates the direction of propagation. If type=linear direction->dcdz=slope

12. Toy profile= soundspeed_east_strat.dat
