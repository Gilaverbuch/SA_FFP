% Analytical solution for downward refracting atmosphere.

clear all;
close all;

freq=0.5;
omega=2*pi*freq;
dz=100.0;
dr=1000;
r_max=1000*1e3;

zmin=0;
zmax=150.0*1e3;
z=zmin:dz:zmax;
nz=size(z,2);

fprintf('\n\n');
fprintf('Analytic modal solution for pseudo-linear sound speed profile\n');
fprintf('\n\n');
fprintf('Frequency = %6.2f Hz\n', freq);
fprintf('\n\n');


%profile
a= 3.5e-11;
b= 4e-6;
c= sqrt(1./(a*(abs(z-zmax))+b));

k_max = omega/min(300);
k_min = omega/450.0;

n_grid = 1024;

k_grid   = linspace(k_min,k_max,n_grid);
n_modes  = 0;
k_mode   = zeros(n_grid,1);
psi_mode = zeros(nz,n_grid);

kH_last  = k_grid(1);
airy_arg = ((omega^2*a)^(-2/3)) * (kH_last^2 - (omega^2)./(c.^2));
psi_last = airy(1,airy_arg);
    
for kk=2:n_grid
    kH=k_grid(kk);

    airy_arg=((omega^2*a)^(-2/3)) * (kH^2 - (omega^2)./(c.^2));
    psi     = airy(1,airy_arg);
    
    if (real(psi(1))*real(psi_last(1)) < 0)
     kHmin    = kH_last;
     kHmax    = kH;
     fprintf('   Mode found in interval %6.2f m/s - %6.2f m/s ->', omega/kHmax, omega/kHmin);
     [k_,psi_] = searchMode(kHmin,kHmax,dz,omega,a,c);
     fprintf(' isolated at %6.2f m/s\n', omega/k_);
     n_modes             = n_modes + 1;
     k_mode(n_modes)     = k_;
     psi_mode(:,n_modes) = psi_';
     
     psi_last = psi;
     kH_last  = kH;
    else
     fprintf('No mode found in interval %6.2f m/s - %6.2f m/s\n', omega/k_grid(kk), omega/k_grid(kk-1));
     psi_last = psi;
     kH_last  = kH;
    end
    %return;
    %fprintf( ' done!\n');
end

k_mode=k_mode(1:n_modes);
psi_mode=psi_mode(:,1:n_modes);

fprintf('\n Search complete. We found %d modes.\n', n_modes);

%%%%%%%%%%%%
% Normalize modes
%%%%%%%%%%%%




%%%%%%%%%%%%
% Compute modal sum
%%%%%%%%%%%%

r       = dr:dr:r_max;
p_refix = ones(nz,1)*(exp(-1i*pi*0.25).*sqrt(1/8/pi./r));

sum = 0;
%the index in the first psi_mode(ind,m) determines the source position 
for m = 1:n_modes
 sum = sum  + psi_mode(100,m)*psi_mode(:,m)*exp(1i*k_mode(m)*r)/sqrt(k_mode(m));
end

p     = p_refix.*sum;
p_gnd = p(1,:);

figure;
plot(r/1000,20*log10(abs(4*pi*p_gnd)));
xlabel('Range [km]');
ylabel('Transmission Loss [dB re 1 km]');

    figure;
    imagesc(r/1000,z/1000,20*log10(abs(4*pi*p)),[-120 -60]);
    %colormap([255/255 255/255 255/255; 205/255 173/255 0; 255/255 48/255 48/255; 255/255 0 0; 139/255 37/255 0; 0 0 255/255; 72/255 61/255 139/255]);
    xlim([0 1000]);
    ylim([0 150000/1000]);
    set(gca,'YDir','normal')

    xlabel ('Range [km]');
    ylabel ('Altitude [km]');
    t = colorbar('peer',gca);
    set(get(t,'ylabel'),'String', 'Transmission Loss [dB re 1 km]');

    
fileID = fopen('N2_TL.txt','w');
fprintf(fileID,'%12.8f\n',20*log10(abs(4*pi*p_gnd)));
fclose(fileID);

fileID = fopen('N2_vel.txt','w');
fprintf(fileID,'%6.2f\n',c);
fclose(fileID);

fileID = fopen('N2_range.txt','w');
fprintf(fileID,'%6.2f\n',r/1000);
fclose(fileID);
%     
%     figure;
% plot(c,z/1e3,'b',10*psi+c_ph,z/1e3,'r');
% xlim([250 550]);
% xlabel('sound speed [m/s]');
% ylabel('altitude [km]');