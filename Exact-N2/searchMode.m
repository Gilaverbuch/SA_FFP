function [kH,psi]=searchMode(kHmin,kHmax,dz,omega,a,c)

n    = 1;
nmax = 100;
TOL  = 1e-12;

while (n < nmax)
 airy_arg  = ((omega^2*a)^(-2/3)) * (kHmin^2 - (omega^2)./(c.^2));
 psi_1 = airy(1,airy_arg);
 
 kHmid = 0.5*(kHmin+kHmax);
 airy_arg = ((omega^2*a)^(-2/3)) * (kHmid^2 - (omega^2)./(c.^2));
 psi_mid  = airy(1,airy_arg);
 
 if ((psi_mid(1) == 0) || ((kHmax-kHmin)/2 < TOL))
     kH  = kHmid;
     airy_arg = ((omega^2*a)^(-2/3)) * (kH^2 - (omega^2)./(c.^2));
     psi = airy(airy_arg); 
     norm = sqrt(dz*trapz(real(psi).*real(psi)));
     psi = real(psi)/norm;
     return;
 end
 n=n+1;
 
 if (sign(real(psi_mid(1))) == sign(real(psi_1(1))))
  kHmin = kHmid;
 else
  kHmax = kHmid;
 end
    
end
fprintf('Try a larger number of iterations (%d)\n', nmax);

end