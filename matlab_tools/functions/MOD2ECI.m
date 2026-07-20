function [reci]= MOD2ECI(r, ttt)
[prec] = precess(ttt);
reci = prec * r';
end
function[prec] = precess(ttt)
factor = pi / (180.0*3600.0);

zeta  = (2306.2181*ttt + 0.30188*(ttt)^2 + 0.017998*(ttt)^3);
theta = (2004.3109*ttt - 0.42665*(ttt)^2 - 0.041833*(ttt)^3);
z     = (2306.2181*ttt + 1.09468*(ttt)^2 + 0.018203*(ttt)^3);

zeta = factor*zeta;
theta = factor*theta;
z = factor*z;

coszeta  = cos(zeta);
sinzeta  = sin(zeta);
costheta = cos(theta);
sintheta = sin(theta);
cosz     = cos(z);
sinz     = sin(z);
% ----------------- form matrix  mod to j2000 -----------------
prec(1,1) =  coszeta * costheta * cosz - sinzeta * sinz;
prec(1,2) =  coszeta * costheta * sinz + sinzeta * cosz;
prec(1,3) =  coszeta * sintheta;
prec(2,1) = -sinzeta * costheta * cosz - coszeta * sinz;
prec(2,2) = -sinzeta * costheta * sinz + coszeta * cosz;
prec(2,3) = -sinzeta * sintheta;
prec(3,1) = -sintheta * cosz;
prec(3,2) = -sintheta * sinz;
prec(3,3) =  costheta;
end
