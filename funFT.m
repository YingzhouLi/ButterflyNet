function res = funFT(k,x)

tmp = (2*pi)* (k*x');
res = complex(cos(tmp),-sin(tmp));


end
