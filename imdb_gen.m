N = 2^8;
n_train = 10000;
n_test = 100;

xx = ((0:N-1)/N)';
kk = (-N/4:(N/4-1))';

in_siz = length(xx);
out_siz = length(kk);

% DFT * f(xx) = g(kk)
DFT = funFT(kk,xx);

% mask = rand(nsamp,N/64);
% mask = interpft(mask',N)';
% mask = mask > 0.5;
% 
% s = (1+rand(nsamp,1))*real(funFT(kk,xx))+rand(nsamp,1)*real(funFT(5*kk,xx));
% f = s + mask.*(0.1*randn(nsamp,N));

x_train = rand(n_train,N);
y_train = real(x_train*DFT');

x_test = rand(n_test,N);
y_test = real(x_test*DFT');

save('data_DFT.mat', 'n_train', 'n_test', 'in_siz', 'out_siz', ...
    'x_train', 'y_train', 'x_test', 'y_test', '-v7');
