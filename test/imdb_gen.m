N = 2^6;
n_train = 10000;
n_test = 100;

smooth = 1;

xx = ((0:N-1)/N)';
kk = (-N/8:(N/8-1))';

in_siz = length(xx);
in_range = [0,1];
out_siz = length(kk);
out_range = [-N/8, N/8];

DFT = funFT(kk,xx);

if ~smooth
    x_train = rand(n_train,N);
    y_train = real(x_train*DFT');
    
    x_test = rand(n_test,N);
    y_test = real(x_test*DFT');
else
    [x_train,y_train] = rand_real_smooth(DFT,n_train,kk);
    [x_test,y_test] = rand_real_smooth(DFT,n_test,kk);
end

% Normalization
y_norm = sqrt(sum(y_train.*y_train,2));
x_train = x_train./(y_norm*ones(1,size(x_train,2)));
y_train = y_train./(y_norm*ones(1,size(y_train,2)));

y_norm = sqrt(sum(y_test.*y_test,2));
x_test = x_test./(y_norm*ones(1,size(x_test,2)));
y_test = y_test./(y_norm*ones(1,size(y_test,2)));

if ~smooth
    fname = 'data_DFT.mat';
else
    fname = 'data_DFT_smooth.mat';
end
save(fname, 'n_train', 'n_test', 'in_siz', 'in_range', ...
    'out_siz', 'out_range', ...
    'x_train', 'y_train', 'x_test', 'y_test', '-v7');

function [x_data,y_data] = rand_real_smooth(DFT,n,kk)
lenk = length(kk);
y_data = zeros(n,lenk);
for itk = 1:lenk
    if kk(itk) == 0
        y_data(:,itk) = rand(n,1);
    else
        negk = find(kk == -kk(itk));
        if ~isempty(negk)
            y_data(:,itk) = rand(n,1);
            y_data(:,negk) = y_data(:,itk);
        end
    end
end
x_data = real(y_data*DFT)/size(DFT,2);
end
