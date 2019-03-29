N = 2^6;
n_train = 25600;
n_test = 100;

smoothlvl = 2; % 0 non smooth, 1 pos non smooth 2 smooth, 3 pos smooth
lfreq = 1;
seqdata = 0;

xx = ((0:N-1)/N)';
if lfreq
    kk = (0:(N/8-1))';
else
    kk = (3*N/8:N/2-1)';
end

in_siz = length(xx);
in_range = [0,1];
out_siz = length(kk)*2;
if lfreq
    out_range = [0, N/8];
else
    out_range = [3*N/8, N/2];
end

DFT = funFT(kk,xx);

if smoothlvl == 3
    [x_train,y_train] = rand_real_possmooth(DFT,n_train,kk);
    [x_test,y_test] = rand_real_possmooth(DFT,n_test,kk);
elseif smoothlvl == 2
    [x_train,y_train] = rand_real_smooth(DFT,n_train,kk);
    [x_test,y_test] = rand_real_smooth(DFT,n_test,kk);
elseif smoothlvl == 1
    x_train = rand(n_train,N);
    y_train = x_train*DFT';
    
    x_test = rand(n_test,N);
    y_test = x_test*DFT';
else
    x_train = randn(n_train,N);
    y_train = x_train*DFT';
    
    x_test = randn(n_test,N);
    y_test = x_test*DFT';
end

% Normalization
y_norm = sqrt(sum(abs(y_train).^2,2));
x_train = x_train./(y_norm*ones(1,size(x_train,2)));
y_train = y_train./(y_norm*ones(1,size(y_train,2)));

y_norm = sqrt(sum(abs(y_test).^2,2));
x_test = x_test./(y_norm*ones(1,size(x_test,2)));
y_test = y_test./(y_norm*ones(1,size(y_test,2)));

y_train = comp2real(y_train);
y_test = comp2real(y_test);

if smoothlvl == 3
    if lfreq
        fname = 'data_DFT_possmooth.mat';
    else
        fname = 'data_DFT_possmooth_hfreq.mat';
    end
elseif smoothlvl == 2
    if lfreq
        fname = 'data_DFT_smooth.mat';
    else
        fname = 'data_DFT_smooth_hfreq.mat';
    end
elseif smoothlvl == 2
    if lfreq
        fname = 'data_DFT_pos.mat';
    else
        fname = 'data_DFT_pos_hfreq.mat';
    end
else
    if lfreq
        fname = 'data_DFT.mat';
    else
        fname = 'data_DFT_hfreq.mat';
    end
end
% save(['data/' fname], 'n_train', 'n_test', 'in_siz', 'in_range', ...
%     'out_siz', 'out_range', ...
%     'x_train', 'y_train', 'x_test', 'y_test', '-v7');

if seqdata
    tot_n_train = n_train;
    tot_x_train = x_train;
    tot_y_train = y_train;
    initsiz = 100;
    while initsiz <= tot_n_train
        if smoothlvl == 3
            if lfreq
                fname = ['data_DFT_possmooth_' sprintf('%06d',initsiz) '.mat'];
            else
                fname = ['data_DFT_possmooth_hfreq_' sprintf('%06d',initsiz) '.mat'];
            end
        elseif smoothlvl == 2
            if lfreq
                fname = ['data_DFT_smooth_' sprintf('%06d',initsiz) '.mat'];
            else
                fname = ['data_DFT_smooth_hfreq_' sprintf('%06d',initsiz) '.mat'];
            end
        elseif smoothlvl == 1
            if lfreq
                fname = ['data_DFT_pos_' sprintf('%06d',initsiz) '.mat'];
            else
                fname = ['data_DFT_pos_hfreq_' sprintf('%06d',initsiz) '.mat'];
            end
        else
            if lfreq
                fname = ['data_DFT_' sprintf('%06d',initsiz) '.mat'];
            else
                fname = ['data_DFT_hfreq_' sprintf('%06d',initsiz) '.mat'];
            end
        end
        n_train = initsiz;
        x_train = tot_x_train(1:n_train,:);
        y_train = tot_y_train(1:n_train,:);
        save(['data/' fname], 'n_train', 'n_test', 'in_siz', 'in_range', ...
            'out_siz', 'out_range', ...
            'x_train', 'y_train', 'x_test', 'y_test', '-v7');
        initsiz = 4*initsiz;
    end
end

function [x_data,y_data] = rand_real_smooth(DFT,n,kk)
lenk = length(kk);
y_data = zeros(n,lenk);
zeroidx = [];
for itk = 1:lenk
    if kk(itk) == 0
        zeroidx = itk;
        y_data(:,itk) = 2*randn(n,1);
    else
        negk = find(kk == -kk(itk));
        y_data(:,itk) = randn(n,1)+1i*randn(n,1);
        if ~isempty(negk)
            y_data(:,negk) = conj(y_data(:,itk));
        end
    end
end
nzidx = setdiff(1:lenk,zeroidx);
x_data = real(y_data*DFT+conj(y_data(:,nzidx)*DFT(nzidx,:)))/size(DFT,2);
end

function [x_data,y_data] = rand_real_possmooth(DFT,n,kk)
lenk = length(kk);
y_data = zeros(n,lenk);
zeroidx = [];
for itk = 1:lenk
    if kk(itk) == 0
        zeroidx = itk;
        y_data(:,itk) = 2*rand(n,1);
    else
        negk = find(kk == -kk(itk));
        y_data(:,itk) = rand(n,1)+1i*rand(n,1);
        if ~isempty(negk)
            y_data(:,negk) = conj(y_data(:,itk));
        end
    end
end
nzidx = setdiff(1:lenk,zeroidx);
x_data = real(y_data*DFT+conj(y_data(:,nzidx)*DFT(nzidx,:)))/size(DFT,2);
end

function rr = comp2real(cc)
rtmp = [real(cc) imag(cc)];
rtmp = reshape(rtmp,[size(cc) 2]);
rtmp = permute(rtmp,[1 3 2]);
rr = reshape(rtmp,size(cc,1),[]);
end
