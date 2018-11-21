filename = 'DFT_smooth_20181115_Batch100_Channel8_Prefix_v1.out';
fid = fopen(filename,'r');
data = textscan(fid, 'Iter # %d: Train Loss: %f; Test Loss: %f.', 'Headerlines', 17);
figure
semilogy(data{1},data{2});
title(regexprep(filename, '_', ' '))
hold on
semilogy(data{1},data{3});
fclose(fid);