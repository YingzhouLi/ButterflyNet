filename = 'DFT_smooth_hfreq_20190114_Bathch1000_Channel8_Samp001600';
fid = fopen([filename '.out'],'r');
data = textscan(fid, 'Iter # %d: Train Loss: %f; Test Loss: %f.', 'Headerlines', 18);
figure
semilogy(data{1},data{2});
title(regexprep(filename, '_', ' '))
hold on
semilogy(data{1},data{3});
ylim([5e-5 1e-1]);
fclose(fid);

%saveas(gcf, filename, 'jpg');