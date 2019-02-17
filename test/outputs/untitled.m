filename = 'DFT_20190208_smooth_Gen_Bathch1000_Channel8_Samp025600';
fid = fopen([filename '.out'],'r');
%data = textscan(fid, 'Iter # %d: Train Loss: %f; Test Loss: %f.', 'Headerlines', 18);
data = textscan(fid, 'Iter # %d: Train Loss: %f; Test Loss: %f; Gen Test Loss: %f.', 'Headerlines', 20);
figure
semilogy(data{1},data{2});
title(regexprep(filename, '_', ' '))
hold on
semilogy(data{1},data{3});
semilogy(data{1},data{4});
ylim([1e-4 10]);
fclose(fid);

%saveas(gcf, filename, 'jpg');