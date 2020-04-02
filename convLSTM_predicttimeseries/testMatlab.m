mynumber = 1
mypath = '/Users/bene/Dropbox/Dokumente/Promotion/PROJECTS/STORMoChip/PYTHON/Learn2Fluct/convLSTM_predicttimeseries/upsamping_gpu0/validation'
mypath = '.'
load(strcat(mypath, '/epoch_',num2str(mynumber),'.mat'))
load(strcat(mypath, '/origin_x',num2str(mynumber+1),'.mat'))
load(strcat(mypath, '/origin_y',num2str(mynumber+1),'.mat'))

result = dip_image(permute(result, [2,3,1]));
result = dip_image(flip(double(result), 3));
sofi = dip_image(permute(sofi, [2,3,4,1]));
gt = dip_image(permute(gt, [2,3,1]));
sofi_result_std = squeeze(std(sofi,[], 3));
sofi_result_mean = squeeze(mean(sofi,[], 3));
cat(4, rotation(result,pi,3,'linear','zero'), sofi_result_mean, sofi_result_std, gt)


open(fn);
 
%%
sofi=(readtimeseries('C:\Users\diederichbenedict\Dropbox\Dokumente\Promotion\PROJECTS\STORMoChip\PYTHON\Learn2Fluct\convLSTM_predicttimeseries\test\MOV_2020_01_07_14_00_42-1.mp4 kept stack.tif'));
sofi=double(resample(sofi,[.5 .5 1]));
save('./test/sofi_ecoli','sofi')