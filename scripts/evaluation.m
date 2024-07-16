clear;
dsa_folder = './test/dsa/';
mask_folder = './test/mask_test/';
pred_dsa_folder = './results/fake_dsa/';
dsa_figs_group_folder = './dsa_jpg_group/';

mild_test_list = dir(strcat(dsa_figs_group_folder,'mild/*.jpg'));

medium_test_list = dir(strcat(dsa_figs_group_folder,'medium/*.jpg'));

severe_test_list = dir(strcat(dsa_figs_group_folder,'severe/*.jpg'));


%flnmlist = dir(strcat(mask_folder,'*.npy'));
mild_num = length(mild_test_list);
medium_num = length(medium_test_list);
severe_num = length(severe_test_list);

minv = 0;
maxv = 4095; 


mild_DSA_RMAE = zeros(mild_num, 1);
mild_DSA_PSNR = zeros(mild_num, 1);
mild_DSA_SSIM = zeros(mild_num, 1);

for i = 1:mild_num  
    dsa_path = strcat(dsa_folder, mild_test_list(i).name(1:end-4), '.npy');
    fake_dsa_path = strcat(pred_dsa_folder, mild_test_list(i).name(1:end-4), '.npy');
    mask_path = strcat(mask_folder, mild_test_list(i).name(1:end-4), '.npy');
    [RMAE, PSNR, SSIM] = EvaImg(dsa_path, fake_dsa_path, mask_path, minv, maxv);
    mild_DSA_RMAE(i) = RMAE;
    mild_DSA_PSNR(i) = PSNR;
    mild_DSA_SSIM(i) = SSIM; 
end

mean(mild_DSA_SSIM)
std(mild_DSA_SSIM)

mean(mild_DSA_PSNR)
std(mild_DSA_PSNR)

mean(mild_DSA_RMAE)
std(mild_DSA_RMAE)



medium_DSA_RMAE = zeros(medium_num, 1);
medium_DSA_PSNR = zeros(medium_num, 1);
medium_DSA_SSIM = zeros(medium_num, 1);

for i = 1:medium_num  
    dsa_path = strcat(dsa_folder, medium_test_list(i).name(1:end-4), '.npy');
    fake_dsa_path = strcat(pred_dsa_folder, medium_test_list(i).name(1:end-4), '.npy');
    mask_path = strcat(mask_folder, medium_test_list(i).name(1:end-4), '.npy');
    [RMAE, PSNR, SSIM] = EvaImg(dsa_path, fake_dsa_path, mask_path, minv, maxv);
    medium_DSA_RMAE(i) = RMAE;
    medium_DSA_PSNR(i) = PSNR;
    medium_DSA_SSIM(i) = SSIM; 
end

mean(medium_DSA_SSIM)
std(medium_DSA_SSIM)

mean(medium_DSA_PSNR)
std(medium_DSA_PSNR)

mean(medium_DSA_RMAE)
std(medium_DSA_RMAE)




severe_DSA_RMAE = zeros(severe_num, 1);
severe_DSA_PSNR = zeros(severe_num, 1);
severe_DSA_SSIM = zeros(severe_num, 1);

for i = 1:severe_num  
    dsa_path = strcat(dsa_folder, severe_test_list(i).name(1:end-4), '.npy');
    fake_dsa_path = strcat(pred_dsa_folder, severe_test_list(i).name(1:end-4), '.npy');
    mask_path = strcat(mask_folder, severe_test_list(i).name(1:end-4), '.npy');
    [RMAE, PSNR, SSIM] = EvaImg(dsa_path, fake_dsa_path, mask_path, minv, maxv);
    severe_DSA_RMAE(i) = RMAE;
    severe_DSA_PSNR(i) = PSNR;
    severe_DSA_SSIM(i) = SSIM; 
end

mean(severe_DSA_SSIM)
std(severe_DSA_SSIM)

mean(severe_DSA_PSNR)
std(severe_DSA_PSNR)

mean(severe_DSA_RMAE)
std(severe_DSA_RMAE)
