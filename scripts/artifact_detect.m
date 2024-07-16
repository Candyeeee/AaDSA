clear all;close all;clc;

dsa_folder = "./data/train/dsa/";
fake_dsa_folder = "./results/fake_dsa/";
save_fig_folder = "./train/fake_dsa_motionfigs/";
save_mask_folder = "./train/mask/";
if ~exist(save_fig_folder, 'dir')
    mkdir(save_fig_folder)
end

if ~exist(save_mask_folder, 'dir')
    mkdir(save_mask_folder)
end

filelist = dir(strcat(fake_dsa_folder, '*.npy'));

for i=1:length(filelist)

    flnm = filelist(i).name;
    
    % Read real dsa image
    dsa = readNPY(strcat(dsa_folder, flnm));
    dsa = double(dsa);
    
    % Read fake dsa image
    fake_dsa = readNPY(strcat(fake_dsa_folder, flnm));
    fake_dsa = double(fake_dsa);
    
    Imatch = imhist(dsa);
    fake_dsa = histeq(fake_dsa, Imatch);

    [motionArtifactRegion, Ap] = GradientAMP(dsa, fake_dsa);
    mask = uint8(~motionArtifactRegion);
    
    writeNPY(mask, strcat(save_mask_folder, flnm))
    
  
    
    img_withmask = uint8(zeros(size(dsa,1), size(dsa,2), 3));
    img_withmask(:, :, 1)=uint8(mat2gray(dsa)*255);
    img_withmask(:, :, 2)=uint8(mat2gray(dsa)*255);
    img_withmask(:, :, 3)=uint8(mat2gray(dsa.*~motionArtifactRegion)*255);
    imwrite(img_withmask, strcat(save_fig_folder, flnm(1:end-4), '.jpg'))

end




