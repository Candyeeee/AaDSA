function [RMAE, PSNR, SSIM] = EvaImg(dsa_path, fake_dsa_path, mask_path, minv, maxv)
% 
        real_dsa = readNPY(dsa_path);
        real_dsa = double(real_dsa);
        real_dsa(real_dsa>maxv) = maxv;
        real_dsa(real_dsa<minv) = minv;

%         fake_dsa_path = strcat(pred_dsa_folder, flnmlist(i).name);
        fake_dsa = readNPY(fake_dsa_path);
        fake_dsa = double(fake_dsa);
        
        
        fake_dsa(fake_dsa>maxv) = maxv;
        fake_dsa(fake_dsa<minv) = minv;

        mask = readNPY(mask_path) > 0;

        range = maxv - minv;

        normalized_real_dsa = mat2gray(real_dsa, [minv, maxv]);
        normalized_fake_dsa = mat2gray(fake_dsa, [minv, maxv]);

        diff = abs(real_dsa(mask) - fake_dsa(mask));
        MSE = sum(sum(diff.^2)) / numel(real_dsa( mask));

        RMAE = mean(abs(real_dsa(mask) - fake_dsa(mask))) / range;
        PSNR = 10 * log(range * range / MSE ) / log(10);
        [~,ssimmap] = ssim(normalized_real_dsa,normalized_fake_dsa);
        SSIM = mean(ssimmap(mask));
    
end
