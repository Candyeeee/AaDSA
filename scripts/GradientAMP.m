function [motionArtifactRegion, Ap] = GradientAMP(dsa, fake_dsa)

    % Threshold for deciding the homogeneous regions
    THRESHOLD_SMALL = 1;


    % zero padding for 2D integration
    PAD = 100;
    dsa = padarray(dsa,[PAD PAD],0,'both');
    fake_dsa = padarray(fake_dsa,[PAD PAD],0,'both');
    [H,W] = size(dsa);

    tic
    disp('Finding cross diffusion tensor')

    sigma = 0.4; %
    ss = floor(6*sigma);
    if(ss<=3)
        ss = 3;
    end
    % K_sigma in the paper; gaussian kernel for smoothing
    ww = fspecial('gaussian',ss,sigma);



    % find G_sigma for real dsa image and fake dsa image
    % T11, T12 and T22 are elements of matrix G_sigma
    [~,~,~,~,EigD_2,X1,X2,Y1,Y2] = TensorAnalysis(fake_dsa,ww);
    [~,~,~,~,EigD_2_2,~,~,~,~] = TensorAnalysis(dsa,ww);
    clear T11_2 T12_2 T22_2 X1_2 X2_2 Y1_2 Y2_2



    % L1 = mu2 , L2 = mu1 of paper
    % initially set to 1,1 to retain all edges in image A
    L1 = ones(H,W);
    L2 = ones(H,W);


    % if there is an edge in Bimage, set mu1 = 0. mu2 = 1 to remove that edge
    % from image A
    idx = find(EigD_2 > THRESHOLD_SMALL);
    L2(idx) = 0;


    % if both A and B are homogeneous
    idx = find(EigD_2 < THRESHOLD_SMALL & EigD_2_2 < THRESHOLD_SMALL);
    L1(idx) = 0;
    L2(idx) = 0;


    % Get cross diffusion tensor terms
    D11 = L1.*(X1.^2) + L2.*(Y1.^2);
    D12 = L1.*(X1.*X2) + L2.*(Y1.*Y2);
    D22 = L1.*(X2.^2) + L2.*(Y2.^2);


    % do for each channel


    disp('=======================================')


    % find gradient field
    [gx,gy] = CalculateGradients(dsa,0);

    % Affine transformation using tensors
    gx1 = (D11.*gx + D12.*gy);
    gy1 = (D12.*gx + D22.*gy);

    % 2D Integration
    Ap = Integration2D(gx1,gy1,zeros(H,W));

    % 2D Integration of residual gradient field
    App = Integration2D(gx-gx1,gy-gy1,zeros(H,W));


    clear gx gy gx1 gy1


    % remove zero padding
    Ap = Ap(PAD+1:end-PAD,PAD+1:end-PAD);

    % for display
    Ap = Ap - min(Ap(:));

    ttime = toc;
    disp('Processing Done...')
    disp(sprintf('Total time taken = %f',ttime))
    

    Ap_max = prctile(Ap(:), 95);
    Ap_min = prctile(Ap(:), 1);
    mask1 = Ap < Ap_min;
    mask2 = Ap > Ap_max;
    mask = (mask1+mask2) >0;

    motionArtifactRegion = bwareaopen(mask,20);

    motionArtifactRegion = bwmorph(motionArtifactRegion,'dilate',4);
    img_reg = regionprops(motionArtifactRegion,  'area', 'boundingbox');
    areas = [img_reg.Area];
    aa = sort(areas, 'descend');
    motionArtifactRegion = bwareaopen(motionArtifactRegion,aa(5));
    
