function [motionArtifactRegion, Ap] = IntensityAMP(dsa, fake_dsa, minVal, maxVal)    

    dsa = mat2gray(dsa, [minVal, maxVal]);
    fake_dsa = mat2gray(fake_dsa, [minVal, maxVal]);
    
    Imatch = imhist(dsa);
    fake_dsa = histeq(fake_dsa, Imatch);
    
    Ap = dsa - fake_dsa;
    
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
end
