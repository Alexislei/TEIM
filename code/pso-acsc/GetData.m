function [ img, trimap,gt,F_mindist,B_mindist,U_rgb,F_rgb,B_rgb,U_s,F_s,B_s ] = GetData(img,trimap,gt )

% function [ img, trimap,gt,F_mindist,B_mindist,U_rgb,F_rgb,B_rgb,U_s,F_s,B_s ] = GetData(ImageNum,iSize,path )
%GETRGBS Summary of this function goes here
%   Detailed explanation goes here
%     InputAdress = strcat(path,'/img/GT',num2str(ImageNum,'%.2d'),'.png');
%     TrimapAdress = strcat(path,'/Trimap1/GT',num2str(ImageNum,'%.2d'),'_triS.png');
%     GTAdress = strcat(path,'/gt/GT',num2str(ImageNum,'%.2d'),'_gt.png');
%     InputAdress = strcat(path,'/img/GT',num2str(ImageNum,'%.2d'),'.png');
%     TrimapAdress = strcat(path,'/Trimap1/GT',num2str(ImageNum,'%.2d'),'_tri',iSize,'.png');
%     GTAdress = strcat(path,'/gt/GT',num2str(ImageNum,'%.2d'),'_gt.png');

    %Unit8-double
%     img = imread(InputAdress);
%     img = single(img);
%     trimap = imread(TrimapAdress);
%     gt = rgb2gray(imread(GTAdress));
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    F_ind = find(trimap == 255);
    B_ind = find(trimap == 0);
    U_ind = find(trimap == 128);
    img_rgb = single(reshape(img,[numel(trimap),3]));
    F_rgb = img_rgb(F_ind,:); 
    B_rgb = img_rgb(B_ind,:); 
    U_rgb = img_rgb(U_ind,:); 
    [F_y,F_x] = ind2sub(size(trimap),F_ind); F_yx = [F_y,F_x];
    [B_y,B_x] = ind2sub(size(trimap),B_ind); B_yx = [B_y,B_x];
    [U_y,U_x] = ind2sub(size(trimap),U_ind); U_yx = [U_y,U_x];
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    F_mindist = bwdist(trimap == 255);F_mindist = F_mindist(U_ind);
    B_mindist = bwdist(trimap == 0);B_mindist = B_mindist(U_ind);
    F_s = [F_y,F_x];
    B_s = [B_y,B_x];
    U_s = [U_y,U_x];
end

