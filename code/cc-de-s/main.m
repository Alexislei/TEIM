function main()
close all;
clear all;
clc;
%for ImageNum = [17, 23, 25, 15, 9, 13, 5, 2, 14, 10, 16, 12, 7, 18, 6, 20, 22, 11, 1, 3, 24, 21, 8, 26, 27, 4, 19]
for ImageNum = 17
    fprintf('No.%d\n', ImageNum);
    %% load image and trimap
    I_address = strcat('./input_training_highres/GT', num2str(ImageNum, '%02d'), '.png');
    Tri_address = strcat('./trimap_training_highres/Trimap1/GT', num2str(ImageNum, '%02d'), '.png');
    I = imread(I_address);
    Trimap = imread(Tri_address);
    I = double(I);
    
    fprintf('Triamp expansion...\n');
    %% Trimap mask
    MaskAct = TrimapExpansion(I, Trimap);
    %% Extract sample sets
    %MaskAct = Trimap;
    %MaskAct(Trimap == 0) = 5;
    %MaskAct(Trimap == 255) = 1;
    %MaskAct(Trimap == 128) = 3;
    GetUn_FBSample(I, MaskAct);
    load SetFSample.mat;
    load SetBSample.mat;
    load SetUn.mat;
    load SetUnNum.mat;
    fprintf('CCDES\n');
    %% Applying algorithm to search the best FB Samples.
    %Best = TestAlgorithm(I, Trimap, MaskAct, Un, FSample, BSample, ImageNum);
    Best = CCDES(I, Trimap, MaskAct, Un, Un_num, FSample, BSample, ImageNum);
    
    fprintf('post-processing...\n');
    %% Post-processing which is mentioned in paper titled ¡°A Global Sampling Method for Alpha Matting¡± ==============
    [TR_FVAlpha, TRobust, ~] = GetAlpha(I, MaskAct, reshape(Best(1, :), 2, size(Un, 1))', FSample, BSample, Un);
    RConf1=sqrt(TRobust) ;
    RConf1(MaskAct==1)=1 ; RConf1(MaskAct==5)=1 ;
    
    pack=[] ;
    pack(:,:,1) = uint8(TR_FVAlpha*255 ) ;
    pack(:,:,2) = uint8((RConf1)*255 ) ;
    pack(:,:,3) = uint8(Trimap) ;
    alpha = Fun_SmoothingMatting(I, pack) ;
    ImageName = ['./high', num2str(ImageNum),'_alphaMatting.png'];
    imwrite(uint8(alpha*255), ImageName);
    ImageName_no_smoothing = ['./high', num2str(ImageNum),'_alphaMatting_no_smoothing.png'];
    imwrite(TR_FVAlpha, ImageName_no_smoothing);
    save(['./high', num2str(ImageNum), '.mat'], 'Best');

end

end

