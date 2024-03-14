function [alpha_matte, fitness, Best, FSample, BSample, Un] = CC_DE_S(img, trimap, max_fitness_evaluation, mask)
    % 如果没有mask的话这里也做预处理
    if ~exist('mask', 'var') || isempty(mask)
        % 预处理
        mask_iter = 11;
        mask = TrimapExpansion(double(img), trimap, mask_iter);
    end
    
    img = double(img);
    
    mask(mask == 0) = 5;
    mask(mask == 255) = 1;
    mask(mask == 128) = 3;
    
    GetUn_FBSample(img, mask);
    
    load SetFSample.mat;
    load SetBSample.mat;
    load SetUn.mat;
    load SetUnNum.mat;
    
    fprintf('CCDES\n');
    %% Applying algorithm to search the best FB Samples.
    %Best = TestAlgorithm(I, Trimap, MaskAct, Un, FSample, BSample, ImageNum);
    [Best, fitness] = CCDES(img, trimap, mask, Un, Un_num, FSample, BSample);
    
    fprintf('post-processing...\n');
    %% Post-processing which is mentioned in paper titled 'A Global Sampling Method for Alpha Matting' ==============
    [TR_FVAlpha, TRobust, ~] = GetAlpha(img, mask, reshape(Best(1, :), 2, size(Un, 1))', FSample, BSample, Un);
    % RConf1=sqrt(TRobust) ;
    % RConf1(mask==1)=1 ; RConf1(mask==5)=1 ;
    
    % pack=[] ;
    % pack(:,:,1) = uint8(TR_FVAlpha*255 ) ;
    % pack(:,:,2) = uint8((RConf1)*255 ) ;
    % pack(:,:,3) = uint8(trimap) ;
    % alpha = Fun_SmoothingMatting(img, pack) ;
    % alpha_matte = uint8(alpha*255);
    
    alpha_matte = TR_FVAlpha;
    
end