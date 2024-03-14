function [alpha_matte, fitness, x] = CSO(img, trimap, max_fitness_evaluation, mask, num)

    % 如果没有mask的话这里也做预处理
    if ~exist('mask', 'var') || isempty(mask)
        % 预处理
        mask_iter = 11;
        mask = TrimapExpansion(double(img), trimap, mask_iter);
    end
    
    % 预处理之后三分图的区域
    F_ind = find(mask == 255); 
    B_ind = find(mask == 0); 
    U_ind = find(mask == 128); 

    % 读取三个区域的颜色数组
    img_rgb = single(reshape(img, [numel(trimap), 3]));
    F_rgb = img_rgb(F_ind, :);
    B_rgb = img_rgb(B_ind, :);
    U_rgb = img_rgb(U_ind, :);
    clear('img_rgb');
    
    [F_y, F_x] = ind2sub(size(trimap), F_ind);
    [B_y, B_x] = ind2sub(size(trimap), B_ind);
    [U_y, U_x] = ind2sub(size(trimap), U_ind);
    F_s = [F_y, F_x];
    B_s = [B_y, B_x];
    U_s = [U_y, U_x];

     
    F_mindist = bwdist(trimap == 255); F_mindist = F_mindist(U_ind);
    B_mindist = bwdist(trimap == 0); B_mindist = B_mindist(U_ind);    

    %% options for ga
    FitnessFcn = @(x) CostFunc(x, F_rgb, B_rgb, U_rgb, F_s, B_s, U_s, F_mindist, B_mindist, true);
    numberOfVariables = 2 * length(U_ind); % Number of decision variables
    lb = ones(1, numberOfVariables); % Lower bound
    ub = [repmat(size(F_rgb, 1), 1, length(U_ind)) repmat(size(B_rgb, 1), 1,length(U_ind))];
    
    if ~exist('num', 'var') || isempty(num)
        num = 0;
    end
    SaveProcessFcn = @(x, FES) save_process(x, FES, img, trimap, mask, num, FitnessFcn);
    
    % [ x,~,~] = MyCSO( FitnessFcn,numberOfVariables,lb,ub,max_fitness_evaluation, false, true, SaveProcessFcn);
    [ x,~,~] = MyCSO(FitnessFcn,numberOfVariables,lb,ub,max_fitness_evaluation);
    %% disp MOGA
    [ fitness, est_alpha] = CostFunc(x,F_rgb,B_rgb,U_rgb,F_s,B_s,U_s,F_mindist,B_mindist, true);
    clear('F_rgb', 'B_rgb', 'U_rgb', 'F_s', 'B_s', 'F_mindist', 'B_mindist');
    est_alpha(est_alpha > 1) = 1;
    est_alpha(est_alpha < 0) = 0;
    est_alpha = 255 * est_alpha;
    
    alpha_matte = mask;
    clear('mask');

    alpha_matte(U_ind) = est_alpha;
    
end