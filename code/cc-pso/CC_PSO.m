function [alpha_matte, fitness, x] = CC_PSO(img, trimap, max_fitness_evaluation, mask)
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

    % 用预处理的结果做聚类
    % 其中cc_rgb为每类的颜色
    % U_cc_ind为原先未知区域的每个点属于的聚类下标（U在cc中的位置）
    % cc_U_ind为聚类后，每类提取的点，在原先未知区域中的下标（cc在U中的位置）
    [cc_rgb, U_cc_ind, cc_U_ind] = color_clustering(U_rgb);
    
    % 读取聚类后的空间信息
    [F_y, F_x] = ind2sub(size(trimap), F_ind);
    [B_y, B_x] = ind2sub(size(trimap), B_ind);
    % 聚类后，每类提取出的点的x，y信息
    [cc_y, cc_xx] = ind2sub(size(trimap), U_ind(cc_U_ind));
    F_s = [F_y, F_x];
    B_s = [B_y, B_x];
    cc_s = [cc_y, cc_xx];
    
    % 聚类后，每类提取出的点，到F和B的最短距离 
    F_mindist = bwdist(mask == 255); 
    B_mindist = bwdist(mask == 0);   
    cc_F_mindist = F_mindist(U_ind(cc_U_ind));
    cc_B_mindist = B_mindist(U_ind(cc_U_ind));

    % 聚类的结果PSO
    FitnessFcn = @(x) CostFunc(x, F_rgb, B_rgb, cc_rgb, F_s, B_s, cc_s, cc_F_mindist, cc_B_mindist);
    number_of_variables = 2 * length(U_ind(cc_U_ind));
    lb = ones(1, number_of_variables); % Lower bound
    ub = [repmat(size(F_rgb, 1), 1, length(U_ind(cc_U_ind))), repmat(size(B_rgb, 1), 1, length(U_ind(cc_U_ind)))]; % Upper bound

    [cc_x, ~] = MyCCPSO(FitnessFcn, number_of_variables, lb, ub, max_fitness_evaluation);

    % 初始化最终保存结果的矩阵
    x = zeros(1, 2 * length(U_ind));
    fitness = zeros(1, 2 * length(U_ind));
    alpha_matte = mask;
    clear('mask');
    % 结果扩充，每类做PSO
    for ii = 1 : length(cc_U_ind)
%         fprintf('%d/%d\n', ii, length(cc_U_ind));
        % 第ii类在U中的下标
        ii_U_ind = find(U_cc_ind == ii);
        % 第ii类在图片中的下标
        ii_ind = U_ind(ii_U_ind);
        % 这里要改下FitnessFcn的定义，改一下U的位置与颜色，以及U到F、B的最短距离   
        [ii_yy, ii_xx] = ind2sub(size(trimap), ii_ind);
        ii_s = [ii_yy, ii_xx];
        ii_rgb = repmat(cc_rgb(ii, :), length(ii_ind), 1); % 这一类的颜色都是相通的，扩充
        % 聚类后，每类提取出的点，到F和B的最短距离    
        ii_F_mindist = F_mindist(ii_ind);
        ii_B_mindist = B_mindist(ii_ind);

        % 按照之前算出的最优解，拼出一个初始解
        initial_x = [repmat(cc_x(ii), 1, length(ii_ind)), repmat(cc_x(ii + length(U_ind(cc_U_ind))), 1, length(ii_ind))];
        
        ii_x = initial_x;
        
        % 计算alpha、fitness、confidence
        [ii_fitness, ii_alpha] = CostFunc(ii_x, F_rgb, B_rgb, ii_rgb, F_s, B_s, ii_s, ii_F_mindist, ii_B_mindist, true);
        ii_alpha(ii_alpha > 1) = 1;
        ii_alpha(ii_alpha < 0) = 0;
        ii_alpha = ii_alpha * 255;
        
        % 结果保存在整体的x，fitness，alpha
        x(ii_U_ind) = ii_x(1 : length(ii_ind));
        x(ii_U_ind + length(U_ind)) = ii_x(length(ii_ind) + 1: end);
        fitness(ii_U_ind) = ii_fitness(1 : length(ii_ind));
        alpha_matte(ii_ind) = ii_alpha(1 : length(ii_ind));

    end
    
end
