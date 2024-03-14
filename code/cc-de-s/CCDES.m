function [gbest, gbest_val] = CCDES (I, Trimap, MaskAct, Un, Un_num, FSample, BSample, ImageNum)
%************************************************************
% -I, Trimap: 输入的图像以及与其对应的trimap.
% -MaskAct: 标记矩阵.
% -Un: 图像中的未知象素点的个数.
% -Fsample: 前景样本点的集合.
% -Bsample: 背景样本点的集合.
% -group_num: 第几组优化.
% -ImageNum: 待抠图的第几幅图片.
%************************************************************

%% 基本参数设置
FEs = 0;
NP = 4;%10;
Max_FEs = 5E3;
SwarmSize = size(Un_num, 1);
dim = 2*size(Un, 1);
F_low_bound = 1;
F_up_bound = size(FSample,1);
B_low_bound = 1;
B_up_bound = size(BSample,1);

%% 初始化
F_mutation = 0.5;
CR = 0.9;
x = zeros(NP, dim);
v = zeros(NP, dim);
u = zeros(NP, dim);
value_of_x = zeros(NP, 1);
value_of_u = zeros(NP, 1);
gbest = zeros(1, dim);
gbest_val = 0;
r1 = 1:2:(dim-1);
r2 = 2:2:dim;
for i = 1:NP
    x(i, r1) = round(F_low_bound + rand(1, size(Un, 1))*(F_up_bound - F_low_bound));
    x(i, r2) = round(B_low_bound + rand(1, size(Un, 1))*(B_up_bound - B_low_bound));
end
u = x;
for i = 1:NP
    fprintf('(%d/%d)', i, NP);
    [~, ~, value_of_x(i, 1)] = GetAlpha(I, MaskAct, reshape(x(i, :), 2, size(Un, 1))', FSample, BSample, Un);
end
FEs = NP;
[gbest_val, gbestid] = max(value_of_x(:));
gbest = x(gbestid, :);

%% CC-DE-S迭代进化
Iteration = 1;
%*********************************************
while (Iteration <= 0)
    count_dim = 0;
    for s = 1:SwarmSize
       %s
       % left与rifht标记当前种群所在的纬度区间
        left = count_dim + 1;
        right = count_dim + Un_num(s, 1);
        count_dim = right;
       % 当前种群的进化
        for i = 1 : NP
            r1 = round(1 + rand*(NP - 1));
            r2 = round(1 + rand*(NP - 1));
            while((r1 == r2) || (r1 == i) || (r2 == i))
                r1 = ceil(1 + rand*(NP - 1));
                r2 = ceil(1 + rand*(NP - 1));
            end
            z = round(left + rand*(right-left)); % 标记一定会进行突变的纬度z.
            for j = left : right
                v(i,j) = x(i,j) + F_mutation*(x(gbestid, j) - x(i, j)) + F_mutation*(x(r1, j) - x(r2, j));
                if(j == z || rand <= CR)
                    u(i,j) = v(i,j);
                else
                    u(i,j) = x(i,j);
                end
            end
       end
        %************边界约束*************************
        u(:, left:right) = round(u(:, left:right));
        for i = 1:NP
            for j = left:right
                if(u(i, j) > F_up_bound || u(i, j) < F_low_bound || u(i, j) > B_up_bound || u(i, j) < B_low_bound)
                    u(i, j) = x(i,j);
                end
            end
        end
       %********************************************
        for i = 1:NP
            fprintf('Iteration:%d,FEs:%d,swarm(%d/%d),(%d/%d)\n', Iteration, FEs, s, SwarmSize, i, NP);
            [~, ~, value_of_u(i, 1)] = GetAlpha(I, MaskAct, reshape(u(i, :), 2, size(Un, 1))', FSample, BSample, Un);
        end
        for i = 1 : NP
            if(value_of_u(i, 1) > value_of_x(i, 1))
                x(i, :) = u(i, :);
                value_of_x(i, 1) = value_of_u(i, 1);
            end
        end
        FEs = FEs+NP;
        %**********************************************
        [gbest_val_mat, gbestid_mat] = max(value_of_x(:));
        if gbest_val_mat > gbest_val
            gbestid = gbestid_mat;
            gbest_val = gbest_val_mat;
            gbest = x(gbestid, :);
        end
        disp(strcat(num2str(gbest_val),', FEs:',num2str(FEs)));
        
    end
    Iteration = Iteration + 1;
end