function [gbest,fitness_monitor,x,pbest] = ACSCPSO(mInfo,MSEFcn,param)
%************************************************************
% -input, trimap: ????????????????????????trimap. input: m*n*3;
% trimap: m*n.
% -ImageNum: ??????????????????.
% -D, Un: ????????????????????????. D: size(Un,1); Un: un*2.
% -iter: ??????????.
% -filename:??????????????????.
%********************************************
% rule2_count = 0;
%%
D=mInfo.D;
F_mindist=mInfo.F_mindist;B_mindist=mInfo.B_mindist;
U_rgb=mInfo.U_rgb;F_rgb=mInfo.F_rgb;B_rgb=mInfo.B_rgb;
U_s=mInfo.U_s;F_s=mInfo.F_s;B_s=mInfo.B_s;
%%
Max_FEs=param.Max_FEs;
NP=param.NP;
w=param.w; c1=param.c1; c2=param.c2;
% low_bound_F = param.low_bound_F; 
up_bound_F = param.up_bound_F;
% low_bound_B = param.low_bound_B; 
up_bound_B = param.up_bound_B;
up_bound = repmat([up_bound_F;up_bound_B],1,D);
up_bound = repmat(up_bound(:)',NP,1);

%% ????????
x = zeros(NP, D*2);
v = zeros(NP,D*2);
% value_of_x = zeros(NP,1);
% gbest = zeros(1, D*2);
% gbest_val = 0;
for i = 1 : NP
    for j = 1 : D
        x(i, j*2-1) = 1 + ceil(rand()*(up_bound_F-1));
        x(i, j*2) = 1 + ceil(rand()*(up_bound_B-1));
    end
end

[value_of_x,est_alpha,fitness_NP] = CSCPSO_CostFunc(x,F_rgb,B_rgb,U_rgb,F_s,B_s,U_s,F_mindist,B_mindist);

FEs=0;
Iteration = 1;
FEs = FEs + NP;
pbest = x;
pbest_val = value_of_x;
pbest_fitness = fitness_NP;
[gbest_val, gbest_id] = min(value_of_x(:));
gbest = x(gbest_id, :);
best_alpha = est_alpha(gbest_id, :);
best_MSE = MSEFcn(best_alpha);
gbest_fitness = pbest_fitness(gbest_id,:);

tau=param.tau;
thr1=param.thr1;
thr2=param.thr2;
% last_gbest = gbest;
% last_gbest_val = gbest_val;
% l_FEs = FEs;
% best_val = gbest_val;
% best_x = gbest;
MaxIteration = Max_FEs/NP;
fitness_monitor = zeros(MaxIteration,4);
fitness_monitor(Iteration,1) = gbest_val;
fitness_monitor(Iteration,4) = best_MSE;

% n=1;
%*******************************************************************
while (FEs <= Max_FEs)
%     FEs
%     tic;
%     FEs
    %% Pixel pair reset operator
    if mod(Iteration, tau) == 0 % tau==1
        p1 = 1 + ceil(rand * (NP-1));
        p2 = 1 + ceil(rand * (NP-1));
        while(p2 == p1)
            p2 = 1 + ceil(rand * (NP-1));
        end
        bw = abs(est_alpha(p1,:)-est_alpha(p2,:))<3/255;
        if nnz(bw)>numel(bw)*0.5
            fitness_monitor(Iteration,2) = 1;
            for i = 1 : NP
                for j = 1 : D
                    x(i, j*2-1) = 1 + ceil(rand()*(up_bound_F-1));
                    x(i, j*2) = 1 + ceil(rand()*(up_bound_B-1));
                end
            end
            [value_of_x,est_alpha,fitness_NP ] = CSCPSO_CostFunc( x,F_rgb,B_rgb,U_rgb,F_s,B_s,U_s,F_mindist,B_mindist );
            FEs = FEs+NP;
            pbest = x; pbest_val = value_of_x;
            
            v = zeros(NP,D*2);
        end
        %% Competitive pixel pair recombination operator
        p3 = randi(NP);
        bw2 = pbest_fitness(p3,:)>gbest_fitness;
%         disp([nnz(bw)/ D/0.5,nnz(bw2)/ D/0.5,gbest_val/1e5]);
        if nnz(bw2)>D*0.5 % nnz: num of nonzero matrix elements
            fitness_monitor(Iteration,3) = 1;
            learning_range = 2*sort(randi(D,1,2));
            learning_range(1) = learning_range(1)-1;
            x(p3,learning_range(1):learning_range(2)) = gbest(learning_range(1):learning_range(2));
%             x = round(x);
%             bw = x(p3,:)>up_bound(1,:)|x(p3,:)<1;
%             x(p3,bw) = temp(bw);
            [value_of_x(p3),est_alpha(p3,:),fitness_NP(p3,:) ] = CSCPSO_CostFunc( x(p3,:),F_rgb,B_rgb,U_rgb,F_s,B_s,U_s,F_mindist,B_mindist );
            FEs = FEs+1;
            %******************************
            if  value_of_x(p3) < pbest_val(p3)                
                pbest(p3,:) = x(p3,:);
                pbest_val(p3) =  value_of_x(p3);
                pbest_fitness(p3,:) = fitness_NP(p3,:);
            end
            if pbest_val(p3) < gbest_val
                gbest_val = pbest_val(p3);
                gbest = pbest(p3,:);
                best_alpha = est_alpha(p3,:);
                best_MSE = MSEFcn(best_alpha);
                gbest_fitness = pbest_fitness(p3,:);
            end
%             l_FEs = FEs;
        end
    end
    
    %% ????
    % w = 0.4+0.5*((Max_FEs-FEs)/Max_FEs);
    temp = x;
    r1 = rand(NP,D*2);
    r2 = rand(NP,D*2);
    gbest_mat = repmat(gbest,NP,1);
    v = w*v+c2*r2.*(gbest_mat-x)+c1*r1.*(pbest-x);
    %[???????]
    x = x+v;
    x = round(x);
    bw = x>up_bound|x<1;
    x(bw) = temp(bw);
    %     for i = 1 : NP
    %         for j = 1 : D
    %             if(x(i, j*2-1) > up_bound_F)
    %                 x(i, j*2-1) = 1 + ceil(rand * (up_bound_F-1));
    %             end
    %             if(x(i, j*2-1) < low_bound_F)
    %                 x(i, j*2-1) = 1 + ceil(rand * (up_bound_F-1));
    %             end
    %             if(x(i, j*2) > up_bound_B)
    %                 x(i, j*2) = 1 + ceil(rand * (up_bound_B-1));
    %             end
    %             if(x(i, j*2) < low_bound_B)
    %                 x(i, j*2) = 1 + ceil(rand * (up_bound_B-1));
    %             end
    %         end
    %     end
    
    %% ??Pbest?Gbest
    [value_of_x,est_alpha,fitness_NP ] = CSCPSO_CostFunc( x,F_rgb,B_rgb,U_rgb,F_s,B_s,U_s,F_mindist,B_mindist );
    FEs = FEs+NP;
    for i = 1:NP
        if  value_of_x(i) < pbest_val(i)
            pbest(i,:) = x(i,:);
            pbest_val(i) =  value_of_x(i);
            pbest_fitness(i,:) = fitness_NP(i,:);
        end
        if pbest_val(i) < gbest_val
            gbest_val = pbest_val(i);
            gbest = pbest(i,:);
            best_alpha = est_alpha(i,:);
            best_MSE = MSEFcn(best_alpha);
            gbest_fitness = pbest_fitness(i,:);
        end
    end
    %************************************************
    fitness_monitor(Iteration,1) = gbest_val;
    fitness_monitor(Iteration,4) = best_MSE;
    Iteration = Iteration +1;
%     t_all(n,:)=toc;
%     n=n+1;
end

function [dis]=Edis(a,b)
[n,d] = size(a);
dis = 0;
for dd = 1 : d
    dis = dis + (a(dd)-b(dd))^2;
end
dis = sqrt(dis);

function [cdis] = Cdis(a,b)
cdis = 0;
cdis = dot(a,b)/(sqrt(dot(a,a))*sqrt(dot(b,b)));