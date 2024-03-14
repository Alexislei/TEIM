clear;clc;

img_nums=['01';'02';'03';'04';'05';'06';'07';'08';'09';'10';'11';'12';'13';'14';'15';'16';'17';'18';'19';'20';'21';'22';'23';'24';'25';'26';'27'];
runtimes=1;
iSize='L';

t_all=zeros(27,1);
 
for im=1:27
    img_num=img_nums(im,:);

    for r=1:runtimes
        %%
        filename=['GT',img_num,'.png'];
        filename_trimap=['GT',img_num,'_tri',iSize,'.png'];
        filename_gt=['GT',img_num,'_gt.png'];

        %Unit8-double
        img = imread(filename);
        img = single(img);
        trimap = imread(filename_trimap);
        gt = rgb2gray(imread(filename_gt));

        %% 
        [img, trimap, gt,F_mindist,B_mindist,U_rgb,F_rgb,B_rgb,U_s,F_s,B_s ] = GetData(img,trimap,gt);
        D = size(U_s,1);
        MSEFcn=@(alpha_U)GetMSE(alpha_U,gt,find(trimap==128));

        %% mInfo
        mInfo.D=D;
        mInfo.F_mindist=F_mindist;mInfo.B_mindist=B_mindist;
        mInfo.U_rgb=U_rgb;mInfo.F_rgb=F_rgb;mInfo.B_rgb=B_rgb;
        mInfo.U_s=U_s;mInfo.F_s=F_s;mInfo.B_s=B_s;

        %% params
        param.NP = 50;
        param.Max_FEs = 5e3; % maximal number of FEs, should be set to 3e+06
        param.w = 0.729;
        param.c1 = 1.5;
        param.c2 = 2.0;

        param.low_bound_F = 1;
        param.up_bound_F = size(F_rgb, 1);
        param.low_bound_B = 1;
        param.up_bound_B = size(B_rgb, 1);

        param.tau = 1;
        param.thr1 = 0.9;
        param.thr2 = 1e-3;

        tic
        [gbest,fitness_monitor,x,pbest] = ACSCPSO(mInfo,MSEFcn,param);
        t_all(im,1)=toc;

        [~,best_alpha,~] = CSCPSO_CostFunc(gbest,F_rgb,B_rgb,U_rgb,F_s,B_s,U_s,F_mindist,B_mindist);
        best_MSE = MSEFcn(best_alpha);

        FB_pairs=reshape(gbest',[2,size(gbest,2)/2])';
        alpha=FB2alpha(FB_pairs, img, trimap, false);
%         save(['pso_acsc_',iSize,'_',num2str(img_num),'_',num2str(r),'.mat'],'best_MSE','alpha','-v7.3');
%         imwrite(alpha,['pso_acsc_',iSize,'_',num2str(img_num),'_',num2str(r),'.png']);
    end
end