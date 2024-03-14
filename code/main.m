clear all;clc; % SC threshlding + TEA

img_nums=['01';'02';'03';'04';'05';'06';'07';'08';'09';'10';'11';'12';'13';'14';'15';'16';'17';'18';
    '19';'20';'21';'22';'23';'24';'25';'26';'27'];

for in=1:27
    img_num=img_nums(in,:);
    iSize='S';
    runtimes=1;

    for r=1:runtimes
        filename=['GT',img_num,'.png'];
        filename_trimap=['GT',img_num,'_tri',iSize,'.png'];
        filename_gt=['GT',img_num,'_gt.png'];

        img = imread(filename);
        img = single(img);

        trimap0=single(imread(filename_trimap));
        gt_alpha=single(rgb2gray(imread(filename_gt)));

        [m,n]=size(trimap0);
        trimap_url = ['GT',img_num,'_mask',iSize,'.png']; % TEA with SpatialColor thresholding
        trimap = single(imread(trimap_url));  

        %%
        F_ind = find(trimap == 255);
        B_ind = find(trimap == 0);
        img_rgb = reshape(img,[numel(trimap),3]);
        F_rgb = img_rgb(F_ind,:); 
        B_rgb = img_rgb(B_ind,:);
        [F_y,F_x] = ind2sub(size(trimap),F_ind); F_yx = [F_y,F_x];
        [B_y,B_x] = ind2sub(size(trimap),B_ind); B_yx = [B_y,B_x];
        F_s = [F_y,F_x];
        B_s = [B_y,B_x];

        U_ind = find(trimap == 128);
        F_mindist = bwdist(trimap == 255);F_mindist = F_mindist(U_ind);
        B_mindist = bwdist(trimap == 0);B_mindist = B_mindist(U_ind);

        U_rgb = img_rgb(U_ind,:); 
        [U_y,U_x] = ind2sub(size(trimap),U_ind); U_yx = [U_y,U_x];
        U_s = [U_y,U_x];

        %%
        MSEFcn=@(alpha_U)GetMSE(alpha_U,gt_alpha,find(trimap0==128));

        %%
        mInfo.F_rgb=F_rgb;mInfo.B_rgb=B_rgb;mInfo.U_rgb=U_rgb;
        mInfo.F_s=F_s;mInfo.B_s=B_s;mInfo.U_s=U_s;
        mInfo.F_mindist=F_mindist;mInfo.B_mindist=B_mindist;
        mInfo.trimap=trimap;

        %%
        param.NP = 50;
        param.Iter=200;
        param.low_bound_F = 1;
        param.up_bound_F = size(F_rgb, 1);
        param.low_bound_B = 1;
        param.up_bound_B = size(B_rgb, 1);

        %%
        e_alpha=double(trimap);
        e_alpha(trimap==255)=1;
        ch_F=zeros(m,n);ch_B=zeros(m,n);
        count=0;
        t=1;
        for t2=1:15:(n+15)
            for t1=1:15:(m+15)
                if t1+15>m && t2+15<=n
                    tri_s=trimap(m-15:m,t2:t2+15);
                elseif t1+15<=m && t2+15>n
                    tri_s=trimap(t1:t1+15,n-15:n);
                elseif t1+15>m && t2+15>n
                    tri_s=trimap(m-15:m,n-15:n);
                else
                    tri_s=trimap(t1:t1+15,t2:t2+15);
                end

                %back:0-1 unknown:128-2 fore:255-3
                if sum(sum(tri_s==128))~=0
                    count=count+1;
                    tri_s(tri_s==0)=1;tri_s(tri_s==128)=2;tri_s(tri_s==255)=3;
                    tri_ind_all=zeros(m,n);
                    if t1+15>m && t2+15<=n
                        tri_ind_all(end-15:end,t2:t2+15)=tri_s;
                    elseif t1+15<=m && t2+15>n
                        tri_ind_all(t1:t1+15,end-15:end)=tri_s;
                    elseif t1+15>m && t2+15>n
                        tri_ind_all(end-15:end,end-15:end)=tri_s;
                    else
                        tri_ind_all(t1:t1+15,t2:t2+15)=tri_s;
                    end
                    
                    
                    [best_fval,best_est_alpha,best_chf,best_chb]=tea_gaussian(tri_s,tri_ind_all,mInfo,param);
    % 
                    alpha_s=tri_s;alpha_s(tri_s==1)=0;alpha_s(tri_s==3)=255;
                    ch_F(tri_ind_all==2)=best_chf(tri_s==2);ch_B(tri_ind_all==2)=best_chb(tri_s==2);
                    e_alpha(tri_ind_all==2)=best_est_alpha;

                    alpha_s(tri_s==2)=best_est_alpha;

                    t=t+1;
                end
            end
        end

        % merge all the patches
        for i=1:size(U_ind,1)
            px=U_s(i,1);py=U_s(i,2);
            if px-8>1 && px+8<m
                rx=px-8:px+8;
            elseif px-8>1 && px+8>m
                rx=px-8:m;
            elseif px-8<1 && px+8<m
                rx=1:px+8;
            end

            if py-8>1 && py+8<n
                ry=py-8:py+8;
            elseif py-8>1 && py+8>n
                ry=py-8:n;
            elseif py-8<1 && py+8<n
                ry=1:py+8;
            end

            tri_s1=trimap(rx,ry);
            tri_s1(tri_s1==0)=1;tri_s1(tri_s1==128)=2;tri_s1(tri_s1==255)=3;
            tri_ind_all1=zeros(m,n);
            tri_ind_all1(rx,ry)=tri_s1;

            size_u=sum(sum(tri_ind_all1==2));
            U_rgb1=repmat(mInfo.U_rgb(i,:),size_u,1);U_s1=repmat(mInfo.U_s(i,:),size_u,1);
            F_mindist1=repmat(mInfo.F_mindist(i,:),size_u,1);
            B_mindist1=repmat(mInfo.B_mindist(i,:),size_u,1);

            F_rgb=mInfo.F_rgb;B_rgb=mInfo.B_rgb;
            F_s=mInfo.F_s;B_s=mInfo.B_s;

            x=zeros(1,2*size_u);
            x(1,1:2:end)=ch_F(tri_ind_all1==2);x(1,2:2:end)=ch_B(tri_ind_all1==2);

            [fit,est_alpha,fit_NP]=TEA_CostFunc(x,F_rgb,B_rgb,U_rgb1,F_s,B_s,U_s1,F_mindist1,B_mindist1);

            [~,midx]=min(fit_NP);
            e_alpha(px,py)=est_alpha(midx);
        end

        best_MSE=immse(gt_alpha,single(e_alpha.*255));

        save(['TEA_',iSize,'_',num2str(img_num),'_',num2str(r),'.mat'],'best_MSE','e_alpha','-v7.3');
        imwrite(e_alpha,['TEA_',iSize,'_',num2str(img_num),'_',num2str(r),'.png']);
    end
end