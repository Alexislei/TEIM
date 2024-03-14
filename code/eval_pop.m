function [fitness,est_alpha,fit_NP]=eval_pop(pop,mInfo,tri_s,tri_ind)
    [~,~,NP]=size(pop);
    NP=NP/2;
    
    x=zeros(NP,2*sum(sum(tri_s==2)));
    for i=1:NP
        chf=pop(:,:,2*i-1);
        chb=pop(:,:,2*i);
        x(i,1:2:end)=chf(tri_s==2);
        x(i,2:2:end)=chb(tri_s==2);
    end
    
    global_ind=find(tri_ind==2);
    U_ind=find(mInfo.trimap==128);
    idx=zeros(size(global_ind,1),1);
    for i=1:size(global_ind,1)
        idx(i)=find(U_ind==global_ind(i));
    end
    
    U_rgb1=mInfo.U_rgb(idx,:);U_s1=mInfo.U_s(idx,:);
    F_mindist1=mInfo.F_mindist(idx,:);B_mindist1=mInfo.B_mindist(idx,:);
    F_rgb=mInfo.F_rgb;B_rgb=mInfo.B_rgb;
    F_s=mInfo.F_s;B_s=mInfo.B_s;

    [fit,est_alpha,fit_NP]=TEA_CostFunc(x,F_rgb,B_rgb,U_rgb1,F_s,B_s,U_s1,F_mindist1,B_mindist1);
    
    fitness=fit;
end