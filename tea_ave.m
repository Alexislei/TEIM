function [best_fval,best_est_alpha,best_chf,best_chb] = tea_ave(tri_s,tri_ind,mInfo,param)

    trimap=mInfo.trimap;
    NP=param.NP;
    max_iter=param.Iter;
    
    [size_m,size_n]=size(tri_s);
    [m,n]=size(trimap);
    
%     low_bound_F=param.low_bound_F;
    up_bound_F=param.up_bound_F;
%     low_bound_B=param.low_bound_B;
    up_bound_B=param.up_bound_B;
    
    UB=zeros(size_m,size_n,2*NP);
    UB(:,:,1:2:end)=repmat(up_bound_F,size_m,size_n,NP);
    UB(:,:,2:2:end)=repmat(up_bound_B,size_m,size_n,NP);
            
    chf0=zeros(size_m,size_n);chb0=zeros(size_m,size_n);  
    for i=1:size_m
        for j=1:size_n
            global_ind=find(tri_ind~=0);
            ind=sub2ind([size_m,size_n],i,j);
            if tri_s(i,j)==1 %back=img
                chb0(i,j)=global_ind(ind);
            end
            if tri_s(i,j)==3 %fore=img
                chf0(i,j)=global_ind(ind);
            end
        end
    end
    
    pop=zeros(size_m,size_n,2*NP);
    for k=1:NP
        chf=pop(:,:,2*k-1); chf(tri_s==3)=chf0(tri_s==3);
        chb=pop(:,:,2*k); chb(tri_s==1)=chb0(tri_s==1);
    
        for i=1:size_m
            for j=1:size_n
                if tri_s(i,j)==2
                    chf(i,j)=1+ceil(rand()*(up_bound_F-1));
                    chb(i,j)=1+ceil(rand()*(up_bound_B-1));
                end
            end
        end
        pop(:,:,2*k-1)=chf;pop(:,:,2*k)=chb;
    end
            
    iter = 0;
    while (iter <= max_iter)
        
        % evolve 1
        [pop]=TPC(pop,UB,tri_s,chf0,chb0);
        
        % evolve 2
        [pop]=LCM(pop,UB,tri_s,chf0,chb0);
        
        % evolve 3
        pop=GGM(pop,UB,tri_s,chf0,chb0);
        
        [fitness,~,fitness_NP]=eval_pop(pop,mInfo,tri_s,tri_ind);
        
         %select
        [pop]=SEL(pop,fitness,fitness_NP,tri_s);
        
        iter=iter+1;
    end
    
    [fitness,est_alpha,fitness_NP]=eval_pop(pop,mInfo,tri_s,tri_ind);
    [best_fval,idx]=min(fitness);
    best_chf=pop(:,:,2*idx-1);best_chb=pop(:,:,2*idx);
    best_est_alpha=est_alpha(idx,:);
end

function [pop]=SEL(pop,fitness,fitness_NP,tri_s)
    [size_m,size_n,NP]=size(pop);
    NP=NP/2;
    
    ch_id=1 + ceil(rand(1,NP)*(NP-1));

    fit1=zeros(size_m,size_n,2*NP);fit2=zeros(size_m,size_n,2*NP);
    for i=1:NP
        fit_tmp1=fit1(:,:,2*i-1);
        fit_tmp1(tri_s==2)=fitness_NP(i,:);
        fit1(:,:,2*i-1)=fit_tmp1;fit1(:,:,2*i)=fit_tmp1;
        
        fit_tmp2=fit2(:,:,2*i-1);
        fit_tmp2(tri_s==2)=fitness_NP(ch_id(1,i),:);
        fit2(:,:,2*i-1)=fit_tmp2;fit2(:,:,2*i)=fit_tmp2;
    end
    
    pop1=pop;
    pop1(:,:,1:2:end)=pop(:,:,2*ch_id-1);pop1(:,:,2:2:end)=pop(:,:,2*ch_id);
    pop(fit2<fit1)=pop1(fit2<fit1);
end

function [pop]=TPC(pop,UB,tri_s,chf0,chb0)
    [size_m,size_n,NP]=size(pop);
    NP=NP/2;
    %portion of individuals to crossover
    Pc=0.5;
    part=round(NP*Pc);
    
    cid=randperm(NP,part);
    pop_part=zeros(size_m,size_n,2*part);
    pop_part(:,:,1:2:end)=pop(:,:,2*cid-1);pop_part(:,:,2:2:end)=pop(:,:,2*cid);

    if rand()<0.5
        TPC_tensor_left=zeros(size_m,size_m,2*part);
        sp_r=1+ceil(rand()*(size_m-1)); L_r=1+ceil(rand()*(size_m-1));
        diag_r=ones(size_m,1);
        for i=1:L_r
            if mod(sp_r+i-1,size_m)==0
                diag_r(size_m,:)=0;
            else
                diag_r(mod(sp_r+i-1,size_m),:)=0;
            end
        end
        TPC_tensor_left(:,:,1)=diag(diag_r);TPC_tensor_left(:,:,3)=diag(1-diag_r);
        %
        pop1_part=tprod(TPC_tensor_left,pop_part);
    else
        TPC_tensor_right=zeros(size_n,size_n,2*part);
        sp_l=1+ceil(rand()*(size_n-1)); L_l=1+ceil(rand()*(size_n-1));
        diag_l=ones(size_n,1);
        for i=1:L_l
            if mod(sp_l+i-1,size_n)==0
                diag_l(size_n,:)=0;
            else
                diag_l(mod(sp_l+i-1,size_n),:)=0;
            end
        end
        TPC_tensor_right(:,:,1)=diag(diag_l);TPC_tensor_right(:,:,3)=diag(1-diag_l);
        pop1_part=tprod(pop_part,TPC_tensor_right);
    end

    pop1=pop;
    pop1(:,:,2*cid-1)=pop1_part(:,:,1:2:end); pop1(:,:,2*cid)=pop1_part(:,:,2:2:end);

    bw=pop1>UB|pop1<1;
    pop1(bw)=pop(bw);

    pop=pop1;
    tri_s_pop=repmat(tri_s,1,1,NP*2);
    ch_pop=zeros(size_m,size_n,2*NP);
    ch_pop(:,:,1:2:end)=repmat(chf0,1,1,NP);
    ch_pop(:,:,2:2:end)=repmat(chb0,1,1,NP);
    pop(tri_s_pop~=2)=ch_pop(tri_s_pop~=2); 
end

function [pop]=LCM(pop,UB,tri_s,chf0,chb0)
    [size_m,size_n,NP]=size(pop);
    NP=NP/2;
    
    %portion of individuals to mutate
    Pml=0.08;
    part=round(NP*Pml);
    
    cid=randperm(NP,part);
    pop_part=zeros(size_m,size_n,2*part);
    pop_part(:,:,1:2:end)=pop(:,:,2*cid-1);pop_part(:,:,2:2:end)=pop(:,:,2*cid);
    % padding
    pop_part_pad=padarray(pop_part,[1,1],'replicate','both');

    kernel=[1/9 1/9 1/9;1/9 1/9 1/9;1/9 1/9 1/9];
    LM_tensor1=zeros(size_m+2,size_m+2,part*2);
    LM_tensor2=zeros(size_m+2,size_m+2,part*2);
    LM_tensor3=zeros(size_m+2,size_m+2,part*2);

    w1=zeros(size_m+2,size_m+2);w2=zeros(size_m+2,size_m+2);w3=zeros(size_m+2,size_m+2);
    for i=2:size_m+1
        w1(i,i-1:i+1)=kernel(:,1)';w2(i,i-1:i+1)=kernel(:,2)';w3(i,i-1:i+1)=kernel(:,3)';
    end

    LM_tensor1(:,:,1)=w1;LM_tensor2(:,:,1)=w2;LM_tensor3(:,:,1)=w3;
    pop_part_pad1=tprod(LM_tensor1,pop_part_pad);
    pop_part_pad2=tprod(LM_tensor2,pop_part_pad);
    pop_part_pad3=tprod(LM_tensor3,pop_part_pad);

    %unpad
    pop1_part=pop_part_pad1(2:end-1,1:end-2,:)+pop_part_pad2(2:end-1,2:end-1,:)+pop_part_pad3(2:end-1,3:end,:);

    pop1=pop;
    pop1(:,:,2*cid-1)=pop1_part(:,:,1:2:end); pop1(:,:,2*cid)=pop1_part(:,:,2:2:end);

    bw=pop1>UB|pop1<1;
    pop1(bw)=pop(bw);

    pop=pop1;
    tri_s_pop=repmat(tri_s,1,1,NP*2);
    ch_pop=zeros(size_m,size_n,2*NP);
    ch_pop(:,:,1:2:end)=repmat(chf0,1,1,NP);
    ch_pop(:,:,2:2:end)=repmat(chb0,1,1,NP);
    pop(tri_s_pop~=2)=ch_pop(tri_s_pop~=2); 
end

function [pop]=GGM(pop,UB,tri_s,chf0,chb0)

    [size_m,size_n,NP]=size(pop);
    NP=NP/2;Pmg=0.01;
    ub_f=UB(1,1,1);ub_b=UB(1,1,2);
    
    GM_tensor=zeros(size_m,size_n,2*NP);
    GM_tensor(:,:,1:2:end)=(rand(size_m,size_n,NP)<Pmg).*(normrnd(0,0.1,[size_m,size_n,NP]).*(ub_f-1));
    GM_tensor(:,:,2:2:end)=(rand(size_m,size_n,NP)<Pmg).*(normrnd(0,0.1,[size_m,size_n,NP]).*(ub_b-1));
    
    pop1=abs(round(GM_tensor+pop));
    bw=pop1>UB|pop1<1;
    pop1(bw)=pop(bw);

    pop=pop1;
    tri_s_pop=repmat(tri_s,1,1,NP*2);
    ch_pop=zeros(size_m,size_n,2*NP);
    ch_pop(:,:,1:2:end)=repmat(chf0,1,1,NP);
    ch_pop(:,:,2:2:end)=repmat(chb0,1,1,NP);
    pop(tri_s_pop~=2)=ch_pop(tri_s_pop~=2);
end


