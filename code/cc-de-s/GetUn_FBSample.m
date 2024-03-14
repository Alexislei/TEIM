function GetUn_FBSample(I, MaskAct)
%% Extract the set of undetermined pixels.
[~, segs, ~, ~, ~, ~] = edison_wrapper(I, @RGB2Luv,...
    'SpatialBandWidth',27,'RangeBandWidth',9,'MinimumRegionArea',30);
segs_mat = double(segs(:,:)) + 1;
segs_mat(MaskAct ~= 3) = 0;
segs_num = max(segs_mat(:));
Un = [];
Un_num = [];
for i = 1:segs_num
   [x, y] = find(segs_mat == i);
   tmp = [x, y];
    if(size(tmp, 1) >= 1)
       Un = [Un; tmp];
       Un_num = [Un_num; size(tmp, 1)];
    end
end

%% Extract the sets of known foreground sample pixels and known background sample pixels.
BW = imdilate(edge(MaskAct, 'canny'), strel('disk', 2));
FSample=[];
BSample=[];

for i = 1:size(I,1)
    for j = 1:size(I,2)
        if BW(i,j) == 1
            if(MaskAct(i,j) == 5)
                BSample = [BSample;I(i,j,1),I(i,j,2),I(i,j,3),i,j,rgb2gray(I(i,j,1:3))];
            end
            if(MaskAct(i,j) == 1)
                FSample = [FSample;I(i,j,1),I(i,j,2),I(i,j,3),i,j,rgb2gray(I(i,j,1:3))];
            end
        end
    end
end
FSample = sortrows(FSample,6);
BSample = sortrows(BSample,6);

%% Save the results
save  SetFSample.mat FSample;
save  SetBSample.mat BSample;
save  SetUn.mat Un;
save  SetUnNum.mat Un_num;