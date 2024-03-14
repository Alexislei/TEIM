function MaskAct = GetMask(I, Trimap)
%   Detailed explanation goes here
[Ih, Iw, ~] = size(I);
RmaskF = (Trimap > 200); RmaskB = (Trimap <50);
MaskAct = zeros(Ih, Iw);
MaskAct = RmaskB*5 + RmaskF;
MaskAct(MaskAct == 0) = 3;
end

