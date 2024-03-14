function [TR_FVAlpha, TRobust, Value] = GetAlpha(I, MaskAct, X, FSample, BSample, Un)
%   Detailed explanation goes here
[Ih, Iw, ~] = size(I);
TR_FVAlpha = zeros(Ih, Iw);
TRobust = zeros(Ih, Iw);

for i = 1:size(Un, 1)
    Cur_F = FSample(X(i, 1), 1:3);
    Cur_B = BSample(X(i, 2), 1:3);
    Cur_I = reshape(I(Un(i, 1), Un(i, 2), 1:3), [1, 3]);
    TR_FVAlpha(Un(i, 1), Un(i, 2)) = sum((Cur_I - Cur_B).*(Cur_F - Cur_B), 2)./(sum((Cur_F - Cur_B).^2, 2));
end
TR_FVAlpha(MaskAct == 5) = 0;
TR_FVAlpha(MaskAct == 1) = 1;

ColorDis = 0;
SpatialDis = 0;
Value = 0;
for i = 1:size(Un, 1)
    % Color distance
    ColorDis = reshape(I(Un(i, 1), Un(i, 2), 1:3), [1, 3]) - ( TR_FVAlpha(Un(i, 1), Un(i, 2)) * FSample(X(i, 1), 1:3) + (1 - TR_FVAlpha(Un(i, 1), Un(i, 2)) ) * BSample(X(i, 2), 1:3) );
    ColorDis = exp(-sqrt(sum(ColorDis.^2, 2)));
    % Spatial distance
    NumF = size(FSample, 1);
    NumB = size(BSample, 1);
    minF = min(sqrt(sum((FSample(:, 4:5) - repmat(Un(i, :), NumF, 1) ).^2, 2)));
    minB = min(sqrt(sum((BSample(:, 4:5) - repmat(Un(i, :), NumB, 1) ).^2, 2)));
    SpatialDis  = exp(-(sqrt(sum((FSample(X(i, 1), 4:5) - Un(i, :)).^2, 2)) / minF)) + exp(-(sqrt(sum((BSample(X(i, 2), 4:5) - Un(i, :)).^2, 2)) / minB));
    % Objective Function Computation
    ValidTolerance=.2 ;
    if TR_FVAlpha(Un(i, 1), Un(i, 2)) > 1+ValidTolerance
        FVAlphaMask = 0;
    elseif TR_FVAlpha(Un(i, 1), Un(i, 2)) < 0-ValidTolerance
        FVAlphaMask = 0;
    elseif isnan(TR_FVAlpha(Un(i, 1), Un(i, 2)))
        FVAlphaMask = 0;
    else
        FVAlphaMask = 1;
    end
    if FVAlphaMask == 1
        TRobust(Un(i, 1), Un(i, 2)) = ColorDis.^2 + SpatialDis.^0.5;
    else
        TR_FVAlpha(Un(i, 1), Un(i, 2)) = 0.5;
        TRobust(Un(i, 1), Un(i, 2)) = 0;
    end
    Value = Value + ColorDis.^2 + SpatialDis.^0.5;
end

end

