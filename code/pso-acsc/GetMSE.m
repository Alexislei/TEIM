function [ MSE ] = GetMSE( alpha_U,GT,U_ind )
%GETMSE Summary of this function goes here
%   Detailed explanation goes here
    alpha_U(alpha_U>1) = 1;
    alpha_U(alpha_U<0) = 0;
    I = GT;
    I(U_ind) = uint8(alpha_U*255);
    MSE = immse(I,GT);
%     imshow(I),pause(0.1);
%     disp(MSE);
end

