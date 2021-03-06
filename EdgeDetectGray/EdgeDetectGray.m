function EdgeDetectGray
I = imread('lena.jpg');
I = rgb2gray(I);
I = im2double(I);
imwrite(I, 'lenagray.jpg');
maskvertical = [-1 -2 -1;
                -1 2 -1;
                -1 2 -1];
maskpi4 = [-1 -1 2;
           -1 2 -1;
           2 -1 -1];
maskrevpi4 = [2 -1 -1;
              -1 2 -1;
              -1 -1 2];
I_vertical = imfilter(I, maskvertical);
I_pi4 = imfilter(I, maskpi4);
I_revpi4 = imfilter(I, maskrevpi4);
I_vertical = Normalized(I_vertical);
I_pi4 = Normalized(I_pi4);
I_revpi4 = Normalized(I_revpi4);
imwrite(I_vertical, 'lena_outvertical.jpg');
imwrite(I_pi4, 'lena_outpi4.jpg');
imwrite(I_revpi4, 'lena_outrevpi4.jpg');
end

function I_out = Normalized(I_in)
    minV = min(min(I_in));
    maxV = max(max(I_in));
    I_out = (I_in - minV)/(maxV - minV);
end
%{
[M,N] = size(I);
I_new1 = zeros(size(I));
I_new2 = zeros(size(I));
tic;
for i = 1:M-2
    for j = 1:N-2
        c=I(i:i+2,j:j+2).*mask; 
        s=sum(sum(c));                 
        I_new1(i+1,j+1)=s/9;
    end
end
toc;
tic;
minV = min(min(I_new2));
maxV = max(max(I_new2));
I_new2 = (I_new2 - minV)/(maxV - minV);
imwrite(I_new2, 'lena_out2.jpg');
%}