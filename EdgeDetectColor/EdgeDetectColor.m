function EdgeDetectColor
I = imread('colorlena.jpg');
I = im2double(I);
I_gray = rgb2gray(I);
I_out = edge(I_gray);
imwrite(I_out, 'graylena.jpg');
I_R = I(:,:,1);
I_G = I(:,:,2);
I_B = I(:,:,3);
I_Redge = edge(I_R);
I_Gedge = edge(I_G);
I_Bedge = edge(I_B);
[M,N] = size(I_G);
I_new = (I_Redge+I_Gedge+I_Bedge)/3.0;
imwrite(I_new, 'average.jpg');
imwrite(I_Redge, 'Redge.jpg');
imwrite(I_Gedge, 'Gedge.jpg');
imwrite(I_Bedge, 'Bedge.jpg');
end