function EdgeDetectColor
I = imread('colorlena.jpg');
I = im2double(I);
I_gray = rgb2gray(I);
I_out = edge(I_gray);
imwrite(I_out, 'graylena.jpg');
I = rgb2hsi(I);
I_R = I(:,:,1);
I_G = I(:,:,2);
I_B = I(:,:,3);
I_Redge = edge(I_R);
I_Gedge = edge(I_G);
I_Bedge = edge(I_B);
[M,N] = size(I_G);
I_new = zeros(size(I_G));
for i = 1:M
    for j = 1:N
        I_new(i,j) = mean([I_Redge(i,j),I_Gedge(i,j),I_Bedge(i,j)]);
    end
end
imwrite(I_new, 'meanhsi.jpg');
imwrite(I_Redge, 'Redgehsi.jpg');
imwrite(I_Gedge, 'Gedgehsi.jpg');
imwrite(I_Bedge, 'Bedgehsi.jpg');
end