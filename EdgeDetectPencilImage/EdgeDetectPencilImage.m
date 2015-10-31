function EdgeDetectPencilImage
I = imread('multifighter.jpg');
[M,N,X] = size(I);
P = imread('pencil.jpg');
P = rgb2gray(P);
P = im2double(P);
P = imresize(P, [M, N]);
I = im2double(I);
I_hsi = rgb2hsv(I);
I_h = I_hsi(:,:,1);
I_s = I_hsi(:,:,2);
I_v = I_hsi(:,:,3);
I_s = I_s .* P;
I_edgeS = edge(I_v, 'Sobel');
I_edgeP = edge(I_v, 'Prewitt');
I_edgeR = edge(I_v, 'Roberts');
I_edgeC = edge(I_v, 'Canny');
I_edge = I_edgeS + I_edgeP + I_edgeR + I_edgeC; 
I_edge(I_edge>1) = 1;
I_edge = logical(I_edge);
I_v(I_edge) = 0;
I_out = cat(3, I_h, I_s, I_v);
I_out_rgb = hsv2rgb(I_out);
imwrite(I_out_rgb, 'pencilcolormultifighter.jpg');

I_gray = rgb2gray(I);
I_gray_edgeS = edge(I_gray, 'Sobel');
I_gray_edgeP = edge(I_gray, 'Prewitt');
I_gray_edgeR = edge(I_gray, 'Roberts');
I_gray_edgeC = edge(I_gray, 'Canny');
I_gray_edge = I_gray_edgeS + I_gray_edgeP + I_gray_edgeR + I_gray_edgeC; 
I_gray_edge(I_gray_edge > 1) = 1;
I_out_gray = 1 - I_gray_edge;
imwrite(I_out_gray, 'pencilgraymultifighter.jpg');

end
%{
IR = I(:,:,1);
IG = I(:,:,2);
IB = I(:,:,3);

I_edge_sobelR = edge(IR, 'Sobel');
I_edge_sobelG = edge(IG, 'Sobel');
I_edge_sobelB = edge(IB, 'Sobel');
I_edge_prewittR = edge(IR, 'Prewitt');
I_edge_prewittG = edge(IG, 'Prewitt');
I_edge_prewittB = edge(IB, 'Prewitt');
I_edge_robertsR = edge(IR, 'Roberts');
I_edge_robertsG = edge(IG, 'Roberts');
I_edge_robertsB = edge(IB, 'Roberts');
I_edge_cannyR = edge(IR, 'Canny');
I_edge_cannyG = edge(IG, 'Canny');
I_edge_cannyB = edge(IB, 'Canny');
%I_edge_log = edge(I, 'log');
%I_edge_zerocross = edge(I, 'zerocross');
I_edge_R = I_edge_sobelR + I_edge_prewittR + I_edge_robertsR + I_edge_cannyR;
I_edge_G = I_edge_sobelG + I_edge_prewittG + I_edge_robertsG + I_edge_cannyG;
I_edge_B = I_edge_sobelB + I_edge_prewittB + I_edge_robertsB + I_edge_cannyB;
I_edge_R(I_edge_R == 1) = 64;
I_edge_R(I_edge_R == 2) = 128;
I_edge_R(I_edge_R == 3) = 192;
I_edge_R(I_edge_R == 4) = 255;
I_edge_G(I_edge_G == 1) = 64;
I_edge_G(I_edge_G == 2) = 128;
I_edge_G(I_edge_G == 3) = 192;
I_edge_G(I_edge_G == 4) = 255;
I_edge_B(I_edge_B == 1) = 64;
I_edge_B(I_edge_B == 2) = 128;
I_edge_B(I_edge_B == 3) = 192;
I_edge_B(I_edge_B == 4) = 255;
I_edge_all = cat(3, I_edge_R, I_edge_G, I_edge_B); 
I_edge_all = 255 - I_edge_all;
imshow(I_edge_all);
%}



%{
 clear all;
clc;
close all;
%enchaned LIC pencil drawing
I=imread('figure.bmp');

[M,map]=gray2ind(I,256);
figure;imshow(I);title('原图');
X=im2double(I);
[m,n,z]=size(X);
II=zeros(m,n);
for i=1:m-1
    for j=1:n-1
        II(i,j,1)=2*sqrt((X(i,j,1)-X(i+1,j,1))^2+(X(i,j,1)-X(i,j+1,1))^2);%R分量
        II(i,j,2)=2*sqrt((X(i,j,2)-X(i+1,j,2))^2+(X(i,j,2)-X(i,j+1,2))^2);%G分量
        II(i,j,3)=2*sqrt((X(i,j,3)-X(i+1,j,3))^2+(X(i,j,3)-X(i,j+1,3))^2);%B分量
    end
end
II1=1-II;
% figure;imshow(II1);title('反相图');
M1=rgb2gray(II1);
figure;imshow(M1);title('灰度图');
I1=rgb2gray(I);
figure;imshow(I1);title('原图灰度化');
 [c,s]=wavedec2(I1,2,'bior3.7');
%分别对各频率成分进行重构
a2=wrcoef2('a',c,s,'bior3.7',2);
h2=wrcoef2('h',c,s,'bior3.7',2);
v2=wrcoef2('v',c,s,'bior3.7',2);
d2=wrcoef2('d',c,s,'bior3.7',2);
a12=uint8(a2);
h12=uint8(h2);
v12=uint8(v2);
d12=uint8(d2);
cc=[a12,h12;v12,d12];

figure,imshow(cc);title('重构图');
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

for i=1:m
    for j=1:n
        p=rand(1);
        T1=0.9*(1-a2(i,j)/255);
        T2=0.7*(1-a2(i,j)/255);
        T3=0.7*(1-a2(i,j)/255);
        if a2(i,j)<=25
            if p>T1
                aa2(i,j)=255;
            else
               aa2(i,j)=0;
            end
            else if a2(i,j)<=80
                    if p>T2
                        aa2(i,j)=255;
                    else
                        aa2(i,j)=0;
                    end
                else if a2(i,j)>80
                    if p>T3
                    aa2(i,j)=255;
                    else
                    aa2(i,j)=0;
                    end
                    end
                end
        end
    end
end
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
m1=mean2(a2);
m2=mean2(h2);
m3=mean2(v2);
m4=mean2(d2);
noise2=imnoise(h2,'gaussian',m2,1 );
noise3=imnoise(v2,'gaussian',m3,1);
noise4=imnoise(d2,'gaussian',m4,1 );
muban=fspecial('motion',15,45);
aa2=imfilter(aa2,muban);
muban1=fspecial('motion',10,-60);
hh2=imfilter(noise2,muban1);
muban2=fspecial('motion',10,45);
vv2=imfilter(noise3,muban2);
muban3=fspecial('motion',10,135);
dd2=imfilter(noise4,muban3);
 max1=max(max(aa2));
aa2=aa2/max1;
cc1=[aa2,hh2;vv2,dd2];
figure,imshow(cc1);title('模糊图');
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% hhh2=imadjust(hh2,[0 1],[0.5 1]);
% vvv2=imadjust(vv2,[0 1],[0.5 1]);
% ddd2=imadjust(dd2,[0 1],[0.5 1]);
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% for i=1:m
%     for j=1:n
%         if hh2(i,j)>0.1
%             hhh2(i,j)=hh2(i,j)-0.1;
%         end
%          if vv2(i,j)>0.1
%             vvv2(i,j)=vv2(i,j)-0.1;
%          end
%          if dd2(i,j)>0.1
%             ddd2(i,j)=dd2(i,j)-0.1;
%          end
%     end
% end
% cc2=[aa2,hhh2;vvv2,ddd2];
% figure,imshow(cc2);title('模糊图');

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
M1m=mean2(M1);
for i=1:m
    for j=1:n
        if M1(i,j)<(M1m+0.08)
        if 0<=M1(i,j)<(M1m-0.4)
            fusion(i,j)=M1(i,j)*dd2(i,j);
        else
            if M1(i,j)<(M1m-0.3)
                fusion(i,j)=M1(i,j)*vv2(i,j);
            else
%                 Banana 0.98
                if M1(i,j)<(M1m-0.2)
                    fusion(i,j)=M1(i,j)*hh2(i,j);
                else
                    if M1(i,j)<=(M1m+0.08)
                        fusion(i,j)=M1(i,j)*aa2(i,j);
                    end
                end
            end
        end
        else
        fusion(i,j)=M1(i,j);
        end
    end
end 


figure;imshow(fusion);
r=0.4;
for i=1:m
    for j=1:n
        touming(i,j)=r*M1(i,j)+(1.0-r)*fusion(i,j);
    end
end 
mm=mean2(fusion);
for i=1:m
    for j=1:n
        if touming(i,j)>mm
            touming1(i,j)=touming(i,j);
        else
            touming1(i,j)=fusion(i,j);
        end
    end
end
for i=1:m
    for j=1:n
       touming2(i,j)=touming1(i,j)*vv2(i,j);
    end
end
figure;imshow(touming);title('铅笔画');
 figure;imshow(touming1);title('最终铅笔画');
% figure;imshow(touming2);title('铅笔画');
% 
% H=cmap(:,:,1);
% S=cmap(:,:,2);
% V=cmap(:,:,3);
% V1=fusion;
% S1=S;
% H1=1-H;
% [r,g,b]=hsv2rgb(H1,S1,V1);
% colorimg(:,:,1)=r;
% colorimg(:,:,2)=g;
% colorimg(:,:,3)=b;
% figure;imshow(colorimg);
% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% 
% [L,a,b]=RGB2Lab(X(:,:,1),X(:,:,2),X(:,:,3));
% L=L/max(max(L));
% imt=fusion;
% L1=imt(:,:,1);
% for i=1:m
%     for j=1:n
%         if L(i,j)==L1(i,j)
%             a1(i,j)=a(i,j);
%             b1(i,j)=b(i,j);
%         else
%             a1(i,j)=L1(i,j);
%             b1(i,j)=L1(i,j);
%         end
%     end
% end
% [r,g,b]=Lab2RGB(L1,a1,b1);
% colorimg(:,:,1)=r;
% colorimg(:,:,2)=g,
% colorimg(:,:,3)=b;
% figure;imshow(colorimg);
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% SECTION TITLE


imt=fusion;%
ims=I;%img2是一幅彩色图像
[sx sy sz]=size(imt);
[tx ty tz]=size(ims);
ims=im2double(ims);
if sz~=1
    imt=rgb2gray(imt);%img1若不是灰度图像将其转换为灰度图像
end
if tz~=3
    disp ('img2 must be a color image (not indexed)');
else
    imt(:,:,2)=imt(:,:,1);
    imt(:,:,3)=imt(:,:,1);

% for i=1:sx
%     for j=1:sy
%         %for k=1:sz
%             imt(i,j,1)=imt(i,j,1)*ims(i,j,1);
%             imt(i,j,2)=imt(i,j,2)*ims(i,j,2);
%             imt(i,j,3)=imt(i,j,3)*ims(i,j,3);
%        % end
%     end
% end
% % imt(:,:,1)=ims(:,:,1);
% % imt(:,:,2)=ims(:,:,2);
% % imt(:,:,3)=ims(:,:,3);
% figure,imshow(imt);
% % Converting to ycbcr color space
    nspace1=rgb2ycbcr(ims);
    nspace2= rgb2ycbcr(imt);

    ms=double(nspace1(:,:,1));
    mt=double(nspace2(:,:,1));
    m1=max(max(ms));
    m2=min(min(ms));
    m3=max(max(mt));
    m4=min(min(mt));
    d1=m1-m2;
    d2=m3-m4;
% Normalization
    dx1=ms;
    dx2=mt;
    dx1=(dx1*1)/(1-d1);
    dx2=(dx2*1)/(1-d2);
    [mx,my,mz]=size(dx2);
%Luminance亮度 Comparison
    disp('Please wait..................');
    for i=1:mx
        for j=1:my
             iy=dx2(i,j);
             tmp=abs(dx1-iy);
             ck=min(min(tmp));
             [r,c] = find(tmp==ck);
             ck=isempty(r);
             if (ck~=1)            
                 nimage(i,j,2)=nspace1(r(1),c(1),2);
                 nimage(i,j,3)=nspace1(r(1),c(1),3);
                 nimage(i,j,1)=nspace2(i,j,1);           
             end
         end
     end
    rslt=ycbcr2rgb(nimage);
    figure,imshow(imt);
    figure,imshow(rslt);
end
%}