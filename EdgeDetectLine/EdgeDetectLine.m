
function EdgeDetectLine
clc
close
I = imread('circuit.png');
I = rgb2gray(I);
I = im2double(I);
I_edge = edge(I, 'canny');
%I_edge = im2double(I_edge);
imwrite(I_edge, 'edgeout.jpg');
[H, theta, rho] = hough(I_edge);
peaks = houghpeaks(H,5);
I_edge = im2double(I_edge);
lines = houghlines(I_edge, theta, rho, peaks, 'FillGap', 5);
figure
imshow(I_edge)
hold on    
for k=1:length(lines)    
    xy=[lines(k).point1;lines(k).point2];    
    plot(xy(:,1),xy(:,2),'LineWidth',4,'Color','red');    
end  
end