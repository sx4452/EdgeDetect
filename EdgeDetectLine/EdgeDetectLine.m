
function EdgeDetectLine
I = imread('circuit.png');
I = rgb2gray(I);
I = im2double(I);
%imwrite(I, 'circuitgray.jpg');
I_edge = edge(I);
k=5;
%imwrite(I_edge, 'edgeout.jpg');
[H, theta, rho] = hough(I_edge);
%nhoodsize = uint8(0.1*size(H)/50);
peaks = houghpeaks(H, k);
lines = houghlines(I_edge, theta, rho, peaks, 'FillGap', 100);
figure
imshow(I_edge)
hold on    
for k=1:length(lines)    
    xy=[lines(k).point1;lines(k).point2];    
    plot(xy(:,1),xy(:,2),'LineWidth',1,'Color','red');    
end  
end
