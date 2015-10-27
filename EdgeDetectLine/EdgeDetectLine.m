function EdgeDetectLine
I = imread('singlefighter.jpg');
I = im2double(I);
I_edge = edge(I);
imwrite(I_edge, 'edgeout.jpg');
[H, theta, rho] = hough(I_edge);
peaks = houghpeaks(H);
lines = houghlines(H, theta, rho, peaks);

end