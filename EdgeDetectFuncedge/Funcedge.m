function Funcedge
I1 = imread('lenagray.jpg');
I1 = im2double(I1);
I_out = edge(I1, 'canny');
imwrite(I_out, 'lenagrayout.jpg');
I2 = imread('singlefighter.jpg');
I2 = im2double(I2);
I_out = edge(I2, 'canny');
imwrite(I_out, 'singlefighterout.jpg');
I3 = imread('multifighter.jpg');
I3 = im2double(I3);
I_out = edge(I3, 'canny');
imwrite(I_out, 'multifighterout.jpg');
end
