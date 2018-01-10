%% image preprocess
clear all
clc
fn='mypictures/haha.jpg';

Img=imread(fn);

% img = rgb2gray(Img(:,:,1:3));
% imwrite(img,'xixi.jpg')

% kernel1 = fspecial('gaussian',9,3.0);
% kernel2 = fspecial('gaussian',3,1.0);
% 
% f1 = filter2(kernel1,Img);
% f11 = imresize(f1,[900,900]);
% 
% f2 = filter2(kernel2,f11);
% f22 = imresize(f2,[500,500]);
% f22 = uint8(f22);

f22 = imresize(Img,[500,500]);

imwrite(f22,'mypictures/haha1.jpg');


%% %%%%%
% figure
% ed = size(phi);
% for i =1:ed(1)
%     scatter(1:ed(2),phi(:,i))
%     hold on
% end