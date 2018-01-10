% This Matlab file demomstrates a level set algorithm based on Chunming Li et al's paper:
% "Implicit Active Contours Driven By Local Binary Fitting Energy" in Proceedings of CVPR'07

clc;clear all;close all;
c0 =2;
imgID=3;
% 
% fn='article_pictures/noisyNonUniform.bmp';
fn='mypictures/qh_wy_s1_c1_gr.png';
fn='mypictures/s1_fl.png';
% img = imread('mypictures/qh_wy_s1_c1.png');
% img = rgb2gray(img);
% imwrite(img, fn);
% Img=((imread(fn)));
% Img=rgb2gray(Img);
% Img=((imread('D:\algorithm\test4.jpg')));
% [height,width]=size(Img);  %??????????????????????????????????  
% p=zeros(1,256);                            %?????????????????????????????????????????????????????????  
% for i=1:height  
%     for j=1:width  
%      p(Img(i,j) + 1) = p(Img(i,j) + 1)  + 1;  
%     end  
% end  
% s=zeros(1,256);  
% s(1)=p(1);  
% for i=2:256  
%      s(i)=p(i) + s(i-1); %??????????????????????<?????????????????????????????????????????????????s(i):0,1,```,i-1  
% end  
%   
% for i=1:256  
%     s(i) = s(i)*256/(width*height); %?????????????????????????  
%     if s(i) > 256  
%         s(i) = 256;  
%     end  
% end  
%   
% %??????????????  
% I_equal = Img;  
% for i=1:height  
%     for j=1:width  
%      I_equal(i,j) = s( Img(i,j) + 1);  
%     end  
% end  
% % % figure,imshow(I_equal)                           %??????????????????????????????????   
% % % title('???????????????????????')  
% % % imwrite(I_equal,'1_equal.bmp');  
% % 
% % % Img=rgb2gray(Img);
% % % % Img=double(Img);
% % Img=I_equal;
% M=2*size(Img,1);  
% N=2*size(Img,2);                        %??????????????????????????????????????    
% u=-M/2:(M/2-1);  
% v=-N/2:(N/2-1);  
% [U,V]=meshgrid(u,v);  
% D=sqrt(U.^2+V.^2);  
% D0=95;  
% H=exp(-(D.^2)./(2*(D0^2)));          %???????????????????????????  
% J=fftshift(fft2(Img,size(H,1),size(H,2)));  
% G=J.*H;  
% L=ifft2(fftshift(G));  
% L=L(1:size(Img,1),1:size(Img,2));  
% 
% I=floor(real(L));
% Img=I;

% % Img=imread('vessel2.bmp');  % uncommont this line to use ther other vessel image
% % I=Img(:,:,1);
% imagesc(I);
% H=fspecial('laplacian');    
%????????laplacian??????????????????????????????????
% laplacianH=filter2(H,Img);
% Img=laplacianH;

% K = wiener2(I,[3 3]);
% I=Img;
% I=uint8(I);
% imwrite(I,'test01.jpg','jpg');
% 
% Img=((imread('test01.jpg')));
% Img=rgb2gray(Img);
% Img=imread('vessel2.bmp');  % uncommont this line to use ther other vessel image
% Img=Img(:,:,3);
% Img=((imread('test01.jpg')));
Img=imread(fn);
Img=double(Img);
switch imgID
     case 1
       phi= ones(size(Img(:,:,1))).*c0;
       a=43;b=51;c=20;d=28;
       phi(a:b,c:d) = -c0;
       figure;
       imshow(I);colormap;
       hold on;
       plotLevelSet(phi, 0, 'g');
       hold off;
    case 2
       [m,n]=size(Img(:,:,1));
       a=m/2; b=n/2;r=5;
       phi= ones(m,n).*c0;
       phi(a-r:a+r,b-r:b+r) = -c0;
       imshow(I);colormap;
       hold on;
       plotLevelSet(phi, 0, 'r');
       hold off;
    case 3
       figure;imagesc(Img, [0, 255]);colormap(gray);hold on; axis off;axis equal;
%        figure;imshow(Img);hold on; axis off;axis equal;
%        text(6,6,'Left click to get points, right click to get end point','FontSize',[12],'Color', 'g');
       BW=roipoly;
       phi=c0*2*(0.5-BW);
       hold on;
       [c,h] = contour(phi,[0 0],'r');
       hold off;
end
pause(0.01);

%????????????????????
iterNum = 500;
lambda1 = 1.0;
lambda2 = 1.0;
nu = 0.03*255*255;
timestep = 0.1;
mu = 1;
epsilon = 2.0;

% scale parameter in Gaussian kernel
sigma=2.0;    
K=fspecial('gaussian',round(2*sigma)*2+1,sigma); % Gaussian kernel
KI=conv2(Img,K,'same');  
KONE=conv2(ones(size(Img)),K,'same');


oriImg=((imread(fn)));
% start level set evolution
imshow(oriImg);hold on; axis off;

%%%%%%%%%%%%%%%%%%%%%%%%%%% video   %%%%%%%%%%%%
write_video = 1;
if write_video
    myObj = VideoWriter('s1.avi');%?????avi??
    myObj.FrameRate = 5;
    open(myObj);
end

time = cputime;
for n=1:iterNum
   numIter=1;
    %level set evolution.  
%     phi=EVOL_LBF(phi,Img,K,KI,KONE,nu,timestep,mu,lambda1,lambda2,epsilon,numIter);
    u=phi;
    for k1=1:numIter
        u=NeumannBoundCond(u);
        C=curvature_central(u);    % div()  
        HeavU=Heaviside(u,epsilon);
        DiracU=Dirac(u,epsilon);

        [f1,f2]=LBF_LocalBinaryFit(K,Img,KI,KONE,HeavU);    
        LBF=LBF_dataForce(Img,K,KONE,f1,f2,lambda1,lambda2);

        areaTerm=-DiracU.*LBF;
        penalizeTerm=mu*(4*del2(u)-C);
        lengthTerm=nu.*DiracU.*C;
        u=u+timestep*(lengthTerm+penalizeTerm+areaTerm);
        phi=u;
    end
%     pause(0.1);
%     imagesc(u);
%     if mod(n,10)==0
        pause(0.001);
        
        imshow(oriImg);hold on; axis off;
        [Ct, h] = contour(phi,[0 0],'r');
        
        if write_video
            sz = size(oriImg);
            v_img = zeros(sz(1), sz(2), 3, 'uint8');
            v_img(1:end, 1:end, 1) = oriImg;
            v_img(1:end, 1:end, 2) = oriImg;
            v_img(1:end, 1:end, 3) = oriImg;
            cl = [255, 0, 0];

            i = 0;
            it_e = 0;
            while i ~= length(Ct)
                it_e = Ct(2, i+1)+it_e+1;
                it_s = i+2;
                for i = it_s:it_e
                    v_img(uint16(Ct(2, i)), uint16(Ct(1, i)), :) = cl;
                end
            end
            writeVideo(myObj,v_img);
        end
        iterNum=['GagoBigData/ iter:',num2str(n) ];
        title(iterNum);
        hold off;
%     end
end
totaltime = cputime - time

if write_video
    close(myObj);
end

oriImg=((imread(fn)));
imshow(oriImg);hold on; axis off;
contour(phi,[0 0],'r','LineWidth',0.5);
iterNum=[num2str(n), ' iter/ GagoBigData'];
title(iterNum);







