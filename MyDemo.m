% This Matlab file demomstrates a level set algorithm based on Chunming Li et al's paper:
% "Implicit Active Contours Driven By Local Binary Fitting Energy" in Proceedings of CVPR'07

clc;clear all;close all;
c0 =2;
imgID=3;

% parameter of energy functional
iterNum = 100;
lambda1 = 1.0;
lambda2 = 1.0;
nu = 0.003*255*255;%length
timestep = 0.1;
mu = 1;%area
epsilon = 2.0;


fn='mypictures/qh_gr_1.png';%iter = 100 
% fn='mypictures/test01.jpg';%iter = 100 
% fn='article_pictures/noisyNonUniform.bmp';%iter = 400
Img=imread(fn);
% Img=rgb2gray(Img);
Img=double(Img);
switch imgID
     case 1
       phi= ones(size(Img(:,:,1))).*c0;
       a=43;b=51;c=20;d=28;
       phi(a:b,c:d) = -c0;
       figure;
       imshow(Img/255);colormap;
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
       
       figure;
       imshow(phi)
       
       
end

pause(0.01);

% scale parameter in Gaussian kernel
sigma=3.0;    
K=fspecial('gaussian',round(2*sigma)*2+1,sigma); % Gaussian kernel
KI=conv2(Img,K,'same');  
KONE=conv2(ones(size(Img)),K,'same');
imshow(KI/255)
oriImg=((imread(fn)));
% start level set evolution
imshow(oriImg);hold on; axis off;

time = cputime;
u=phi;
for n=1:iterNum
%    numIter=1;
    %level set evolution.  
%     phi=EVOL_LBF(phi,Img,K,KI,KONE,nu,timestep,mu,lambda1,lambda2,epsilon,numIter);
%     u=phi;
%     for k1=1:numIter
        
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
%         phi=u;
 
        imshow(oriImg);hold on; axis off;
        contour(u,[0 0],'r');
        iterNum=['GagoBigData/ iter:',num2str(n) ];
        title(iterNum);
        hold off;
        
%        if n<3
%          pause(3)
%        else
%            pause(0.5)
%        end
    
end
totaltime = cputime - time

oriImg=((imread(fn)));
imshow(oriImg);hold on; axis off;
contour(u,[0 0],'b','LineWidth',2);
iterNum=[num2str(n), ' iter/ GagoBigData'];
title(iterNum);

figure;
imshow(u>0)
% contour(phi,[0 0])


% % Make a function satisfy Neumann boundary condition
% function g = NeumannBoundCond(f)
% [nrow,ncol] = size(f);
% g = f;
% g([1 nrow],[1 ncol]) = g([3 nrow-2],[3 ncol-2]);  
% g([1 nrow],2:end-1) = g([3 nrow-2],2:end-1);          
% g(2:end-1,[1 ncol]) = g(2:end-1,[3 ncol-2]);  
% ------------------------------------
% function k = curvature_central(u)
% % compute curvature for u with central difference scheme
% [ux,uy] = gradient(u);
% normDu = sqrt(ux.^2+uy.^2+1e-10);
% Nx = ux./normDu;
% Ny = uy./normDu;
% [nxx,junk] = gradient(Nx);
% [junk,nyy] = gradient(Ny);
% k = nxx+nyy;
% -------------------------------------
% function [f1,f2] = LBF_LocalBinaryFit(K,Img,KI,KONE,H)
% I=Img.*H;
% c1=conv2(H,K,'same');
% c2=conv2(I,K,'same');
% f1=c2./(c1);
% f2=(KI-c2)./(KONE-c1);
% --------------------------------------
% function h = Heaviside(x,epsilon)     % function (11)
% h=0.5*(1+(2/pi)*atan(x./epsilon));
% --------------------------------------
% function f = Dirac(x, epsilon)    % function (12)
% f=(epsilon/pi)./(epsilon^2.+x.^2);
% --------------------------------------
% function f=LBF_dataForce(Img,K,KONE,f1,f2,lamda1,lamda2)
% s1=lamda1.*f1.^2-lamda2.*f2.^2;
% s2=lamda1.*f1-lamda2.*f2;
% f=(lamda1-lamda2)*KONE.*Img.*Img+conv2(s1,K,'same')-2.*Img.*conv2(s2,K,'same');



