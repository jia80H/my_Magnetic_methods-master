clear all;close all;clc;
u0=4*pi*10^-7;
h=1;    %%深度
r=0.5;        %%%半径
V=(4/3)*pi*r^3;
M=10;       %%%磁化强度
I0=0.5*pi/4;   %%倾斜磁化
I1=pi/2;    %%%垂直磁化
A=pi/2;     %%磁偏角
m=M*V;
X=-10:0.1:10;
Y=-10:0.1:10;
[x,y]=meshgrid(X,Y);
HX=u0*m*((2*x.^2-y.^2-h^2).*cos(I0)*sin(A)+3*x.*y.*cos(I0)*sin(A)-3*x.*h*sin(I0))./(4*pi*(x.^2+y.^2+h^2).^(5/2));
ZZ1=u0*m*((2*h^2-x.^2-y.^2).*sin(I1)-3*x.*h*cos(I1)*sin(A)-3*y.*h*cos(I1)*sin(A))./(4*pi*(x.^2+y.^2+h^2).^(5/2));
ZZ=u0*m*((2*h^2-x.^2-y.^2).*sin(I0)-3*x.*h*cos(I0)*sin(A)-3*y.*h*cos(I0)*sin(A))./(4*pi*(x.^2+y.^2+h^2).^(5/2));
% WW=((2*h^2-x.^2-y.^2).*sin(I0)^2+(2*x.^2-y.^2-h^2).*cos(I0)^2*cos(A)^2+(2*y.^2-x.^2-h^2).*cos(I0)^2*sin(A)^2-3*x.*h*sin(2*I0)*cos(A)+3*x.*y.*cos(I0)^2*sin(2*A)-3*y.h*sin(2*I0)*sin(A));
TT=u0*m*((2*h^2-x.^2-y.^2).*sin(I0)^2+(2*x.^2-y.^2-h^2).*cos(I0)^2*cos(A)^2+(2*y.^2-x.^2-h^2).*cos(I0)^2*sin(A)^2-3*x.*h*sin(2*I0)*cos(A)+3*x.*y.*cos(I0)^2*sin(2*A)-3*y.*h*sin(2*I0)*sin(A))./(4*pi*(x.^2+y.^2+h^2).^(5/2));
subplot(2,3,1)
imagesc(ZZ1);
colormap jet;colorbar;
subplot(2,3,4)
mesh(ZZ1);
subplot(2,3,2)
imagesc(ZZ);
colormap jet;colorbar;
subplot(2,3,5)
mesh(ZZ);
subplot(2,3,3)
imagesc(TT);
colormap jet;colorbar;
subplot(2,3,6)
mesh(TT);