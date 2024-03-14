clear all
clc
JJD=csvread('./welldata.csv',0,0);%csvread只能读取纯数据,excel中只保留数据
data=JJD;%读入数据
%DENPTH  AC  CNC  DEN GR   RT  RXO SP

%% 1、泥质含量计算
%%GR 自然伽马法
GR=data(:,5);
Vsh1=(data(:,5)-min(data(:,5)))./(max(data(:,5))-min(data(:,5)));
GCUR=3.7;%经验系数老地层用2，新地层3.7
Vsh1=(2.^(GCUR*Vsh1)-1)./(2.^GCUR-1);
%%SP 自然电位法
SP=data(:,8);
Vsh2=(data(:,8)-min(data(:,8)))./(max(data(:,8))-min(data(:,8)));

%%泥质含量计算
GR_SP_muddy=[Vsh1,Vsh2]';
GR_SP_muddy_min=min(GR_SP_muddy);

depth=data(:,1);
figure(1)
plot(GR_SP_muddy_min',-depth)
xlabel("泥质含量")
ylabel("深度/m")
title("泥质含量计算结果")
hold off

%%地层孔隙度计算
rho_ma=2.65;
rho_f=1.0;
%DEN
rho_b=data(:,4);
figure(2)
plot(SP,-depth)
xlabel("自然电位读数")
ylabel("深度/m")

figure(3)
plot(GR,-depth)
xlabel("自然伽马读数")
ylabel("深度/m")


rho_sh=3; %泥质密度
fai=(rho_b-rho_ma)./(rho_f-rho_ma)-GR_SP_muddy_min'.*(rho_sh-rho_ma)./(rho_f-rho_ma);
plot(fai,-depth)
figure(4)
plot(fai,-depth)
xlabel("孔隙度")
ylabel("深度/m")
title("孔隙度计算结果")
% 地层水饱和度计算
%采用阿尔奇公式计算地层含水饱和度
a=0.5;m=1.8;n=2;Rw=0.4;
Rt=data(:,6);
Sw=((a*Rw)./(fai.^n.*Rt)).^(1/n);
figure(5)
plot(Sw,-depth)
xlabel("含水饱和度")
ylabel("深度/m")
title("地层水含水饱和度计算结果")
%xlim([0,1])

% 4、地层渗透率计算
Swb=0.2;%束缚水饱和度
K=(0.136*fai.^4.4)./(Swb^2);%地层绝对渗透率K
figure(6)
plot(K,-depth)
xlabel("渗透率")
ylabel("深度/m")
title("地层渗透率计算结果")

plot(rho_b,-depth)
xlabel("密度")
ylabel("深度/m")
title("密度测井读数")

