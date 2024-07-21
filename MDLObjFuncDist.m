function [Wstore,Hstore,LtotStore,errorStore]=MDLObjFuncDistNew(dataSet,delta,r,nRuns)

[V,W0,H0,E0,parmW,parmH,muE,sigE,m,n]=initialisationMDLObjFunc(r,dataSet);
deltaE=delta(1);deltaW=delta(2);deltaH=delta(3);

%%% Start implementation of the gradient descent
W=W0;H=H0;E=E0;

Vp=W*H;
tinyVal=0.000001;
wLearnRate=0.000005;hLearnRate=0.000005;
LtotStore=zeros(nRuns+1,4);
errorStore=zeros(nRuns+1,1);
errorStore(1,1)=norm(V-Vp,'fro');
alphaW=parmW(1,1);betaW=1/parmW(1,2);
alphaH=parmH(1,1);betaH=1/parmH(1,2);
tempW1=W.^(alphaW-1);tempW2=exp(-betaW*W);
Pw=(1/gamma(alphaW))*deltaW*(betaW^alphaW)*tempW1.*tempW2;
tempH1=H.^(alphaH-1);tempH2=exp(-betaH*H);
Ph=(1/gamma(alphaH))*deltaH*(betaH^alphaH)*tempH1.*tempH2;
Etemp=E0-muE;Etemp2=(Etemp.^2)/(2*sigE^2);
Pe=(deltaE/(sqrt(2*(sigE^2)*pi)))*exp(-Etemp2);
logPe=log2(Pe(:));
% Then calculate the log probabilities
Le=-sum(logPe(logPe>-inf));
Lw=-sum(log2(Pw(:)));Lh=-sum(log2(Ph(:)));
LtotStore(1,1)=Le+Lw+Lh;LtotStore(1,2)=Le;
LtotStore(1,3)=Lw;LtotStore(1,4)=Lh;
num1=11;
%Wstore=cell(round(nRuns/num1),1);Hstore=cell(round(nRuns/num1),1);
Wstore=cell(num1+5,1);Hstore=cell(num1+5,1);
Wstore{1,1}=W0;Hstore{1,1}=H0;

for k=1:nRuns
    alphaW=parmW(1,1);betaW=1/parmW(1,2);
    alphaH=parmH(1,1);betaH=1/parmH(1,2);

    deltaW1=-((alphaW-1)*(1./(W+tinyVal))-betaW)/log(2);
    deltaW2=-(1/(log(2)*sigE^2))*(E-muE)*H';
    gradW=deltaW1+deltaW2;
    Wnew=W-wLearnRate*gradW;
    
    deltaH1=-((alphaH-1)*(1./(H+tinyVal))-betaH)/log(2);
    deltaH2=-(1/(log(2)*sigE^2))*Wnew'*(E-muE);
    gradH=deltaH1+deltaH2;
    Hnew=H-hLearnRate*gradH;

    VpAfter=Wnew*Hnew;E=V-VpAfter;
    smallNum1=0.002;
    Wnew=max(Wnew,smallNum1);Hnew=max(Hnew,smallNum1);
    parmW=gamfit(Wnew(:));
    parmH=gamfit(Hnew(:));
    [muE,sigE]=normfit(E(:));
    W=Wnew;H=Hnew;
    % Calculate the probabilities
    tempW1=W.^(alphaW-1);tempW2=exp(-betaW*W);
    Pw=(1/gamma(alphaW))*deltaW*(betaW^alphaW)*tempW1.*tempW2;
    tempH1=H.^(alphaH-1);tempH2=exp(-betaH*H);
    Ph=(1/gamma(alphaH))*deltaH*(betaH^alphaH)*tempH1.*tempH2;
    Etemp=E-muE;Etemp2=(Etemp.^2)/(2*sigE^2);
    Pe=(deltaE/(sqrt(2*(sigE^2)*pi)))*exp(-Etemp2);
    logPe=log2(Pe(:));
    % Then calculate the log probabilities
    Le=-sum(logPe(logPe>-inf));Lw=-sum(log2(Pw(:)));Lh=-sum(log2(Ph(:)));
    LtotStore(k+1,1)=Le+Lw+Lh;LtotStore(k+1,2)=Le;
    LtotStore(k+1,3)=Lw;LtotStore(k+1,4)=Lh;
    Vp=W*H;
    errorStore(k+1,1)=norm(V-Vp,'fro');
end
end

function [V,W0,H0,E0,parmW,parmH,muE,sigE,m,n]=initialisationMDLObjFunc(r,dataSet)
Vorig=xlsread('path-to-data');
V=Vorig/max(Vorig(:));
[m,n]=size(V);
meanV=mean(V(:));
meanWH=sqrt(meanV/r);
Aw=0.8;Bw=meanWH/Aw;
Ah=0.8;Bh=meanWH/Ah;
W0=gamrnd(Aw,Bw,m,r);
H0=gamrnd(Ah,Bh,r,n);
Vp=W0*H0;
E0=V-Vp;
parmW=gamfit(W0(:));
parmH=gamfit(H0(:));
[muE,sigE]=normfit(E0(:));
end









