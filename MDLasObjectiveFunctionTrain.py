import numpy as np
from scipy.stats import gamma,norm
from scipy import special
### requires subspace size r and original matrix V

def initialise(r,Vorig):  ### convert Vorig values to maximum of 1. Set W0, H0 to a
#### sensible starting set of values. Produce first set of parameters
    V=Vorig/np.amax(Vorig.flatten())
    m,n=V.shape
    meanV=np.mean(V.flatten())
    meanWH=np.sqrt(meanV/r)
    Aw,Ah=0.8,0.8
    Bw,Bh=meanWH/Aw,meanWH/Ah
    W0=np.random.gamma(Aw,scale=Bw,size=(m,r))
    H0=np.random.gamma(Ah,Bh,size=(r,n))
    Vp=np.matmul(W0,H0)
    E0=V-Vp
    alphaW,betaW,aH,bH,muE,sigmaE=returnParams(W0,H0,E0)
    return V,W0,H0,E0,alphaW,betaW,aH,bH,muE,sigmaE,m,n
def returnParams(W,H,E):#### fit the gamma and Gaussian distribution
    alphaW,locW,scaleW=gamma.fit(W.flatten())
    betaW=1/scaleW
    aH,locH,scaleH=gamma.fit(H.flatten())
    bH=1/scaleH
    muE,sigmaE=norm.fit(E)
    return alphaW,betaW,aH,bH,muE,sigmaE
def returnWorHtilde(WorH,alpha,beta):
    tinyVal=1e-5 #### stop infinities
    fac1=-((alpha-1)*(1/(WorH+tinyVal))-beta)
    return fac1/(np.log(2))
def returnEtilde(E,mu,sigma):
    fac1=-(1/(np.log(2)*sigma**2))
    fac2=(E-mu)
    return fac1*fac2
def returnGammaProbs(alpha,beta,WorH,deltaWorH):
    tinyVal=1e-5 #### stop infinities
    fac1=(1/(tinyVal+special.gamma(alpha)))*(beta**alpha)
    fac2=WorH**(alpha-1)
    fac3=np.exp(-beta*WorH)
    probs=deltaWorH*fac1*fac2*fac3
    return probs
def returnNormProbs(E,mu,sigma,deltaE):
    return deltaE*(1/(np.sqrt(2*np.pi*sigma**2)))*np.exp(-((E-mu)**2)/(2*sigma**2))
def returnObjFunc(ProbW,ProbH,ProbE):
    tinyVal=1e-5 #### stop infinities
    sumW=np.sum(np.log2(ProbW).flatten())
    sumH=np.sum(np.log2(ProbH).flatten())
    sumE=np.sum(np.log2(ProbE+tinyVal).flatten())
    return -(sumW+sumH+sumE)
def runTraining(r,Vorig,nRuns,wLR=1e-5,hLR=1e-5,deltaW=0.004,deltaH=0.004,deltaE=0.004):
    V,W0,H0,E0,alphaW,betaW,aH,bH,muE,sigmaE,m,n=initialise(r,Vorig) #### normalise V, initialise W and H to srt gamma distributions
    W,H,E=W0.copy(),H0.copy(),E0.copy() ### use W,H,E and leave original alone
    ProbW=returnGammaProbs(alphaW,betaW,W,deltaW)
    ProbH=returnGammaProbs(aH,bH,H,deltaH)
    ProbE=returnNormProbs(E,muE,sigmaE,deltaE)
    fObStore=[returnObjFunc(ProbW,ProbH,ProbE)]
    minF=fObStore[0]
    for i in range(nRuns): ### begin training run.
        Wtilde=returnWorHtilde(W, alphaW, betaW)
        Htilde=returnWorHtilde(H, aH, bH)
        Etilde=returnEtilde(E,muE,sigmaE)
        changeInW=Wtilde+np.matmul(Etilde,H.T)
        Wtemp=W-wLR*changeInW
        Wtemp=np.maximum(Wtemp,0.002)
        changeInH=Htilde+np.matmul(Wtemp.T,Etilde)
        Htemp=H-hLR*changeInH
        Htemp=np.maximum(Htemp,0.002)
        ProbW=returnGammaProbs(alphaW,betaW,Wtemp,deltaW)
        ProbH=returnGammaProbs(aH,bH,Htemp,deltaH)
        ProbE=returnNormProbs(E,muE,sigmaE,deltaE)
        fObStore.append(returnObjFunc(ProbW,ProbH,ProbE))
        W,H=Wtemp,Htemp
        E=E=V-np.matmul(W,H)
        alphaW,betaW,aH,bH,muE,sigmaE=returnParams(W,H,E)
        print(i,fObStore[i+1])
        if fObStore[i+1]<minF:
            Wbest,Hbest=W,H
    return Wbest,Hbest,fObStore
        














