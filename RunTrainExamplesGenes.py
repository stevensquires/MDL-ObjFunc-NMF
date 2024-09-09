import MDLasObjectiveFunctionTrain 
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

def pathData():
    return 'Data/'
dataNames=['Vfaces.csv','Vfinance.csv','Vgenes.txt']
r=4
nRuns=500
V=pd.read_csv(pathData()+dataNames[2],index_col=0,header=None,sep='\t').values
V,W0,H0,E0,alphaW,betaW,aH,bH,muE,sigmaE,m,n=MDLasObjectiveFunctionTrain.initialise(r,V)
W,H,fObStore=MDLasObjectiveFunctionTrain.runTraining(r,V,nRuns,wLR=5e-6,hLR=5e-6)
Vp=np.matmul(W,H)
figsize=4
plt.close('all')
plt.figure(1)
randPerm=np.random.permutation(V.shape[1])
for i in range(5):    
    plt.plot(V[:,randPerm[i]],Vp[:,randPerm[i]],'k.')
plt.axis([0,1,0,1])
plt.grid()
plt.xlabel('Real=');plt.ylabel('Reconstructed')
plt.figure(2)
plt.plot(np.array(fObStore),'k--')
