import MDLasObjectiveFunctionTrain 
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

def pathData():
    return 'Data/'
dataNames=['Vfaces.csv','Vfinance.csv','Vgenes.txt']
r=10
nRuns=2000
V=pd.read_csv(pathData()+dataNames[1],index_col=0,header=None).values

W,H,fObStore=MDLasObjectiveFunctionTrain.runTraining(r,V,nRuns,wLR=1e-6,hLR=1e-6)
Vp=np.matmul(W,H)
figsize=4
plt.close('all')
plt.figure(1,figsize=(5*figsize,figsize))
randPerm=np.random.permutation(V.shape[1])
for i in range(5):
    
    plt.subplot(1,5,i+1)
    plt.plot(V[:,randPerm[i]],'k--')
    plt.plot(Vp[:,randPerm[i]],'r--')
plt.figure(2)
plt.plot(np.array(fObStore),'k--')
