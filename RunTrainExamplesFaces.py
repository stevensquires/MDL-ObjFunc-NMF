import MDLasObjectiveFunctionTrain 
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

def pathData():
    return 'Data/'
dataNames=['Vfaces.csv','Vfinance.csv','Vgenes.txt']
r=50
nRuns=300
V=pd.read_csv(pathData()+dataNames[0],index_col=0,header=None).values

W,H,fObStore=MDLasObjectiveFunctionTrain.runTraining(r,V,nRuns,wLR=1e-5,hLR=1e-5)
Vp=np.matmul(W,H)
figsize=4
plt.close('all')
plt.figure(1)
randPerm=np.random.permutation(V.shape[1])
for i in range(5):
    plt.figure(i+1,figsize=(2*figsize,figsize))
    plt.subplot(1,2,1)
    im0=V[:,randPerm[i]].reshape(19,19)
    plt.imshow(im0.T,cmap='gray')
    plt.subplot(1,2,2)
    im1=Vp[:,randPerm[i]].reshape(19,19)
    plt.imshow(im1.T,cmap='gray')
plt.figure(6)
plt.plot(np.array(fObStore),'k--')
