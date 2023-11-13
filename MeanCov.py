# -*- coding: utf-8 -*-
"""
Created on Wed Mar  8 17:00:19 2023

@author: Danial Khatamsaz
"""
import numpy as np
import scipy
import scipy.linalg
from copy import deepcopy
    
def Kernel(dataset1,dataset2,sf,l):
    
    dataset11=deepcopy(dataset1)
    dataset22=deepcopy(dataset2)
    
    
    dataset11[:,2]=dataset11[:,2]**0.5
    dataset11[:,1]=1/dataset11[:,1]
    
    dataset22[:,2]=dataset22[:,2]**0.5
    dataset22[:,1]=1/dataset22[:,1]
    
    S1=dataset11.shape[0]
    S2=dataset22.shape[0]
    K=np.zeros([S1,S2])
    
    for i in range(S2):
        temp = ((dataset11-dataset22[i])**2)/l**2
        temp_sum = -0.5*np.sum(temp,axis=1)
        K[:,i]=sf*np.exp(temp_sum)
        
    return K

def MeanCov(xtrain,ytrain,l,sf,sn,test):
    
    Kxz = Kernel(xtrain,test,sf,l)
    Kzz = Kernel(test,test,sf,l)
    # Kzx=np.transpose(Kxz)
    
    GPKxx=Kernel(xtrain,xtrain,sf,l)
    H = GPKxx+np.eye(xtrain.shape[0])*sn**2
    R = scipy.linalg.cholesky(H, lower=False)
    
    ad = np.linalg.solve(np.transpose(R), ytrain)
    bd = np.linalg.solve(np.transpose(R), Kxz)
    
    m = np.transpose(bd)@ad
    co = Kzz-np.transpose(bd)@bd
    
    return m,co