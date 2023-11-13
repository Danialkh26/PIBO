# -*- coding: utf-8 -*-
"""
Created on Wed Mar  8 20:40:46 2023

@author: Danial Khatamsaz
"""
import numpy as np

def Kernel(dataset1,dataset2,sf,l):
    
    dataset1[:,2]=dataset1[:,2]**0.5
    dataset1[:,1]=1/dataset1[:,1]
    
    dataset2[:,2]=dataset2[:,2]**0.5
    dataset2[:,1]=1/dataset2[:,1]
    
    S1=dataset1.shape[0]
    S2=dataset2.shape[0]
    K=np.zeros([S1,S2])
    
    for i in range(S2):
        temp = ((dataset1-dataset2[i])**2)/l**2
        temp_sum = -0.5*np.sum(temp,axis=1)
        K[:,i]=sf*np.exp(temp_sum)
        
    return K
