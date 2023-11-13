"""
Created on Sat Nov 12 13:31:21 2022

@author: Danial Khatamsaaz
"""

### sf,sn,l to be 1D np arrays
### xtrain to be 2D np array
### ytrain to be 1Dnp array and create more GPs if more than 1 ouput

import numpy as np
import pandas as pd
from pyDOE import *
from acquisitionFunc import knowledge_gradient , expected_improvement
import matplotlib.pyplot as plt
from sklearn.preprocessing import normalize
import timeit
import random
from precipitator import precipitator
from MeanCov import MeanCov
from sklearn.ensemble import RandomForestClassifier

# composition = 50.8, temp = 723K (450C), and time = 7200s
 # transformation temperature of 315.256 K and a mean interparticle distance of 2.733e-8

iteration=30
N_dim=3
lb1 = 0.500 # lower bound of dimension 1 Composition
ub1 = 0.520 # upper bound of dimension 1
lb2 = 650 # lower bound of dimension 2 Temp
ub2 = 1050 # upper bound of dimension 2
lb3 = 1 # lower bound of dimension 3 Time
ub3 = 18000 # upper bound of dimension 3
N_test=1000


# x_init= pd.DataFrame(pd.read_csv('x_init.csv', header=None)).to_numpy()
# y_init= pd.DataFrame(pd.read_csv('y_init.csv', header=None)).to_numpy()

label_distance=pd.DataFrame(pd.read_csv('label_distance.csv', header=None)).to_numpy()
label_vf=pd.DataFrame(pd.read_csv('label_vf.csv', header=None)).to_numpy()
label_in=pd.DataFrame(pd.read_csv('input.csv', header=None)).to_numpy()

clf1 = RandomForestClassifier()
clf1.fit(label_in,label_distance.ravel())

clf2 = RandomForestClassifier()
clf2.fit(label_in,label_vf.ravel())

#x_init = np.array([[0.51, 800 ,5000],[0.512, 700 ,12000]])
#y_init = np.array([[252.01422903],[340.48955394]])
#dist_init = np.array([[9.4044465e-06],[1.04258271e-08]])

x_init = np.array([[0.5082111364583605,726.2354260088387,3759.601375083552]])
y_init = np.array([[290.439776559818]])
dist_init = np.array([[2.977867961513454e-08]])
vf_init = np.array([[0.02805699166816168]])


improvements=[]
y_max_total=[]
x_max_total=[]
dist_total=[]
sig_total=[]
database_x=x_init
database_y=y_init
database_dist = dist_init
database_vf = vf_init

for rep in range(50):
    
    sf=175**2 ### variance of the GP where no observation exists
    sn=0.0001
    l=np.array([0.001,0.025,42.4])
    y_max_found=np.array([np.max(y_init)])
    ind=np.argmax(y_init)
    x_max_found=x_init[ind,:].reshape(1,N_dim)
    sigs=[]
    train_x=x_init
    train_y=y_init.reshape(train_x.shape[0],1)
    dist=dist_init
    vf=vf_init
    
    itr=0

    while itr<iteration:

        itr=itr+1
        
        xx=lhs(N_dim,100000)*[ub1-lb1,ub2-lb2,ub3-lb3]+[lb1,lb2,lb3]
        temp1=clf1.predict(xx)
        cfs=xx[temp1==1,:]
        temp2=clf2.predict(cfs)
        cfs2=cfs[temp2==1,:]
        
        if cfs2.shape[0]>=N_test:
            x_test=cfs2[0:N_test,:]
        else:
            x_test=cfs2
        
        
        # x_test=lhs(N_dim,N_test)*[ub1-lb1,ub2-lb2,ub3-lb3]+[lb1,lb2,lb3]

        y,sig=MeanCov(train_x,train_y,l,sf,sn,x_test)
                
        
        sigs.append(np.sum(np.diag(sig)**0.5))
        
        nu_star,x_star,nu=expected_improvement(y_max_found[-1], y,(np.diag(sig))**0.5)
        improvements.append(nu_star)
        
        x_query=np.array(x_test[x_star]).reshape(1,N_dim)
        y_query,dist_query,vf_query=precipitator(x_query)
        y_query = y_query.reshape(1,1)
        dist_query = dist_query.reshape(1,1)
        vf_query = vf_query.reshape(1,1)
        dist=np.concatenate((dist,dist_query),axis=0)
        vf=np.concatenate((vf,vf_query),axis=0)
        train_x=np.concatenate((train_x,x_query),axis=0)
        train_y=np.concatenate((train_y,y_query),axis=0)
        
        
        y_max_found=np.append(y_max_found,np.max(train_y))
        ind=np.argmax(train_y)
        x_max_found=np.append(x_max_found,train_x[ind].reshape(1,N_dim),axis=0)
        pd.DataFrame(np.array(y_max_found)).to_csv("y_max_found.csv", header=None, index=None)
        pd.DataFrame(np.array(x_max_found)).to_csv("x_max_found.csv", header=None, index=None)

    database_x=np.concatenate((database_x,train_x),axis=0)
    database_y=np.concatenate((database_y,train_y),axis=0)
    database_dist=np.concatenate((database_dist,dist),axis=0)
    database_vf=np.concatenate((database_vf,vf),axis=0)
    
    y_max_total.append(y_max_found)
    sig_total.append(sigs)
    # x_max_total.append(x_max_found)

    pd.DataFrame(np.array(y_max_total)).to_csv("y_max_total.csv", header=None, index=None)
    pd.DataFrame(database_x).to_csv("database_x.csv", header=None, index=None)
    pd.DataFrame(database_y).to_csv("database_y.csv", header=None, index=None)
    pd.DataFrame(database_dist).to_csv("database_dist.csv", header=None, index=None)
    pd.DataFrame(database_vf).to_csv("database_vf.csv", header=None, index=None)
    pd.DataFrame(np.array(sig_total)).to_csv("sig_total.csv", header=None, index=None)

