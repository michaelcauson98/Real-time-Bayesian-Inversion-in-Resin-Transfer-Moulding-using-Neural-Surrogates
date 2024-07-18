# -*- coding: utf-8 -*-
"""
Created on Mon Nov 13 12:06:29 2023

@author: pmymc12
"""

import numpy as np
import matplotlib.pyplot as plt
import scipy

# Import permeability/pressure files and save as a numpy array, size (samples)x(regions)
def read_txt(txt_file):
    f = open(txt_file, 'r')
    content = f.read()
    lines = content.splitlines()
    dat = [x.split(',') for x in lines]
    dat = np.array(dat,dtype=float)
    f.close()
    return dat[0] if len(dat) == 1 else dat

# Converts permeability to porosity according to power law used in paper.
def permeability_to_porosity(K):
    return 1 - (10**(-10)/K)**(1/1.4)

# Converts porosity to permeability according to power law used in paper.
def porosity_to_permeability(phi):
    return 10**(-10)*((1-phi)**(-1.4))

# PLots (interpolated) pressure using exported simulations from Matlab.
def plot_press(Exp,pressure_data,mesh_nodes,t,rm_axis=False,plotting=False):
    
    x=np.linspace(0,Exp.Lx,200)
    y=np.linspace(0,Exp.Ly,200)
    [X,Y]=np.meshgrid(x,y)
    px = mesh_nodes[:,0]
    py = mesh_nodes[:,1]

    Ti = scipy.interpolate.griddata((px, py), pressure_data[:,t-1], (X, Y), method='linear')
    Ti_level_set = (Ti > 0)
    
    if plotting:
        plt.imshow(Ti_level_set,extent=[0,0.3,0,0.3],origin="lower",vmin=0,vmax=1,cmap="plasma")
        plt.xlim([0,0.3])
        plt.ylim([0,0.3])
        if rm_axis:
            plt.xticks([])
            plt.yticks([])
        
    return Ti_level_set
        

# Pushes ensemble through the (surrogate) forward map and plots
def push_ensemble(U,Data,NN,EKI,pressure_data,sensor_inds):
    
    U = np.hstack( (U,
                    np.ones((EKI.n_ensemble,1))*EKI.p_I,
                    np.ones((EKI.n_ensemble,1))*EKI.mu) )
    U = Data.UnitTransformX(data = U, direction="FWD")
    
    y_pred = NN.F(U)
    
    y_pred_m = np.mean(y_pred,axis=0)
    y_pred_s = np.std(y_pred,axis=0)
    k=0
    
    for sensor in sensor_inds:
        inds1 = []
        inds2 = []
        for j in range(len(EKI.t)):
            inds1.append(k + int(len(pressure_data)/len(EKI.Experiment.observation_times)*(EKI.t[j]-1)))
            inds2.append(sensor + int(len(y_pred_m)/len(EKI.Experiment.observation_times)*(EKI.t[j]-1)))
        k += 1
        plt.plot(np.array(EKI.t),y_pred_m[inds2],color="black")
        plt.fill_between(np.array(EKI.t),y_pred_m[inds2]-2*y_pred_s[inds2],y_pred_m[inds2]+2*y_pred_s[inds2],color="green",alpha=0.5)
        plt.plot(np.array(EKI.t),pressure_data[inds1],color="red",linestyle="--")
    plt.ylim([-5_000,110_000])
    plt.xticks(np.array(EKI.t))
    plt.show()
    
    
    