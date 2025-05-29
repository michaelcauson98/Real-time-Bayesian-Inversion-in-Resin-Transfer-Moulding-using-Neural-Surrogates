# -*- coding: utf-8 -*-
"""
Created on Mon Nov 13 12:00:08 2023

@author: pmymc12
"""
# New comment

###############################################################################
############################     Imports     ##################################
###############################################################################

# Standard imports
import matplotlib.pyplot as plt
from matplotlib import cm
import matplotlib.image as mpimg
import matplotlib.patches as patches
import numpy as np
import scipy
import torch
import json
import tqdm
from copy import copy
import timeit

# Local imports
from Experiment import Experiment
from Data import Data
from Utils import read_txt, push_ensemble, plot_press
from NeuralNetwork import NeuralNetwork
from EKI import EKI
from PlotEKI import PlotEKI, nlcmap

###############################################################################
########################     Initialisation     ###############################
###############################################################################

# Experiment object holds all of the experimental details
Exp = Experiment()

# Read in data 
# X has 71 columns (49 permeability values, 17 porosity values, 4 p_I, 1 \mu)
# Y has 300 coloumns (20 sensor locs with 15 observation times)
X, Y = read_txt(r"Data\Inputs.txt"), read_txt(r"Data\Outputs.txt")
# X, Y = read_txt(r"Data\Inputs_surrogate_1.txt"), read_txt(r"Data\Outputs_surrogate_1.txt")
filling_times = read_txt(r"Data\filling_times.txt")
# Data object holds data and useful data-related functions
Dat = Data(Exp,X,Y,filling_times)


###############################################################################
####################     Neural network training    ###########################
###############################################################################

# Initialise neural network and train 
NN = NeuralNetwork(Data = Dat,
                   architecture = [len(Dat.TrainX[0]),200,200,200,len(Dat.TrainY[0])],
                   activation = "Sigmoid",
                   epochs = 5000, 
                   learning_rate = 0.001,
                   batch_size = 128,
                   plotting = 1)
NN.train_nn()
NN.assess_surrogate(n = 10)

###############################################################################
####################     Visual surrogate error    ############################
###############################################################################

# Uncomment to upload saved NN model
# NN = NeuralNetwork(Data = Dat,
#                    architecture = [len(Dat.TrainX[0]),1000,len(Dat.TrainY[0])],
#                    activation = "Sigmoid",
#                    epochs = 1000, 
#                    learning_rate = 0.001,
#                    batch_size = 128,
#                    plotting = 1)
# NN.upload_nn(r"path_to_NN.pth") # Upload pre-trained model

plt.figure(figsize=(14,4))
plt.subplot(1,2,1)
plt.plot(NN.surr_error)
plt.title(r"$\bar{\epsilon}$"), plt.ylabel("Pressure (Pa)"), plt.xlabel("Index")
plt.subplot(1,2,2)
plt.plot(np.sqrt(np.diagonal(NN.surr_cov)))
plt.title(r"$\sqrt{diag(\Sigma)}$"), plt.ylabel("Pressure (Pa)"), plt.xlabel("Index")
plt.tight_layout()
plt.show()

###############################################################################
###########################     Best model     ################################
###############################################################################

# Finds the optimal surrogate architecture

# Single layer
train_mse_vec_single = []
dev_mse_vec_single = []
rel_error_single = []

# Double layer
train_mse_vec_double = []
dev_mse_vec_double = []
rel_error_double = []

# Triple layer
train_mse_vec_triple = []
dev_mse_vec_triple = []
rel_error_triple = []

for i in [200,400,600,800,1000,1250,1500]:
    print(i)
    
    print("Single layer:")
    NN = NeuralNetwork(Data = Dat,
                       architecture = [len(Dat.TrainX[0]),i,len(Dat.TrainY[0])],
                       activation = "Sigmoid",
                       epochs = 50_000, 
                       learning_rate = 0.001,
                       batch_size = 128,
                       plotting = 0)
    NN.train_nn()
    torch.save(NN.model, r"neuralnet" + str(i) + r".pth")
    dev_mse_vec_single.append(NN.dev_mse)
    train_mse_vec_single.append(NN.train_mse)
    rel_error_single.append(NN.assess_surrogate(0))
    
    print("Double layer:")
    NN = NeuralNetwork(Data = Dat,
                       architecture = [len(Dat.TrainX[0]),i,i,len(Dat.TrainY[0])],
                       activation = "Sigmoid",
                       epochs = 50_000, 
                       learning_rate = 0.001,
                       batch_size = 128,
                       plotting = 0)
    NN.train_nn()
    torch.save(NN.model, r"neuralnet" + str(i) + "x" + str(i) + r".pth")
    dev_mse_vec_double.append(NN.dev_mse)
    train_mse_vec_double.append(NN.train_mse)
    rel_error_double.append(NN.assess_surrogate(0))
    
    print("Triple layer:")
    NN = NeuralNetwork(Data = Dat,
                       architecture = [len(Dat.TrainX[0]),i,i,i,len(Dat.TrainY[0])],
                       activation = "Sigmoid",
                       epochs = 50_000, 
                       learning_rate = 0.001,
                       batch_size = 128,
                       plotting = 0)
    NN.train_nn()
    torch.save(NN.model, r"neuralnet" + str(i) + "x" + str(i) + "x" + str(i) + r".pth")
    dev_mse_vec_triple.append(NN.dev_mse)
    train_mse_vec_triple.append(NN.train_mse)
    rel_error_triple.append(NN.assess_surrogate(0))
    
    # Plot relative error scores
    plt.plot(rel_error_single,color="black")
    plt.plot(rel_error_double,color="red")
    plt.plot(rel_error_triple,color="blue")
    plt.show()
    
    # Plot train/validation MSE of training process
    plt.plot(np.log10(dev_mse_vec_single[-1]),color="black")
    plt.plot(np.log10(train_mse_vec_single[-1]),color="black",linestyle="--")
    plt.plot(np.log10(dev_mse_vec_double[-1]),color="red")
    plt.plot(np.log10(train_mse_vec_double[-1]),color="red",linestyle="--")
    plt.plot(np.log10(dev_mse_vec_triple[-1]),color="blue")
    plt.plot(np.log10(train_mse_vec_triple[-1]),color="blue",linestyle="--")
    plt.show()

# Save as matrices for later reference
scipy.io.savemat(r"surrogate_rel_error.mat",
                  dict(rel_error_single = rel_error_single,
                       rel_error_double = rel_error_double,
                       rel_error_triple = rel_error_triple))
scipy.io.savemat(r"surrogate_dev_mse.mat",
                  dict(dev_mse_vec_single = dev_mse_vec_single,
                       dev_mse_vec_double = dev_mse_vec_double,
                       dev_mse_vec_triple = dev_mse_vec_triple))
scipy.io.savemat(r"surrogate_train_mse.mat",
                  dict(train_mse_vec_single = train_mse_vec_single,
                       train_mse_vec_double = train_mse_vec_double,
                       train_mse_vec_triple = train_mse_vec_triple))


# Plot for paper
plt.figure(figsize=(10.5,3.5),dpi=200)
plt.subplot(1,2,1)
plt.scatter([200,400,600,800,1000,1250,1500],[rel_error_single[i] for i in range(7)],color="black",label="1 layer")
plt.ylabel(r"$E_{val}$"), plt.xlabel("Hidden layer size"), plt.title("Average relative error on validation set")

plt.subplot(1,2,2)
plt.plot(np.log10(dev_mse_vec_single[4]),color="black",label="Validation MSE")
plt.plot(np.log10(train_mse_vec_single[4]),color="red",label="Train MSE")
plt.axvline(x=len(dev_mse_vec_single[4])-50,color="black",linestyle="--")
plt.xlabel("Epochs")
plt.ylabel(r"$\log_{10}(MSE_{val})$")
plt.title("MSE of validation set (1000 nodes)")
plt.legend(loc="upper right")
plt.tight_layout()
#plt.savefig("Figures/NN_errors.png")
plt.show()

###############################################################################
########################     EKI on test data    ##############################
###############################################################################

# Test surrogate evaluation time
%timeit NN.F(Dat.DevelopX[0])

# Set various inversion times
all_times = list(range(1,19))

# Generate data for test row i
i = 5
x_i = Dat.UnitTransformX(Dat.TestX, "BWD")[i]
y_i = Dat.ParameteriseY(Dat.TestY,"BWD")[i]

# Create data
data_obj_virt = Dat.generate_data(y_i,
                                  sensor_inds = list(range(20)),
                                  sigma1 = 0.000,sigma2 = 0.005,
                                  surr_error = NN.surr_error)

# Create EKI object
eki = EKI(Experiment = Exp, Data = Dat, NeuralNetwork = NN,
          Data_obj = data_obj_virt, t = all_times,
          n_ensemble = 10_000, iter_max = 100,
          p_I = x_i[-5:-1], mu = x_i[-1],
          u_true = x_i[:len(x_i)-5])

# Run EKI
posterior_ensemble = eki.run_EKI(plotting = False,ensemble_dep=0)
eki.diagnostic_check(posterior_ensemble[0][-1])



###############################################################################
######################     EKI on all test data    ############################
###############################################################################

# Generate testX, testY, inlet pressures and viscosities
X_test_og = Dat.UnitTransformX(Dat.TestX, "BWD")[:,:len(Dat.TestX[0])-2]
Y_test_og = Dat.ParameteriseY(Dat.TestY,"BWD")
p_I_og = Dat.UnitTransformX(Dat.TestX, "BWD")[:,-2]
mu_og = Dat.UnitTransformX(Dat.TestX, "BWD")[:,-1]

# Initial data matrices
mean_relerror_K = np.zeros((6,4))
std_relerror_K = np.zeros((6,4))
mean_relerror_phi = np.zeros((6,4))
std_relerror_phi = np.zeros((6,4))
mean_time = np.zeros((6,4))
std_time = np.zeros((6,4))

for idi,i in enumerate(["2x2","3x3","4x4","5x5","Half","All"]): # for each sensor config.
    print(i)
    for idj,j in enumerate([0.1,0.05,0.01,0.005]): # for each sensor precision
        print(j)
        relerror_K = []
        relerror_phi = []
        time_vec = []
        for k in tqdm.tqdm(range(len(Dat.TestX))): # for each element of test set
            data_obj = Dat.generate_data(Y_test_og[k],
                                     sensor_inds = Exp.sensor_dict[i],
                                     sigma1 = 0,
                                     sigma2 = j)
            eki = EKI(Experiment = Exp,
                      Data = Dat,
                      NeuralNetwork = NN,
                      Data_obj = data_obj,
                      t = seven_times,
                      n_ensemble = 10_000,
                      iter_max = 100,
                      p_I = p_I_og[k], mu = mu_og[k])
            
            # Run EKI
            posterior_ensemble = eki.run_EKI(plotting = False)
            
            # Extract info
            K_mean = np.mean(posterior_ensemble[0][-1][:,:85],axis=0)
            phi_mean = np.mean(posterior_ensemble[0][-1][:,85:],axis=0)
            time_taken = np.sum(posterior_ensemble[1])
            
            # Compute relative error and time taken
            relerror_K.append( np.linalg.norm(X_test_og[k,:85]-K_mean)/np.linalg.norm(X_test_og[k,:85]) )
            relerror_phi.append( np.linalg.norm(X_test_og[k,85:]-phi_mean)/np.linalg.norm(X_test_og[k,85:]) )
            time_vec.append(time_taken)
        
        # Once all tests are complete, find average and add to matrices
        mean_relerror_K[idi,idj] = np.mean(relerror_K)
        std_relerror_K[idi,idj] = np.std(relerror_K)
        mean_relerror_phi[idi,idj] = np.mean(relerror_phi)
        std_relerror_phi[idi,idj] = np.std(relerror_phi)
        mean_time[idi,idj] = np.mean(time_vec)
        std_time[idi,idj] = np.std(time_vec)

# Save matrices
# scipy.io.savemat(r"mean_relerror_K.mat", dict(mean_relerror_K = mean_relerror_K))
# scipy.io.savemat(r"mean_relerror_phi.mat", dict(mean_relerror_phi = mean_relerror_phi))
# scipy.io.savemat(r"mean_time.mat", dict(mean_time = mean_time))
# scipy.io.savemat(r"std_relerror_K.mat", dict(std_relerror_K = std_relerror_K))
# scipy.io.savemat(r"std_relerror_phi.mat", dict(std_relerror_phi = std_relerror_phi))
# scipy.io.savemat(r"std_time.mat", dict(std_time = std_time))

# Compute prior means and error
K_prior = np.concatenate( (np.ones(81)*(Exp.min_perm_central+Exp.max_perm_central)/2,
                           np.ones(4)*(Exp.min_perm_RT+Exp.max_perm_RT)/2) )
phi_prior = np.concatenate( (np.ones(81)*(Exp.min_poro_central+Exp.max_poro_central)/2,
                           np.ones(4)*(Exp.min_poro_RT+Exp.max_poro_RT)/2) )
K_prior_error = np.mean(np.linalg.norm(X_test_og[:,:85]- K_prior,axis=1) / np.linalg.norm(X_test_og[:,:85],axis=1))
phi_prior_error = np.mean(np.linalg.norm(X_test_og[:,85:]- phi_prior,axis=1) / np.linalg.norm(X_test_og[:,85:],axis=1))

# Plot
plt.figure(figsize=(12,4),dpi=200)
plt.subplot(1,3,1)
[plt.scatter([0.1,0.05,0.01,0.005],mean_relerror_K[i],s=48,label=["2x2","3x3","4x4","5x5","Half","All"][i]) for i in range(len(mean_relerror_K))]
plt.axhline(K_prior_error,color="k",linestyle="--")
plt.legend(loc=1)
plt.title(r"$K$")
plt.ylabel(r"Average  $E_{K}$")
plt.xlabel(r"$\sigma_0$")
plt.xticks([0,0.05,0.1])
plt.ylim([0,0.6])
plt.subplot(1,3,2)
[plt.scatter([0.1,0.05,0.01,0.005],mean_relerror_phi[i],s=48,label=["2x2","3x3","4x4","5x5","Half","All"][i]) for i in range(len(mean_relerror_phi))]
plt.axhline(phi_prior_error,color="k",linestyle="--")
plt.legend(loc=4)
plt.title(r"$\phi$")
plt.ylabel(r"Average  $E_{\phi}$")
plt.xlabel(r"$\sigma_0$")
plt.ylim([0.0,0.23])
plt.xticks([0,0.05,0.1])
plt.subplot(1,3,3)
[plt.scatter([0.1,0.05,0.01,0.005],mean_time[i],s=48,label=["2x2","3x3","4x4","5x5","Half","All"][i]) for i in range(len(mean_time))]
plt.legend()
plt.title("Average inversion time")
plt.xlabel(r"$\sigma_0$")
plt.ylabel("Time (secs)")
plt.xticks([0,0.05,0.1])
plt.tight_layout()
plt.show()

