# -*- coding: utf-8 -*-
"""
Created on Mon Nov 13 12:00:08 2023

@author: pmymc12
"""

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
# X has 172 columns (85 permeability values, 85 porosity values, 1 p_I, 1 \mu)
# Y has 1624 coloumns (116 sensor locs with 14 observation times)
X, Y = read_txt(r"Data\X_9x9.txt"), read_txt(r"Data\Y_9x9.txt")
filling_times = read_txt(r"Data\filling_times.txt")

# Data object holds data and useful data-related functions
Dat = Data(Exp,X,Y,filling_times)


###############################################################################
####################     Neural network training    ###########################
###############################################################################

# Initialise neural network and train
NN = NeuralNetwork(Data = Dat,
                   architecture = [len(Dat.TrainX[0]),1000,len(Dat.TrainY[0])],
                   activation = "Sigmoid",
                   epochs = 1000, 
                   learning_rate = 0.001,
                   batch_size = 128,
                   plotting = 1)
NN.train_nn()
NN.assess_surrogate(n = 20)

###############################################################################
####################     Visual surrogate error    ############################
###############################################################################

NN = NeuralNetwork(Data = Dat,
                   architecture = [len(Dat.TrainX[0]),1000,len(Dat.TrainY[0])],
                   activation = "Sigmoid",
                   epochs = 1000, 
                   learning_rate = 0.001,
                   batch_size = 128,
                   plotting = 1)
NN.upload_nn(r"Data\neuralnet1000.pth") # Upload pre-trained model

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
########################     Best model plot     ##############################
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
    torch.save(NN.model, r"NNs\newneuralnet" + str(i) + r".pth")
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
    torch.save(NN.model, r"NNs\newneuralnet" + str(i) + "x" + str(i) + r".pth")
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
    torch.save(NN.model, r"NNs\newneuralnet" + str(i) + "x" + str(i) + "x" + str(i) + r".pth")
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
scipy.io.savemat(r"NNs/surrogate_rel_error.mat",
                  dict(rel_error_single = rel_error_single,
                       rel_error_double = rel_error_double,
                       rel_error_triple = rel_error_triple))
scipy.io.savemat(r"NNs/surrogate_dev_mse.mat",
                  dict(dev_mse_vec_single = dev_mse_vec_single,
                       dev_mse_vec_double = dev_mse_vec_double,
                       dev_mse_vec_triple = dev_mse_vec_triple))
scipy.io.savemat(r"NNs/surrogate_train_mse.mat",
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

NN = NeuralNetwork(Data = Dat,
                   architecture = [len(Dat.TrainX[0]),1000,len(Dat.TrainY[0])],
                   activation = "Sigmoid",
                   epochs = 50_000, 
                   learning_rate = 0.001,
                   batch_size = 128,
                   plotting = 1)
NN.upload_nn(r"NNs\newneuralnet1000.pth")

# Test surrogate evaluation time
%timeit NN.F(Dat.DevelopX[0])

# Set various inversion times
all_times = list(range(1,15))
five_times = [1,4,7,10,14]
seven_times = [1,3,5,7,9,11,14]

# Generate data for test row i
i = 14
x_14 = Dat.UnitTransformX(Dat.TestX, "BWD")[i]
y_14 = Dat.ParameteriseY(Dat.TestY,"BWD")[i]

# Create data
data_obj_virt = Dat.generate_data(y_14,
                                  sensor_inds = Exp.sensor_dict['All'],
                                  sigma1 = 0.000,sigma2 = 0.005,
                                  surr_error = NN.surr_error)

# Create EKI object
eki = EKI(Experiment = Exp, Data = Dat, NeuralNetwork = NN,
          Data_obj = data_obj_virt, t = seven_times,
          n_ensemble = 10_000, iter_max = 100,
          p_I = x_14[-2], mu = x_14[-1],
          u_true = x_14[:len(x_14)-2])

# Run EKI
posterior_ensemble = eki.run_EKI(plotting = False,ensemble_dep=0)

# Import mesh and pressures from Matlab (expensive) simulation for this
# particular test example to generate front positions
mesh = read_txt(r"Data\Virtual\mesh14.txt")
all_pressures = read_txt(r"Data\Virtual\pressures14.txt")
images = []
front_positions = []
for j in range(1,15):
    image = plot_press(Exp,all_pressures,mesh,j,plotting=False)
    image_sums = np.sum(image,axis=1)
    front_position = np.zeros((len(image),2))
    for k in range(200):
        front_position[k] = [image_sums[k],k]
    front_position[:,0] = np.flipud(front_position[:,0])
    images.append(image)
    front_positions.append(front_position)
front_positions = np.array(front_positions)/(len(image)/Exp.Lx)

# Plot inversion
plotter = PlotEKI(posterior_ensemble, eki)
plotter.plot_sequential(permeability_only = 0,log_perm=1,
                        sensor_locs=Exp.sensor_dict['All'],
                        plot_prob = 0,defect_type="low",
                        defect_tresholds=[3e-10,7e-10,0.5,0.7],
                        images_available = 0, images = None,
                        truth_available = 1, truth = eki.u_true,
                        front_available = 1, front = [front_positions[i-1] for i in seven_times],
                        plot_circle=0,circle_centre=[0,0],radius=0.1)

# Plot posterior ensemble vs actual values
eki.diagnostic_check(posterior_ensemble[0][-1])


###############################################################################
########################     EKI on RTM words    ##############################
###############################################################################

relative_path = "Data\\Virtual\\"
images = []
front_positions = []

experiment = 3
p1 = read_txt(relative_path + r"pressures" + str(experiment) + ".txt")
K1 = np.flipud(read_txt(relative_path + r"K"  + str(experiment) + ".txt"))
phi1 = np.flipud(read_txt(relative_path + r"phi"  + str(experiment) + ".txt"))
mesh1 = read_txt(relative_path + r"mesh.txt")
all_p1 = read_txt(relative_path + r"all_pressures"  + str(experiment) + ".txt")
u_true1 = np.concatenate( (np.reshape(K1,4096),np.reshape(phi1,4096)) )
params1 = read_txt(relative_path + r"params"  + str(experiment) + ".txt")

k_true = np.reshape(u_true1[:4096],(64,64))
k_true[1,:] = k_true[0,:]
k_true[-2,:] = k_true[-1,:]
phi_true = np.reshape(u_true1[4096:],(64,64))
phi_true[1,:] = phi_true[0,:]
phi_true[-2,:] = phi_true[-1,:]
u_true1 = np.concatenate( (np.reshape(k_true,4096),np.reshape(phi_true,4096)) )

for i in range(1,15):
    image = plot_press(Exp,all_p1,mesh1,0,i,rm_axis=False,plotting=False)
    image_sums = np.sum(image,axis=1)
    front_position = np.zeros((200,2))
    for j in range(200):
        front_position[j] = [image_sums[j],j]
    images.append(image)
    front_positions.append(front_position)
front_positions = np.array(front_positions)/(200/0.3)

data_obj = Dat.generate_data(p1,
                             sensor_inds = Exp.sensor_dict['All'],
                             sigma1 = 0.0,
                             sigma2 = 0.005)
eki = EKI(Experiment = Exp,
          Data = Dat,
          NeuralNetwork = NN,
          Data_obj = data_obj,
          t = seven_times,
          n_ensemble = 10_000,
          iter_max = 100,
          p_I = params1[0],
          mu = params1[1],
          virtual=1,
          u_true = u_true1)
posterior_ensemble = eki.run_EKI(plotting = True,ensemble_dep=1)

plotter = PlotEKI(posterior_ensemble, eki)
plotter.plot_sequential(permeability_only = 0,log_perm=True,sensor_locs=Exp.sensor_dict['All'],
                    plot_prob = 1,defect_type="low",defect_tresholds=[3e-10,6e-10,0.544,0.722],
                    images_available = 1, images = [images[i-1] for i in seven_times],
                    truth_available = 1, truth = u_true1,
                    front_available = 1, front = [front_positions[i-1] for i in seven_times],
                    plot_circle=0,circle_centre=[0.1,0.1],radius=0.04)

eki.diagnostic_check(posterior_ensemble[0][-1])


###############################################################################
######################     EKI on all test data    ############################
###############################################################################

# Instantiate NN
NN = NeuralNetwork(Data = Dat,
                   architecture = [len(Dat.TrainX[0]),1000,len(Dat.TrainY[0])],
                   activation = "Sigmoid",
                   epochs = 10_000, 
                   learning_rate = 0.001,
                   batch_size = 128,
                   plotting = 1)
NN.upload_nn(r"NNs\newneuralnet1000.pth")

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
                      p_I = p_I_og[k], mu = mu_og[k],
                      virtual = 1,
                      u_true =  X_test_og[k])
            
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
# scipy.io.savemat(r"Data3/mean_relerror_K.mat", dict(mean_relerror_K = mean_relerror_K))
# scipy.io.savemat(r"Data3/mean_relerror_phi.mat", dict(mean_relerror_phi = mean_relerror_phi))
# scipy.io.savemat(r"Data3/mean_time.mat", dict(mean_time = mean_time))
# scipy.io.savemat(r"Data3/std_relerror_K.mat", dict(std_relerror_K = std_relerror_K))
# scipy.io.savemat(r"Data3/std_relerror_phi.mat", dict(std_relerror_phi = std_relerror_phi))
# scipy.io.savemat(r"Data3/std_time.mat", dict(std_time = std_time))

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
#plt.ylim([0,0.5])
plt.xticks([0,0.05,0.1])
plt.tight_layout()
plt.show()


###############################################################################
######################     EKI on experimental data    ########################
###############################################################################

# p_I
PI_DICT = dict({
    "Batch1Test1": 1.0146e+05,
    "Batch1Test2": 1.0078e+05,
    "Batch1Test3": 1.0131e+05,
    "Batch1Test4": 9.8095e+04,
    "Batch1Test5": 9.7555e+04,

    "Batch2Test1": 9.6066e+04,
    "Batch2Test2": 9.2325e+04,
    "Batch2Test3": 8.8987e+04,
    "Batch2Test4": 1.0018e+05,
    "Batch2Test5": 9.7633e+04,
    "Batch2Test6": 9.6012e+04,
    "Batch2Test7": 8.9516e+04,
    "Batch2Test8": 9.3959e+04,

    "Batch3Test1": 1.0863e+05,
    "Batch3Test2": 1.0596e+05,
    "Batch3Test3": 1.0605e+05,
    "Batch3Test4": 1.0277e+05,
    "Batch3Test5": 1.0614e+05,
    "Batch3Test6": 1.0441e+05
    })

MU_DICT = dict({
    "Batch1Test1": 0.1020,
    "Batch1Test2": 0.1020,
    "Batch1Test3": 0.1020,
    "Batch1Test4": 0.1020,
    "Batch1Test5": 0.1020,

    "Batch2Test1": 0.1088,
    "Batch2Test2": 0.1088,
    "Batch2Test3": 0.1088,
    "Batch2Test4": 0.1088,
    "Batch2Test5": 0.1088,
    "Batch2Test6": 0.1088,
    "Batch2Test7": 0.1088,
    "Batch2Test8": 0.1088,

    "Batch3Test1": 0.0922,
    "Batch3Test2": 0.0922,
    "Batch3Test3": 0.0922,
    "Batch3Test4": 0.0922,
    "Batch3Test5": 0.0922,
    "Batch3Test6": 0.0922,
    })

DEFLOC_DICT = dict({
    "Batch1Test1": [0.15,0.15],
    "Batch1Test2": [0.15,0.15],
    "Batch1Test3": [0.21,0.09],
    "Batch1Test4": [0.15,0.15],
    "Batch1Test5": [0.09,0.21],

    "Batch2Test1": [0.15,0.15],
    "Batch2Test2": [0.15,0.15],
    "Batch2Test3": [0.15,0.15],
    "Batch2Test4": [0.15,0.15],
    "Batch2Test5": [0.15,0.15],
    "Batch2Test6": [0.15,0.15],
    "Batch2Test7": [0.09,0.09],
    "Batch2Test8": [0.09,0.09],

    "Batch3Test1": [0.09,0.15],
    "Batch3Test2": [0.09,0.15],
    "Batch3Test3": [0.09,0.09],
    "Batch3Test4": [0.18,0.18],
    "Batch3Test5": [0.09,0.09],
    "Batch3Test6": [0.09,0.09],
    })

batch = '3'
test = '6'
sensors = '3x3'

images = []
for i in range(1,15):
    images.append(mpimg.imread(r"Matlab\Processing\Batch" + batch +
                                r"\Test" + test + 
                                r"\time" + str(i) + r".png"))

NN = NeuralNetwork(Data = Dat,
                    architecture = [len(Dat.TrainX[0]),800,len(Dat.TrainY[0])],
                    activation = "Sigmoid",
                    epochs = 150_000, 
                    learning_rate = 0.001,
                    batch_size = 32,
                    plotting = 1)
NN.upload_nn(r"Data3\newneuralnet1000.pth")

noisy_pressure = read_txt(r"Data\ExperimentalData\Batch" + batch + "_Test" + test + ".txt")
# K_true = read_txt(r"Data\ExperimentalData\Kb" + batch + "t" + test + ".txt")
# phi_true = read_txt(r"Data\ExperimentalData\phib" + batch + "t" + test + ".txt")
# u_true = np.concatenate( (np.reshape(K_true,4096),np.reshape(phi_true,4096)) )
p_defect = np.zeros(1624)
for i in range(14):
    for j in Exp.sensor_dict[sensors]:
        p_defect[j + 116*i] = noisy_pressure[j+23*i]


ob_times = seven_times
inlet_pressure = PI_DICT["Batch" + batch + "Test" + test]
mu_exp = MU_DICT["Batch" + batch + "Test" + test]
sigma1, sigma2 = 0.01, 0.005
def_type = "low"

data_obj = Dat.generate_data(p_defect,
                             sensor_inds = Exp.sensor_dict[sensors],
                             sigma1 = sigma1,
                             sigma2 = sigma2,
                             surr_error = NN.surr_error)
data_obj = [data_obj[1],data_obj[0],data_obj[2],data_obj[3]]

# Create EKI object
eki = EKI(Experiment = Exp,
          Data = Dat,
          NeuralNetwork = NN,
          Data_obj = data_obj,
          t = ob_times,
          n_ensemble = 2_000,
          iter_max = 100,
          p_I = 109_000,
          mu = 0.1,
          virtual = 0)

# Run EKI
posterior_ensemble = eki.run_EKI(plotting = False,ensemble_dep=1)
def_thresholds = [np.quantile(np.concatenate(posterior_ensemble[0][-1][:,:81]),0.15),
                  np.quantile(np.concatenate(posterior_ensemble[0][-1][:,:81]),0.8),
                  np.quantile(np.concatenate(posterior_ensemble[0][-1][:,85:166]),0.15),
                  np.quantile(np.concatenate(posterior_ensemble[0][-1][:,85:166]),0.8)]

plotter = PlotEKI(posterior_ensemble, eki)
plotter.plot_sequential(permeability_only = 0,log_perm=True,sensor_locs=Exp.sensor_dict[sensors],
                    plot_prob = 1,defect_type=def_type,defect_tresholds=def_thresholds,
                    images_available = 1, images = [images[i-1] for i in ob_times],
                    truth_available = 0, truth = 0,
                    plot_circle=1,circle_centre=DEFLOC_DICT["Batch" + batch + "Test" + test],radius=0.04,rm_first_col=0)
plt.show()
push_ensemble(posterior_ensemble[0][-1], Dat, NN, eki, data_obj[1], Exp.sensor_dict[sensors])


scipy.io.savemat(r"Matlab\Processing\Batch" + batch + r"\Test" + test + r"\posterior_samples.mat",
                 dict(x=posterior_ensemble[0][-1][np.random.choice(5_000, 500, replace=False)]))

EXP_DICT = dict({"batch": batch, "test" : test,
                 "sigma1" : sigma1, "sigma2" : sigma2,
                 "observation times" : ob_times, "sensor config." : sensors,
                "p_I" : inlet_pressure, "mu" : mu_exp,
                "central thresholds" : def_thresholds,
                "RT thresholds" : [10e-10,0.8069], "thresholds chosen" : def_type})
json.dump(EXP_DICT,open(r"Matlab\Processing\Batch" + batch + r"\Test" + test + r"\parameters.txt",'w'))


###############################################################################
############################    Plot partition   ##############################
###############################################################################

import matplotlib.image as mpimg
img = mpimg.imread('Figures\glassfibre.jpg')

plt.figure(figsize=(15,10))
rt_width = 0.003
imgplot = plt.imshow(img,cmap="gray",extent=[0,0.3,0,0.3])
for i in range(1,9):    
    plt.axvline(0.3/9*i,rt_width*3.333,3.333*(0.3-rt_width),color='k')
    plt.axhline((0.3-2*rt_width)/9*i+rt_width,0,3.3333*0.3,color='k')
plt.axhline(rt_width,0,3.3333*0.3,color='k')
plt.axhline(0.3-rt_width,0,3.3333*0.3,color='k')
plt.axvline(0.15,0,3.3333*rt_width,color='k')
plt.axvline(0.15,3.3333*(0.3-rt_width),3.3333*0.3,color='k')
for j in range(9):
    for i in range(9):
        plt.text(0.3/18 + 0.3/9*j-0.00674,rt_width + (0.3-2*rt_width)/18 + (0.3-2*rt_width)/9*i - 0.004,
                 r"$R_{{{k}}}$".format(k=9*i+j+1),fontsize=22)


###############################################################################
######################    Plot sensor configurations   ########################
###############################################################################

img = np.flipud(mpimg.imread(r"Figures\glassfibre.jpg"))

plt.figure(dpi=300,figsize=(12,8))
rt_width = 0.003
plt.subplot(2,3,1)
imgplot = plt.imshow(img,cmap="gray",extent=[0,0.3,0,0.3])
for i in range(1,9):    
    plt.axvline(0.3/9*i,rt_width*3.333,3.333*(0.3-rt_width),color='k')
    plt.axhline((0.3-2*rt_width)/9*i+rt_width,0,3.3333*0.3,color='k')
plt.axhline(rt_width,0,3.3333*0.3,color='k')
plt.axhline(0.3-rt_width,0,3.3333*0.3,color='k')
plt.axvline(0.15,0,3.3333*rt_width,color='k')
plt.axvline(0.15,3.3333*(0.3-rt_width),3.3333*0.3,color='k')
plt.scatter(Exp.all_sensor_locs[Exp.sensor_dict['2x2'],0],
            Exp.all_sensor_locs[Exp.sensor_dict['2x2'],1],
            color="black",edgecolors="black",s=40)
plt.ylim([0,0.3])
plt.xlim([0,0.3])
plt.title("2x2 (Experimental)")

plt.subplot(2,3,2)
imgplot = plt.imshow(img,cmap="gray",extent=[0,0.3,0,0.3])
for i in range(1,9):    
    plt.axvline(0.3/9*i,rt_width*3.333,3.333*(0.3-rt_width),color='k')
    plt.axhline((0.3-2*rt_width)/9*i+rt_width,0,3.3333*0.3,color='k')
plt.axhline(rt_width,0,3.3333*0.3,color='k')
plt.axhline(0.3-rt_width,0,3.3333*0.3,color='k')
plt.axvline(0.15,0,3.3333*rt_width,color='k')
plt.axvline(0.15,3.3333*(0.3-rt_width),3.3333*0.3,color='k')
plt.scatter(Exp.all_sensor_locs[Exp.sensor_dict['3x3'],0],
            Exp.all_sensor_locs[Exp.sensor_dict['3x3'],1],
            color="black",edgecolors="black",s=40)
plt.ylim([0,0.3])
plt.xlim([0,0.3])
plt.title("3x3 (Experimental)")

plt.subplot(2,3,3)
imgplot = plt.imshow(img,cmap="gray",extent=[0,0.3,0,0.3])
for i in range(1,9):    
    plt.axvline(0.3/9*i,rt_width*3.333,3.333*(0.3-rt_width),color='k')
    plt.axhline((0.3-2*rt_width)/9*i+rt_width,0,3.3333*0.3,color='k')
plt.axhline(rt_width,0,3.3333*0.3,color='k')
plt.axhline(0.3-rt_width,0,3.3333*0.3,color='k')
plt.axvline(0.15,0,3.3333*rt_width,color='k')
plt.axvline(0.15,3.3333*(0.3-rt_width),3.3333*0.3,color='k')
plt.scatter(Exp.all_sensor_locs[Exp.sensor_dict['4x4'],0],
            Exp.all_sensor_locs[Exp.sensor_dict['4x4'],1],
            color="black",edgecolors="black",s=40)
plt.ylim([0,0.3])
plt.xlim([0,0.3])
plt.title("4x4")

plt.subplot(2,3,4)
imgplot = plt.imshow(img,cmap="gray",extent=[0,0.3,0,0.3])
for i in range(1,9):    
    plt.axvline(0.3/9*i,rt_width*3.333,3.333*(0.3-rt_width),color='k')
    plt.axhline((0.3-2*rt_width)/9*i+rt_width,0,3.3333*0.3,color='k')
plt.axhline(rt_width,0,3.3333*0.3,color='k')
plt.axhline(0.3-rt_width,0,3.3333*0.3,color='k')
plt.axvline(0.15,0,3.3333*rt_width,color='k')
plt.axvline(0.15,3.3333*(0.3-rt_width),3.3333*0.3,color='k')
plt.scatter(Exp.all_sensor_locs[Exp.sensor_dict['5x5'],0],
            Exp.all_sensor_locs[Exp.sensor_dict['5x5'],1],
            color="black",edgecolors="black",s=40)
plt.ylim([0,0.3])
plt.xlim([0,0.3])
plt.title("5x5")

plt.subplot(2,3,5)
imgplot = plt.imshow(img,cmap="gray",extent=[0,0.3,0,0.3])
for i in range(1,9):    
    plt.axvline(0.3/9*i,rt_width*3.333,3.333*(0.3-rt_width),color='k')
    plt.axhline((0.3-2*rt_width)/9*i+rt_width,0,3.3333*0.3,color='k')
plt.axhline(rt_width,0,3.3333*0.3,color='k')
plt.axhline(0.3-rt_width,0,3.3333*0.3,color='k')
plt.axvline(0.15,0,3.3333*rt_width,color='k')
plt.axvline(0.15,3.3333*(0.3-rt_width),3.3333*0.3,color='k')
plt.scatter(Exp.all_sensor_locs[Exp.sensor_dict['Half'],0],
            Exp.all_sensor_locs[Exp.sensor_dict['Half'],1],
            color="black",edgecolors="black",s=40)
plt.ylim([0,0.3])
plt.xlim([0,0.3])
plt.title("Half")

plt.subplot(2,3,6)
imgplot = plt.imshow(img,cmap="gray",extent=[0,0.3,0,0.3])
for i in range(1,9):    
    plt.axvline(0.3/9*i,rt_width*3.333,3.333*(0.3-rt_width),color='k')
    plt.axhline((0.3-2*rt_width)/9*i+rt_width,0,3.3333*0.3,color='k')
plt.axhline(rt_width,0,3.3333*0.3,color='k')
plt.axhline(0.3-rt_width,0,3.3333*0.3,color='k')
plt.axvline(0.15,0,3.3333*rt_width,color='k')
plt.axvline(0.15,3.3333*(0.3-rt_width),3.3333*0.3,color='k')
plt.scatter(Exp.all_sensor_locs[Exp.sensor_dict['All'],0],
            Exp.all_sensor_locs[Exp.sensor_dict['All'],1],
            color="black",edgecolors="black",s=40)
plt.ylim([0,0.3])
plt.xlim([0,0.3])
plt.title("All")

plt.tight_layout()
plt.show()

###############################################################################
########################       Plot NN routine      ###########################
###############################################################################

i = 20
plt.figure(figsize=(10,10))
plotter.plot_permeability(Dat.X[i,:85],np.min(Dat.X[0,:85]),np.max(Dat.X[0,:85]),
                          0,1, cbar = False,c_map="jet",rm_axis="False",
                          plt_sensors=False,axis=None)
plt.show()
plt.figure(figsize=(10,10))
plotter.plot_permeability(Dat.X[i,85:170],0,1,0,1, cbar = False,c_map="jet",
                          rm_axis="False",plt_sensors=False,axis=None)
plt.show()
pred_y = NN.F(Dat.TrainX[i])
true_y = Dat.ParameteriseY(Dat.TrainY[i], "BWD")
pred_v = np.diagonal(NN.surr_cov)

plt.figure(figsize=(5,5))
for i in Exp.sensor_dict['2x2']:
    sensor_pred = [pred_y[i+116*j] for j in range(14)]
    sensor_true = [true_y[i+116*j] for j in range(14)]
    sensor_var = [pred_v[i+116*j] for j in range(14)]
    plt.plot([1, 3, 5, 7, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55],sensor_pred,color="black")
    plt.plot([1, 3, 5, 7, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55],sensor_true,color="red",linestyle="--")
    plt.fill_between([1, 3, 5, 7, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55],
                     sensor_pred-2*np.sqrt(sensor_var),sensor_pred+2*np.sqrt(sensor_var),color="grey",alpha=0.6)
plt.xticks([1, 3, 5, 7, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55])
plt.yticks([0,25000,50000,75000,100000])
plt.ylabel("Pressure")
plt.xlabel("Observation time")
plt.title("Sensor pressure (for 2x2 configuration)")
plt.show()


###############################################################################
######################       Plot all synthetic       #########################
###############################################################################

sensor_list = ['2x2','3x3','4x4','5x5','Half','All']
p_list = []
K_list = []
phi_list = []
time_list = []
def_type = ["high","both","low","low"]
inlet_pressures = [110_000,105_000,102_500,100_000]
viscosities = [0.091,0.105,0.095,0.100]

for i in range(1,5):
    file_dir = r'Data3\\Virtual\\'
    p_list.append(read_txt(file_dir + "pressures" + str(i) + ".txt"))
    K_list.append(read_txt(file_dir + "K" + str(i) + ".txt"))
    phi_list.append(read_txt(file_dir + "phi" + str(i) + ".txt"))

posteriors = []

for i in range(4):
    for j in range(6):
        print(i,j)
        data_obj = Dat.generate_data(p_list[i],
                                 sensor_inds = Exp.sensor_dict[sensor_list[j]],
                                 sigma1 = 0.00,
                                 sigma2 = 0.005,
                                 surr_error = NN.surr_error)
        eki = EKI(Experiment = Exp,
                  Data = Dat,
                  NeuralNetwork = NN,
                  Data_obj = data_obj,
                  t = 14,
                  n_ensemble = 10_000,
                  iter_max = 100,
                  p_I = inlet_pressures[i],
                  mu = viscosities[i],
                  virtual = 0)
        posterior_ensemble = eki.run_EKI(plotting = False,ensemble_dep=1)
        posteriors.append(posterior_ensemble[0])
        time_list.append(posterior_ensemble[1])
plotter = PlotEKI(posterior_ensemble, eki)

scipy.io.savemat(r"VirtualPosteriors.mat",dict(VirtualPosteriors=posteriors))

###############################################################################

fullmodel_low = [scipy.io.loadmat("Data3/Probfile_two_by_two.mat")['Ensemble'].T]
fullmodel_low.append(scipy.io.loadmat("Data3/Probfile_three_by_three.mat")['Ensemble'].T)
fullmodel_low.append(scipy.io.loadmat("Data3/Probfile_four_by_four.mat")['Ensemble'].T)
fullmodel_low.append(scipy.io.loadmat("Data3/Probfile_five_by_five.mat")['Ensemble'].T)
fullmodel_low.append(scipy.io.loadmat("Data3/Probfile_half_sensors.mat")['Ensemble'].T)
fullmodel_low.append(scipy.io.loadmat("Data3/Probfile_all_sensors.mat")['Ensemble'].T)

defect_tresholds = [3e-10,6e-10,0.5437539644525995,0.7219149264955291]
x = np.linspace( 0.0 , 0.3 , 50 )
y = np.linspace( 0.0 , 0.3 , 50 )  
a, b = np.meshgrid( x , y )
C1 = (a-0.075) ** 2 + (b-0.15) ** 2 - 0.04**2
circle_cmap = "binary"
circle_cmap2 = "gray"
fig,axes = plt.subplots(2,6,figsize=(16,5),constrained_layout=True)
i = 0
for j in range(len(sensor_list)):
    
    prop_mat1 = np.zeros(85)
    prop_mat2 = np.zeros(85)
   
    prop_mat1[:eki.Experiment.M**2] = np.mean(np.logical_and(
        posteriors[len(sensor_list)*i+j][:,:eki.Experiment.M**2] > defect_tresholds[1],
        posteriors[len(sensor_list)*i+j][:,(eki.Experiment.M**2+2*eki.Experiment.M_RT):(2*eki.Experiment.M**2+2*eki.Experiment.M_RT)] > defect_tresholds[3]),axis=0)
    prop_mat1[eki.Experiment.M**2:] = np.mean(np.logical_and(
        posteriors[len(sensor_list)*i+j][:,eki.Experiment.M**2:(eki.Experiment.M**2+2*eki.Experiment.M_RT)] > 45e-10,
        posteriors[len(sensor_list)*i+j][:,(2*eki.Experiment.M**2+2*eki.Experiment.M_RT):] > 0.934),axis=0)
    prop_mat2[:eki.Experiment.M**2] = np.mean(fullmodel_low[j][:,:eki.Experiment.M**2] > defect_tresholds[1],axis=0)
    prop_mat2[eki.Experiment.M**2:] = np.mean(fullmodel_low[j][:,eki.Experiment.M**2:(eki.Experiment.M**2+2*eki.Experiment.M_RT)] > 45e-10,axis=0)

    im1 = plotter.plot_permeability(prop_mat1,0, 1,
                                    [0],virtual=1,axis=axes[0,j],c_map="Reds")
    axes[0,j].set_xticks([0,0.15,0.3])
    axes[0,j].set_yticks([0,0.15,0.3])
    axes[0,j].set_title(sensor_list[j])
    im1 = plotter.plot_permeability(prop_mat2,0, 1,
                                    [0],virtual=1,axis=axes[1,j],c_map="Reds")
    axes[1,j].set_xticks([0,0.15,0.3])
    axes[1,j].set_yticks([0,0.15,0.3])


fig.colorbar(im1, ax=axes[:,:],shrink=0.96,pad=0.01,location="right",aspect=20)

[axes[i,j].contour( a , b , C1 , [0] ,cmap=circle_cmap2,linewidths=4,linestyles="-") for i in range(2) for j in range(6)]
[axes[i,j].contour( a , b , C1 , [0] ,cmap=circle_cmap,linewidths=2,linestyles="-") for i in range(2) for j in range(6)]
plt.show()

###############################################################################

for i in range(len(K_list)):
    K_list[i][1,:] = K_list[i][0,:]
    K_list[i][-2,:] = K_list[i][-1,:]
    phi_list[i][1,:] = phi_list[i][0,:]
    phi_list[i][-2,:] = phi_list[i][-1,:]
    K_list[i] = np.flipud(K_list[i])
    phi_list[i] = np.flipud(phi_list[i])
    
fontsize = 20
log_ensemble_K = [np.log(posteriors[i][:,:85]) for i in range(len(p_list)*len(sensor_list))]
fig,axes = plt.subplots(len(p_list),len(sensor_list)+1,figsize=(fontsize,0.6*fontsize),constrained_layout=True)
for i in range(len(p_list)):
    
    im1 = plotter.plot_permeability(np.log(np.reshape(K_list[i],64**2)),
                             np.log(Exp.min_perm_central), np.log(Exp.max_perm_RT),
                             [0],virtual=0,rm_axis=0,axis=axes[i,0],
                             c_map=nlcmap(cm.turbo, [0, 5, 10, 15, 20, 40, 70, 100]))
    for j in range(len(sensor_list)):
        im2 = plotter.plot_permeability(np.mean(log_ensemble_K[len(sensor_list)*i+j],axis=0),
                                 np.log(Exp.min_perm_central), np.log(Exp.max_perm_RT),
                                 [0],virtual=1,axis=axes[i,j+1],
                                 c_map=nlcmap(cm.turbo, [0, 5, 10, 15, 20, 40, 70, 100]))
        
fig.colorbar(im1, ax=axes[1:,0],shrink=0.90,pad=0.01,location="bottom",aspect=13)
fig.colorbar(im2, ax=axes[1:,1:],shrink=0.98,pad=0.01,location="bottom",aspect=80)
[axes[0,j+1].set_title(['2x2','3x3','4x4','5x5','Half','All'][j],fontsize=fontsize) for j in range(len(sensor_list))]
axes[0,0].set_title(r"$\log K^{\dagger}$",fontsize=fontsize)
line = plt.Line2D([0.155,0.155],[0,1], transform=fig.transFigure, color="black",linewidth=3,linestyle="--")
fig.add_artist(line)
plt.show()

ensemble_phi = [posteriors[i][:,85:] for i in range(len(p_list)*len(sensor_list))]
fig,axes = plt.subplots(len(p_list),len(sensor_list)+1,figsize=(fontsize,0.6*fontsize),constrained_layout=True)
for i in range(len(p_list)):
    
    im1 = plotter.plot_permeability(np.reshape(phi_list[i],64**2),
                                    Exp.min_poro_central,Exp.max_poro_RT,
                             [0],virtual=0,rm_axis=0,axis=axes[i,0])
    for j in range(len(sensor_list)):
        im2 = plotter.plot_permeability(np.mean(ensemble_phi[len(sensor_list)*i+j],axis=0),
                                        Exp.min_poro_central, Exp.max_poro_RT,
                                 [0],virtual=1,axis=axes[i,j+1])
fig.colorbar(im1, ax=axes[1:,0],shrink=0.90,pad=0.01,location="bottom",aspect=13)
fig.colorbar(im2, ax=axes[1:,1:],shrink=0.98,pad=0.01,location="bottom",aspect=80)
[axes[0,j+1].set_title(['2x2','3x3','4x4','5x5','Half','All'][j],fontsize=fontsize) for j in range(len(sensor_list))]
axes[0,0].set_title(r"$\phi^{\dagger}$",fontsize=fontsize)
line = plt.Line2D([0.155,0.155],[0,1], transform=fig.transFigure, color="black",linewidth=3,linestyle="--")
fig.add_artist(line)
plt.show()
       
###############################################################################

std_max = np.max([np.max(np.std(np.log(posteriors[i][:,:85]),axis=0)) for i in range(len(p_list)*len(sensor_list))])

fig,axes = plt.subplots(len(p_list),len(sensor_list)+1,figsize=(fontsize,0.6*fontsize),constrained_layout=True)
for i in range(len(p_list)):
    
    im1 = plotter.plot_permeability(np.log(np.reshape(K_list[i],64**2)),
                             np.log(Exp.min_perm_central), np.log(Exp.max_perm_RT),
                             [0],virtual=0,rm_axis=0,axis=axes[i,0],c_map=nlcmap(cm.turbo, [0, 5, 10, 15, 20, 40, 70, 100]))
    for j in range(len(sensor_list)):
        im2 = plotter.plot_permeability(np.std(log_ensemble_K[len(sensor_list)*i+j],axis=0),
                                 0, std_max,
                                 [0],virtual=1,axis=axes[i,j+1],c_map="binary")
fig.colorbar(im1, ax=axes[1:,0],shrink=0.90,pad=0.01,location="bottom",aspect=13)
fig.colorbar(im2, ax=axes[1:,1:],shrink=0.98,pad=0.01,location="bottom",aspect=80)
[axes[0,j+1].set_title(['2x2','3x3','4x4','5x5','Half','All'][j],fontsize=fontsize) for j in range(len(sensor_list))]
axes[0,0].set_title(r"$\log K^{\dagger}$",fontsize=fontsize)
line = plt.Line2D([0.155,0.155],[0,1], transform=fig.transFigure, color="black",linewidth=3,linestyle="--")
fig.add_artist(line)
plt.show()

std_max = np.max([np.max(np.std(posteriors[i][:,85:],axis=0)) for i in range(len(p_list)*len(sensor_list))])

fig,axes = plt.subplots(len(p_list),len(sensor_list)+1,figsize=(fontsize,0.6*fontsize),constrained_layout=True)
for i in range(len(p_list)):
    
    im1 = plotter.plot_permeability(np.reshape(phi_list[i],64**2),
                                    Exp.min_poro_central, Exp.max_poro_RT,
                             [0],virtual=0,rm_axis=0,axis=axes[i,0],c_map="jet")
    for j in range(len(sensor_list)):
        im2 = plotter.plot_permeability(np.std(ensemble_phi[len(sensor_list)*i+j],axis=0),
                                 0, std_max,
                                 [0],virtual=1,axis=axes[i,j+1],c_map="binary")
fig.colorbar(im1, ax=axes[1:,0],shrink=0.90,pad=0.01,location="bottom",aspect=13)
fig.colorbar(im2, ax=axes[1:,1:],shrink=0.98,pad=0.01,location="bottom",aspect=80)
[axes[0,j+1].set_title(['2x2','3x3','4x4','5x5','Half','All'][j],fontsize=fontsize) for j in range(len(sensor_list))]
axes[0,0].set_title(r"$\phi^{\dagger}$",fontsize=fontsize)
line = plt.Line2D([0.155,0.155],[0,1], transform=fig.transFigure, color="black",linewidth=3,linestyle="--")
fig.add_artist(line)
plt.show()

###############################################################################

fig,axes = plt.subplots(len(p_list),len(sensor_list)+1,figsize=(fontsize,0.6*fontsize),constrained_layout=True)
for i in range(len(p_list)):
    
    im1 = plotter.plot_permeability(np.log(np.reshape(K_list[i],64**2)),
                             np.log(Exp.min_perm_central), np.log(Exp.max_perm_RT),
                             [0],virtual=0,rm_axis=0,axis=axes[i,0],c_map=nlcmap(cm.turbo, [0, 5, 10, 15, 20, 40, 70, 100]))
    for j in range(len(sensor_list)):
        
        prop_mat = np.zeros(85)
        
        if def_type[i] == "low":
            prop_mat[:eki.Experiment.M**2] = np.mean(np.logical_and(posteriors[len(sensor_list)*i+j][:,:eki.Experiment.M**2] < defect_tresholds[0],
                                                                    posteriors[len(sensor_list)*i+j][:,(eki.Experiment.M**2+2*eki.Experiment.M_RT):(2*eki.Experiment.M**2+2*eki.Experiment.M_RT)] < defect_tresholds[2]),axis=0)
            prop_mat[eki.Experiment.M**2:] = np.mean(np.logical_and(posteriors[len(sensor_list)*i+j][:,eki.Experiment.M**2:(eki.Experiment.M**2+2*eki.Experiment.M_RT)] > 45e-10,
                                                                    posteriors[len(sensor_list)*i+j][:,(2*eki.Experiment.M**2+2*eki.Experiment.M_RT):] > 0.934),axis=0)

        elif def_type[i] == "both":
            prop_mat[:eki.Experiment.M**2] = np.mean(np.logical_and((posteriors[len(sensor_list)*i+j][:,:eki.Experiment.M**2] < defect_tresholds[0]) + (posteriors[len(sensor_list)*i+j][:,:eki.Experiment.M**2] > defect_tresholds[1]),
                                                                        (posteriors[len(sensor_list)*i+j][:,(eki.Experiment.M**2+2*eki.Experiment.M_RT):(2*eki.Experiment.M**2+2*eki.Experiment.M_RT)] < defect_tresholds[2]) + (posteriors[len(sensor_list)*i+j][:,(eki.Experiment.M**2+2*eki.Experiment.M_RT):(2*eki.Experiment.M**2+2*eki.Experiment.M_RT)] > defect_tresholds[3])),
                                                          axis=0)
            prop_mat[eki.Experiment.M**2:] = np.mean(np.logical_and(posteriors[len(sensor_list)*i+j][:,eki.Experiment.M**2:(eki.Experiment.M**2+2*eki.Experiment.M_RT)] > 45e-10,
                                                                        posteriors[len(sensor_list)*i+j][:,(2*eki.Experiment.M**2+2*eki.Experiment.M_RT):] > 0.934),
                                                          axis=0)
        else:
            prop_mat[:eki.Experiment.M**2] = np.mean(np.logical_and(posteriors[len(sensor_list)*i+j][:,:eki.Experiment.M**2] > defect_tresholds[1],
                                                                        posteriors[len(sensor_list)*i+j][:,(eki.Experiment.M**2+2*eki.Experiment.M_RT):(2*eki.Experiment.M**2+2*eki.Experiment.M_RT)] > defect_tresholds[3]),
                                                          axis=0)
            prop_mat[eki.Experiment.M**2:] = np.mean(np.logical_and(posteriors[len(sensor_list)*i+j][:,eki.Experiment.M**2:(eki.Experiment.M**2+2*eki.Experiment.M_RT)] > 45e-10,
                                                                        posteriors[len(sensor_list)*i+j][:,(2*eki.Experiment.M**2+2*eki.Experiment.M_RT):] > 0.934),
                                                          axis=0)

        im2 = plotter.plot_permeability(prop_mat,0, 1,
                                        [0],virtual=1,axis=axes[i,j+1],c_map="Reds")

        
fig.colorbar(im1, ax=axes[1:,0],shrink=0.90,pad=0.01,location="bottom",aspect=13)
fig.colorbar(im2, ax=axes[1:,1:],shrink=0.98,pad=0.01,location="bottom",aspect=80)
[axes[0,j+1].set_title(['2x2','3x3','4x4','5x5','Half','All'][j],fontsize=fontsize) for j in range(len(sensor_list))]
axes[0,0].set_title(r"$\log K^{\dagger}$",fontsize=fontsize)
line = plt.Line2D([0.155,0.155],[0,1], transform=fig.transFigure, color="black",linewidth=3,linestyle="--")
fig.add_artist(line)

rect1 = patches.Rectangle((0.035, 0.045), 0.029, 0.21, linewidth=4, edgecolor='black', facecolor='none',linestyle="-")
rect2 = patches.Rectangle((0.035, 0.045), 0.029, 0.21, linewidth=2, edgecolor='white', facecolor='none',linestyle="-")
rect3 = patches.Rectangle((0.15, 0.215), 0.15, 0.029, linewidth=4, edgecolor='black', facecolor='none',linestyle="-")
rect4 = patches.Rectangle((0.15, 0.215), 0.15, 0.029, linewidth=2, edgecolor='white', facecolor='none',linestyle="-")
C2 = (a-0.1) ** 2 + (b-0.1) ** 2 - 0.04**2
C3 = (a-0.2) ** 2 + (b-0.2) ** 2 - 0.04**2
C4 = (a-0.2) ** 2 + (b-0.1) ** 2 - 0.035**2


[axes[0,i].contour( a , b , C1 , [0] ,cmap=circle_cmap2,linewidths=4,linestyles="-") for i in range(1,7)]
[axes[0,i].contour( a , b , C1 , [0] ,cmap=circle_cmap,linewidths=2,linestyles="-") for i in range(1,7)]

[axes[2,i].contour( a , b , C2 , [0] ,cmap=circle_cmap2,linewidths=4,linestyles="-") for i in range(1,7)]
[axes[2,i].contour( a , b , C2 , [0] ,cmap=circle_cmap,linewidths=2,linestyles="-") for i in range(1,7)]
[axes[2,i].contour( a , b , C3 , [0] ,cmap=circle_cmap2,linewidths=4,linestyles="-") for i in range(1,7)]
[axes[2,i].contour( a , b , C3 , [0] ,cmap=circle_cmap,linewidths=2,linestyles="-") for i in range(1,7)]

[axes[3,i].contour( a , b , C4 , [0] ,cmap=circle_cmap2,linewidths=4,linestyles="-") for i in range(1,7)]
[axes[3,i].contour( a , b , C4 , [0] ,cmap=circle_cmap,linewidths=2,linestyles="-") for i in range(1,7)]
[axes[3,i].add_patch(copy(rect1)) for i in range(1,7)]
[axes[3,i].add_patch(copy(rect2)) for i in range(1,7)]

