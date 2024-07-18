# -*- coding: utf-8 -*-
"""
Created on Mon Nov 13 12:21:55 2023

@author: pmymc12
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import torch.optim as optim
import tqdm
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

class NeuralNetwork:
    
    """
    Description
    -----------
    Neural network class contains all data and functions that are used to train
    the surrogate. This class has been written to simplify the process of using
    PyTorch by letting users provide only the data and architecture.

    Parameters
    ----------
    Data : class
        Data object found in Data.py.
    architecture : list
        A list containing the number of nodes in each layer (incl. input/output).
    epochs : int
        Max. number of epochs NN is trained for (unless early stopping used).
    lr: float
        Learning rate of Adam algorithm.
    plotting : bool
        Logical stating whether training is to be plotted.
    model_trained : bool
        Logical stating whether model has been trained or is awaiting training.
    surr_cov : numpy.ndarray
        Surrogate covariance matrix of surrogate error evaluated on validation.
    activation : str
        The type of activation function chosen. Defaults to Sigmoid.
    model : torch model
        A neural network model
    
    Functions
    -------
    Public train_nn :
        Trains the neural network and stores training data (e.g. MSE).
    Private _plot_training :
        Plots the training process. Compares train and validation MSE.
    Private _early_stopper :
        Stops the training process early if validation error fails to improve.
    Public upload_nn :
        Uploads a neural network if already trained previously.
    Public F :
        Evaluates the forward map using the surrogate neural network.
    Public assess_surrogate :
        Assesses the performance of surrogate on validation set incl. relative
        error and calibration score. Also plots n examples on validation set.

    Examples
    --------
    >>> NN_object = NeuralNetwork(Data_obj,[5,100,2],"ReLU",5_000,0.1,False)
    >>> NN_object.train_nn()
    >>> NN_object.assess_surrogate(20)
    >>> torch.save(NN_object.model, r"path_to_model.pth")
    
    
    >>> NN_object = NeuralNetwork(Data_obj,[5,100,2],"ReLU",5_000, 0.1,False)
    >>> NN_object.upload_nn("path_to_model")
    >>> NN_object.F(Test_X)
    """
    
    def __init__(self,Data,architecture,activation,epochs,learning_rate,batch_size,plotting):
        
        # Various sets
        self.Data = Data
        
        assert all(isinstance(layer, int) for layer in architecture), "Nodes in each layer must be integer valued"
        self.architecture = architecture
        
        assert epochs > 0, "Epochs must be positive"
        self.epochs = epochs
        
        assert learning_rate > 0, "Learning rate must be positive"
        self.lr = learning_rate
        
        assert batch_size >= 0, "Batch size must be non-negative"
        self.batch_size = batch_size
        
        assert plotting in [0,1], "Plotting must be 0 or 1"
        self.plotting = plotting
        
        self.model_trained = 0
        self.param_count = 0
        self.surr_cov = "Training required first"
        
        # Early stopping parameters
        self.patience = 50
        self.min_delta = 0.00
        self.counter = 0
        self.min_validation_loss = float('inf')
        
        # Check activation function type
        if activation == "ReLU":
            self.activation = nn.ReLU()
        elif activation == "LeakyReLU":
            self.activation = nn.LeakyReLU()
        elif activation == "Tanh":
            self.activation = nn.Tanh() 
        else:
            self.activation = nn.Sigmoid()
            
        # Check architecture is valid
        assert architecture[0] == len(Data.TrainX[0]), "Nodes in first layer must be input size"
        assert architecture[-1] == len(Data.TrainY[0]), "Nodes in final layer must be output size"
    
        # Generate model object and count parameters
        self.model = nn.Sequential()
        for i in range(len(architecture)-1):
            self.model.append(nn.Linear(architecture[i],architecture[i+1]))
            self.model.append(self.activation)
            self.param_count += architecture[i]*architecture[i+1]+architecture[i+1]
        self.model.pop(-1)
        print("Model initialised with " + str(self.param_count) + " parameters.") 
    
    # Trains NN using train data and evaluates with develop data
    def train_nn(self):
        
        # Define data as tensors
        train_x, train_y = torch.tensor(self.Data.TrainX).float(), torch.tensor(self.Data.TrainY).float()
        develop_x, develop_y = torch.tensor(self.Data.DevelopX).float(), torch.tensor(self.Data.DevelopY).float()
        
        # Move data to GPU
        model = self.model.to('cuda:0')
        train_x, train_y = train_x.to('cuda:0'), train_y.to('cuda:0')
        develop_x, develop_y = develop_x.to('cuda:0'), develop_y.to('cuda:0')
        
        # Train the model
        loss_fn = nn.MSELoss()
        optimizer = optim.Adam(model.parameters(), lr=self.lr)
        mse_train = []
        mse_dev = []
        
        # If batch_size > 0, use minibatching, else do not use minibatching
        if self.batch_size > 0:
            print("Using mini-batching...")
            dataset = TensorDataset(train_x, train_y)
            dataloader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)
        
            for epoch in tqdm.tqdm(range(self.epochs)):
                for batch_X, batch_Y in dataloader:
                    y_pred = model(batch_X.cuda())
                    loss = loss_fn(y_pred, batch_Y.cuda())
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                mse_train.append(loss.item())
                
                # Assess validation MSE at epoch
                with torch.no_grad():
                    y_pred = model(develop_x)
                mse_curr = np.mean( ((y_pred - develop_y)**2).cpu().numpy())
                mse_dev.append(mse_curr)
                
                # Check early stopping criterion
                if mse_curr < self.min_validation_loss:
                    best_model = model
                    best_iterate = epoch
                if self._early_stopper(mse_curr):
                    print("Early exit")
                    break
                
                # Plot train/develop MSE
                if self.plotting:
                    self._plot_training(epoch,mse_curr,mse_train,mse_dev)
                del loss, y_pred
        else:
            print("Not using mini-batching...")
      
            for epoch in tqdm.tqdm(range(self.epochs)):
                y_pred = model(train_x)
                loss = loss_fn(y_pred, train_y)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                mse_train.append(loss.item())
                
                # Assess validation MSE at epoch
                with torch.no_grad():
                    y_pred = model(develop_x)
                mse_curr = np.mean( ((y_pred - develop_y)**2).cpu().numpy())
                mse_dev.append(mse_curr)
                
                # Check early stopping criterion
                if mse_curr < self.min_validation_loss:
                    best_model = model
                    best_iterate = epoch
                if self._early_stopper(mse_curr):
                    print("Early exit")
                    break
                    
                # Plot train/develop MSE
                if self.plotting:
                    self._plot_training(epoch,mse_curr,mse_train,mse_dev)
                del loss, y_pred
        
        # Move back to CPU to offload memory
        model = best_model.to('cpu')
        train_x, train_y = train_x.to('cpu'), train_y.to('cpu')
        develop_x, develop_y = develop_x.to('cpu'), develop_y.to('cpu')
        
        # (Re)define variables
        self.model = model
        self.best_iterate = best_iterate
        self.model_trained = 1
        self.surr_cov = np.cov( (self.F(self.Data.DevelopX)-
                               self.Data.ParameteriseY(self.Data.DevelopY,"BWD")).T )
        self.surr_error = np.mean(self.F(self.Data.DevelopX)-self.Data.ParameteriseY(self.Data.DevelopY,"BWD"),
                                  axis=0)
        self.train_mse = mse_train
        self.dev_mse = mse_dev
    
    # Plotting function for training
    def _plot_training(self,epoch,mse_curr,loss_vec,mse_dev):
        
        if epoch % int(self.epochs/20) == 0:
            plt.close()
            plt.plot(np.log10(loss_vec),label="Train log-MSE")
            print('- MSE : %2.4f' % mse_curr)
            plt.plot(np.log10(mse_dev),label="Develop log-MSE")
            plt.legend()
            plt.ylabel("log-MSE")
            plt.xlabel("Epochs")
            plt.show()
    
    # Stops training process if validation MSE fails to improve over time
    def _early_stopper(self,validation_loss):
        if validation_loss < self.min_validation_loss:
            self.min_validation_loss = validation_loss
            self.counter = 0
            self.lr = 0.001
        elif validation_loss > (self.min_validation_loss + self.min_delta):
            self.counter += 1
            self.lr = 0.1*self.lr
            if self.counter >= self.patience:
                return True
        return False
    
    # Upload an existing NN which is saved locally
    def upload_nn(self,path_to_model):
        
        if self.model_trained:
            print("Overwriting previous neural network...")
        else:
            print("Uploading neural network...")
            self.model_trained = 1
        self.model = torch.load(path_to_model)
        
        # Must recalculate surrogate error mean/covariance
        data_misfit = self.F(self.Data.DevelopX)-self.Data.ParameteriseY(self.Data.DevelopY,"BWD")
        self.surr_cov = np.cov(data_misfit.T)
        self.surr_error = np.mean(data_misfit,axis=0)
    
    
    # The surrogate forward map
    def F(self,X):
        X = torch.tensor(X).float().to('cuda:0')
        with torch.no_grad():
            y_pred = self.model.cuda()(X.cuda())
       
        return self.Data.ParameteriseY(y_pred.cpu().numpy(),"BWD")

    
    # Assess quality of surrogate
    def assess_surrogate(self,n):
        
        assert self.model_trained, "Neural network not yet trained" 
        
        # Form developing data set
        DevelopY = self.Data.ParameteriseY(self.Data.DevelopY,"BWD")
        preds_m = self.F(self.Data.DevelopX)
        preds_v = np.diagonal(self.surr_cov)
        
        # Plot n examples of surrogate on validation set
        if self.plotting:
            k = int(116*14)
            for i in range(n):
                plt.plot(list(range(k)),preds_m[i,:k],color="black")
                plt.plot(list(range(k)),DevelopY[i,:k],color="red")
                plt.fill_between(list(range(k)),preds_m[i,:k]-2*np.sqrt(preds_v[:k]),preds_m[i,:k]+2*np.sqrt(preds_v[:k]))
                plt.ylim([-10_000,110_000])
                plt.show()
        
        # Relative errors in the develop set
        rel_errors = [np.linalg.norm(DevelopY[i]-preds_m[i])/np.linalg.norm(DevelopY[i]) for i in range(len(DevelopY))]
        
        # Assess the uncertainty estimate
        percs = np.linspace(0,1,10)
        perc_within = np.zeros(len(percs))
        k = 0
        for i in percs:
            z_crit = stats.norm.ppf(1-i/2)
            within_conf = np.sum( ((preds_m - z_crit*np.sqrt(preds_v)) < DevelopY) & ((preds_m + z_crit*np.sqrt(preds_v)) > DevelopY) )
            perc_within[k] = within_conf/(len(DevelopY)*len(DevelopY[0]))
            k += 1
        
        if self.plotting:
            plt.figure(figsize=(5,5))
            plt.plot(1-percs,perc_within)
            plt.plot(percs,percs,linestyle = "--",color="black")
            plt.xlabel("Predicted proportion")
            plt.ylabel("Observed proportion")
            miscallibration = (percs[1]-percs[0])*np.sum(np.abs(perc_within - (1-percs)))
            plt.title("Miscallibration score: " + str(miscallibration))
            plt.show()
            
            # Plot histogram of relative errors
            plt.hist(rel_errors)
            plt.show()
        
        # Average/std of relative errors
        print("Average relative error: " + str(np.mean(rel_errors)))
        print("Std. relative error: " + str(np.std(rel_errors)))
        return np.mean(rel_errors)
    