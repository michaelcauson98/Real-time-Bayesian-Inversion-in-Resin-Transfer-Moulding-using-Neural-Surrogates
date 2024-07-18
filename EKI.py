# -*- coding: utf-8 -*-
"""
Created on Mon Nov 13 15:17:53 2023

@author: pmymc12
"""
import numpy as np
from Utils import permeability_to_porosity
import torch
import matplotlib.pyplot as plt
import scipy
import time

class EKI:
    
    """
    Description
    -----------
    Neural network class contains all data and functions that are used to train
    the surrogate. This class has been written to simplify the process of using
    PyTorch by letting users provide only the data and architecture.

    Parameters
    ----------
    Experiment : class
        Experiment object found in Experiment.py.
    Data : class
        Data object found in Data.py.
    NeuralNetwork : class
        Neural network object found in NeuralNetwork.py.
    Data_obj : list
        Different from class. Contains the actual data and noise covariance used.
    t : list
        Contains the indices of the observation times used (1-indexed)
    n_ensemble : int
        Number of ensemble memebers used.
    iter_max : int
        Maximum number of iterations used.
    p_I : float
        Inlet pressure used for the experiment.
    mu : float
        Viscosity used for the experiment.
    u_true : numpy.ndarray
        Supply u_true if known (e.g. virtual experiment), leave None otherwise.
        
    
    Functions
    -------
    Private _ParameteriseXTensor :
        Parameterises and normalises X data.
    Private _UnitTransformXTensor :
        Transforms X data to [0,1]^N in accordance with the NN training data.
    Private _ParameteriseYTensor :
        Parameterises Y data in accordance with NN training data.
    Public initialise_ensemble :
        Produces the initial ensemble, either from the prior or previous inversions.
    Public forward_map :
        Evaluates the forward map - this is designed to be faster than NN.F
    Public run_EKI_t :
        Runs EKI for data up to and including time t.
    Public run_EKI :
        Runs EKI sequentially using run_EKI_t function.
    Public diagnostic_check :
        Boxplot of the posterior ensemble.
    

    Examples
    --------
    >>> eki_object = EKI(Experiment = Exp, Data = Dat, NeuralNetwork = NN,
                         Data_obj = data_obj, t = [1,3,5,7,9,11,14],
                         n_ensemble = 10_000, iter_max = 100,
                         p_I = 100_000, mu = 0.1, u_true = None)
    >>> posterior_ensemble = eki.run_EKI(plotting = True, ensemble_dep = 0)
    """
    
    def __init__(self,Experiment,Data,NeuralNetwork,Data_obj,t,
                 n_ensemble,iter_max,p_I,mu,u_true=None):
        
        # Objects
        self.Experiment = Experiment
        self.Data = Data
        self.NN = NeuralNetwork
        
        # EKI parameters
        assert n_ensemble > 0, "Number of ensemble members must be positive"
        self.n_ensemble = n_ensemble
        
        assert iter_max > 0, "Max. iterations must be positive"
        self.iter_max = iter_max
        
        self.u_true = u_true
        
        assert p_I < self.Experiment.p_I[1], "p_I must be within the trained range"
        assert p_I > self.Experiment.p_I[0], "p_I must be within the trained range"
        self.p_I = p_I
        
        assert mu < self.Experiment.mu[1], "\mu must be within the trained range"
        assert mu > self.Experiment.mu[0], "\mu must be within the trained range"
        self.mu = mu
        
        # Ensure list of observation times is consistent
        if type(t) == list:
            t = [round(t[i]) for i in range(len(t))]
            t = sorted(list(set(t)))
            self.t = [t[i] for i in range(len(t)) if t[i] <= self.Experiment.times and t[i]>=1]
            self.sequential = True
        elif type(t) == int:
            if t <= self.Experiment.times and t >= 1:
                self.t = t
            else:
                print("t must be an integer or list of integers in [1,...," + str(self.Experiment.times) + "]. Setting t = " + str(self.Experiment.times) + ".")
                self.t = self.Experiment.times
            self.sequential = False
        else:
            print("t must be an integer or list of integers in [1,...," + str(self.Experiment.times) + "]. Setting t = " + str(self.Experiment.times) + ".")
            self.t = self.Experiment.times
            self.sequential = False
        
        
        # Convert useful objects to torch tensors
        self.param_max = torch.tensor(self.Experiment.param_max).float().cuda()
        self.param_min = torch.tensor(self.Experiment.param_min).float().cuda()
        self.param_max_no_press = torch.tensor(self.Experiment.param_max[:len(self.Experiment.param_max)-2]).float().cuda()
        self.param_min_no_press = torch.tensor(self.Experiment.param_min[:len(self.Experiment.param_min)-2]).float().cuda()
        self._logX_means = torch.tensor(self.Data._logX_means[:len(self.Data._logX_means)-2]).float().cuda()
        self._logX_stds = torch.tensor(self.Data._logX_stds[:len(self.Data._logX_stds)-2]).float().cuda()
        self._logY_means = torch.tensor(self.Data._logY_means).float().cuda()
        self._logY_stds = torch.tensor(self.Data._logY_stds).float().cuda()
        self.model = self.NN.model.cuda()
        self.pressures = torch.ones((self.n_ensemble,1)).float().cuda()*self.p_I
        self.viscosities = torch.ones((self.n_ensemble,1)).float().cuda()*self.mu
    
        
        # Add/update data provided
        self.press = torch.tensor(Data_obj[0]).float().cuda()
        self.data = torch.tensor(Data_obj[1]).float().cuda()
        self.gamma = torch.tensor(Data_obj[2]).float().cuda()
        self.all_sensor_inds = Data_obj[3]
        surr_cov = self.NN.surr_cov[np.ix_(Data_obj[3],Data_obj[3])]
        gamma_infl = np.diag(Data_obj[2]) + surr_cov
        Gamma_minus_half = scipy.linalg.sqrtm(scipy.linalg.inv(gamma_infl))
        self.surr_cov = torch.tensor(surr_cov).float().cuda()
        self.Gamma_minus_half = torch.tensor(Gamma_minus_half).float().cuda()
    
    
    # X -> X_new = log( (b-X)/(X-a) ) -> normalised X_new (EKI parameterisation)
    def _ParameteriseXTensor(self,data,direction):
        if direction == "FWD":
            logdata = torch.log( (self.param_max_no_press-data)/(data-self.param_min_no_press) )
            return (logdata - self._logX_means)/self._logX_stds
        else:
            logdata = data*self._logX_stds + self._logX_means
            return (self.param_max_no_press+self.param_min_no_press*torch.exp(logdata))/(torch.exp(logdata)+1)
    
    # Affinely transforms raw X into [0,1]^{170} (surrogate parameterisation)
    # Tensor form of function in Data.py
    def _UnitTransformXTensor(self,data,direction):
        
        if direction == "FWD":
            return (data - self.param_min)/(self.param_max-self.param_min)
        else:
            return data*(self.param_max-self.param_min) + self.param_min

        
    # Y -> log(Y) -> normalise log(Y) (add 1,000 to avoid log(0))
    # Tensor form of function in Data.py
    def _ParameteriseYTensor(self,data,direction):
        
        if direction == "FWD":
            logdata = torch.log(data+1_000)
            return (logdata-self._logY_means)/self._logY_stds
        else:
            logdata = data*self._logY_stds + self._logY_means
            return torch.exp(logdata)-1_000
    
        
    # Set and transform ensemble members.
    # Continuation used in sequential cases or when prior ensemble is provided.
    # If continuation = 1, U_prev must contain the prior ensemble.
    # If prior permeability/porosity are assumed dependent, set ensemble_dependence = 1.
    def initialise_ensemble(self,continuation=0,U_prev = 0,ensemble_dependence = 0):
        
        if continuation:
            #print("Ensemble provided...")
            return U_prev.clone().detach()
        
        # K central
        U_1 = np.random.uniform(self.Experiment.min_perm_central+0.005e-10,
                                self.Experiment.max_perm_central-0.005e-10,
                                (self.n_ensemble,self.Experiment.M**2))
        # K RT
        U_2 = np.random.uniform(self.Experiment.min_perm_central+0.005e-10,
                                self.Experiment.max_perm_RT-0.005e-10,
                                (self.n_ensemble,self.Experiment.M_RT*2))
        
        if not ensemble_dependence:
        
            # Phi central
            U_3 = np.random.uniform(self.Experiment.min_poro_central+0.0005,
                                    self.Experiment.max_poro_central-0.0005,
                                    (self.n_ensemble,self.Experiment.M**2))
            # Phi RT
            U_4 = np.random.uniform(self.Experiment.min_poro_central+0.0005,
                                    self.Experiment.max_poro_RT-0.0005,
                                    (self.n_ensemble,self.Experiment.M_RT*2))
        else:
            
            # Phi central
            U_3 = np.minimum(np.maximum(permeability_to_porosity(U_1),self.Experiment.min_poro_central+0.0005),
                              self.Experiment.max_poro_central-0.0005)
            # Phi RT
            U_4 = np.minimum(np.maximum(permeability_to_porosity(U_2),self.Experiment.min_poro_central+0.0005),
                              self.Experiment.max_poro_RT-0.0005)


        U = torch.tensor(np.hstack( (U_1,U_2,U_3,U_4) )).float().cuda()
        U = self._ParameteriseXTensor(U,"FWD").T # Parameterise and tensorise
        return U
     
    # Evaluates G_s(u) = F_s(P(\theta)) at selected sensors
    def forward_map(self,U,all_sensor_inds):
        
        # For FWD map, un-parameterise ensemble and move to [0,1]^170 (where model is trained).
        U = self._ParameteriseXTensor(U.T, "BWD")
        U = torch.hstack( (U,self.pressures.cuda(),self.viscosities.cuda()) )
        U = self._UnitTransformXTensor(U, "FWD")
                
        with torch.no_grad():
            y_pred = self.model(U)
        
        # Output of model is log(Y+1000), must convert back to compare with data.
        y_pred = self._ParameteriseYTensor(y_pred,"BWD")
                
        return y_pred[:,all_sensor_inds]
        
    # Runs EKI for data up to and including time t with prior ensemble U
    def run_EKI_t(self,U,i):
        
        # Define data/error covariance
        if self.sequential:
            if i == 0:
                data = self.data[:int(len(self.data)/self.Experiment.times)*self.t[i]]
                all_sensor_inds = self.all_sensor_inds[:int(len(self.data)/self.Experiment.times)*self.t[i]]
                M = len(data)
                Gamma_minus_half = self.Gamma_minus_half[:int(len(self.data)/self.Experiment.times)*self.t[i],
                                                         :int(len(self.data)/self.Experiment.times)*self.t[i]]
            else:
                data = self.data[int(len(self.data)/self.Experiment.times)*self.t[i-1]:int(len(self.data)/self.Experiment.times)*self.t[i]]
                all_sensor_inds = self.all_sensor_inds[int(len(self.data)/self.Experiment.times)*self.t[i-1]:int(len(self.data)/self.Experiment.times)*self.t[i]]
                M = len(data)
                Gamma_minus_half = self.Gamma_minus_half[int(len(self.data)/self.Experiment.times)*self.t[i-1]:int(len(self.data)/self.Experiment.times)*self.t[i],
                                                         int(len(self.data)/self.Experiment.times)*self.t[i-1]:int(len(self.data)/self.Experiment.times)*self.t[i]]
        else:
            data = self.data[:int(len(self.data)/self.Experiment.times)*self.t]
            all_sensor_inds = self.all_sensor_inds[:int(len(self.data)/self.Experiment.times)*self.t]
            M = len(data)
            Gamma_minus_half = self.Gamma_minus_half[:int(len(self.data)/self.Experiment.times)*self.t,
                                                     :int(len(self.data)/self.Experiment.times)*self.t]
  
        
        # Initialise
        t_vec = [torch.tensor(0)]
        Cond = 1
        iterate = -1
        misfit_vec = []
        ensemble_iter = [U]
        
        while Cond == 1 and iterate < self.iter_max:
            
            iterate += 1
            
            # Forward maps
            Gu = self.forward_map(U,all_sensor_inds)
            Gu = Gu.T
        
            # Repeat the data into N_ensemble columns
            data_mat = torch.tile(data,(self.n_ensemble,1)).T.cuda()
            
            # Calculate data misfit (scaled by Gamma inflated)
            Dat_Misfit = torch.matmul(Gamma_minus_half, data_mat - Gu)
            
            # Mean data misfit
            Dat_Misfit_Mean = torch.mean(Dat_Misfit,axis = 1)
            
            misfit_report = (torch.linalg.norm(Dat_Misfit_Mean) ** 2) / M
            misfit_vec.append(misfit_report)
            
            alpha = torch.mean(torch.linalg.norm(Dat_Misfit,axis=0) ** 2) / M
            #alpha = misfit_report
            
            if t_vec[iterate] + 1/alpha > 1:
                alpha = (1/(1-t_vec[iterate])).clone().detach()
                Cond = 0
            
            # Recenter data misfit and ensemble members
            Dat_Misfit_Recentered = Dat_Misfit - Dat_Misfit_Mean[:,np.newaxis]
            U_mean = torch.mean(U,axis=1)
            U_Recentered = U - U_mean[:,np.newaxis]
            
            # Covariance matrices
            C_GG = 1/(self.n_ensemble - 1) * torch.matmul(Dat_Misfit_Recentered, Dat_Misfit_Recentered.T)
            C_uG = 1/(self.n_ensemble - 1) * torch.matmul(U_Recentered, Dat_Misfit_Recentered.T)
            
            # Noise perturbation and covariance update
            RHS = Dat_Misfit + torch.sqrt(alpha) * torch.normal(0,1,(M,self.n_ensemble)).cuda()
            U = U - torch.matmul(C_uG, torch.linalg.solve(C_GG + alpha * torch.eye(M).cuda() , RHS))

            t_vec.append(t_vec[iterate] + 1/alpha)
            ensemble_iter.append(U)
            #print(t[iterate])
              
        return U, t_vec, misfit_vec, ensemble_iter
    
    # Runs EKI based on whether observation time is singular or list
    def run_EKI(self,plotting = 1,ensemble_dep = 0):
        EKI_times = []
        
        # Singular observation time
        if not self.sequential:
            U_prior = self.initialise_ensemble(ensemble_dependence = ensemble_dep)
            start_time = time.time()
            U_posterior = self.run_EKI_t(U_prior,self.t)
            time_diff = time.time()-start_time
            EKI_times.append(time_diff)
            # Convert posterior ensemble to familiar range
            U_plotting = self._ParameteriseXTensor(U_posterior[0].T, "BWD").cpu().numpy() 
            print("t = " + str(self.t) + ": " + "EKI completed in " + str(time_diff) + "s.")
            # Plot (not included in EKI time).
            if plotting:
                self.diagnostic_check(U_plotting)
            return U_plotting, EKI_times, len(U_posterior[2])
        
        # Multiple observation times.
        else:
            U = self.initialise_ensemble(ensemble_dependence = ensemble_dep)
            U_list = []
            U_list.append(U)
            iters_list = []
            # total_time = time.time()
            for i in range(len(self.t)):
                start_time = time.time()
                U_posterior = self.run_EKI_t(U_list[i],i)
                time_diff = time.time()-start_time
                EKI_times.append(time_diff)
                #print("t = " + str(self.t[i]) + ": " + "EKI completed in " + str(time_diff) + "s.")
                U_list.append(U_posterior[0])
                iters_list.append(len(U_posterior[2]))
            # print("Total inversion time: " + str(time.time()-total_time))
            
            # Convert posterior ensemble to familiar range.
            U_plotting = [self._ParameteriseXTensor(U_list[i+1].T, "BWD").cpu().numpy() for i in range(len(self.t))]
            # Plot (not included in EKI time).
            if plotting:
                [self.diagnostic_check(U_plotting[i]) for i in range(len(self.t))]
            return U_plotting, EKI_times,iters_list
    
        
    # Boxplot of posteriors. Better for visualising uncertainty intervals
    def diagnostic_check(self,U):

        halfway = int(len(U[0])/2) # Length 85
        
        plt.figure(figsize=(12,4),dpi=130)
        
        # Posterior boxplot of log(K)
        plt.subplot(2,1,1)
        plt.title("Log-permeability")
        plt.boxplot(np.log(U[:,:halfway]),showfliers=False)
        
        # Plot u_true if known (i.e. if virtual experiment)
        if self.u_true is not None:
            plt.scatter(list(range(1,halfway+1)),np.log(self.u_true[:halfway]),s=14,color="red")
        plt.ylim([-22.5,np.log(self.Experiment.max_perm_RT)])
        plt.xticks(rotation = 90)
        
        # Posterior boxplot of \phi
        plt.subplot(2,1,2)
        plt.title("Porosity")
        plt.boxplot(U[:,halfway:],showfliers=False)
        
        # Plot u_true if known (i.e. if virtual experiment)
        if self.u_true is not None:
            plt.scatter(list(range(1,halfway+1)),self.u_true[halfway:],s=14,color="red")
        plt.ylim([0,1])
        plt.xticks(rotation = 90)
        
        plt.tight_layout()
        plt.show()
    
        
        
    
    