# -*- coding: utf-8 -*-
"""
Created on Mon Nov 13 11:21:16 2023

@author: pmymc12
"""

import numpy as np
import scipy.io as io

class Experiment:
    
    """
    Description
    -----------
    The Experiment class holds details regarding the experimental setup used.
    Perhaps unneccessary to write this as a class rather than a dictionary.

    Parameters
    ----------
    Lx, Ly : float
        Dimensions of the preform.
    p_I : list
        Range of inlet pressure used.
    p_O : int
        Outlet pressure used.
    darcy_thickness : float
        Thickness of preform (not relevant in 2D).
    mu : list
        Range of resin viscosities used.
    observation_times : list
        Times at which pressure measurements are acquired.
    min_perm_central, max_perm_central : float
        Bounds of the parameterisation for permeability in central regions.
    min_perm_RT, max_perm_RT : float
        Bounds of the parameterisation for permeability in racetracking regions.
    min_poro_central, max_poro_central : float
        Bounds of the parameterisation for porosity in central regions.
    min_poro_RT, max_poro_RT : float
        Bounds of the parameterisation for porosity in racetracking regions.
    Nx : int
        Mesh refinement used for Matlab simulations.
    M : int
        Central zone is partitioned into MxM regions.
    M_RT : int
        Racetracking zone is partioned into M_RT zones on top and bottom
    sigma1 : float
        Scales size of noise (prop. to G(u)) added to virtual experiments.
    sigma2 : float
        Scales size of noise (prop. to max G(u)) added to virtual experiments.

    Returns
    -------
    No returns

    Examples
    --------
    >>> Experiment_object = Experiment()
    """
    
    def __init__(self,
                 Lx = 0.3, Ly = 0.3,
                 p_I = [90_000, 110_000], p_O = 0,
                 darcy_thickness = 0.001, mu = [0.09, 0.11], 
                 observation_times = [1,3,5,7,10,15,20,25,30,35,40,45,50,55],
                 min_perm_central = 2.0 * 10**(-10), max_perm_central = 7 * 10**(-10),
                 min_perm_RT = 2.0 * 10**(-10), max_perm_RT = 500 * 10**(-10),
                 min_poro_central = 0.4, max_poro_central = 0.85,
                 min_poro_RT = 0.4, max_poro_RT = 0.9882,
                 Nx = 64, M = 9, M_RT = 2,
                 sigma1 = 0.00, sigma2 = 0.01):
        
        # Various sets
        assert Lx > 0, "Lx must be greater than 0"
        assert Ly > 0, "Ly must be greater than 0"
        self.Lx = Lx
        self.Ly = Ly
        
        assert len(p_I) == 2, "p_I must have a min and max value"
        assert p_I[0] > p_O, "p_I must be greater than p_O"
        self.p_I = p_I
        self.p_O = p_O
        
        self.darcy_thickness = darcy_thickness
        
        assert len(mu) == 2, "\mu must have a min and max value"
        self.mu = mu
    
        self.observation_times = observation_times
        self.times = len(observation_times)
        
        assert min_perm_central < max_perm_central, "Max K must be greater than min K"
        self.min_perm_central = min_perm_central
        self.max_perm_central = max_perm_central
        
        assert min_perm_RT < max_perm_RT, "Max K must be greater than min K"
        self.min_perm_RT = min_perm_RT
        self.max_perm_RT = max_perm_RT
        
        assert min_poro_central < max_poro_central, "Max \phi must be greater than min \phi"
        self.min_poro_central = min_poro_central
        self.max_poro_central = max_poro_central
        
        assert min_poro_RT < max_poro_RT, "Max \phi must be greater than min \phi"
        self.min_poro_RT = min_poro_RT
        self.max_poro_RT = max_poro_RT
        
        assert Nx > 0, "Nx must be positive"
        self.Nx = Nx
        
        assert M > 0, "M must be positive"
        self.M = M
        
        assert M_RT > 0, "M_RT must be positive"
        self.M_RT = M_RT
        
        assert sigma1 >= 0, "sigma1 must be non-negative"
        self.sigma1 = sigma1
        
        assert sigma2 >= 0, "sigma2 must be non-negative"
        self.sigma2 = sigma2
        
        # Max and min vectors (used in other classes)
        self.param_min = np.concatenate( (np.ones(self.M**2)*self.min_perm_central,
                                          np.ones(self.M_RT*2)*self.min_perm_RT,
                                          np.ones(self.M**2)*self.min_poro_central,
                                          np.ones(self.M_RT*2)*self.min_poro_RT,
                                          np.array([self.p_I[0]]),
                                          np.array([self.mu[0]])
                                          ) )
        self.param_max = np.concatenate( (np.ones(self.M**2)*self.max_perm_central,
                                          np.ones(self.M_RT*2)*self.max_perm_RT,
                                          np.ones(self.M**2)*self.max_poro_central,
                                          np.ones(self.M_RT*2)*self.max_poro_RT,
                                          np.array([self.p_I[1]]),
                                          np.array([self.mu[1]])
                                          ) )
        
        print("Parameters set.")
        

        # Import sensor locations
        self.all_sensor_locs = io.loadmat('Data\sensor_locs_9x9.mat')['all_sensor_locs']
        self.all_sensor_locs_mesh = io.loadmat('Data\sensor_locs_mesh_9x9.mat')['all_sensor_locs_mesh']
        self.exp_sensor_locs = self.all_sensor_locs[0:23]
        self.exp_sensor_locs_mesh = self.all_sensor_locs_mesh[0:23]
        self.sensors_per_time = len(self.all_sensor_locs)
        self.total_sensors = self.sensors_per_time*len(observation_times)
        
        # Generate dictionary for each sensor configuration
        file_names = ["all_sensors","half_sensors",
                      "two_by_two","three_by_three",
                      "four_by_four","five_by_five"]
        dict_names = ["All","Half",
                      "2x2","3x3",
                      "4x4","5x5"]
        sensor_dict = {}
        k=0
        for name in file_names:
            sensor_dict[dict_names[k]] = []
            with open("SensorLocs/" + name + ".txt", "r") as f:
              for line in f:
                sensor_dict[dict_names[k]].append(int(line.strip()))
            k += 1
        self.sensor_dict = sensor_dict
        
        print("Sensor locations loaded.")
    
            
     

   