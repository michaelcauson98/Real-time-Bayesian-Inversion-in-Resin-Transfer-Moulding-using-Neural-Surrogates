# -*- coding: utf-8 -*-
"""
Created on Thu Dec  7 08:37:40 2023

@author: pmymc12
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import scipy
from matplotlib import cm
import matplotlib.colors as mcolors

class nlcmap(LinearSegmentedColormap):
    """A nonlinear colormap"""
    
    name = 'nlcmap'
    
    def __init__(self, cmap, levels):
        self.cmap = cmap
        # @MRR: Need to add N for backend
        self.N = cmap.N
        self.monochrome = self.cmap.monochrome
        self.levels = np.asarray(levels, dtype='float64')
        self._x = self.levels / self.levels.max()
        self._y = np.linspace(0.0, 1.0, len(self.levels))
    
    #@MRR Need to add **kw for 'bytes'
    def __call__(self, xi, alpha=1.0, **kw):
        """docstring for fname"""
        # @MRR: Appears broken? 
        # It appears something's wrong with the
        # dimensionality of a calculation intermediate
        #yi = stineman_interp(xi, self._x, self._y)
        yi = np.interp(xi, self._x, self._y)
        return self.cmap(yi, alpha)

class PlotEKI:
    
    def __init__(self,posterior_ensemble,EKI):
        
        self.posterior_ensemble = posterior_ensemble[0]
        self.inversion_times = posterior_ensemble[1]
        self.EKI = EKI
    
    def plot_permeability(self,K,v_min,v_max,sensor_locs,virtual, cbar = True,c_map="jet",rm_axis="False",plt_sensors=False,axis=None,flipud=1):
        
        Lx = self.EKI.Experiment.Lx
        Ly = self.EKI.Experiment.Ly
        
        if virtual:
            
            M = self.EKI.Experiment.M
            M_RT = self.EKI.Experiment.M_RT
            
            central_patch = np.flipud(np.rot90(np.reshape(K[0:M**2],(M,M)),1))
            RT_patch = np.reshape(K[M**2:],(2,M_RT) ).T
                    
            # Various definitions
            Nx = 300
            Ny = 300
            x = np.linspace(0,self.EKI.Experiment.Lx,Nx)
            y = x
    
            #Initialise
            K_mat = np.zeros( (Nx,Ny) )
            
            # Find the (x,y) coordinates of the region boundaries
            coarse_grid_exact_locs_x = round(Nx/M)
            coarse_grid_boundaries_x = np.concatenate( ([0],np.array(list(range(1,M)))*coarse_grid_exact_locs_x,[Nx]) )
        
            q = np.where(abs(y-0.002) == min(abs(y-0.002)))[0][0]+5
            coarse_grid_exact_locs_y = round((Nx-2*q)/M)
            coarse_grid_boundaries_y = np.concatenate( ([q],q + np.array(list(range(1,M)))*coarse_grid_exact_locs_y,[Nx-q]) )
            
            course_x_RT = np.concatenate( ([0],np.array(list(range(1,M_RT)))*round(Nx/M_RT),[Nx]) )

            # Fill the central zones
            for RegX in range(M):
                for RegY in range(M):
                    K_mat[coarse_grid_boundaries_y[RegY]:coarse_grid_boundaries_y[RegY+1],
                          coarse_grid_boundaries_x[RegX]:coarse_grid_boundaries_x[RegX+1]] = central_patch[RegX,RegY]
        
                
            # Fill RT zones
            for j in range(M_RT):
                K_mat[0:q+3, course_x_RT[j]:course_x_RT[j+1]] = RT_patch[0,j]
                K_mat[(Nx-q-2):, course_x_RT[j]:course_x_RT[j+1]] = RT_patch[1,j]
                
                
            X, Y = np.meshgrid(np.linspace(0,self.EKI.Experiment.Lx, Nx),
                               np.linspace(0,self.EKI.Experiment.Ly, Ny))
        
        else:
            K_mat = np.reshape(K,(64,64))
            X, Y = np.meshgrid(np.linspace(0,self.EKI.Experiment.Lx, 64),
                               np.linspace(0,self.EKI.Experiment.Ly, 64))
            
            f = scipy.interpolate.NearestNDInterpolator(list(zip(np.reshape(X,4096),np.reshape(Y,4096))),K)

            x = np.linspace(0,self.EKI.Experiment.Lx, 300)
            y = np.linspace(0,self.EKI.Experiment.Lx, 300)
            X, Y = np.meshgrid(x, y)
            K_mat = f(X,Y)
            
            if flipud:
                K_mat = np.flipud(K_mat)
        
            
        
        # Add plot or subplot functionality
        if axis == None:
            
            im = plt.pcolor(X,Y,K_mat,shading='auto',cmap=c_map,vmin=v_min, vmax=v_max)

            if cbar:
                plt.colorbar(im,fraction=0.046, pad=0.04)
                
            plt.xlim([0,Lx])
            plt.ylim([0,Ly])
            
            if plt_sensors:
                plt.scatter(self.EKI.Experiment.all_sensor_locs[sensor_locs,0],
                            self.EKI.Experiment.all_sensor_locs[sensor_locs,1],
                            c='white',edgecolors='black',s=20)
            
            if rm_axis:
                plt.xticks([])
                plt.yticks([])
            else:
                plt.xticks([0,0.15,0.3])
                plt.yticks([0,0.15,0.3])
            
        else:
            
            im = axis.pcolor(X,Y,K_mat,shading='auto',cmap=c_map,vmin=v_min, vmax=v_max)
                
            axis.set_xlim([0,Lx])
            axis.set_ylim([0,Ly])
            
            if plt_sensors:
                axis.scatter(self.EKI.Experiment.all_sensor_locs[sensor_locs,0],
                             self.EKI.Experiment.all_sensor_locs[sensor_locs,1],
                             c='white',edgecolors='black',s=20)
            
            if rm_axis:
                axis.set_xticks([])
                axis.set_yticks([])
            else:
                axis.set_xticks([0,0.15,0.3])
                axis.set_yticks([0,0.15,0.3])
                
            axis.set_aspect('equal')
            
        return im
    
    def plot_sequential(self,permeability_only = 0,log_perm=True,sensor_locs=list(range(23)),
                        plot_prob = 0,defect_type="low",defect_tresholds=[3e-10,0.6],
                        images_available = 0, images = None,
                        truth_available = 0, truth = 0,
                        front_available = 0, front = 0,
                        plot_circle=0,circle_centre=[0.15,0.15],radius=0.04,rm_first_col=0):
        
        perm_map = "turbo"
        poro_map = "jet"
        perm_std_map = "binary"
        poro_std_map = "binary"
        prob_map = "Reds"
        #colors2 = cm.Reds.reversed()(np.linspace(0., 1, 181))
        #colors1 = cm.jet(np.linspace(0, 1, 75))

        # combine them and build a new colormap
        #colors = np.vstack((colors1, colors2))
        #perm_map = mcolors.LinearSegmentedColormap.from_list('my_colormap', colors)
        perm_map = nlcmap(cm.turbo, [0, 5, 10, 15, 20, 40, 70, 100])
        
        
        if log_perm:
            ensemble_K = [np.log(self.posterior_ensemble[i][:,:85]) for i in range(len(self.posterior_ensemble))]
            K_min = np.log(self.EKI.Experiment.min_perm_central)
            K_max = np.log(self.EKI.Experiment.max_perm_RT)
            #K_min = np.min([np.mean(ensemble_K[i],axis=0) for i in range(len(ensemble_K))])
            #K_max = np.max([np.mean(ensemble_K[i],axis=0) for i in range(len(ensemble_K))])
            if truth_available:
                hw = int(len(truth)/2)
                K_true = np.log(truth[:hw])
            
        else:
            ensemble_K = [self.posterior_ensemble[i][:,:85] for i in range(len(self.posterior_ensemble))]
            K_min = self.EKI.Experiment.min_perm_central
            K_max = self.EKI.Experiment.max_perm_RT
            #K_min = np.min([np.mean(ensemble_K[i],axis=0) for i in range(len(ensemble_K))])
            #K_max = np.max([np.mean(ensemble_K[i],axis=0) for i in range(len(ensemble_K))])
            if truth_available:
                hw = int(len(truth)/2)
                K_true = truth[:hw]
            
        ensemble_phi = [self.posterior_ensemble[i][:,85:] for i in range(len(self.posterior_ensemble))]
        if truth_available:
            phi_true = truth[hw:]
        
        Lx = self.EKI.Experiment.Lx
        Ly = self.EKI.Experiment.Ly
        x = np.linspace(0.0,Lx,50)
        y = np.linspace(0.0,Ly,50)  
        a, b = np.meshgrid( x , y )
        C = (a-circle_centre[0]) ** 2 + (b-circle_centre[1]) ** 2 - radius**2
        circle_cmap = "binary"
        
        columns = len(self.EKI.t) + truth_available
        rows = 2 * (2-permeability_only) + images_available + plot_prob
        
        figsizex = 23
        figsizey = figsizex * (rows/columns)
        font = 18
        
        fig,axes = plt.subplots(rows,columns,figsize=(figsizex,figsizey),constrained_layout=True)
        
        # Plot resin flow images if available
        if images_available:
            [axes[0,i+truth_available].imshow(np.flipud(images[i]),extent=[0,Lx,0,Ly],cmap="bwr") for i in range(len(images))]
            [axes[0,i+truth_available].scatter(self.EKI.Experiment.all_sensor_locs[sensor_locs,0],self.EKI.Experiment.all_sensor_locs[sensor_locs,1],s=font,color="white",edgecolor="black") for i in range(len(images))]
            
        K_std_max = np.max(np.std(ensemble_K[0],axis=0))
        phi_std_max = np.max(np.std(ensemble_phi[0],axis=0))
        
        
        # Plot mean permeability
        for i in range(columns - truth_available):
            
            im1 = self.plot_permeability(np.mean(ensemble_K[i],axis=0),K_min,K_max,
                                   virtual=1,sensor_locs=0,cbar = False,c_map=perm_map,rm_axis="False",plt_sensors=False,
                                   axis=axes[images_available,i+truth_available])
            im2 = self.plot_permeability(np.std(ensemble_K[i],axis=0),0,K_std_max,
                                   virtual=1,sensor_locs=0,cbar = False,c_map=perm_std_map,rm_axis="False",plt_sensors=False,
                                   axis=axes[images_available+1,i+truth_available])
            
        
        if not plot_prob and plot_circle:
            [axes[images_available,i+truth_available].contour( a , b , C , [0] ,cmap=circle_cmap,linewidths=4,linestyles="--")  for i in range(columns - truth_available)]

        axes[images_available,truth_available].set_ylabel([r"$\mu_K$",r"$\mu_{\log K}$"][log_perm],fontsize=font)
        axes[images_available+1,truth_available].set_ylabel([r"$\sigma_K$",r"$\sigma_{\log K}$"][log_perm],fontsize=font)
        
        # If porosity required, plot mean/std porosity
        if not permeability_only:
            
            for i in range(columns - truth_available):
                im3 = self.plot_permeability(np.mean(ensemble_phi[i],axis=0),self.EKI.Experiment.min_poro_central,self.EKI.Experiment.max_poro_RT,
                                       virtual=1,sensor_locs=0,cbar = False,c_map=poro_map,rm_axis="False",plt_sensors=False,
                                       axis=axes[images_available+2,i+truth_available])
                im4 = self.plot_permeability(np.std(ensemble_phi[i],axis=0),0,phi_std_max,
                                       virtual=1,sensor_locs=0,cbar = False,c_map=poro_std_map,rm_axis="False",plt_sensors=False,
                                       axis=axes[images_available+3,i+truth_available])
            if not plot_prob and plot_circle:
                [axes[images_available+2,i+truth_available].contour( a , b , C , [0] ,cmap="gray",linewidths=4,linestyles="-")  for i in range(columns - truth_available)]
                [axes[images_available+2,i+truth_available].contour( a , b , C , [0] ,cmap="binary",linewidths=2,linestyles="-")  for i in range(columns - truth_available)]

                
            axes[images_available+2,truth_available].set_ylabel(r"$\mu_{\phi}$",fontsize=font)
            axes[images_available+3,truth_available].set_ylabel(r"$\sigma_{\phi}$",fontsize=font)
        
        
        # If probability required, plot (also for porosity if required)
        if plot_prob:
            
            for i in range(columns - truth_available):
                im5 = self.plot_permeability(self.defect_probability(i,defect_type,defect_tresholds,rm_first_col),0,1,
                                       sensor_locs=0,virtual=1,cbar = False,c_map=prob_map,rm_axis="False",plt_sensors=False,
                                       axis=axes[-1,i+truth_available])
            if plot_circle:
                [axes[-1,i+truth_available].contour( a , b , C , [0] ,cmap="gray",linewidths=4,linestyles="-")  for i in range(columns - truth_available)]
                [axes[-1,i+truth_available].contour( a , b , C , [0] ,cmap="binary",linewidths=2,linestyles="-")  for i in range(columns - truth_available)]
            axes[-1,truth_available].set_ylabel(r"Defect prob.",fontsize=font)


        # If virtual, u_true is known - plot it and remove empty axis
        if truth_available:
            self.plot_permeability(K_true,K_min,K_max,
                                   virtual=(hw<64**2),sensor_locs=0,cbar = False,c_map=perm_map,rm_axis="False",plt_sensors=False,
                                   axis=axes[images_available,0])
            axes[images_available,0].set_title([r"$K^{\dagger}$",r"$\log K^{\dagger}$"][log_perm],fontsize=font)
            axes[images_available+1,0].axis('off')
            if not permeability_only:
                self.plot_permeability(phi_true,self.EKI.Experiment.min_poro_central,self.EKI.Experiment.max_poro_RT,
                                       virtual=(hw<64**2),sensor_locs=0,cbar = False,c_map=poro_map,rm_axis="False",plt_sensors=False,
                                       axis=axes[images_available+2,0])
                axes[images_available+2,0].set_title(r"$\phi^{\dagger}$",fontsize=font)
                axes[images_available+3,0].axis('off')
                
            if images_available:
                axes[0,0].axis('off')
            
            if plot_prob:
                axes[-1,0].axis('off')
                
        if front_available:
            print("plotting front")
            for i in range(images_available,rows):
                for j in range(len(front)):
                    axes[i,truth_available+j].plot(front[j][:,0],front[j][:,1],
                                                   color="black",linewidth=4)
                    axes[i,truth_available+j].plot(front[j][:,0],front[j][:,1],
                                                   color="white",linewidth=2)
        

        # Reduce number of ticks for all plots
        [axes[i,j].set_xticks([0,0.1,0.2,0.3]) for i in range(rows) for j in range(columns)]
        [axes[i,j].set_yticks([0,0.1,0.2,0.3]) for i in range(rows) for j in range(columns)]
        [axes[0,i+truth_available].set_title(r"$t_{{{}}}$".format(self.EKI.t[i]) + " = " + str(self.EKI.Experiment.observation_times[self.EKI.t[i]-1]) + ", " + r"$t_{{{}}}^{{({})}} = $".format("c",self.EKI.t[i]) + str(round(self.inversion_times[i],2)),fontsize=font) for i in range(len(self.EKI.t))]
        
        fig.colorbar(im1, ax=axes[images_available,:],fraction=0.006, pad=0.005)
        fig.colorbar(im2, ax=axes[images_available+1,:],fraction=0.006, pad=0.005)
        
        if not permeability_only:
            fig.colorbar(im3, ax=axes[images_available+2,:],fraction=0.006, pad=0.005)
            fig.colorbar(im4, ax=axes[images_available+3,:],fraction=0.006, pad=0.005)
        if plot_prob:
            fig.colorbar(im5, ax=axes[-1,:],fraction=0.006, pad=0.005) 
            
        # line = plt.Line2D((0,1),(.4975,.4975), color="k", linewidth=2,linestyle="--")
        # fig.add_artist(line)
        line = plt.Line2D((0,1),(.495,.495), color="k", linewidth=2,linestyle="--")
        fig.add_artist(line)
        line = plt.Line2D((0,1),(.825,.825), color="k", linewidth=2,linestyle="--")
        fig.add_artist(line)
        line = plt.Line2D((0,1),(.1675,.1675), color="k", linewidth=2,linestyle="--")
        fig.add_artist(line)
        # line = plt.Line2D((0,1),(.3925,.3925), color="k", linewidth=2,linestyle="--")
        # fig.add_artist(line)
        # line = plt.Line2D((0.11875,0.11875),(0,1), color="k", linewidth=2,linestyle="--")
        fig.add_artist(line)

    
    def defect_probability(self,t,def_type,defect_tresholds):
        
        # Thresholds: [K_low,K_high,phi_low,phi_high]
        ensemble_t = self.posterior_ensemble[t]
        
        prop_mat = np.zeros(85)
        
        if def_type == "low":
            prop_mat[:self.EKI.Experiment.M**2] = np.mean(np.logical_and(ensemble_t[:,:self.EKI.Experiment.M**2] < defect_tresholds[0],
                                                                        ensemble_t[:,(self.EKI.Experiment.M**2+2*self.EKI.Experiment.M_RT):(2*self.EKI.Experiment.M**2+2*self.EKI.Experiment.M_RT)] < defect_tresholds[2]),axis=0)
            prop_mat[self.EKI.Experiment.M**2:] = np.mean(np.logical_and(ensemble_t[:,self.EKI.Experiment.M**2:(self.EKI.Experiment.M**2+2*self.EKI.Experiment.M_RT)] > 45e-10,
                                                                        ensemble_t[:,(2*self.EKI.Experiment.M**2+2*self.EKI.Experiment.M_RT):] > 0.934),axis=0)

        elif def_type == "both":
            prop_mat[:self.EKI.Experiment.M**2] = np.mean(np.logical_and((ensemble_t[:,:self.EKI.Experiment.M**2] < defect_tresholds[0]) + (self.posterior_ensemble[t][:,:self.EKI.Experiment.M**2] > defect_tresholds[1]),
                                                                        (ensemble_t[:,(self.EKI.Experiment.M**2+2*self.EKI.Experiment.M_RT):(2*self.EKI.Experiment.M**2+2*self.EKI.Experiment.M_RT)] < defect_tresholds[2]) + (self.posterior_ensemble[t][:,(self.EKI.Experiment.M**2+2*self.EKI.Experiment.M_RT):(2*self.EKI.Experiment.M**2+2*self.EKI.Experiment.M_RT)] > defect_tresholds[3])),
                                                          axis=0)
            prop_mat[self.EKI.Experiment.M**2:] = np.mean(np.logical_and(ensemble_t[:,self.EKI.Experiment.M**2:(self.EKI.Experiment.M**2+2*self.EKI.Experiment.M_RT)] > 45e-10,
                                                                        ensemble_t[:,(2*self.EKI.Experiment.M**2+2*self.EKI.Experiment.M_RT):] > 0.934),
                                                          axis=0)
        else:
            prop_mat[:self.EKI.Experiment.M**2] = np.mean(np.logical_and(ensemble_t[:,:self.EKI.Experiment.M**2] > defect_tresholds[1],
                                                                        ensemble_t[:,(self.EKI.Experiment.M**2+2*self.EKI.Experiment.M_RT):(2*self.EKI.Experiment.M**2+2*self.EKI.Experiment.M_RT)] > defect_tresholds[3]),
                                                          axis=0)
            prop_mat[self.EKI.Experiment.M**2:] = np.mean(np.logical_and(ensemble_t[:,self.EKI.Experiment.M**2:(self.EKI.Experiment.M**2+2*self.EKI.Experiment.M_RT)] > 45e-10,
                                                                        ensemble_t[:,(2*self.EKI.Experiment.M**2+2*self.EKI.Experiment.M_RT):] > 0.934),
                                                          axis=0)
        
        
        return prop_mat
    
    
    
    
    
    