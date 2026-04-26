"""
---------------------------------
BY : Haoyu Tang
Github : Jerry_Haoyu 
---------------------------------
"""

import numpy as np
import matplotlib.pyplot as plt
import tqdm 
import os
import numpy.typing as npt
from collections.abc import Callable

class pendulumODE:
    def __init__(self, 
                 out_dir : str, 
                 L : Callable[[npt.NDArray[np.float64]], npt.NDArray[np.float64]],
                 dt = 1e-2,
                 T = 10
        ):
        """
        Solve pendulumn equation by simple Heun's method, i.e., second order Runge-Kutta
        Note this is an explict method by resolving implicty through a euler guess, i.e., 
        predictor-corrector method. See README.md for more details 
        
        Args:
            out_dir(str): the out directory 
            L(Callable): L(t), note this must be numpy compatible 
            dt : the step size in foward step
            T : total simulation time 
        """
        os.makedirs(out_dir, exist_ok=True)
        self.out_dir = out_dir
        
        self.total_steps = int(T / dt)
        
        self.L = L  # arm length, callalbe 
        
        self.x1s = np.zeros(self.total_steps) # history of displacement 
        self.x2s = np.zeros(self.total_steps) # history of velocity
        
        # initila condition
        self.x1s[0] = np.pi/2
        self.x2s[0] = 0.0
        
        self.i = 1
        self.T = T
        self.dt = dt
        self.ts = np.linspace(0.0, float(self.T), int(T / dt), dtype=float)
        self.Ls = self.L(self.ts)
    
    def _rhs(self, x1, x2):
        return x2, - 9.8 / self.Ls[self.i] * np.sin(x1)
    
    def _get_euler_guess(self):
        x1_curr,  x2_curr = self.x1s[self.i-1], self.x2s[self.i-1]
        dx1, dx2 = self._rhs(x1_curr, x2_curr)
        return x1_curr + dx1 * self.dt, x2_curr + dx2 * self.dt
    
    def _plot_and_save(self):
        fig, ax = plt.subplots(2, 2)
        ax[0,0].set_xlabel("time")
        ax[0,0].set_ylabel(r"$\theta$")
        ax[0,1].set_xlabel("time")
        ax[0,1].set_ylabel(r"$\theta'$")
        ax[1,0].set_xlabel("time")
        ax[1,0].set_ylabel(r"$L$")
        
        ax[0,0].plot(self.ts, self.x1s)
        ax[0,1].plot(self.ts, self.x2s)
        ax[1,0].plot(self.ts, self.Ls)
        
        ax[1,1].axis('off')
        
        plt.subplots_adjust(wspace=0.5, hspace=0.5)
        
        
        fig.savefig(os.path.join(self.out_dir, "time_series"))
    
    def simulate(self, save_fig=True, save_data=True):
        pbar = tqdm.tqdm(total=self.total_steps, desc="Pendulum_ODE Simulating")
        
        while self.i < self.total_steps:
            pbar.update(1)
            x1_curr,  x2_curr = self.x1s[self.i-1], self.x2s[self.i-1]
            x1_eg, x2_eg = self._get_euler_guess()
            dx1_pred, dx2_pred =  self._rhs(x1=x1_eg, x2=x2_eg) 
            dx1_corr, dx2_corr = self._rhs(x1=x1_curr, x2=x2_curr) 
            dx1 = 0.5 * (dx1_pred + dx1_corr)
            dx2 = 0.5 * (dx2_pred + dx2_corr)
            self.x1s[self.i] = x1_curr + dx1 * self.dt
            self.x2s[self.i] = x2_curr + dx2 * self.dt
            self.i += 1

        
        data = np.column_stack([self.ts, self.x1s, self.x2s, self.Ls])
        
        if save_data :
            np.save(os.path.join(self.out_dir, "data.npy"), data)
            
        if save_fig :
            self._plot_and_save()
        