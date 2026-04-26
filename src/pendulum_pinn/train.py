"""
---------------------------------
BY : Haoyu Tang
Github : Jerry_Haoyu 
---------------------------------
"""
import torch
from torch.utils.data import DataLoader
import torch.nn as nn

import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt

import os
import tqdm 
import time

from src.pendulum_pinn.dataset import pendulumPINNDataSet
from src.pendulum_pinn.pinn import pendulumPINN

def print_check(msgs : list[str]):
    print(50*"=")
    for msg in msgs:
        print(msg)
        print()
    print(50*"=")

class PendulumParametrizer:
    def __init__(self, 
                 data_path,
                 output_dir,
                 num_data,
                 epochs
        ):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print_check([f"Using device: {self.device}"])
        if self.device.type == 'cuda':
            print_check([f"GPU: {torch.cuda.get_device_name(0)}"])
            
        self.model = pendulumPINN().to(self.device)
        
        num_params = sum(p.numel() for p in self.model.parameters())
        print_check([f"Model has {num_params} paramters"])
        
        self.num_data = num_data
        self.data_path = data_path 
        
        # setting up the optimizer and scheduler
        self.epochs = epochs 
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=1e-3, weight_decay=0.01)
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=self.epochs)
        
        # making some directories
        os.makedirs(output_dir, exist_ok=True)
        self.out_dir = output_dir
        self.parametrized_func_vis_dir = os.path.join(self.out_dir, "paramterized_functions_visuliazation") 
        os.makedirs(self.parametrized_func_vis_dir, exist_ok=True)
        
        # call creater loader to declaure the loaders
        self._create_loader()
        # save the loader visualization to output
        self.plot_loader()
        
        
    def _create_loader(self, train_ratio=0.8):
        full_dataset = pendulumPINNDataSet(npy_data = self.data_path, num_data=self.num_data)
        
        train_size = int(train_ratio * len(full_dataset))
        test_size = len(full_dataset) - train_size
        
        train_dataset, test_dataset = torch.utils.data.random_split(full_dataset, [train_size, test_size])
        
        train_loader = DataLoader(train_dataset, shuffle=True)
        test_loader = DataLoader(test_dataset, shuffle=False)
        
        # store the entire time domain with a coarsen factor of 5
        self.time_domain : npt.NDArray = full_dataset.all_data[::5, 0]
        
        self.train_loader = train_loader
        self.test_loader = test_loader
        
        print(f"Training samples: {len(self.train_loader):,}")
        print(f"Test samples:     {len(self.test_loader):,}")

    # for inspecting what the data acutally look like
    def plot_loader(self):
        fig, ax = plt.subplots(1, 2, figsize=(10,3))
        for t, x1, x2, _ in self.train_loader:
            ax[0].plot(t.item(), x1.item(), marker='o', color='r', markersize=2)
            ax[0].plot(t.item(), x2.item(), marker='o', color='g', markersize=2)
            
        for t, x1, x2, _ in self.test_loader:
            ax[1].plot(t.item(), x1.item(), marker='o', color='r', markersize=2)
            ax[1].plot(t.item(), x2.item(), marker='o', color='g', markersize=2)

        fig.savefig(os.path.join(self.out_dir, "data_visulization"))
            
    def _train_one_epoch(self, update_global_weights=False):
        self.model.train() 
        total_loss : float = 0.0
        for t, x1, x2, _ in self.train_loader:
            t, x1_true, x2_true = t.to(self.device), x1.to(self.device), x2.to(self.device)
            
            loss = self.model.get_loss(t, x1_true, x2_true, update_global_weights=update_global_weights)
            
            total_loss += loss.item()
            
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
        
        average_loss = total_loss / len(self.train_loader)
        
        return average_loss
        
    def _evaluate(self):
        self.model.eval() 
        total_error : float = 0.0
        with torch.no_grad():
            for t, x1, x2, _ in self.test_loader:
                t, x1_true, x2_true = t.to(self.device), x1.to(self.device), x2.to(self.device)
                result = self.model.forward(t)
                x1, x2 = result['x1'], result['x2']
                total_error += (x1_true - x1) ** 2 + (x2_true - x2) ** 2
        
        average_error = total_error / len(self.test_loader)
        
        return average_error
    
    def _evaluate_parametrized_functions(self):
        l = np.empty_like(self.time_domain)
        x1 = np.empty_like(self.time_domain)
        x2 = np.empty_like(self.time_domain)
        for i, t in enumerate(self.time_domain):
            with torch.no_grad():
                result = self.model.forward(torch.tensor([t], requires_grad=False, dtype=torch.float32).to(self.device))
                l[i] = result['l'].item()
                x1[i] = result['x1'].item()
                x2[i] = result['x2'].item()
        
        return x1, x2, l
    
    def _plot_pendulum(self, x1, x2, l, epoch):
        fig, ax = plt.subplots(2, 2)
        ax[0,0].set_xlabel("time")
        ax[0,0].set_ylabel(r"$\theta$")
        ax[0,1].set_xlabel("time")
        ax[0,1].set_ylabel(r"$\theta'$")
        ax[1,0].set_xlabel("time")
        ax[1,0].set_ylabel(r"$L$")
        
        ax[0,0].plot(self.time_domain, x1)
        ax[0,1].plot(self.time_domain, x2)
        ax[1,0].plot(self.time_domain, l)
        
        ax[1,1].axis('off')
        
        plt.subplots_adjust(wspace=0.5, hspace=0.5)
        
        fig.savefig(os.path.join(self.parametrized_func_vis_dir, f"epoch_{epoch}"))
        
    def train(self):
        total_start = time.time()
        best_error = float('inf')
        for epoch in range(self.epochs):
            epoch_start = time.time()

            train_loss = self._train_one_epoch(update_global_weights=(epoch % 1000 == 0))

            test_error = self._evaluate()

            epoch_time = time.time() - epoch_start
            
            lr = self.optimizer.param_groups[0]['lr']
            
            print(f"Epoch {epoch+1:3d}/{self.epochs} | "
              f"Train Loss: {float(train_loss):.3f}  | "
              f"Test Loss: {float(test_error):.3f} | "
              f"LR: {lr:.2e} | Time: {epoch_time:.1f}s")
            
            if test_error < best_error:
                best_error = best_error
                torch.save(self.model, os.path.join(self.out_dir, 'pendulum_pinn_model.pt'))
            
            if epoch % 1000 == 0:
                x1, x2, l = self._evaluate_parametrized_functions()
                self._plot_pendulum(x1, x2, l, epoch)
        
        total_time = time.time() - total_start
        print_check([f"\nTotal training time: {total_time / 60:.1f} minutes"])
    