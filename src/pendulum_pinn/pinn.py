"""
---------------------------------
BY : Haoyu Tang
Github : Jerry_Haoyu 
---------------------------------
"""
import torch
import torch.nn as nn


class pendulumPINN(nn.Module):
    """
    A set of two FCN to learn the arm length L(t) function 
    """
    def __init__(self, hidden_layers = {'parameter' : 1, 'oscillator' : 3}):
        super().__init__()
        
        # The first FCN paramterize L(t)
        self.parameter_net = nn.Sequential(
            nn.Linear(1, 32),
            self.__make_hidden__(hidden_layers['parameter']),
            nn.Linear(32, 1)
        )
        
        # The second FCN paramterize P(D,L)x = 0
        self.oscillator_net = nn.Sequential(
            nn.Linear(2, 32),
            self.__make_hidden__(hidden_layers['oscillator']),
            nn.Linear(32, 2)
        )
        
        self.mse = nn.MSELoss()
        self.lambda1 = 1.0
        self.lambda2 = 1.0
        
    def __make_hidden__(self,num_hidden) -> nn.Module:
        hidden = nn.Sequential(
            *[
                nn.Sequential(*[nn.Linear(32, 32), nn.Tanh()])
                for _ in range(num_hidden)
            ]
        )
        return hidden

    def forward(self, t):
        """
        Returns:
            l : l(t)
            x1 : x1(t), i.e. theta(t)
            x2 : x2(t), i.e. theta_prime(t)
        """
        l = self.parameter_net(t)
        input = torch.concat([l,t])
        x = self.oscillator_net(input)
        return {'l' : l, 'x1' : x[0:1], 'x2' : x[1:2]}
    
    def get_loss(self, 
                 t, 
                 x1_true, 
                 x2_true,
                 update_global_weights=False
        ):
        """
            Compute the physics loss and the data loss
            
            In the paper "an expert's guide in training physics-informed neural networks" by Sifan Wang et. al., 
            the authors recomputed the weigts put on physics loss and data loss by weighting the norm of their gradient 
            every f steps this scheme is implemented here. This function will be called every f step in the 
            training loop where f default to 1000
            
            https://arxiv.org/pdf/2308.08468
        """
        l = self.parameter_net(t)
        input = torch.concat([l,t])
        x = self.oscillator_net(input)
        x1, x2 = x[0:1], x[1:2]
        dx2_dt = torch.autograd.grad(outputs=x2, inputs=t, grad_outputs=torch.ones_like(x2), create_graph=True)[0]
        dx1_dt = torch.autograd.grad(outputs=x1, inputs=t, grad_outputs=torch.ones_like(x1), create_graph=True)[0]
        
        physics_loss = self.mse(dx1_dt, x2) + self.mse(dx2_dt, (-9.8/l) * torch.sin(x1))
        data_loss = self.mse(x1, x1_true) + self.mse(x2, x2_true)
        
        if update_global_weights == True:
            grad_norm1 = torch.norm(physics_loss, p=2).item()
            grad_norm2 = torch.norm(data_loss, p=2).item()
            self.lambda1 = 0.9 * self.lambda1 + 0.1 * (grad_norm1 + grad_norm2)/grad_norm1
            self.lambda2 = 0.9 * self.lambda1 + 0.1 * (grad_norm1 + grad_norm2)/grad_norm2
        
        return self.lambda1 * physics_loss + self.lambda2 * data_loss


