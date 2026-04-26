# Pendulum With A Varying Arm Length 
This is a toy problem I came up for myself to quickly become familiar with PINNs.
## Problem Description
> Suppose we are given some data $(t_i, \theta_i)_i$ of pendulum with time_varying arm legnth $L(t)$. We consider how to uncover $L(t)$ using a simple PINN(physics-informed neural network)

The pendulum equation is given by:

$$
\theta'' =  - \frac{g}{L}\sin(\theta)
$$

Rewriting as a system:

$$
\frac{d}{dt}
\begin{pmatrix}
x_1 \\
x_2
\end{pmatrix}= \begin{pmatrix}
x_2 \\
-\frac{g}{L}\sin(x_1)
\end{pmatrix}
$$

## Numerical Simulation/Data Generation
We use a simple *heun's scheme* , specifically with the predictor-corrector scheme. 

$$
x_{t+1}=x_t + \frac{\Delta t}{2}(F(x+\Delta x)+F(x))
$$

where $F(x+\Delta x)$ would be replaced by an *euler guess*, i.e., $F(x+\Delta x)\approx F(x_t +F(x)\Delta t)$

## Optimizations
### Global weight adjustment 
- Adjust the weight of physics loss and data loss every $f$ steps:

$$
w_{\text{physics}}\gets \alpha w_{\text{physics}} +(1-\alpha)\frac{\|\nabla_{\theta } L_{\text{total}}\|_2 }{\|\nabla_{\theta} L_{\text{physics}}\|_2} 
$$

$$
w_{\text{data}}\gets \alpha w_{\text{data}} +(1-\alpha)\frac{\|\nabla_{\theta } L_{\text{total}}\|_2 }{\|\nabla_{\theta} L_{\text{data}}\|_2}
$$

where $f$ is  $1000$ by default.
