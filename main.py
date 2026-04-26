
from src.pendulum_pinn.pendulum_ode import pendulumODE
from src.pendulum_pinn.train import PendulumParametrizer

def simulate():
    out_dir = "data/linear_decrease"
    total_time = 10
    model = pendulumODE(
        T = total_time,
        out_dir = out_dir,
        L = lambda t : 2.0 - 1.9 * t / total_time
    )
    model.simulate()

def train():
    pipe_line = PendulumParametrizer(data_path="data/linear_decrease/data.npy", 
                         output_dir="output/linear_decrease1",
                         num_data = 128,
                         epochs = 10000
                         )
    pipe_line.train()
    
def main():
    # simulate()
    train()

if __name__ == "__main__":
    main()
