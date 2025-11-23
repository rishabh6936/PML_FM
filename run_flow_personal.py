import time
import torch
from mlp_flow_class import DiscreteVelocityMLP, ProbWrapper
from torch import nn, Tensor


from flow_matching.path import MixtureDiscreteProbPath
from flow_matching.path.scheduler import PolynomialConvexScheduler
from flow_matching.solver import MixtureDiscreteEulerSolver
from flow_matching.utils import ModelWrapper
from flow_matching.loss import MixturePathGeneralizedKL


import numpy as np
import matplotlib.cm as cm
import matplotlib.pyplot as plt
from tqdm import tqdm


import imageio.v2 as imageio
import os
import shutil
from visualization_utils import FlowVisualizer


if torch.cuda.is_available():
    device = 'cuda:0'
elif torch.backends.mps.is_available():
    device = 'mps' # For Mac 
else:
    device = 'cpu'
print(f"Using device: {device}")

torch.manual_seed(42)
np.random.seed(42) 

def load_checkerboard(n_grid_points: int = 128, batch_size: int = 200, device: str = "cpu") -> torch.Tensor:
    """ We implement our dataset loaded here, it takes the checkboard grid size, number of points as the batch size and
    returns the coordinates of these points inside the checkboard grid."""

    block_width = n_grid_points // 4
    

    valid_blocks = torch.tensor([
        [0, 0], [0, 2], [2, 0], [2, 2],
        [1, 1], [1, 3], [3, 1], [3, 3]   
    ], device=device)
    

    block_indices = torch.randint(0, 8, size=(batch_size,), device=device)
    chosen_blocks = valid_blocks[block_indices]  # Shape: (batch_size, 2)
    
    offsets = torch.randint(0, block_width, size=(batch_size, 2), device=device)
    
    x_end = chosen_blocks * block_width + offsets
    return x_end.long()
    
#We define the following values for our configuration
config = {
    "lr": 1e-3,
    "batch_size": 4096,
    "iters": 10000,      
    "log_freq": 3000,     
    "vocab": 128,
    "hidden_dim": 128,
    "time_dim": 64,       
    "mode": "uniform",   
    "epsilon": 1e-3
}


if config["mode"] == "mask":
    mask_token_id = config["vocab"]
    total_vocab = config["vocab"] + 1
else:
    total_vocab = config["vocab"]


#We initialize our model for training
velocity_model = DiscreteVelocityMLP(
    input_dim=total_vocab, 
    time_dim=config["time_dim"], 
    hidden_dim=config["hidden_dim"]
).to(device)
noise_scheduler = PolynomialConvexScheduler(n=2.0)
flow_path = MixtureDiscreteProbPath(scheduler=noise_scheduler)
optimizer = torch.optim.AdamW(velocity_model.parameters(), lr=config["lr"], weight_decay=1e-5)
criterion = MixturePathGeneralizedKL(path=flow_path)
start_time = time.time()
print(f"Model initialized. Training on {device} with {config['mode']} source.")
    


print(f"Starting training for {config['iters']} iterations...")
# We initialize a progress bar
pbar = tqdm(range(config["iters"]), desc="Training", unit="step")
running_loss = 0.0

for step in pbar:
    #we generate the target distribution
    clean_data = load_checkerboard(
        n_grid_points=config["vocab"], 
        batch_size=config["batch_size"], 
        device=device
    )
    
    if config["mode"] == "uniform":
        noise_data = torch.randint_like(clean_data, high=config["vocab"])
    elif config["mode"] == "mask":
        noise_data = torch.zeros_like(clean_data) + mask_token_id
    else:
        raise ValueError("Invalid mode in config")
    
    #We sample random times to train the model to predict
    target = torch.rand(clean_data.shape[0], device=device) * (1 - config["epsilon"])
    
    #according to optimal path  we calculate the intermediate states between noise and data
    trajectory = flow_path.sample(t=target, x_0=noise_data, x_1=clean_data)
    x_t = trajectory.x_t

    #model predicts the velocity
    pred_logits = velocity_model(x_t, target)

    
    loss = criterion(logits=pred_logits, x_1=clean_data, x_t=x_t, t=target)

    #optimisation step
    optimizer.zero_grad()
    loss.backward()

    torch.nn.utils.clip_grad_norm_(velocity_model.parameters(), max_norm=1.0)
    
  
    optimizer.step()
    
    #log
    loss_val = loss.item()
    running_loss = 0.9 * running_loss + 0.1 * loss_val if step > 0 else loss_val
    
    if step % 100 == 0:
        pbar.set_postfix(loss=f"{running_loss:.4f}")

print("Training finished successfully.")

#prepare for inference
prob_model = ProbWrapper(velocity_model)

#we initialize the Euler Solver
euler_solver = MixtureDiscreteEulerSolver(
    model=prob_model, 
    path=flow_path, 
    vocabulary_size=config["vocab"]
)

viz = FlowVisualizer(solver=euler_solver, config=config, device=device)

viz.save_snapshots(n_samples=100000)
viz.create_gif(n_samples=50000, filename="my_checkerboard.gif")
acc = viz.calculate_validity_accuracy()
viz.plot_elbo_heatmap()
viz.plot_streamlines(t_val=0.5)