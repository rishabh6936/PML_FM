import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm


from mlp_flow_class import DiscreteVelocityMLP


from flow_matching.path import MixtureDiscreteProbPath
from flow_matching.path.scheduler import PolynomialConvexScheduler
from flow_matching.loss import MixturePathGeneralizedKL


if torch.cuda.is_available():
    device = 'cuda:0'
elif torch.backends.mps.is_available():
    device = 'mps' # For Mac 
else:
    device = 'cpu'
print(f"Using device: {device}")

torch.manual_seed(42)
np.random.seed(42)


config = {
    "lr": 1e-3,
    "batch_size": 4096,
    "iters": 3000,        
    "vocab": 128,
    "hidden_dim": 128,
    "time_dim": 64,
    "mode": "uniform",    
    "epsilon": 1e-3
}


if config["mode"] == "mask":
    mask_token_id = config["vocab"]
    total_vocab_size = config["vocab"] + 1
else:
    mask_token_id = None
    total_vocab_size = config["vocab"]


def load_checkerboard(n_grid_points: int = 128, batch_size: int = 200, device: str = "cpu") -> torch.Tensor:
    """Generate checkerboard data."""
    block_width = n_grid_points // 4
    
    valid_blocks = torch.tensor([
        [0, 0], [0, 2], [2, 0], [2, 2],  
        [1, 1], [1, 3], [3, 1], [3, 3]   
    ], device=device)
    
    block_indices = torch.randint(0, 8, size=(batch_size,), device=device)
    chosen_blocks = valid_blocks[block_indices]
    
    offsets = torch.randint(0, block_width, size=(batch_size, 2), device=device)
    x_end = chosen_blocks * block_width + offsets
    
    return x_end.long() 


noise_scheduler = PolynomialConvexScheduler(n=2.0)
flow_path = MixtureDiscreteProbPath(scheduler=noise_scheduler)
criterion = MixturePathGeneralizedKL(path=flow_path)

#we define different activation functions we want to compare
activation_candidates = ["relu", "silu", "tanh"]
results = {}



for act_name in activation_candidates:
    """Does the whole training like our main script just with different activation functions for comparison"""
    print(f"Training model with {act_name.upper()}...")
    
    #initialise the model
    model = DiscreteVelocityMLP(
        input_dim=total_vocab_size, 
        time_dim=config["time_dim"], 
        hidden_dim=config["hidden_dim"],
        activation=act_name
    ).to(device)
    

    optimizer = torch.optim.AdamW(model.parameters(), lr=config["lr"], weight_decay=1e-5)
    
 
    current_losses = []
    pbar = tqdm(range(config["iters"]), desc=f"Activ: {act_name}", unit="step")
    

    avg_loss = 0.0
    
    for step in pbar:
        
        # target distribution samples
        clean_data = load_checkerboard(
            n_grid_points=config["vocab"], 
            batch_size=config["batch_size"], 
            device=device
        )
        

        if config["mode"] == "uniform":
            noise_data = torch.randint_like(clean_data, high=config["vocab"])
        elif config["mode"] == "mask":
            noise_data = (torch.zeros_like(clean_data) + mask_token_id).long()
            
        
        t = torch.rand(clean_data.shape[0], device=device) * (1 - config["epsilon"])

        #according to optimal path  we calculate the intermediate states between noise and data
        traj = flow_path.sample(t=t, x_0=noise_data, x_1=clean_data)
        

        pred_logits = model(traj.x_t, t)
        loss = criterion(logits=pred_logits, x_1=clean_data, x_t=traj.x_t, t=t)
        
        #optimisation steps and model parameter update
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        

        val = loss.item()
        if avg_loss == 0.0:
            avg_loss = val
        else:
            avg_loss = 0.95 * avg_loss + 0.05 * val
            
        current_losses.append(avg_loss)
        
        if step % 100 == 0:
            pbar.set_postfix(loss=f"{avg_loss:.4f}")
            

    results[act_name] = current_losses
    print(f"Finished {act_name}. Final Loss: {avg_loss:.4f}\n")


print("Plotting results...")
plt.figure(figsize=(12, 7))

colors = {'relu': 'red', 'silu': 'blue', 'tanh': 'green'}

for act_name, loss_curve in results.items():
    plt.plot(loss_curve, label=f"{act_name.upper()}", linewidth=2, color=colors.get(act_name))

plt.yscale('log') 
plt.xlabel("Training Steps")
plt.ylabel("Loss (Log Scale)")
plt.title(f"Impact of Activation Function on Flow Matching Convergence\n(Hidden Dim: {config['hidden_dim']}, Time Dim: {config['time_dim']})")
plt.legend()
plt.grid(True, which="both", ls="-", alpha=0.2)


output_filename = "experiment_activations.png"
plt.savefig(output_filename, dpi=300)
print(f"Plot saved to {output_filename}")
plt.show()