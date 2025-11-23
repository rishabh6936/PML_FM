import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm


from mlp_flow_class import DiscreteVelocityMLP, ProbWrapper
from visualization_utils import FlowVisualizer


from flow_matching.path import MixtureDiscreteProbPath
from flow_matching.path.scheduler import PolynomialConvexScheduler
from flow_matching.solver import MixtureDiscreteEulerSolver
from flow_matching.loss import MixturePathGeneralizedKL

if torch.cuda.is_available(): device = 'cuda:0'
elif torch.backends.mps.is_available(): device = 'mps'
else: device = 'cpu'

torch.manual_seed(42)
np.random.seed(42)

config = {
    "lr": 1e-3,
    "vocab": 128,
    "hidden_dim": 128,
    "time_dim": 64,
    "iters": 5000,        
    "batch_size": 4096,
    "mode": "uniform",
    "epsilon": 1e-3
}


def get_data(n=4096):
    grid = 128 // 4
    idx = torch.randint(0, 8, (n,), device=device)
    blocks = torch.tensor([[0,0],[0,2],[2,0],[2,2],[1,1],[1,3],[3,1],[3,3]], device=device)[idx]
    offsets = torch.randint(0, grid, (n, 2), device=device)
    return (blocks * grid + offsets).long()


candidates = ["relu", "silu", "tanh"]
accuracy_results = {}




flow_path = MixtureDiscreteProbPath(scheduler=PolynomialConvexScheduler(n=2.0))
criterion = MixturePathGeneralizedKL(path=flow_path)

for act_name in candidates:
    print(f"Training {act_name.upper()} Model...")
    

    model = DiscreteVelocityMLP(
        input_dim=config["vocab"], 
        time_dim=config["time_dim"], 
        hidden_dim=config["hidden_dim"],
        activation=act_name 
    ).to(device)
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=config["lr"])
    

    pbar = tqdm(range(config["iters"]), desc=f"Training {act_name}", leave=False)
    for _ in pbar:
        #drawing valid samples from target distribution
        x1 = get_data(config["batch_size"])

        #draw noise sample
        x0 = torch.randint(0, config["vocab"], x1.shape, device=device)
        t = torch.rand(x1.shape[0], device=device) * (1 - config["epsilon"])

        #according to optimal path  we calculate the intermediate states between noise and data
        traj = flow_path.sample(t=t, x_0=x0, x_1=x1)
        
        #skip pred logits and calc it directly in loss
        loss = criterion(model(traj.x_t, t), x1, traj.x_t, t)
        #optimisation steps and model parameter update
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()


    print(f"   Evaluating {act_name.upper()}...")
    
    #prepare for inference and initialise ode solver
    prob_model = ProbWrapper(model)
    solver = MixtureDiscreteEulerSolver(model=prob_model, path=flow_path, vocabulary_size=config["vocab"])
    

    viz = FlowVisualizer(solver, config, device)
    
    #calculate constraint accuracy
    acc = viz.calculate_validity_accuracy(n_samples=5000)
    

    accuracy_results[act_name] = acc * 100 # Convert to percentage


plt.figure(figsize=(8, 6))

names = list(accuracy_results.keys())
values = list(accuracy_results.values())
colors = ['red', 'green', 'blue']

bars = plt.bar(names, values, color=colors, alpha=0.7)


for bar in bars:
    yval = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2, yval + 1, f"{yval:.1f}%", ha='center', va='bottom', fontweight='bold')

plt.ylim(0, 105) 
plt.ylabel("Validity Accuracy (%)")
plt.title(f"Which Activation Function Learns Constraints Best?\n(After {config['iters']} iters)")
plt.grid(axis='y', alpha=0.3)

plt.savefig("activation_accuracy_comparison.png")
print("Chart saved to activation_accuracy_comparison.png")
#plt.show()