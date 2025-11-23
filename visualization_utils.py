import os
import shutil
import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import imageio.v2 as imageio
from flow_matching.loss import MixturePathGeneralizedKL
from matplotlib.patches import Patch
class FlowVisualizer:
    """We make this class for visualising our results, we would use these visualisations in the paper. There are multiple
    functions which generate gifs, snapshots, graphs etc. """
    def __init__(self, solver, config, device):
        self.solver = solver
        self.config = config
        self.device = device
        self.mask_token_id = config["vocab"]  

    def _get_initial_noise(self, n_samples):
        #Helper  to sample the prior distribution p(x0)
        dim = 2
        if self.config["mode"] == "uniform":
            return torch.randint(size=(n_samples, dim), high=self.config["vocab"], device=self.device)
        elif self.config["mode"] == "mask":
            return (torch.zeros(size=(n_samples, dim), device=self.device) + self.mask_token_id).long()
        else:
            raise ValueError(f"Unknown mode: {self.config['mode']}")

    def _filter_data(self, data_np):
        #helper to prevent plotting the mask token
        if self.config["mode"] == "mask":
            mask_arr = np.array([self.mask_token_id, self.mask_token_id])
            
            valid_rows = np.all(data_np != mask_arr, axis=1)
            return data_np[valid_rows]
        return data_np

    def save_snapshots(self, n_samples=100000, steps=64, output_folder="snapshots"):
        # Check if folder exists, if not make it
        if not os.path.exists(output_folder):
            os.mkdir(output_folder)
            
        print("Starting to generate snapshots...")

        # Setup time array
        times = torch.linspace(0, 1 - self.config["epsilon"], 9).to(self.device)
        
        # Get start noise
        start_noise = self._get_initial_noise(n_samples)
        
        # we get the trajectory from the solver
        with torch.no_grad():
            trajectory = self.solver.sample(
                x_init=start_noise,
                step_size=1.0/steps,
                return_intermediates=True,
                time_grid=times
            )
        print("Trjectory:",trajectory)    
            
        # Looping through the time
        for i in range(len(times)):
            # Get the data for this specific time step
            current_data = trajectory[i]
            
            # Convert to numpy array
            data_np = current_data.cpu().numpy()
            
            # Filter the data
            valid_data = self._filter_data(data_np)
            
            
            plt.figure(figsize=(5, 5))
            
            # Only plot if we have points
            if len(valid_data) > 0:
                x_points = valid_data[:, 0]
                y_points = valid_data[:, 1]
                
                plt.hist2d(
                    x_points, y_points, 
                    bins=self.config["vocab"],
                    range=[[0, self.config["vocab"]], [0, self.config["vocab"]]],
                    cmap="viridis"
                )
            
            # Set titles and remove axis
            current_time = float(times[i].item())
            plt.title("t=" + str(round(current_time, 2)))
            plt.axis('off')
            
            # Save the file
            save_path = output_folder + "/step_" + str(i) + ".png"
            plt.savefig(save_path, bbox_inches='tight')
            
            
            plt.close()
            
        print("Finished saving snapshots.")

    def create_gif(self, n_samples=50000, steps=64, filename="flow_animation.gif"):
        
        print("Starting gif generation...")
        
        # Create a temp folder. 
        temp_folder = "temp_frames"
        if not os.path.exists(temp_folder):
            os.mkdir(temp_folder)

        # Setup time
        times = torch.linspace(0, 1 - self.config["epsilon"], steps + 1).to(self.device)
        start_x = self._get_initial_noise(n_samples)

        # Get trajectory
        with torch.no_grad():
            trajectory = self.solver.sample(
                x_init=start_x,
                step_size=1.0/steps,
                return_intermediates=True,
                time_grid=times
            )
        
        # Convert to numpy
        traj_np = trajectory.cpu().numpy()
        
        # We need to save the paths to load them later
        saved_files = []

        # Loop through time
        for i in range(len(traj_np)):
            
            # Get data for this frame
            current_data = self._filter_data(traj_np[i])
            
            plt.figure(figsize=(6, 6))
            
            # Only plot if not empty
            if len(current_data) > 0:
                plt.hist2d(
                    current_data[:, 0], current_data[:, 1], 
                    bins=self.config["vocab"], 
                    range=[[0, self.config["vocab"]], [0, self.config["vocab"]]],
                    cmap="viridis"
                )
            
            plt.axis('off')
            
            
            t_val = float(times[i])
            plt.title("Time: " + str(round(t_val, 2)))
            
            # Save file
            save_name = temp_folder + "/frame_" + str(i) + ".png"
            plt.savefig(save_name, bbox_inches='tight')
            plt.close()
            
            saved_files.append(save_name)
 
        # Load images back
        images = []
        for file_path in saved_files:
            img = imageio.imread(file_path)
            images.append(img)
            
        # Save final gif
        imageio.mimsave(filename, images, fps=15, loop=0)
        
        # Clean up
        for f in saved_files:
            if os.path.exists(f):
                os.remove(f)
        os.rmdir(temp_folder)
        
        print("Gif saved as " + filename)

    
    
    def calculate_validity_accuracy(self, n_samples=10000):
        """
        We calculate the number of points in valid blocks and present it as an accuracy statistic
        """
        
        
        x_init = self._get_initial_noise(n_samples)
        with torch.no_grad():
            traj = self.solver.sample(
                x_init=x_init, 
                step_size=1.0/64, 
                verbose=False
            )
        gen_samples = traj.cpu().numpy() # Shape [N, 2]


        vocab_size = self.config["vocab"]
        block_width = vocab_size // 4
        
        #calculate the block coordinates
        grid_x = np.floor(gen_samples[:, 0] / block_width).astype(int)
        grid_y = np.floor(gen_samples[:, 1] / block_width).astype(int)


        #blocks are valid when they are sum is even, plot the block to understand
        is_valid = (grid_x + grid_y) % 2 == 0
        
        accuracy = np.mean(is_valid)
        
        print(f"Validity Accuracy: {accuracy * 100:.2f}%")
        return accuracy   
    
    
    def plot_elbo_heatmap(self, filename="elbo_heatmap.png"):
            """
            Calculate the ELBO for EVERY pixel in the 128x128 grid
            and plot it as a heatmap.
            """
            print("\n Generating ELBO Heatmap ")
            vocab_size = self.config["vocab"]
            
            #first construct the grid
            x = torch.arange(vocab_size, device=self.device)
            y = torch.arange(vocab_size, device=self.device)
            grid_x, grid_y = torch.meshgrid(x, y, indexing='ij')


            x_1_all = torch.stack([grid_x.reshape(-1), grid_y.reshape(-1)], dim=1).long()
            n_points = x_1_all.shape[0]

            # Prepare time integration
            n_steps = 100
            time_points = torch.linspace(0, 1 - self.config["epsilon"], n_steps, device=self.device)
 
            pixel_elbo = torch.zeros(n_points, device=self.device)
            loss_fn = MixturePathGeneralizedKL(path=self.solver.path, reduction='none') 

            print("Integrate over time...")
            with torch.no_grad():
                for t_val in time_points:
                    t_batch = torch.ones(n_points, device=self.device) * t_val
                    
                    #sample noise prior
                    x_0 = torch.randint(0, vocab_size, x_1_all.shape, device=self.device)
                    
                    #interpolate flow state
                    path_sample = self.solver.path.sample(t=t_batch, x_0=x_0, x_1=x_1_all)
                    
                    # Predict
                    logits = self.solver.model.model(path_sample.x_t, t_batch)
                    
    
                    loss = loss_fn(logits=logits, x_1=x_1_all, x_t=path_sample.x_t, t=t_batch).sum(dim=1)
                    
                    # ELBO = Negative Loss
                    pixel_elbo -= loss

            # Average over time steps
            pixel_elbo /= n_steps
            

            heatmap = pixel_elbo.reshape(vocab_size, vocab_size).cpu().numpy()
            

            plt.figure(figsize=(8, 8))
            
            # Transform Log-Likelihood -> Probability Density
            density_map = np.exp(heatmap)
            
            #clip the outliers
            cmax = np.quantile(density_map, 0.99)
            norm = cm.colors.Normalize(vmin=0, vmax=cmax)

            plt.imshow(
                density_map, 
                origin='lower', 
                cmap='viridis', 
                norm=norm
            )
            
            plt.colorbar(label='Likelihood (Density)')
            plt.title("ELBO Heatmap")
            plt.axis("off")
            
            plt.savefig(filename, bbox_inches='tight')
            print(f"Heatmap saved to {filename}")
            #plt.show()    


    def plot_streamlines(self, t_val=0.5, title="Flow Streamlines"):
        """
        Visualizes marginal vector field using  streamlines.
        """
        print(f"Generating Streamlines at t={t_val}")
        
        vocab_size = self.config["vocab"]
        
        #define the grid
        grid_size = 32 
        x = np.linspace(0, vocab_size-1, grid_size)
        y = np.linspace(0, vocab_size-1, grid_size)
        grid_x, grid_y = np.meshgrid(x, y)

        
        points_np = np.stack([grid_x, grid_y], axis=-1).reshape(-1, 2)
        points = torch.tensor(points_np).long().to(self.device)
        t = torch.ones(points.shape[0]).to(self.device) * t_val
        
        with torch.no_grad():
            # Get probability distribution over next states
            probs = self.solver.model(points, t) 

            # calculate expected position
            indices = torch.arange(vocab_size).to(self.device).float()
            expected_x = torch.sum(probs[:, 0, :] * indices, dim=1)
            expected_y = torch.sum(probs[:, 1, :] * indices, dim=1)
            
            # current position
            curr_x = points[:, 0].float()
            curr_y = points[:, 1].float()
            #get velocity vector (direction and mag)
            vel_x = expected_x - curr_x
            vel_y = expected_y - curr_y

        #horizontan and vertical velocities    
        U = vel_x.reshape(grid_size, grid_size).cpu().numpy()
        V = vel_y.reshape(grid_size, grid_size).cpu().numpy()
        speed = np.sqrt(U**2 + V**2)
        
        # Visualization Setup
        fig, ax = plt.subplots(figsize=(10, 8))
        
        # green region valid and grey region invalid
        block_w = vocab_size / 4
        for i in range(4):
            for j in range(4):
                if (i + j) % 2 == 0: 
                    color = 'green'
                else:
                    color = 'gray'
                rect = plt.Rectangle((i*block_w, j*block_w), block_w, block_w, 
                                   color=color, alpha=0.15)
                ax.add_patch(rect)
        
    
        strm = ax.streamplot(x, y, U, V, color=speed, cmap='autumn', 
                             density=1.5, linewidth=1.0, arrowsize=1.5)
        
        ax.set_xlim(0, vocab_size)
        ax.set_ylim(0, vocab_size)
        ax.set_title(f"{title} (t={t_val})\nYellow=Fast, Red=Slow")
        
       
        cbar = plt.colorbar(strm.lines, ax=ax)
        cbar.set_label('Particle Speed (Velocity Magnitude)')
        
   
        legend_elements = [Patch(facecolor='green', alpha=0.15, label='Valid Region (Data)'),
                           Patch(facecolor='gray', alpha=0.15, label='Invalid Region (Empty)')]
        ax.legend(handles=legend_elements, loc='upper right')

        filename = f"streamlines_t{t_val}.png"
        plt.savefig(filename, bbox_inches='tight', dpi=150)
        print(f"Clean Streamlines saved to {filename}")
        #plt.show()