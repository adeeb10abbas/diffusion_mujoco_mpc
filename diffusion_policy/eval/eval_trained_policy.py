import time
# from multiprocessing.managers import SharedMemoryManager
import click
import numpy as np
import torch
import dill
import hydra
import scipy.spatial.transform as st

from matplotlib.backends.backend_pdf import PdfPages
from diffusion_policy.workspace.base_workspace import BaseWorkspace
from diffusion_policy.policy.base_lowdim_policy import BaseLowdimPolicy

ckpt_path = "/home/ali/super_isolated/mujoco_mpc/diffusion_policy/data/outputs/2024.08.09/21.29.13_train_diffusion_unet_lowdim_mjpc_cartpole_dp_transformer/checkpoints/latest.ckpt"
pkl_path = "/home/ali/super_isolated/mujoco_mpc/python/mujoco_mpc/demos/agent/trajectories/pkls/cartpole_traj_0.pkl"
import matplotlib.pyplot as plt

def load_pkl_obs(pkl_path):
    print("Processing pkl file: %s" % pkl_path)
    with open(pkl_path, "rb") as f:
        data = torch.load(f)
        # import pdb; pdb.set_trace()
    # Prepare data to be saved in Zarr
    data_to_save = {}
    for key, tensor_list in data.items():
        # print(key, len(tensor_list))

        data_to_save[key] = np.array(tensor_list)

    pos = data_to_save["pos"]
    vel = data_to_save["vel"]
    u = data_to_save["u"]
    spline_params = data_to_save["spline_params"]

    data_to_save["obs"] = np.concatenate([pos, #2 
                                            vel, #2
                                            spline_params[:,0,:], #10 
                                            ], axis = 1)
    # obs is 14
    # action is 11
    
    data_to_save["action"] = np.concatenate([u, # 1
                                        spline_params[:,0,:], # 10
                                        ], axis=1)


    obs_dict = {}
    # obs_dict['obs'] = data_to_save['obs']
    # obs_dict['action'] = data_to_save['action']
    #     obs_dict = {}
    obs_dict['obs'] = data_to_save['obs']
    obs_dict['action'] = data_to_save['action']
    
    return obs_dict

    return obs_dict

def get_obs_dict(loaded_pkl, index, size):
    """
    Extract a window of data from each key in the dictionary.
    
    Args:
        loaded_pkl (dict): Dictionary containing various time series or batched data.
        index (int): Start index from which to extract the data.
        size (int): Number of data points to extract from the start index.
    
    Returns:
        dict: A new dictionary with the same keys as `loaded_pkl` but containing only the window of data.
    """
    obs_dict = {}
    if index < 2:
        index = 2
    for key, data in loaded_pkl.items():
        if isinstance(data, np.ndarray) and data.ndim > 1:
            obs_dict[key] = torch.from_numpy(data[index:index+size]).unsqueeze(0)
        else:
            raise ValueError("Data under key '{}' is not in the expected format.".format(key))
    return obs_dict

raw_dict = load_pkl_obs(pkl_path=pkl_path)

obs_dict_sub = get_obs_dict(raw_dict, 1, 2)
# import pdb; pdb.set_trace()
# load checkpoint
# ckpt_path = input
payload = torch.load(open(ckpt_path, 'rb'), pickle_module=dill)
cfg = payload['cfg']
cls = hydra.utils.get_class(cfg._target_)
workspace = cls(cfg)
workspace: BaseWorkspace
workspace.load_payload(payload, exclude_keys=None, include_keys=None)

# diffusion model
policy: BaseLowdimPolicy
policy = workspace.model
if cfg.training.use_ema:
    policy = workspace.ema_model

device = torch.device('cuda')
policy.eval().to(device)

# set inference params
policy.num_inference_steps = 16 # DDIM inference iterations
# policy.n_action_steps = policy.horizon - policy.n_obs_steps + 1
policy.n_action_steps = 8
# import pdb; pdb.set_trace()

inferred_actions = []

for i in range(0, len(raw_dict["action"])//policy.n_action_steps):
        obs_dict_sub = get_obs_dict(raw_dict, i*policy.n_action_steps, 2)
        # import pdb; pdb.set_trace()
        # obs_dict_torched = dict_apply(get_real_obs_dict(env_obs=obs_dict_sub, 
                                                        # shape_meta=cfg.task.obs_dim), lambda x: torch.from_numpy(x).unsqueeze(0).to(device=device))
        result = policy.predict_action(obs_dict_sub)
        action = result["action"][0].detach().to("cpu").numpy()
        for j in [*action]:
            inferred_actions.append(j)

inferred = np.array(inferred_actions)
ground_truth = np.array(raw_dict['action'])


# Create a PDF to save all plots
with PdfPages('action_comparison_plots.pdf') as pdf:
    for ind in range(11):  # u(1) + spline_params(10)
        plt.figure(figsize=(10, 6))  # Increase the figure size for better visibility
        plt.plot([i[ind] for i in inferred], label=f'Inferred Action {ind}')
        plt.plot([i[ind] for i in ground_truth[:]], label=f'Ground Truth Action {ind}')
        plt.legend(bbox_to_anchor=(1.04, 1), loc="upper left", fontsize=12)  # Increase the font size of the legend
        plt.title(f'Comparison of Inferred and Ground Truth Actions for Element {ind}', fontsize=14)  # Increase the font size of the title
        plt.xlabel('Time Step', fontsize=12)  # Add an x-label
        plt.ylabel('Action Value', fontsize=12)  # Add a y-label
        plt.tight_layout()  # Adjust layout to ensure everything fits well
        pdf.savefig()  # Save the current figure to the PDF
        plt.close()  # Close the figure to avoid display and memory issues

print("PDF with action comparison plots has been created.")

# import pdb; pdb.set_trace()


