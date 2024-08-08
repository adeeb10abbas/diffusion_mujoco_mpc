import os
import pickle
import zarr
import numpy as np
import torch
from diffusion_policy.common.replay_buffer import ReplayBuffer
from tqdm import tqdm

def main(input_pkl_file_path):
    # Determine the output directory and name for the Zarr file
    output_dir = os.path.dirname(input_pkl_file_path)
    zarr_name = os.path.basename(input_pkl_file_path) + "_replay_buffer.zarr"
    zarr_output_path = os.path.join(output_dir, zarr_name)
    print(zarr_output_path)
    # Create or open an existing Zarr file
    replay_buffer = ReplayBuffer.create_from_path(zarr_output_path, mode="a")

    # List all pkl files in the provided directory and subdirectories
    pkl_list = []
    for root, dirs, files in os.walk(input_pkl_file_path):
        for file in files:
            if file.endswith(".pkl"):
                pkl_list.append(os.path.join(root, file))
                
    for pkl_file in tqdm(pkl_list, desc="Processing pkl files"):
        print("Processing pkl file: %s" % pkl_file)
        with open(pkl_file, "rb") as f:
            data = torch.load(f)
            # import pdb; pdb.set_trace()
        data_to_save = {}
        for key, tensor_list in data.items():
            print(key, np.array(tensor_list).shape)
            
            data_to_save[key] = np.array(tensor_list)
        # import pdb; pdb.set_trace()
        ## action_space
        u = data_to_save["u"]
        spline_params = data_to_save["spline_params"]


        # Stack the arrays along the 0th dimension
        data_to_save["action"] = np.concatenate([u, # 1
                                            spline_params[:,0,:], # 10
                                            ], axis=1)

        del data
        # keys_to_delete = ["rdda_right_act", "right_operator_pose", "rdda_left_act", "left_operator_pose"]
        # for key in keys_to_delete:
        #     if key in data_to_save:
        #         del data_to_save[key]

        # Add processed data to replay buffer
        # assert len(set(data_to_save.keys()))== len(data_to_save.keys()) 
        # import pdb; pdb.set_trace()
        replay_buffer.add_episode(data_to_save, compressors='disk')
        print("Added data from pkl file to Zarr: %s" % pkl_file)
        
if __name__=="__main__":
    import sys
    if len(sys.argv) != 2:
        print("Usage: generate_zarr_episode.py <input_pkl_directory>")
        sys.exit(1)
    main(sys.argv[1])
    
    
