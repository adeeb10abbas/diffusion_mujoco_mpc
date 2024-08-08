import zarr
zarr_path="/home/ali/super_isolated/mujoco_mpc/python/mujoco_mpc/demos/agent/trajectories/pkls/_replay_buffer.zarr"

def list_keys(zarr_path):
    # Open the Zarr file
    z = zarr.open(zarr_path, mode='r')
    
    # Function to recursively list keys
    def recursive_list_keys(group, prefix=''):
        keys = []
        for key in group:
            if isinstance(group[key], zarr.hierarchy.Group):
                # If the key is a group, list keys within it
                keys.extend(recursive_list_keys(group[key], prefix + key + '/'))
            else:
                # If the key is an array, add it to the keys list
                keys.append(prefix + key)
        return keys

    # List all keys
    all_keys = recursive_list_keys(z)
    return all_keys

keys = list_keys(zarr_path)
for key in keys:
    print(key)