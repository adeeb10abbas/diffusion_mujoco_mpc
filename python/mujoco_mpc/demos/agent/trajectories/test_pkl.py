import torch
import sys

def load_and_inspect_pkl(file_path):
    data_structure = torch.load(file_path)

    # Iterate through each category ('observations' and 'actions')
    for key in data_structure.keys():
        print(f"Key: {key}")
        print(f"Length: {len(data_structure[key])}")
        if len(data_structure[key]) > 0 and hasattr(data_structure[key][0], 'shape'):
            print(f"Shape of first element: {data_structure[key][0].shape}")
        print("-" * 40)

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python3 test_pkl.py <path_to_pkl_file>")
        sys.exit(1)

    file_path = sys.argv[1]
    load_and_inspect_pkl(file_path)
