import sys

import os

#sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

sys.path.insert(0, os.path.abspath(os.path.join('', '..'))) 

 

# Import the required modules

import numpy as np



import h5py



 

np.random.seed(42)

 

def load_networks_by_spectral_radius():

    filename = f'gaussian_networks.hdf5'

    spectral_radius = 0.8

    networks = []

    readouts = []

    with h5py.File(filename, 'r') as f:

        spectral_radius_group = str(spectral_radius)  # Ensure the spectral radius is in the correct format

        if spectral_radius_group in f:

            for i in range(10):  # Assuming there are always 10 networks

                network_dataset_name = f'{spectral_radius_group}/networks/network_{i}'

                readout_dataset_name = f'{spectral_radius_group}/readouts/readout_{i}'

                if network_dataset_name in f:

                    network = f[network_dataset_name][()]

                    networks.append(network)#

                    readout = f[readout_dataset_name][()]

                    readouts.append(readout)

                else:

                    print(f"Dataset {network_dataset_name} not found.")

        else:

            print(f"Spectral radius group {spectral_radius_group} not found.")
    readout = np.array([readout[2],readout[6]])
    return networks[2],readout


    

    # the file should be in the data store directory above the current directory
