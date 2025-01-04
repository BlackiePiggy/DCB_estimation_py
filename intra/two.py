import os
import numpy as np
import scipy.io as sio
from glob import glob

# Read .mat files from M_OBS directory
obs_path = 'E:\projects\FlexPower\M_DCB_data_2024_1_1\o_C1C_C1W'
mat_files = sorted(glob(os.path.join(obs_path, '*.mat')))

# Process each .mat file
for mat_file in mat_files:
    # Load .mat file
    data = sio.loadmat(mat_file)
    obs = data['obs']

    # Get DOY from filename
    doy = os.path.basename(mat_file)[4:9]

    # Calculate P4
    P4 = np.zeros(obs['P1'][0][0].shape)
    valid_idx = (obs['P1'][0][0] != 0) & (obs['P2'][0][0] != 0)
    P4[valid_idx] = obs['P1'][0][0][valid_idx] - obs['P2'][0][0][valid_idx]

    # Create output directory if not exists
    out_dir = os.path.join('E:\projects\FlexPower\M_DCB_data_2024_1_1\M_P4_C1C_C1W', doy)
    os.makedirs(out_dir, exist_ok=True)

    # Save P4 as .mat file
    out_file = os.path.basename(mat_file).replace('.mat', 'P4.mat')
    out_path = os.path.join(out_dir, out_file)
    sio.savemat(out_path, {'P4': P4})

print('Step two: completing!')