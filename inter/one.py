import os
import numpy as np
from datetime import datetime
from typing import Dict, Tuple, List
import scipy.io as sio


def format_sites_info(sites_info: Dict) -> Dict:
    """Format Sites_Info into the required structure."""
    names = np.array([[np.array([name], dtype='<U4') for name in sites_info['name']]], dtype=object)
    doys = np.array(sites_info['doy'], dtype='float64').reshape(-1, 1)  # 8x1 double
    coords = np.array(sites_info['coor'], dtype='float64')  # 8x3 double

    return np.array([(names, doys, coords)],
                    dtype=[('name', 'O'), ('doy', 'O'), ('coor', 'O')])

def read_rinex(r_ipath: str, r_opath: str, signal_P1: str, signal_P2: str, signal_L1: str, signal_L2: str) -> Dict:
    """Read RINEX files and process observation data."""
    os.makedirs(r_opath, exist_ok=True)

    # Get list of RINEX files
    obs_files = []
    for ext in ['*.*O', '*.rnx', '*.*o']:
        obs_files.extend([f for f in os.listdir(r_ipath) if f.endswith(ext[-1])])

    if not obs_files:
        raise FileNotFoundError(f'No RINEX files found in {r_ipath}')

    obs_files.sort()

    Sites_Info = {
        'name': [],
        'doy': [],
        'coor': []
    }

    for obsn in obs_files:
        try:
            fname = os.path.join(r_ipath, obsn)

            # Check RINEX version
            with open(fname, 'r') as f:
                first_line = f.readline()
                if len(first_line) > 79 and first_line[60:80] == 'RINEX VERSION / TYPE':
                    rinex_ver = first_line[5:9]
                    if float(rinex_ver[0]) <= 2:
                        print(f'Skipping {obsn}: Only RINEX v3 is supported')
                        continue
                else:
                    print(f'Skipping {obsn}: Invalid RINEX format')
                    continue

            # Process v3 file
            obs, coor, year, doy = read_rinex_v3(fname, signal_P1, signal_P2, signal_L1, signal_L2)

            # Check for valid observations
            if (np.all(obs['P1'] == 0) or np.all(obs['P2'] == 0) or
                np.all(obs['L1'] == 0) or np.all(obs['L2'] == 0)):
                print(f'Skipping {obsn}: Missing observation types')
                continue

            # Save processed data
            outname = f"{obsn[:4]}{doy}.mat"
            outfile = os.path.join(r_opath, outname)
            sio.savemat(outfile, {'obs': obs})

            Sites_Info['name'].append(obsn[:4])
            Sites_Info['doy'].append(doy)
            Sites_Info['coor'].append(coor)

            print(f'Successfully processed file {obsn}')

        except Exception as e:
            print(f'Error processing file {obsn}: {str(e)}')
            continue

    Sites_Info['doy'] = np.array(Sites_Info['doy'])
    Sites_Info['coor'] = np.array(Sites_Info['coor'])
    return Sites_Info

def read_rinex_v3(path: str, signal_P1: str, signal_P2: str, signal_L1: str, signal_L2: str) -> Tuple[Dict, List, int, int]:
    """Read RINEX v3 observation file."""
    obs = {
        'P1': np.zeros((2880, 32)),
        'P2': np.zeros((2880, 32)),
        'L1': np.zeros((2880, 32)),
        'L2': np.zeros((2880, 32))
    }

    coor = [0, 0, 0]
    G_signal_name = []

    with open(path, 'r') as f:
        print(f'Reading RINEX file: {path}')

        while True:
            line = f.readline()
            if not line:
                break

            if len(line) > 76 and line[60:77] == 'TIME OF FIRST OBS':
                dt = datetime.strptime(line[2:18], '%Y    %m    %d')
                yy = int(dt.strftime('%y'))
                doy = dt.timetuple().tm_yday
                doy = yy * 1000 + doy

            elif len(line) > 78 and line[60:79] == 'APPROX POSITION XYZ':
                coor = [float(line[0:14]), float(line[14:28]), float(line[28:42])]

            elif len(line) > 78 and line[60:79] == 'SYS / # / OBS TYPES':
                if line[0] == 'G':
                    signal_num = int(line[1:6])
                    G_signal_name = []

                    # Read first line of signals
                    for i in range(min(13, signal_num)):
                        G_signal_name.append(line[7+4*i:10+4*i])

                    # Read additional signals if needed
                    if signal_num > 13:
                        line = f.readline()
                        for i in range(signal_num - 13):
                            G_signal_name.append(line[7+4*i:10+4*i])

            elif len(line) > 72 and line[60:73] == 'END OF HEADER':
                # Process observation data
                while True:
                    line = f.readline()
                    if not line:
                        break

                    if line[0] == '>':
                        # Parse epoch
                        h = int(line[13:15])
                        m = int(line[16:18])
                        s = float(line[19:31])
                        ep = h*120 + m*2 + s/30 + 1

                        if ep != int(ep):
                            continue

                        nsat = int(line[32:35])

                        # Read satellite data
                        for _ in range(nsat):
                            line = f.readline()
                            if not line:
                                break

                            if line[0] == 'G':
                                sat_num = int(line[1:3])
                                data = line[3:]
                                values = read_observation_values(data, G_signal_name,
                                                              signal_P1, signal_P2,
                                                              signal_L1, signal_L2)

                                obs['P1'][int(ep)-1, sat_num-1] = values[signal_P1]
                                obs['P2'][int(ep)-1, sat_num-1] = values[signal_P2]
                                obs['L1'][int(ep)-1, sat_num-1] = values[signal_L1]
                                obs['L2'][int(ep)-1, sat_num-1] = values[signal_L2]

    return obs, coor, yy, doy

def read_observation_values(data: str, G_signal_name: List[str],
                          signal_P1: str, signal_P2: str,
                          signal_L1: str, signal_L2: str) -> Dict[str, float]:
    """Parse observation values from RINEX data line."""
    target_signals = [signal_P1, signal_P2, signal_L1, signal_L2]
    values = {sig: 0.0 for sig in target_signals}

    for sig in target_signals:
        try:
            idx = G_signal_name.index(sig)
            start_idx = idx * 16
            if len(data) >= start_idx + 14:
                val_str = data[start_idx:start_idx+14].strip()
                if val_str:
                    values[sig] = float(val_str)
        except ValueError:
            continue

    return values

""""""""""""""""""""""""""""""""""""""""""""
"""Main function to process multiple days"""
""""""""""""""""""""""""""""""""""""""""""""
base_path = 'E:\\projects\\FlexPower\\M_DCB_data_2024_1_1'
signal_P1 = "C1W"
signal_P2 = "C2W"
signal_L1 = "L1C"
signal_L2 = "L2W"

r_ipath = os.path.join(base_path, 'o')
r_opath = os.path.join(base_path, f'o_{signal_P1}_{signal_P2}')

Sites_Info = read_rinex(r_ipath, r_opath, signal_P1, signal_P2, signal_L1, signal_L2)
formatted_sites_info = format_sites_info(Sites_Info)
sio.savemat(os.path.join(base_path, f'Sites_Info.mat'),
            {'Sites_Info': formatted_sites_info},
            format='5')

print('Step one: completing !')