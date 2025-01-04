import os
import numpy as np
import scipy.io as sio
from glob import glob
from scipy.sparse import csr_matrix, issparse

def diagnose_rank_deficiency(bb, l, solution_vector):
    """Diagnose rank deficiency in sparse matrix"""
    print('=== Sparse Matrix Rank Deficiency Diagnosis Report ===\n')

    m, n = bb.shape
    if not issparse(bb):
        bb = csr_matrix(bb)

    print('1. Basic Information:')
    print(f' Matrix size: {m} x {n}')
    print(f' Number of non-zero elements: {bb.nnz}')
    print(f' Matrix sparsity: {100*bb.nnz/(m*n):.4f}%')

    actual_rank = np.linalg.matrix_rank(bb.toarray())
    print('\n2. Rank Information:')
    print(f' Theoretical maximum rank: {min(m,n)}')
    print(f' Actual rank: {actual_rank}')
    print(f' Rank deficiency: {min(m,n) - actual_rank}')

    print('\n3. Column Analysis:')
    col_norms = np.sqrt((bb.power(2)).sum(axis=0)).A1
    zero_cols = np.where(col_norms < 1e-10)[0]
    if len(zero_cols) > 0:
        print(f' Found near-zero columns: {zero_cols}')
    else:
        print(' No near-zero columns found')

    print('\n5. Condition Number Analysis:')
    try:
        cond_val = np.linalg.cond(bb.toarray())
        print(f' Condition number estimate: {cond_val:.2e}')
        if cond_val > 1e12:
            print(' Warning: High condition number, numerical instability possible')
    except:
        print(' Condition number estimation failed, matrix may be near singular')

    if solution_vector is not None:
        print('\n6. Residual Analysis:')
        residual = l - bb @ solution_vector
        rel_residual = np.linalg.norm(residual) / np.linalg.norm(l)
        print(f' Relative residual: {rel_residual:.2e}')
        if rel_residual > 1e-6:
            print(' Warning: Large relative residual')

    print('\n=== Diagnosis Complete ===')

base_path = 'E:\projects\FlexPower\M_DCB_data_2024_1_1'
date_dirs = sorted([d for d in os.listdir(f'{base_path}\M_P4_C1C_C1W') if os.path.isdir(os.path.join(f'{base_path}\M_P4_C1C_C1W', d))])

constraint_type = 'ref_sat'
ref_sat = 2
print('Step five: DCB estimate day by day!')

for doy in date_dirs:
    time_window = 24*60  # minutes
    epochs_per_window = time_window * 2  # 30s sampling
    total_windows = 24 * 60 // time_window
    path_P4 = os.path.join(f'{base_path}\M_P4_C1C_C1W', doy)

    obs_files = sorted(glob(os.path.join(path_P4, '*.mat')))
    n_r = len(obs_files)
    n_s = 32

    DCB_R_time = np.zeros((n_r, total_windows))
    DCB_S_time = np.zeros((n_s, total_windows))

    for window in range(total_windows):
        start_epoch = window * epochs_per_window
        end_epoch = (window + 1) * epochs_per_window

        # Check satellite availability
        PRN = np.zeros(32)
        for i in range(n_r):
            data = sio.loadmat(obs_files[i])
            P4 = data['P4']
            for j in range(32):
                PRN[j] += np.sum(P4[start_epoch:end_epoch, j] != 0)

        if np.sum(PRN > 0) < 4:
            print(f'Warning: Window {window} has insufficient data. Skipping...')
            continue

        d_sat = np.where(PRN == 0)[0]
        n_s = 32 - len(d_sat)

        if len(d_sat) > 0:
            print(f'Unavailable satellites: {len(d_sat)}')
            print(f'Satellite indices: {d_sat}')

        B = []
        ll = []

        # Constraint conditions
        if constraint_type == 'zero_sum':
            C = np.zeros(n_s + n_r)
            C[n_r:n_r + n_s] = 1
            Wx = 0
        else:
            if ref_sat in d_sat:
                print(f'Warning: Reference satellite G{ref_sat:02d} unavailable in window {window}')
                continue
            C = np.zeros(n_s + n_r)
            ref_sat_col = ref_sat - np.sum(d_sat < ref_sat)
            C[n_r + ref_sat_col - 1] = 1
            Wx = 0

        # Process P4 files
        for i in range(n_r):
            data = sio.loadmat(obs_files[i])
            P4 = data['P4']

            for j in range(32):
                mask = P4[start_epoch:end_epoch, j] != 0
                if not np.any(mask):
                    continue

                M_col = np.zeros(n_s + n_r)
                M_col[i] = 1
                M_col[n_r + j] = 1

                B.extend([M_col] * np.sum(mask))
                ll.extend(P4[start_epoch:end_epoch, j][mask])

        if not B:
            continue

        B = np.array(B)
        ll = np.array(ll)

        if len(d_sat) > 0:
            B = np.delete(B, d_sat + n_r, axis=1)

        BB = np.vstack([B, C])
        L = np.append(ll, Wx)

        m, n = BB.shape
        if m < n:
            print(f'Warning: Window {window}: Underdetermined system')
            continue

        R = np.linalg.lstsq(BB, L, rcond=None)[0]

        diagnose_rank_deficiency(BB, L, R)

        DCB_R_time[:, window] = R[:n_r] / 299792458*1e9
        valid_sats = np.setdiff1d(np.arange(32), d_sat)
        DCB_S_time[valid_sats, window] = R[n_r:n_r + n_s] / 299792458*1e9

    # Save results
    os.makedirs('M_Result', exist_ok=True)
    out_file = os.path.join('M_Result', f'M_R{doy}_15min.mat')
    sio.savemat(out_file, {'DCB_R_time': DCB_R_time, 'DCB_S_time': DCB_S_time})

print('end')

