import os
import numpy as np
import scipy.io as sio
from datetime import datetime, timedelta
from math import pi, sin, cos, atan, sqrt, factorial, asin
from scipy.sparse import csr_matrix, issparse
from scipy.special import lpmn, legendre

def find_sp3(list_sp3, doy):
    """Find corresponding sp3 file index"""
    doys = [int(name[2:7]) for name in list_sp3]
    return doys.index(int(doy))

def xyz_to_blh(x, y, z):
    """Convert XYZ coordinates to BLH (latitude, longitude, height)"""
    a = 6378137
    e2 = 0.0066943799013

    l = atan(abs(y / x))
    if y > 0:
        if x <= 0:
            l = pi - l
    else:
        if x > 0:
            l = 2 * pi - l
        else:
            l = pi + l

    b0 = atan(z / sqrt(x ** 2 + y ** 2))
    while True:
        n = a / sqrt(1 - e2 * sin(b0) ** 2)
        h = z / sin(b0) - n * (1 - e2)
        b = atan(z * (n + h) / (sqrt(x ** 2 + y ** 2) * (n * (1 - e2) + h)))
        if abs(b - b0) < 1e-6:
            break
        b0 = b

    n = a / sqrt(1 - e2 * sin(b) ** 2)
    h = z / sin(b) - n * (1 - e2)

    return b, l, h

def get_ea(sx, sy, sz, x, y, z):
    """Get elevation and azimuth angles"""
    sb, sl = xyz_to_blh(sx, sy, sz)[:2]
    t = np.array([
        [-sin(sb) * cos(sl), -sin(sb) * sin(sl), cos(sb)],
        [-sin(sl), cos(sl), 0],
        [cos(sb) * cos(sl), cos(sb) * sin(sl), sin(sb)]
    ])

    deta_xyz = np.array([x - sx, y - sy, z - sz])
    neu = t @ deta_xyz

    e = atan(neu[2] / sqrt(neu[0] ** 2 + neu[1] ** 2))
    a = atan(abs(neu[1] / neu[0]))

    if neu[0] > 0:
        if neu[1] <= 0:
            a = 2 * pi - a
    else:
        if neu[1] > 0:
            a = pi - a
        else:
            a = pi + a

    return e, a


def get_ipp(e, a, b, l, z, k):
    """Get ionospheric pierce point coordinates"""
    t = pi / 2 - e - z
    lat_ipp = asin(sin(b) * cos(t) + cos(b) * sin(t) * cos(a))
    lon_ipp = l + asin(sin(t) * sin(a) / cos(lat_ipp))
    lon_sf = lon_ipp + 30 * (k - 1) * pi / 43200 - pi
    return lat_ipp, lon_ipp, lon_sf


def norm_diy(n, m):
    """Calculate normalization factor"""
    if m == 0:
        return sqrt(factorial(n - m) * (2 * n + 1) / factorial(n + m))
    else:
        return sqrt(factorial(n - m) * (4 * n + 2) / factorial(n + m))


def get_coef(b, s, order):
    """Get coefficients for spherical harmonics"""
    cof_p = np.zeros((order + 1) ** 2)
    ms = np.linspace(s, order * s, order)
    i = 0
    x = sin(b)

    for n in range(order + 1):
        p, _ = lpmn(n, n, x)
        for m in range(n + 1):
            if m == 0:
                cof_p[i] = p[m][n] * norm_diy(n, m)
                i += 1
            else:
                cof_p[i] = p[m][n] * norm_diy(n, m) * cos(ms[m - 1])
                i += 1
                cof_p[i] = p[m][n] * norm_diy(n, m) * sin(ms[m - 1])
                i += 1

    return cof_p


def get_matrix(p4, x, y, z, sx, sy, sz, n_r, ith, order, ion_data, doy, time_window, epo_perwin, stec_gim, calc_gim):
    """Build design matrix and observation vector"""
    m_matrix = []
    l_vector = []
    sb, sl = xyz_to_blh(sx, sy, sz)[:2]

    if calc_gim:
        stec_station_gim = np.zeros((epo_perwin, 32))
        year = 2000 + int(int(doy) / 1000)
        day = int(doy) % 1000
        base_date = datetime(year, 1, 1) + timedelta(days=day - 1)

    for j in range(32):
        for k in range((epo_perwin) * (time_window - 1), epo_perwin * time_window):
            if p4[k, j] == 0:
                continue

            m_col = np.zeros((order + 1) ** 2 + 32 + n_r)
            e, a = get_ea(sx, sy, sz, x[k, j] * 1000, y[k, j] * 1000, z[k, j] * 1000)
            ippz = asin(6378137 * sin(0.9782 * (pi / 2 - e)) / (6378137 + 506700))
            lat_ipp, lon_ipp, lon_sf = get_ipp(e, a, sb, sl, ippz, k+1)

            if calc_gim:
                epoch_time = base_date + timedelta(seconds=(k - 1) * 30)
                stec_gim_val = vtec2stec(ion_data, epoch_time, lat_ipp, lon_ipp, e)
                if isinstance(stec_gim_val, (list, np.ndarray)) and len(stec_gim_val) > 1:
                    stec_gim_val = np.mean(stec_gim_val)
                if stec_gim_val is not None and not np.isnan(stec_gim_val):
                    epoch_index = k - (time_window - 1) * epo_perwin
                    stec_station_gim[epoch_index, j] = stec_gim_val

            m_col[ith] = (-9.52437) * cos(ippz)
            m_col[n_r + j] = (-9.52437) * cos(ippz)
            st = n_r + 32
            ed = (order + 1) ** 2 + n_r + 32
            ionc_coef = get_coef(lat_ipp, lon_sf, order)
            m_col[st:ed] = ionc_coef

            m_matrix.append(m_col)
            l_vector.append(p4[k, j] * (-9.52437) * cos(ippz))

    if calc_gim:
        stec_gim[ith] = stec_station_gim

    return np.array(m_matrix), np.array(l_vector), stec_gim


def calculate_parameter_rms(b, l, r, n_r, n_s):
    """Calculate RMS for station and satellite DCBs"""
    v = l - b @ r
    n = len(l)
    u = len(r)
    dof = n - u

    sigma0 = np.sqrt(np.sum(v ** 2) / dof)
    q = np.linalg.inv(b.T @ b)

    rms_station = np.zeros(n_r)
    rms_sat = np.zeros(32)

    for i in range(n_r):
        rms_station[i] = sigma0 * np.sqrt(q[i, i])

    for i in range(n_s):
        idx = n_r + i
        rms_sat[i] = sigma0 * np.sqrt(q[idx, idx])

    return rms_station * 1e9 / 299792458, rms_sat * 1e9 / 299792458


def get_mdcb(doy, sites_info, sate, order, time_window, window_num, constra, ref_sat, input_path, ion_data=None,
             calc_stec=False, calc_gim=False):
    """Main DCB estimation function"""
    if calc_gim and ion_data is None:
        raise ValueError('ion_data is required when calc_gim is true')

    coor = sites_info['coor'][0]
    stations = sites_info['name'][0]
    doys = sites_info['doy'][0]
    x, y, z = sate['x'][0][0], sate['y'][0][0], sate['z'][0][0]

    list_obs = [f for f in os.listdir(os.path.join(input_path, str(doy))) if f.endswith('.mat')]
    n_r = len(list_obs)

    prn = np.zeros(32)
    prn_pre = np.zeros(32)
    dcb_s = np.zeros(32)
    epo_perwin = 2880 // window_num
    prn_num = 0

    stec_values = []
    stec_gim = []
    if calc_stec:
        stec_values = [None] * n_r
    if calc_gim:
        stec_gim = [None] * n_r

    # Check observations per satellite
    for i in range(n_r):
        p4_filename = os.path.join(input_path, str(doy), list_obs[i])
        p4_data = sio.loadmat(p4_filename)
        p4 = p4_data['P4']

        for j in range(32):
            prn[j] += np.count_nonzero(p4[(time_window - 1) * epo_perwin:(time_window) * epo_perwin, j])

        prn_diff = prn - prn_pre
        prn_pre = prn.copy()
        prn_num += np.count_nonzero(prn_pre)

    d_sat = np.where(prn == 0)[0]
    if len(d_sat) == 0:
        n_s = 32
    else:
        raise ValueError(f'On day {doy}, PRN {d_sat} has no observations.')

    # LS estimate initialization
    b_matrix = []
    l_vector = []
    c_vector = np.zeros((order + 1) ** 2 + n_s + n_r)

    if constra == 1:
        c_vector[n_r:n_r + n_s] = 1  # Zero-mean constraint
    elif constra == 2:
        c_vector[n_r + ref_sat - 1] = 1  # Reference satellite constraint
    wx = 0

    for i in range(n_r):
        p4_data = sio.loadmat(os.path.join(input_path, str(doy), list_obs[i]))
        p4 = p4_data['P4']
        site = list_obs[i][:4]

        index = np.where(stations[0] == np.array([site], dtype='<U4'))[0][0]
        sx, sy, sz = coor[index]

        sn, sl, stec_gim = get_matrix(p4, x, y, z, sx, sy, sz, n_r, i, order,
                                      ion_data, doy, time_window, epo_perwin,
                                      stec_gim if calc_gim else None, calc_gim)

        b_matrix.extend(sn)
        l_vector.extend(sl)

    b_matrix = np.array(b_matrix)
    l_vector = np.array(l_vector)

    if len(d_sat) > 0:
        b_matrix = np.delete(b_matrix, d_sat + n_r, axis=1)

    bb = np.vstack([b_matrix, c_vector])
    ll = np.append(l_vector, wx)

    r = np.linalg.lstsq(bb, ll, rcond=None)[0]

    diagnose_rank_deficiency(bb, ll, r)

    rms_station, rms_sat = calculate_parameter_rms(bb, ll, r, n_r, n_s)

    dcb_r = r[:n_r] * 1e9 / 299792458
    temp = np.arange(1, 33)
    temp = np.delete(temp, d_sat)
    dcb_s[temp - 1] = r[n_r:n_r + n_s] * 1e9 / 299792458
    ionc = r[n_r + n_s:]

    return dcb_r, dcb_s, ionc, stec_values, stec_gim, rms_station, rms_sat

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

    print('\n4. Similar Column Analysis:')
    similar_cols = find_similar_columns(bb)
    if len(similar_cols) > 0:
        print(' Found highly similar column pairs:')
        for col1, col2 in similar_cols:
            print(f' Columns {col1} and {col2} may be linearly dependent')
    else:
        print(' No highly similar columns found')

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

def find_similar_columns(bb):
    """Find similar columns in matrix"""
    m, n = bb.shape
    similar_cols = []
    max_cols_to_check = min(n, 1000)

    if n > max_cols_to_check:
        cols_to_check = np.sort(np.random.permutation(n)[:max_cols_to_check])
    else:
        cols_to_check = np.arange(n)

    bb_array = bb.toarray()
    for i in range(len(cols_to_check)):
        col_i = bb_array[:, cols_to_check[i]]
        norm_i = np.linalg.norm(col_i)
        if norm_i < 1e-10:
            continue
        col_i = col_i / norm_i

        for j in range(i+1, len(cols_to_check)):
            col_j = bb_array[:, cols_to_check[j]]
            norm_j = np.linalg.norm(col_j)
            if norm_j < 1e-10:
                continue
            col_j = col_j / norm_j

            cos_theta = abs(np.dot(col_i, col_j))
            if cos_theta > 0.9999:
                similar_cols.append([cols_to_check[i], cols_to_check[j]])

    return np.array(similar_cols)

def vtec2stec(ion_data, epoch, ippz_lat, ippz_lon, e):
    """Convert VTEC to STEC"""
    re = 6378137  # Earth radius in meters
    hion = 506700  # Ionosphere height

    ipp_lat = np.rad2deg(ippz_lat)
    ipp_lon = np.rad2deg(ippz_lon)

    if ipp_lon > 180:
        ipp_lon -= 360
    elif ipp_lon < -180:
        ipp_lon += 360

    unique_epochs = np.sort(np.unique(ion_data['Epoch']))
    prev_epochs = unique_epochs[unique_epochs <= epoch]

    if len(prev_epochs) == 0:
        print('Warning: No previous epoch available, using first available epoch')
        nearest_epoch = unique_epochs[0]
    else:
        nearest_epoch = prev_epochs[-1]

    epoch_data = ion_data[ion_data['Epoch'] == nearest_epoch]

    lat_grid = np.unique(epoch_data['Latitude'])
    lon_grid = np.unique(epoch_data['Longitude'])

    lat_idx = np.argsort(np.abs(lat_grid - ipp_lat))
    lon_idx = np.argsort(np.abs(lon_grid - ipp_lon))

    lat1, lat2 = lat_grid[lat_idx[:2]]
    lon1, lon2 = lon_grid[lon_idx[:2]]

    vtec11 = epoch_data.loc[(epoch_data['Latitude'] == lat1) & (epoch_data['Longitude'] == lon1), 'TEC'].values[0]
    vtec12 = epoch_data.loc[(epoch_data['Latitude'] == lat1) & (epoch_data['Longitude'] == lon2), 'TEC'].values[0]
    vtec21 = epoch_data.loc[(epoch_data['Latitude'] == lat2) & (epoch_data['Longitude'] == lon1), 'TEC'].values[0]
    vtec22 = epoch_data.loc[(epoch_data['Latitude'] == lat2) & (epoch_data['Longitude'] == lon2), 'TEC'].values[0]

    if any(v == 9999 for v in [vtec11, vtec12, vtec21, vtec22]):
        print('Warning: Invalid TEC values found at interpolation points')
        return np.nan

    p = (ipp_lon - lon1)/(lon2 - lon1)
    q = (ipp_lat - lat1)/(lat2 - lat1)

    vtec = (1-p)*(1-q)*vtec11 + p*(1-q)*vtec12 + (1-p)*q*vtec21 + p*q*vtec22

    z_ipp = asin(re*sin(0.9782*(pi/2 - e))/(re + hion))
    mf = cos(z_ipp)

    return vtec/10 * (1/mf)


""""""""""""""""""""""""""""""""""""""""""""
"""Main function to process multiple days"""
""""""""""""""""""""""""""""""""""""""""""""
calc_stec = False
calc_gim = False
signal_P1 = "C1W"
signal_P2 = "C2W"
signal_L1 = "L1C"
signal_L2 = "L2W"
time_window = 24 * 60  # 2 hours
window_num = 24 * 60 // time_window
base_path = 'E:/projects/FlexPower/M_DCB_data_2024_1_1'

# Load required data
sites_info = sio.loadmat(os.path.join(base_path, 'Sites_Info.mat'))['Sites_Info'][0]

input_path = os.path.join(base_path, f'M_P4_{signal_P1}_{signal_P2}')
output_path = os.path.join(base_path, f'M_Result_{signal_P1}_{signal_P2}')
i_ipath = os.path.join(base_path, 'i')

list_sp3 = [f for f in os.listdir(os.path.join(base_path, 'SP3')) if f.endswith('.mat')]
list_days = [d for d in os.listdir(input_path) if os.path.isdir(os.path.join(input_path, d))]

ion_data = None
if calc_gim:
    print('calc_gim')
    #ion_data = load_multiple_ionex(i_ipath)

print('Step five: DCB estimate day by day!')

for doy in list_days:
    index = find_sp3(list_sp3, doy)
    sate = sio.loadmat(os.path.join(base_path, 'SP3', list_sp3[index]))['sate']

    dcb_r_time = []
    dcb_s_time = []
    ionc_time = []
    rms_station_time = []
    rms_sat_time = []

    for j in range(1, window_num + 1):
        dcb_r, dcb_s, ionc, stec_values, stec_gim, rms_station, rms_sat = get_mdcb(
            doy, sites_info, sate, order=4, time_window=j,
            window_num=window_num, constra=2, ref_sat=2,
            input_path=input_path, ion_data=ion_data,
            calc_stec=calc_stec, calc_gim=calc_gim
        )

        dcb_r_time.append(dcb_r)
        dcb_s_time.append(dcb_s)
        ionc_time.append(ionc)
        rms_station_time.append(rms_station)
        rms_sat_time.append(rms_sat)

        print(f'Window {j}/{window_num} estimation complete!')

    if not os.path.exists(output_path):
        os.makedirs(output_path)

    save_data = {
        'DCB_R_time': np.array(dcb_r_time).T,
        'DCB_S_time': np.array(dcb_s_time).T,
        'IONC_time': np.array(ionc_time).T,
        'RMS_Station_time': np.array(rms_station_time).T,
        'RMS_Sat_time': np.array(rms_sat_time).T
    }

    if calc_stec:
        save_data['STEC_time'] = stec_values
    if calc_gim:
        save_data['STEC_GIM_time'] = stec_gim

    sio.savemat(os.path.join(output_path, f'M_R{doy}.mat'), save_data)
    print(f'DCB estimate of doy {doy} complete!')

print('Step five: completing!')