import os
import numpy as np
import scipy.io as sio


def interp_lag(x, y, x0):
    """Manual implementation of Lagrange interpolation"""
    n = len(x)
    y0 = np.zeros_like(x0)

    for k in range(n):
        t = 1
        for i in range(n):
            if i != k:
                t = t * (x0 - x[i]) / (x[k] - x[i])
        y0 = y0 + t * y[k]

    return y0


def interplotation(x1, y1, z1, x2, y2, z2, x3, y3, z3):
    """Interpolation function for satellite coordinates"""
    interp_x2 = np.zeros((2880, 32))
    interp_y2 = np.zeros((2880, 32))
    interp_z2 = np.zeros((2880, 32))

    x2 = np.vstack((x1[92:96, :], x2, x3[0:5, :]))
    y2 = np.vstack((y1[92:96, :], y2, y3[0:5, :]))
    z2 = np.vstack((z1[92:96, :], z2, z3[0:5, :]))

    m_t = np.linspace(-120, 3000, 105)

    for i in range(32):
        for j in range(96):
            tt = m_t[j:j + 10]
            x = x2[j:j + 10, i]
            y = y2[j:j + 10, i]
            z = z2[j:j + 10, i]

            t0 = np.linspace(m_t[j + 4], m_t[j + 5] - 1, 30)

            interp_x2[30 * j:30 * (j + 1), i] = interp_lag(tt, x, t0)
            interp_y2[30 * j:30 * (j + 1), i] = interp_lag(tt, y, t0)
            interp_z2[30 * j:30 * (j + 1), i] = interp_lag(tt, z, t0)

    return interp_x2, interp_y2, interp_z2


def r_sp3(path):
    """Read SP3 file and extract coordinates"""
    X = np.zeros((96, 32))
    Y = np.zeros((96, 32))
    Z = np.zeros((96, 32))

    with open(path, 'r') as f:
        ep = 0
        for line in f:
            if line.startswith('*'):
                h = int(line[14:16])
                m = int(line[17:19])
                ep = h * 4 + round(m / 15) + 1
                continue

            if ep > 96:
                # 检查X有多少行
                Xep = X.shape[0]
                if ep-Xep == 1:
                    X = np.vstack((X, np.zeros((1, 32))))
                    Y = np.vstack((Y, np.zeros((1, 32))))
                    Z = np.vstack((Z, np.zeros((1, 32))))

            if len(line) > 1 and line.startswith('PG'):
                sv = int(line[2:4])
                X[ep - 1, sv - 1] = float(line[4:18])
                Y[ep - 1, sv - 1] = float(line[18:32])
                Z[ep - 1, sv - 1] = float(line[32:46])

    return X, Y, Z


def GwToDoy(GN, Day):
    """Convert GPS week and day to year and day of year"""
    import datetime
    gps_epoch = datetime.datetime(1980, 1, 6)
    current_date = gps_epoch + datetime.timedelta(weeks=GN, days=Day)
    year = current_date.year
    doy = current_date.timetuple().tm_yday
    return year, doy


def read_sp3(s_ipath, s_opath):
    """Main function to process SP3 files"""
    list_obs = sorted([f for f in os.listdir(s_ipath) if f.endswith(('.sp3', '.SP3'))])

    if len(list_obs) < 3:
        raise ValueError('Need at least three SP3 files; see Readme')

    os.makedirs(s_opath, exist_ok=True)

    for i in range(len(list_obs) - 2):
        pr_obs = os.path.join(s_ipath, list_obs[i])
        cu_obs = os.path.join(s_ipath, list_obs[i + 1])
        nx_obs = os.path.join(s_ipath, list_obs[i + 2])

        x1, y1, z1 = r_sp3(pr_obs)
        x2, y2, z2 = r_sp3(cu_obs)
        x3, y3, z3 = r_sp3(nx_obs)

        sate_x, sate_y, sate_z = interplotation(x1, y1, z1, x2, y2, z2, x3, y3, z3)
        sate = {'x': sate_x, 'y': sate_y, 'z': sate_z}

        filename = list_obs[i + 1]
        if filename.endswith('.sp3'):
            GN = int(filename[3:7])
            Day = int(filename[7])
            y, doy = GwToDoy(GN, Day)
        else:  # .SP3
            y = int(filename[11:15])
            doy = int(filename[15:18])

        outfile = f"{y:04d}{doy:03d}sp3.mat"
        sio.savemat(os.path.join(s_opath, outfile), {'sate': sate})


""""""""""""""""""""""""""""""""""""""""""""
"""Main function to process multiple days"""
""""""""""""""""""""""""""""""""""""""""""""
base_path = 'E:\\projects\\FlexPower\\M_DCB_data_2024_1_1'
s_ipath = os.path.join(base_path, 'sp3file')
s_opath = os.path.join(base_path, 'SP3')

print('Step two: read SP3 files!')
read_sp3(s_ipath, s_opath)
print('Step two: completing!')