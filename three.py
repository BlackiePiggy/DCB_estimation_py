import os
import numpy as np
import scipy.io as sio
from math import pi, atan, sqrt, sin, cos

def Get_P12(path_obs, path_sp3, Sites_Info, lim, flag, f1, f2, output_path):
    list_obs = [f for f in os.listdir(path_obs) if f.endswith('.mat')]
    list_sp3 = [f for f in os.listdir(path_sp3) if f.endswith('.mat')]

    # MATLAB的struct数据在Python中以字典形式加载
    Coor = Sites_Info['coor'][0]
    stations = Sites_Info['name'][0]
    doys = Sites_Info['doy'][0]

    if Coor is None or stations is None or doys is None:
        print("Warning: Required fields not found in Sites_Info")
        print("Available fields:", Sites_Info.keys())
        return

    if not os.path.exists(output_path):
        os.makedirs(output_path)

    for station_name in list_obs:
        obs_data = sio.loadmat(os.path.join(path_obs, station_name))
        obs = obs_data['obs'][0]
        site = station_name[:4]
        doy = station_name[4:9]

        indices = doys == int(doy)
        index = np.where(stations[0] == np.array([site], dtype='<U4'))[0][0]
        sx, sy, sz = Coor[index]

        if sx == 0 and sy == 0 and sz == 0:
            continue

        sp3index = find_sp3(list_sp3, doy)
        sate = sio.loadmat(os.path.join(path_sp3, list_sp3[sp3index]))['sate']
        obs = cutobs(sate, sx, sy, sz, obs, lim)

        if np.all(obs['P1'][0] == 0) or np.all(obs['P2'][0] == 0):
            continue

        P4 = prepro(obs, f1, f2)

        if flag == 0:
            save_path = os.path.join(output_path, doy)
        else:
            save_path = os.path.join(output_path, site)

        if not os.path.exists(save_path):
            os.makedirs(save_path)

        filenameP4 = os.path.join(save_path, f"{site}{doy}P4.mat")
        sio.savemat(filenameP4, {'P4': P4})


def cutobs(sate, sx, sy, sz, obs, lim):
    x, y, z = sate[0][0][0], sate[0][0][1], sate[0][0][2]
    for i in range(32):
        for j in range(2880):
            if (obs['L1'][0][j, i] == 0 or obs['L2'][0][j, i] == 0 or
                    obs['P1'][0][j, i] == 0 or obs['P2'][0][j, i] == 0):
                obs['P1'][0][j, i] = obs['P2'][0][j, i] = obs['L1'][0][j, i] = obs['L2'][0][j, i] = 0
                continue

            el, _ = Get_EA(sx, sy, sz, x[j, i] * 1000, y[j, i] * 1000, z[j, i] * 1000)
            if el < lim:
                obs['P1'][0][j, i] = obs['P2'][0][j, i] = obs['L1'][0][j, i] = obs['L2'][0][j, i] = 0
    return obs


def prepro(obs, f1, f2):
    P4 = np.zeros((2880, 32))
    L4 = np.zeros((2880, 32))
    c = 299792458
    lamda_w = c / (f1 - f2)

    L6 = lamda_w * (obs['L1'][0] - obs['L2'][0]) - (f1 * obs['P1'][0] + f2 * obs['P2'][0]) / (f1 + f2)
    Li = obs['L1'][0] - f1 * obs['L2'][0] / f2
    Nw = L6 / lamda_w

    for i in range(32):
        arc = Get_arc(L6[:, i])
        if len(arc) == 0:
            continue

        # Delete arcs less than 10 epochs
        arc_d = []
        for j in range(len(arc)):
            n_epoch = arc[j, 1] - arc[j, 0]
            if n_epoch < 10:
                for k in range(arc[j, 0], arc[j, 1] + 1):
                    obs['P1'][0][k, i] = obs['P2'][0][k, i] = obs['L1'][0][k, i] = obs['L2'][0][k, i] = 0
                    L6[k, i] = Nw[k, i] = Li[k, i] = 0
                arc_d.append(j)

        arc = np.delete(arc, arc_d, axis=0)
        if len(arc) == 0:
            continue

        j = 0
        while j < len(arc):
            e = arc[j, 0]
            while True:
                if e + 1 >= arc[j, 1]:
                    break

                fir, sec, thi = Nw[e, i], Nw[e + 1, i], Nw[e + 2, i]
                firl, secl, thil = Li[e, i], Li[e + 1, i], Li[e + 2, i]
                sub, sub2 = abs(fir - sec), abs(sec - thi)
                subl, subl2 = abs(firl - secl), abs(secl - thil)

                if sub > 1 or sub2 > 1 or subl > 1 or subl2 > 1:
                    L6[e, i] = obs['L1'][0][e, i] = obs['L2'][0][e, i] = obs['P1'][0][e, i] = obs['P2'][0][e, i] = Nw[e, i] = Li[
                        e, i] = 0
                    e += 1
                    arc[j, 0] = e
                else:
                    arc[j, 0] = e
                    break

            if arc[j, 1] - arc[j, 0] < 10:
                for k in range(arc[j, 0], arc[j, 1] + 1):
                    obs['P1'][0][k, i] = obs['P2'][0][k, i] = obs['L1'][0][k, i] = obs['L2'][0][k, i] = 0
                    L6[k, i] = Nw[k, i] = Li[k, i] = 0
                arc = np.delete(arc, j, axis=0)
                continue

            ave_N = np.zeros(arc[j, 1] - arc[j, 0] + 1)
            sigma2 = np.zeros_like(ave_N)
            sigma = np.zeros_like(ave_N)
            ave_N[0] = Nw[arc[j, 0], i]

            count = 1
            for k in range(arc[j, 0] + 1, arc[j, 1]):
                ave_N[count] = ave_N[count - 1] + (Nw[k, i] - ave_N[count - 1]) / (count+1)
                sigma2[count] = sigma2[count - 1] + ((Nw[k, i] - ave_N[count - 1]) ** 2 - sigma2[count - 1]) / (count+1)
                sigma[count] = sqrt(sigma2[count])
                T = abs(Nw[k + 1, i] - ave_N[count])
                I1 = abs(Li[k + 1, i] - Li[k, i])

                if T < 4 * sigma[count] and I1 < 0.28:
                    count += 1
                    continue
                else:
                    if k + 1 == arc[j, 1]:
                        if k + 1 - arc[j, 0] > 10:
                            L6[k + 1, i] = obs['P1'][0][k + 1, i] = obs['P2'][0][k + 1, i] = 0
                            obs['L1'][0][k + 1, i] = obs['L2'][0][k + 1, i] = Nw[k + 1, i] = Li[k, i] = 0
                            arc[j, 1] = k
                        else:
                            for l in range(arc[j, 0], k + 2):
                                L6[l, i] = obs['P1'][0][l, i] = obs['P2'][0][l, i] = 0
                                obs['L1'][0][l, i] = obs['L2'][0][l, i] = Nw[l, i] = Li[k, i] = 0
                            arc = np.delete(arc, j, axis=0)
                            j -= 1
                        break

                    I2 = abs(Li[k + 2, i] - Li[k + 1, i])
                    if abs(Nw[k + 2, i] - Nw[k + 1, i]) < 1 and I2 < 1:
                        if k + 1 - arc[j, 0] > 10:
                            new_arc = np.array([[arc[j, 0], k], [k + 1, arc[j, 1]]])
                            arc = np.insert(arc, j + 1, new_arc[1], axis=0)
                            arc[j, 1] = k
                        else:
                            for l in range(arc[j, 0], k + 1):
                                L6[l, i] = obs['P1'][0][l, i] = obs['P2'][0][l, i] = 0
                                obs['L1'][0][l, i] = obs['L2'][0][l, i] = Nw[l, i] = Li[k, i] = 0
                            arc[j, 0] = k + 1
                            j -= 1
                    else:
                        if k + 1 - arc[j, 0] > 10:
                            L6[k + 1, i] = obs['P1'][0][k + 1, i] = obs['P2'][0][k + 1, i] = 0
                            obs['L1'][0][k + 1, i] = obs['L2'][0][k + 1, i] = Nw[k + 1, i] = Li[k, i] = 0
                            new_arc = np.array([[arc[j, 0], k], [k + 2, arc[j, 1]]])
                            arc = np.insert(arc, j + 1, new_arc[1], axis=0)
                            arc[j, 1] = k
                        else:
                            for l in range(arc[j, 0], k + 2):
                                L6[l, i] = obs['P1'][0][l, i] = obs['P2'][0][l, i] = 0
                                obs['L1'][0][l, i] = obs['L2'][0][l, i] = Nw[l, i] = Li[k, i] = 0
                            arc[j, 0] = k + 2
                            j -= 1
                    break
            j += 1

        P4[:, i] = obs['P1'][0][:, i] - obs['P2'][0][:, i]
        L4[:, i] = (c / f1) * obs['L1'][0][:, i] - (c / f2) * obs['L2'][0][:, i]

        # Smoothing
        for j in range(len(arc)):
            t = 2
            for k in range(arc[j, 0] + 1, arc[j, 1] + 1):
                P4[k, i] = P4[k, i] / t + (P4[k - 1, i] + L4[k - 1, i] - L4[k, i]) * (t - 1) / t
                t += 1
            P4[arc[j, 0]:arc[j, 0] + 5, i] = 0

        # Remove bad P4
        arc = Get_arc(P4[:, i])
        for j in range(len(arc)):
            ave = np.mean(P4[arc[j, 0]:arc[j, 1] + 1, i])
            if abs(ave) > 10:
                P4[arc[j, 0]:arc[j, 1] + 1, i] = 0

    return P4


def find_sp3(list_sp3, doy):
    doys = [int(name[2:7]) for name in list_sp3]
    return doys.index(int(doy))


def Get_arc(array):
    arc = []
    for i in range(len(array)):
        if i == len(array) - 1:
            if array[i] != 0:
                arc.append(i)
            continue

        if i == 0 and array[i] != 0:
            arc.append(i)

        if array[i] == 0 and array[i + 1] != 0:
            arc.append(i + 1)
            continue

        if array[i] != 0 and array[i + 1] == 0:
            arc.append(i)
            continue

    if len(arc) == 0:
        return np.array([])

    return np.array(arc).reshape(-1, 2)


def Get_EA(sx, sy, sz, x, y, z):
    sb, sl, _ = XYZtoBLH(sx, sy, sz)
    T = np.array([
        [-sin(sb) * cos(sl), -sin(sb) * sin(sl), cos(sb)],
        [-sin(sl), cos(sl), 0],
        [cos(sb) * cos(sl), cos(sb) * sin(sl), sin(sb)]
    ])

    deta_xyz = np.array([x - sx, y - sy, z - sz])
    NEU = T @ deta_xyz

    E = atan(NEU[2] / sqrt(NEU[0] ** 2 + NEU[1] ** 2))
    A = atan(abs(NEU[1] / NEU[0]))

    if NEU[0] > 0:
        if NEU[1] <= 0:
            A = 2 * pi - A
    else:
        if NEU[1] > 0:
            A = pi - A
        else:
            A = pi + A

    return E, A


def XYZtoBLH(X, Y, Z):
    a = 6378137
    e2 = 0.0066943799013

    L = atan(abs(Y / X))
    if Y > 0:
        if X <= 0:
            L = pi - L
    else:
        if X > 0:
            L = 2 * pi - L
        else:
            L = pi + L

    B0 = atan(Z / sqrt(X ** 2 + Y ** 2))
    while True:
        N = a / sqrt(1 - e2 * sin(B0) * sin(B0))
        H = Z / sin(B0) - N * (1 - e2)
        B = atan(Z * (N + H) / (sqrt(X ** 2 + Y ** 2) * (N * (1 - e2) + H)))
        if abs(B - B0) < 1e-6:
            break
        B0 = B

    N = a / sqrt(1 - e2 * sin(B) * sin(B))
    H = Z / sin(B) - N * (1 - e2)

    return B, L, H

""""""""""""""""""""""""""""""""""""""""""""
"""Main function to process multiple days"""
""""""""""""""""""""""""""""""""""""""""""""
base_path = 'E:\\projects\\FlexPower\\M_DCB_data_2024_1_1'
lim = 10
order = 4
signal_P1 = "C1W"
signal_P2 = "C2W"
signal_L1 = "L1C"
signal_L2 = "L2W"

Sites_Info = sio.loadmat(os.path.join(base_path, 'Sites_Info.mat'))
Sites_Info = Sites_Info['Sites_Info'][0]

# 假设 Sites_Info 已经加载
r_opath = os.path.join(base_path, f'o_{signal_P1}_{signal_P2}')
s_opath = os.path.join(base_path, 'SP3')
p4_opath = os.path.join(base_path, f'M_P4_{signal_P1}_{signal_P2}')

print('Step four: data pre-processing and get ionosphere observations !')
f1 = 1575.42e6
f2 = 1227.6e6
Get_P12(r_opath, s_opath, Sites_Info, lim*pi/180, 0, f1, f2, p4_opath)
print('Step four: completing !')