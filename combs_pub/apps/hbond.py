
from numba import jit
import numpy as np
import math

# X is [c_D_x, c_D_y, c_D_z, c_A_x, c_A_y, c_A_z]
@jit(nopython=True, cache=True, nogil=True)
def dist(X):
    d = 0.0
    tmp = X[0] - X[3]
    d += tmp * tmp
    tmp = X[1] - X[4]
    d += tmp * tmp
    tmp = X[2] - X[5]
    d += tmp * tmp
    return math.sqrt(d)


@jit(nopython=True, cache=True, nogil=True)
def dist_test_DA(X):
    d = dist(X)
    if d <= 3.9:
        return True
    else:
        return False

@jit(nopython=True, cache=True, nogil=True)
def dist_test_DS(X):
    d = dist(X)
    if d <= 4.1:
        return True
    else:
        return False


@jit(nopython=True, cache=True, nogil=True)
def dist_test_HA(X):
    d = dist(X)
    if d <= 2.5:   # Note this is too strict a cutoff for S acceptor!
        return True
    else:
        return False


@jit(nopython=True, cache=True, nogil=True)
def dist_test_HS(X):
    d = dist(X)
    if d <= 3.0:  # Satisfies S acceptor.  See Scientific Reports volume 6, Article number: 30369 (2016)
        return True
    else:
        return False


# X is [c_D_x, c_D_y, c_D_z, c_H_x, c_H_y, c_H_z, c_A_x, c_A_y, c_A_z]
# [atom1, origin atom, atom2]
@jit(nopython=True, cache=True, nogil=True)
def norm(X):
    s = 0.0
    for i in range(3):
        s += X[i] ** 2
    return math.sqrt(s)


@jit(nopython=True, cache=True, nogil=True)
def dot(X, Y):
    s = 0.0
    for i in range(3):
        s += X[i] * Y[i]
    return s


@jit(nopython=True, cache=True, nogil=True)
def ang(X, v1, v2):
    v1[0] = X[0] - X[3]
    v1[1] = X[1] - X[4]
    v1[2] = X[2] - X[5]

    v2[0] = X[6] - X[3]
    v2[1] = X[7] - X[4]
    v2[2] = X[8] - X[5]

    n1 = norm(v1)
    n2 = norm(v2)
    for i in range(3):
        v1[i] = v1[i] / n1
        v2[i] = v2[i] / n2
    return math.acos(dot(v1, v2))

# X is [c_D_x, c_D_y, c_D_z, c_H_x_1, c_H_y_1, c_H_z_1, c_A_x_1, c_A_y_1, c_A_z_1]

@jit(nopython=True, cache=True, nogil=True)
def ang_test(X, v1, v2):
    ang1 = ang(X, v1, v2)
    # if ang1 > np.pi / 2:
    crit = 80 / 180 * np.pi
    if ang1 > crit:
        return True
    else:
        return False


# X is [c_D_x, c_D_y, c_D_z, c_H_x_1, c_H_y_1, c_H_z_1, c_A_x_1, c_A_y_1, c_A_z_1, c_A_x_2, c_A_y_2, c_A_z_2]
@jit(nopython=True, cache=True, nogil=True)
def hbond_test(X, v1, v2, x):
    x[:3] = X[:3]
    x[3:6] = X[6:9]
    t1 = dist_test_DA(x[:6])
    t2 = dist_test_HA(X[3:9])
    t3 = ang_test(X[:9], v1, v2)

    x[:3] = X[9:]
    x[3:6] = X[6:9]
    x[6:] = X[:3]
    t4 = ang_test(x, v1, v2)  # A2-A1-D

    x[6:] = X[3:6]
    t5 = ang_test(x, v1, v2)  # A2-A1-H
    return t1 & t2 & t3 & t4 & t5


# X is [c_D_x, c_D_y, c_D_z, c_H_x_1, c_H_y_1, c_H_z_1, c_A_x_1, c_A_y_1, c_A_z_1, c_A_x_2, c_A_y_2, c_A_z_2]
@jit(nopython=True, cache=True, nogil=True)
def hbond_test_S_acceptor(X, v1, v2, x):
    x[:3] = X[:3]
    x[3:6] = X[6:9]
    t1 = dist_test_DS(x[:6])
    t2 = dist_test_HS(X[3:9])
    t3 = ang_test(X[:9], v1, v2)

    x[:3] = X[9:]
    x[3:6] = X[6:9]
    x[6:] = X[:3]
    t4 = ang_test(x, v1, v2)  # A2-A1-D

    x[6:] = X[3:6]
    t5 = ang_test(x, v1, v2)  # A2-A1-H
    return t1 & t2 & t3 & t4 & t5


# X is [ c_D_x, c_D_y, c_D_z, c_H_x_1, c_H_y_1, c_H_z_1, c_H_x_2, c_H_y_2, c_H_z_2,
# c_H_x_3, c_H_y_3, c_H_z_3, c_H_x_4, c_H_y_4, c_H_z_4,
# c_A_x_1, c_A_y_1, c_A_z_1, c_A_x_2, c_A_y_2, c_A_z_2]
@jit(nopython=True, cache=True, nogil=True)
def is_hbond(X):
    # Needs to treat D and each H and A, one at a time.
    # 4 possible Hs for each D, so
    # D H1 A1 A2
    # D H2 A1 A2
    # D H3 A1 A2
    # D H4 A1 A2
    # if any of these passes, just return True
    # need function that takes D H1 A1 A2 coords as input, returns True or False
    M = X.shape[0]
    ts = np.zeros(M)
    x = np.zeros(12)
    v1 = np.zeros(3)
    v2 = np.zeros(3)
    x_ = np.zeros(9)
    for i in range(M):
        if ~np.isnan(X[i, 3]):
            x[:3] = X[i, :3]
            x[3:6] = X[i, 3:6]
            x[6:] = X[i, 15:]
            t = hbond_test(x, v1, v2, x_)
            if t:
                ts[i] = True
            else:
                if ~np.isnan(X[i, 6]):
                    x[3:6] = X[i, 6:9]
                    t = hbond_test(x, v1, v2, x_)
                    if t:
                        ts[i] = True
                    else:
                        if ~np.isnan(X[i, 9]):
                            x[3:6] = X[i, 9:12]
                            t = hbond_test(x, v1, v2, x_)
                            if t:
                                ts[i] = True
                            else:
                                if ~np.isnan(X[i, 12]):
                                    x[3:6] = X[i, 12:15]
                                    t = hbond_test(x, v1, v2, x_)
                                    if t:
                                        ts[i] = True
                                    else:
                                        ts[i] = False
                                else:
                                    ts[i] = False
                        else:
                            ts[i] = False
                else:
                    ts[i] = False
        else:
            ts[i] = False
    return ts


# X is [ c_D_x, c_D_y, c_D_z, c_H_x_1, c_H_y_1, c_H_z_1, c_H_x_2, c_H_y_2, c_H_z_2,
# c_H_x_3, c_H_y_3, c_H_z_3, c_H_x_4, c_H_y_4, c_H_z_4,
# c_A_x_1, c_A_y_1, c_A_z_1, c_A_x_2, c_A_y_2, c_A_z_2]
@jit(nopython=True, cache=True, nogil=True)
def is_hbond_S_acceptor(X):
    # Needs to treat D and each H and A, one at a time.
    # 4 possible Hs for each D, so
    # D H1 A1 A2
    # D H2 A1 A2
    # D H3 A1 A2
    # D H4 A1 A2
    # if any of these passes, just return True
    # need function that takes D H1 A1 A2 coords as input, returns True or False
    M = X.shape[0]
    ts = np.zeros(M)
    x = np.zeros(12)
    v1 = np.zeros(3)
    v2 = np.zeros(3)
    x_ = np.zeros(9)
    for i in range(M):
        if ~np.isnan(X[i, 3]):
            x[:3] = X[i, :3]
            x[3:6] = X[i, 3:6]
            x[6:] = X[i, 15:]
            t = hbond_test_S_acceptor(x, v1, v2, x_)
            if t:
                ts[i] = True
            else:
                if ~np.isnan(X[i, 6]):
                    x[3:6] = X[i, 6:9]
                    t = hbond_test_S_acceptor(x, v1, v2, x_)
                    if t:
                        ts[i] = True
                    else:
                        if ~np.isnan(X[i, 9]):
                            x[3:6] = X[i, 9:12]
                            t = hbond_test_S_acceptor(x, v1, v2, x_)
                            if t:
                                ts[i] = True
                            else:
                                if ~np.isnan(X[i, 12]):
                                    x[3:6] = X[i, 12:15]
                                    t = hbond_test_S_acceptor(x, v1, v2, x_)
                                    if t:
                                        ts[i] = True
                                    else:
                                        ts[i] = False
                                else:
                                    ts[i] = False
                        else:
                            ts[i] = False
                else:
                    ts[i] = False
        else:
            ts[i] = False
    return ts
