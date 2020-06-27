__all__ = ['get_rot_trans']

import numpy as np

def get_rot_trans(mob_coords, targ_coords):
    mob_coords_com = mob_coords.mean(0)
    targ_coords_com = targ_coords.mean(0)
    mob_coords_cen = mob_coords - mob_coords_com
    targ_coords_cen = targ_coords - targ_coords_com
    cov_matrix = np.dot(mob_coords_cen.T, targ_coords_cen)
    U, S, Wt = np.linalg.svd(cov_matrix)
    R = np.dot(U, Wt)
    if np.linalg.det(R) < 0.:
        Wt[-1] *= -1
        R = np.dot(U, Wt)
    return R, mob_coords_com, targ_coords_com

def get_rot(mob_coords, targ_coords):
    cov_matrix = np.dot(mob_coords.T, targ_coords)
    U, S, Wt = np.linalg.svd(cov_matrix)
    R = np.dot(U, Wt)
    if np.linalg.det(R) < 0.:
        Wt[-1] *= -1
        R = np.dot(U, Wt)
    return R

def get_rotation_axis(R, normalized=False):
    """rotation matrix R"""
    u = np.zeros(3)
    u[0] = R[2, 1] - R[1, 2]
    u[1] = R[0, 2] - R[2, 0]
    u[2] = R[1, 0] - R[0, 1]
    if normalized:
        u = u / np.linalg.norm(u)
    return u

def get_rotation_angle(R):
    """rotation matrix R"""
    return np.arccos((np.trace(R) - 1) / 2)

def rotate_by_angle(u, angle):
    """Takes rotation axis u and angle to rotate about this axis, returns rotation matrix."""
    norm = np.linalg.norm(u)
    if norm != 1:
        u = u / norm
    R = np.zeros((3, 3))
    ux, uy, uz = u
    cos = np.cos
    sin = np.sin
    R[0, 0] = cos(angle) + ux**2 * (1 - cos(angle))
    R[0, 1] = ux*uy*(1-cos(angle)) - uz*sin(angle)
    R[0, 2] = ux*uz*(1-cos(angle)) + uy*sin(angle)
    R[1, 0] = uy*ux*(1-cos(angle)) + uz*sin(angle)
    R[1, 1] = cos(angle) + uy**2*(1 - cos(angle))
    R[1, 2] = uy*uz*(1-cos(angle)) - ux*sin(angle)
    R[2, 0] = uz*ux*(1-cos(angle)) - uy*sin(angle)
    R[2, 1] = uz*uy*(1-cos(angle)) + ux*sin(angle)
    R[2, 2] = cos(angle) + uz**2*(1 - cos(angle))
    return R