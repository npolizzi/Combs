__all__ = ['score_fit', 'rigid_body', 'score_fit_1']

import numpy as np


def rigid_body_highdim(mob_coord, x):
    dim = int(mob_coord.shape[1]/3)
    return np.dot(mob_coord + np.matlib.repmat(np.array([x[3], x[4], x[5]]), 1, dim),
                  np.kron(np.eye(dim), R(x[0], x[1], x[2])).T)


def rigid_body(mob_coord, x):
    return np.dot(mob_coord + x[3:], R(x[0], x[1], x[2]).T)


def score_fit(mob_coords_densities, x):
    return np.sum(-1 * density.logpdf(rigid_body_highdim(mob_coord, x))
                  for mob_coord, density in mob_coords_densities)


def score_fit_1(mob_coords_densities, x):
    return -1 * mob_coords_densities[1].logpdf(rigid_body_highdim(mob_coords_densities[0], x))


def sin(x):
    return np.sin(x)


def cos(x):
    return np.cos(x)


def R(phi, thet, psi):
    return np.array([[-sin(phi) * cos(thet) * sin(psi) + cos(phi) * cos(psi),
                      cos(phi) * cos(thet) * sin(psi) + sin(phi) * cos(psi), sin(thet) * sin(psi)],
                     [-sin(phi) * cos(thet) * cos(psi) - cos(phi) * sin(psi),
                      cos(phi) * cos(thet) * cos(psi) - sin(phi) * sin(psi), sin(thet) * cos(psi)],
                     [sin(phi) * sin(thet), -cos(phi) * sin(thet), cos(thet)]])


def R_vec(x):
    return np.array([[-sin(x[:, 0]) * cos(x[:, 1]) * sin(x[:, 2]) + cos(x[:, 0]) * cos(x[:, 2]),
                      cos(x[:, 0]) * cos(x[:, 1]) * sin(x[:, 2]) + sin(x[:, 0]) * cos(x[:, 2]), sin(x[:, 1]) * sin(x[:, 2])],
                     [-sin(x[:, 0]) * cos(x[:, 1]) * cos(x[:, 2]) - cos(x[:, 0]) * sin(x[:, 2]),
                      cos(x[:, 0]) * cos(x[:, 1]) * cos(x[:, 2]) - sin(x[:, 0]) * sin(x[:, 2]), sin(x[:, 1]) * cos(x[:, 2])],
                     [sin(x[:, 0]) * sin(x[:, 1]), -cos(x[:, 0]) * sin(x[:, 1]), cos(x[:, 1])]])

