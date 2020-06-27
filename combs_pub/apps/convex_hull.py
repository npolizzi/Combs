from scipy.spatial import Delaunay
import prody as pr
import numpy as np
from .pointTriangleDistance import pointTriangleDistance as distance
from numba import jit
import copy


@jit("f8(f8[:],f8[:],f8[:],f8[:])", nopython=True, cache=True)
def vol(a, b, c, d):
    M = np.zeros((3, 3))
    M[0, :] = np.subtract(a, d)
    M[1, :] = np.subtract(b, d)
    M[2, :] = np.subtract(c, d)
    return np.abs(np.linalg.det(M)) / 6


@jit("f8(f8[:,:])", nopython=True, cache=True)
def get_radius(points):
    a = np.linalg.norm(points[0] - points[1])
    a1 = np.linalg.norm(points[2] - points[3])
    b = np.linalg.norm(points[0] - points[2])
    b1 = np.linalg.norm(points[1] - points[3])
    c = np.linalg.norm(points[0] - points[3])
    c1 = np.linalg.norm(points[1] - points[2])
    p = (a * a1 + b * b1 + c * c1) / 2
    V = vol(points[0], points[1], points[2], points[3])
    if V > 0:
        return 1 / (6 * V) * np.sqrt(p * (p - a * a1) * (p - b * b1) * (p - c * c1))
    else:
        return np.inf


@jit("i4[:,:](f8[:,:], i4[:,:], f8)", nopython=True, cache=True)
def _calc_alpha_simplex(C, S, a):
    M = S.shape[0]
    N = S.shape[1]
    Result = np.zeros((M, N))
    j = 0
    for i in range(M):
        s = S[i, :]
        ps = C[s]
        r = get_radius(ps)
        if r < a:
            Result[j, :] = s
            j += 1
    return Result[:j, :].astype(np.int32)


combos = np.array([[0, 1, 2], [0, 1, 3], [0, 2, 3], [1, 2, 3]])


@jit("i4[:,:](i4[:,:], i8[:,:])", nopython=True, cache=True)
def make_simplex_set(S, combos):
    M = S.shape[0] * 4
    N = S.shape[1] - 1
    R = np.zeros((M, N), dtype=np.int32)
    for k, s in enumerate(range(0, M, 4)):
        for i in range(4):
            for j in range(3):
                R[s + i, j] = S[k, combos[i, j]]
    return R


def normal(ps):
    v1 = ps[1] - ps[0]
    v2 = ps[2] - ps[0]
    crossprod = np.cross(v1, v2)
    return crossprod / np.linalg.norm(crossprod)


class AlphaHull:

    def __init__(self, alpha=9):
        self.alpha = alpha
        self.hull = None
        self._hull = None
        self.tri = None
        self._tri = None
        self.coords = None
        self.simplices = None
        self.resindices = None

    def set_coords(self, pdb):
        type1 = isinstance(pdb, pr.atomic.selection.Selection)
        type2 = isinstance(pdb, pr.atomic.atomgroup.AtomGroup)
        if type1 or type2:
            self._set_coords(pdb)
        elif isinstance(pdb, np.ndarray):
            self.coords = pdb
        else:
            raise ValueError('*pdb* must be prody instance or numpy array')

    def _set_coords(self, pdb):
        """pdb is a prody object. pdb should have CB atoms where appropriate."""
        self.coords = pdb.select('name CB or (resname GLY and name CA)').getCoords()

    def set_tri(self):
        self.tri = Delaunay(self.coords)
        self._tri = copy.deepcopy(self.tri)

    def set_resindices(self, pdb):
        """pdb is a prody object. pdb should have CB atoms where appropriate."""
        self.resindices = pdb.select('name CB or (resname GLY and name CA)').getResindices()

    def calc_alpha_simplices(self):
        if self.tri is None:
            self.set_tri()
        self.tri.simplices.sort()
        self.tri.simplices = self.tri.simplices[self.tri.simplices[:, 0].argsort()]
        self.simplices = _calc_alpha_simplex(self.coords, self.tri.simplices, self.alpha)
        self._tri.simplices = self.simplices
        self._tri.neighbors = self.simplices


    def calc_hull(self):
        if self.simplices is None:
            self.calc_alpha_simplices()
        simpl_set = make_simplex_set(self.simplices, combos)
        un, ind, co = np.unique(simpl_set, axis=0,
                                return_counts=True, return_index=True)
        self.hull = np.array([simpl_set[i] for i in ind[co == 1]], dtype=np.int32)

    def pnts_in_hull(self, pnts):
        return self._tri.find_simplex(pnts) >= 0

    def get_pnt_distance(self, pnt):
        distances = []
        inout = self.pnts_in_hull(pnt)
        if inout:
            minmax = min
            inout = 1
        else:
            minmax = max
            inout = -1
        for i in range(len(self.hull)):
            distances.append(inout * distance(pnt, self.coords[self.hull[i]]))
        return minmax(distances)

    def get_pnts_distance(self, pnts):
        return [self.get_pnt_distance(pnt) for pnt in pnts]


def partition_res_by_burial(pdb_ala, alpha=9):
    """Returns residue indices of exposed, intermediate, and buried residues
    based on CA hull and CB hull."""
    ahull_ca = AlphaHull(alpha=alpha)
    ahull_ca.coords = pdb_ala.select('name CA').getCoords()
    ahull_ca.calc_hull()
    ahull_cb = AlphaHull(alpha=alpha)
    ahull_cb.set_coords(pdb_ala)
    ahull_cb.calc_hull()
    ahull_cb.set_resindices(pdb_ala)
    cb_in_ca_hull = ahull_ca.pnts_in_hull(ahull_cb.coords)
    resindices_cb_in_ca_hull = set(ahull_cb.resindices[cb_in_ca_hull])
    resindices_cb_hull = set(ahull_cb.resindices[np.unique(ahull_cb.hull)])
    resindices_not_cb_hull = set(ahull_cb.resindices) - resindices_cb_hull
    resindices_exposed = resindices_cb_hull - resindices_cb_in_ca_hull
    resindices_intermediate = resindices_cb_in_ca_hull - resindices_not_cb_hull
    resindices_buried = resindices_cb_in_ca_hull - resindices_intermediate
    res_ = resindices_not_cb_hull - resindices_buried
    resindices_intermediate |= res_
    return resindices_exposed, resindices_intermediate, resindices_buried

