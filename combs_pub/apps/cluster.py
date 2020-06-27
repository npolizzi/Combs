import numpy as np
from scipy.sparse import csr_matrix
from numba import jit
import prody as pr
from pandas import DataFrame
from .transformation import get_rot_trans
from os import makedirs, listdir
from ..analysis.analyze import Analyze
from sklearn.neighbors import NearestNeighbors

class Cluster:

    def __init__(self, **kwargs):

        self.bb_sel = kwargs.get('bb_sel', [(10, ['N', 'CA', 'C'])])
        self.cluster_type = kwargs.get('cluster_type', 'unlabeled')
        self.ifg_dict = kwargs.get('ifg_dict', None)
        self.num_atoms_ifg = len([v for v in self.ifg_dict.values()][0]) \
            if self.ifg_dict else None
        self.rmsd_cutoff = kwargs.get('rmsd_cutoff', 0.5)
        od = kwargs.get('rmsd_mat_outdir', '.')
        self.rmsd_mat_outdir = od if od[-1] == '/' else od + '/'
        od = kwargs.get('clusters_outdir', '.')
        self.clusters_outdir = od if od[-1] == '/' else od + '/'
        od = kwargs.get('pickle_file_outdir', '.')
        self.pickle_file_outdir = od if od[-1] == '/' else od + '/'
        self.pdbs = None
        self.resname = None
        self.pdb_coords = list()
        self.pdbs_errorfree = list()
        self.rmsd_mat = None
        self.adj_mat = None
        self.mems = None
        self.cents = None
        self._square = False
        self._adj_mat = False
        self._data_set = False
        self._ifg_count = list()
        self._vdm_count = list()
        self._query_name = list()
        self._centroid = list()
        self._cluster_num = list()
        self._cluster_size = list()
        self._rmsd_from_centroid = list()
        self._resname = list()
        self.__resname = list()
        self._cluster_type = list()
        self.df = None
        self.query_names_errorfree = list()
        self.query_names = kwargs.get('query_names')

    def set_pdbs(self, pdbs):
        """Load pdbs into the Cluster object to be clustered.
        VdMs must be all one residue type."""
        self.pdbs = pdbs
        resname = set([pdb.select('chain X resnum 10 name CA').getResnames()[0]
                       for pdb in pdbs])
        if len(resname) == 1:
            self.resname = resname.pop()

    def _set_coords(self, pdb, query_name=None):
        """Grabs coordinates of atoms in bb selection and iFG dict."""
        coords = list()
        try:
            for resnum, bb_atoms in self.bb_sel:
                for bb_atom in bb_atoms:
                    coords.append(pdb.select('chain X resnum ' + str(resnum)
                                             + ' name ' + bb_atom).getCoords()[0])
            resname = set(pdb.select('chain Y resnum 10').getResnames()).pop()
            for ifg_atom in self.ifg_dict[resname]:
                coords.append(pdb.select('chain Y resnum 10 name '
                                         + ifg_atom).getCoords()[0])
            self.pdb_coords.append(coords)
            self.pdbs_errorfree.append(pdb)
            if query_name is not None:
                self.query_names_errorfree.append(query_name)
        except AttributeError:
            pass

    def set_coords(self):
        """Grabs coords for every vdM in pdbs."""
        if self.query_names is not None:
            for pdb, query_name in zip(self.pdbs, self.query_names):
                self._set_coords(pdb, query_name)
                self.__resname.append(pdb.select('chain X resnum 10 name CA').getResnames()[0])
        else:
            for pdb in self.pdbs:
                self._set_coords(pdb)
                self.__resname.append(pdb.select('chain X resnum 10 name CA').getResnames()[0])
        self.pdb_coords = np.array(self.pdb_coords, dtype='float32')

    def make_pairwise_rmsd_mat(self):
        """Uses C-compiled numba code for fast pairwise superposition
        and RMSD calculation of coords (all against all)."""
        assert isinstance(self.pdb_coords, np.ndarray), 'PDB coords must be ' \
                                                       'numpy array'
        assert self.pdb_coords.dtype == 'float32', 'PDB coords must ' \
                                                   'be dtype of float32'
        self.rmsd_mat = _make_pairwise_rmsd_mat(self.pdb_coords)

    def save_rmsd_mat(self):
        """Saves RMSD matrix (lower triangular) as a numpy array.  This also
        saves a text file that lists the file names in the order of the matrix
        indices."""
        outpath = self.rmsd_mat_outdir + self.cluster_type + '/' + self.resname + '/'
        try:
            makedirs(outpath)
        except FileExistsError:
            pass

        np.save(outpath + 'rmsd_mat_half_'
                + self.resname, self.rmsd_mat)

        with open(outpath + 'filenames.txt', 'w') as outfile:
            outfile.write('bb_sel = ' + ' '.join(str(t) for t in self.bb_sel) + '\n')
            outfile.write('ifg_dict = ' + ', '.join(key + ': ' + str(val) for key, val
                                                    in self.ifg_dict.items()) + '\n')
            for pdb in self.pdbs_errorfree:
                outfile.write(str(pdb).split()[-1] + '\n')

    @staticmethod
    def greedy(adj_mat):
        """Takes an adjacency matrix as input.
            All values of adj_mat are 1 or 0:  1 if <= to cutoff, 0 if > cutoff.
            Can generate adj_mat from data in column format with:
            sklearn.neighbors.NearestNeighbors(metric='euclidean',
            radius=cutoff).fit(data).radius_neighbors_graph(data)"""

        if not isinstance(adj_mat, csr_matrix):
            try:
                adj_mat = csr_matrix(adj_mat)
            except:
                print('adj_mat distance matrix must be scipy csr_matrix '
                      '(or able to convert to one)')
                return

        assert adj_mat.shape[0] == adj_mat.shape[1], 'Distance matrix is not square.'

        all_mems = []
        cents = []
        indices = np.arange(adj_mat.shape[0])

        while adj_mat.shape[0] > 0:
            cent = adj_mat.sum(axis=1).argmax()
            cents.append(indices[cent])
            row = adj_mat.getrow(cent)
            tf = ~row.toarray().astype(bool)[0]
            mems = indices[~tf]
            all_mems.append(mems)
            indices = indices[tf]
            adj_mat = adj_mat[tf][:, tf]

        return all_mems, cents

    def make_square(self):
        self.rmsd_mat = self.rmsd_mat.T + self.rmsd_mat
        self._square = True

    def make_adj_mat_no_superpose(self):
        num_atoms = len(self.pdb_coords[0])
        nbrs = NearestNeighbors(radius=self.rmsd_cutoff * np.sqrt(num_atoms))
        nbrs_coords = np.array([s.getCoords().flatten() for s in self.pdb_coords])
        nbrs.fit(nbrs_coords)
        self.adj_mat = nbrs.radius_neighbors_graph(nbrs_coords)
        self._adj_mat = True

    def make_adj_mat(self):
        """Makes an adjacency matrix from the RMSD matrix"""
        self.adj_mat = np.zeros(self.rmsd_mat.shape)
        self.adj_mat[self.rmsd_mat <= self.rmsd_cutoff] = 1
        self.adj_mat = csr_matrix(self.adj_mat)
        self._adj_mat = True

    def cluster(self):
        """Performs greedy clustering of the RMSD matrix with a given
        RMSD cutoff (Default cutoff is 0.5 A)."""
        assert self.rmsd_mat is not None, 'Must create rmsd matrix first with ' \
                                          'make_pairwise_rmsd_mat()'
        if not self._square:
            self.make_square()

        if not self._adj_mat:
            self.make_adj_mat()

        self.mems, self.cents = self.greedy(self.adj_mat)

    def print_cluster(self, cluster_number, label='', outpath=None):
        """Prints PDBs of a cluster after superposition of the backbone (bb_sel)
        onto that of the cluster centroid.  The backbone of the cluster centroid
        is itself superposed onto that of the largest cluster's centroid."""
        if not self._data_set:
            self.set_data()

        if not outpath:
            outpath = self.clusters_outdir + self.cluster_type + '/' \
                      + self.resname + '/' + str(cluster_number) + '/'

        try:
            makedirs(outpath)
        except FileExistsError:
            pass

        n = self.num_atoms_ifg
        cluster_index = cluster_number - 1
        cent = self.cents[cluster_index]
        mems = self.mems[cluster_index]

        # Align backbone of cluster centroid to backbone of centroid of largest cluster.
        R, m_com, t_com = get_rot_trans(self.pdb_coords[cent][:-n],
                                        self.pdb_coords[self.cents[0]][:-n])
        cent_coords = np.dot((self.pdb_coords[cent] - m_com), R) + t_com

        for mem in mems:
            filename = str(self.pdbs_errorfree[mem]).split()[-1].split('_')
            ifg_count = filename[1]
            vdm_count = filename[3]
            if self.query_names_errorfree:
                query_name = self.query_names_errorfree[mem]
            else:
                query_name = '_'.join(filename[4:])
            R, m_com, t_com = get_rot_trans(self.pdb_coords[mem], cent_coords)
            pdb = self.pdbs_errorfree[mem].copy()
            pdb_coords = pdb.getCoords()
            coords_transformed = np.dot((pdb_coords - m_com), R) + t_com
            pdb.setCoords(coords_transformed)
            is_cent = '_centroid' if mem == cent else ''
            pr.writePDB(outpath + 'cluster_' + str(cluster_number) + '_iFG_'
                        + ifg_count + '_vdM_'
                        + vdm_count + '_' + query_name
                        + is_cent + label + '.pdb.gz', pdb)

    def print_cluster_from_dataframe(self, df, cluster_num, paths_to_vdms, label='', outpath='./', **kwargs):

        cluster_type = kwargs['cluster_type']
        if cluster_type == 'SC':
            bb_sel = [(10, ['N', 'CA', 'C'])]

        if cluster_type == 'HNCA':
            bb_sel = [(9, ['C']), (10, ['N', 'CA'])]

        if cluster_type == 'CO':
            bb_sel = [(10, ['CA', 'C']), (11, ['N'])]

        if cluster_type == 'PHI_PSI':
            bb_sel = [(9, ['C']), (10, ['N', 'CA', 'C']), (11, ['N'])]

        self.bb_sel = bb_sel
        self.cluster_type = cluster_type
        self.ifg_dict = kwargs.get('ifg_dict', None)
        self.num_atoms_ifg = len([v for v in self.ifg_dict.values()][0]) \
            if self.ifg_dict else None
        self.rmsd_cutoff = kwargs.get('rmsd_cutoff', 0.5)

        an = Analyze('', './')
        df_clu = df[df['cluster_number'] == cluster_num]
        query_names = set(df_clu.query_name)
        pdbs = list()
        for query_name in query_names:
            df_clu_qn = df_clu[df_clu['query_name'] == query_name]
            for path in paths_to_vdms:
                try:
                    pdbs_ = an.parse_vdms(df_clu_qn, path_to_vdm=path)
                    pdbs.extend(pdbs_)
                except OSError:
                    pass

        self.set_pdbs(pdbs)
        self.set_coords()
        self.make_pairwise_rmsd_mat()
        self.cluster()
        self.set_data()
        self.print_cluster(1, label=label, outpath=outpath)

    def print_clusters(self, clusters):
        """Prints all clusters in the list *clusters*"""
        for cluster_number in clusters:
            self.print_cluster(cluster_number)

    def set_data(self):
        """Creates cluster data (columns) for creation of a pandas dataframe."""
        for i, (cent, mems) in enumerate(zip(self.cents, self.mems)):
            cluster_number = i + 1
            cluster_size = len(mems)
            for mem in mems:
                filename = str(self.pdbs_errorfree[mem]).split()[-1].split('_')
                self._ifg_count.append(filename[1])
                self._vdm_count.append(filename[3])
                if self.query_names_errorfree:
                    query_name = self.query_names_errorfree[mem]
                else:
                    query_name = '_'.join(filename[4:])
                self._query_name.append(query_name)
                centroid = True if mem == cent else False
                self._centroid.append(centroid)
                self._cluster_num.append(cluster_number)
                self._cluster_size.append(cluster_size)
                rmsd_from_centroid = np.round(self.rmsd_mat[mem][cent], 2)
                self._rmsd_from_centroid.append(rmsd_from_centroid)
        self._data_set = True

    def make_dataframe(self):
        """Creates a pandas dataframe of cluster data."""
        if not self._data_set:
            self.set_data()

        if self.resname is not None:
            self._resname = [self.resname] * len(self._ifg_count)
        else:
            self._resname = self.__resname
        self._cluster_type = [self.cluster_type] * len(self._ifg_count)

        all_attrs = [self._ifg_count, self._vdm_count, self._resname,
                     self._query_name, self._centroid, self._cluster_num,
                     self._cluster_size, self._rmsd_from_centroid,
                     self._cluster_type]

        assert len(set(map(len, all_attrs))) == 1, 'all attributes must be same length'

        self.df = DataFrame(list(zip(*all_attrs)),
                            columns=['iFG_count', 'vdM_count', 'resname',
                                     'query_name', 'centroid', 'cluster_number',
                                     'cluster_size', 'rmsd_from_centroid',
                                     'cluster_type'])

    def save_dataframe(self):
        """Saves cluster dataframe to a pickle file."""
        outpath = self.pickle_file_outdir + self.cluster_type + '/'

        try:
            makedirs(outpath)
        except FileExistsError:
            pass

        self.df.to_pickle(outpath + self.resname + '.pkl')

    def run_full_protocol(self, pdbs, clusters=range(1, 21)):
        """Runs a series of methods that will cluster the
        PDBs and output a pandas dataframe, RMSD matrix, and
        PDBs of top 20 clusters."""
        print('clustering ' + self.cluster_type + ':')
        self.set_pdbs(pdbs)
        print('setting PDB coords...')
        self.set_coords()
        print('constructing RMSD matrix...')
        self.make_pairwise_rmsd_mat()
        self.save_rmsd_mat()
        print('greedy clustering...')
        self.cluster()
        num_clusts = len(self.cents)
        if len(clusters) > num_clusts:
            clusters = range(1, num_clusts + 1)
        print('making dataframe...')
        self.make_dataframe()
        self.save_dataframe()
        self.print_clusters(clusters)
        
    def run_min_protocol(self, pdbs):
        """Runs a series of methods that will cluster the
        PDBs and output a pandas dataframe, RMSD matrix, and
        PDBs of top 20 clusters."""
        print('clustering ' + self.cluster_type + ':')
        self.set_pdbs(pdbs)
        print('setting PDB coords...')
        self.set_coords()
        print('constructing RMSD matrix...')
        self.make_pairwise_rmsd_mat()
        print('greedy clustering...')
        self.cluster()
        print('making dataframe...')
        self.make_dataframe()
        self.save_dataframe()
        print('done')

    def run_min_protocol_no_superpose(self, pdbs):
        """Runs a series of methods that will cluster the
        PDBs and output a pandas dataframe, RMSD matrix, and
        PDBs of top 20 clusters."""
        print('clustering ' + self.cluster_type + ':')
        self.set_pdbs(pdbs)
        print('setting PDB coords...')
        self.set_coords()
        print('constructing adj matrix...')
        self.make_adj_mat_no_superpose()
        print('greedy clustering...')
        self.cluster()
        print('making dataframe...')
        self.make_dataframe()
        self.save_dataframe()
        print('done')


@jit("f4[:,:](f4[:,:,:])", nopython=True, cache=True)
def _make_pairwise_rmsd_mat(X):
    M = X.shape[0]
    N = X.shape[1]
    O = X.shape[2]
    D = np.zeros((M, M), dtype=np.float32)
    m_com = np.zeros(O, dtype=np.float32)
    t_com = np.zeros(O, dtype=np.float32)
    m = np.zeros((N, O), dtype=np.float32)
    mtrans = np.zeros((O, N), dtype=np.float32)
    mtr = np.zeros((N, O), dtype=np.float32)
    t = np.zeros((N, O), dtype=np.float32)
    c = np.zeros((O, O), dtype=np.float32)
    U = np.zeros((O, O), dtype=np.float32)
    S = np.zeros(O, dtype=np.float32)
    Wt = np.zeros((O, O), dtype=np.float32)
    R = np.zeros((O, O), dtype=np.float32)
    mtr_re = np.zeros(N * O, dtype=np.float32)
    t_re = np.zeros(N * O, dtype=np.float32)
    sub = np.zeros(N * O, dtype=np.float32)
    for i in range(M):
        for j in range(i + 1, M):
            for k in range(O):
                m_com[k] = np.mean(X[i, :, k])
                t_com[k] = np.mean(X[j, :, k])
            m = np.subtract(X[i, :, :], m_com)
            for a in range(N):
                for b in range(O):
                    mtrans[b, a] = m[a, b]
            t = np.subtract(X[j, :, :], t_com)
            c = np.dot(mtrans, t)
            U, S, Wt = np.linalg.svd(c)
            R = np.dot(U, Wt)
            if np.linalg.det(R) < 0.0:
                Wt[-1, :] *= -1.0
                R = np.dot(U, Wt)
            mtr = np.add(np.dot(m, R), t_com)
            q = 0
            for a in range(N):
                for b in range(O):
                    mtr_re[q] = mtr[a, b]
                    t_re[q] = X[j, :, :][a, b]
                    q += 1
            sub = np.subtract(mtr_re, t_re)
            D[i, j] = np.sqrt(1.0 / N * np.dot(sub, sub))
    return D


def make_clusters_from_dataframe(df, an, **kwargs):
    for resname in set(df.resname_vdm):
        pdbs = an.parse_vdms(df[df['resname_vdm'] == resname])
        clu = Cluster(**kwargs)
        clu.run_full_protocol(pdbs)


def make_clusters_all_types_from_dataframe(df, an, **kwargs):
    for resname in set(df.resname_vdm):
        pdbs = an.parse_vdms(df[df['resname_vdm'] == resname])
        kwargs['cluster_type'] = 'SC'
        kwargs['bb_sel'] = [(10, ['N', 'CA', 'C'])]
        clu = Cluster(**kwargs)
        clu.run_full_protocol(pdbs)

        kwargs['cluster_type'] = 'HNCA'
        kwargs['bb_sel'] = [(9, ['C']), (10, ['N', 'CA'])]
        clu = Cluster(**kwargs)
        clu.run_full_protocol(pdbs)

        kwargs['cluster_type'] = 'CO'
        kwargs['bb_sel'] = [(10, ['CA', 'C']), (11, ['N'])]
        clu = Cluster(**kwargs)
        clu.run_full_protocol(pdbs)

        kwargs['cluster_type'] = 'PHI_PSI'
        kwargs['bb_sel'] = [(9, ['C']), (10, ['N', 'CA', 'C']), (11, ['N'])]
        clu = Cluster(**kwargs)
        clu.run_full_protocol(pdbs)

def make_clusters_all_types_from_pdbs(pdbs, clusters=range(1, 21), **kwargs):
    """Makes all types of clusters from given PDBs as input."""
    kwargs['cluster_type'] = 'SC'
    kwargs['bb_sel'] = [(10, ['N', 'CA', 'C'])]
    clu = Cluster(**kwargs)
    clu.run_full_protocol(pdbs, clusters)

    kwargs['cluster_type'] = 'HNCA'
    kwargs['bb_sel'] = [(9, ['C']), (10, ['N', 'CA'])]
    clu = Cluster(**kwargs)
    clu.run_full_protocol(pdbs, clusters)

    kwargs['cluster_type'] = 'CO'
    kwargs['bb_sel'] = [(10, ['CA', 'C']), (11, ['N'])]
    clu = Cluster(**kwargs)
    clu.run_full_protocol(pdbs, clusters)

    kwargs['cluster_type'] = 'PHI_PSI'
    kwargs['bb_sel'] = [(9, ['C']), (10, ['N', 'CA', 'C']), (11, ['N'])]
    clu = Cluster(**kwargs)
    clu.run_full_protocol(pdbs, clusters)


def make_clusters_all_types_from_pdbs_min(pdbs, **kwargs):
    """Makes all types of clusters from given PDBs as input,
    using the minimal protocol (no saving RMSD matrix or printing
    PDBs of clusters."""

    kwargs['cluster_type'] = 'SC'
    kwargs['bb_sel'] = [(10, ['N', 'CA', 'C'])]
    clu = Cluster(**kwargs)
    clu.run_min_protocol(pdbs)

    kwargs['cluster_type'] = 'HNCA'
    kwargs['bb_sel'] = [(9, ['C']), (10, ['N', 'CA'])]
    clu = Cluster(**kwargs)
    clu.run_min_protocol(pdbs)

    kwargs['cluster_type'] = 'CO'
    kwargs['bb_sel'] = [(10, ['CA', 'C']), (11, ['N'])]
    clu = Cluster(**kwargs)
    clu.run_min_protocol(pdbs)

    kwargs['cluster_type'] = 'PHI_PSI'
    kwargs['bb_sel'] = [(9, ['C']), (10, ['N', 'CA', 'C']), (11, ['N'])]
    clu = Cluster(**kwargs)
    clu.run_min_protocol(pdbs)


def _create_alt_vdM(pdb, name_dicts, outdir, gzip=True):
    """Prints new vdms (including original) with coords in name_dicts
    swapped. dicts are key,val=name1,name2.  New pdbs are given hyphenated
    vdM numbers, i.e. vdM_1_ becomes vdM_1-1_."""

    filename = str(pdb).split()[-1].split('_')
    prefix = '_'.join(filename[:4])
    if gzip:
        postfix = '_' + filename[4] + '.pdb.gz'
    else:
        postfix = '_' + filename[4] + '.pdb'

    for i, name_dict in enumerate(name_dicts):
        pdb_ = pdb.copy()
        version = str(i + 2)
        for name1, name2 in name_dict.items():
            coords1 = pdb_.select('chain Y resnum 10 name ' + name1).getCoords()
            coords2 = pdb_.select('chain Y resnum 10 name ' + name2).getCoords()
            pdb_.select('chain Y resnum 10 name ' + name1).setCoords(coords2)
            pdb_.select('chain Y resnum 10 name ' + name2).setCoords(coords1)
        pr.writePDB(outdir + prefix + '-' + version + postfix, pdb_)
    pr.writePDB(outdir + prefix + '-1' + postfix, pdb)


def create_alt_vdMs(indir, name_dicts, outdir, gzip=True):
    if outdir[-1] != '/':
        outdir += '/'

    if indir[-1] != '/':
        indir += '/'

    try:
        makedirs(outdir)
    except FileExistsError:
        pass

    pdbfiles = [f for f in listdir(indir) if f[0] != '.']

    for file in pdbfiles:
        pdb = pr.parsePDB(indir + file)
        try:
            _create_alt_vdM(pdb, name_dicts, outdir, gzip)
        except AttributeError:
            print('AttributeError: ', pdb)

