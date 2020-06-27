import random
import prody as pr
import numpy as np
from collections import defaultdict, deque
from itertools import groupby
from .convex_hull import partition_res_by_burial
from scipy.spatial.distance import cdist
import traceback


# Energy table for Monte Carlo. -1 is neg, 0 neutral, +1 positive
En = defaultdict(dict)
En[-1][-1] = 3
En[1][1] = 2
En[1][-1] = -1
En[-1][1] = -1
En[1][0] = -0.1
En[0][1] = -0.1
En[0][0] = 0
En[0][-1] = -0.1
En[-1][0] = -0.1


class Topology:
    """Calculates surface charge distribution for helical bundles to
    stabilize forward topology over reverse topology.

    Example of typical usage:

    top = combs.apps.topology.Topology()
    top.load_pdb(pdb, selection='resnum 1to31 39to68 79to106 111to140')
    top.set_topologies(outdir='surf_test/')
    top.set_surface_res()
    top.set_contacts()
    top.run_mc()
    top.find_pareto_front()
    top.map_seq_resnums_to_pdb(top.pdb_f)
    top.set_charge_groups(top.seqs[162063])
    top.print_charge_groups()

    """
    def __init__(self, **kwargs):
        self.pdb = None
        self.pdb_sel = None
        self.pdb_ala = None
        self.pdb_ala_sel = None
        self.pdb_f = None
        self.pdb_r = None
        self.surf_rns_f = None
        self.surf_rns_r = None
        self.surf_rns = None
        self.surf_sel_f = None
        self.surf_sel_r = None
        self.nbrs_f = None
        self.nbrs_r = None
        self.contacts_f = None
        self.contacts_r = None
        self._seq = None
        self.seq = dict()
        self.seqs = list()
        self.en_gaps = list()
        self.en_fs = list()
        self.seq_rep_lens = list()
        self.resnum_conv = dict()
        self.resnum_conv_reverse = dict()
        self.negs = list()
        self.neuts = list()
        self.poss = list()
        self._surf_rns = None
        self.pareto_front = None
        self.nearest_utopian_pt = None
        self.constrained_rns = kwargs.get('constrained_rns', list())
        self.constrained_rns_vals = kwargs.get('constrained_rns_vals', list())
        self.constrained_surf_rns_f = list()
        self.constrained_surf_rns_vals_f = dict()

    def load_pdb(self, pdb, selection=None):
        """Loads a prody pdb object (pdb). Takes a selection string (selection).
        Selection string is needed if the pdb is not disconnected helices. Note
        that the selection needs to contain swapped helices of equal lengths, e.g.
        helices 1 and 3 of a 4-helix bundle (with helices 0,1,2,3) must be the
        same length"""
        pdb = pdb.select('protein').copy()
        self.pdb = pdb
        if selection is not None:
            self.pdb_sel = pdb.select(selection)
        else:
            self.pdb_sel = pdb

    def load_pdb_ala(self, pdb, selection=None):
        """Loads a prody pdb object (pdb) that is all alanine residues. Takes a
        selection string (selection).
        Selection string is needed if the pdb is not disconnected helices. Note
        that the selection needs to contain swapped helices of equal lengths, e.g.
        helices 1 and 3 of a 4-helix bundle (with helices 0,1,2,3) must be the
        same length"""
        pdb = pdb.select('protein').copy()
        if (len(set(pdb.getResnames())) != 1) or (set(pdb.getResnames()).pop() != 'ALA'):
            raise "*pdb* must be all alanine residues."
        self.pdb_ala = pdb
        if selection is not None:
            self.pdb_ala_sel = pdb.select(selection)
        else:
            self.pdb_ala_sel = pdb

    def set_topologies(self, outdir=None, tag=''):
        """Takes a prody pdb object or prody selection.
        Outputs two prody pdb objects: pdb with topology 1 and
        pdb with topology 2. This function finds the best cyclic permutation
        of the helices so that helix 0 and n (if n exists, e.g. n = 2
        in a 4 helix bundle, n = 3 in a 6 helix bundle) can have arbitrary length, but
        other helices must be the same length (so that they can be swapped
        in the structure).  Ideally, the pdb should have CB atoms for
        subsequent alpha hull calculations."""
        self.pdb_f = number_helices(self.pdb_sel, reverse=False)
        self.pdb_r = number_helices(self.pdb_sel, reverse=True)
        if outdir is not None:
            if outdir[-1] != '/':
                outdir += '/'
            pr.writePDB(outdir + 'pdb_f' + tag + '.pdb', self.pdb_f)
            pr.writePDB(outdir + 'pdb_r' + tag + '.pdb', self.pdb_r)

    @staticmethod
    def _map_resnums_to_pdb(resnums_of_pdb1, pdb1, pdb2):
        """Maps resnums of pdb1 (list of integers, resnums_of_pdb1) to corresponding
        resnums of pdb2. Returns forward and reverse resnum conversion dictionaries."""
        resnum_conv = dict()
        resnum_conv_reverse = dict()
        for rn in resnums_of_pdb1:
            try:
                sel = pdb1.select('name CA and resnum ' + str(rn))
                pdb_rn = pdb2.select('within 0.05 of sel', sel=sel).getResnums()[0]
                resnum_conv[rn] = pdb_rn
                resnum_conv_reverse[pdb_rn] = rn
            except:
                traceback.print_exc()
        return resnum_conv, resnum_conv_reverse

    def set_surface_res(self, alpha=9, selection_only=False):
        """Calculates surface residues of forward and reverse topologies
        based on alpha hull calculation."""
        if selection_only:
            exp, inter, bur = partition_res_by_burial(self.pdb_ala_sel, alpha=alpha)
            rns = self.pdb_ala_sel.select('name CB or (resname GLY and name CA)').getResnums()
        else:
            exp, inter, bur = partition_res_by_burial(self.pdb_ala, alpha=alpha)
            rns = self.pdb_ala.select('name CB or (resname GLY and name CA)').getResnums()

        self._surf_rns = [rns[rn] for rn in exp]
        resnum_conv_f, resnum_conv_reverse_f = self._map_resnums_to_pdb(self._surf_rns,
                                                                        self.pdb, self.pdb_f)
        self.surf_rns_f = set(list(resnum_conv_f.values()))
        resnum_conv_r, resnum_conv_reverse_r = self._map_resnums_to_pdb(self._surf_rns,
                                                                        self.pdb, self.pdb_r)
        self.surf_rns_r = set(list(resnum_conv_r.values()))
        self.surf_rns = list(self.surf_rns_f) # list(self.surf_rns_f | self.surf_rns_r)
        self.surf_sel_f = self.pdb_f.select('name CA and resnum ' +
                                            ' '.join(str(rn) for rn in self.surf_rns))
        self.surf_sel_r = self.pdb_r.select('name CA and resnum ' +
                                            ' '.join(str(rn) for rn in self.surf_rns))

        if self.constrained_rns:
            constrained_surf_rns = [c_rn for c_rn in self.constrained_rns if c_rn in self._surf_rns]
            constrained_surf_rns_vals = [val for c_rn, val in zip(self.constrained_rns, self.constrained_rns_vals)
                                              if c_rn in self._surf_rns]
            if constrained_surf_rns:
                constrained_surf_rns_f, resnum_cons_conv_reverse_f = self._map_resnums_to_pdb(constrained_surf_rns,
                                                                                              self.pdb, self.pdb_f)
                self.constrained_surf_rns_f = list(constrained_surf_rns_f.values())
                for key, val in zip(self.constrained_surf_rns_f, constrained_surf_rns_vals):
                    self.constrained_surf_rns_vals_f[key] = val


    def set_contacts(self, calpha_distance=10.5):
        """Sets contacts between surface residues. A contact is by default defined
        as a pair of C alpha atoms with a distance less than 10 angstroms."""
        self.nbrs_f = pr.findNeighbors(self.surf_sel_f, calpha_distance)
        self.nbrs_r = pr.findNeighbors(self.surf_sel_r, calpha_distance)
        self.contacts_f = list()
        for nbr in self.nbrs_f:
            resnum_0 = nbr[0].getResnum()
            resnum_1 = nbr[1].getResnum()
            filter = True
            if np.abs(resnum_0 - resnum_1) > 6:
                resind_0 = nbr[0].getResindex()
                resind_1 = nbr[1].getResindex()
                ca_0 = nbr[0]
                cb_0 = self.pdb_f.select('name CB and resindex ' + str(resind_0))
                ca_1 = nbr[1]
                cb_1 = self.pdb_f.select('name CB and resindex ' + str(resind_1))
                ang1 = pr.calcAngle(ca_0, cb_0, cb_1)
                ang2 = pr.calcAngle(ca_1, cb_1, cb_0)
                if (ang1 < 80) or (ang2 < 80):
                    filter = False
            if filter:
                self.contacts_f.append((resnum_0, resnum_1))
        self.contacts_r = list()
        for nbr in self.nbrs_r:
            resnum_0 = nbr[0].getResnum()
            resnum_1 = nbr[1].getResnum()
            filter = True
            if np.abs(resnum_0 - resnum_1) > 6:
                resind_0 = nbr[0].getResindex()
                resind_1 = nbr[1].getResindex()
                ca_0 = nbr[0]
                cb_0 = self.pdb_r.select('name CB and resindex ' + str(resind_0))
                ca_1 = nbr[1]
                cb_1 = self.pdb_r.select('name CB and resindex ' + str(resind_1))
                ang1 = pr.calcAngle(ca_0, cb_0, cb_1)
                ang2 = pr.calcAngle(ca_1, cb_1, cb_0)
                if (ang1 < 80) or (ang2 < 80):
                    filter = False
            if filter:
                self.contacts_r.append((resnum_0, resnum_1))
            
    def initialize_sequence(self):
        """Sets the sequence to all neutral residues."""
        for i in self.surf_rns:
            if i in self.constrained_surf_rns_f:
                self.seq[i] = self.constrained_surf_rns_vals_f[i]
            else:
                self.seq[i] = 0

    def calc_gap(self):
        """Returns the energy gap of forward and reverse sequences."""
        e1 = np.sum(self.contact_En(con) for con in self.contacts_f)
        e2 = np.sum(self.contact_En(con) for con in self.contacts_r)
        return e1 - e2

    def contact_En(self, con):
        """Returns the contact energy between two residues in seq."""
        return En[self.seq[con[0]]][self.seq[con[1]]]

    def calc_En_f(self):
        """Returns the energy of the sequence mapped onto the pdb with forward topology."""
        return np.sum(self.contact_En(con) for con in self.contacts_f)

    def run_mc(self, num_iterations=1000000, kt_en_gap=1, kt_en_f=1, kt_seq_rep_len=0.4):
        """Runs Monte Carlo simulation that simultaneously optimizes:
        1. energy gap between topologies (maximizes),
        2. energy of forward topology sequence (minimizes),
        3. average length of repeat sequences (i.e. continuous strings
        such as 11111 in the sequence. minimizes)"""
        self.initialize_sequence()
        en_gap_old = 0
        en_f_old = 0
        seq_rep_len_old = len(self.surf_rns)
        for _ in range(num_iterations):
            key = random.choice(self.surf_rns)
            if key in self.constrained_surf_rns_f:  #accounts for constraints
                continue
            val = random.choice([0, 1, -1])
            old_val = self.seq[key]
            self.seq[key] = val
            en_gap_new = self.calc_gap()
            p_en_gap_new = np.exp((en_gap_old - en_gap_new) / kt_en_gap)
            en_f_new = self.calc_En_f()
            p_en_f_new = np.exp((en_f_old - en_f_new) / kt_en_f)
            seq_rep_len_new = np.mean([len(list(g)) for k, g in groupby(self.seq.values())])
            p_m_new = np.exp((seq_rep_len_old - seq_rep_len_new) / kt_seq_rep_len)
            if (p_en_f_new >= np.random.rand()) and (p_en_gap_new >= np.random.rand()) \
                    and (p_m_new >= np.random.rand()):
                en_gap_old = en_gap_new
                en_f_old = en_f_new
                seq_rep_len_old = seq_rep_len_new
                self.en_gaps.append(en_gap_new)
                self.en_fs.append(en_f_new)
                self.seq_rep_lens.append(seq_rep_len_new)
                self.seqs.append(self.seq.copy())
            else:
                self.seq[key] = old_val

    def find_pareto_front(self):
        """Returns the indices of the sequences that are in the pareto front,
        considering energy gap, energy of forward topology, and sequence repeat lengths."""
        costs = np.array(list(zip(self.en_gaps, self.en_fs, self.seq_rep_lens)))
        self.pareto_front = self.is_pareto_efficient_indexed(costs)

    @staticmethod
    def is_pareto_efficient_indexed(costs, return_mask=False):
        """
        :param costs: An (n_points, n_costs) array
        :param return_mask: True to return a mask, False to return integer indices of efficient points.
        :return: An array of indices of pareto-efficient points.
            If return_mask is True, this will be an (n_points, ) boolean array
            Otherwise it will be a (n_efficient_points, ) integer array of indices.

        This code is from username Peter at
        https://stackoverflow.com/questions/32791911/fast-calculation-of-pareto-front-in-python
        """
        is_efficient = np.arange(costs.shape[0])
        n_points = costs.shape[0]
        next_point_index = 0  # Next index in the is_efficient array to search for

        while next_point_index < len(costs):
            nondominated_point_mask = np.any(costs <= costs[next_point_index], axis=1)
            is_efficient = is_efficient[nondominated_point_mask]  # Remove dominated points
            costs = costs[nondominated_point_mask]
            next_point_index = np.sum(nondominated_point_mask[:next_point_index]) + 1

        if return_mask:
            is_efficient_mask = np.zeros(n_points, dtype=bool)
            is_efficient_mask[is_efficient] = True
            return is_efficient_mask
        else:
            return is_efficient

    def find_nearest_utopian_pt(self, weight_en_gap=1, weight_en_f=1, weight_seq_rep_len=1):
        en_gaps_pareto = [self.en_gaps[i] for i in self.pareto_front]
        en_fs_pareto = [self.en_fs[i] for i in self.pareto_front]
        seq_rep_lens_pareto = [self.seq_rep_lens[i] for i in self.pareto_front]
        max_en_gap = max(en_gaps_pareto)
        max_en_f = max(en_fs_pareto)
        max_seq_rep_len = max(seq_rep_lens_pareto)
        min_en_gap = min(en_gaps_pareto)
        min_en_f = min(en_fs_pareto)
        min_seq_rep_len = min(seq_rep_lens_pareto)

        pareto_points = np.array([((self.en_gaps[i] - max_en_gap) / (min_en_gap - max_en_gap),
                                   (self.en_fs[i] - max_en_f) / (min_en_f - max_en_f),
                                   (self.seq_rep_lens[i] - max_seq_rep_len) / (min_seq_rep_len - max_seq_rep_len))
                                  for i in self.pareto_front])
        dists = cdist(pareto_points, np.array([[weight_en_gap, weight_en_f, weight_seq_rep_len]]))
        self.nearest_utopian_pt = self.pareto_front[(dists == min(dists)).flatten()]
        if len(self.nearest_utopian_pt) > 1:
            self.nearest_utopian_pt = self.nearest_utopian_pt[0]
        else:
            self.nearest_utopian_pt = int(self.nearest_utopian_pt)

    def map_seq_resnums_to_pdb(self, pdb):
        """Maps resnums of sequence to resnums of pdb."""
        self.resnum_conv = dict()
        self.resnum_conv_reverse = dict()
        for rn in self.seq.keys():
            try:
                sel = self.pdb_f.select('name CA and resnum ' + str(rn))
                pdb_rn = pdb.select('within 0.05 of sel', sel=sel).getResnums()[0]
                self.resnum_conv[rn] = pdb_rn
                self.resnum_conv_reverse[pdb_rn] = rn
            except:
                print('resnum ' + str(rn) + ' not in protein.')

    def set_charge_groups(self, seq):
        """Sets charged groups from sequence seq, which may be found via find_top(n)."""
        self.negs = list()
        self.neuts = list()
        self.poss = list()
        for rn, q in seq.items():
            try:
                if q == -1:
                    self.negs.append(self.resnum_conv[rn])
                if q == 0:
                    self.neuts.append(self.resnum_conv[rn])
                if q == 1:
                    self.poss.append(self.resnum_conv[rn])
            except:
                print('resnum ' + str(rn) + ' not in protein.')


    def print_charge_groups(self):
        """Prints pymol resnum selection strings of the residues mapped to pdb
        via map_seq_resnums_to_pdb."""
        print('neutral residues= ' + '+'.join([str(i) for i in self.neuts]))
        print('negative residues= ' + '+'.join([str(i) for i in self.negs]))
        print('positive residues= ' + '+'.join([str(i) for i in self.poss]))

    def save_sequence(self, seq, outdir='./', filetag=''):
        if outdir[-1] != '/':
            outdir += '/'
        with open(outdir + 'surface_sequence' + filetag + '.txt', 'w') as outfile:
            outfile.write('resnum charge \n')
            for rn, q in seq.items():
                try:
                    outfile.write(str(self.resnum_conv[rn]) + ' ' + str(q) + ' \n')
                except:
                    print('resnum ' + str(rn) + ' not in protein.')


def set_bonds(prody_pdb):
    """Sets backbone bonds of chain based on proximity of atoms."""
    bb_sel = prody_pdb.select('protein and name N C CA')
    dm = pr.buildDistMatrix(bb_sel)
    ind = np.where((np.tril(dm) < 1.7) & (np.tril(dm) > 0))
    atom_ind = bb_sel.getIndices()
    prody_pdb.setBonds([(atom_ind[i], atom_ind[j]) for i, j in zip(ind[0], ind[1])])


def check_helix_lengths(hels):
    """Checks the helices that need to be equal length"""
    hels = list(hels)
    num_helices = len(hels)
    f = np.arange(1, num_helices)
    r = np.array(list(reversed(range(1, num_helices))))
    eq_len_helices = r != f
    lens = np.array([len(hel) for hel in hels[1:]])
    if len(np.unique(lens[eq_len_helices])) == 1:
        return True
    else:
        return False


def order_helices(hels):
    """Cyclically permutes the helices until they achieve the right length"""
    hels = deque(hels)
    i = 0
    while not check_helix_lengths(hels):
        hels.rotate(1)
        i += 1
        if i > 10:
            raise ValueError('Helices are not same length.')
    return list(hels)


def number_helices(pdb, reverse=True):
    """Renumbers a prody pdb/selection for forward or reverse topology.
    Returns a new prody object that is renumbered."""
    pdb = pdb.copy()
    set_bonds(pdb)
    i_end = 0
    hels = list()
    hel_inds = set()
    for i in pdb.iterBonds():
        if np.abs(i.getIndices()[0] - i_end) > 1:
            hels.append(hel_inds)
            hel_inds = set()
            i_end = i.getIndices()[-1]
            continue
        hel_inds |= set([b for b in i.getIndices()])
        i_end = i.getIndices()[-1]
    hels.append(hel_inds)
    hels = order_helices(hels)
    if reverse:
        order = [0]
        for n in list(reversed(range(1, len(hels)))):
            order.append(n)
    else:
        order = list(range(len(hels)))
    j = 1
    all_resnums = list()
    for o in order:
        resnums = list()
        hel = hels[o]
        ris = sorted(set(pdb.select('index ' + ' '.join([str(i) for i in hel])).getResindices()))
        for ri in ris:
            hel_sel = pdb.select('resindex ' + str(ri))
            resnums.extend(len(hel_sel) * [j])
            j += 1
        all_resnums.append(resnums)
    new_resnums = list()
    for o in order:
        new_resnums.extend(all_resnums[o])
    pdb.setResnums(new_resnums)
    return pdb








