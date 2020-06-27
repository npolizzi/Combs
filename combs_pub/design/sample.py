from collections import defaultdict
from ..apps.clashfilter import df_ideal_ala, rel_coords_dict, Clash, ClashVDM, make_pose_df, \
    backbone_str, Contact, make_df_corr, VdmReps
import pickle
import numpy as np
import pandas as pd
from ..apps.transformation import get_rot_trans
from prody import calcPhi, calcPsi, writePDB, AtomGroup
from sklearn.neighbors import NearestNeighbors
from ..apps.convex_hull import AlphaHull
from numba import jit
import time
import os
import copy
import random
import itertools
from scipy.spatial.distance import cdist



coords = ['c_x', 'c_y', 'c_z', 'c_D_x', 'c_D_y',
            'c_D_z', 'c_H1_x', 'c_H1_y', 'c_H1_z',
            'c_H2_x', 'c_H2_y', 'c_H2_z',
            'c_H3_x', 'c_H3_y', 'c_H3_z',
            'c_H4_x', 'c_H4_y', 'c_H4_z',
            'c_A1_x', 'c_A1_y', 'c_A1_z',
            'c_A2_x', 'c_A2_y', 'c_A2_z']

class Template:

    def __init__(self, pdb):
        self.pdb = pdb  # pdb should be prody object poly-gly with CA hydrogens for design.
        self.dataframe = make_pose_df(self.pdb)
        self.alpha_hull = None

    @staticmethod
    def get_bb_sel(pdb):
        return pdb.select(backbone_str).copy()

    def get_phi_psi(self, seg, chain, resnum):
        res = self.pdb[seg, chain, resnum]
        
        try:
            phi = calcPhi(res)
        except ValueError:
            phi = None
        
        try:
            psi = calcPsi(res)
        except ValueError:
            psi = None
            
        return phi, psi

    def set_alpha_hull(self, pdb_w_CB, alpha=9):
        self.pdb_w_CB = pdb_w_CB
        self.alpha_hull = AlphaHull(alpha)
        self.alpha_hull.set_coords(pdb_w_CB)
        self.alpha_hull.calc_hull()


class Load:
    """Doesn't yet deal with terminal residues (although phi/psi does)"""

    def __init__(self, **kwargs):
        self.name = kwargs.get('name')
        self.path = kwargs.get('path', './')  # path to sig reps
        self.sequence_csts = kwargs.get('sequence_csts') # keys1 are tuples (seq, ch, #), keys2 are label,
                                               # vals are allowed residue names (three letter code).
        self.dataframe = pd.DataFrame()
        self.dataframe_grouped = None
        self._rot = defaultdict(dict)
        self._mobile_com = defaultdict(dict)
        self._target_com = defaultdict(dict)
        self._sig_reps = defaultdict(dict)
        self._ideal_ala_df = defaultdict(dict)
        self._nonclashing = list()
        self.remove_from_df = kwargs.get('remove_from_df') # e.g. {1: {'chain': 'Y', 'name': 'CB', 'resname': 'ASN'},
                                                           #       2: {'chain': 'Y', 'name': 'CG', 'resname': 'GLN'}}

    @staticmethod
    def _get_targ_coords(template, label, seg, chain, resnum):
        sel_str = 'segment ' + seg + ' chain ' + chain + ' resnum ' + str(resnum) + ' name '
        cs = []
        for n in rel_coords_dict[label]:
            try:
                cs.append(template.pdb.select(sel_str + n).getCoords()[0])
            except AttributeError:
                try:
                    cs = []
                    for n in ['N', '1H', 'CA']:
                        cs.append(template.pdb.select(sel_str + n).getCoords()[0])
                    return np.stack(cs)
                except AttributeError:
                    try:
                        cs = []
                        for n in ['N', 'H1', 'CA']:
                            cs.append(template.pdb.select(sel_str + n).getCoords()[0])
                        return np.stack(cs)
                    except AttributeError:
                        sel_str = 'chain ' + chain + ' resnum ' + str(resnum) + ' name '
                        cs = []
                        for n in rel_coords_dict[label]:
                            try:
                                cs.append(template.pdb.select(sel_str + n).getCoords()[0])
                            except AttributeError:
                                cs = []
                                for n in ['N', '1H', 'CA']:
                                    cs.append(template.pdb.select(sel_str + n).getCoords()[0])
                                return np.stack(cs)
                        return np.stack(cs)
        return np.stack(cs)

    @staticmethod
    def _get_mob_coords(df, label):
        return np.stack(df[df['name'] == n][['c_x', 'c_y', 'c_z']].values.flatten()
                        for n in rel_coords_dict[label])

    def set_rot_trans(self, template):
        for seg, chain, resnum in self.sequence_csts.keys():
            for label, df in df_ideal_ala.items():
                mob_coords = self._get_mob_coords(df, label)
                targ_coords = self._get_targ_coords(template, label, seg, chain, resnum)
                R, m_com, t_com = get_rot_trans(mob_coords, targ_coords)
                self._rot[label][(seg, chain, resnum)] = R
                self._mobile_com[label][(seg, chain, resnum)] = m_com
                self._target_com[label][(seg, chain, resnum)] = t_com
                df_ = df.copy()
                df_[['c_x', 'c_y', 'c_z']] = np.dot(df_[['c_x', 'c_y', 'c_z']] - m_com, R) + t_com
                self._ideal_ala_df[label][(seg, chain, resnum)] = df_

    def _import_sig_reps(self):
        labels_resns = defaultdict(set)
        for tup in self.sequence_csts.keys():
            for label in self.sequence_csts[tup].keys():
                labels_resns[label] |= set(self.sequence_csts[tup][label])
        for label in labels_resns.keys():
            for resn in labels_resns[label]:
                try:
                    with open(self.path + label + '/' + resn + '.pkl', 'rb') as infile:
                        self._sig_reps[label][resn] = pickle.load(infile)
                except FileNotFoundError:
                    pass

    @staticmethod
    def _get_phi_psi_df(df, phi, psi, phipsi_width=60):
        if phi is not None:
            phi_high = df['phi'] < (phi + (phipsi_width / 2))
            phi_low = df['phi'] > (phi - (phipsi_width / 2))
        else:
            phi_high = np.array([True] * len(df))
            phi_low = phi_high
        if psi is not None:
            psi_high = df['psi'] < (psi + (phipsi_width / 2))
            psi_low = df['psi'] > (psi - (phipsi_width / 2))
        else:
            psi_high = np.array([True] * len(df))
            psi_low = psi_high
        return df[phi_high & phi_low & psi_high & psi_low]

    @staticmethod
    def chunk_df(df_gr, gr_chunk_size=100):
        grs = list()
        for i, (n, gr) in enumerate(df_gr):
            grs.append(gr)
            if (i + 1) % gr_chunk_size == 0:
                yield pd.concat(grs)
                grs = list()

    def _load(self, template, seg, chain, resnum, **kwargs):
        phipsi_width = kwargs.get('phipsi_width', 60)

        dfs = list()
        for label in self.sequence_csts[(seg, chain, resnum)].keys():
            print('loading ' + str((seg, chain, resnum)) + ' , ' + label)
            if label == 'PHI_PSI':
                df_list = list()
                phi, psi = template.get_phi_psi(seg, chain, resnum)
                for resn in self.sequence_csts[(seg, chain, resnum)][label]:
                    df_phipsi = self._get_phi_psi_df(self._sig_reps[label][resn],
                                                     phi, psi, phipsi_width)
                    df_list.append(df_phipsi)
                df = pd.concat(df_list)
            else:
                df = pd.concat([self._sig_reps[label][resn]
                                for resn in self.sequence_csts[(seg, chain, resnum)][label]])

            if self.remove_from_df is not None:
                for d in self.remove_from_df.values():
                    tests = []
                    for col, val in d.items():
                        tests.append(df[col] == val)
                    tests = np.array(tests).T
                    tests = tests.all(axis=1)
                    df = df.loc[~tests]

            m_com = self._mobile_com[label][(seg, chain, resnum)]
            t_com = self._target_com[label][(seg, chain, resnum)]
            R = self._rot[label][(seg, chain, resnum)]
            print('transforming coordinates...')
            df[coords[:3]] = np.dot(df[coords[:3]] - m_com, R) + t_com
            df[coords[3:6]] = np.dot(df[coords[3:6]] - m_com, R) + t_com
            df[coords[6:9]] = np.dot(df[coords[6:9]] - m_com, R) + t_com
            df[coords[9:12]] = np.dot(df[coords[9:12]] - m_com, R) + t_com
            df[coords[12:15]] = np.dot(df[coords[12:15]] - m_com, R) + t_com
            df[coords[15:18]] = np.dot(df[coords[15:18]] - m_com, R) + t_com
            df[coords[18:21]] = np.dot(df[coords[18:21]] - m_com, R) + t_com
            df[coords[21:]] = np.dot(df[coords[21:]] - m_com, R) + t_com
            df['seg_chain_resnum'] = [(seg, chain, resnum)] * len(df)
            df['seg_chain_resnum_'] = [seg + '_' + chain + '_' + str(resnum)] * len(df)
            ###NEW STUFF FOR CLASH FILTER TEST
            df['str_index'] = df['iFG_count'] + '_' + df['vdM_count'] + '_' + df['query_name'] + '_' + df['seg_chain_resnum_']
            print('making transformed dataframe...')
            dfs.append(df)
        dataframe = pd.concat(dfs, sort=False, ignore_index=True)
    
        print('removing clashes...')
        df_nonclash = self._remove(dataframe, template, seg, chain, resnum, **kwargs)
        self._nonclashing.append(df_nonclash)
    @staticmethod
    def _remove(dataframe, template, seg, chain, resnum, **kwargs):
        t0 = time.time()
        cla = ClashVDM(dfq=dataframe, dft=template.dataframe)
        cla.set_grouping('str_index')
        cla.set_exclude((resnum, chain, seg))
        cla.setup()
        cla.find(**kwargs)
        tf = time.time()
        print('time:', tf-t0)
        return cla.dfq_clash_free

    def load(self, template, **kwargs):
        if not self._sig_reps:
            self._import_sig_reps()

        if not self._rot:
            self.set_rot_trans(template)

        for seg, chain, resnum in self.sequence_csts.keys():
            self._load(template, seg, chain, resnum, **kwargs)

        print('concatenating non-clashing to dataframe')
        t0 = time.time()
        self.dataframe = pd.concat(self._nonclashing, sort=False, ignore_index=True)
        self._nonclashing = list()
        self._sig_reps = defaultdict(dict)
        tf = time.time() - t0
        print('concatenated in ' + str(tf) + ' seconds.')
        self._set_grouped_dataframe()

    def load_additional(self, template, sequence_csts, **kwargs):
        seq_csts = defaultdict(dict)
        seq_csts_copy = copy.deepcopy(self.sequence_csts)
        for seg_ch_rn in sequence_csts.keys():
            if seg_ch_rn not in self.sequence_csts.keys():
                seq_csts[seg_ch_rn] = sequence_csts[seg_ch_rn]
                seq_csts_copy[seg_ch_rn] = sequence_csts[seg_ch_rn]

        if len(seq_csts.keys()) > 0:
            self.path = kwargs.get('path', self.path)
            self.sequence_csts = seq_csts
            self._import_sig_reps()
            self.set_rot_trans(template)
            self._nonclashing = list()

            for seg, chain, resnum in self.sequence_csts.keys():
                self._load(template, seg, chain, resnum, **kwargs)

            print('concatenating non-clashing to dataframe')
            t0 = time.time()
            _dataframe = pd.concat(self._nonclashing, sort=False, ignore_index=True)
            self.dataframe = pd.concat((self.dataframe, _dataframe), sort=False, ignore_index=True)
            self._nonclashing = list()
            self._sig_reps = defaultdict(dict)
            tf = time.time() - t0
            print('concatenated in ' + str(tf) + ' seconds.')
            self._set_grouped_dataframe()
            self.sequence_csts = seq_csts_copy
            return True
        else:
            return False

    def _set_grouped_dataframe(self):
        self.dataframe_grouped = self.dataframe.groupby('str_index')


class VdM(Load):
    """Doesn't yet deal with terminal residues (although phi/psi does)"""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.name = kwargs.get('name', 'iFG_type')
        self.neighbors = None
        self.num_iFG_atoms = 0
        self.dataframe_iFG_coords = None
        self.neighbors = None
        self.ligand_iFG_corr_sorted = None
        self.path_to_pdbs = kwargs.get('path_to_pdbs', './')  # for printing vdMs
        if self.path_to_pdbs[-1] != '/':
            self.path_to_pdbs += '/'
        self.ligand_iFG_corr = kwargs.get('ligand_iFG_corr')  # use make_df_corr for dataframe

    def _get_num_iFG_atoms(self):
        d = defaultdict(set)
        for n1, g1 in self.ligand_iFG_corr.groupby('lig_resname'):
            for n2, g2 in g1.groupby('resname'):
                d[n1] |= {len(g2)}
        for key in d.keys():
            assert len(d[key]) == 1, 'Problem with ligand iFG correspondence?'
            self.num_iFG_atoms += d[key].pop()

    def set_sorted_lig_corr(self):
        self.ligand_iFG_corr_sorted = self.ligand_iFG_corr.sort_values(by=['lig_resname', 'lig_name'])

    def set_neighbors(self, rmsd=0.4):
        if self.ligand_iFG_corr_sorted is None:
            self.set_sorted_lig_corr()

        if self.num_iFG_atoms == 0:
            self._get_num_iFG_atoms()

        df_ifg = self.dataframe[self.dataframe['chain'] == 'Y']
        df_ifg = pd.merge(self.ligand_iFG_corr_sorted[['resname', 'name']], df_ifg,
                          how='inner', on=['resname', 'name'], sort=False)
        M = int(len(df_ifg) / self.num_iFG_atoms)
        N = self.num_iFG_atoms
        R = np.arange(len(df_ifg))
        inds = np.array([R[i::M] for i in range(M)]).flatten()
        self.dataframe_iFG_coords = df_ifg[:M]['str_index']
        self.neighbors = NearestNeighbors(radius=np.sqrt(self.num_iFG_atoms) * rmsd, algorithm='ball_tree')
        self.neighbors.fit(df_ifg.iloc[inds][['c_x', 'c_y', 'c_z']].values.reshape(M, N * 3))

    def _print_vdm(self, group_name, df_group, outpath, out_name_tag='', full_fragments=False, with_bb=False):
        """group_name is (ifg_count, vdm_count, query_name, seg_chain_resnum)"""
        if not full_fragments:
            if not with_bb:
                ProdyAG().print_ag(group_name, df_group, outpath, out_name_tag)
            elif with_bb:
                label = set(df_group.label).pop()
                bb_names = rel_coords_dict[label]
                seg_chain_resnum = set(df_group.seg_chain_resnum).pop()
                df_ala = self._ideal_ala_df[label][seg_chain_resnum].copy()
                df_ala['segment'] = set(df_group.segment).pop()
                df_ala['resname'] = set(df_group.resname_vdm).pop()
                df_ala_bbsel = df_ala[df_ala['name'].isin(bb_names)]
                df = pd.concat((df_group, df_ala_bbsel))
                ProdyAG().print_ag(group_name, df, outpath, out_name_tag='_' + label + out_name_tag)
        else:
            pass # update to print full fragments

    def print_vdms(self):
        pass


class ProdyAG:
    def __init__(self):
        self.resnums = list()
        self.names = list()
        self.resnames = list()
        self.coords = list()
        self.chids = list()
        self.segments = list()
        self.elements = list()
        self.ag = None
        self.df_group = None
        self.group_name = None

    def set(self, group_name, df_group):
        self.group_name = group_name
        self.df_group = df_group
        
        for n, d in df_group.iterrows():
            self.resnums.append(d['resnum'])
            name = d['name']
            self.names.append(name)
            self.resnames.append(d['resname'])
            self.coords.append(d[['c_x', 'c_y', 'c_z']])
            self.chids.append(d['chain'])
            self.segments.append(d['segment'])
            if name[0].isdigit():
                self.elements.append(name[1])
            else:
                self.elements.append(name[0])

    def set_ag(self, group_name, df_group):
        if not self.resnums:
            self.set(group_name, df_group)
        
        self.ag = AtomGroup(self.group_name)
        self.ag.setResnums(self.resnums)
        self.ag.setNames(self.names)
        self.ag.setResnames(self.resnames)
        self.ag.setCoords(np.array(self.coords, dtype='float'))
        self.ag.setChids(self.chids)
        self.ag.setSegnames(self.segments)
        self.ag.setElements(self.elements)

    def print_ag(self, group_name, df_group, outpath, out_name_tag=''):
        if self.ag is None:
            self.set_ag(group_name, df_group)

        file = list()
        if isinstance(self.group_name, list):
            for e in self.group_name:
                if isinstance(e, tuple):
                    e = '-'.join([str(i) for i in e])
                file.append(str(e))
        else:
            file.append(str(self.group_name))
        filename = 'ag_' + '_'.join(file) + out_name_tag + '.pdb.gz'

        if outpath[-1] != '/':
            outpath += '/'
        writePDB(outpath + filename, self.ag)


class Ligand(Load):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.dataframe_frag_coords = None
        self.poses = list()
        self.csts = None
        self.csts_gr = None
        self.num_heavy = kwargs.get('num_heavy', 26) #34)
        self.num_total = kwargs.get('num_total', 59)
        self.percent_buried = kwargs.get('percent_buried', 0.5)
        self.isin_field = kwargs.get('isin_field', 'name')
        self.isin = kwargs.get('isin', ['C10', 'C12', 'C13', 'C8', 'C17',
                                        'C24', 'C1', 'C2', 'C3', 'C4', 'C5',
                                        'C6', 'C15', 'N5', 'C7', 'C22', 'C18',
                                        'C16', 'C14', 'C44', 'N2', 'C19',
                                        'C23', 'C20', 'C21', 'C25'])

    def set_csts(self, path_to_cst_file):
        self.csts = make_cst_df(path_to_cst_file)
        self.csts_gr = self.csts.groupby('cst_group')

    def _get_frag_coords(self, df, vdm):
        df_corr = pd.merge(vdm.ligand_iFG_corr_sorted[['lig_resname', 'lig_name']].drop_duplicates(), df,
                           how='inner', on=['lig_resname', 'lig_name'], sort=False)
        return df_corr[['c_x', 'c_y', 'c_z']].values.reshape(1, -1)

    def _load(self, template, seg, chain, resnum, **kwargs):
        t0 = time.time()
        phipsi_width = kwargs.get('phipsi_width', 40)

        dfs = list()
        for label in self.sequence_csts[(seg, chain, resnum)].keys():
            print('loading ' + str((seg, chain, resnum)) + ' , ' + label)
            if label == 'PHI_PSI':
                df_list = list()
                phi, psi = template.get_phi_psi(seg, chain, resnum)
                for resn in self.sequence_csts[(seg, chain, resnum)][label]:
                    df_phipsi = self._get_phi_psi_df(self._sig_reps[label][resn],
                                                     phi, psi, phipsi_width)
                    df_list.append(df_phipsi)
                df = pd.concat(df_list)
            else:
                df = pd.concat([self._sig_reps[label][resn]
                                for resn in self.sequence_csts[(seg, chain, resnum)][label]])

            m_com = self._mobile_com[label][(seg, chain, resnum)]
            t_com = self._target_com[label][(seg, chain, resnum)]
            R = self._rot[label][(seg, chain, resnum)]
            print('transforming coordinates...')
            df[coords[:3]] = np.dot(df[coords[:3]] - m_com, R) + t_com
            df[coords[3:6]] = np.dot(df[coords[3:6]] - m_com, R) + t_com
            df[coords[6:9]] = np.dot(df[coords[6:9]] - m_com, R) + t_com
            df[coords[9:12]] = np.dot(df[coords[9:12]] - m_com, R) + t_com
            df[coords[12:15]] = np.dot(df[coords[12:15]] - m_com, R) + t_com
            df[coords[15:18]] = np.dot(df[coords[15:18]] - m_com, R) + t_com
            df[coords[18:21]] = np.dot(df[coords[18:21]] - m_com, R) + t_com
            df[coords[21:]] = np.dot(df[coords[21:]] - m_com, R) + t_com
            df['seg_chain_resnum'] = [(seg, chain, resnum)] * len(df)
            df['seg_chain_resnum_'] = [seg + '_' + chain + '_' + str(resnum)] * len(df)
            df['str_index'] = df['iFG_count'] + '_' + df['vdM_count'] + '_' + df['query_name'] + '_' + df['seg_chain_resnum_']
            print('making transformed dataframe...')
            dfs.append(df)
        dataframe = pd.concat(dfs, sort=False)

        print('removing clashes...')
        df_nonclash = self._remove(dataframe, template, seg, chain, resnum, **kwargs)
        if len(df_nonclash) > 0:
            print('removing exposed ligands...')
            df_alpha = self.remove_alpha_hull(df_nonclash, template)
            self._nonclashing.append(df_alpha)
        tf = time.time()
        print('loaded ligand for (' + seg + ', ' + chain + ', ' + str(resnum) + ') in ' + str(tf - t0) + ' seconds.')

    @staticmethod
    def _remove(dataframe, template, seg, chain, resnum, **kwargs):
        t0 = time.time()
        cla = Clash(dfq=dataframe, dft=template.dataframe)
        # cla.set_grouping(['iFG_count', 'vdM_count',
        #                   'query_name', 'seg_chain_resnum'])
        # cla.set_grouping(['iFG_count', 'vdM_count', 'query_name'])
        cla.set_grouping('str_index')
        cla.find()
        tf = time.time()
        print('time:', tf-t0)
        return cla.dfq_clash_free

    @staticmethod
    def print_lig(group_name, df_group, outpath, out_name_tag=''):
        """group_name is (ifg_count, vdm_count, query_name, seg_chain_resnum)"""
        ProdyAG().print_ag(group_name, df_group, outpath, out_name_tag)

    def remove_alpha_hull(self, df, template):
        """Removes ligands with less than *percent_buried* heavy atoms
         within the alpha hull of the template.  This also adds the columns
         *in_hull* and *dist_to_hull* to the (surviving) ligand dataframe"""

        dfh = df[df[self.isin_field].isin(self.isin)]
        inout = template.alpha_hull.pnts_in_hull(dfh[['c_x', 'c_y', 'c_z']].values)
        index = dfh.index
        df.loc[index, 'in_hull'] = inout
        inout = inout.reshape(-1, self.num_heavy)
        bur = inout.sum(axis=1) > np.floor((self.percent_buried * self.num_heavy))
        df_inds = np.arange(len(df)).reshape(-1, self.num_total)
        buried_inds = df_inds[bur].flatten()
        df_bur = df.iloc[buried_inds]
        return df_bur

    def find_frag_neighbors(self, vdms, template, hb_only=False, return_rmsd=False):
        if not self.dataframe_grouped:
            self._set_grouped_dataframe()

        for ind, lig in self.dataframe_grouped:
            pose = Pose()
            dfs_to_append = list()
            lig = lig.copy()
            for vdm in vdms:
                lig_coords = self._get_frag_coords(lig, vdm)
                if return_rmsd:
                    dist, ind_neighbors = vdm.neighbors.radius_neighbors(lig_coords, return_distance=True)
                    dist = dist[0]
                    ind_neighbors = ind_neighbors[0]
                    rmsds = dist / np.sqrt(vdm.num_iFG_atoms)
                else:
                    ind_neighbors = vdm.neighbors.radius_neighbors(lig_coords, return_distance=False)[0]
                if ind_neighbors.size > 0:
                    if return_rmsd:
                        df_uniq = pd.DataFrame(vdm.dataframe_iFG_coords.iloc[ind_neighbors])
                        df_uniq['rmsd_to_query'] = rmsds
                        df_uniq = df_uniq.drop_duplicates()
                        to_concat = []
                        for str_index, rmsd in df_uniq[['str_index', 'rmsd_to_query']].values:
                            d = vdm.dataframe_grouped.get_group(str_index).copy()
                            d['rmsd_to_query'] = rmsd
                            to_concat.append(d)
                        df_to_append = pd.concat(to_concat, sort=False, ignore_index=True)
                    else:
                        df_uniq = vdm.dataframe_iFG_coords.iloc[ind_neighbors].drop_duplicates()
                        df_to_append = pd.concat([vdm.dataframe_grouped.get_group(g) for g in
                                              df_uniq.values], sort=False, ignore_index=True)
                    print('appending to possible pose...')
                    dfs_to_append.append(df_to_append)
            if len(dfs_to_append) > 0:
                pose._vdms = pd.concat(dfs_to_append)
                if 'num_tag' in pose._vdms.columns:
                    pose._vdms = pose._vdms.drop('num_tag', axis=1).drop_duplicates()
                else:
                    pose._vdms = pose._vdms.drop_duplicates()
                pose.ligand = lig
                pose.ligand_gr_name = ind
                print('getting non clashing pose...')
                pose.set_nonclashing_vdms()
                if len(pose.vdms) > 0:
                    print('checking pose constraints...')
                    sc_bb = pd.concat((pose.vdms_sidechains, template.dataframe), sort=False)
                    lig_con = Contact(sc_bb, lig)
                    lig_con.find()
                    if len(lig_con.df_contacts) > 0:
                        inout = template.alpha_hull.pnts_in_hull(lig[['c_x', 'c_y', 'c_z']].values)
                        lig.loc[:, 'in_hull'] = inout
                        dist_bur = template.alpha_hull.get_pnts_distance(lig[['c_x', 'c_y', 'c_z']].values)
                        lig.loc[:, 'dist_in_hull'] = dist_bur
                        if self.check_csts(lig, lig_con.df_contacts):
                            print('pose found...')
                            pose.lig_contacts = lig_con.df_contacts
                            if hb_only:
                                hb_contacts = pose.lig_contacts[pose.lig_contacts.contact_type == 'hb']
                                df_hb = get_vdms_hbonding_to_lig(pose, hb_contacts)
                                bb_contacts = hb_contacts[hb_contacts.iFG_count_q.isnull()]
                                all_vdms = []
                                all_vdms.append(df_hb)
                                if len(bb_contacts) > 0:
                                    bb_only_vdms = pose.vdms[
                                        pose.vdms.seg_chain_resnum.isin(set(bb_contacts.seg_chain_resnum_q))].groupby(
                                        pose.groupby).filter(lambda x: 'X' not in set(x.chain))
                                    for n, g in bb_only_vdms.groupby('seg_chain_resnum_'):
                                        min_clust_num = min(g.cluster_number)
                                        for n_, g_ in g[g.cluster_number == min_clust_num].groupby(pose.groupby):
                                            all_vdms.append(g_)
                                            break
                                df_all_vdms = pd.concat(all_vdms, sort=False)
                                df = pd.merge(pose.vdms, df_all_vdms[pose.groupby].drop_duplicates(), on=pose.groupby)
                                pose.vdms = df
                                pose.vdms_sidechains = df_hb
                                pose.lig_contacts = hb_contacts
                            pose._vdms = None
                            pose.num_heavy_buried = lig[lig[self.isin_field].isin(self.isin)].in_hull.sum()
                            pose.lig_csts = self.csts
                            pose.lig_csts_gr = self.csts_gr
                            self.poses.append(pose)

    def check_csts(self, lig, lig_contacts):
        if not self.csts_gr:
            return True
        for n, cst_gr in self.csts_gr:
            atom_cst_tests = list()
            for i, cst in cst_gr.iterrows():
                if cst['contact_type']:
                    resname = lig_contacts['resname_t'] == cst['lig_resname']
                    name = lig_contacts['name_t'] == cst['lig_name']
                    lig_atom = lig_contacts[resname & name]
                    if any(lig_atom['contact_type'].isin(cst['contact_type'])):
                        atom_cst_tests.append(True)
                    else:
                        atom_cst_tests.append(False)
                if pd.notnull(cst['burial']):
                    resname = lig['lig_resname'] == cst['lig_resname']
                    name = lig['lig_name'] == cst['lig_name']
                    lig_atom = lig[resname & name]
                    lig_burial = lig_atom['in_hull'].item()
                    if cst['burial'] == lig_burial:
                        atom_cst_tests.append(True)
                    else:
                        atom_cst_tests.append(False)
                if pd.notnull(cst['dist_buried']):
                    resname = lig['lig_resname'] == cst['lig_resname']
                    name = lig['lig_name'] == cst['lig_name']
                    lig_atom = lig[resname & name]
                    lig_dist_buried = lig_atom['dist_in_hull'].item()
                    if cst['dist_buried_lessthan']:
                        if lig_dist_buried < cst['dist_buried']:
                            atom_cst_tests.append(True)
                        else:
                            atom_cst_tests.append(False)
                    else:
                        if lig_dist_buried > cst['dist_buried']:
                            atom_cst_tests.append(True)
                        else:
                            atom_cst_tests.append(False)
            if not any(atom_cst_tests):  # If any group cst fails, the function returns False
                return False
        return True


class Pose:

    def __init__(self, **kwargs):
        self.score_designability = None
        self.score_opt = None
        self.ligand = None  # ligand dataframe
        self.ligand_gr_name = None
        self._vdms = None #dict()  # keys are vdm names, vals are dataframes, pre-clash removal
        self.vdms = None #dict()  # keys are vdm names, vals are dataframes, post-clash removal
        self.vdms_sidechains = None
        self.groupby = kwargs.get('groupby', ['str_index'])
        self.lig_contacts = None
        self.num_heavy_buried = None
        self.hb_net = list()
        self.num_buns_lig = 0
        self._poseleg_number = 0
        self.lig_csts = None
        self.lig_csts_gr = None

    def set_lig_csts(self, path_to_cst_file):
        self.lig_csts = make_cst_df(path_to_cst_file)
        self.lig_csts_gr = self.lig_csts.groupby('cst_group')

    def set_nonclashing_vdms(self):
        vdms_x = self._vdms[self._vdms.chain == 'X'].copy()
        cla = Clash(vdms_x, self.ligand, **dict(tol=0.1))
        cla.set_grouping(self.groupby)
        cla.find(return_clash_free=True, return_clash=True)
        df = pd.merge(self._vdms, cla.dfq_clash[self.groupby],
                    on=self.groupby, how='outer', indicator=True, sort=False)
        df = df[df['_merge'] == 'left_only'].drop(columns='_merge')
        self.vdms = df
        self.vdms_sidechains = cla.dfq_clash_free

    def find_opt(self):
        pass

    def set_lig_buns(self, template):
        bun_acc, bun_don = get_bun_hb_atoms(df=self.ligand, template=template)
        self.num_buns_lig = len(bun_acc) + len(bun_don)
        self.lig_acc_buns = bun_acc
        self.lig_don_buns = bun_don

    def set_num_hb_net_lig_bun_interactions(self, template):
        for hbnet in self.hb_net:
            num_lig_ints = self._set_num_hb_net_lig_bun_interactions(hbnet, template)
            hbnet.num_lig_ints = num_lig_ints

    def _set_num_hb_net_lig_bun_interactions(self, hbnet, template):
        con = Contact(hbnet.primary, self.ligand)
        con.find()
        bun_acc, bun_don = get_bun_hb_atoms(df=self.ligand, template=template, contacts=con.df_contacts)
        return self.num_buns_lig - (len(bun_acc) + len(bun_don))

    def _set_num_pl_lig_bun_interactions(self, pl, template):
        con = Contact(pl, self.ligand)
        con.find()
        bun_acc, bun_don, sat_acc, sat_don = get_bun_hb_atoms(df=self.ligand, template=template, contacts=con.df_contacts, return_satisfied=True)
        num_lig_bun_ints = self.num_buns_lig - (len(bun_acc) + len(bun_don))
        sat_acc_set = set()
        sat_don_set = set()
        if len(sat_acc) > 0:
            sat_acc_set = set(sat_acc.name)
        if len(sat_don) > 0:
            sat_don_set = set(sat_don.name)
        return num_lig_bun_ints, sat_acc_set, sat_don_set

    @staticmethod
    def _print_vdm(vdm, group_name, df_group, outpath, out_name_tag='', full_fragments=False, with_bb=False):
        """group_name is (ifg_count, vdm_count, query_name, seg_chain_resnum)"""
        if not full_fragments:
            if not with_bb:
                ProdyAG().print_ag(group_name, df_group, outpath, out_name_tag)
            elif with_bb:
                label = set(df_group.label).pop()
                bb_names = rel_coords_dict[label]
                seg_chain_resnum = set(df_group.seg_chain_resnum).pop()
                df_ala = vdm._ideal_ala_df[label][seg_chain_resnum].copy()
                df_ala['segment'] = set(df_group.segment).pop()
                df_ala['resname'] = set(df_group.resname_vdm).pop()
                df_ala_bbsel = df_ala[df_ala['name'].isin(bb_names)]
                df = pd.concat((df_group, df_ala_bbsel))
                ProdyAG().print_ag(group_name, df, outpath, out_name_tag='_' + label + out_name_tag)
        else:
            pass

    def print_vdms(self, vdm, outpath, out_name_tag='', full_fragments=False, with_bb=False):
        for n, gr in self.vdms.groupby(self.groupby):
            self._print_vdm(vdm, n, gr, outpath, out_name_tag, full_fragments, with_bb)

    def print_hb_nets(self, min_bun_only=True, outdir='./'):
        for k, hbnet in enumerate(self.hb_net):
            self._print_hb_net(hbnet, k, min_bun_only, outdir)

    def _print_hb_net(self, hbnet, k, min_bun_only=True, outdir='./'):
        if outdir[-1] != '/':
            outdir += '/'

        try:
            os.makedirs(outdir)
        except FileExistsError:
            pass

        pl = PoseLegs()
        pl.get_poselegs(hbnet)
        pl.drop_duplicates()
        min_bun = min(pl.num_buns_uniq)
        for i, (nbuns, df) in enumerate(zip(pl.num_buns_uniq, pl.poselegs_uniq)):
            can_print = True
            if min_bun_only:
                if nbuns != min_bun:
                    can_print = False
            if can_print:
                ProdyAG().print_ag('h', df, outdir, '_' + str(k) + '_' + str(i) + '_' + str(nbuns) + 'numbuns')

    def make_pose_legs(self, template):
        for hbnet in self.hb_net:
            hbnet.pose_legs = self._make_pose_legs(hbnet, template)

    def _make_pose_legs(self, hbnet, template):
        pl = PoseLegs()
        pl.get_poselegs(hbnet)
        pl.drop_duplicates()
        for n, p in zip(pl.num_buns_uniq, pl.poselegs_uniq):
            p['poseleg_number'] = self._poseleg_number
            p['num_buns'] = n
            num_lig_ints, sat_acc_set, sat_don_set = self._set_num_pl_lig_bun_interactions(p, template)
            p['num_lig_ints'] = num_lig_ints
            acc = '_'.join(i for i in sat_acc_set)
            don = '_'.join(i for i in sat_don_set)
            p['sat_acc_set'] = acc
            p['sat_don_set'] = don
            self._poseleg_number += 1
        return pl

    def _get_clash_pose_legs(self, pl1, pl2, inds1, inds2):
        """returns true if unique residues from each pose are clashing"""
        if len(inds1) == len(pl1) and len(inds2) == len(pl2):
            # cla = Clash(pl1, pl2, **dict(q_grouping=['iFG_count', 'vdM_count', 'query_name', 'seg_chain_resnum']))
            cla = Clash(pl1, pl2, **dict(q_grouping='str_index'))
        elif len(inds2) == 0:
            return False
        else:
            cla = Clash(pl1.iloc[inds1], pl2.iloc[inds2], **dict(q_grouping='str_index', tol=0.05))
        cla.find()
        return len(cla.dfq_clash_free) != len(pl1)

    def _get_pairwise_hb_int_pose_legs(self, pl1, pl2, template, inds1, inds2, **kwargs):
        if len(inds1) == len(pl1) and len(inds2) == len(pl2):
            df = pd.concat((pl1, pl2), sort=False)
        elif len(inds2) == 0:
            df = pl1
        else:
            if ('num_tag' in pl1.columns) or ('num_tag' in pl2.columns):
                df = pd.concat((pl1.iloc[inds1], pl2.iloc[inds2]), sort=False).drop('num_tag', axis=1)
            else:
                df = pd.concat((pl1.iloc[inds1], pl2.iloc[inds2]), sort=False)
        num_acc, num_don = get_num_bun_hb_atoms(df, template, self, **kwargs)
        return num_acc + num_don

    def make_pls_sorted(self):
        pls = []
        pls_nums = []
        for hbnet in self.hb_net:
            for p in hbnet.pose_legs.poselegs_uniq:
                pls.append(p)
                pls_nums.append(set(p.poseleg_number).pop())
        inds_sorted = sorted(range(len(pls)), key=lambda x: pls_nums[x])
        pls_sorted = [pls[i] for i in inds_sorted]
        self.pls_sorted = pls_sorted

    def set_lig_bb_contacts(self):
        self.lig_bb_contacts = self.lig_contacts[self.lig_contacts.iFG_count_q.isnull()]

    def make_energy_arrays(self, template, **kwargs):
        pls = []
        pls_nums = []
        for hbnet in self.hb_net:
            for p in hbnet.pose_legs.poselegs_uniq:
                pls.append(p)
                pls_nums.append(set(p.poseleg_number).pop())
        inds_sorted = sorted(range(len(pls)), key=lambda x: pls_nums[x])
        pls_sorted = [pls[i] for i in inds_sorted]
        self.pls_sorted = pls_sorted
        pairwise_array = np.zeros((len(pls_sorted), len(pls_sorted)), dtype='float')
        single_array = np.zeros(len(pls_sorted), dtype='float')
        for i in range(len(pls_sorted)):
            pl1 = pls_sorted[i]
            single_array[i] = set(pl1.num_buns).pop() - set(pl1.num_lig_ints).pop() # * 1.5
            for k in range(i + 1, len(pls)):
                pl2 = pls_sorted[k]
                a1 = pl1[['c_x', 'c_y', 'c_z']].values
                a2 = pl2[['c_x', 'c_y', 'c_z']].values
                dists = cdist(a1, a2)
                print(set(pl1.poseleg_number).pop(), set(pl2.poseleg_number).pop())
                if (dists <= 4.8).any():
                    if is_subset(a1, a2):  # one poseleg is a subset of the other, so make them clash
                        pairwise_array[i, k] = 50
                    else:
                        a1_, a2_, inds1, inds2 = remove_dups(a1, a2, return_inds=True)
                        dists_ = cdist(a1_, a2_)
                        if (dists_ <= 1).any():
                            pairwise_array[i, k] = 50
                        elif self._get_clash_pose_legs(pl1, pl2, inds1, inds2):
                            pairwise_array[i, k] = 50
                        else:
                            atoms_acc1 = pl1[pl1.c_A1_x.notna()][['c_x', 'c_y', 'c_z']].values
                            atoms_don1 = pl1[pl1.c_D_x.notna()][['c_x', 'c_y', 'c_z']].values
                            atoms_acc2 = pl2[pl2.c_A1_x.notna()][['c_x', 'c_y', 'c_z']].values
                            atoms_don2 = pl2[pl2.c_D_x.notna()][['c_x', 'c_y', 'c_z']].values
                            d1 = cdist(atoms_acc1, atoms_don2)
                            d2 = cdist(atoms_don1, atoms_acc2)
                            if (d1 <= 3.25).any() or (d2 <= 3.25).any():
                                nb = self._get_pairwise_hb_int_pose_legs(pl1, pl2, template, inds1, inds2, **kwargs)
                                new_hbs = max(max(set(pl1.num_buns).pop(), set(pl2.num_buns).pop()) - nb, 0)
                                if new_hbs == 0:
                                    pairwise_array[i, k] = 0
                                else:
                                    pairwise_array[i, k] = (-1 * new_hbs)
                            else:
                                pairwise_array[i, k] = 0
                else:
                    pairwise_array[i, k] = 0
                pairwise_array[k, i] = pairwise_array[i, k]
        self.single_array = single_array
        self.pairwise_array = pairwise_array

    def dee(self):
        n = self.single_array.size
        es = self.single_array
        ep = self.pairwise_array
        to_del = _dee(es, ep)
        es = np.delete(es, to_del)
        ep = np.delete(ep, to_del, axis=0)
        ep = np.delete(ep, to_del, axis=1)
        self.pls_sorted_dee = [self.pls_sorted[i] for i in range(n) if i not in to_del]
        self.Es_dee = es
        self.Ep_dee = ep
        self.to_delete = to_del

    def _make_nonpairwise_en(self, _keys, template, burial_depth=1, **kwargs):
        clash_en = self._calc_clash_en(_keys)
        if clash_en > 0:
            return clash_en

        df = pd.concat((self.pls_sorted[i] for i in _keys), sort=False)
        if ('num_tag' in df.columns):
            df = df.drop('num_tag', axis=1).drop_duplicates()
        else:
            df = df.drop_duplicates()
        acc, don = _get_bun_hb_atoms(df, template, self, burial_depth, **kwargs)
        costs = 0
        if len(acc) > 0:
            costs += np.sum(acc['distance_to_hull'].values/50)
        if len(don) > 0:
            costs += np.sum(don['distance_to_hull'].values/50)
        df_con = Contact(df, self.ligand)
        df_con.find()
        contacts_ = df_con.df_contacts
        all_lig_contacts = pd.concat((contacts_, self.lig_bb_contacts), sort=False)
        lig = Ligand()
        lig.csts_gr = self.lig_csts_gr
        if not lig.check_csts(self.ligand, all_lig_contacts):
            return 50
        lig_bun_acc, lig_bun_don = get_bun_hb_atoms(df=self.ligand, template=template, contacts=contacts_)
        if len(lig_bun_acc) > 0:
            costs += np.sum(lig_bun_acc['distance_to_hull'].values/40)
        if len(lig_bun_don) > 0:
            costs += np.sum(lig_bun_don['distance_to_hull'].values/40)
        return costs

    def make_clash_en_matrix(self):
        print('making clash matrix...')
        pairwise_array = np.zeros((len(self.pls_sorted), len(self.pls_sorted)), dtype='float')
        for i in range(len(self.pls_sorted)):
            pl1 = self.pls_sorted[i]
            for k in range(i + 1, len(self.pls_sorted)):
                pl2 = self.pls_sorted[k]
                a1 = pl1[['c_x', 'c_y', 'c_z']].values
                a2 = pl2[['c_x', 'c_y', 'c_z']].values
                dists = cdist(a1, a2)
                if (dists <= 4.8).any():
                    if is_subset(a1, a2):  # one poseleg is a subset of the other, so make them clash
                        pairwise_array[i, k] = 50
                    else:
                        a1_, a2_, inds1, inds2 = remove_dups(a1, a2, return_inds=True)
                        dists_ = cdist(a1_, a2_)
                        if (dists_ <= 1).any():
                            pairwise_array[i, k] = 50
                        elif self._get_clash_pose_legs(copy.deepcopy(pl1), copy.deepcopy(pl2), inds1, inds2):
                            pairwise_array[i, k] = 50
        pairwise_array = pairwise_array + pairwise_array.T
        self.clash_matrix = pairwise_array

    def make_single_array(self):
        single_array = np.zeros(len(self.pls_sorted), dtype='float')
        for i in range(len(self.pls_sorted)):
            pl1 = self.pls_sorted[i]
            single_array[i] = set(pl1.num_buns).pop() - set(pl1.num_lig_ints).pop()
        self.single_array = single_array

    def make_weights(self):
        un = np.unique(self.single_array)
        n = len(un)
        self.weights = np.zeros(len(self.single_array))
        for u in un:
            self.weights[self.single_array == u] = n
            if u > 0:
                n -= 1

    def _calc_clash_en(self, _keys):
        return sum(self.clash_matrix[i, j] for i, j in itertools.combinations(_keys, 2))

    def run_mc_nonpairwise(self, num_pose_legs, template, burial_depth=1, trials=10000, kT=1, num_non_clash=500, **kwargs):
        all_keys = range(len(self.pls_sorted))
        _keys = random.sample(list(all_keys), num_pose_legs)
        e_old = self._make_nonpairwise_en(_keys, template, burial_depth, **kwargs)
        energies = []
        keys = []
        energies.append(e_old)
        keys.append(_keys)
        for_pop = list(range(num_pose_legs))
        j = 0
        for i in range(trials):
            if i % 100 == 0:
                print('iteration', i + 1)
            old_keys = copy.deepcopy(_keys)
            oldkey = 1
            newkey = 1
            while oldkey == newkey:
                oldkey = _keys.pop(random.choice(for_pop))
                newkey = random.choices(all_keys, weights=self.weights, k=1)[0]
                if newkey in _keys:
                    _keys.append(oldkey)
                    oldkey = newkey
                else:
                    _keys.append(newkey)
            e = self._make_nonpairwise_en(_keys, template, burial_depth, **kwargs)
            if e < 10:
                j += 1
            p = min(1, np.exp(-(e - e_old) / kT))
            if np.random.rand() <= p:
                print('old', e_old, 'new', e, 'accepted')
                energies.append(e)
                keys.append(copy.deepcopy(_keys))
                e_old = e
            else:
                print('old', e_old, 'new', e, 'rejected')
                _keys = old_keys
            if j == num_non_clash:
                break
            if j % 50 == 0:
                print('nonclashing', j + 1)
        self.mc_energies = energies
        self.mc_keys = keys

    def run_sim_anneal(self, num_pose_legs, template, annealing_sched, burial_depth=1, trials=10000,
                       **kwargs):
        all_keys = range(len(self.pls_sorted))
        _keys = random.sample(list(all_keys), num_pose_legs)
        e_old = self._make_nonpairwise_en(_keys, template, burial_depth, **kwargs)
        energies = []
        keys = []
        energies.append(e_old)
        keys.append(_keys)
        for_pop = list(range(num_pose_legs))
        for i in range(trials):
            kT = annealing_sched[i]
            if i % 100 == 0:
                print('iteration', i + 1, 'kT=', kT)
            old_keys = copy.deepcopy(_keys)
            oldkey = 1
            newkey = 1
            while oldkey == newkey:
                oldkey = _keys.pop(random.choice(for_pop))
                newkey = random.choices(all_keys, weights=self.weights, k=1)[0]
                if newkey in _keys:
                    _keys.append(oldkey)
                    oldkey = newkey
                else:
                    _keys.append(newkey)
            e = self._make_nonpairwise_en(_keys, template, burial_depth, **kwargs)
            p = min(1, np.exp(-(e - e_old) / kT))
            if np.random.rand() <= p:
                print('old', e_old, 'new', e, 'accepted')
                energies.append(e)
                keys.append(copy.deepcopy(_keys))
                e_old = e
            else:
                print('old', e_old, 'new', e, 'rejected')
                _keys = old_keys
        self.mc_energies = energies
        self.mc_keys = keys

    def _energy_fn_singles(self, keys):
        return sum(self.single_array[key] for key in keys)

    def _energy_fn_pairs(self, keys):
        return sum(self.pairwise_array[key1, key2] for key1, key2 in itertools.combinations(keys, 2))


def make_cst_df(path_to_cst_file):
    groups = list()
    resnames = list()
    names = list()
    contacts = list()
    burials = list()
    dists_buried = list()
    dists_lessthan = list()
    with open(path_to_cst_file, 'r') as infile:
        for line in infile:
            if line.startswith('#'):
                continue
            spl = line.split()
            if len(spl) < 1:
                continue
            group = int(spl[0].strip())
            resname = spl[1].strip()
            name = spl[2].strip()
            try:
                CO_ind = spl.index('CO')
            except ValueError:
                CO_ind = None
            try:
                BU_ind = spl.index('BU')
            except ValueError:
                BU_ind = None
            try:
                DB_ind = spl.index('DB')
            except ValueError:
                DB_ind = None

            if BU_ind:
                burial = spl[BU_ind + 1]
                if burial == 'buried':
                    burial = True
                elif burial == 'exposed':
                    burial = False
                else:
                    raise ValueError('burial must be "exposed" or "buried"')
            else:
                burial = None

            if DB_ind:
                dist = spl[DB_ind + 1]
                if dist[0] == '<':
                    dist_lessthan = True
                elif dist[0] == '>':
                    dist_lessthan = False
                else:
                    raise ValueError('distance buried must be "<" or ">" a number, e.g. "<0.5')
                dist = float(dist[1:])
            else:
                dist = None
                dist_lessthan = None

            CO_set = set()
            if CO_ind:
                CO_set = set(spl[CO_ind + 1].split(','))

            groups.append(group)
            resnames.append(resname)
            names.append(name)
            contacts.append(CO_set)
            burials.append(burial)
            dists_buried.append(dist)
            dists_lessthan.append(dist_lessthan)
    data = dict(cst_group=groups, lig_resname=resnames, lig_name=names,
                contact_type=contacts, burial=burials, dist_buried=dists_buried,
                dist_buried_lessthan=dists_lessthan)
    return pd.DataFrame(data)


acceptor_atom_types = ['n', 'o', 'f']

path_to_sig_dict = defaultdict(dict)
path_to_sig_dict['SER'] = '~/combs/database/representatives/hb_only/hydroxyl/20181009/'
path_to_sig_dict['THR'] = '~/combs/database/representatives/hb_only/hydroxyl/20181009/'
path_to_sig_dict['TYR'] = '~/combs/database/representatives/hb_only/hydroxyl/20181009/'
path_to_sig_dict['ASN'] = '~/combs/database/representatives/hb_only/carboxamide/20181002/'
path_to_sig_dict['GLN'] = '~/combs/database/representatives/hb_only/carboxamide/20181002/'
path_to_sig_dict['ASP'] = '~/combs/database/representatives/hb_only/carboxylate/20181009/'
path_to_sig_dict['GLU'] = '~/combs/database/representatives/hb_only/carboxylate/20181009/'
path_to_sig_dict['ARG'] = '~/combs/database/representatives/hb_only/arginine/20181009/'
path_to_sig_dict['LYS'] = '~/combs/database/representatives/hb_only/lysine/20181009/'
path_to_sig_dict['backboneNH'] = '~/combs/database/representatives/hb_only/backboneNH/20181002/'
path_to_sig_dict['backboneCO'] = '~/combs/database/representatives/hb_only/backboneCO/20181002/'
path_to_sig_dict['HIS'] = defaultdict(dict)
path_to_sig_dict['HIS']['ND1']['ACC'] = '~/combs/database/representatives/hb_only/imidazoleacc/20181009/'
path_to_sig_dict['HIS']['NE2']['ACC'] = '~/combs/database/representatives/hb_only/imidazoleacc/20181009/'
path_to_sig_dict['HIS']['ND1']['DON'] = '~/combs/database/representatives/hb_only/imidazoledon/20181009/'
path_to_sig_dict['HIS']['NE2']['DON'] = '~/combs/database/representatives/hb_only/imidazoledon/20181009/'
path_to_sig_dict['HIS']['HD1']['DON'] = '~/combs/database/representatives/hb_only/imidazoledon/20181009/'
path_to_sig_dict['HIS']['HE2']['DON'] = '~/combs/database/representatives/hb_only/imidazoledon/20181009/'

dict_corr_dict = defaultdict(dict)
#hydroxyl
dict_corr_dict['SER'] = dict(SER=dict(SER=dict(CB='CB', OG='OG'),
                                      THR=dict(CB='CB', OG1='OG'),
                                      TYR=dict(CZ='CB', OH='OG')))
dict_corr_dict['THR'] = dict(THR=dict(SER=dict(CB='CB', OG='OG1'),
                                      THR=dict(CB='CB', OG1='OG1'),
                                      TYR=dict(CZ='CB', OH='OG1')))
dict_corr_dict['TYR'] = dict(TYR=dict(SER=dict(CB='CZ', OG='OH'),
                                      THR=dict(CB='CZ', OG1='OH'),
                                      TYR=dict(CZ='CZ', OH='OH')))
#carboxamide
dict_corr_dict['ASN'] = dict(ASN=dict(GLN=dict(NE2='ND2', CD='CG', OE1='OD1', CG='CB'),
                                      ASN=dict(ND2='ND2', CG='CG', OD1='OD1', CB='CB')))
dict_corr_dict['GLN'] = dict(GLN=dict(GLN=dict(NE2='NE2', CD='CD', OE1='OE1', CG='CG'),
                                      ASN=dict(ND2='NE2', CG='CD', OD1='OE1', CB='CG')))

#carboxylate
dict_corr_dict['ASP'] = dict(ASP=dict(GLU=dict(OE2='OD2', CD='CG', OE1='OD1', CG='CB'),
                                      ASP=dict(OD2='OD2', CG='CG', OD1='OD1', CB='CB')))
dict_corr_dict['GLU'] = dict(GLU=dict(GLU=dict(OE2='OE2', CD='CD', OE1='OE1', CG='CG'),
                                      ASP=dict(OD2='OE2', CG='CD', OD1='OE1', CB='CG')))

#guanidinium
dict_corr_dict['ARG'] = dict(ARG=dict(ARG=dict(NE='NE', CZ='CZ', NH1='NH1', NH2='NH2')))

#amino
dict_corr_dict['LYS'] = dict(LYS=dict(LYS=dict(CE='CE', NZ='NZ')))
dict_corr_dict['HIS'] = defaultdict(dict)
dict_corr_dict['HIS']['NE2']['ACC'] = dict(HIS=dict(HID=dict(CE1='CE1', NE2='NE2', CD2='CD2'),
                                                    HIE=dict(CE1='CE1', ND1='NE2', CG='CD2')
                                                    ))
dict_corr_dict['HIS']['HE2']['DON'] = dict(HIS=dict(HIE=dict(CE1='CE1', NE2='NE2', CD2='CD2'),
                                                    HID=dict(CE1='CE1', ND1='NE2', CG='CD2'),
                                                    TRP=dict(CD1='CE1', NE1='NE2', CE2='CD2')
                                                    ))
dict_corr_dict['HIS']['ND1']['ACC'] = dict(HIS=dict(HID=dict(CE1='CE1', NE2='ND1', CD2='CG'),
                                                    HIE=dict(CE1='CE1', ND1='ND1', CG='CG')
                                                    ))
dict_corr_dict['HIS']['HD1']['DON'] = dict(HIS=dict(HIE=dict(CE1='CE1', NE2='ND1', CD2='CG'),
                                                    HID=dict(CE1='CE1', ND1='ND1', CG='CG'),
                                                    TRP=dict(CD1='CE1', NE1='ND1', CE2='CG')
                                                    ))

dict_corr_dict['backboneNH'] = defaultdict(dict)
dict_corr_dict['backboneCO'] = defaultdict(dict)
dict_corr_dict['backboneNH']['HIS'] = defaultdict(dict)
dict_corr_dict['backboneCO']['HIS'] = defaultdict(dict)
dict_corr_dict['backboneNH']['ASN'] = defaultdict(dict)
dict_corr_dict['backboneNH']['ASN']['HD22'] = dict(ASN=dict(GLN={'HE21': 'HD22', 'NE2': 'ND2'},
                                                            ASN={'HD21': 'HD22', 'ND2': 'ND2'},
                                                            ALA={'H': 'HD22', 'N': 'ND2'},
                                                            GLY={'H': 'HD22', 'N': 'ND2'}))
dict_corr_dict['backboneNH']['ASN']['HD21'] = dict(ASN=dict(GLN={'HE21': 'HD21', 'NE2': 'ND2'},
                                                            ASN={'HD21': 'HD21', 'ND2': 'ND2'},
                                                            ALA={'H': 'HD21', 'N': 'ND2'},
                                                            GLY={'H': 'HD21', 'N': 'ND2'}))
dict_corr_dict['backboneNH']['GLN'] = defaultdict(dict)
dict_corr_dict['backboneNH']['GLN']['HE22'] = dict(GLN=dict(GLN=dict(HE21='HE22', NE2='NE2'),
                                                    ASN=dict(HD21='HE22', ND2='NE2'),
                                                    ALA=dict(H='HE22', N='NE2'),
                                                    GLY=dict(H='HE22', N='NE2')))
dict_corr_dict['backboneNH']['GLN']['HE21'] = dict(GLN=dict(GLN=dict(HE21='HE21', NE2='NE2'),
                                                    ASN=dict(HD21='HE21', ND2='NE2'),
                                                    ALA=dict(H='HE21', N='NE2'),
                                                    GLY=dict(H='HE21', N='NE2')))
dict_corr_dict['backboneNH']['ARG'] = defaultdict(dict)
dict_corr_dict['backboneNH']['ARG']['HE'] = dict(ARG=dict(GLN=dict(HE21='HE', NE2='NE'),
                                                    ASN=dict(HD21='HE', ND2='NE'),
                                                    ALA=dict(H='HE', N='NE'),
                                                    GLY=dict(H='HE', N='NE')))
dict_corr_dict['backboneNH']['ARG']['HH21'] = dict(ARG=dict(GLN=dict(HE21='HH21', NE2='NH2'),
                                                    ASN=dict(HD21='HH21', ND2='NH2'),
                                                    ALA=dict(H='HH21', N='NH2'),
                                                    GLY=dict(H='HH21', N='NH2')))
dict_corr_dict['backboneNH']['ARG']['HH22'] = dict(ARG=dict(GLN=dict(HE21='HH22', NE2='NH2'),
                                                    ASN=dict(HD21='HH22', ND2='NH2'),
                                                    ALA=dict(H='HH22', N='NH2'),
                                                    GLY=dict(H='HH22', N='NH2')))
dict_corr_dict['backboneNH']['ARG']['HH12'] = dict(ARG=dict(GLN=dict(HE21='HH12', NE2='NH1'),
                                                    ASN=dict(HD21='HH12', ND2='NH1'),
                                                    ALA=dict(H='HH12', N='NH1'),
                                                    GLY=dict(H='HH12', N='NH1')))
dict_corr_dict['backboneNH']['ARG']['HH11'] = dict(ARG=dict(GLN=dict(HE21='HH11', NE2='NH1'),
                                                    ASN=dict(HD21='HH11', ND2='NH1'),
                                                    ALA=dict(H='HH11', N='NH1'),
                                                    GLY=dict(H='HH11', N='NH1')))
dict_corr_dict['backboneNH']['LYS'] = defaultdict(dict)
dict_corr_dict['backboneNH']['LYS']['HZ1'] = dict(LYS=dict(GLN=dict(HE21='HZ1', NE2='NZ'),
                                                    ASN=dict(HD21='HZ1', ND2='NZ'),
                                                    ALA=dict(H='HZ1', N='NZ'),
                                                    GLY=dict(H='HZ1', N='NZ')))
dict_corr_dict['backboneNH']['LYS']['HZ2'] = dict(LYS=dict(GLN=dict(HE21='HZ2', NE2='NZ'),
                                                    ASN=dict(HD21='HZ2', ND2='NZ'),
                                                    ALA=dict(H='HZ2', N='NZ'),
                                                    GLY=dict(H='HZ2', N='NZ')))
dict_corr_dict['backboneNH']['LYS']['HZ3'] = dict(LYS=dict(GLN=dict(HE21='HZ3', NE2='NZ'),
                                                    ASN=dict(HD21='HZ3', ND2='NZ'),
                                                    ALA=dict(H='HZ3', N='NZ'),
                                                    GLY=dict(H='HZ3', N='NZ')))

dict_corr_dict['backboneNH']['HIS']['HE2'] = dict(HIS=dict(GLN=dict(HE21='HE2', NE2='NE2'),
                                                           ASN=dict(HD21='HE2', ND2='NE2'),
                                                           ALA=dict(H='HE2', N='NE2'),
                                                           GLY=dict(H='HE2', N='NE2')))
dict_corr_dict['backboneNH']['HIS']['HD1'] = dict(HIS=dict(GLN=dict(HE21='HD1', NE2='ND1'),
                                                                  ASN=dict(HD21='HD1', ND2='ND1'),
                                                                  ALA=dict(H='HD1', N='ND1'),
                                                                  GLY=dict(H='HD1', N='ND1')))
dict_corr_dict['backboneCO']['HIS']['ND1'][1] = dict(HIS=dict(GLN=dict(OE1='ND1', CD='CG'),
                                                                  ASN=dict(OD1='ND1', CG='CG'),
                                                                  ALA=dict(O='ND1', C='CG'),
                                                                  GLY=dict(O='ND1', C='CG')))
dict_corr_dict['backboneCO']['HIS']['ND1'][2] = dict(HIS=dict(GLN=dict(OE1='ND1', CD='CE1'),
                                                                  ASN=dict(OD1='ND1', CG='CE1'),
                                                                  ALA=dict(O='ND1', C='CE1'),
                                                                  GLY=dict(O='ND1', C='CE1')))
dict_corr_dict['backboneCO']['HIS']['NE2'][1] = dict(HIS=dict(GLN=dict(OE1='NE2', CD='CE1'),
                                                                  ASN=dict(OD1='NE2', CG='CE1'),
                                                                  ALA=dict(O='NE2', C='CE1'),
                                                                  GLY=dict(O='NE2', C='CE1')))
dict_corr_dict['backboneCO']['HIS']['NE2'][2] = dict(HIS=dict(GLN=dict(OE1='NE2', CD='CD2'),
                                                                  ASN=dict(OD1='NE2', CG='CD2'),
                                                                  ALA=dict(O='NE2', C='CD2'),
                                                                  GLY=dict(O='NE2', C='CD2')))
dict_corr_dict['backboneCO']['ASN'] = defaultdict(dict)
dict_corr_dict['backboneCO']['GLN'] = defaultdict(dict)
dict_corr_dict['backboneCO']['ASP'] = defaultdict(dict)
dict_corr_dict['backboneCO']['GLU'] = defaultdict(dict)
dict_corr_dict['backboneCO']['ASN']['OD1'] = dict(ASN=dict(GLN=dict(OE1='OD1', CD='CG'),
                                                    ASN=dict(OD1='OD1', CG='CG'),
                                                    ALA=dict(O='OD1', C='CG'),
                                                    GLY=dict(O='OD1', C='CG')))
dict_corr_dict['backboneCO']['GLN']['OE1'] = dict(GLN=dict(GLN=dict(OE1='OE1', CD='CD'),
                                                    ASN=dict(OD1='OE1', CG='CD'),
                                                    ALA=dict(O='OE1', C='CD'),
                                                    GLY=dict(O='OE1', C='CD')))
dict_corr_dict['backboneCO']['ASP']['OD1'] = dict(ASP=dict(GLN=dict(OE1='OD1', CD='CG'),
                                                    ASN=dict(OD1='OD1', CG='CG'),
                                                    ALA=dict(O='OD1', C='CG'),
                                                    GLY=dict(O='OD1', C='CG')))
dict_corr_dict['backboneCO']['ASP']['OD2'] = dict(ASP=dict(GLN=dict(OE1='OD2', CD='CG'),
                                                    ASN=dict(OD1='OD2', CG='CG'),
                                                    ALA=dict(O='OD2', C='CG'),
                                                    GLY=dict(O='OD2', C='CG')))
dict_corr_dict['backboneCO']['GLU']['OE1'] = dict(GLU=dict(GLN=dict(OE1='OE1', CD='CD'),
                                                    ASN=dict(OD1='OE1', CG='CD'),
                                                    ALA=dict(O='OE1', C='CD'),
                                                    GLY=dict(O='OE1', C='CD')))
dict_corr_dict['backboneCO']['GLU']['OE2'] = dict(GLU=dict(GLN=dict(OE1='OE2', CD='CD'),
                                                    ASN=dict(OD1='OE2', CG='CD'),
                                                    ALA=dict(O='OE2', C='CD'),
                                                    GLY=dict(O='OE2', C='CD')))

backboneNHs = ['HIS', 'ASN', 'GLN', 'ARG', 'LYS']
backboneCOs = ['HIS', 'ASN', 'GLN', 'ASP', 'GLU']

remove_from_df_dict = defaultdict(dict)
remove_from_df_dict['SER'] = {1: {'chain': 'Y', 'name': 'CA'}, 2: {'chain': 'Y', 'name': 'CE1'}}
remove_from_df_dict['THR'] = {1: {'chain': 'Y', 'name': 'CA'}, 2: {'chain': 'Y', 'name': 'CE1'}}
remove_from_df_dict['TYR'] = {1: {'chain': 'Y', 'name': 'CA'}, 2: {'chain': 'Y', 'name': 'CE1'}}
remove_from_df_dict['LYS'] = {1: {'chain': 'Y', 'name': 'CD'}}
remove_from_df_dict['backboneNH'] = {1: {'chain': 'Y', 'name': 'CA'}, 2: {'chain': 'Y', 'name': 'CG', 'resname': 'ASN'},
                                     3: {'chain': 'Y', 'name': 'CD', 'resname': 'GLN'}}
remove_from_df_dict['backboneCO'] = {1: {'chain': 'Y', 'name': 'CA'}, 2: {'chain': 'Y', 'name': 'CB'},
                                     3: {'chain': 'Y', 'name': 'CG', 'resname': 'GLN'}}

class HBNet:
    def __init__(self):
        self.primary = pd.DataFrame()
        self.num_buns = 0
        self.secondary = dict()  # Pose()
        self.num_lig_ints = 0
        self.pose_legs = None


class SecondaryPoses:
    def __init__(self, **kwargs):
        self.poses = None
        self.template = None
        self.vdm_dict = defaultdict(dict)
        self.num_recusion = 0
        self.recursion_limit = kwargs.get('recursion_limit', 2)
        self.path_to_sig_dict = kwargs.get('path_to_sig_dict', path_to_sig_dict)
        self.dict_corr_dict = kwargs.get('dict_corr_dict', dict_corr_dict)
        self.remove_from_df_dict = kwargs.get('remove_from_df_dict', remove_from_df_dict)
        self.do_not_design = kwargs.get('do_not_design')  # format of list of seg_chain_resnums, e.g. [('A', 'A', 5), ...]

    def load_primary_poses(self, poses):
        self.poses = poses

    def load_template(self, template):
        self.template = template

    def find_secondary(self, outdir=None, **kwargs):
        reset_dict = kwargs.get('reset_dict', True)
        j = 0
        for k, pose in enumerate(self.poses):
            self._find_secondary(pose, outdir, k, **kwargs)
            j += 1
            if reset_dict and j == 3:
                self.vdm_dict = defaultdict(dict)
                j = 0
    def _find_secondary(self, pose, outdir=None, pose_num=1, **kwargs):
        """

        :param pose:
        :param template:
        :return:


        1). get set of hb atoms of ligand that are hbonding.
        2a). for each hb atom:
            get vdms that are hbonding.
            2b). for each vdm:
                    get set of possible hb atoms.
                    for each hb atom of vdm:
                        if hb atom is not hbonding (or is not solvent exposed):
                            -load vdms (SC and SC-phipsi only) onto residues of template with CA atoms w/in 10A of hb atom
                            -prune clashing vdms (template and ligand and vdm)
                            -find neighbors (include first shell vdms from pose)
                            -if there are neighbors, check if they satisfy hbonds of the vdm.

        """

        burialdepth = kwargs.get('burialdepth', 1)
        hb_contacts = pose.lig_contacts[pose.lig_contacts.contact_type == 'hb']
        df_hb = get_vdms_hbonding_to_lig(pose, hb_contacts)
        if len(df_hb) > 0:
            vdmrep = VdmReps(df_hb, **dict(grouping=pose.groupby))
            vdmrep.find_all_reps_dict(rmsd=0.4)
            df_hb = pd.concat(vdmrep.reps_dict.values())
            has_hbs = ''
        else:
            has_hbs = '_no_hb'

        df_hb_gr = df_hb.groupby(pose.groupby)
        print('Searching through ' + str(len(df_hb_gr)) + ' vdMs for secondary H-bonders...')
        recurse = kwargs.get('recurse')
        if recurse is not None:
            if self.num_recusion >= self.recursion_limit:
                print('Breaking recursion')
                self.num_recusion -= 1
                for i, (vdm_name, vdm_df) in enumerate(df_hb_gr):
                    if recurse is not None:
                        print('recursive vdM number ' + str(i + 1) + '...')
                    else:
                        print('first-shell vdM number ' + str(i + 1) + '...')
                    vdm_df = vdm_df.copy()
                    print(set(vdm_df.resname_vdm).pop())
                    num_accs, num_dons = get_bun_hb_atoms(vdm_df, self.template, burial_depth=burialdepth,
                                                          contacts=None, pose=pose, grouping=None, **kwargs)
                    hbnet = HBNet()
                    hbnet.num_buns = num_accs + num_dons
                    hbnet.primary = vdm_df
                    pose.hb_net.append(hbnet)
            else:
                for i, (vdm_name, vdm_df) in enumerate(df_hb_gr):
                    if recurse is not None:
                        print('recursive vdM number ' + str(i + 1) + '...')
                    else:
                        print('first-shell vdM number ' + str(i + 1) + '...')
                    vdm_df = vdm_df.copy()
                    print(set(vdm_df.resname_vdm).pop())
                    bun_accs, bun_dons = get_bun_hb_atoms(vdm_df, self.template, burial_depth=burialdepth,
                                                          contacts=None, pose=pose, grouping=None)
                    hbnet = HBNet()
                    hbnet.num_buns = len(bun_accs) + len(bun_dons)
                    hbnet.primary = vdm_df

                    if recurse is not None:
                        num_accs, num_dons = get_bun_hb_atoms(vdm_df, self.template, burial_depth=burialdepth,
                                                              contacts=None, pose=pose, grouping=None, **kwargs)
                        hbnet.num_buns = num_accs + num_dons
                    num_vdm_recusion = self.num_recusion
                    for n, atom_row in bun_accs.iterrows():
                        self.num_recusion = num_vdm_recusion
                        print('loading and searching for H-bonders to buried unsatisfied ' + atom_row['name'] + '...')
                        self.load_sec(atom_row, hb_type='ACC')
                        self.find_sec(pose, vdm_df, hbnet, atom_row, hb_type='ACC', **kwargs)
                    for n, atom_row in bun_dons.iterrows():
                        self.num_recusion = num_vdm_recusion
                        print('loading and searching for H-bonders to buried unsatisfied ' + atom_row['name'] + '...')
                        self.load_sec(atom_row, hb_type='DON')
                        self.find_sec(pose, vdm_df, hbnet, atom_row, hb_type='DON', **kwargs)
                    pose.hb_net.append(hbnet)
        else:
            self.num_recusion = 0

            for i, (vdm_name, vdm_df) in enumerate(df_hb_gr):
                print('first-shell vdM number ' + str(i + 1) + '...')
                vdm_df = vdm_df.copy()
                print(set(vdm_df.resname_vdm).pop())
                bun_accs, bun_dons = get_bun_hb_atoms(vdm_df, self.template, burial_depth=burialdepth,
                                                      contacts=None, pose=pose, grouping=None)
                accs, dons = get_hb_atoms(vdm_df)
                hbnet = HBNet()
                hbnet.num_buns = len(bun_accs) + len(bun_dons)
                hbnet.primary = vdm_df
                num_vdm_recusion = self.num_recusion
                for n, atom_row in accs.iterrows():  # find buttressing interactions for all polar atoms in first shell
                    if atom_row['resname_vdm'] != 'TRP':
                        self.num_recusion = num_vdm_recusion
                        print('loading and searching for H-bonders to buried unsatisfied ' + atom_row['name'] + '...')
                        self.load_sec(atom_row, hb_type='ACC')
                        self.find_sec(pose, vdm_df, hbnet, atom_row, hb_type='ACC', **kwargs)
                # for n, atom_row in bun_dons.iterrows():
                for n, atom_row in dons.iterrows():
                    if atom_row['resname_vdm'] != 'TRP':
                        self.num_recusion = num_vdm_recusion
                        print('loading and searching for H-bonders to buried unsatisfied ' + atom_row['name'] + '...')
                        self.load_sec(atom_row, hb_type='DON')
                        self.find_sec(pose, vdm_df, hbnet, atom_row, hb_type='DON', **kwargs)
                pose.hb_net.append(hbnet)

        if outdir is not None:
            if outdir[-1] != '/':
                outdir += '/'

            try:
                os.makedirs(outdir)
            except FileExistsError:
                pass
            with open(outdir + 'pose' + str(pose_num) + has_hbs + '.pkl', 'wb') as outfile:
                pickle.dump(pose, outfile)

    def load_sec(self, atom_row, hb_type):
        sel = self.template.pdb.select('name CA within 10 of c',
                                       c=np.array([atom_row.c_x, atom_row.c_y,
                                                   atom_row.c_z]))
        if self.do_not_design is None:
            template_seg_chain_resnums = set(
                zip(sel.getSegnames(), sel.getChids(), sel.getResnums())) - \
                                     {atom_row.seg_chain_resnum}
        else:
            template_seg_chain_resnums = set(
                zip(sel.getSegnames(), sel.getChids(), sel.getResnums())) - \
                                         {atom_row.seg_chain_resnum} - set(self.do_not_design)

        ## MAKE DF of only hbonding vdms since that's all we are looking for here.
        resname = atom_row.resname_vdm
        if resname == 'HIS':
            try:
                name = atom_row['name']
                path_to_sig = self.path_to_sig_dict[resname][name][hb_type]
                dict_corr = self.dict_corr_dict[resname][name][hb_type]
            except KeyError:
                print('HIS KeyError:' + resname + ', ' + name + ', ' + hb_type)
                return
        else:
            try:
                path_to_sig = self.path_to_sig_dict[resname]
                dict_corr = self.dict_corr_dict[resname]
            except KeyError:
                print('KeyError:' + resname)
                return
        df_lig_corr = make_df_corr(dict_corr)
        remove_from_df = self.remove_from_df_dict[resname]

        sc_set = {f[:3] for f in [ff for ff in os.listdir(path_to_sig + 'SC') if ff[-3:] == 'pkl']}
        phi_psi_set = {f[:3] for f in [ff for ff in os.listdir(path_to_sig + 'PHI_PSI') if ff[-3:] == 'pkl']}
        hydrophobes_set = {'ALA', 'PHE', 'LEU', 'VAL', 'PRO', 'ILE', 'MET'}
        cys_set = {'CYS'}
        gly_set = {'GLY'}
        charged_set = {'ASP', 'GLU', 'ARG', 'LYS'}
        seq_csts = defaultdict(dict)
        for seg, ch, rn in template_seg_chain_resnums:
            for label in ['SC', 'PHI_PSI']:
                if label == 'SC':
                    resnames = sc_set - hydrophobes_set - cys_set - gly_set
                if label == 'PHI_PSI':
                    resnames = phi_psi_set - hydrophobes_set - cys_set - gly_set
                seq_csts[(seg, ch, rn)][label] = resnames

        if (resname in self.vdm_dict.keys()) and (hb_type in self.vdm_dict[resname].keys()):
            additional_loaded = self.vdm_dict[resname][hb_type].load_additional(self.template, seq_csts)
            if additional_loaded:
                self.vdm_dict[resname][hb_type].set_neighbors(rmsd=1.5)

        else:
            kwargs = dict(name='vdm', path=path_to_sig, sequence_csts=seq_csts,
                          ligand_iFG_corr=df_lig_corr, remove_from_df=remove_from_df)
            self.vdm_dict[resname][hb_type] = VdM(**kwargs)
            self.vdm_dict[resname][hb_type].load(self.template)
            self.vdm_dict[resname][hb_type].set_neighbors(rmsd=1.5)

        atom_type = atom_row['atom_type_label']
        if (resname in backboneNHs) and (hb_type == 'DON') and (atom_type) == 'h_pol':
            print('loading backboneNH vdMs')
            try:
                name = atom_row['name']
                path_to_sig = self.path_to_sig_dict['backboneNH']
                dict_corr = self.dict_corr_dict['backboneNH'][resname][name]
                df_lig_corr = make_df_corr(dict_corr)
                remove_from_df = self.remove_from_df_dict['backboneNH']
            except KeyError:
                print('KeyError: backboneNH and ' + resname)
                return

            if 'backboneNH' in self.vdm_dict.keys():
                kwargs = dict(name='vdm', path=path_to_sig,
                              ligand_iFG_corr=df_lig_corr, remove_from_df=remove_from_df)
                additional_loaded = self.vdm_dict['backboneNH']['parent'].load_additional(self.template, seq_csts)
                if additional_loaded:
                    self.vdm_dict['backboneNH']['parent'].set_neighbors(rmsd=0.9)
                    print('copying additional backboneNH vdms to ' + resname + '...')
                    self.vdm_dict['backboneNH'][resname][name] = VdM(**kwargs)
                    self.vdm_dict['backboneNH'][resname][name].ligand_iFG_corr_sorted = pd.merge(
                        self.vdm_dict['backboneNH']['parent'].ligand_iFG_corr_sorted[['resname', 'name']],
                        self.vdm_dict['backboneNH'][resname][name].ligand_iFG_corr, on=['resname', 'name'])
                    
            else:
                kwargs = dict(name='vdm', path=path_to_sig, sequence_csts=seq_csts,
                              ligand_iFG_corr=df_lig_corr, remove_from_df=remove_from_df)
                self.vdm_dict['backboneNH'] = defaultdict(dict)
                self.vdm_dict['backboneNH']['parent'] = VdM(**kwargs)
                self.vdm_dict['backboneNH']['parent'].load(self.template)
                self.vdm_dict['backboneNH']['parent'].set_neighbors(rmsd=0.9)

                print('copying backboneNH vdms to ' + resname + '...')
                self.vdm_dict['backboneNH'][resname][name] = VdM(**kwargs)
                self.vdm_dict['backboneNH'][resname][name].ligand_iFG_corr_sorted = pd.merge(self.vdm_dict['backboneNH']['parent'].ligand_iFG_corr_sorted[['resname', 'name']],
                                                                                       self.vdm_dict['backboneNH'][resname][name].ligand_iFG_corr, on=['resname', 'name'])

        if (resname in backboneCOs) and (hb_type == 'ACC'):
            print('loading backboneCO vdMs')
            name = atom_row['name']
            path_to_sig = self.path_to_sig_dict['backboneCO']
            remove_from_df = self.remove_from_df_dict['backboneCO']
            if resname == 'HIS':
                try:
                    dict_corr = self.dict_corr_dict['backboneCO'][resname][name][1]
                    df_lig_corr = make_df_corr(dict_corr)
                except KeyError:
                    print('HIS KeyError:' + resname + ', ' + name + ', ' + hb_type)
                    return
            else:
                try:
                    dict_corr = self.dict_corr_dict['backboneCO'][resname][name]
                    df_lig_corr = make_df_corr(dict_corr)
                except KeyError:
                    print('KeyError: backboneCO and ' + resname)
                    return
            kwargs = dict(name='vdm', path=path_to_sig, sequence_csts=seq_csts,
                          ligand_iFG_corr=df_lig_corr, remove_from_df=remove_from_df)

            if 'backboneCO' in self.vdm_dict.keys():
                additional_loaded = self.vdm_dict['backboneCO']['parent'].load_additional(self.template, seq_csts)
                if additional_loaded:
                    self.vdm_dict['backboneCO']['parent'].set_neighbors(rmsd=0.9)
                    if resname == 'HIS':
                        print('copying additional backboneCO vdms to ' + resname + '...')
                        self.vdm_dict['backboneCO'][resname][name] = dict()
                        for ii in range(1, 3):
                            dict_corr = self.dict_corr_dict['backboneCO'][resname][name][ii]
                            df_lig_corr = make_df_corr(dict_corr)
                            kwargs = dict(name='vdm', path=path_to_sig,
                                          ligand_iFG_corr=df_lig_corr, remove_from_df=remove_from_df)
                            self.vdm_dict['backboneCO'][resname][name][ii] = VdM(**kwargs)
                            self.vdm_dict['backboneCO'][resname][name][ii].ligand_iFG_corr_sorted = pd.merge(
                                self.vdm_dict['backboneCO']['parent'].ligand_iFG_corr_sorted[['resname', 'name']],
                                self.vdm_dict['backboneCO'][resname][name][ii].ligand_iFG_corr, on=['resname', 'name'])
                    else:
                        print('copying additional backboneCO vdms to ' + resname + '...')
                        self.vdm_dict['backboneCO'][resname][name] = VdM(**kwargs)
                        self.vdm_dict['backboneCO'][resname][name].ligand_iFG_corr_sorted = pd.merge(
                            self.vdm_dict['backboneCO']['parent'].ligand_iFG_corr_sorted[['resname', 'name']],
                            self.vdm_dict['backboneCO'][resname][name].ligand_iFG_corr, on=['resname', 'name'])

            else:
                self.vdm_dict['backboneCO'] = defaultdict(dict)
                self.vdm_dict['backboneCO']['parent'] = VdM(**kwargs)
                self.vdm_dict['backboneCO']['parent'].load(self.template)
                self.vdm_dict['backboneCO']['parent'].set_neighbors(rmsd=0.9)

                print('copying backboneCO vdms to ' + resname + '...')
                if resname == 'HIS':
                    self.vdm_dict['backboneCO'][resname][name] = dict()
                    for ii in range(1, 3):
                        dict_corr = self.dict_corr_dict['backboneCO'][resname][name][ii]
                        df_lig_corr = make_df_corr(dict_corr)
                        kwargs = dict(name='vdm', path=path_to_sig, sequence_csts=seq_csts,
                                      ligand_iFG_corr=df_lig_corr, remove_from_df=remove_from_df)
                        self.vdm_dict['backboneCO'][resname][name][ii] = VdM(**kwargs)
                        self.vdm_dict['backboneCO'][resname][name][ii].ligand_iFG_corr_sorted = pd.merge(
                            self.vdm_dict['backboneCO']['parent'].ligand_iFG_corr_sorted[['resname', 'name']],
                            self.vdm_dict['backboneCO'][resname][name][ii].ligand_iFG_corr, on=['resname', 'name'])
                else:
                    self.vdm_dict['backboneCO'][resname][name] = VdM(**kwargs)
                    self.vdm_dict['backboneCO'][resname][name].ligand_iFG_corr_sorted = pd.merge(
                        self.vdm_dict['backboneCO']['parent'].ligand_iFG_corr_sorted[['resname', 'name']],
                        self.vdm_dict['backboneCO'][resname][name].ligand_iFG_corr, on=['resname', 'name'])


    def find_sec(self, pose, vdm_df, hbnet, atom_row, hb_type, **kwargs):
        first_shell_df = pose.vdms_sidechains[pose.vdms_sidechains.seg_chain_resnum
                                              != atom_row.seg_chain_resnum]
        resname = atom_row.resname_vdm
        name = atom_row['name']
        sec_pose = Pose()
        coords = _get_frag_coords(vdm_df, self.vdm_dict[resname][hb_type])
        try:
            ind_neighbors = \
                self.vdm_dict[resname][hb_type].neighbors.radius_neighbors(coords, return_distance=False)[0]
        except ValueError:
            print('ValueError: duplicated vdM? skipping...')
            print(resname, hb_type)
            print(coords)
            ind_neighbors = np.array([])

        df_to_append = []
        if ind_neighbors.size > 0:
            df_uniq = self.vdm_dict[resname][hb_type].dataframe_iFG_coords.iloc[
                ind_neighbors].drop_duplicates()
            df_to_append = pd.concat(
                [self.vdm_dict[resname][hb_type].dataframe_grouped.get_group(g) for g in
                 df_uniq.values],
                sort=False, ignore_index=True)
            print('found secondary vdms...')
            
        if ('backboneNH' in self.vdm_dict.keys()) and (resname in self.vdm_dict['backboneNH']) and (name in self.vdm_dict['backboneNH'][resname]):
            coords = _get_frag_coords(vdm_df, self.vdm_dict['backboneNH'][resname][name])
            try:
                ind_neighbors_bb = \
                    self.vdm_dict['backboneNH']['parent'].neighbors.radius_neighbors(coords,
                                                                                          return_distance=False)[0]
            except ValueError:
                print('ValueError: duplicated vdM? skipping...')
                print(resname, name)
                print(coords)
                ind_neighbors_bb = np.array([])

            if ind_neighbors_bb.size > 0:
                df_uniq_bb = self.vdm_dict['backboneNH']['parent'].dataframe_iFG_coords.iloc[
                    ind_neighbors_bb].drop_duplicates()
                df_to_append_bb = pd.concat(
                    [self.vdm_dict['backboneNH']['parent'].dataframe_grouped.get_group(g) for g in
                     df_uniq_bb.values],
                    sort=False, ignore_index=True)
                print('found backboneNH secondary vdms...')
                if len(df_to_append) > 0:
                    df_to_append = pd.concat((df_to_append, df_to_append_bb))
                else:
                    df_to_append = df_to_append_bb

        if ('backboneCO' in self.vdm_dict.keys()) and (resname in self.vdm_dict['backboneCO']) and (
            name in self.vdm_dict['backboneCO'][resname]):
            if resname == 'HIS':
                for jj in range(1, 3):
                    coords = _get_frag_coords(vdm_df, self.vdm_dict['backboneCO'][resname][name][jj])
                    try:
                        ind_neighbors_bb = \
                            self.vdm_dict['backboneCO']['parent'].neighbors.radius_neighbors(coords,
                                                                                                      return_distance=False)[0]
                    except ValueError:
                        print('ValueError: duplicated vdM? skipping...')
                        print(resname, name)
                        print(coords)
                        ind_neighbors_bb = np.array([])

                    if ind_neighbors_bb.size > 0:
                        df_uniq_bb = self.vdm_dict['backboneCO']['parent'].dataframe_iFG_coords.iloc[
                            ind_neighbors_bb].drop_duplicates()
                        df_to_append_bb = pd.concat(
                            [self.vdm_dict['backboneCO']['parent'].dataframe_grouped.get_group(g) for g in
                             df_uniq_bb.values],
                            sort=False, ignore_index=True)
                        print('found backboneCO secondary vdms...')
                        if len(df_to_append) > 0:
                            df_to_append = pd.concat((df_to_append, df_to_append_bb))
                        else:
                            df_to_append = df_to_append_bb
            else:
                coords = _get_frag_coords(vdm_df, self.vdm_dict['backboneCO'][resname][name])
                try:
                    ind_neighbors_bb = \
                        self.vdm_dict['backboneCO']['parent'].neighbors.radius_neighbors(coords,
                                                                                              return_distance=False)[0]
                except ValueError:
                    print('ValueError: duplicated vdM? skipping...')
                    ind_neighbors_bb = np.array([])

                if ind_neighbors_bb.size > 0:
                    df_uniq_bb = self.vdm_dict['backboneCO']['parent'].dataframe_iFG_coords.iloc[
                        ind_neighbors_bb].drop_duplicates()
                    df_to_append_bb = pd.concat(
                        [self.vdm_dict['backboneCO']['parent'].dataframe_grouped.get_group(g) for g in
                         df_uniq_bb.values], sort=False, ignore_index=True)
                    print('found backboneCO secondary vdms...')
                    if len(df_to_append) > 0:
                        df_to_append = pd.concat((df_to_append, df_to_append_bb))
                    else:
                        df_to_append = df_to_append_bb

        if len(first_shell_df) > 0 and len(df_to_append) > 0:
            _vdms = pd.concat((first_shell_df, df_to_append)).drop_duplicates()
        elif len(first_shell_df) > 0:
            _vdms = first_shell_df
        elif len(df_to_append) > 0:
            _vdms = df_to_append
        else:
            _vdms = []

        if len(_vdms) > 0:
            sec_pose._vdms = _vdms
            vdm_df = vdm_df.copy()
            vdm_df['lig_resname'] = vdm_df.resname
            vdm_df['lig_name'] = vdm_df.name
            sec_pose.ligand = pd.concat((vdm_df, pose.ligand), sort=False)
            sec_pose.set_nonclashing_vdms()
            if len(sec_pose.vdms) > 0:
                sec_pose._vdms = None
                print('checking secondary pose for H-bond...')
                lig_con = Contact(sec_pose.vdms_sidechains, vdm_df)
                lig_con.find()
                if len(lig_con.df_contacts) > 0:
                    hb_contacts = lig_con.df_contacts[
                        (lig_con.df_contacts.name_t == atom_row['name'])
                        & (lig_con.df_contacts.contact_type == 'hb')]
                    if hb_type == 'ACC':
                        hb_contacts = hb_contacts[hb_contacts.atom_type_label_q == 'h_pol']
                    if len(hb_contacts) > 0:
                        df_hb_sec = get_vdms_hbonding_to_lig(sec_pose, hb_contacts)
                        if len(df_hb_sec) > 0:
                            df_hb_sec = df_hb_sec.drop('num_tag', axis=1).drop_duplicates()
                            sec_pose.vdms_sidechains = df_hb_sec
                            sec_pose.vdms = df_hb_sec
                            print('secondary H-bonders found!')
                            hbnet.secondary[name] = sec_pose
                            lig_con = Contact(sec_pose.vdms_sidechains, vdm_df)
                            lig_con.find()
                            sec_pose.lig_contacts = lig_con.df_contacts
                            kwargs['recurse'] = True
                            print('Beginning recursion...')
                            self.num_recusion += 1
                            self._find_secondary(sec_pose, **kwargs)


def get_vdms_hbonding_to_lig(pose, hb_contacts=None):
    groupby_q = [p + '_q' for p in pose.groupby]
    if hb_contacts is None:
        df_hb = pd.merge(pose.vdms_sidechains,
                         pose.lig_contacts[pose.lig_contacts.contact_type == 'hb'][groupby_q].drop_duplicates(),
                         left_on=pose.groupby,
                         right_on=groupby_q).drop(groupby_q, axis=1)
    else:
        df_hb = pd.merge(pose.vdms_sidechains,
                         hb_contacts[groupby_q].drop_duplicates(),
                         left_on=pose.groupby,
                         right_on=groupby_q).drop(groupby_q, axis=1)
    return df_hb


def get_hb_atoms(df, grouping=None):
    atoms_acc = df[df.c_A1_x.notna()]
    atoms_don = df[df.c_D_x.notna()]
    heavy_atoms_don = df[df.atom_type_label != 'h_pol']
    if grouping is None:
        grouping = ['str_index', 'name']
    if len(atoms_don) > 0 and len(heavy_atoms_don) > 0:
        don_merged = pd.merge(atoms_don, heavy_atoms_don[grouping].drop_duplicates(), on=grouping, how='outer',
                              indicator=True)
        atoms_don = don_merged[don_merged._merge == 'left_only'].drop('_merge', axis=1)
    return atoms_acc, atoms_don


def count_num_donors_acceptors(df):
    num_acc = len(df[df.c_A1_x.notna()])
    num_don = len(df[df.c_D_x.notna()])
    num_don -= len(df[df.c_H2_x.notna()])  # removes heavy atoms from donor count, such that donors only counts Hs.
    return num_acc + num_don


def get_bun_hb_atoms(df, template, burial_depth=1, contacts=None, pose=None, grouping=None, return_satisfied=False, **kwargs):
    """returns dataframe of buried unsatisfied h-bonding atoms from df.
    returns acc dataframe and don dataframe"""
    bun_atoms_acc = pd.DataFrame()
    bun_atoms_don = pd.DataFrame()
    df['distance_to_hull'] = get_distance_to_hull(df, template)

    if kwargs:
        omit = kwargs.get('omit')
        groupby = kwargs.get('groupby', 'str_index')
        pl = pose.ligand.copy()
        pl2 = pose.ligand.copy()
        pl.set_index(groupby, inplace=True, drop=False)
        num_accs = []
        num_dons = []
        for n, g in pl2.groupby(groupby):
            if set(g.resname).pop() == omit:
                continue
            g = g.copy()
            g.set_index(groupby, inplace=True, drop=False)
            lig_bb = pd.concat((pl[~pl.index.isin(g.index)], df, template.dataframe), sort=False)
            lig_bb.reset_index(inplace=True)
            g.reset_index(inplace=True, drop=True)
            df_con = Contact(lig_bb, g)
            df_con.find()
            contacts_ = df_con.df_contacts
            acc, don = get_bun_hb_atoms(g, template, contacts=contacts_)
            num_accs.append(len(acc))
            num_dons.append(len(don))
        acc, don = get_bun_hb_atoms(df, template, burial_depth, contacts, pose, grouping)
        num_accs.append(len(acc))
        num_dons.append(len(don))
        return sum(num_accs), sum(num_dons)
    
    atoms_acc = df[df.c_A1_x.notna()]
    atoms_don = df[df.c_D_x.notna()]
    heavy_atoms_don = df[df.atom_type_label != 'h_pol']
    if grouping is None:
        grouping = ['str_index', 'name']
    if len(atoms_don) > 0 and len(heavy_atoms_don) > 0:
        don_merged = pd.merge(atoms_don, heavy_atoms_don[grouping].drop_duplicates(), on=grouping, how='outer',
                              indicator=True)
        atoms_don = don_merged[don_merged._merge == 'left_only'].drop('_merge', axis=1)
        
    if contacts is None and pose is not None:
        lig_bb = pd.concat((pose.ligand, template.dataframe), sort=False)
        df_con = Contact(lig_bb, df)
        df_con.find()
        contacts = df_con.df_contacts
    elif contacts is None and pose is None:
        if len(atoms_acc) > 0:
            bun_atoms_acc = atoms_acc[atoms_acc.distance_to_hull >= burial_depth]
        if len(atoms_don) > 0:
            bun_atoms_don = atoms_don[atoms_don.distance_to_hull >= burial_depth]
        return bun_atoms_acc, bun_atoms_don
    hb_contacts = contacts[contacts.contact_type == 'hb']

    if return_satisfied:
        sat_acc = pd.DataFrame()
        sat_don = pd.DataFrame()
        if len(atoms_acc) > 0:
            atoms_acc, sat_acc = remove_satisfied_accs(atoms_acc, hb_contacts, grouping, return_satisfied)
            atoms_acc = atoms_acc.copy()
            sat_acc = sat_acc.copy()

        if len(atoms_don) > 0:
            atoms_don, sat_don = remove_satisfied_dons(atoms_don, hb_contacts, grouping, return_satisfied)
            atoms_don = atoms_don.copy()
            sat_don = sat_don.copy()
    else:
        if len(atoms_acc) > 0:
            atoms_acc = remove_satisfied_accs(atoms_acc, hb_contacts, grouping).copy()

        if len(atoms_don) > 0:
            atoms_don = remove_satisfied_dons(atoms_don, hb_contacts, grouping).copy()

    if len(atoms_acc) > 0:
        bun_atoms_acc = atoms_acc[atoms_acc.distance_to_hull >= burial_depth]
        
    if len(atoms_don) > 0:
        bun_atoms_don = atoms_don[atoms_don.distance_to_hull >= burial_depth]

    if return_satisfied:
        return bun_atoms_acc, bun_atoms_don, sat_acc, sat_don
    else:
        return bun_atoms_acc, bun_atoms_don


def get_num_bun_hb_atoms(df, template, pose, burial_depth=1, **kwargs):
    """returns dataframe of buried unsatisfied h-bonding atoms from df.
    returns acc dataframe and don dataframe"""
    df = pd.concat((df, pose.ligand), sort=False)

    if kwargs:
        omit = kwargs.get('omit')
        groupby = kwargs.get('groupby', 'str_index')
        df2 = df.copy()
        df.set_index(groupby, inplace=True, drop=False)
        num_accs = []
        num_dons = []
        for n, g in df2.groupby(groupby):
            if set(g.resname).pop() == omit:
                continue
            g = g.copy()
            g.set_index(groupby, inplace=True, drop=False)
            lig_bb = pd.concat((df[~df.index.isin(g.index)], template.dataframe), sort=False)
            lig_bb.reset_index(inplace=True)
            g.reset_index(inplace=True, drop=True)
            df_con = Contact(lig_bb, g)
            df_con.find()
            contacts_ = df_con.df_contacts
            acc, don = get_bun_hb_atoms(g, template, burial_depth, contacts=contacts_)
            num_accs.append(len(acc))
            num_dons.append(len(don))
        return sum(num_accs), sum(num_dons)


def _get_bun_hb_atoms(df, template, pose, burial_depth=1, **kwargs):
    """returns dataframe of buried unsatisfied h-bonding atoms from df.
    returns acc dataframe and don dataframe"""
    df = pd.concat((df, pose.ligand), sort=False)
    if 'num_tag' in df.columns:
        df = df.drop('num_tag', axis=1)

    if kwargs:
        omit = kwargs.get('omit')
        groupby = kwargs.get('groupby', 'str_index')
        df2 = df.copy()
        df.set_index(groupby, inplace=True, drop=False)
        accs = []
        dons = []
        for n, g in df2.groupby(groupby):
            if set(g.resname).pop() == omit:
                continue
            g = g.copy()
            g.set_index(groupby, inplace=True, drop=False)
            lig_bb = pd.concat((df[~df.index.isin(g.index)], template.dataframe), sort=False)
            lig_bb.reset_index(inplace=True)
            g.reset_index(inplace=True, drop=True)
            df_con = Contact(lig_bb, g)
            df_con.find()
            contacts_ = df_con.df_contacts
            acc, don = get_bun_hb_atoms(g, template, burial_depth, contacts=contacts_)
            if len(acc) > 0:
                accs.append(acc)
            if len(don) > 0:
                dons.append(don)
        if accs:
            df_accs = pd.concat(accs, sort=False).drop_duplicates()
        else:
            df_accs = pd.DataFrame(columns=df.columns)
        if dons:
            df_dons = pd.concat(dons, sort=False).drop_duplicates()
        else:
            df_dons = pd.DataFrame(columns=df.columns)

        return df_accs, df_dons


def remove_satisfied_accs(atoms_acc, hb_contacts, grouping, return_satisfied=False):
    hb_contacts = hb_contacts[hb_contacts.atom_type_label_q == 'h_pol']
    if return_satisfied:
        if len(hb_contacts) > 0:
            t_keys = [p + '_t' for p in grouping]
            merged_acc = pd.merge(atoms_acc, hb_contacts[t_keys].drop_duplicates(), how='outer', left_on=grouping,
                                  right_on=t_keys, indicator=True)
            atoms_acc = merged_acc[merged_acc._merge == 'left_only'].drop('_merge', axis=1).drop(t_keys, axis=1)
            sat_atoms_acc = merged_acc[merged_acc._merge == 'both'].drop('_merge', axis=1).drop(t_keys, axis=1)
            return atoms_acc, sat_atoms_acc
        else:
            return atoms_acc, pd.DataFrame(columns=atoms_acc.columns)
    else:
        if len(hb_contacts) > 0:
            t_keys = [p + '_t' for p in grouping]
            merged_acc = pd.merge(atoms_acc, hb_contacts[t_keys].drop_duplicates(), how='outer', left_on=grouping,
                                  right_on=t_keys, indicator=True)
            atoms_acc = merged_acc[merged_acc._merge == 'left_only'].drop('_merge', axis=1).drop(t_keys, axis=1)
            return atoms_acc
        else:
            return atoms_acc
    
    
def remove_satisfied_dons(atoms_don, hb_contacts, grouping, return_satisfied=False):
    if return_satisfied:
        if len(hb_contacts) > 0:
            t_keys = [p + '_t' for p in grouping]
            merged_don = pd.merge(atoms_don, hb_contacts[t_keys].drop_duplicates(), how='outer', left_on=grouping,
                                  right_on=t_keys, indicator=True)
            atoms_don = merged_don[merged_don._merge == 'left_only'].drop('_merge', axis=1).drop(t_keys, axis=1)
            sat_atoms_don = merged_don[merged_don._merge == 'both'].drop('_merge', axis=1).drop(t_keys, axis=1)
            return atoms_don, sat_atoms_don
        else:
            return atoms_don, pd.DataFrame(columns=atoms_don.columns)
    else:
        t_keys = [p + '_t' for p in grouping]
        merged_don = pd.merge(atoms_don, hb_contacts[t_keys].drop_duplicates(), how='outer', left_on=grouping,
                              right_on=t_keys, indicator=True)
        atoms_don = merged_don[merged_don._merge == 'left_only'].drop('_merge', axis=1).drop(t_keys, axis=1)
        return atoms_don


def get_distance_to_hull(df, template):
    return template.alpha_hull.get_pnts_distance(df[['c_x', 'c_y', 'c_z']].values)


def _get_frag_coords(df, vdm):
    df_corr = pd.merge(vdm.ligand_iFG_corr_sorted[['lig_resname', 'lig_name']].drop_duplicates(), df.drop_duplicates(),
                       how='inner', left_on=['lig_resname', 'lig_name'], right_on=['resname_vdm', 'name'], sort=False)
    return df_corr[['c_x', 'c_y', 'c_z']].values.reshape(1, -1)


class PoseLegs:
    def __init__(self):
        self.poselegs = []
        self.num_buns = []
        self.poselegs_uniq = []
        self.num_buns_uniq = []

    def get_poselegs(self, hbnet):
        self.poselegs.append(hbnet.primary)
        self.num_buns.append(hbnet.num_buns)

        if hbnet.secondary:
            for pose in hbnet.secondary.values():
                for hbnet_ in pose.hb_net:
                    hbnet_ = copy.deepcopy(hbnet_)
                    concated = pd.concat((hbnet.primary, hbnet_.primary), sort=False)
                    hbnet_.primary = concated
                    self.get_poselegs(hbnet_)

    def drop_duplicates(self):
        dropped = [pl.drop(['num_tag', 'rmsd_from_centroid'], axis=1).sort_values(
            ['iFG_count', 'vdM_count', 'query_name', 'seg_chain_resnum', 'name']).reset_index(drop=True) for pl in
                   self.poselegs]
        for i, (nb, pl) in enumerate(zip(self.num_buns, dropped)):
            drop = False
            for pl_ in dropped[i + 1:]:
                if pl.equals(pl_):
                    drop = True
            if not drop:
                self.poselegs_uniq.append(pl)
                self.num_buns_uniq.append(nb)


@jit("f8[:](f8[:], f8[:,:])", nopython=True)
def _dee(es, ep):
    to_del = np.zeros(es.size)
    pair_ens = np.zeros(es.size)
    w = 0
    for i in range(es.size):
        for j in range(es.size):
            if j == i:
                continue
            for k in range(es.size):
                pair_ens[k] = ep[i, j] - ep[k, j]
            cond = es[i] - es[j] + pair_ens.min()
            if cond > 0.0:
                to_del[w] = i
                w += 1
                break
    return to_del[:w]


def remove_dups(ar1, ar2, return_inds=False):
    sh1 = ar1.shape[0]
    stacked = np.vstack((ar1, ar2))
    un, inds = np.unique(stacked, axis=0, return_index=True)
    inds_ = set(inds)
    inds1 = set(range(sh1)) & inds_
    inds__ = list(np.array(list(inds_ - inds1)) - sh1)
    if return_inds:
        inds1 = list(inds1)
        return ar1[inds1], ar2[inds__], inds1, inds__
    else:
        return ar1[np.array(list(inds1))], ar2[inds__]


def is_subset(ar1, ar2):
    if ar1.shape[0] < ar2.shape[0]:
        ar1_ = ar1
        ar1 = ar2
        ar2 = ar1_
    sh1 = ar1.shape[0]
    stacked = np.vstack((ar1, ar2))
    un, inds = np.unique(stacked, axis=0, return_index=True)
    inds_ = set(inds)
    inds1 = set(range(sh1)) & inds_
    inds__ = list(np.array(list(inds_ - inds1)) - sh1)
    if inds__:
        return False
    else:
        return True
