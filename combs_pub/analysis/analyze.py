__all__ = ['Analyze']

import pandas as pd
import numpy as np
from ..apps.constants import one_letter_code
import traceback
import os
import prody as pr


co_bb_atoms = ['C', 'O']
hnca_bb_atoms = ['N', 'H', 'HA', '1HA', '2HA', '3HA', 'CA', 'HA1', 'HA2', 'HA3']
o_bb_atoms = ['O']
h_bb_atoms = ['H']
term_set = {'OXT', 'H1', 'H2', 'H3', '1H', '2H', '3H'}
nterm_atoms = ['1H', '2H', '3H', 'H1', 'H2', 'H3']
bb_atoms = ['N', 'H', 'HA', '1HA', '2HA', '3HA', 'H1', 'H2', 'H3', 'C', 'CA',
            'O', 'OXT', '1H', '2H', '3H', 'HA1', 'HA2', 'HA3']
ca_atoms = ['HA', '1HA', '2HA', '3HA', 'CA', 'HA1', 'HA2', 'HA3']


class Analyze:
    def __init__(self, comb_atoms, csv_directory):
        self._directory = csv_directory
        if self._directory[-1] != '/':
            self._directory += '/'
        self.ifg_atom_density = None
        self.ifg_contact_vdm = None
        self.ifg_hbond_water = None
        self.ifg_ca_hbond_vdm = None
        self.ifg_contact_water = None
        self.ifg_pdb_info = None
        self.ifg_contact_ligand = None
        self.ifg_hbond_ligand = None
        self.vdm_pdb_info = None
        self.ifg_contact_metal = None
        self.ifg_hbond_vdm = None
        self.comb_atoms = comb_atoms

        for file in [file for file in os.listdir(self._directory) if file[0] != '.']:
            if 'ifg_atom_density' in file:
                try:
                    self.ifg_atom_density = pd.read_csv(self._directory + file)
                    self._header = file.split('ifg_atom_density')[0]
                except:
                    print('ifg_atom_density not loaded')
            elif 'alt_vdm_pdb_info' in file:
                try:
                    self.alt_vdm_pdb_info = pd.read_csv(self._directory + file)
                except:
                    pass
            elif 'ifg_contact_vdm' in file:
                try:
                    self.ifg_contact_vdm = pd.read_csv(self._directory + file)
                    self._header = file.split('ifg_contact_vdm')[0]
                except:
                    print('ifg_contact_vdm not loaded')
            elif 'ifg_hbond_water' in file:
                try:
                    self.ifg_hbond_water = pd.read_csv(self._directory + file)
                    self._header = file.split('ifg_hbond_water')[0]
                except:
                    print('ifg_hbond_water not loaded')
            elif 'ifg_ca_hbond_vdm' in file:
                try:
                    self.ifg_ca_hbond_vdm = pd.read_csv(self._directory + file)
                    self._header = file.split('ifg_ca_hbond_vdm')[0]
                except:
                    print('ifg_ca_hbond_vdm not loaded')
            elif 'ifg_contact_water' in file:
                try:
                    self.ifg_contact_water = pd.read_csv(self._directory + file)
                    self._header = file.split('ifg_contact_water')[0]
                except:
                    print('ifg_contact_water not loaded')
            elif 'ifg_pdb_info' in file:
                try:
                    self.ifg_pdb_info = pd.read_csv(self._directory + file)
                    self._header = file.split('ifg_pdb_info')[0]
                except:
                    print('ifg_pdb_info not loaded')
            elif 'ifg_contact_ligand' in file:
                try:
                    self.ifg_contact_ligand = pd.read_csv(self._directory + file)
                    self._header = file.split('ifg_contact_ligand')[0]
                except:
                    print('ifg_contact_ligand not loaded')
            elif 'ifg_hbond_ligand' in file:
                try:
                    self.ifg_hbond_ligand = pd.read_csv(self._directory + file)
                    self._header = file.split('ifg_hbond_ligand')[0]
                except:
                    print('ifg_hbond_ligand not loaded')
            elif 'vdm_pdb_info' in file:
                try:
                    self.vdm_pdb_info = pd.read_csv(self._directory + file)
                    self._header = file.split('vdm_pdb_info')[0]
                except:
                    print('vdm_pdb_info not loaded')
            elif 'ifg_contact_metal' in file:
                try:
                    self.ifg_contact_metal = pd.read_csv(self._directory + file)
                    self._header = file.split('ifg_contact_metal')[0]
                except:
                    print('ifg_contact_metal not loaded')
            elif 'ifg_hbond_vdm' in file:
                try:
                    self.ifg_hbond_vdm = pd.read_csv(self._directory + file)
                    self._header = file.split('ifg_hbond_vdm')[0]
                except:
                    print('ifg_hbond_vdm not loaded')

    def get_distant_vdms(self, seq_distance=10):
        mer = pd.merge(self.ifg_pdb_info, self.vdm_pdb_info, on='iFG_count', suffixes=('_ifg', '_vdm'))
        return mer[np.abs(mer['resindex_ifg'] - mer['resindex_vdm']) > seq_distance]

    def parse_vdms(self, df, path_to_vdm=None):
        path = path_to_vdm or self._directory.split('csv')[0] + 'vdM/'
        with os.scandir(path) as it:
            for entry in it:
                if entry.name[0] != '.':
                    filename_end = '_'.join(entry.name.split('_')[4:])
                    break
        parsed_vdms = []

        if 'query_name' in df.columns:
            for n, row in df[['iFG_count', 'vdM_count', 'query_name']].iterrows():
                try:
                    parsed_vdms.append(pr.parsePDB(path + 'iFG_' + str(row['iFG_count'])
                                                   + '_vdM_' + str(row['vdM_count']) + '_'
                                                   + filename_end))
                except Exception:
                    traceback.print_exc()
            return parsed_vdms
        else:
            for n, row in df[['iFG_count', 'vdM_count']].iterrows():
                try:
                    parsed_vdms.append(pr.parsePDB(path + 'iFG_' + str(row['iFG_count'])
                                                   + '_vdM_' + str(row['vdM_count']) + '_'
                                                   + filename_end))
                except Exception:
                    traceback.print_exc()
            return parsed_vdms

    def get_vdms(self, df, path_to_vdm=None):
        path = path_to_vdm or self._directory.split('csv')[0] + 'vdM/'
        with os.scandir(path) as it:
            for entry in it:
                if entry.name[0] != '.':
                    filename_end = '_'.join(entry.name.split('_')[4:])
                    break

        if 'query_name' in df.columns:
            for n, row in df[['iFG_count', 'vdM_count', 'query_name']].iterrows():
                try:
                    yield  pr.parsePDB(path + 'iFG_' + str(row['iFG_count'])
                                + '_vdM_' + str(row['vdM_count']) + '_'
                                + filename_end)
                except Exception:
                    traceback.print_exc()
        else:
            for n, row in df[['iFG_count', 'vdM_count']].iterrows():
                try:
                    yield pr.parsePDB(path + 'iFG_' + str(row['iFG_count'])
                                                    + '_vdM_' + str(row['vdM_count']) + '_'
                                                    + filename_end)
                except Exception:
                    traceback.print_exc()

    @staticmethod
    def _get_phipsi_contacts(row, exclude_wc=True, exclude=None):
        cons = row['probe_contact_pairs'].split(') (')
        cons = [c.replace('(', '') for c in cons]
        cons = [c.replace(')', '') for c in cons]
        ccs = [c.split() for c in cons]
        if exclude_wc and exclude is None:
            protein_atoms = set([c[1] for c in ccs if c[2] != 'wc'])
        elif exclude is not None:
            protein_atoms = set([c[1] for c in ccs if c[2] not in exclude])
        else:
            protein_atoms = set([c[1] for c in ccs])
        if protein_atoms:
            if 'O' in protein_atoms:
                if 'H' in protein_atoms:
                    if not any([a in term_set for a in protein_atoms]):
                        return True
            if 'O' in protein_atoms:
                if not all([a in bb_atoms for a in protein_atoms]):
                    if not any([a in term_set for a in protein_atoms]):
                        return True
            if 'H' in protein_atoms:
                if not all([a in bb_atoms for a in protein_atoms]):
                    if not any([a in term_set for a in protein_atoms]):
                        return True
        return False

    @staticmethod
    def label_phipsi_contacts(row, exclude_wc=True, exclude=None):
        """To be used after get_phipsi_contacts is applied to dataframe.
        This will label the vdms as either backbone or sidechain."""
        cons = row['probe_contact_pairs'].split(') (')
        cons = [c.replace('(', '') for c in cons]
        cons = [c.replace(')', '') for c in cons]
        ccs = [c.split() for c in cons]
        if exclude_wc and exclude is None:
            protein_atoms = set([c[1] for c in ccs if c[2] != 'wc'])
        elif exclude is not None:
            protein_atoms = set([c[1] for c in ccs if c[2] not in exclude])
        else:
            protein_atoms = set([c[1] for c in ccs])
        if protein_atoms:
            if 'O' in protein_atoms:
                if 'H' in protein_atoms:
                    if not any([a in term_set for a in protein_atoms]):
                        return 'backbone'
            if 'O' in protein_atoms:
                if not all([a in bb_atoms for a in protein_atoms]):
                    if not any([a in term_set for a in protein_atoms]):
                        return 'sidechain'
            if 'H' in protein_atoms:
                if not all([a in bb_atoms for a in protein_atoms]):
                    if not any([a in term_set for a in protein_atoms]):
                        return 'sidechain'

    @staticmethod
    def _get_co_only_contacts(row, exclude_wc=True, exclude=None):
        cons = row['probe_contact_pairs'].split(') (')
        cons = [c.replace('(', '') for c in cons]
        cons = [c.replace(')', '') for c in cons]
        ccs = [c.split() for c in cons]
        if exclude_wc and exclude is None:
            protein_atoms = set([c[1] for c in ccs if c[2] != 'wc'])
        elif exclude is not None:
            protein_atoms = set([c[1] for c in ccs if c[2] not in exclude])
        else:
            protein_atoms = set([c[1] for c in ccs])
        if protein_atoms:
            if all([a in co_bb_atoms for a in protein_atoms]):
                return True
        return False

    @staticmethod
    def _get_hnca_only_contacts(row, exclude_wc=True, exclude=None):
        cons = row['probe_contact_pairs'].split(') (')
        cons = [c.replace('(', '') for c in cons]
        cons = [c.replace(')', '') for c in cons]
        ccs = [c.split() for c in cons]
        if exclude_wc and exclude is None:
            protein_atoms = set([c[1] for c in ccs if c[2] != 'wc'])
        elif exclude is not None:
            protein_atoms = set([c[1] for c in ccs if c[2] not in exclude])
        else:
            protein_atoms = set([c[1] for c in ccs])
        if protein_atoms:
            if all([a in hnca_bb_atoms for a in protein_atoms]):
                return True
        return False

    @staticmethod
    def _get_nterm_only_contacts(row, exclude_wc=True, exclude=None):
        cons = row['probe_contact_pairs'].split(') (')
        cons = [c.replace('(', '') for c in cons]
        cons = [c.replace(')', '') for c in cons]
        ccs = [c.split() for c in cons]
        if exclude_wc and exclude is None:
            protein_atoms = set([c[1] for c in ccs if c[2] != 'wc'])
        elif exclude is not None:
            protein_atoms = set([c[1] for c in ccs if c[2] not in exclude])
        else:
            protein_atoms = set([c[1] for c in ccs])
        if protein_atoms:
            if any([a in nterm_atoms for a in protein_atoms]):
                return True
        return False

    @staticmethod
    def _get_oxt_only_contacts(row, exclude_wc=True, exclude=None):
        cons = row['probe_contact_pairs'].split(') (')
        cons = [c.replace('(', '') for c in cons]
        cons = [c.replace(')', '') for c in cons]
        ccs = [c.split() for c in cons]
        if exclude_wc and exclude is None:
            protein_atoms = set([c[1] for c in ccs if c[2] != 'wc'])
        elif exclude is not None:
            protein_atoms = set([c[1] for c in ccs if c[2] not in exclude])
        else:
            protein_atoms = set([c[1] for c in ccs])
        if protein_atoms:
            if any([a in ['OXT'] for a in protein_atoms]):
                return True
        return False

    def _get_sc_only_contacts(self, row, exclude_wc=True, exclude=None):
        cons = row['probe_contact_pairs'].split(') (')
        cons = [c.replace('(', '') for c in cons]
        cons = [c.replace(')', '') for c in cons]
        ccs = [c.split() for c in cons]
        if exclude_wc:
            protein_atoms = set([c[1] for c in ccs if c[2] != 'wc'])
        elif exclude is not None:
            protein_atoms = set([c[1] for c in ccs if c[2] not in exclude])
        else:
            protein_atoms = set([c[1] for c in ccs])
        if not protein_atoms:
            return False
        if any([a in bb_atoms for a in protein_atoms]):
            return False
        if any([a in term_set for a in protein_atoms]):
            return False
        tf_phipsi = self._get_phipsi_contacts(row, exclude_wc=exclude_wc, exclude=exclude)
        tf_co = self._get_co_only_contacts(row, exclude_wc=exclude_wc, exclude=exclude)
        tf_hnca = self._get_hnca_only_contacts(row, exclude_wc=exclude_wc, exclude=exclude)
        tf_nterm = self._get_nterm_only_contacts(row, exclude_wc=exclude_wc, exclude=exclude)
        tf_cterm = self._get_oxt_only_contacts(row, exclude_wc=exclude_wc, exclude=exclude)
        if any([tf_phipsi, tf_co, tf_hnca, tf_nterm, tf_cterm]):
            return False
        return True

    def get_ifg_contacts(self, row, exclude_wc=True, exclude=None):
        cons = row['probe_contact_pairs'].split(') (')
        cons = [c.replace('(', '') for c in cons]
        cons = [c.replace(')', '') for c in cons]
        ccs = [c.split() for c in cons]
        if exclude_wc and exclude is None:
            ifg_atoms = set([c[0] for c in ccs if c[2] != 'wc'])
        elif exclude is not None:
            ifg_atoms = set([c[0] for c in ccs if c[2] not in exclude])
        else:
            ifg_atoms = set([c[0] for c in ccs])
        if ifg_atoms:
            if any([a in ifg_atoms for a in self.comb_atoms]):
                return True
        return False

    def get_vdms_contacting_comb_atoms(self, exclude_wc=True, exclude=None):
        return self.ifg_contact_vdm[self.ifg_contact_vdm.apply(self.get_ifg_contacts,
                                                               **dict(exclude_wc=exclude_wc, exclude=exclude), axis=1)]

    def get_distant_contacting(self, seq_distance=10, exclude_wc=True, exclude=None):
        dist_vdms = self.get_distant_vdms(seq_distance)
        cont_vdms = self.get_vdms_contacting_comb_atoms(exclude_wc=exclude_wc, exclude=exclude)
        return pd.merge(dist_vdms, cont_vdms[['iFG_count', 'vdM_count']], on=['iFG_count', 'vdM_count'])

    def get_sc_only_contacts(self, exclude_wc=True, exclude=None):
        return self.ifg_contact_vdm[self.ifg_contact_vdm.apply(self._get_sc_only_contacts,
                                                               **dict(exclude_wc=exclude_wc, exclude=exclude), axis=1)]

    def get_oxt_only_contacts(self, exclude_wc=True, exclude=None):
        return self.ifg_contact_vdm[self.ifg_contact_vdm.apply(self._get_oxt_only_contacts,
                                                               **dict(exclude_wc=exclude_wc, exclude=exclude), axis=1)]

    def get_nterm_only_contacts(self, exclude_wc=True, exclude=None):
        return self.ifg_contact_vdm[self.ifg_contact_vdm.apply(self._get_nterm_only_contacts,
                                                               **dict(exclude_wc=exclude_wc, exclude=exclude), axis=1)]

    def get_hnca_only_contacts(self, exclude_wc=True, exclude=None):
        return self.ifg_contact_vdm[self.ifg_contact_vdm.apply(self._get_hnca_only_contacts,
                                                               **dict(exclude_wc=exclude_wc, exclude=exclude), axis=1)]

    def get_co_only_contacts(self, exclude_wc=True, exclude=None):
        return self.ifg_contact_vdm[self.ifg_contact_vdm.apply(self._get_co_only_contacts,
                                                               **dict(exclude_wc=exclude_wc, exclude=exclude), axis=1)]

    def get_phipsi_contacts(self, exclude_wc=True, exclude=None):
        return self.ifg_contact_vdm[self.ifg_contact_vdm.apply(self._get_phipsi_contacts,
                                                               **dict(exclude_wc=exclude_wc, exclude=exclude), axis=1)]

    def add_label_phipsi_contacts(self, df, exclude_wc=True, exclude=None):
        labels = df.apply(self.label_phipsi_contacts, **dict(exclude_wc=exclude_wc, exclude=exclude), axis=1)
        df['sc_bb_label'] = labels

    @staticmethod
    def _get_phi(row):
        phipsi = row.split()
        try:
            phi = float(phipsi[0])
        except:
            phi = np.nan
        return phi

    @staticmethod
    def _get_psi(row):
        phipsi = row.split()
        try:
            psi = float(phipsi[1])
        except:
            psi = np.nan
        return psi

    def get_vdm_phipsi_df(self, df):
        try:
            phi = df['sec_struct_phi_psi_vdm'].apply(self._get_phi)
            psi = df['sec_struct_phi_psi_vdm'].apply(self._get_psi)
        except KeyError:
            phi = df['sec_struct_phi_psi'].apply(self._get_phi)
            psi = df['sec_struct_phi_psi'].apply(self._get_psi)
        except AttributeError:
            raise AttributeError('Need to make sure NaN is dropped '
                                 'from dataframe in phi psi column. Try using'
                                 'pandas.isnull')
        phi.name = 'phi'
        psi.name = 'psi'
        return pd.concat([phi, psi, df[['iFG_count', 'vdM_count']]], axis=1)

    def parse_vdms_by_aa(self, df, subset=None, path_to_vdm=None):
        """subset is a list of residue names"""
        gr_df = self.group_vdms_by_aa(df)
        parsed_by_aa = {}
        subset = subset or gr_df.groups
        for group in set(gr_df.groups).intersection(set(subset)):
            if group in one_letter_code.keys():
                parsed_by_aa[group] = self.parse_vdms(gr_df.get_group(group), path_to_vdm)
        return parsed_by_aa

    def make_csv(self, df, path=None):
        path = path or self._directory
        df.to_csv(path + self._header + 'distant_vdm.csv', index=False)

    def get_hbonding_vdms(self, df, hbond_seq_pattern, mode='sidechain'):
        mer = pd.merge(df, self.ifg_hbond_vdm, on=['iFG_count', 'vdM_count'], suffixes=('_vdm', '_hb'))
        mer = mer[mer['rel_resnums_hb'].str.contains(hbond_seq_pattern)]
        if mode == 'sidechain':
            def func(row):
                index = [i for i, rrn in enumerate([n for n in row['rel_resnums_hb'] if n != '-']) if rrn == '0']
                return any([any(
                    {y for names in [row['vdM_atom_names'].strip('()').split(') (')[ind]] for y in names.split() if
                     y not in ['N', 'O', 'H', 'OXT', 'H1', 'H2', 'H3']}) for ind in index])

            mer = mer[mer.apply(func, axis=1)]
        elif mode == 'mainchain':
            def func(row):
                index = [i for i, rrn in enumerate([n for n in row['rel_resnums_hb'] if n != '-']) if rrn == '0']
                return any([any(
                    {y for names in [row['vdM_atom_names'].strip('()').split(') (')[ind]] for y in names.split() if
                     y in ['N', 'O', 'H']}) for ind in index])

            mer = mer[mer.apply(func, axis=1)]
        else:
            raise NameError("arg *mode* must be 'sidechain' or 'mainchain'")

        return mer

    @staticmethod
    def group_vdms_by_aa(df):
        try:
            return df.groupby('resname')
        except:
            return df.groupby('resname_vdm')






