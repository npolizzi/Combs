from sklearn.neighbors import NearestNeighbors
from prody import calcCenter, writePDB, parsePDB
from collections import defaultdict
from .constants import resnames_aa_20
from pandas import DataFrame, merge, concat, notnull
import numpy as np
from os import scandir, makedirs, path, listdir
from .cluster import Cluster
import pickle
from .transformation import get_rot_trans
from .hbond import is_hbond, is_hbond_S_acceptor
from .convex_hull import AlphaHull
from sklearn.neighbors import BallTree


SEL_C_ALKYL = '(name CA CB) or (resname VAL and name CG1 CG2) ' \
          'or (resname ILE and name CG1 CG2 CD1) ' \
          'or (resname LEU and name CG CD1 CD2) ' \
          'or (resname MET and name CG CE) ' \
          'or (resname PRO and name CG CD) ' \
          'or (resname GLN and name CG) ' \
          'or (resname GLU and name CG) ' \
          'or (resname THR and name CG2) ' \
          'or (resname LYS and name CG CD CE) ' \
          'or (resname ARG and name CG CD)'

SEL_CO = '(name C) or (resname GLN and name CD) ' \
          'or (resname ASN and name CG) ' \
          'or (resname GLU and name CD) ' \
          'or (resname ASP and name CG) ' \
          'or (resname ARG and name CZ)'

SEL_C_ARO = '(resname TRP and name CG CD1 CD2 CE2 CE3 CZ2 CZ3 CH2) ' \
          'or (resname TYR and name CG CD1 CD2 CE1 CE2 CZ) ' \
          'or (resname PHE and name CG CD1 CD2 CE1 CE2 CZ) ' \
          'or (resname HIS HSE HID HIE and name CG CD2 CE1)'

SEL_N = 'element N'

SEL_O = 'element O'

SEL_S = 'element S'

SEL_H_ALKYL = '(name 1H 2H 3H H1 H2 H3 HA HA2 HA3 HB HB1 HB2 HB3) ' \
              'or (resname VAL and name HG11 HG12 HG13 HG21 HG22 ' \
              'HG23 1HG1 1HG2 1HG3 2HG1 2HG2 2HG3) ' \
              'or (resname ILE and name HD12 HG23 HG21 HD13 HG12 ' \
              'HG13 HG22 HD11 1HG1 1HG2 1HD1 3HG2 2HD1 2HG1 3HD1 2HG2) ' \
              'or (resname LEU and name HD12 HD22 HG HD23 HD13 HD21 ' \
              'HD11 1HD1 2HD1 HG 3HD2 3HD1 1HD2 2HD2) ' \
              'or (resname MET and name HE1 HG3 HG2 HE3 HE2 ' \
              '3HE 2HE 1HG 2HG 1HE) ' \
              'or (resname PRO and name HG3 HG2 HD3 HD2 ' \
              '1HG 2HG 2HD 1HD) ' \
              'or (resname GLN and name 1HG 2HG HG3 HG2) ' \
              'or (resname GLU and name 1HG 2HG HG3 HG2) ' \
              'or (resname THR and name HG23 HG21 HG22 3HG2 2HG2 1HG2) ' \
              'or (resname LYS and name HG3 HG2 HE3 HD3 HD2 HE2 ' \
              '2HE 1HG 2HG 2HD 1HE 1HD) ' \
              'or (resname ARG and name HG3 HG2 HD3 HD2 1HG 2HG 2HD 1HD)'

SEL_H_POL = '(name H) or (resname TRP and name HE1) ' \
            'or (resname TYR and name HH) ' \
            'or (resname GLN and name HE21 HE22 2HE2 1HE2) ' \
            'or (resname ASN and name HD21 HD22 2HD2 1HD2) ' \
            'or (resname HIS HSE HID HIE and name HD1 HE2) ' \
            'or (resname LYS and name HZ2 HZ3 HZ1 2HZ 3HZ 1HZ) ' \
            'or (resname SER and name HG) ' \
            'or (resname THR and name HG1) ' \
            'or (resname CYS and name HG) ' \
            'or (resname ARG and name 2HH1 2HH2 1HH2 HE 1HH1 ' \
            'HH11 HH21 HH12 HE HH22)'

SEL_H_ARO = '(resname TRP and name HZ2 HE3 HZ3 HH2 HD1) ' \
            'or (resname TYR and name HE1 HD1 HD2 HE2) ' \
            'or (resname HIS HSE HID HIE and name HD2 HE1) ' \
            'or (resname PHE and name HE1 HD1 HD2 HE2 HZ)'

atom_type_dict = defaultdict(dict,
            {'ARG': {'1HD': 'h_alkyl',
              '1HG': 'h_alkyl',
              '1HH1': 'h_pol',
              '1HH2': 'h_pol',
              '2HD': 'h_alkyl',
              '3HD': 'h_alkyl',
              '2HG': 'h_alkyl',
              '3HG': 'h_alkyl',
              '2HH1': 'h_pol',
              '2HH2': 'h_pol',
              'CD': 'c_alkyl',
              'CG': 'c_alkyl',
              'CZ': 'co',
              'HD2': 'h_alkyl',
              'HD3': 'h_alkyl',
              'HE': 'h_pol',
              'HG2': 'h_alkyl',
              'HG3': 'h_alkyl',
              'HH11': 'h_pol',
              'HH12': 'h_pol',
              'HH21': 'h_pol',
              'HH22': 'h_pol',
              'NE': 'n',
              'NH1': 'n',
              'NH2': 'n'},
             'ASN': {'1HD2': 'h_pol',
              '2HD2': 'h_pol',
              'CG': 'co',
              'HD21': 'h_pol',
              'HD22': 'h_pol',
              'ND2': 'n',
              'OD1': 'o'},
             'ASP': {'CG': 'co', 'OD1': 'o', 'OD2': 'o'},
             'CYS': {'HG': 'h_pol', 'SG': 's'},
             'GLN': {'1HE2': 'h_pol',
              '1HG': 'h_alkyl',
              '2HE2': 'h_pol',
              '2HG': 'h_alkyl',
              '3HG': 'h_alkyl',
              'CD': 'co',
              'CG': 'c_alkyl',
              'HE21': 'h_pol',
              'HE22': 'h_pol',
              'HG2': 'h_alkyl',
              'HG3': 'h_alkyl',
              'NE2': 'n',
              'OE1': 'o'},
             'GLU': {'1HG': 'h_alkyl',
              '2HG': 'h_alkyl',
              'CD': 'co',
              'CG': 'c_alkyl',
              'HG2': 'h_alkyl',
              'HG3': 'h_alkyl',
              'OE1': 'o',
              'OE2': 'o'},
             'HIS': {'CD2': 'c_aro',
              'CE1': 'c_aro',
              'CG': 'c_aro',
              'HD1': 'h_pol',
              'HD2': 'h_aro',
              'HE1': 'h_aro',
              'HE2': 'h_pol',
              'ND1': 'n',
              'NE2': 'n'},
             'ILE': {'1HD1': 'h_alkyl',
              '1HG1': 'h_alkyl',
              '1HG2': 'h_alkyl',
              '2HD1': 'h_alkyl',
              '2HG1': 'h_alkyl',
              '2HG2': 'h_alkyl',
              '3HD1': 'h_alkyl',
              '3HG1': 'h_alkyl',
              '3HG2': 'h_alkyl',
              'CD1': 'c_alkyl',
              'CG1': 'c_alkyl',
              'CG2': 'c_alkyl',
              'HD11': 'h_alkyl',
              'HD12': 'h_alkyl',
              'HD13': 'h_alkyl',
              'HG12': 'h_alkyl',
              'HG13': 'h_alkyl',
              'HG21': 'h_alkyl',
              'HG22': 'h_alkyl',
              'HG23': 'h_alkyl'},
             'LEU': {'1H': 'h_alkyl',
              '1HD1': 'h_alkyl',
              '1HD2': 'h_alkyl',
              '2H': 'h_alkyl',
              '2HD1': 'h_alkyl',
              '2HD2': 'h_alkyl',
              '3H': 'h_alkyl',
              '3HD1': 'h_alkyl',
              '3HD2': 'h_alkyl',
              'CD1': 'c_alkyl',
              'CD2': 'c_alkyl',
              'CG': 'c_alkyl',
              'HD11': 'h_alkyl',
              'HD12': 'h_alkyl',
              'HD13': 'h_alkyl',
              'HD21': 'h_alkyl',
              'HD22': 'h_alkyl',
              'HD23': 'h_alkyl',
              'HG': 'h_alkyl'},
             'LYS': {'1HD': 'h_alkyl',
              '1HE': 'h_alkyl',
              '1HG': 'h_alkyl',
              '1HZ': 'h_pol',
              '2HD': 'h_alkyl',
              '2HE': 'h_alkyl',
              '2HG': 'h_alkyl',
              '2HZ': 'h_pol',
              '3HZ': 'h_pol',
              'CD': 'c_alkyl',
              'CE': 'c_alkyl',
              'CG': 'c_alkyl',
              'HD2': 'h_alkyl',
              'HD3': 'h_alkyl',
              'HE2': 'h_alkyl',
              'HE3': 'h_alkyl',
              'HG2': 'h_alkyl',
              'HG3': 'h_alkyl',
              'HZ1': 'h_pol',
              'HZ2': 'h_pol',
              'HZ3': 'h_pol',
              'NZ': 'n'},
             'MET': {'1HE': 'h_alkyl',
              '1HG': 'h_alkyl',
              '2HE': 'h_alkyl',
              '2HG': 'h_alkyl',
              '3HE': 'h_alkyl',
              'CE': 'c_alkyl',
              'CG': 'c_alkyl',
              'HE1': 'h_alkyl',
              'HE2': 'h_alkyl',
              'HE3': 'h_alkyl',
              'HG2': 'h_alkyl',
              'HG3': 'h_alkyl',
              'SD': 's'},
             'MSE': {'1HE': 'h_alkyl',
              '1HG': 'h_alkyl',
              '2HE': 'h_alkyl',
              '2HG': 'h_alkyl',
              '3HE': 'h_alkyl',
              'CE': 'c_alkyl',
              'CG': 'c_alkyl',
              'HE1': 'h_alkyl',
              'HE2': 'h_alkyl',
              'HE3': 'h_alkyl',
              'HG2': 'h_alkyl',
              'HG3': 'h_alkyl',
              'SE': 's'},
             'PHE': {'CD1': 'c_aro',
              'CD2': 'c_aro',
              'CE1': 'c_aro',
              'CE2': 'c_aro',
              'CG': 'c_aro',
              'CZ': 'c_aro',
              'HD1': 'h_aro',
              'HD2': 'h_aro',
              'HE1': 'h_aro',
              'HE2': 'h_aro',
              'HZ': 'h_aro'},
             'PRO': {'1HD': 'h_alkyl',
              '1HG': 'h_alkyl',
              '2HD': 'h_alkyl',
              '2HG': 'h_alkyl',
              'CD': 'c_alkyl',
              'CG': 'c_alkyl',
              'HD2': 'h_alkyl',
              'HD3': 'h_alkyl',
              'HG2': 'h_alkyl',
              'HG3': 'h_alkyl',
              '3HG': 'h_alkyl',
              '3HD': 'h_alkyl'},
             'SER': {'HG': 'h_pol', 'OG': 'o'},
             'THR': {'1HG2': 'h_alkyl',
              '2HG2': 'h_alkyl',
              '3HG2': 'h_alkyl',
              'CG2': 'c_alkyl',
              'HG1': 'h_pol',
              '1HG': 'h_pol',
              'HG21': 'h_alkyl',
              'HG22': 'h_alkyl',
              'HG23': 'h_alkyl',
              'OG1': 'o'},
             'TRP': {'CD1': 'c_aro',
              'CD2': 'c_aro',
              'CE2': 'c_aro',
              'CE3': 'c_aro',
              'CG': 'c_aro',
              'CH2': 'c_aro',
              'CZ2': 'c_aro',
              'CZ3': 'c_aro',
              'HD1': 'h_aro',
              'HE1': 'h_pol',
              'HE3': 'h_aro',
              'HH2': 'h_aro',
              'HZ2': 'h_aro',
              'HZ3': 'h_aro',
              'NE1': 'n'},
             'TYR': {'CD1': 'c_aro',
              'CD2': 'c_aro',
              'CE1': 'c_aro',
              'CE2': 'c_aro',
              'CG': 'c_aro',
              'CZ': 'c_aro',
              'HD1': 'h_aro',
              'HD2': 'h_aro',
              'HE1': 'h_aro',
              'HE2': 'h_aro',
              'HH': 'h_pol',
              'OH': 'o'},
             'VAL': {'1HG1': 'h_alkyl',
              '1HG2': 'h_alkyl',
              '2HG1': 'h_alkyl',
              '2HG2': 'h_alkyl',
              '3HG1': 'h_alkyl',
              '3HG2': 'h_alkyl',
              'CG1': 'c_alkyl',
              'CG2': 'c_alkyl',
              'HG11': 'h_alkyl',
              'HG12': 'h_alkyl',
              'HG13': 'h_alkyl',
              'HG21': 'h_alkyl',
              'HG22': 'h_alkyl',
              'HG23': 'h_alkyl'},
             'HEM': {'FE': 'fe',
                     'NA': 'n',
                     'NB': 'n',
                     }
             })


for aa in resnames_aa_20:
    atom_type_dict[aa]['H'] = 'h_pol'
    atom_type_dict[aa]['1H'] = 'h_pol'
    atom_type_dict[aa]['2H'] = 'h_pol'
    atom_type_dict[aa]['3H'] = 'h_pol'
    atom_type_dict[aa]['H1'] = 'h_pol'
    atom_type_dict[aa]['H2'] = 'h_pol'
    atom_type_dict[aa]['H3'] = 'h_pol'
    atom_type_dict[aa]['N'] = 'n'
    atom_type_dict[aa]['O'] = 'o'
    atom_type_dict[aa]['OXT'] = 'o'
    atom_type_dict[aa]['1HA'] = 'h_alkyl'
    atom_type_dict[aa]['2HA'] = 'h_alkyl'
    atom_type_dict[aa]['HA'] = 'h_alkyl'
    atom_type_dict[aa]['HA2'] = 'h_alkyl'
    atom_type_dict[aa]['HA3'] = 'h_alkyl'
    atom_type_dict[aa]['HB'] = 'h_alkyl'
    atom_type_dict[aa]['HB1'] = 'h_alkyl'
    atom_type_dict[aa]['HB2'] = 'h_alkyl'
    atom_type_dict[aa]['HB3'] = 'h_alkyl'
    atom_type_dict[aa]['1HB'] = 'h_alkyl'
    atom_type_dict[aa]['2HB'] = 'h_alkyl'
    atom_type_dict[aa]['3HB'] = 'h_alkyl'
    atom_type_dict[aa]['CB'] = 'c_alkyl'
    atom_type_dict[aa]['CA'] = 'c_alkyl'
    atom_type_dict[aa]['C'] = 'co'

# METALS
atom_type_dict['NA']['NA'] = 'na'


def rec_dd():
    """returns a recursive dictionary"""
    return defaultdict(rec_dd)

# make dictionary of (resname, name, donor/acceptor) keys and their h-bond atom vectors.
can_hbond = rec_dd()
for aa in resnames_aa_20:
    can_hbond[aa]['H']['donor'] = [('H', 'N')]
    can_hbond[aa]['1H']['donor'] = [('1H', 'N')]
    can_hbond[aa]['2H']['donor'] = [('2H', 'N')]
    can_hbond[aa]['3H']['donor'] = [('3H', 'N')]
    can_hbond[aa]['H1']['donor'] = [('H1', 'N')]
    can_hbond[aa]['H2']['donor'] = [('H2', 'N')]
    can_hbond[aa]['H3']['donor'] = [('H3', 'N')]
    can_hbond[aa]['N']['donor'] = [('H', 'N')]
    can_hbond[aa]['O']['acceptor'] = ('O', 'C', 'C')
    can_hbond[aa]['OXT']['acceptor'] = ('OXT', 'C', 'C')
can_hbond['N_term']['N']['donor'] = [('H1 1H', 'N'), ('H2 2H', 'N'), ('H3 3H', 'N')]
can_hbond['SER']['OG']['donor'] = [('HG', 'OG')]
can_hbond['SER']['HG']['donor'] = [('HG', 'OG')]
can_hbond['SER']['OG']['acceptor'] = ('OG', 'HG', 'CB')
can_hbond['THR']['OG1']['donor'] = [('HG1', 'OG1')]
can_hbond['THR']['HG1']['donor'] = [('HG1', 'OG1')]
can_hbond['THR']['OG1']['acceptor'] = ('OG1', 'HG1', 'CB')
can_hbond['TRP']['NE1']['donor'] = [('HE1', 'NE1')]
can_hbond['TRP']['HE1']['donor'] = [('HE1', 'NE1')]
can_hbond['TYR']['OH']['donor'] = [('HH', 'OH')]
can_hbond['TYR']['HH']['donor'] = [('HH', 'OH')]
can_hbond['TYR']['OH']['acceptor'] = ('OH','HH','CZ')
can_hbond['MET']['SD']['acceptor'] = ('SD','CE','CG')
can_hbond['MSE']['SE']['acceptor'] = ('SE','CE','CG')
can_hbond['CYS']['SG']['donor'] = [('HG', 'SG')]
can_hbond['CYS']['HG']['donor'] = [('HG', 'SG')]
can_hbond['CYS']['SG']['acceptor'] = ('SG', 'HG', 'CB')
can_hbond['GLN']['NE2']['donor'] = [('1HE2 HE21', 'NE2'), ('2HE2 HE22', 'NE2')]
can_hbond['GLN']['1HE2']['donor'] = [('1HE2 HE21', 'NE2')]
can_hbond['GLN']['HE21']['donor'] = [('1HE2 HE21', 'NE2')]
can_hbond['GLN']['2HE2']['donor'] = [('2HE2 HE22', 'NE2')]
can_hbond['GLN']['HE22']['donor'] = [('2HE2 HE22', 'NE2')]
can_hbond['GLN']['OE1']['acceptor'] = ('OE1', 'CD', 'CD')
can_hbond['ASN']['ND2']['donor'] = [('1HD2 HD21', 'ND2'), ('2HD2 HD22', 'ND2')]
can_hbond['ASN']['1HD2']['donor'] = [('1HD2 HD21', 'ND2')]
can_hbond['ASN']['HD21']['donor'] = [('1HD2 HD21', 'ND2')]
can_hbond['ASN']['2HD2']['donor'] = [('2HD2 HD22', 'ND2')]
can_hbond['ASN']['HD22']['donor'] = [('2HD2 HD22', 'ND2')]
can_hbond['ASN']['OD1']['acceptor'] = ('OD1', 'CG', 'CG')
can_hbond['GLU']['OE1']['acceptor'] = ('OE1', 'CD', 'CD')
can_hbond['GLU']['OE2']['acceptor'] = ('OE2', 'CD', 'CD')
can_hbond['ASP']['OD1']['acceptor'] = ('OD1', 'CG', 'CG')
can_hbond['ASP']['OD2']['acceptor'] = ('OD2', 'CG', 'CG')
can_hbond['LYS']['NZ']['donor'] = [('1HZ HZ1', 'NZ'), ('2HZ HZ2', 'NZ'), ('3HZ HZ3', 'NZ')]
can_hbond['LYS']['1HZ']['donor'] = [('1HZ HZ1', 'NZ')]
can_hbond['LYS']['HZ1']['donor'] = [('1HZ HZ1', 'NZ')]
can_hbond['LYS']['2HZ']['donor'] = [('2HZ HZ2', 'NZ')]
can_hbond['LYS']['HZ2']['donor'] = [('2HZ HZ2', 'NZ')]
can_hbond['LYS']['3HZ']['donor'] = [('3HZ HZ3', 'NZ')]
can_hbond['LYS']['HZ3']['donor'] = [('3HZ HZ3', 'NZ')]
can_hbond['HIS']['ND1']['donor'] = [('HD1', 'ND1')]
can_hbond['HIS']['HD1']['donor'] = [('HD1', 'ND1')]
can_hbond['HIS']['ND1']['acceptor'] = ('ND1', 'CG', 'CD2')
can_hbond['HIS']['NE2']['donor'] = [('HE2', 'NE2')]
can_hbond['HIS']['HE2']['donor'] = [('HE2', 'NE2')]
can_hbond['HIS']['NE2']['acceptor'] = ('NE2', 'CD2', 'CE1')
can_hbond['ARG']['NE']['donor'] = [('HE', 'NE')]
can_hbond['ARG']['HE']['donor'] = [('HE', 'NE')]
can_hbond['ARG']['NH1']['donor'] = [('HH11 1HH1', 'NH1'), ('HH12 2HH1', 'NH1')]
can_hbond['ARG']['HH11']['donor'] = [('HH11 1HH1', 'NH1')]
can_hbond['ARG']['1HH1']['donor'] = [('HH11 1HH1', 'NH1')]
can_hbond['ARG']['HH12']['donor'] = [('HH12 2HH1', 'NH1')]
can_hbond['ARG']['2HH1']['donor'] = [('HH12 2HH1', 'NH1')]
can_hbond['ARG']['NH2']['donor'] = [('HH21 1HH2', 'NH2'), ('HH22 2HH2', 'NH2')]
can_hbond['ARG']['HH21']['donor'] = [('HH21 1HH2', 'NH2')]
can_hbond['ARG']['1HH2']['donor'] = [('HH21 1HH2', 'NH2')]
can_hbond['ARG']['HH22']['donor'] = [('HH22 2HH2', 'NH2')]
can_hbond['ARG']['2HH2']['donor'] = [('HH22 2HH2', 'NH2')]

hbond_types = {'h_pol', 'o', 'n', 's'}
hbond_donor_types = {'h_pol', 'o', 'n', 's'}
hbond_acceptor_types = {'o', 'n', 's', 'f'}
charged_atoms = {('LYS', 'NZ'),
                 ('LYS', '1HZ'),
                 ('LYS', 'HZ1'),
                 ('LYS', '2HZ'),
                 ('LYS', 'HZ2'),
                 ('LYS', '3HZ'),
                 ('LYS', 'HZ3'),
                 ('GLU', 'OE1'),
                 ('GLU', 'OE2'),
                 ('ASP', 'OD1'),
                 ('ASP', 'OD2'),
                 ('ARG', 'NE'),
                 ('ARG', 'HE'),
                 ('ARG', 'NH1'),
                 ('ARG', 'HH11'),
                 ('ARG', '1HH1'),
                 ('ARG', 'HH12'),
                 ('ARG', '2HH1'),
                 ('ARG', 'NH2'),
                 ('ARG', 'HH21'),
                 ('ARG', '1HH2'),
                 ('ARG', 'HH22'),
                 ('ARG', '2HH2')}


num_sc_atoms_residue = dict(ALA=4, GLY=0, ASP=6, ASN=8, GLN=11, GLU=9, ARG=18,
                            ILE=13, TYR=15, THR=8, PHE=14, LYS=16, LEU=13, SER=5,
                            VAL=10, TRP=18, MET=11, MSE=11, HIS=11, CYS=5, PRO=9)

num_sc_atoms_residue_deprotonated = dict(TYR=14, THR=7, SER=4, HIS=10, CYS=4)



def get_hbond_atom_coords(name, resname, selection):
    """Gets the hydrogen bond vectors from a 1-residue prody selection, returns
    donor vectors as a list of coord arrays, returns acc vectors as a coord array."""

    assert len(set(selection.getResindices())) == 1, 'input must be only 1 residue'

    ev = np.empty(3)
    ev[:] = np.nan
    h_don_coords = [ev, ev, ev, ev]
    don_coords = ev
    try:
        for i, (name1, name2) in enumerate(can_hbond[resname][name]['donor']):
            v1 = selection.select('name ' + name1).getCoords()[0]
            v2 = selection.select('name ' + name2).getCoords()[0]
            h_don_coords[i] = v1
            don_coords = v2
    except (AttributeError, ValueError, KeyError):
        pass

    if name == 'N':
        try:
            for i, (name1, name2) in enumerate(can_hbond['N_term'][name]['donor']):
                v1 = selection.select('name ' + name1).getCoords()[0]
                v2 = selection.select('name ' + name2).getCoords()[0]
                h_don_coords[i] = v1
                don_coords = v2
        except (AttributeError, ValueError, KeyError):
            pass


    # Need to account for special case of histidine
    # If a His N is a donor it cannot be an acceptor.
    if resname == 'HIS' and ~np.isnan(don_coords[0]):
        acc1_coords = ev
        acc2_coords = ev
        return don_coords, h_don_coords, acc1_coords, acc2_coords

    try:
        name1, name2, name3 = can_hbond[resname][name]['acceptor']
        acc1_coords = selection.select('name ' + name1).getCoords()[0]
        sel = selection.select('name ' + name2 + ' ' + name3)
        acc2_coords = calcCenter(sel).flatten()
    except (AttributeError, ValueError, KeyError):
        acc1_coords = ev
        acc2_coords = ev

    return don_coords, h_don_coords, acc1_coords, acc2_coords


def make_pose_df(pose):
    """Returns a dataframe with rows of every atom in
    every residue of a prody object or selection (pose)"""

    resnums = list()
    chains = list()
    segments = list()
    names = list()
    resnames = list()
    c_x = list()
    c_y = list()
    c_z = list()
    cd_x = list()
    cd_y = list()
    cd_z = list()
    ch1_x = list()
    ch1_y = list()
    ch1_z = list()
    ch2_x = list()
    ch2_y = list()
    ch2_z = list()
    ch3_x = list()
    ch3_y = list()
    ch3_z = list()
    ch4_x = list()
    ch4_y = list()
    ch4_z = list()
    ca1_x = list()
    ca1_y = list()
    ca1_z = list()
    ca2_x = list()
    ca2_y = list()
    ca2_z = list()
    atom_type_labels = list()
    seg_chain_resnums = list()
    for res_sel in pose.getHierView().iterResidues():
        resname = set(res_sel.getResnames()).pop()
        for atom in res_sel:
            resnames.append(resname)
            name = atom.getName()
            names.append(name)
            atom_type_labels.append(atom_type_dict[resname][name])
            resnum = atom.getResnum()
            resnums.append(resnum)
            chain = atom.getChid()
            chains.append(chain)
            segment = atom.getSegname()
            segments.append(segment)
            seg_chain_resnums.append((segment, chain, resnum))
            don_coords, h_don_coords, acc1_coords, acc2_coords = get_hbond_atom_coords(name, resname, res_sel)
            cd_x.append(don_coords[0].astype('float32'))
            cd_y.append(don_coords[1].astype('float32'))
            cd_z.append(don_coords[2].astype('float32'))
            ch1_x.append(h_don_coords[0][0].astype('float32'))
            ch1_y.append(h_don_coords[0][1].astype('float32'))
            ch1_z.append(h_don_coords[0][2].astype('float32'))
            ch2_x.append(h_don_coords[1][0].astype('float32'))
            ch2_y.append(h_don_coords[1][1].astype('float32'))
            ch2_z.append(h_don_coords[1][2].astype('float32'))
            ch3_x.append(h_don_coords[2][0].astype('float32'))
            ch3_y.append(h_don_coords[2][1].astype('float32'))
            ch3_z.append(h_don_coords[2][2].astype('float32'))
            ch4_x.append(h_don_coords[3][0].astype('float32'))
            ch4_y.append(h_don_coords[3][1].astype('float32'))
            ch4_z.append(h_don_coords[3][2].astype('float32'))
            ca1_x.append(acc1_coords[0].astype('float32'))
            ca1_y.append(acc1_coords[1].astype('float32'))
            ca1_z.append(acc1_coords[2].astype('float32'))
            ca2_x.append(acc2_coords[0].astype('float32'))
            ca2_y.append(acc2_coords[1].astype('float32'))
            ca2_z.append(acc2_coords[2].astype('float32'))
            c = atom.getCoords()
            c_x.append(c[0].astype('float32'))
            c_y.append(c[1].astype('float32'))
            c_z.append(c[2].astype('float32'))
    df = DataFrame(list(zip(resnums, chains, segments, resnames, names,
                            c_x, c_y, c_z, cd_x, cd_y, cd_z,
                            ch1_x, ch1_y, ch1_z, ch2_x, ch2_y, ch2_z,
                            ch3_x, ch3_y, ch3_z, ch4_x, ch4_y, ch4_z,
                            ca1_x, ca1_y, ca1_z, ca2_x, ca2_y, ca2_z,
                            atom_type_labels, seg_chain_resnums)),
                   columns=['resnum', 'chain', 'segment', 'resname', 'name',
                            'c_x', 'c_y', 'c_z', 'c_D_x', 'c_D_y',
                            'c_D_z', 'c_H1_x', 'c_H1_y', 'c_H1_z',
                            'c_H2_x', 'c_H2_y', 'c_H2_z',
                            'c_H3_x', 'c_H3_y', 'c_H3_z',
                            'c_H4_x', 'c_H4_y', 'c_H4_z',
                            'c_A1_x', 'c_A1_y', 'c_A1_z',
                            'c_A2_x', 'c_A2_y', 'c_A2_z', 'atom_type_label', 'seg_chain_resnum'])
    return df



def make_lig_df(pose, **kwargs):
    """Returns a dataframe with rows of every atom in
    every residue of a prody object or selection (pose)"""

    lig_atom_type_dict = kwargs.get('lig_atom_types_dict')
    can_hbond_dict = kwargs.get('can_hbond_dict')
    can_hbond.update(can_hbond_dict)
    resnums = list()
    chains = list()
    segments = list()
    names = list()
    resnames = list()
    c_x = list()
    c_y = list()
    c_z = list()
    cd_x = list()
    cd_y = list()
    cd_z = list()
    ch1_x = list()
    ch1_y = list()
    ch1_z = list()
    ch2_x = list()
    ch2_y = list()
    ch2_z = list()
    ch3_x = list()
    ch3_y = list()
    ch3_z = list()
    ch4_x = list()
    ch4_y = list()
    ch4_z = list()
    ca1_x = list()
    ca1_y = list()
    ca1_z = list()
    ca2_x = list()
    ca2_y = list()
    ca2_z = list()
    atom_type_labels = list()
    for res_sel in pose.getHierView().iterResidues():
        resname = set(res_sel.getResnames()).pop()
        for atom in res_sel:
            resnames.append(resname)
            name = atom.getName()
            names.append(name)
            atom_type_labels.append(lig_atom_type_dict[resname][name])
            resnums.append(atom.getResnum())
            chains.append(atom.getChid())
            segments.append(atom.getSegname())
            don_coords, h_don_coords, acc1_coords, acc2_coords = get_hbond_atom_coords(name, resname, res_sel)
            cd_x.append(don_coords[0].astype('float32'))
            cd_y.append(don_coords[1].astype('float32'))
            cd_z.append(don_coords[2].astype('float32'))
            ch1_x.append(h_don_coords[0][0].astype('float32'))
            ch1_y.append(h_don_coords[0][1].astype('float32'))
            ch1_z.append(h_don_coords[0][2].astype('float32'))
            ch2_x.append(h_don_coords[1][0].astype('float32'))
            ch2_y.append(h_don_coords[1][1].astype('float32'))
            ch2_z.append(h_don_coords[1][2].astype('float32'))
            ch3_x.append(h_don_coords[2][0].astype('float32'))
            ch3_y.append(h_don_coords[2][1].astype('float32'))
            ch3_z.append(h_don_coords[2][2].astype('float32'))
            ch4_x.append(h_don_coords[3][0].astype('float32'))
            ch4_y.append(h_don_coords[3][1].astype('float32'))
            ch4_z.append(h_don_coords[3][2].astype('float32'))
            ca1_x.append(acc1_coords[0].astype('float32'))
            ca1_y.append(acc1_coords[1].astype('float32'))
            ca1_z.append(acc1_coords[2].astype('float32'))
            ca2_x.append(acc2_coords[0].astype('float32'))
            ca2_y.append(acc2_coords[1].astype('float32'))
            ca2_z.append(acc2_coords[2].astype('float32'))
            c = atom.getCoords()
            c_x.append(c[0].astype('float32'))
            c_y.append(c[1].astype('float32'))
            c_z.append(c[2].astype('float32'))
    df = DataFrame(list(zip(resnums, chains, segments, resnames, names,
                            c_x, c_y, c_z, cd_x, cd_y, cd_z,
                            ch1_x, ch1_y, ch1_z, ch2_x, ch2_y, ch2_z,
                            ch3_x, ch3_y, ch3_z, ch4_x, ch4_y, ch4_z,
                            ca1_x, ca1_y, ca1_z, ca2_x, ca2_y, ca2_z,
                            atom_type_labels, names, resnames)),
                   columns=['resnum', 'chain', 'segment', 'resname', 'name',
                            'c_x', 'c_y', 'c_z', 'c_D_x', 'c_D_y',
                            'c_D_z', 'c_H1_x', 'c_H1_y', 'c_H1_z',
                            'c_H2_x', 'c_H2_y', 'c_H2_z',
                            'c_H3_x', 'c_H3_y', 'c_H3_z',
                            'c_H4_x', 'c_H4_y', 'c_H4_z',
                            'c_A1_x', 'c_A1_y', 'c_A1_z',
                            'c_A2_x', 'c_A2_y', 'c_A2_z', 'atom_type_label', 'lig_name', 'lig_resname'])
    return df


rel_coords_dict = dict(SC=['CA', 'N', 'C'], HNCA=['N', 'H', 'CA'],
                       CO=['C', 'O', 'CA'], PHI_PSI=['CA', 'N', 'C'])


def listdir_mac(path):
    return [f for f in listdir(path) if f[0] != '.']


class SigReps:

    def __init__(self, **kwargs):
        self.path_to_reps = kwargs.get('path_to_reps')
        self.min_cluster_method = kwargs.get('min_cluster_method', 'mean')
        self.path_to_clusters = kwargs.get('path_to_clusters', '.')
        self.path_to_rel_vdms = kwargs.get('path_to_rel_vdms', '.')
        self.outdir = kwargs.get('outdir', '.')
        if self.outdir[-1] != '/':
            self.outdir += '/'

    @staticmethod
    def _combine_rel_vdms(path_to_relvdm1, path_to_relvdm2, outdir):
        """These dataframes should have a *query_name* column before
        being combined."""

        if path_to_relvdm1[-1] != '/':
            path_to_relvdm1 += '/'

        if path_to_relvdm2[-1] != '/':
            path_to_relvdm2 += '/'

        if outdir[-1] != '/':
            outdir += '/'

        try:
            makedirs(outdir)
        except FileExistsError:
            pass

        for f in [f for f in listdir(path_to_relvdm1) if f[0] != '.']:
            with open(path_to_relvdm1 + f, 'rb') as infile:
                df_relvdm1 = pickle.load(infile)

            with open(path_to_relvdm2 + f, 'rb') as infile:
                df_relvdm2 = pickle.load(infile)

            df = concat((df_relvdm1, df_relvdm2))
            df.to_pickle(outdir + f)

    @staticmethod
    def _make_df_rep(path_to_rep):
        with open(path_to_rep, 'rb') as infile:
            df_rep = pickle.load(infile)

        df_rep = df_rep.rename(columns={'resname': 'resname_vdm'})
        df_rep['iFG_count'] = df_rep['iFG_count'].astype('str')
        df_rep['vdM_count'] = df_rep['vdM_count'].astype('str')
        return df_rep

    @staticmethod
    def _make_sig_reps(df_relvdms, df_rep_min_clu, outpath):
        df_sig_reps = merge(df_relvdms, df_rep_min_clu, on=['iFG_count', 'vdM_count',
                                                           'resname_vdm', 'query_name'])
        df_sig_reps.to_pickle(outpath)

    @staticmethod
    def _make_sig_reps_lig(df_rep, df_clu, outpath):
        df_sig_reps = merge(df_rep, df_clu, on=['iFG_count', 'vdM_count', 'query_name'])
        df_sig_reps.to_pickle(outpath)

    @staticmethod
    def _get_min_cluster_size_std(df, num_std):
        mean = df[df.centroid == True].cluster_size.mean()
        std = df[df.centroid == True].cluster_size.std()
        return mean + num_std * std

    def _get_min_cluster_size(self, df_clu):
        if self.min_cluster_method == 'mean':
            return df_clu[df_clu.centroid == True].cluster_size.mean()

        elif self.min_cluster_method[-3:] == 'std':
            # example is '1std'
            num_std = int(self.min_cluster_method[0])
            return self._get_min_cluster_size_std(df_clu, num_std)

        elif self.min_cluster_method == 'none':
            return 0

    def make_sig_reps(self):

        for label in listdir_mac(self.path_to_reps):
            with open(self.path_to_rel_vdms + label + '.pkl', 'rb') as infile:
                df_relvdms = pickle.load(infile)
                # needed for appropriate merging because of Cluster data type:
                df_relvdms['iFG_count'] = df_relvdms['iFG_count'].astype('str')
                df_relvdms['vdM_count'] = df_relvdms['vdM_count'].astype('str')

            for file in listdir_mac(self.path_to_reps + label):
                path_to_clu = self.path_to_clusters + label + '/' + file
                with open(path_to_clu, 'rb') as infile:
                    df_clu = pickle.load(infile)
                min_cluster_size = self._get_min_cluster_size(df_clu)
                print(label, file.split('.')[0], 'min cluster size=', min_cluster_size)

                path_to_rep = self.path_to_reps + label + '/' + file
                df_rep = self._make_df_rep(path_to_rep)
                df_rep_min_clu = df_rep[df_rep['cluster_size'] > min_cluster_size]

                outpath = self.outdir + label + '/'
                try:
                    makedirs(outpath)
                except FileExistsError:
                    pass

                self._make_sig_reps(df_relvdms, df_rep_min_clu, outpath + file)

    def make_sig_reps_ligand(self):

        for label in listdir_mac(self.path_to_reps):

            for file in listdir_mac(self.path_to_reps + label):
                path_to_clu = self.path_to_clusters + label + '/' + file
                with open(path_to_clu, 'rb') as infile:
                    df_clu = pickle.load(infile)
                df_clu['iFG_count'] = df_clu['iFG_count'].astype('str')
                df_clu['vdM_count'] = df_clu['vdM_count'].astype('str')
                min_cluster_size = self._get_min_cluster_size(df_clu)
                df_clu_min = df_clu[df_clu['cluster_size'] > min_cluster_size]
                df_clu_min = df_clu_min.rename(columns={'resname': 'resname_vdm'})
                print(label, file.split('.')[0], 'min cluster size=', min_cluster_size)

                path_to_rep = self.path_to_reps + label + '/' + file
                with open(path_to_rep, 'rb') as infile:
                    df_rep = pickle.load(infile)
                df_rep['iFG_count'] = df_rep['iFG_count'].astype('str')
                df_rep['vdM_count'] = df_rep['vdM_count'].astype('str')

                outpath = self.outdir + label + '/'
                try:
                    makedirs(outpath)
                except FileExistsError:
                    pass

                self._make_sig_reps_lig(df_rep, df_clu_min, outpath + file)


class RelCoords:
    def __init__(self, vdms, label):
        self.vdms = vdms
        self.label = label
        self.ori, self.pla1, self.pla2 = rel_coords_dict[self.label]

    def set_new_coord_frame(self, vdm):
        origin_coords = vdm.select('chain X resnum 10 name ' + self.ori).getCoords()[0]
        vdm_coords = vdm.getCoords()
        vdm_coords_neworigin = vdm_coords - origin_coords
        plane_atom1_coords = vdm.select('chain X resnum 10 name ' + self.pla1).getCoords()[0] - origin_coords
        plane_atom2_coords = vdm.select('chain X resnum 10 name ' + self.pla2).getCoords()[0] - origin_coords
        x_norm = plane_atom1_coords / np.linalg.norm(plane_atom1_coords)
        orthvec = np.cross(plane_atom1_coords, plane_atom2_coords)
        z_norm = -1 * orthvec / np.linalg.norm(orthvec)
        orthvec2 = np.cross(plane_atom1_coords, orthvec)
        y_norm = orthvec2 / np.linalg.norm(orthvec2)
        R = np.array([x_norm, y_norm, z_norm])
        vdm_coords_neworigin_rot = np.dot(vdm_coords_neworigin, R.T)
        vdm.setCoords(vdm_coords_neworigin_rot)

    def _print_rel_vdms(self, prefix='', postfix='', outdir='./', gzip=True):
        if outdir[-1] != '/':
            outdir += '/'

        if gzip:
            file_end = '.pdb.gz'
        else:
            file_end = '.pdb'

        for vdm in self.vdms:
            self.set_new_coord_frame(vdm)
            pdbname = str(vdm).split()[-1]
            writePDB(outdir + prefix + pdbname + postfix + file_end, vdm)

    def print_rel_vdms(self, df, path_to_vdm, prefix='', postfix='', outdir='./', gzip=True):
        try:
            makedirs(outdir)
        except FileExistsError:
            pass

        self.vdms = self.get_vdms(df, path_to_vdm)
        self._print_rel_vdms(prefix, postfix, outdir, gzip)

    @staticmethod
    def get_vdms(df, path_to_vdm):
        with scandir(path_to_vdm) as it:
            for entry in it:
                if entry.name[0] != '.':
                    filename_end = '_'.join(entry.name.split('_')[4:])
                    break

            for n, row in df[['iFG_count', 'vdM_count']].iterrows():
                yield parsePDB(path_to_vdm + 'iFG_' + str(row['iFG_count'])
                               + '_vdM_' + str(row['vdM_count']) + '_'
                               + filename_end)


class VdmReps:
    def __init__(self, df, **kwargs):
        self.grouping = kwargs.get('grouping', ['iFG_count', 'vdM_count', 'query_name'])
        df['iFG_count'] = df['iFG_count'].astype('str')
        df['vdM_count'] = df['vdM_count'].astype('str')
        df.set_index(self.grouping, inplace=True)
        self.df = df
        self.df_xy = None
        self.df_y = None
        self.df_x = None
        self.aa = None
        self.num_atoms = None
        self.df_aa = None
        self.df_aa_gr = None
        self.df_aa_gr_coords = None
        self.coords = None
        self.df_pairwise = None
        self._reps = list()
        self.reps = None
        self.reps_dict = dict()
        self.mems = None
        self.cents = None
        self.path_to_df_pairwise = kwargs.get('path_to_df_pairwise', './')
        if self.path_to_df_pairwise[-1] != '/':
            self.path_to_df_pairwise += '/'

    def partition_df(self, aa_):
        df = self.df[self.df['resname_vdm'] == aa_]
        gr = df.groupby(self.grouping)
        self.df_xy = gr.filter(lambda g: len(set(g.chain)) == 2)
        self.df_y = gr.filter(lambda g: len(set(g.chain)) == 1)

    def partition_df_x_only(self, aa_):
        df = self.df[self.df['resname_vdm'] == aa_]
        self.df_x = df[df.chain == 'X']

    def set_dataframe_aa(self, df, aa):
        self.aa = aa
        self.df_aa = df[df['resname_vdm'] == aa]

    def get_df_pairwise(self):
        with open(self.path_to_df_pairwise + self.aa + '.pkl', 'rb') as infile:
            self.df_pairwise = pickle.load(infile)
            self.df_pairwise['iFG_count'] = self.df_pairwise['iFG_count'].astype('str')
            self.df_pairwise['vdM_count'] = self.df_pairwise['vdM_count'].astype('str')
            self.df_pairwise.set_index(self.grouping, inplace=True)

    @staticmethod
    def _reshape(g):
        g = g.sort_values(by=['chain', 'name'])
        return g[['c_x', 'c_y', 'c_z']].values.reshape(-1)

    def group_dataframe_aa(self):
        self.df_aa_gr = self.df_aa.groupby(self.grouping)
        un, coun = np.unique(self.df_aa_gr.size(), return_counts=True)
        if len(un) > 1:
            ind_max = coun.argmax()
            size_max = un[ind_max]
            df = self.df_aa_gr.filter(lambda x: len(x) == size_max)
            self.df_aa_gr = df.groupby(self.grouping)
        assert len(set(self.df_aa_gr.size())) == 1, 'Atom groups are not all uniform size!'
        self.num_atoms = set(self.df_aa_gr.size()).pop()

    def get_dataframe_coords(self):
        self.df_aa_gr_coords = self.df_aa_gr.apply(self._reshape)
        self.coords = np.stack(self.df_aa_gr_coords)

    def cluster_coords(self, rmsd=0.2):
        nbrs = NearestNeighbors(radius=np.sqrt(self.num_atoms) * rmsd)
        nbrs.fit(self.coords)
        adj_mat = nbrs.radius_neighbors_graph(self.coords)
        self.mems, self.cents = Cluster().greedy(adj_mat)
        print('number of clusters=', len(self.mems))

    def _find_rep(self, mems, cent):
        rep_mask = self.df_aa.index.isin(self.df_aa_gr_coords.iloc[[cent]].index)
        rep = self.df_aa[rep_mask].copy()
        mem_indices = self.df_aa_gr_coords.iloc[mems].index
        clu_mask = self.df_pairwise.index.isin(mem_indices)
        df_clu = self.df_pairwise[clu_mask]
        try:
            df_clu_top = df_clu.sort_values(by='cluster_number').iloc[0]
            rep.loc[:, 'cluster_number'] = df_clu_top['cluster_number']
            rep.loc[:, 'cluster_size'] = df_clu_top['cluster_size']
            rep.loc[:, 'cluster_type'] = df_clu_top['cluster_type']
            rep.loc[:, 'rmsd_from_centroid'] = df_clu_top['rmsd_from_centroid']
            self._reps.append(rep)
        except IndexError:
            print('IndexError')

    def _find_rep_dict(self, mems, cent):
        if len(mems) > 1:
            rep_mask = self.df_aa.index.isin(self.df_aa_gr_coords.iloc[[cent]].index)
            rep = self.df_aa[rep_mask].copy()
            mems_mask = self.df_aa.index.isin(self.df_aa_gr_coords.iloc[mems].index)
            mems_ = self.df_aa[mems_mask].copy()
            try:
                df_clu_top = mems_.sort_values(by='cluster_number').iloc[0]
                rep.loc[:, 'cluster_number'] = df_clu_top['cluster_number']
                rep.loc[:, 'cluster_size'] = df_clu_top['cluster_size']
                rep.loc[:, 'cluster_type'] = df_clu_top['cluster_type']
                rep.loc[:, 'rmsd_from_centroid'] = df_clu_top['rmsd_from_centroid']
                self._reps.append(rep)
            except IndexError:
                print('IndexError')
        else:
            rep_mask = self.df_aa.index.isin(self.df_aa_gr_coords.iloc[[cent]].index)
            rep = self.df_aa[rep_mask].copy()
            self._reps.append(rep)

    def find_reps(self, rmsd=0.2):
        if self.df_pairwise is None:
            self.get_df_pairwise()

        if self.df_aa_gr is None:
            self.group_dataframe_aa()

        if self.coords is None:
            self.get_dataframe_coords()

        if self.mems is None:
            self.cluster_coords(rmsd)

        for mems, cent in zip(self.mems, self.cents):
            print('cluster size=', len(mems))
            self._find_rep(mems, cent)

    def find_reps_dict(self, rmsd=0.2):

        if self.df_aa_gr is None:
            self.group_dataframe_aa()

        if self.coords is None:
            self.get_dataframe_coords()

        if self.mems is None:
            self.cluster_coords(rmsd)

        for mems, cent in zip(self.mems, self.cents):
            print('cluster size=', len(mems))
            self._find_rep_dict(mems, cent)

    def set_reps_df(self):
        self.reps = concat(self._reps)
        self.reps.reset_index(inplace=True)

    def _find_all_reps(self, aa_, rmsd=0.2, outdir='./'):
        if outdir[-1] != '/':
            outdir += '/'

        try:
            makedirs(outdir)
        except FileExistsError:
            pass

        self.partition_df(aa_)
        if len(self.df_xy) > 0:
            self._find_all_reps_cycle(self.df_xy, aa_, rmsd)
        if len(self.df_y) > 0:
            self._find_all_reps_cycle(self.df_y, aa_, rmsd)
        self.set_reps_df()
        self.reps.to_pickle(outdir + aa_ + '.pkl')
        self._reps = list()

    def find_all_reps(self, rmsd=0.2, outdir='./', omit=set()):
        aas = set(self.df.resname_vdm) - {'MSE'} - omit
        for aa_ in aas:
            print('finding reps of', aa_)
            self._find_all_reps(aa_, rmsd, outdir)

    def _find_all_reps_dict(self, aa_, rmsd=0.2):
        self.partition_df_x_only(aa_)
        if len(self.df_x) > 0:
            self._find_all_reps_cycle_dict(self.df_x, aa_, rmsd)
        self.set_reps_df()
        self.reps_dict[aa_] = self.reps
        self._reps = list()

    def find_all_reps_dict(self, rmsd=0.2):
        aas = set(self.df.resname_vdm) - {'MSE'}
        for aa_ in aas:
            print('finding reps of', aa_)
            self._find_all_reps_dict(aa_, rmsd)

    def _find_all_reps_cycle(self, df, aa_, rmsd=0.2):
        self.set_dataframe_aa(df, aa_)
        self.get_df_pairwise()
        self.group_dataframe_aa()
        self.get_dataframe_coords()
        self.cluster_coords(rmsd)
        self.find_reps(rmsd)

    def _find_all_reps_cycle_dict(self, df, aa_, rmsd=0.2):
        self.set_dataframe_aa(df, aa_)
        self.group_dataframe_aa()
        self.get_dataframe_coords()
        self.cluster_coords(rmsd)
        self.find_reps_dict(rmsd)



backbone_str = 'name CA 1H 2H 3H H1 H2 H3 HA HA1 HA2 HA3 1HA 2HA 3HA C O N H OXT'



class DataFrameVdm(RelCoords):
    def __init__(self, vdms, label, ifg_atom_names, **kwargs):
        """Will make a dataframe of vdms that are selections of
        protein (chain X) and iFG atoms (chain Y).  Ligands can be present
        as chain Z."""

        super().__init__(vdms, label)
        self.chains = list()
        self.resnums = list()
        self.segments = list()
        self.resnames = list()
        self.resnames_vdm = list()
        self.names = list()
        self.c_x = list()
        self.c_y = list()
        self.c_z = list()
        self.cd_x = list()
        self.cd_y = list()
        self.cd_z = list()
        self.ch1_x = list()
        self.ch1_y = list()
        self.ch1_z = list()
        self.ch2_x = list()
        self.ch2_y = list()
        self.ch2_z = list()
        self.ch3_x = list()
        self.ch3_y = list()
        self.ch3_z = list()
        self.ch4_x = list()
        self.ch4_y = list()
        self.ch4_z = list()
        self.ca1_x = list()
        self.ca1_y = list()
        self.ca1_z = list()
        self.ca2_x = list()
        self.ca2_y = list()
        self.ca2_z = list()
        self.res_types = list()
        self.ifg_counts = list()
        self.vdm_counts = list()
        self.res_types = list()
        self.atom_type_labels = list()
        self.labels = list()
        self.df = None
        self.set = False
        self.ifg_atom_names = ifg_atom_names
        self.num_atoms_ifg = len(self.ifg_atom_names)
        self.sels = None
        self.sc_bb_list = kwargs.get('sc_bb_list', None)
        self.phipsi_dataframe = kwargs.get('phipsi_dataframe', None)
        if self.phipsi_dataframe is not None:
            self.phipsi_dataframe['iFG_count'] = self.phipsi_dataframe['iFG_count'].astype('str')
            self.phipsi_dataframe['vdM_count'] = self.phipsi_dataframe['vdM_count'].astype('str')
        self.query_name = kwargs.get('query_name', None)
        self.sel_set = False

    def append_attr(self, vdm, resname_vdm):
        spl_repr = repr(vdm).split()[-3].split('_')
        self.ifg_counts.extend([spl_repr[1]] * len(vdm))
        self.vdm_counts.extend([spl_repr[3]] * len(vdm))
        self.labels.extend([self.label] * len(vdm))
        for res_sel in vdm.getHierView().iterResidues():
            resname = set(res_sel.getResnames()).pop()
            chain = set(res_sel.getChids()).pop()
            _res_sel = res_sel
            if chain == 'Y':
                _res_sel = [atom for atom_name in self.ifg_atom_names
                            for atom in res_sel.select('name ' + atom_name)]
            if chain == 'X':
                _res_sel = [atom for atom_name in atom_type_dict[resname].keys()
                            if res_sel.select('name ' + atom_name) is not None
                            for atom in res_sel.select('name ' + atom_name)]
            for atom in _res_sel:
                name = atom.getName()
                chain = atom.getChid()
                self.resnames.append(resname)
                self.resnames_vdm.append(resname_vdm)
                self.names.append(name)
                self.atom_type_labels.append(atom_type_dict[resname][name])
                self.resnums.append(atom.getResnum())
                self.chains.append(chain)
                self.segments.append(atom.getSegname())
                don_coords, h_don_coords, acc1_coords, acc2_coords = get_hbond_atom_coords(name, resname, res_sel)
                self.cd_x.append(don_coords[0].astype('float32'))
                self.cd_y.append(don_coords[1].astype('float32'))
                self.cd_z.append(don_coords[2].astype('float32'))
                self.ch1_x.append(h_don_coords[0][0].astype('float32'))
                self.ch1_y.append(h_don_coords[0][1].astype('float32'))
                self.ch1_z.append(h_don_coords[0][2].astype('float32'))
                self.ch2_x.append(h_don_coords[1][0].astype('float32'))
                self.ch2_y.append(h_don_coords[1][1].astype('float32'))
                self.ch2_z.append(h_don_coords[1][2].astype('float32'))
                self.ch3_x.append(h_don_coords[2][0].astype('float32'))
                self.ch3_y.append(h_don_coords[2][1].astype('float32'))
                self.ch3_z.append(h_don_coords[2][2].astype('float32'))
                self.ch4_x.append(h_don_coords[3][0].astype('float32'))
                self.ch4_y.append(h_don_coords[3][1].astype('float32'))
                self.ch4_z.append(h_don_coords[3][2].astype('float32'))
                self.ca1_x.append(acc1_coords[0].astype('float32'))
                self.ca1_y.append(acc1_coords[1].astype('float32'))
                self.ca1_z.append(acc1_coords[2].astype('float32'))
                self.ca2_x.append(acc2_coords[0].astype('float32'))
                self.ca2_y.append(acc2_coords[1].astype('float32'))
                self.ca2_z.append(acc2_coords[2].astype('float32'))
                c = atom.getCoords()
                self.c_x.append(c[0].astype('float32'))
                self.c_y.append(c[1].astype('float32'))
                self.c_z.append(c[2].astype('float32'))
                if chain == 'Y':
                    self.res_types.append('iFG')
                if chain == 'X':
                    self.res_types.append('protein')
                if chain == 'Z':
                    self.res_types.append('ligand')

    def set_df_info(self):
        if not self.sel_set:
            if self.label == 'SC':
                self.select_SC_iFG()
            elif self.label in ['HNCA', 'CO']:
                self.select_iFG()
            elif self.label == 'PHI_PSI':
                self.select_phipsi_vdms()
            else:
                raise 'Label must be PHI_PSI, SC, HNCA, or CO.'

        for sel, resname_vdm in self.sels:
            self.append_attr(sel, resname_vdm)
        self.set = True

    def make(self):
        """make dataframe from vdms"""

        if not self.set:
            self.set_df_info()

        all_attrs = [self.ifg_counts, self.vdm_counts, self.resnums,
                     self.chains, self.segments, self.resnames, self.resnames_vdm,
                     self.names, self.c_x, self.c_y, self.c_z, self.cd_x, self.cd_y, self.cd_z,
                     self.ch1_x, self.ch1_y, self.ch1_z, self.ch2_x, self.ch2_y, self.ch2_z,
                     self.ch3_x, self.ch3_y, self.ch3_z, self.ch4_x, self.ch4_y, self.ch4_z,
                     self.ca1_x, self.ca1_y, self.ca1_z, self.ca2_x, self.ca2_y, self.ca2_z, self.atom_type_labels,
                     self.res_types, self.labels]

        assert len(set(map(len, all_attrs))) == 1, 'all attributes must be same length'

        self.df = DataFrame(list(zip(*all_attrs)),
                            columns=['iFG_count', 'vdM_count', 'resnum', 'chain',
                                     'segment', 'resname', 'resname_vdm', 'name',
                                     'c_x', 'c_y', 'c_z', 'c_D_x', 'c_D_y',
                                    'c_D_z', 'c_H1_x', 'c_H1_y', 'c_H1_z',
                                    'c_H2_x', 'c_H2_y', 'c_H2_z',
                                    'c_H3_x', 'c_H3_y', 'c_H3_z',
                                    'c_H4_x', 'c_H4_y', 'c_H4_z',
                                    'c_A1_x', 'c_A1_y', 'c_A1_z',
                                    'c_A2_x', 'c_A2_y', 'c_A2_z', 'atom_type_label',
                                     'res_type', 'label'])

        if self.query_name:
            self.df['query_name'] = self.query_name

        if self.label == 'PHI_PSI':
            df = merge(self.df, self.phipsi_dataframe, on=['iFG_count', 'vdM_count'])
            self.df = df

    def _select_SC_iFG(self):
        """sc interactions only. Selects iFG + SC atoms"""

        for vdm in self.vdms:
            resname = vdm.select('chain X resnum 10 name CA').getResnames()[0]
            sc = vdm.select('chain X resnum 10 not ' + backbone_str)
            if sc is not None:
                sc_check = (len(sc) == num_sc_atoms_residue[resname])
                sc_deprot_check = False
                if resname in num_sc_atoms_residue_deprotonated.keys():
                    sc_deprot_check = (len(sc) == num_sc_atoms_residue_deprotonated[resname])
            else:
                sc_check = True
                sc_deprot_check = True

            ifg = vdm.select('chain Y resnum 10 name ' + ' '.join(self.ifg_atom_names))
            check_sel = vdm.select('chain X resnum 10 name '
                                   + ' '.join([self.ori, self.pla1, self.pla2]))
            if len(ifg) == self.num_atoms_ifg and len(check_sel) == 3 and (sc_check or sc_deprot_check):
                if sc is not None:
                    sel = sc | ifg
                else:
                    sel = ifg
                self.set_new_coord_frame(vdm)
                yield sel, resname

    def select_SC_iFG(self):
        """sc interactions only. Selects iFG + SC atoms"""

        self.sels = self._select_SC_iFG()
        self.sel_set = True

    def _select_iFG(self):
        """Relevant to bb CA N H interactions only, or bb C=O interactions only.
         This only selects the iFG atoms."""

        for vdm in self.vdms:
            resname = vdm.select('chain X resnum 10 name CA').getResnames()[0]
            ifg = vdm.select('chain Y resnum 10 name ' + ' '.join(self.ifg_atom_names))
            check_sel = vdm.select('chain X resnum 10 name '
                                   + ' '.join([self.ori, self.pla1, self.pla2]))
            if len(ifg) == self.num_atoms_ifg and len(check_sel) == 3:
                self.set_new_coord_frame(vdm)
                yield ifg, resname

    def select_iFG(self):
        """Relevant to bb CA N H interactions only, or bb C=O interactions only.
         This only selects the iFG atoms."""

        self.sels = self._select_iFG()
        self.sel_set = True

    def _select_phipsi_vdms(self):
        """bb CA N H + C=O interactions, as well as CA N H + SC, or C=O + SC.
        If SC is interacting, this will select iFG + SC atoms.  Otherwise,
        it only selects the iFG atoms"""

        for vdm, sc_bb in zip(self.vdms, self.sc_bb_list):
            resname = vdm.select('chain X resnum 10 name CA').getResnames()[0]
            sel = vdm.select('chain Y resnum 10 name ' + ' '.join(self.ifg_atom_names))
            check_sel = vdm.select('chain X resnum 10 name '
                                   + ' '.join([self.ori, self.pla1, self.pla2]))
            if len(sel) == self.num_atoms_ifg and len(check_sel) == 3:
                append = True
                if sc_bb == 'sidechain':
                    append = False
                    sc = vdm.select('chain X resnum 10 not ' + backbone_str)
                    resname = vdm.select('chain X resnum 10 name CA').getResnames()[0]
                    if len(sc) == num_sc_atoms_residue[resname]:
                        sel = sc | sel
                        append = True
                if append:
                    self.set_new_coord_frame(vdm)
                    yield sel, resname

    def select_phipsi_vdms(self):
        """bb CA N H + C=O interactions, as well as CA N H + SC, or C=O + SC.
        If SC is interacting, this will select iFG + SC atoms.  Otherwise,
        it only selects the iFG atoms"""

        self.sels = self._select_phipsi_vdms()
        self.sel_set = True


def make_rel_vdms(an, ifg_atom_names, outdir, label, seq_dist=7, **kwargs):
    """Makes relative van der Mers of type label and for ifg atoms included
    in ifg_atom_names."""

    if outdir[-1] != '/':
        outdir += '/'

    df = kwargs.get('df', None)
    custom_label = kwargs.get('custom_label', 'SC')
    exclude = kwargs.get('exclude')
    path_to_vdm = kwargs.get('path_to_vdm', None)
    if df is None:
        dis_con = an.get_distant_contacting(seq_dist)
    else:
        dis_con = df

    if label == 'PHI_PSI':
        label_df = an.get_phipsi_contacts(exclude=exclude)
    elif label == 'SC':
        label_df = an.get_sc_only_contacts(exclude=exclude)
    elif label == 'HNCA':
        label_df = an.get_hnca_only_contacts(exclude=exclude)
    elif label == 'CO':
        label_df = an.get_co_only_contacts(exclude=exclude)
    elif label == 'CUSTOM':
        label_df = dis_con[['iFG_count', 'vdM_count']]
        label = custom_label
    else:
        raise 'Label must be PHI_PSI, SC, HNCA, or CO.'

    dis_con_label = merge(dis_con, label_df, on=['iFG_count', 'vdM_count'])

    if label == 'PHI_PSI':
        an.add_label_phipsi_contacts(dis_con_label)
        try:
            dis_con_label = dis_con_label[notnull(dis_con_label['sec_struct_phi_psi_vdm'])]
        except:
            dis_con_label = dis_con_label[notnull(dis_con_label['sec_struct_phi_psi'])]
        phipsi_df = an.get_vdm_phipsi_df(dis_con_label)
        sc_bb_list = dis_con_label['sc_bb_label']
        kwargs['sc_bb_list'] = sc_bb_list
        kwargs['phipsi_dataframe'] = phipsi_df

    if path_to_vdm:
        vdms = an.get_vdms(dis_con_label, path_to_vdm=path_to_vdm)
    else:
        vdms = an.get_vdms(dis_con_label)
    df_relvdm = DataFrameVdm(vdms, label, ifg_atom_names, **kwargs)
    df_relvdm.make()
    pickle_dump(df_relvdm.df, outdir + label + '.pkl')


class MacOSFile(object):

    def __init__(self, f):
        self.f = f

    def __getattr__(self, item):
        return getattr(self.f, item)

    def read(self, n):
        if n >= (1 << 31):
            buffer = bytearray(n)
            idx = 0
            while idx < n:
                batch_size = min(n - idx, 1 << 31 - 1)
                buffer[idx:idx + batch_size] = self.f.read(batch_size)
                idx += batch_size
            return buffer
        return self.f.read(n)

    def write(self, buffer):
        n = len(buffer)
        print("writing total_bytes=%s..." % n, flush=True)
        idx = 0
        while idx < n:
            batch_size = min(n - idx, 1 << 31 - 1)
            print("writing bytes [%s, %s)... " % (idx, idx + batch_size), end="", flush=True)
            self.f.write(buffer[idx:idx + batch_size])
            print("done.", flush=True)
            idx += batch_size


def pickle_dump(obj, file_path):
    with open(file_path, "wb") as f:
        return pickle.dump(obj, MacOSFile(f), protocol=pickle.HIGHEST_PROTOCOL)


def pickle_load(file_path):
    with open(file_path, "rb") as f:
        return pickle.load(MacOSFile(f))


def make_rel_vdms_all_types(an, ifg_atom_names, outdir, seq_dist=7, **kwargs):
    for label in ['SC', 'CO', 'HNCA', 'PHI_PSI']:
        make_rel_vdms(an, ifg_atom_names, outdir, label, seq_dist, **kwargs)


my_path = path.abspath(path.dirname(__file__))


def load_ideal(df, label):
    path_ = path.join(my_path, 'ideal_ala_' + label + '.pkl')
    with open(path_, 'rb') as infile:
        df[label] = pickle.load(infile)


df_ideal_ala = dict()
load_ideal(df_ideal_ala, 'SC')
load_ideal(df_ideal_ala, 'HNCA')
load_ideal(df_ideal_ala, 'CO')
load_ideal(df_ideal_ala, 'PHI_PSI')


df_ideal_ala_atoms = dict(SC=['HA', 'C', 'CA', 'N'],
                          HNCA=['HA', 'H', 'CA', 'N'],
                          CO=['HA', 'C', 'O'],
                          PHI_PSI=['HA', 'C', 'CA', 'N'])


def make_df_corr(dict_corr):
    """Example of dict_corr for apixaban and carboxamide:
    dict_corr = dict(APX=dict(GLN=dict(NE2='N3', CD='C11', OE1='O1', CG='C10'),
                              ASN=dict(ND2='N3',CG='C11',OD1='O1',CB='C10')))"""
    names = list()
    lig_names = list()
    resnames = list()
    lig_resnames = list()
    for lig_resname, lig_dict in dict_corr.items():
        for resname, names_dict in lig_dict.items():
            names.extend(names_dict.keys())
            lig_names.extend(names_dict.values())
            len_ = len(names_dict.keys())
            resnames.extend([resname] * len_)
            lig_resnames.extend([lig_resname] * len_)
    return DataFrame(list(zip(names, resnames, lig_names, lig_resnames)),
                     columns=['name', 'resname', 'lig_name', 'lig_resname'])


coords = ['c_x', 'c_y', 'c_z', 'c_D_x', 'c_D_y',
            'c_D_z', 'c_H1_x', 'c_H1_y', 'c_H1_z',
            'c_H2_x', 'c_H2_y', 'c_H2_z',
            'c_H3_x', 'c_H3_y', 'c_H3_z',
            'c_H4_x', 'c_H4_y', 'c_H4_z',
            'c_A1_x', 'c_A1_y', 'c_A1_z',
            'c_A2_x', 'c_A2_y', 'c_A2_z']


class SuperposeLig:
    """This really only needs to be the coords of the ligand,
    superposed onto a vdm, but it need not include the vdm itself, right?"""

    def __init__(self, df_reps, df_lig, df_corr, label, **kwargs):
        self.df_reps = df_reps
        self.df_reps_gr = self.df_reps.groupby(['iFG_count', 'vdM_count', 'query_name'])
        self.df_lig = df_lig
        self.df_corr = df_corr
        self.df_nonclashing_lig = DataFrame()
        self._nonclashing_lig = list()
        self.lig_coords = None
        self.label = label
        self.ligand_iFG_corr_sorted = None
        self.truncate_lig_atoms = kwargs.get('truncate_lig_atoms', [])  # list of (lig_resname, lig_name) tuples

    def set_sorted_lig_corr(self):
        self.ligand_iFG_corr_sorted = self.df_corr.sort_values(by=['lig_resname', 'lig_name'])

    def get_lig_coords(self):
        df_corr = self.ligand_iFG_corr_sorted[['lig_resname', 'lig_name']].drop_duplicates()
        self.lig_coords = merge(df_corr, self.df_lig,
                                on=['lig_resname', 'lig_name'], sort=False)[['c_x', 'c_y', 'c_z']].values
        
    def _get_ifg_coords(self, rep):
        df_ifg = rep[rep.chain == 'Y']
        df_ifg_c = merge(self.ligand_iFG_corr_sorted[['resname', 'name']], df_ifg,
                          how='inner', on=['resname', 'name'], sort=False)
        return df_ifg_c[['c_x', 'c_y', 'c_z']].values

    @staticmethod
    def _get_rep_info(rep):
        ifg = set(rep.iFG_count).pop()
        vdm = set(rep.vdM_count).pop()
        query_name = set(rep.query_name).pop()
        return ifg, vdm, query_name

    @staticmethod
    def _set_info(df, ifg, vdm, query_name):
        df['iFG_count'] = ifg
        df['vdM_count'] = vdm
        df['query_name'] = query_name
    
    def _find(self, rep):
        ifg_coords = self._get_ifg_coords(rep)
        R, m_com, t_com = get_rot_trans(self.lig_coords, ifg_coords)
        df_lig = self.df_lig.copy()
        df_lig[coords[:3]] = np.dot(df_lig[coords[:3]] - m_com, R) + t_com
        df_lig[coords[3:6]] = np.dot(df_lig[coords[3:6]] - m_com, R) + t_com
        df_lig[coords[6:9]] = np.dot(df_lig[coords[6:9]] - m_com, R) + t_com
        df_lig[coords[9:12]] = np.dot(df_lig[coords[9:12]] - m_com, R) + t_com
        df_lig[coords[12:15]] = np.dot(df_lig[coords[12:15]] - m_com, R) + t_com
        df_lig[coords[15:18]] = np.dot(df_lig[coords[15:18]] - m_com, R) + t_com
        df_lig[coords[18:21]] = np.dot(df_lig[coords[18:21]] - m_com, R) + t_com
        df_lig[coords[21:]] = np.dot(df_lig[coords[21:]] - m_com, R) + t_com
        df_rep_vdm = rep[rep.chain == 'X']
        ifg, vdm, query_name = self._get_rep_info(rep)
        df_ideal = df_ideal_ala[self.label].copy()
        self._set_info(df_ideal, ifg, vdm, query_name)
        self._set_info(df_lig, ifg, vdm, query_name)
        df_ideal_subset = df_ideal[df_ideal.name.isin(df_ideal_ala_atoms[self.label])]
        df_rep_vdm_append = concat((df_rep_vdm, df_ideal_subset), sort=False)

        df_corr = self.ligand_iFG_corr_sorted[['lig_resname', 'lig_name']].drop_duplicates()
        df_lig_trunc = merge(df_lig, df_corr, on=['lig_resname', 'lig_name'], sort=False, how='outer', indicator=True)
        df_lig_trunc = df_lig_trunc[df_lig_trunc._merge == 'left_only'].drop(columns='_merge')

        if self.truncate_lig_atoms:
            df_trunc = DataFrame(self.truncate_lig_atoms, columns=['lig_resname', 'lig_name'])
            df_lig_trunc = merge(df_lig_trunc, df_trunc, on=['lig_resname', 'lig_name'], sort=False, how='outer',
                                 indicator=True)
            df_lig_trunc = df_lig_trunc[df_lig_trunc._merge == 'left_only'].drop(columns='_merge')

        clash = Clash(df_lig_trunc, df_rep_vdm_append)
        clash.set_grouping(['resnum'])
        clash.find()
        if len(clash.dfq_clash_free) > 0:
            self._nonclashing_lig.append(df_lig)

    def find(self):
        if self.ligand_iFG_corr_sorted is None:
            self.set_sorted_lig_corr()

        if self.lig_coords is None:
            self.get_lig_coords()

        for name, rep in self.df_reps_gr:
            self._find(rep)
        if len(self._nonclashing_lig) > 0:
            self.df_nonclashing_lig = concat(self._nonclashing_lig, ignore_index=True)

    def print_lig(self, lig, ifg_count, vdm_count, query_name, out_path, out_name):
        df_lig = self.df_nonclashing_lig[(self.df_nonclashing_lig['iFG_count'] == ifg_count)
                                         & (self.df_nonclashing_lig['vdM_count'] == vdm_count)
                                         & (self.df_nonclashing_lig['query_name'] == query_name)]
        coords = np.stack(df_lig[df_lig['name'] == n][['c_x', 'c_y', 'c_z']].item() for n in lig.getNames())
        lig_copy = lig.copy()
        lig_copy.setCoords(coords)
        writePDB(out_path + out_name, lig_copy)


class Neighbors:
    def __init__(self, df, exclude=None, **kwargs):
        """makes atom-type dataframes and nearest neighbors
        of coordinates found in dataframe df"""
        
        self.ex = exclude
        if self.ex:
            self.ex_resnum = self.ex[0]
            self.ex_chain = self.ex[1]
            self.ex_seg = self.ex[2]

            gen_exclude = ((df['resnum'] == self.ex_resnum) &
                           (df['chain'] == self.ex_chain) &
                           (df['segment'] == self.ex_seg) &
                           ~(df['name'].isin(['H', 'O'])))

            # cbeta_exclude is for a 4-bond distance condition to start
            # measuring clashes between atoms.  For Cbeta atoms, this means
            # excluding the i-1 residue's C atom and the i+1 residue's N atom.

            res = ((df['resnum'] == self.ex_resnum) &
                   (df['chain'] == self.ex_chain) &
                   (df['segment'] == self.ex_seg))
            resm1 = ((df['resnum'] == self.ex_resnum - 1) &
                     (df['chain'] == self.ex_chain) &
                     (df['segment'] == self.ex_seg) &
                     (df['name'] == 'C'))
            resp1 = ((df['resnum'] == self.ex_resnum + 1) &
                     (df['chain'] == self.ex_chain) &
                     (df['segment'] == self.ex_seg) &
                     (df['name'] == 'N'))

            if resm1.any() and resp1.any():
                cbeta_exclude = (resm1 & res & resp1)
            elif resm1.any():
                cbeta_exclude = (resm1 & res)
            else:
                cbeta_exclude = (resp1 & res)

            self.df = df[~gen_exclude]
            self.df_excluded = df[gen_exclude]
            self.df_cb = df[~cbeta_exclude]
            self.df_cb_excluded = df[cbeta_exclude]

        else:
            self.df = df
            self.df_excluded = None
            self.df_cb = None
            self.df_cb_excluded = None

        # Using default e-cloud vdW params from D and J Richardson's Probe program.
        self.vdw_radii = dict(co=kwargs.get('r_co', 1.65),
                              c_alkyl=kwargs.get('r_c_alkyl', 1.70),
                              c_aro=kwargs.get('r_c_aro', 1.75),
                              n=kwargs.get('r_n', 1.55),
                              o=kwargs.get('r_o', 1.40),
                              s=kwargs.get('r_s', 1.80),
                              h_pol=kwargs.get('r_h_pol', 1.05),
                              h_aro=kwargs.get('r_h_aro', 1.05),
                              h_alkyl=kwargs.get('r_h_alkyl', 1.22))

        self.atom_types = set(self.df.atom_type_label)
        self.neighbors_atom_types = dict()
        self.df_atom_types = dict()

    def make_df(self, atom_type_label):
        """returns subset of dataframe df that corresponds to
        atom_type_label"""

        return self.df[self.df['atom_type_label'] == atom_type_label]

    def get_coords(self, atom_type_label):
        """retrieves coordinates as a numpy array of atom_type_label
        subset of dataframe df"""

        return self.df_atom_types[atom_type_label][['c_x', 'c_y', 'c_z']].values

    def make_neighbors(self, atom_type_label, radius=5):
        """Makes nearest neighbors object of atom coords
        from atom_type_label subset of dataframe df"""

        try:
            return NearestNeighbors(radius=radius,
                                    n_neighbors=1,
                                    algorithm='ball_tree').fit(self.get_coords(atom_type_label))
        except (ValueError, AttributeError):
            return None

    def set_neighbors(self, radius=5):
        """Sets the attributes df_atom_types(dict) and neighbors_atom_types(dict).
        This is the main method that needs to be called"""

        for atom_type in self.atom_types:
            self.df_atom_types[atom_type] = self.make_df(atom_type)
            self.neighbors_atom_types[atom_type] = self.make_neighbors(atom_type, radius)


atom_types_sortkey = ['c_alkyl', 'c_aro', 'h_alkyl', 'h_aro', 's', 'n', 'o', 'co', 'h_pol', 'f']


#CLASH FILTER INCORPORATING S ACCEPTOR HBOND DISTANCE EXCEPTION
class Clash:
    """Will find members of dataframe dfq that do not clash with dft"""

    def __init__(self, dfq, dft, **kwargs):
        self.q_grouping = kwargs.get('q_grouping')
        dfq.loc[:, 'num_tag'] = np.arange(len(dfq))
        self.dfq = dfq
        self.dft = dft
        self.atom_types_dfq = None
        self.atom_types_dft = None
        self.dfq_atom_type = dict()
        self.dft_atom_type = dict()
        self._balltrees = dict()
        self.clash_indices = list()
        self.dfq_clash_free = None
        self.overlap_hb = kwargs.get('overlap_hb', 0.7)
        self.overlap_hb_heavy_nn = kwargs.get('overlap_hb_heavy_nn', 0.6)
        self.overlap_hb_heavy_no = kwargs.get('overlap_hb_heavy_no', 0.45)
        self.overlap_hb_heavy_oo = kwargs.get('overlap_hb_heavy_oo', 0.3)
        self.tol = kwargs.get('tol', 0.1)
        self.tol_h_alkyl = kwargs.get('tol_h_alkyl', 0.1)
        # Using default e-cloud vdW params from D and J Richardson's Probe program.
        self.vdw_radii = dict(co=kwargs.get('r_co', 1.65),
                              c_alkyl=kwargs.get('r_c_alkyl', 1.70),
                              c_aro=kwargs.get('r_c_aro', 1.75),
                              c_aro_met=kwargs.get('r_c_aro_met', 1.2),
                              n=kwargs.get('r_n', 1.55),
                              n_met=kwargs.get('r_n_met', 1.0),
                              f=kwargs.get('r_f', 1.30),
                              o=kwargs.get('r_o', 1.40),
                              s=kwargs.get('r_s', 1.80),
                              p=kwargs.get('r_p', 1.80),
                              h_pol=kwargs.get('r_h_pol', 1.05),
                              h_aro=kwargs.get('r_h_aro', 1.05),
                              h_alkyl=kwargs.get('r_h_alkyl', 1.22),
                              cl=kwargs.get('r_cl', 1.77),
                              na=kwargs.get('r_na', 0.95),
                              fe=kwargs.get('r_fe', 0.6),
                              zn=kwargs.get('r_zn', 0.71))

    def set_atom_types(self):
        self.atom_types_dfq = set(self.dfq.atom_type_label)
        self.atom_types_dft = set(self.dft.atom_type_label)

    def set_grouping(self, grouping):
        self.q_grouping = grouping

    def set_index(self):
        self.dfq.set_index(self.q_grouping, inplace=True, drop=False)

    def split_dfs_to_atom_types(self):
        for atom_type in self.atom_types_dfq:
            self.dfq_atom_type[atom_type] = self.dfq[self.dfq.atom_type_label.isin([atom_type])]
        for atom_type in self.atom_types_dft:
            self.dft_atom_type[atom_type] = self.dft[self.dft.atom_type_label.isin([atom_type])]

    @staticmethod
    def make_tree(dfq_):
        return BallTree(dfq_[['c_x', 'c_y', 'c_z']].values)

    def make_trees(self):
        for atom_type, dfq_ in self.dfq_atom_type.items():
            self._balltrees[atom_type] = self.make_tree(dfq_)

    @staticmethod
    def prune_empty(dists, inds):
        t_inds = []
        dists_ = []
        q_inds = []
        for t_ind, (dist, q_ind) in enumerate(zip(dists, inds)):
            if dist.size > 0:
                t_inds.append([t_ind])
                dists_.append(dist)
                q_inds.append(q_ind)
        return t_inds, q_inds, dists_

    @staticmethod
    def get_clashes_hb_hard_cutoff(dists, q_inds, t_inds, hb_hard_cutoff):
        q_inds_clashes = []
        q_inds_poss_hbonds = []
        t_inds_poss_hbonds = []
        for d, i_q, i_t in zip(dists, q_inds, t_inds):
            clashing = d < hb_hard_cutoff
            clashes = i_q[clashing]
            poss_hbonds = i_q[~clashing]
            if clashes.size > 0:
                q_inds_clashes.extend(clashes)
            if poss_hbonds.size > 0:
                q_inds_poss_hbonds.append(poss_hbonds)
                t_inds_poss_hbonds.append(i_t)
        return q_inds_clashes, q_inds_poss_hbonds, t_inds_poss_hbonds


    def _angle_test(self, dfq, dft, q_inds_poss_hbonds):
        t_is_don = ~np.isnan(dft.c_D_x.values)
        t_is_acc = ~np.isnan(dft.c_A1_x.values)
        if ~t_is_don and ~t_is_acc:  # dft is only 1 row, so .values produces a scalar
            return list(q_inds_poss_hbonds)

        q_is_don = ~np.isnan(dfq.c_D_x.values)
        q_is_acc = ~np.isnan(dfq.c_A1_x.values)
        if (~q_is_don).all() and (~q_is_acc).all():
            return list(q_inds_poss_hbonds)

        clashing = set(q_inds_poss_hbonds)
        hbonds = set()

        if t_is_acc and (q_is_don).any():
            #q is donor, t is acceptor
            donor_inds = q_inds_poss_hbonds[q_is_don]
            donor = dfq[q_is_don]
            d_arr = donor[coords[3:18]].values
            s = d_arr.shape
            if len(s) == 1:
                m = 1
            else:
                m = s[0]
            a_arr = np.tile(dft[coords[18:]].values, (m, 1))
            X = np.hstack((d_arr, a_arr))
            is_hb = is_hbond(X)
            is_hb = is_hb.astype(bool)
            hbonds |= set(donor_inds[is_hb])

        if t_is_don and (q_is_acc).any():
            #q is acceptor, t is donor
            acc_inds = q_inds_poss_hbonds[q_is_acc]
            acc = dfq[q_is_acc]
            a_arr = acc[coords[18:]].values
            s = a_arr.shape
            if len(s) == 1:
                m = 1
            else:
                m = s[0]
            d_arr = np.tile(dft[coords[3:18]].values, (m, 1))
            X = np.hstack((d_arr, a_arr))
            is_hb = is_hbond(X)
            is_hb = is_hb.astype(bool)
            hbonds |= set(acc_inds[is_hb])

        clashing -= hbonds
        return list(clashing)
    
    def _angle_test_S_acceptor(self, dfq, dft, q_inds_poss_hbonds):
        t_is_don = ~np.isnan(dft.c_D_x.values)
        t_is_acc = ~np.isnan(dft.c_A1_x.values)
        if ~t_is_don and ~t_is_acc:  # dft is only 1 row, so .values produces a scalar
            return list(q_inds_poss_hbonds)

        q_is_don = ~np.isnan(dfq.c_D_x.values)
        q_is_acc = ~np.isnan(dfq.c_A1_x.values)
        if (~q_is_don).all() and (~q_is_acc).all():
            return list(q_inds_poss_hbonds)

        clashing = set(q_inds_poss_hbonds)
        hbonds = set()

        if t_is_acc and (q_is_don).any():
            #q is donor, t is acceptor
            donor_inds = q_inds_poss_hbonds[q_is_don]
            donor = dfq[q_is_don]
            d_arr = donor[coords[3:18]].values
            s = d_arr.shape
            if len(s) == 1:
                m = 1
            else:
                m = s[0]
            a_arr = np.tile(dft[coords[18:]].values, (m, 1))
            X = np.hstack((d_arr, a_arr))
            is_hb = is_hbond_S_acceptor(X)
            is_hb = is_hb.astype(bool)
            hbonds |= set(donor_inds[is_hb])

        if t_is_don and (q_is_acc).any():
            #q is acceptor, t is donor
            acc_inds = q_inds_poss_hbonds[q_is_acc]
            acc = dfq[q_is_acc]
            a_arr = acc[coords[18:]].values
            s = a_arr.shape
            if len(s) == 1:
                m = 1
            else:
                m = s[0]
            d_arr = np.tile(dft[coords[3:18]].values, (m, 1))
            X = np.hstack((d_arr, a_arr))
            is_hb = is_hbond_S_acceptor(X)
            is_hb = is_hb.astype(bool)
            hbonds |= set(acc_inds[is_hb])

        clashing -= hbonds
        return list(clashing)

    def angle_test(self, q_inds_poss_hbonds, t_inds_poss_hbonds, dfq_, dft_):
        clashes = []
        for q_inds_poss_hbond, t_inds_poss_hbond in zip(q_inds_poss_hbonds, t_inds_poss_hbonds):
            df_poss_hb_t = dft_.iloc[t_inds_poss_hbond]
            df_poss_hb_q = dfq_.iloc[q_inds_poss_hbond]
            clashes.extend(self._angle_test(df_poss_hb_q, df_poss_hb_t,
                                            q_inds_poss_hbond))
        return clashes

    def angle_test_S_acceptor(self, q_inds_poss_hbonds, t_inds_poss_hbonds, dfq_, dft_):
        clashes = []
        for q_inds_poss_hbond, t_inds_poss_hbond in zip(q_inds_poss_hbonds, t_inds_poss_hbonds):
            df_poss_hb_t = dft_.iloc[t_inds_poss_hbond]
            df_poss_hb_q = dfq_.iloc[q_inds_poss_hbond]
            clashes.extend(self._angle_test_S_acceptor(df_poss_hb_q, df_poss_hb_t,
                                            q_inds_poss_hbond))
        return clashes

    def _find_clash_indices_hb(self):
        pass

    def _find_clash_indices(self, atom_type_q, atom_type_t):
        tree = self._balltrees[atom_type_q]
        dft_ = self.dft_atom_type[atom_type_t]
        if (atom_type_q == 'h_alkyl') or (atom_type_t == 'h_alkyl'):
            cutoff = self.vdw_radii[atom_type_q] + self.vdw_radii[atom_type_t] - self.tol - self.tol_h_alkyl
        else:
            cutoff = self.vdw_radii[atom_type_q] + self.vdw_radii[atom_type_t] - self.tol
        i, d = tree.query_radius(dft_[['c_x', 'c_y', 'c_z']].values,
                                 r=cutoff, return_distance=True)
        t_inds, q_inds, dists = self.prune_empty(d, i)

        if t_inds:
            D_q = atom_type_q in hbond_donor_types
            A_t = atom_type_t in hbond_acceptor_types
            A_q = atom_type_q in hbond_acceptor_types
            D_t = atom_type_t in hbond_donor_types
            if not ((D_q and A_t) or (D_t and A_q)):
                return [j for i in q_inds for j in i]

            if (atom_type_q in {'n', 'p', 's'}) and (atom_type_t in {'n', 'p', 's'}):
                hb_hard_cutoff = cutoff - self.overlap_hb_heavy_nn
            elif (atom_type_q == 'o') and (atom_type_t in {'n', 'p', 's'}):
                hb_hard_cutoff = cutoff - self.overlap_hb_heavy_no
            elif (atom_type_t == 'o') and (atom_type_q in {'n', 'p', 's'}):
                hb_hard_cutoff = cutoff - self.overlap_hb_heavy_no
            elif (atom_type_q == 'o') and (atom_type_t == 'o'):
                hb_hard_cutoff = cutoff - self.overlap_hb_heavy_oo
            else:
                hb_hard_cutoff = cutoff - self.overlap_hb

            q_inds_hard_clashes, q_inds_poss_hbonds, t_inds_poss_hbonds = \
                self.get_clashes_hb_hard_cutoff(dists, q_inds, t_inds, hb_hard_cutoff)

            if q_inds_poss_hbonds:
                dfq_ = self.dfq_atom_type[atom_type_q]
                if atom_type_q in {'s'} or atom_type_t in {'s'}:
                    q_inds_soft_clashes = self.angle_test_S_acceptor(q_inds_poss_hbonds, t_inds_poss_hbonds, dfq_, dft_)
                else:
                    q_inds_soft_clashes = self.angle_test(q_inds_poss_hbonds, t_inds_poss_hbonds, dfq_, dft_)
                q_inds_hard_clashes.extend(q_inds_soft_clashes)
            return q_inds_hard_clashes
        else:
            return list()

    def find_clash_indices(self):
        for atom_type_q in self.atom_types_dfq:
            for atom_type_t in self.atom_types_dft:
                local_clash_inds = self._find_clash_indices(atom_type_q, atom_type_t)
                global_clash_inds = self.dfq_atom_type[atom_type_q]['num_tag'].iloc[local_clash_inds].values
                self.clash_indices.extend(global_clash_inds)

    def drop(self, return_clash_free=True, return_clash=False):
        cind = self.dfq.iloc[np.unique(self.clash_indices)].index
        qind = self.dfq.index
        mask = qind.isin(cind)
        if return_clash_free:
            self.dfq_clash_free = self.dfq.loc[~mask].reset_index(drop=True)
        if return_clash:
            self.dfq_clash = self.dfq.loc[mask].reset_index(drop=True)

    def find(self, return_clash_free=True, return_clash=False):
        self.set_index()

        if self.atom_types_dfq is None:
            self.set_atom_types()

        if not self.dfq_atom_type:
            self.split_dfs_to_atom_types()

        if not self._balltrees:
            self.make_trees()

        self.find_clash_indices()
        self.drop(return_clash_free, return_clash)


class Contact:
    """Will find contacts between dataframe dfq and dft"""

    def __init__(self, dfq, dft, **kwargs):
        dfq.loc[:, 'num_tag'] = np.arange(len(dfq))
        dft.loc[:, 'num_tag'] = np.arange(len(dft))
        self.dfq = dfq
        self.dft = dft
        self.atom_types_dfq = None
        self.atom_types_dft = None
        self.dfq_atom_type = dict()
        self.dft_atom_type = dict()
        self._balltrees = dict()
        self.q_global_indices = list()
        self.t_global_indices = list()
        self.contact_types = list()
        self.df_contacts = None
        self.dfq_clash_free = None
        self.gap_close_contact = kwargs.get('gap_close_contact', 0.3)
        self.gap_wide_contact = kwargs.get('gap_wide_contact', 0.5)
        self.overlap_hb = kwargs.get('overlap_hb', 0.7)
        self.overlap_hb_heavy_nn = kwargs.get('overlap_hb_heavy_nn', 0.6)
        self.overlap_hb_heavy_no = kwargs.get('overlap_hb_heavy_no', 0.45)
        self.overlap_hb_heavy_oo = kwargs.get('overlap_hb_heavy_oo', 0.3)
        self.tol = kwargs.get('tol', 0.1)
        # Using default e-cloud vdW params from D and J Richardson's Probe program.
        self.vdw_radii = dict(co=kwargs.get('r_co', 1.65),
                              c_alkyl=kwargs.get('r_c_alkyl', 1.70),
                              c_aro=kwargs.get('r_c_aro', 1.75),
                              c_aro_met=kwargs.get('r_c_aro_met', 1.2),
                              n=kwargs.get('r_n', 1.55),
                              n_met=kwargs.get('r_n_met', 1.0),
                              f=kwargs.get('r_f', 1.30),
                              o=kwargs.get('r_o', 1.40),
                              s=kwargs.get('r_s', 1.80),
                              p=kwargs.get('r_p', 1.80),
                              h_pol=kwargs.get('r_h_pol', 1.05),
                              h_aro=kwargs.get('r_h_aro', 1.05),
                              h_alkyl=kwargs.get('r_h_alkyl', 1.22),
                              cl=kwargs.get('r_cl', 1.77),
                              na=kwargs.get('r_na', 0.95),
                              # fe=kwargs.get('r_fe', 0.74),
                              fe=kwargs.get('r_fe', 0.6),
                              zn=kwargs.get('r_zn', 0.71))

    def set_atom_types(self):
        self.atom_types_dfq = set(self.dfq.atom_type_label)
        self.atom_types_dft = set(self.dft.atom_type_label)

    def split_dfs_to_atom_types(self):
        for atom_type in self.atom_types_dfq:
            self.dfq_atom_type[atom_type] = self.dfq[self.dfq.atom_type_label.isin([atom_type])]
        for atom_type in self.atom_types_dft:
            self.dft_atom_type[atom_type] = self.dft[self.dft.atom_type_label.isin([atom_type])]

    @staticmethod
    def make_tree(dfq_):
        return BallTree(dfq_[['c_x', 'c_y', 'c_z']].values)

    def make_trees(self):
        for atom_type, dfq_ in self.dfq_atom_type.items():
            self._balltrees[atom_type] = self.make_tree(dfq_)

    @staticmethod
    def prune_empty(dists, inds):
        t_inds = []
        dists_ = []
        q_inds = []
        for t_ind, (dist, q_ind) in enumerate(zip(dists, inds)):
            if dist.size > 0:
                t_inds.append([t_ind])
                dists_.append(dist)
                q_inds.append(q_ind)
        return t_inds, q_inds, dists_

    @staticmethod
    def partition_contacts_hb_hard_cutoff(dists, q_inds, t_inds,
                                          cc_low, cc_low_hb, cc_high, wc_high,
                                          hb_hard_cutoff):
        q_inds_clashes = []
        t_inds_clashes = []
        q_inds_cc = []
        t_inds_cc = []
        q_inds_wc = []
        t_inds_wc = []
        q_inds_poss_hbonds_cl = []
        t_inds_poss_hbonds_cl = []
        q_inds_poss_hbonds_cc = []
        t_inds_poss_hbonds_cc = []
        for d, i_q, i_t in zip(dists, q_inds, t_inds):
            clashing = d < hb_hard_cutoff
            poss_hbonds_cl_test = (d >= hb_hard_cutoff) & (d < cc_low)
            poss_hbonds_cc_test = (d >= cc_low) & (d < cc_low_hb)
            clashes = i_q[clashing]
            cc_test = (d >= cc_low_hb) & (d < cc_high)
            wc_test = (d >= cc_high) & (d < wc_high)
            ccs = i_q[cc_test]
            wcs = i_q[wc_test]
            poss_hbonds_cl = i_q[poss_hbonds_cl_test]
            poss_hbonds_cc = i_q[poss_hbonds_cc_test]
            if clashes.size > 0:
                q_inds_clashes.append(clashes)
                t_inds_clashes.append(i_t)
            if ccs.size > 0:
                q_inds_cc.append(ccs)
                t_inds_cc.append(i_t)
            if wcs.size > 0:
                q_inds_wc.append(wcs)
                t_inds_wc.append(i_t)
            if poss_hbonds_cl.size > 0:
                q_inds_poss_hbonds_cl.append(poss_hbonds_cl)
                t_inds_poss_hbonds_cl.append(i_t)
            if poss_hbonds_cc.size > 0:
                q_inds_poss_hbonds_cc.append(poss_hbonds_cc)
                t_inds_poss_hbonds_cc.append(i_t)
        return q_inds_clashes, t_inds_clashes, q_inds_cc, t_inds_cc, \
               q_inds_wc, t_inds_wc, q_inds_poss_hbonds_cl, t_inds_poss_hbonds_cl, \
               q_inds_poss_hbonds_cc, t_inds_poss_hbonds_cc

    @staticmethod
    def partition_contacts_no_hb(dists, q_inds, t_inds, cc_low, cc_high, wc_high):
        q_inds_clashes = []
        t_inds_clashes = []
        q_inds_cc = []
        t_inds_cc = []
        q_inds_wc = []
        t_inds_wc = []
        for d, i_q, i_t in zip(dists, q_inds, t_inds):
            clashing = d < cc_low
            clashes = i_q[clashing]
            cc_test = (d >= cc_low) & (d < cc_high)
            wc_test = (d >= cc_high) & (d < wc_high)
            ccs = i_q[cc_test]
            wcs = i_q[wc_test]
            if clashes.size > 0:
                q_inds_clashes.append(clashes)
                t_inds_clashes.append(i_t)
            if ccs.size > 0:
                q_inds_cc.append(ccs)
                t_inds_cc.append(i_t)
            if wcs.size > 0:
                q_inds_wc.append(wcs)
                t_inds_wc.append(i_t)
        return q_inds_clashes, t_inds_clashes, q_inds_cc, t_inds_cc, \
               q_inds_wc, t_inds_wc, [], []

    def _angle_test(self, dfq, dft, q_inds_poss_hbonds):
        t_is_don = ~np.isnan(dft.c_D_x.values)
        t_is_acc = ~np.isnan(dft.c_A1_x.values)
        if ~t_is_don and ~t_is_acc:  # dft is only 1 row, so .values produces a scalar
            return list(q_inds_poss_hbonds), []

        q_is_don = ~np.isnan(dfq.c_D_x.values)
        q_is_acc = ~np.isnan(dfq.c_A1_x.values)
        if (~q_is_don).all() and (~q_is_acc).all():
            return list(q_inds_poss_hbonds), []

        clashing = set(q_inds_poss_hbonds)
        hbonds = set()

        if t_is_acc and (q_is_don).any():
            # q is donor, t is acceptor
            donor_inds = q_inds_poss_hbonds[q_is_don]
            donor = dfq[q_is_don]
            d_arr = donor[coords[3:18]].values
            s = d_arr.shape
            if len(s) == 1:
                m = 1
            else:
                m = s[0]
            a_arr = np.tile(dft[coords[18:]].values, (m, 1))
            X = np.hstack((d_arr, a_arr))
            is_hb = is_hbond(X)
            is_hb = is_hb.astype(bool)
            hbonds |= set(donor_inds[is_hb])

        if t_is_don and (q_is_acc).any():
            # q is acceptor, t is donor
            acc_inds = q_inds_poss_hbonds[q_is_acc]
            acc = dfq[q_is_acc]
            a_arr = acc[coords[18:]].values
            s = a_arr.shape
            if len(s) == 1:
                m = 1
            else:
                m = s[0]
            d_arr = np.tile(dft[coords[3:18]].values, (m, 1))
            X = np.hstack((d_arr, a_arr))
            is_hb = is_hbond(X)
            is_hb = is_hb.astype(bool)
            hbonds |= set(acc_inds[is_hb])

        clashing -= hbonds
        return list(clashing), list(hbonds)

    def _angle_test_S_acceptor(self, dfq, dft, q_inds_poss_hbonds):
        t_is_don = ~np.isnan(dft.c_D_x.values)
        t_is_acc = ~np.isnan(dft.c_A1_x.values)
        if ~t_is_don and ~t_is_acc:  # dft is only 1 row, so .values produces a scalar
            return list(q_inds_poss_hbonds), []

        q_is_don = ~np.isnan(dfq.c_D_x.values)
        q_is_acc = ~np.isnan(dfq.c_A1_x.values)
        if (~q_is_don).all() and (~q_is_acc).all():
            return list(q_inds_poss_hbonds), []

        clashing = set(q_inds_poss_hbonds)
        hbonds = set()

        if t_is_acc and (q_is_don).any():
            # q is donor, t is acceptor
            donor_inds = q_inds_poss_hbonds[q_is_don]
            donor = dfq[q_is_don]
            d_arr = donor[coords[3:18]].values
            s = d_arr.shape
            if len(s) == 1:
                m = 1
            else:
                m = s[0]
            a_arr = np.tile(dft[coords[18:]].values, (m, 1))
            X = np.hstack((d_arr, a_arr))
            is_hb = is_hbond_S_acceptor(X)
            is_hb = is_hb.astype(bool)
            hbonds |= set(donor_inds[is_hb])

        if t_is_don and (q_is_acc).any():
            # q is acceptor, t is donor
            acc_inds = q_inds_poss_hbonds[q_is_acc]
            acc = dfq[q_is_acc]
            a_arr = acc[coords[18:]].values
            s = a_arr.shape
            if len(s) == 1:
                m = 1
            else:
                m = s[0]
            d_arr = np.tile(dft[coords[3:18]].values, (m, 1))
            X = np.hstack((d_arr, a_arr))
            is_hb = is_hbond_S_acceptor(X)
            is_hb = is_hb.astype(bool)
            hbonds |= set(acc_inds[is_hb])

        clashing -= hbonds
        return list(clashing), list(hbonds)

    def angle_test(self, q_inds_poss_hbonds, t_inds_poss_hbonds, dfq_, dft_):
        q_inds_clash = []
        t_inds_clash = []
        q_inds_hbond = []
        t_inds_hbond = []
        for q_inds_poss_hbond, t_ind_poss_hbond in zip(q_inds_poss_hbonds, t_inds_poss_hbonds):
            df_poss_hb_t = dft_.iloc[t_ind_poss_hbond]
            df_poss_hb_q = dfq_.iloc[q_inds_poss_hbond]

            q_inds_clash_, q_inds_hbond_ = self._angle_test(df_poss_hb_q, df_poss_hb_t,
                                                            q_inds_poss_hbond)
            if q_inds_clash_:
                q_inds_clash.append(q_inds_clash_)
                t_inds_clash.append(t_ind_poss_hbond)
            if q_inds_hbond_:
                q_inds_hbond.append(q_inds_hbond_)
                t_inds_hbond.append(t_ind_poss_hbond)
        return q_inds_clash, t_inds_clash, q_inds_hbond, t_inds_hbond

    def angle_test_S_acceptor(self, q_inds_poss_hbonds, t_inds_poss_hbonds, dfq_, dft_):
        q_inds_clash = []
        t_inds_clash = []
        q_inds_hbond = []
        t_inds_hbond = []
        for q_inds_poss_hbond, t_ind_poss_hbond in zip(q_inds_poss_hbonds, t_inds_poss_hbonds):
            df_poss_hb_t = dft_.iloc[t_ind_poss_hbond]
            df_poss_hb_q = dfq_.iloc[q_inds_poss_hbond]

            q_inds_clash_, q_inds_hbond_ = self._angle_test_S_acceptor(df_poss_hb_q, df_poss_hb_t,
                                                                       q_inds_poss_hbond)
            if q_inds_clash_:
                q_inds_clash.append(q_inds_clash_)
                t_inds_clash.append(t_ind_poss_hbond)
            if q_inds_hbond_:
                q_inds_hbond.append(q_inds_hbond_)
                t_inds_hbond.append(t_ind_poss_hbond)
        return q_inds_clash, t_inds_clash, q_inds_hbond, t_inds_hbond

    def _find_contact_indices(self, atom_type_q, atom_type_t):
        tree = self._balltrees[atom_type_q]
        dft_ = self.dft_atom_type[atom_type_t]
        vdw_sum = self.vdw_radii[atom_type_q] + self.vdw_radii[atom_type_t]
        cc_low = vdw_sum  # - self.tol
        cc_high = vdw_sum + self.gap_close_contact
        cutoff = wc_high = vdw_sum + self.gap_wide_contact
        i, d = tree.query_radius(dft_[['c_x', 'c_y', 'c_z']].values,
                                 r=cutoff, return_distance=True)
        t_inds, q_inds, dists = self.prune_empty(d, i)
        if t_inds:
            D_q = atom_type_q in hbond_donor_types
            A_t = atom_type_t in hbond_acceptor_types
            A_q = atom_type_q in hbond_acceptor_types
            D_t = atom_type_t in hbond_donor_types
            if not ((D_q and A_t) or (D_t and A_q)):
                return self.partition_contacts_no_hb(dists, q_inds, t_inds, cc_low, cc_high, wc_high)
                # This returns q_inds_clashes, t_inds_clashes, q_inds_cc, t_inds_cc,
                # q_inds_wc, t_inds_wc, q_inds_hb, t_inds_hb (hb are empty lists)

            if (atom_type_q in {'n', 'p', 's'}) and (atom_type_t in {'n', 'p', 's'}):
                hb_hard_cutoff = cc_low - self.overlap_hb_heavy_nn
            elif (atom_type_q in {'o', 'f'}) and (atom_type_t in {'f', 'n', 'p', 's'}):
                hb_hard_cutoff = cc_low - self.overlap_hb_heavy_no
            elif (atom_type_t in {'o', 'f'}) and (atom_type_q in {'f', 'n', 'p', 's'}):
                hb_hard_cutoff = cc_low - self.overlap_hb_heavy_no
            elif (atom_type_q in {'o', 'f'}) and (atom_type_t in {'o', 'f'}):
                hb_hard_cutoff = cc_low - self.overlap_hb_heavy_oo
            if (atom_type_q in {'n', 'o', 'p', 's', 'f'}) and (atom_type_t in {'n', 'o', 'p', 's', 'f'}):
                cc_low_hb = max(3.3, cc_low)
                cc_high = max(3.6, cc_low + 0.3)
                wc_high = max(4.0, cc_high + 0.4)
            else:
                hb_hard_cutoff = cc_low - self.overlap_hb - self.tol
                cc_low_hb = max(2.5, cc_low)
                cc_high = max(2.8, cc_low + 0.3)
                wc_high = max(3.2, cc_high + 0.4)
            q_inds_clashes, t_inds_clashes, q_inds_cc, t_inds_cc, \
            q_inds_wc, t_inds_wc, q_inds_poss_hbonds_cl, t_inds_poss_hbonds_cl, \
            q_inds_poss_hbonds_cc, t_inds_poss_hbonds_cc = \
                self.partition_contacts_hb_hard_cutoff(dists, q_inds, t_inds, cc_low,
                                                       cc_low_hb, cc_high, wc_high,
                                                       hb_hard_cutoff)

            q_inds_hbond = []
            t_inds_hbond = []
            if q_inds_poss_hbonds_cl:
                dfq_ = self.dfq_atom_type[atom_type_q]
                if atom_type_q in {'s'} or atom_type_t in {'s'}:
                    q_inds_clash, t_inds_clash, q_inds_hbond_cl, t_inds_hbond_cl \
                        = self.angle_test_S_acceptor(q_inds_poss_hbonds_cl, t_inds_poss_hbonds_cl, dfq_, dft_)
                else:
                    q_inds_clash, t_inds_clash, q_inds_hbond_cl, t_inds_hbond_cl \
                        = self.angle_test(q_inds_poss_hbonds_cl, t_inds_poss_hbonds_cl, dfq_, dft_)
                q_inds_clashes.extend(q_inds_clash)
                t_inds_clashes.extend(t_inds_clash)
                q_inds_hbond.extend(q_inds_hbond_cl)
                t_inds_hbond.extend(t_inds_hbond_cl)
            if q_inds_poss_hbonds_cc:
                dfq_ = self.dfq_atom_type[atom_type_q]
                if atom_type_q in {'s'} or atom_type_t in {'s'}:
                    q_inds_cc_, t_inds_cc_, q_inds_hbond_cc, t_inds_hbond_cc \
                        = self.angle_test_S_acceptor(q_inds_poss_hbonds_cc, t_inds_poss_hbonds_cc, dfq_, dft_)
                else:
                    q_inds_cc_, t_inds_cc_, q_inds_hbond_cc, t_inds_hbond_cc \
                        = self.angle_test(q_inds_poss_hbonds_cc, t_inds_poss_hbonds_cc, dfq_, dft_)
                q_inds_cc.extend(q_inds_cc_)
                t_inds_cc.extend(t_inds_cc_)
                q_inds_hbond.extend(q_inds_hbond_cc)
                t_inds_hbond.extend(t_inds_hbond_cc)
                return q_inds_clashes, t_inds_clashes, q_inds_cc, t_inds_cc, \
                       q_inds_wc, t_inds_wc, q_inds_hbond, t_inds_hbond
            else:
                return q_inds_clashes, t_inds_clashes, q_inds_cc, t_inds_cc, \
                       q_inds_wc, t_inds_wc, q_inds_hbond, t_inds_hbond
        else:
            return [[]] * 8

    def extend_global_indices(self, atom_type_q, atom_type_t, q_inds, t_inds, contact_type):
        q_global_inds = list()
        t_global_inds = list()
        dfq = self.dfq_atom_type[atom_type_q]
        dft = self.dft_atom_type[atom_type_t]
        for q_inds_, t_ind in zip(q_inds, t_inds):
            q_global_inds.extend(dfq['num_tag'].iloc[q_inds_].values)
            t_global_inds.extend([dft['num_tag'].iat[t_ind[0]]] * len(q_inds_))

        self.q_global_indices.extend(q_global_inds)
        self.t_global_indices.extend(t_global_inds)
        self.contact_types.extend([contact_type] * len(q_global_inds))

    def _find(self):
        for atom_type_q in self.atom_types_dfq:
            for atom_type_t in self.atom_types_dft:
                q_inds_clashes, t_inds_clashes, q_inds_cc, t_inds_cc, \
                q_inds_wc, t_inds_wc, q_inds_hbond, t_inds_hbond \
                    = self._find_contact_indices(atom_type_q, atom_type_t)
                if q_inds_clashes:
                    self.extend_global_indices(atom_type_q, atom_type_t,
                                               q_inds_clashes, t_inds_clashes, 'cl')
                if q_inds_cc:
                    self.extend_global_indices(atom_type_q, atom_type_t,
                                               q_inds_cc, t_inds_cc, 'cc')
                if q_inds_wc:
                    self.extend_global_indices(atom_type_q, atom_type_t,
                                               q_inds_wc, t_inds_wc, 'wc')
                if q_inds_hbond:
                    self.extend_global_indices(atom_type_q, atom_type_t,
                                               q_inds_hbond, t_inds_hbond, 'hb')
        dfq_ = self.dfq.iloc[self.q_global_indices]
        dft_ = self.dft.iloc[self.t_global_indices]
        df = dfq_.reset_index(drop=True).join(dft_.reset_index(drop=True),
                                              how='outer', lsuffix='_q', rsuffix='_t')
        df.loc[:, 'contact_type'] = self.contact_types
        self.df_contacts = df

    def find(self):

        if self.atom_types_dfq is None:
            self.set_atom_types()

        if not self.dfq_atom_type:
            self.split_dfs_to_atom_types()

        if not self._balltrees:
            self.make_trees()

        self._find()


class ClashVDM:
    '''Do Cbeta, do non-Cbeta, combine results'''
    def __init__(self, dfq, dft):
        self.dfq = dfq
        self.dft = dft
        self.exclude = None
        self.q_grouping = None
        self.dfq_cb_clash_free = None
        self.dfq_non_cb_clash_free = None
        self.dfq_clash_free = None
        self.dft_for_non_cb = None
        self.dft_for_cb = None
        self.dfq_non_cb = None
        self.dfq_cb = None

    def set_grouping(self, grouping):
        self.q_grouping = grouping

    def set_exclude(self, exclude):
        self.exclude = exclude

    def setup(self):
        df = self.dft
        resnum = self.exclude[0]
        chain = self.exclude[1]
        seg = self.exclude[2]
        gen_exclude = ((df['resnum'] == resnum) &
                       (df['chain'] == chain) &
                       (df['segment'] == seg) &
                       ~(df['name'].isin(['H', 'O'])))
        # cbeta_exclude is for a 4-bond distance condition to start
        # measuring clashes between atoms.  For Cbeta atoms, this means
        # excluding the i-1 residue's C atom and the i+1 residue's N atom.
        res = ((df['resnum'] == resnum) &
               (df['chain'] == chain) &
               (df['segment'] == seg))
        resm1 = ((df['resnum'] == resnum - 1) &
                 (df['chain'] == chain) &
                 (df['segment'] == seg) &
                 (df['name'] == 'C'))
        resp1 = ((df['resnum'] == resnum + 1) &
                 (df['chain'] == chain) &
                 (df['segment'] == seg) &
                 (df['name'] == 'N'))
        if resm1.any() and resp1.any():
            cbeta_exclude = (resm1 | res | resp1)
        elif resm1.any():
            cbeta_exclude = (resm1 | res)
        else:
            cbeta_exclude = (resp1 | res)
        self.dft_for_non_cb = df[~gen_exclude].copy()
        self.dft_for_cb = df[~cbeta_exclude].copy()

        cb_crit = (self.dfq.name == 'CB') & (self.dfq.chain == 'X')
        self.dfq_non_cb = self.dfq[~cb_crit].copy()
        self.dfq_cb = self.dfq[cb_crit].copy()

    def set_index(self, df):
        df.set_index(self.q_grouping, inplace=True, drop=False)

    def find_cb_clash(self, **kwargs):
        cla = Clash(self.dfq_cb, self.dft_for_cb, **kwargs)
        cla.set_grouping(self.q_grouping)
        cla.find(return_clash_free=True, return_clash=True)
        self.dfq_cb_clash = cla.dfq_clash
        self.dfq_cb_clash_free = cla.dfq_clash_free

    def find_non_cb_clash(self, **kwargs):
        cla = Clash(self.dfq_non_cb, self.dft_for_non_cb, **kwargs)
        cla.set_grouping(self.q_grouping)
        cla.find(return_clash_free=True, return_clash=True)
        self.dfq_non_cb_clash = cla.dfq_clash
        self.dfq_non_cb_clash_free = cla.dfq_clash_free

    def find(self, **kwargs):
        self.find_cb_clash(**kwargs)
        self.find_non_cb_clash(**kwargs)
        self.set_index(self.dfq_non_cb_clash_free)
        self.set_index(self.dfq_cb_clash)
        self.set_index(self.dfq_cb_clash_free)
        self.set_index(self.dfq_non_cb_clash)
        isin1 = self.dfq_non_cb_clash_free.index.isin(self.dfq_cb_clash.index)
        df1 = self.dfq_non_cb_clash_free.loc[~isin1]
        isin2 = self.dfq_cb_clash_free.index.isin(self.dfq_non_cb_clash.index)
        df2 = self.dfq_cb_clash_free.loc[~isin2]
        self.dfq_clash_free = concat((df1, df2), sort=False, ignore_index=True)




def find_buried_unsatisfied_hbonds(pdb, lig_resname=None, lig_can_hbond_dict=None,
                                   lig_atom_type_dict=None, append_to_file=False,
                                   outdir='./', ignore=set(), alpha=9):
    try:
        if isinstance(pdb, str):
            pdb_name = pdb.split('/')[-1].split('.')[0]
            pdb = parsePDB(pdb)
        else:
            pdb_name = str(pdb).split()[1]
    except:
        raise TypeError('*pdb* must be a pdb file or prody object')

    if set(pdb.getSegnames()) == {''}:
        pdb.setSegnames('A')

    ahull = AlphaHull(alpha=alpha)
    ahull.set_coords(pdb)
    ahull.calc_hull()

    if (lig_resname is not None) and (lig_can_hbond_dict is not None) \
            and (lig_atom_type_dict is not None):
        dflig = make_lig_df(pdb.select('resname ' + lig_resname),
                                       **{'lig_atom_types_dict': lig_atom_type_dict,
                                          'can_hbond_dict': lig_can_hbond_dict})
        columns_to_file = ['resnum_q', 'chain_q', 'segment_q', 'resname_q',
                           'name_q', 'resnum_t', 'chain_t', 'segment_t',
                           'resname_t', 'name_t', 'lig_resname', 'lig_name',
                           'contact_type']
    else:
        columns_to_file = ['resnum_q', 'chain_q', 'segment_q', 'resname_q',
                           'name_q', 'resnum_t', 'chain_t', 'segment_t',
                           'resname_t', 'name_t', 'contact_type']

    if append_to_file:
        status = 'a'
        pdb_filename = pdb_name + '.pdb'
    else:
        status = 'w'
        pdb_filename = 'buried_unsat_hb_' + pdb_name + '.txt'

    if outdir[-1] != '/':
        outdir += '/'

    with open(outdir + pdb_filename, status) as outfile:
        outfile.write('\n')
        num_bur_unsatisfied = 0
        num_bur_unsatisfied_ignore = 0
        df_pdb = make_pose_df(pdb.select('protein and sidechain'))
        df_pdb_polar = df_pdb[df_pdb.atom_type_label.isin({'h_pol', 'o', 'n'})]
        for n, row in df_pdb_polar.iterrows():
            if (row.atom_type_label == 'n') and ~np.isnan(row.c_D_x):
                continue
            seg, ch, rn = row.seg_chain_resnum
            q_sel = pdb.select('resnum ' + str(rn) + ' chain ' + ch + ' segment ' + seg + ' and sidechain')
            if q_sel is None:
                continue
            if q_sel.select('element O N') is None:
                continue
            name = row['name']
            q_sel_atom = q_sel.select('name ' + name)
            is_buried = ahull.get_pnt_distance(q_sel_atom.getCoords().flatten()) > 1
            if is_buried:
                dfq = make_pose_df(q_sel)
                t_sel = pdb.select(
                    'protein and not (resnum ' + str(rn) + ' chain ' + ch + ' segment ' + seg + ' and sidechain)')
                dft = make_pose_df(t_sel)
                if lig_resname is not None:
                    dft = concat((dft, dflig), sort=False)
                con = Contact(dfq=dfq, dft=dft)
                con.find()
                isON = con.df_contacts.name_q == name
                df = con.df_contacts[isON]
                if not any(df.contact_type == 'hb'):
                    outfile.write('(' + seg + ', ' + ch + ', ' + str(rn) + ', ' + name + '), ' + df[columns_to_file].to_string() + ' \n')
                    num_bur_unsatisfied += 1
                    if (seg, ch, rn, name) not in ignore:
                        num_bur_unsatisfied_ignore += 1
        outfile.write('\n')
        outfile.write('Total_buried_unsatisfied_hbonds: ' + str(num_bur_unsatisfied) + ' \n')
        outfile.write('Total_buried_unsatisfied_hbonds_ignore: ' + str(num_bur_unsatisfied_ignore) + ' \n')