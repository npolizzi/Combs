import time
t0 = time.time()
import sys
sys.path.append('/Users/npolizzi/Projects/combs/src/')
import combs
import prody as pr
from collections import defaultdict
import copy
import pandas as pd
import numpy as np


########  LOAD PROTEIN TEMPLATE  ###########
pdb_gly = pr.parsePDB('path_to_template_pdb_glycine')
pdb_ala = pr.parsePDB('path_to_template_pdb_alanine')
designable_resis = np.array([2, 3, 4, 5, 6, 7, 8, 9, 10,
                    11, 12, 13, 14, 15, 16, 17,
                    18, 19, 20, 35, 36, 37, 38,
                    39, 40, 41, 42, 43, 44, 45,
                    46, 47, 48, 49, 50, 51, 52,
                    53, 54, 55, 56, 57, 58, 59,
                    60, 61, 62, 63, 64, 65, 66,
                    67, 68, 69, 70, 71, 72, 73,
                    74, 75, 90, 91, 92, 93, 94,
                    95, 96, 97, 98, 99, 100, 101,
                    102, 103, 104, 105, 106, 107,
                    108, 109])
template = combs.apps.sample.Template(pdb_gly)
template.set_alpha_hull(pdb_w_CB=pdb_ala, alpha=9)
template_resnums = template.pdb.select('name CA').getResnums()
template_chids = template.pdb.select('name CA').getChids()
template_segments = template.pdb.select('name CA').getSegnames()
surf_inds = np.unique(template.alpha_hull.hull.reshape(-1))

surf_resis = template_resnums[surf_inds]
surf_chids = template_chids[surf_inds]
surf_segments = template_segments[surf_inds]
ind_surf_in_designable = np.in1d(surf_resis, designable_resis)
surf_resis_in_designable = surf_resis[ind_surf_in_designable]
surf_chains_in_designable = surf_chids[ind_surf_in_designable]
surf_segs_in_designable = surf_segments[ind_surf_in_designable]

bool_buried = np.ones(len(template_resnums), dtype=bool)
bool_buried[surf_inds] = False
buried_resis = template_resnums[bool_buried]
buried_chids = template_chids[bool_buried]
buried_segments = template_segments[bool_buried]
ind_buried_in_designable = np.in1d(buried_resis, designable_resis)
buried_resis_in_designable = buried_resis[ind_buried_in_designable]
buried_chains_in_designable = buried_chids[ind_buried_in_designable]
buried_segs_in_designable = buried_segments[ind_buried_in_designable]


########  LOAD VDMS  ###########
seq_csts_carbonyl = defaultdict(dict)

for i in surf_resis_in_designable:
    for label in ['SC', 'PHI_PSI', 'CO', 'HNCA']:
        if label == 'SC':
            resnames = ['SER', 'TRP', 'TYR', 'HIS', 'THR',
                        'ASN', 'GLN', 'ALA', 'LYS', 'ARG',
                        'GLU', 'ASP']
        else:
            resnames = ['SER', 'TRP', 'TYR', 'HIS', 'THR',
                        'ASN', 'GLN', 'ALA', 'LYS', 'ARG',
                        'GLU', 'ASP', 'GLY', 'LEU', 'PHE',
                        'VAL', 'ILE', 'MET']
        seq_csts_carbonyl[('A', 'A', i)][label] = resnames

for i in buried_resis_in_designable:
    for label in ['SC', 'PHI_PSI', 'CO', 'HNCA']:
        if label == 'SC':
            resnames = ['SER', 'TRP', 'TYR', 'HIS', 'THR',
                        'ALA']
        else:
            resnames = ['SER', 'TRP', 'TYR', 'HIS', 'THR',
                        'ALA', 'GLY', 'LEU', 'PHE',
                        'VAL', 'ILE', 'MET']
        seq_csts_carbonyl[('A', 'A', i)][label] = resnames

print('loading carbonyl_1 vdms...')
path_to_rep = '~/combs/database/representatives/hb_only/backboneCO/'
dict_corr = dict(APX=dict(ALA=dict(C='C8', O='O3'),
                          GLY=dict(C='C8', O='O3')))
df_lig_corr_carbonyl_1 = combs.apps.clashfilter.make_df_corr(dict_corr)
remove_from_df = {1: {'chain': 'Y', 'name': 'CA'}}
kwargs = dict(name='backboneCO', path=path_to_rep, sequence_csts=seq_csts_carbonyl,
              ligand_iFG_corr=df_lig_corr_carbonyl_1, remove_from_df=remove_from_df)
vdm_carbonyl_1 = combs.apps.sample.VdM(**kwargs)
vdm_carbonyl_1.load(template)
vdm_carbonyl_1.set_neighbors()

## For making an identical VDM on a separate part of the ligand:
print('loading carbonyl_2 vdms...')
dict_corr = dict(APX=dict(ALA=dict(C='C19', O='O2'),
                          GLY=dict(C='C19', O='O2')))
df_lig_corr_carbonyl_2 = combs.apps.clashfilter.make_df_corr(dict_corr)
kwargs = dict(name='backboneCO', path=path_to_rep, sequence_csts=seq_csts_carbonyl,
              ligand_iFG_corr=df_lig_corr_carbonyl_2, remove_from_df=remove_from_df)
vdm_carbonyl_2 = combs.apps.sample.VdM(**kwargs)
vdm_carbonyl_2.neighbors = copy.copy(vdm_carbonyl_1.neighbors)
vdm_carbonyl_2.dataframe_iFG_coords = vdm_carbonyl_1.dataframe_iFG_coords
vdm_carbonyl_2.dataframe = vdm_carbonyl_1.dataframe
vdm_carbonyl_2.dataframe_grouped = vdm_carbonyl_1.dataframe_grouped
vdm_carbonyl_2.ligand_iFG_corr_sorted = pd.merge(vdm_carbonyl_1.ligand_iFG_corr_sorted[['resname', 'name']],
                                                 vdm_carbonyl_2.ligand_iFG_corr, on=['resname', 'name'])


seq_csts_carboxamide = defaultdict(dict)

for i in surf_resis_in_designable:
    for label in ['SC', 'PHI_PSI', 'CO', 'HNCA']:
        if label == 'SC':
            resnames = ['SER', 'TRP', 'TYR', 'HIS', 'THR',
                        'ASN', 'GLN', 'ALA', 'LYS', 'ARG',
                        'GLU', 'ASP']
        else:
            resnames = ['SER', 'TRP', 'TYR', 'HIS', 'THR',
                        'ASN', 'GLN', 'ALA', 'LYS', 'ARG',
                        'GLU', 'ASP', 'GLY']
        seq_csts_carboxamide[('A', 'A', i)][label] = resnames

for i in buried_resis_in_designable:
    for label in ['SC', 'PHI_PSI', 'CO', 'HNCA']:
        if label == 'SC':
            resnames = ['SER', 'TRP', 'TYR', 'HIS', 'THR',
                        'ALA']
        else:
            resnames = ['SER', 'TRP', 'TYR', 'HIS', 'THR',
                        'ALA', 'GLY']
        seq_csts_carboxamide[('A', 'A', i)][label] = resnames


print('loading carboxamide vdms...')
path_to_rep = '~/combs/database/representatives/hb_only/carboxamide/'
dict_corr = dict(APX=dict(GLN=dict(NE2='N3', CD='C11', OE1='O1', CG='C10'),
                          ASN=dict(ND2='N3', CG='C11', OD1='O1', CB='C10')))
df_lig_corr_carboxamide = combs.apps.clashfilter.make_df_corr(dict_corr)
kwargs = dict(name='carboxamide', path=path_to_rep, sequence_csts=seq_csts_carboxamide,
              ligand_iFG_corr=df_lig_corr_carboxamide)
vdm_carboxamide = combs.apps.sample.VdM(**kwargs)
vdm_carboxamide.load(template)
vdm_carboxamide.set_neighbors()


########  LOAD LIGS and FIND POSES  ###########
path_to_cst = '~/combs/src/runs/apixaban/APX.cst'

seq_csts_lig = defaultdict(dict)
for i in designable_resis:
    for label in ['PHI_PSI', 'SC', 'HNCA', 'CO']:
        if label == 'SC':
            resnames = ['SER', 'TRP', 'TYR', 'HIS', 'THR',
                       'ASN', 'GLN', 'ALA', 'LYS', 'ARG',
                       'GLU', 'ASP', 'PHE', 'LEU', 'VAL',
                        'MET', 'CYS', 'ILE']
        else:
            resnames = ['SER', 'TRP', 'TYR', 'HIS', 'THR',
                        'ASN', 'GLN', 'ALA', 'LYS', 'ARG',
                        'GLU', 'ASP', 'PHE', 'LEU', 'VAL',
                        'MET', 'CYS', 'ILE', 'GLY']
        seq_csts_lig[('A', 'A', i)][label] = resnames

def load_and_prune_ligand(path_to_sig_wlig, lig_name, df_lig_corr):
    kwargs = dict(name=lig_name, path=path_to_sig_wlig,
                  sequence_csts=seq_csts_lig, ligand_iFG_corr=df_lig_corr,
                  percent_buried=0.7, num_heavy=26)
    print('loading ligands for ', kwargs['name'], '...')
    lig = combs.apps.sample.Ligand(**kwargs)
    lig.load(template)
    lig.set_csts(path_to_cst)
    return lig

path_to_sig_wlig = '~/combs/database/representatives_w_ligand/hb_only/APX1/carboxamide/'
lig_carboxamide = load_and_prune_ligand(path_to_sig_wlig, lig_name='carboxamide',
                                        df_lig_corr=df_lig_corr_carboxamide)
print('finding poses carboxamide...')
lig_carboxamide.find_frag_neighbors([vdm_carboxamide, vdm_carbonyl_1, vdm_carbonyl_2], template)

path_to_sig_wlig = '~/combs/database/representatives_w_ligand/hb_only/APX1/backboneCO_1/'
lig_carbonyl_1 = load_and_prune_ligand(path_to_sig_wlig, lig_name='backboneCO',
                                        df_lig_corr=df_lig_corr_carbonyl_1)
print('finding poses CO_1...')
lig_carbonyl_1.find_frag_neighbors([vdm_carboxamide, vdm_carbonyl_1, vdm_carbonyl_2], template)

path_to_sig_wlig = '~/combs/database/representatives_w_ligand/hb_only/APX1/backboneCO_2/'
lig_carbonyl_2 = load_and_prune_ligand(path_to_sig_wlig, lig_name='backboneCO',
                                        df_lig_corr=df_lig_corr_carbonyl_2)
print('finding poses CO_2...')
lig_carbonyl_2.find_frag_neighbors([vdm_carboxamide, vdm_carbonyl_1, vdm_carbonyl_2], template)

