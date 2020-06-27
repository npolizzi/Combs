import sys
sys.path.append('~/combs/src/')
import combs
import pickle
from os import makedirs, listdir
import pandas as pd
import prody as pr
from collections import defaultdict


def rec_dd():
    """returns a recursive dictionary"""
    return defaultdict(rec_dd)

path_to_reps = '~/combs/database/representatives/hb_only/carboxamide/'
outdir_base = '~/combs/database/representatives_w_ligand/hb_only/APX1/carboxamide/'

lig = pr.parsePDB('/Users/npolizzi/Projects/combs/src/runs/apixaban/APX_0001.pdb')

with open('/Users/npolizzi/Projects/combs/src/runs/apixaban/apx_can_hbond_dict.pkl', 'rb') as infile:
    lig_can_hbond_dict = pickle.load(infile)

with open('/Users/npolizzi/Projects/combs/src/runs/apixaban/apx_atom_type_dict.pkl', 'rb') as infile:
    lig_atom_type_dict = pickle.load(infile)

kwargs = dict(can_hbond_dict=lig_can_hbond_dict, lig_atom_types_dict=lig_atom_type_dict)
df_lig = combs.apps.clashfilter.make_lig_df(lig, **kwargs)

dict_corr = dict(APX=dict(ASN=dict(ND2='N3', CG='C11', OD1='O1', CB='C10'),
                          GLN=dict(NE2='N3', CD='C11', OE1='O1', CG='C10')))
df_corr = combs.apps.clashfilter.make_df_corr(dict_corr)

for label in ['PHI_PSI', 'SC', 'CO', 'HNCA']:
    print(label)
    outdir = outdir_base + label + '/'
    try:
        makedirs(outdir)
    except FileExistsError:
        pass

    for file in [f for f in listdir(path_to_reps + label) if f[0] != '.']:
        print(file)
        with open(path_to_reps + label + '/' + file, 'rb') as infile:
            df_reps = pickle.load(infile)
            suplig = combs.apps.clashfilter.SuperposeLig(df_reps, df_lig, df_corr, label)
            suplig.find()
            if label == 'PHI_PSI':
                suplig.df_nonclashing_lig = pd.merge(suplig.df_nonclashing_lig,
                                                     df_reps[['phi', 'psi', 'iFG_count',
                                                             'vdM_count', 'query_name']].drop_duplicates(),
                                                     on=['iFG_count', 'vdM_count', 'query_name'])
            suplig.df_nonclashing_lig.to_pickle(outdir + file)



