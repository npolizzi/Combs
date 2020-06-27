import sys
sys.path.append('~/combs/src/')
import combs
import pickle
from pandas import concat
import os

def make_reps(label):
    print('Making ' + label + ' vdM reps...')
    outdir = '~/combs/database/representatives/hb_only/carboxamide/' + label + '/'

    with open('~/combs/database/relative_vdMs/hb_only/asparagine/CONH/' + label + '.pkl', 'rb') as infile:
        asn = pickle.load(infile)
    with open('~/combs/database/relative_vdMs/hb_only/glutamine/CONH/' + label + '.pkl', 'rb') as infile:
        gln = pickle.load(infile)
    carboxamide = concat([asn, gln])
    outdir_relvdms = '~/combs/database/relative_vdMs/hb_only/carboxamide/'
    try:
        os.makedirs(outdir_relvdms)
    except FileExistsError:
        pass
    carboxamide.to_pickle(outdir_relvdms + label + '.pkl')

    path_to_df_pairwise = '~/combs/database/clusters/hb_only/carboxamide/' + label + '/'
    kwargs = dict(path_to_df_pairwise=path_to_df_pairwise)

    vdmreps = combs.apps.clashfilter.VdmReps(carboxamide, **kwargs)
    vdmreps.find_all_reps(outdir=outdir)

make_reps('SC')
make_reps('HNCA')
make_reps('CO')
make_reps('PHI_PSI')