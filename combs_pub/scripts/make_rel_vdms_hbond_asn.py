import sys                                                                   
sys.path.append('~/combs/src/')
import combs
import pandas as pd

ifg_dict = {'ASN': 'CB CG OD1 ND2'}
csv_path = 'path_to_asn_comb_csv_file'
an = combs.analyze.Analyze(csv_path)
dist_vdms = an.get_distant_vdms(7)
dist_vdms_hbond = pd.merge(dist_vdms, an.ifg_hbond_vdm, on=['iFG_count', 'vdM_count'])
outpath='relative_vdM_output_directory'
cb = combs.parse.comb.Comb(ifg_dict)
combs.make_all_rel_vdms_hbond_df(dist_vdms_hbond, an, outpath, cb)

