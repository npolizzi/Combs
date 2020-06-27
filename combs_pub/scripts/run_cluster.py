import sys
sys.path.append('~/combs/src/')
import combs
import pandas as pd

def main():
    resname = sys.argv[1]
    print('Resname = ', resname)
    print('Getting contacting vdMs...')
    csvpath = '~/combs/results/asparagine/csv/'
    comb_atoms_asn = 'CB CG OD1 ND2 HD21 HD22'.split()
    an_asn = combs.analyze.Analyze(comb_atoms_asn, csvpath)
    asn_distcon = an_asn.get_distant_contacting(7)
    asn_distcon_hb = pd.merge(asn_distcon, an_asn.ifg_hbond_vdm, on=['iFG_count', 'vdM_count'])
    csvpath = '~/combs/results/glutamine/csv/'
    comb_atoms_gln = 'CG CD OE1 NE2 HE21 HE22 2HE2'.split()
    an_gln = combs.analyze.Analyze(comb_atoms_gln, csvpath)
    gln_distcon = an_gln.get_distant_contacting(7)
    gln_distcon_hb = pd.merge(gln_distcon, an_gln.ifg_hbond_vdm, on=['iFG_count', 'vdM_count']) 
    gln_distcon_hb['query_name'] = 'glutamine'
    asn_distcon_hb['query_name'] = 'asparagine'
    ifg_dict = dict(ASN='CB CG OD1 ND2'.split(), GLN='CG CD OE1 NE2'.split())
    pickle_file_outdir = '~/combs/results/carboxamide/hb_only/clusters/'
    kwargs = dict(ifg_dict=ifg_dict, pickle_file_outdir=pickle_file_outdir)
    if resname in set(asn_distcon_hb.resname_vdm) | set(gln_distcon_hb.resname_vdm):
        print('parsing PDBs...')
        pdbs = an_asn.parse_vdms(asn_distcon_hb[asn_distcon_hb['resname_vdm'] == resname])
        pdbs.extend(an_gln.parse_vdms(gln_distcon_hb[gln_distcon_hb['resname_vdm'] == resname]))
        combs.apps.cluster.make_clusters_all_types_from_pdbs_min(pdbs, **kwargs)

if __name__ == '__main__':
    main()
