__all__ = ['search_loop', 'pick_loop', 'paste_loop', 'print_loop_pdb', 'check_loop_clash',
           'check_loop_connectivity', 'make_loop']

import prody as pr
import os
import numpy as np
from .functions import writePDBStream
import pyrosetta as py

def setup_scorefxn():
    py.init()
    scorefxn = py.ScoreFunction()
    fa_rep = py.rosetta.core.scoring.ScoreType.fa_rep
    scorefxn.set_weight(fa_rep, 1.0)
    return scorefxn

def norm_score(x, max_, min_):
    return (x - min_) / (max_ - min_)


def loop_score(rmsd, min_rmsd, max_rmsd, cum_rmsd, min_cum_rmsd, max_cum_rmsd):
    return norm_score(rmsd, max_rmsd, min_rmsd) + norm_score(cum_rmsd, max_cum_rmsd, min_cum_rmsd)


def search_loop(path_to_pdb, query_selection, outdir, path_to_master, tag='', low=3, high=10, rmsd_cutoff=1):
    if outdir[-1] != '/':
        outdir += '/'
    if path_to_master[-1] != '/':
        path_to_master += '/'
    try:
        os.makedirs(outdir)
    except:
        pass
    pdb = pr.parsePDB(path_to_pdb)
    query = pdb.select(query_selection)
    pr.writePDB(outdir + 'query' + tag + '.pdb', query.select('backbone'))
    _cwd = os.getcwd()
    os.chdir(outdir)
    os.system(path_to_master + 'createPDS --type query --pdb ' + 'query' + tag + '.pdb')
    os.system('for i in {' + str(low) + '..' + str(high) + '}; do ' + path_to_master
              + '/master --query query' + tag + '.pds '
              '--targetList ' + path_to_master + '/database/list --rmsdCut ' + str(rmsd_cutoff) + ' --gapLen $i '
              '--matchOut output' + tag + '_' + '$i/match.txt --structOut output' + tag + '_' + '$i '
              '--outType wgap --topN 50 >/dev/null; done')
    os.chdir(_cwd)


def set_bonds(pdb):
    """Sets backbone bonds of chain based on proximity of atoms."""
    try:
        bb_sel = pdb.select('protein and name N C CA')
        dm = pr.buildDistMatrix(bb_sel)
        ind = np.where((np.tril(dm) < 1.7) & (np.tril(dm) > 0))
        atom_ind = bb_sel.getIndices()
        pdb.setBonds([(atom_ind[i], atom_ind[j]) for i, j in zip(ind[0], ind[1])])
        return True
    except:
        return False


def check_loop_connectivity(best_loop, query_length=8):
    pdb_name = sorted([p for p in os.listdir(best_loop[1]) if p[0] == 'w'])[0]
    loop = pr.parsePDB(best_loop[1] + '/' + pdb_name)
    bond_test = set_bonds(loop)
    loop_len = int(best_loop[1].split('_')[-1])
    resind1 = loop.getResindices()[0]
    if bond_test:
        un_res = np.unique(loop.select('backbone and bonded 100 to resindex ' + str(resind1)).getResnums())
        real_loop_len = len(un_res) - query_length
    else:
        return False
    if real_loop_len == loop_len:
        return True
    else:
        return False


def check_loop_clash(path_to_pdb_w_loop, scorefxn, clash_en_threshold=5):
    pose = py.pose_from_pdb(path_to_pdb_w_loop)
    scorefxn(pose)
    if any([pose.energies().residue_total_energy(i) > clash_en_threshold
            for i in range(1, pose.pdb_info().nres() + 1)]):
        return False
    else:
        return True


def pick_loop(path_to_loop_dir, topN=50, query_length=8):
    _cwd = os.getcwd()
    os.chdir(path_to_loop_dir)
    low_rmsd_dict = {}
    cum_rmsd_dict = {}
    for dir_ in [d for d in os.listdir() if d[-3:-1] != 'pd']:
        if len(os.listdir(dir_)) > 0:
            with open(dir_ + '/match.txt', 'r') as infile:
                len_ = len(infile.readlines())
            if len_ == topN:
                rmsds = []
                with open(dir_ + '/match.txt', 'r') as infile:
                    for line in infile:
                        rmsds.append(float(line.split()[0]))
                low_rmsd = min(rmsds)
                cum_rmsd = np.sum(rmsds)
                low_rmsd_dict[dir_] = low_rmsd
                cum_rmsd_dict[dir_] = cum_rmsd

    min_rmsd = min([val for val in low_rmsd_dict.values()])
    max_rmsd = max([val for val in low_rmsd_dict.values()])
    min_cum_rmsd = min([val for val in cum_rmsd_dict.values()])
    max_cum_rmsd = max([val for val in cum_rmsd_dict.values()])

    loop_scores = {}
    for key, rmsd in low_rmsd_dict.items():
        loop_scores[key] = loop_score(rmsd, min_rmsd, max_rmsd, cum_rmsd_dict[key], min_cum_rmsd, max_cum_rmsd)

    best_loop = min([(val, key) for key, val in loop_scores.items()])

    if not check_loop_connectivity(best_loop, query_length=query_length):
        pdb_name = sorted([p for p in os.listdir(best_loop[1]) if p[0] == 'w'])[0]
        os.system('tail -49 ' + best_loop[1] + '/match.txt > ' + best_loop[1] + '/match_new.txt')
        os.system('tail -1 ' + best_loop[1] + '/match.txt >> ' + best_loop[1] + '/match_new.txt')
        os.system('mv ' + best_loop[1] + '/match_new.txt ' + best_loop[1] + '/match.txt')
        os.system('rm ' + best_loop[1] + '/' + pdb_name)
        os.chdir(_cwd)
        return pick_loop(path_to_loop_dir, topN=topN, query_length=query_length)
    os.chdir(_cwd)
    return best_loop


def paste_loop(best_loop, path_to_loop_dir, path_to_pdb, query_selection_N, query_selection_C, query_length=8):
    _cwd = os.getcwd()
    os.chdir(path_to_loop_dir)
    pdb_name = sorted([p for p in os.listdir(best_loop[1]) if p[0] == 'w'])[0]
    loop = pr.parsePDB(best_loop[1] + '/' + pdb_name)
    loop_bb = loop.select('backbone')
    pdb = pr.parsePDB(path_to_pdb)
    query_N = pdb.select(query_selection_N)
    query_N_bb = query_N.select('name N C CA')
    query_C = pdb.select(query_selection_C)
    query_C_bb = query_C.select('name N C CA')
    num = int(query_length / 2 - 1)

    first_resnum_loop = loop_bb.getResnums()[0]
    last_resnum_loop = loop_bb.getResnums()[-1]
    loop_N_bb = loop_bb.select('name N C CA and resnum ' + str(first_resnum_loop) + 'to' + str(first_resnum_loop + num))
    loop_C_bb = loop_bb.select('name N C CA and resnum ' + str(last_resnum_loop - num) + 'to' + str(last_resnum_loop))

    try:
        coords_diff_N = loop_N_bb.getCoords() - query_N_bb.getCoords()
        coords_diff_C = loop_C_bb.getCoords() - query_C_bb.getCoords()
    except ValueError:
        pdb_name = sorted([p for p in os.listdir(best_loop[1]) if p[0] == 'w'])[0]
        os.system('tail -49 ' + best_loop[1] + '/match.txt > ' + best_loop[1] + '/match_new.txt')
        os.system('tail -1 ' + best_loop[1] + '/match.txt >> ' + best_loop[1] + '/match_new.txt')
        os.system('mv ' + best_loop[1] + '/match_new.txt ' + best_loop[1] + '/match.txt')
        os.system('rm ' + best_loop[1] + '/' + pdb_name)
        os.chdir(_cwd)
        best_loop = pick_loop(path_to_loop_dir, topN=50, query_length=query_length)
        return paste_loop(best_loop, path_to_loop_dir, path_to_pdb, query_selection_N, query_selection_C)

    ind_match_N = np.argmin([np.linalg.norm(i) for i in coords_diff_N])
    ind_match_C = np.argmin([np.linalg.norm(i) for i in coords_diff_C])

    loop_N_bb_index = loop_N_bb.getIndices()[ind_match_N]
    loop_C_bb_index = loop_C_bb.getIndices()[ind_match_C]
    query_N_bb_index = query_N_bb.getIndices()[ind_match_N]
    query_C_bb_index = query_C_bb.getIndices()[ind_match_C]
    first_index_pdb = pdb.select('backbone').getIndices()[0]
    last_index_pdb = pdb.select('backbone').getIndices()[-1]

    loop_slice = loop_bb.select('index ' + str(loop_N_bb_index) + 'to' + str(loop_C_bb_index))
    pdb_N = pdb.select('backbone and index ' + str(first_index_pdb) + 'to' + str(query_N_bb_index - 1))
    pdb_C = pdb.select('backbone and index ' + str(query_C_bb_index + 1) + 'to' + str(last_index_pdb))
    os.chdir(_cwd)
    return pdb_N, loop_slice, pdb_C


def print_loop_pdb(pdb_Ns, loop_slices, pdb_Cs, outdir):
    """pdb_Ns, loop_slices, pdb_Cs are lists that should be in order of increasing residue number (i.e., N to C)"""
    if outdir[-1] != '/':
        outdir += '/'
    try:
        os.makedirs(outdir)
    except:
        pass
    with open(outdir + repr(pdb_Ns[0]).split()[-3] + '_loops.pdb', 'w') as outfile:
        l = len(loop_slices)
        pdbC_first_index = pdb_Ns[0].getIndices()[0]
        pdbN_first_index_resnum = 0
        loop_last_resnum = -1
        for i, (pdbN, loop, pdbC) in enumerate(zip(pdb_Ns, loop_slices, pdb_Cs)):
            pdbN_first_index = pdbC_first_index
            pdbN_resnums = pdbN.getResnums()
            if pdbN.select('index ' + str(pdbN_first_index)).getNames()[0] == 'N':
                pdbN.setResnums(pdbN_resnums - pdbN_first_index_resnum + loop_last_resnum + 1)
            else:
                pdbN.setResnums(pdbN_resnums - pdbN_first_index_resnum + loop_last_resnum)
            num_ind = len(pdbN.getIndices())
            start = 1
            finish = start + num_ind
            pdbN.setBetas(list(range(start, finish)))
            pdbN.setResnames('GLY')
            pdbN.setChids('A')
            pdbN_last_resnum = pdbN.getResnums()[-1]
            loop_resnums = loop.getResnums()
            loop_first_resnum = loop_resnums[0]
            if loop.getNames()[0] == 'N':
                loop.setResnums(loop_resnums - loop_first_resnum + pdbN_last_resnum + 1)
            else:
                loop.setResnums(loop_resnums - loop_first_resnum + pdbN_last_resnum)
            loop.setChids('A')

            num_ind = len(loop.getIndices())
            start = finish
            finish = start + num_ind
            loop.setBetas(list(range(start, finish)))
            loop.setResnames('GLY')
            pdbN_last_index = pdbN.getIndices()[-1]
            loop_last_resnum = loop.getResnums()[-1]
            pdbC_first_index = pdbC.getIndices()[0]
            pdbN_first_index_resnum = pdbC.select('index ' + str(pdbC_first_index)).getResnums()[0]
            writePDBStream(outfile, pdbN.select('index ' + str(pdbN_first_index) + 'to' + str(pdbN_last_index)))
            writePDBStream(outfile, loop)
            pdbN.setResnums(pdbN_resnums)
            loop.setResnums(loop_resnums)

            if i + 1 == l:
                num_ind = len(pdbC.getIndices())
                start = finish
                finish = start + num_ind
                pdbC.setBetas(list(range(start, finish)))
                pdbC.setResnames('GLY')
                pdbC_resnums = pdbC.getResnums()
                pdbC_first_resnum = pdbC_resnums[0]
                if pdbC.getNames()[0] == 'N':
                    pdbC.setResnums(pdbC_resnums - pdbC_first_resnum + loop_last_resnum + 1)
                else:
                    pdbC.setResnums(pdbC_resnums - pdbC_first_resnum + loop_last_resnum)
                pdbC.setChids('A')
                writePDBStream(outfile, pdbC)
                pdbC.setResnums(pdbC_resnums)


def make_loop(path_to_loop_dirs, path_to_pdb, query_selection_Ns, query_selection_Cs, scorefxn,
              outdir, topN=50, query_length=8):

    pdb_Ns = []
    pdb_Cs = []
    loop_slices =[]
    best_loops = []
    for i in range(len(path_to_loop_dirs)):
        best_loop = pick_loop(path_to_loop_dirs[i], topN=topN, query_length=query_length)
        best_loops.append(best_loop)
        pdb_N, loop_slice, pdb_C = paste_loop(best_loop, path_to_loop_dirs[i],
                                              path_to_pdb, query_selection_Ns[i], query_selection_Cs[i],
                                              query_length)
        pdb_Ns.append(pdb_N)
        loop_slices.append(loop_slice)
        pdb_Cs.append(pdb_C)
        print_loop_pdb(pdb_Ns, loop_slices, pdb_Cs, outdir)

        if not check_loop_clash(outdir + repr(pdb_Ns[0]).split()[-3] + '_loops.pdb', scorefxn):
            _cwd = os.getcwd()
            os.chdir(path_to_loop_dirs[i])
            pdb_name = sorted([p for p in os.listdir(best_loop[1]) if p[0] == 'w'])[0]
            os.system('tail -49 ' + best_loop[1] + '/match.txt > ' + best_loop[1] + '/match_new.txt')
            os.system('tail -1 ' + best_loop[1] + '/match.txt >> ' + best_loop[1] + '/match_new.txt')
            os.system('mv ' + best_loop[1] + '/match_new.txt ' + best_loop[1] + '/match.txt')
            os.system('rm ' + best_loop[1] + '/' + pdb_name)
            os.chdir(_cwd)
            return make_loop(path_to_loop_dirs, path_to_pdb, query_selection_Ns, query_selection_Cs, scorefxn,
                      outdir, topN=topN, query_length=query_length)
    return best_loops





