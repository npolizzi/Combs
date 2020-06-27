import os
import prody as pr
import numpy as np
from .functions import writePDBStream

#query_sel_N = 'chain C and resnum 39to44'
#query_sel_C = 'chain D and resnum 4to12'


def paste_loop(path_to_loop, path_to_pdb, query_selection_N, query_selection_C, query_length_N=4, query_length_C=4,
               include_sidechains=False):
    loop = pr.parsePDB(path_to_loop)
    loop.setSegnames('A')
    loop_bb = loop.select('backbone')
    pdb = pr.parsePDB(path_to_pdb)
    query_N = pdb.select(query_selection_N)
    query_N_bb = query_N.select('name N C CA')
    query_C = pdb.select(query_selection_C)
    query_C_bb = query_C.select('name N C CA')

    first_resnum_loop = loop_bb.getResnums()[0]
    last_resnum_loop = loop_bb.getResnums()[-1]
    loop_N_bb = loop_bb.select('name N C CA and resnum `' + str(first_resnum_loop) + 'to' + str(first_resnum_loop + query_length_N - 1) + '`')
    loop_C_bb = loop_bb.select('name N C CA and resnum `' + str(last_resnum_loop - query_length_C + 1) + 'to' + str(last_resnum_loop) + '`')

    try:
        coords_diff_N = loop_N_bb.getCoords() - query_N_bb.getCoords()
        coords_diff_C = loop_C_bb.getCoords() - query_C_bb.getCoords()
    except ValueError:
        print('Loop failure')

    ind_match_N = np.argmin([np.linalg.norm(i) for i in coords_diff_N])
    ind_match_C = np.argmin([np.linalg.norm(i) for i in coords_diff_C])

    loop_N_bb_index = loop_N_bb.getIndices()[ind_match_N]
    loop_C_bb_index = loop_C_bb.getIndices()[ind_match_C]
    query_N_bb_index = query_N_bb.getIndices()[ind_match_N]
    query_C_bb_index = query_C_bb.getIndices()[ind_match_C]
    first_index_pdb = pdb.select('backbone').getIndices()[0]
    last_index_pdb = pdb.select('backbone').getIndices()[-1]

    loop_slice = loop_bb.select('index ' + str(loop_N_bb_index) + 'to' + str(loop_C_bb_index))
    if not include_sidechains:
        pdb_N = pdb.select('backbone and index ' + str(first_index_pdb) + 'to' + str(query_N_bb_index - 1))
        pdb_C = pdb.select('backbone and index ' + str(query_C_bb_index + 1) + 'to' + str(last_index_pdb))
    else:
        pdb_N = pdb.select('index ' + str(first_index_pdb) + 'to' + str(query_N_bb_index - 1))
        pdb_C = pdb.select('index ' + str(query_C_bb_index + 1) + 'to' + str(last_index_pdb))
    return pdb_N, loop_slice, pdb_C


def print_loop_pdb(pdb_Ns, loop_slices, pdb_Cs, outdir, keep_resnames=False, keep_ligand=True, original_pdb=None):
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
                pdbN.setResnums(pdbN_resnums - pdbN_first_index_resnum + loop_last_resnum + 0)
            num_ind = len(pdbN.getIndices())
            start = 1
            finish = start + num_ind
            pdbN.setBetas(list(range(start, finish)))
            if not keep_resnames:
                pdbN.setResnames('GLY')
            pdbN.setChids('A')
            pdbN.setSegnames('A')
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
                if not keep_resnames:
                    pdbC.setResnames('GLY')
                pdbC_resnums = pdbC.getResnums()
                pdbC_first_resnum = pdbC_resnums[0]
                if pdbC.getNames()[0] == 'N':
                    pdbC.setResnums(pdbC_resnums - pdbC_first_resnum + loop_last_resnum + 1)
                else:
                    pdbC.setResnums(pdbC_resnums - pdbC_first_resnum + loop_last_resnum)
                pdbC.setChids('A')
                pdbC.setSegnames('A')
                writePDBStream(outfile, pdbC)
                pdbC.setResnums(pdbC_resnums)

        if keep_ligand and original_pdb is not None:
            pdb = pr.parsePDB(original_pdb)
            lig = pdb.select('not protein')
            num_ind = len(lig.getIndices())
            start = finish
            finish = start + num_ind
            lig.setBetas(list(range(start, finish)))
            writePDBStream(outfile, lig)




def paste_bulge(path_to_loop, path_to_pdb, query_selection_N, query_selection_C, query_length_N=4, query_length_C=4):
    loop = pr.parsePDB(path_to_loop)
    loop.setSegnames('A')
    keep = list()
    psi_prev = 0
    for i, res in enumerate(loop.iterResidues()):
        if i > 0:
            try:
                phi = pr.calcPhi(res)
            except:
                phi = 0
            try:
                psi = pr.calcPsi(res)
                if psi > -32 and (phi + psi_prev <= -125):
                    resnum = set(res.getResnums()).pop()
                    keep.append(resnum + 4)
                psi_prev = psi
            except:
                pass

    if len(keep) > 0:
        loop_bb = loop.select('backbone or resnum ' + ' '.join([str(i) for i in keep]))
        loop.select('not resnum ' + ' '.join([str(i) for i in keep])).setResnames('GLY')
    else:
        loop_bb = loop.select('backbone')
        loop.setResnames('GLY')
    pdb = pr.parsePDB(path_to_pdb)
    query_N_bb = pdb.select(query_selection_N + ' and name N C CA')
    query_C_bb = pdb.select(query_selection_C + ' and name N C CA')

    first_resnum_loop = loop_bb.getResnums()[0]
    last_resnum_loop = loop_bb.getResnums()[-1]
    n_last = first_resnum_loop + query_length_N - 1
    c_first = last_resnum_loop - query_length_C + 1
    if len(keep) > 0:
        if any([k <= n_last for k in keep]):
            n_last = min(keep) - 1
        if any([k >= c_first for k in keep]):
            c_first = max(keep) + 1

    if n_last <= first_resnum_loop:
        n_last = first_resnum_loop + 1
    if c_first >= last_resnum_loop:
        c_first = last_resnum_loop - 1
    loop_N_bb = loop_bb.select('name N C CA and resnum `' + str(first_resnum_loop) + 'to' + str(n_last) + '`')
    loop_C_bb = loop_bb.select('name N C CA and resnum `' + str(c_first) + 'to' + str(last_resnum_loop) + '`')

    len_loop_N = n_last - first_resnum_loop + 1
    len_loop_C = last_resnum_loop - c_first + 1

    print('len loop N bb=', len_loop_N)
    print('len loop C bb=', len_loop_C)

    try:
        coords_diff_N = loop_N_bb.getCoords() - query_N_bb.getCoords()[:len_loop_N*3+1]
        coords_diff_C = loop_C_bb.getCoords() - query_C_bb.getCoords()[-len_loop_C*3:]
    except ValueError:
        print('Loop failure')

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
    return pdb_N, loop_slice, pdb_C


def print_bulge_pdb(pdb_Ns, loop_slices, pdb_Cs, outdir):
    """pdb_Ns, loop_slices, pdb_Cs are lists that should be in order of increasing residue number (i.e., N to C)"""
    if outdir[-1] != '/':
        outdir += '/'
    try:
        os.makedirs(outdir)
    except:
        pass

    with open(outdir + repr(loop_slices[0]).split()[-3] + '_' + repr(pdb_Ns[0]).split()[-3] + '_loop.pdb', 'w') as outfile:
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
                pdbN.setResnums(pdbN_resnums - pdbN_first_index_resnum + loop_last_resnum + 0)
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