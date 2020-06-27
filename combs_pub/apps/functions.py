__all__ = ['writePDBStream']

import numpy as np
import prody as pr
import os


def writePDBStream(stream, atoms, csets=None, **kwargs):
    """Write *atoms* in PDB format to a *stream*.

    Needs selection with bfactors to be printed as atom indices.

    :arg stream: anything that implements a :meth:`write` method (e.g. file,
        buffer, stdout)"""

    # remark = str(atoms)
    PDBLINE = ('{0:6s}{1:5d} {2:4s}{3:1s}'
               '{4:4s}{5:1s}{6:4d}{7:1s}   '
               '{8:8.3f}{9:8.3f}{10:8.3f}'
               '{11:6.2f}{12:6.2f}      '
               '{13:4s}{14:2s}\n')

    PDBLINE_LT100K = ('%-6s%5d %-4s%1s%-4s%1s%4d%1s   '
                      '%8.3f%8.3f%8.3f%6.2f%6.2f      '
                      '%4s%2s\n')

    PDBLINE_GE100K = ('%-6s%5x %-4s%1s%-4s%1s%4d%1s   '
                      '%8.3f%8.3f%8.3f%6.2f%6.2f      '
                      '%4s%2s\n')

    try:
        coordsets = atoms._getCoordsets(csets)
    except AttributeError:
        try:
            coordsets = atoms._getCoords()
        except AttributeError:
            raise TypeError('atoms must be an object with coordinate sets')
        if coordsets is not None:
            coordsets = [coordsets]
    else:
        if coordsets.ndim == 2:
            coordsets = [coordsets]
    if coordsets is None:
        raise ValueError('atoms does not have any coordinate sets')

    try:
        acsi = atoms.getACSIndex()
    except AttributeError:
        try:
            atoms = atoms.getAtoms()
        except AttributeError:
            raise TypeError('atoms must be an Atomic instance or an object '
                            'with `getAtoms` method')
        else:
            if atoms is None:
                raise ValueError('atoms is not associated with an Atomic '
                                 'instance')
            try:
                acsi = atoms.getACSIndex()
            except AttributeError:
                raise TypeError('atoms does not have a valid type')

    try:
        atoms.getIndex()
    except AttributeError:
        pass
    else:
        atoms = atoms.select('all')

    n_atoms = atoms.numAtoms()

    occupancy = kwargs.get('occupancy')
    if occupancy is None:
        occupancies = atoms._getOccupancies()
        if occupancies is None:
            occupancies = np.zeros(n_atoms, float)
    else:
        occupancies = np.array(occupancy)
        if len(occupancies) != n_atoms:
            raise ValueError('len(occupancy) must be equal to number of atoms')

    beta = kwargs.get('beta')
    if beta is None:
        bfactors = atoms._getBetas()
        if bfactors is None:
            bfactors = np.zeros(n_atoms, float)
    else:
        bfactors = np.array(beta)
        if len(bfactors) != n_atoms:
            raise ValueError('len(beta) must be equal to number of atoms')

    atomnames = atoms.getNames()
    if atomnames is None:
        raise ValueError('atom names are not set')
    for i, an in enumerate(atomnames):
        if len(an) < 4:
            atomnames[i] = ' ' + an

    s_or_u = np.array(['a']).dtype.char

    altlocs = atoms._getAltlocs()
    if altlocs is None:
        altlocs = np.zeros(n_atoms, s_or_u + '1')

    resnames = atoms._getResnames()
    if resnames is None:
        resnames = ['UNK'] * n_atoms

    chainids = atoms._getChids()
    if chainids is None:
        chainids = np.zeros(n_atoms, s_or_u + '1')

    resnums = atoms._getResnums()
    if resnums is None:
        resnums = np.ones(n_atoms, int)

    icodes = atoms._getIcodes()
    if icodes is None:
        icodes = np.zeros(n_atoms, s_or_u + '1')

    hetero = ['ATOM'] * n_atoms
    heteroflags = atoms._getFlags('hetatm')
    if heteroflags is None:
        heteroflags = atoms._getFlags('hetero')
    if heteroflags is not None:
        hetero = np.array(hetero, s_or_u + '6')
        hetero[heteroflags] = 'HETATM'

    elements = atoms._getElements()
    if elements is None:
        elements = np.zeros(n_atoms, s_or_u + '1')
    else:
        elements = np.char.rjust(elements, 2)

    segments = atoms._getSegnames()
    if segments is None:
        segments = np.zeros(n_atoms, s_or_u + '6')

    # stream.write('REMARK {0}\n'.format(remark))

    multi = len(coordsets) > 1
    write = stream.write
    for m, coords in enumerate(coordsets):
        pdbline = PDBLINE_LT100K
        if multi:
            write('MODEL{0:9d}\n'.format(m + 1))
        for i, xyz in enumerate(coords):
            if i == 99999:
                pdbline = PDBLINE_GE100K
            write(pdbline % (hetero[i], bfactors[i],
                             atomnames[i], altlocs[i],
                             resnames[i], chainids[i], resnums[i],
                             icodes[i],
                             xyz[0], xyz[1], xyz[2],
                             occupancies[i], 0,
                             segments[i], elements[i]))
        if multi:
            write('ENDMDL\n')
            altlocs = np.zeros(n_atoms, s_or_u + '1')


def make_pdb_wholebb_noligand(poi, outdir, vdm):
    """prints pdbs of the top number of hotspots (number) to the output directory (outdir)."""
    if outdir[-1] != '/':
        outdir += '/'
    poi = poi.copy()

    sc = vdm.select('not element H and chain X and resnum 10 and sidechain')
    resnum = str(vdm).split()[-1].split('_')[3][:-1]
    chid = str(vdm).split()[-1].split('_')[3][-1]
    resnum_chid = [int(resnum), chid]
    resn = vdm.select('chain X and resnum 10 and name CA').getResnames()[0]

    bb_first_resnum = np.min(poi.select('backbone').getResnums())
    bb_last_resnum = np.max(poi.select('backbone').getResnums())

    bb1 = poi.select('not element H and resnum ' + str(bb_first_resnum) + 'to' + str(resnum_chid[0]))
    num_ind = len(bb1.getIndices())
    start = 1
    finish = start + num_ind
    bb1.setBetas(list(range(start, finish)))
    poi.select('resnum ' + str(resnum_chid[0])).setResnames(resn)

    sc.setResnums(str(resnum_chid[0]))
    sc.setChids(str(resnum_chid[1]))
    num_ind = len(sc.getIndices())
    start = finish
    finish = start + num_ind
    sc.setBetas(list(range(start, finish)))

    bb2 = poi.select('not element H and resnum ' + str(resnum_chid[0] + 1) + 'to' + str(bb_last_resnum))
    num_ind = len(bb2.getIndices())
    start = finish
    finish = start + num_ind
    bb2.setBetas(list(range(start, finish)))

    try:
        os.makedirs(outdir)
    except:
        pass
    with open(outdir + 'pose_with_bb_template.pdb', 'w') as outfile:
        writePDBStream(outfile, bb1)
        writePDBStream(outfile, sc)
        writePDBStream(outfile, bb2)


def make_pdb_wholebb_ligand(poi, outdir, vdm, ligand):
    """prints pdbs of the top number of hotspots (number) to the output directory (outdir)."""
    if outdir[-1] != '/':
        outdir += '/'
    poi = poi.copy()

    sc = vdm.select('not element H and chain X and resnum 10 and sidechain')
    resnum = str(vdm).split()[-1].split('_')[3][:-1]
    chid = str(vdm).split()[-1].split('_')[3][-1]
    resnum_chid = [int(resnum), chid]
    resn = vdm.select('chain X and resnum 10 and name CA').getResnames()[0]

    bb_first_resnum = np.min(poi.select('backbone').getResnums())
    bb_last_resnum = np.max(poi.select('backbone').getResnums())

    bb1 = poi.select('not element H and resnum ' + str(bb_first_resnum) + 'to' + str(resnum_chid[0]))
    num_ind = len(bb1.getIndices())
    start = 1
    finish = start + num_ind
    bb1.setBetas(list(range(start, finish)))
    poi.select('resnum ' + str(resnum_chid[0])).setResnames(resn)
    sc.setResnums(str(resnum_chid[0]))
    sc.setChids(str(resnum_chid[1]))
    num_ind = len(sc.getIndices())
    start = finish
    finish = start + num_ind
    sc.setBetas(list(range(start, finish)))

    bb2 = poi.select('not element H and resnum ' + str(resnum_chid[0] + 1) + 'to' + str(bb_last_resnum))
    num_ind = len(bb2.getIndices())
    start = finish
    finish = start + num_ind
    bb2.setBetas(list(range(start, finish)))

    num_ind = len(ligand.select('all').getIndices())
    start = finish
    finish = start + num_ind
    ligand.setBetas(list(range(start, finish)))

    try:
        os.makedirs(outdir)
    except:
        pass
    with open(outdir + 'pose_with_bb_template.pdb', 'w') as outfile:
        writePDBStream(outfile, bb1)
        writePDBStream(outfile, sc)
        writePDBStream(outfile, bb2)
        writePDBStream(outfile, ligand)



def make_protein_from_dee_pose(indir, outdir, path_to_bb_template):
    if indir[-1] != '/':
        indir = indir + '/'
    if outdir[-1] != '/':
        outdir = outdir + '/'
    bb_template = pr.parsePDB(path_to_bb_template)
    ligand = pr.parsePDB(indir + 'lig.pdb')
    num_fns = len([fn for fn in os.listdir(indir) if fn[-2:] == 'gz'])
    for i, f in enumerate([fn for fn in os.listdir(indir) if fn[-2:] == 'gz']):
        vdm = pr.parsePDB(indir + f)
        if i + 1 < num_fns:
            if i == 0:
                make_pdb_wholebb_noligand(bb_template, outdir, vdm)
            else:
                bb_template = pr.parsePDB(outdir + 'pose_with_bb_template.pdb')
                make_pdb_wholebb_noligand(bb_template, outdir, vdm)
        if i + 1 == num_fns:
            bb_template = pr.parsePDB(outdir + 'pose_with_bb_template.pdb')
            make_pdb_wholebb_ligand(bb_template, outdir, vdm, ligand)


def make_resfile(path_to_pdb, outdir):
    if outdir[-1] != '/':
        outdir = outdir + '/'
    pdb = pr.parsePDB(path_to_pdb)
    sel = pdb.select('name CA and not resname GLY')
    resnums = sel.getResnums()
    chains = sel.getChids()

    with open(outdir + 'resfile.txt', 'w') as resfile:
        resfile.write('ALLAAxc \n')
        resfile.write('USE_INPUT_SC \n')
        resfile.write('EX 1 LEVEL 1 \n')
        resfile.write('start \n')
        resfile.write('\n')
        for resnum, chid in zip(resnums, chains):
            resfile.write(str(resnum) + ' ' + chid + ' NATRO \n')
        resfile.write('1 X NATRO \n')




