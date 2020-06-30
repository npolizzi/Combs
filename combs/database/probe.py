""""""
import os
import pandas as pd
from collections import defaultdict


probe = '/Users/npolizzi/Applications/MolProbity/build/bin/phenix.probe'


#dtypes of dataframe columns in :func:`parse_probe`
probe_dtype_dict = {'interaction': 'category',
                    'segid1': 'category',
                    'chain1': 'category',
                    'resnum1': int,
                    'resname1': 'category',
                    'name1': 'category',
                    'atomtype1': 'category',
                    'segid2': 'category',
                    'chain2': 'category',
                    'resnum2': int,
                    'resname2': 'category',
                    'name2': 'category',
                    'atomtype2': 'category'}


# dataframe columns in :func:`parse_probe`
probe_col_names = list(probe_dtype_dict.keys())


def _make_segname(segname):
    return 'SEG' + '____'[:-len(segname)] + segname


def _make_cmd(segname1, pdb_file, segname2='',
             probe_sel_criteria='blt40 ogt99 not metal',
             outdir=None):
    """ """
    seg1 = _make_segname(segname1)
    seg2 = ''

    if segname2 != '':
        probe_action = '-B'
        seg2 = _make_segname(segname2)
    else:
        probe_action = '-SE'

    cmd = [probe,'-U -CON -DOCHO -MC -DE32',
           probe_action, '"', seg1, probe_sel_criteria,
           '"']

    if segname2 != '':
        seg2_cmd = ['"', seg2, probe_sel_criteria, '"']
        cmd.extend(seg2_cmd)

    cmd.append(pdb_file)

    if outdir is not None:
        if outdir[-1] != '/':
            outdir += '/'
        outfile = ''.join([outdir, pdb_file.split('/')[-1][:-4],
                          '_', segname1, segname2, '.probe'])
        cmd.extend(['>', outfile])

    return ' '.join(cmd)


def _parse_probe_line(line, segname1, segname2):
    """ """
    if segname2 == '':
        segname2 = segname1
        
    spl = line.split(':')[1:]
    INTERACTION = spl[1]
    CHAIN1 = spl[2][:2].strip()
    RESNUM1 = int(spl[2][2:6])
    RESNAME1 = spl[2][6:10].strip()
    NAME1 = spl[2][10:15].strip()
    ATOMTYPE1 = spl[12]
    CHAIN2 = spl[3][:2].strip()
    RESNUM2 = int(spl[3][2:6])
    RESNAME2 = spl[3][6:10].strip()
    NAME2 = spl[3][10:15].strip()
    ATOMTYPE2 = spl[13]
    if RESNAME1 == 'HOH':
        NAME1 = 'O'
    if RESNAME2 == 'HOH':
        NAME2 = 'O'
    return (INTERACTION, segname1, CHAIN1,
            RESNUM1, RESNAME1, NAME1, ATOMTYPE1,
            segname2, CHAIN2, RESNUM2, RESNAME2,
            NAME2, ATOMTYPE2)


def parse_probe(pdb_file, segname1, segname2='',
                probe_sel_criteria='blt40 ogt99 not metal',
                outdir=None):
    """ Creates a :class:`pandas.DataFrame` of Probe
    contacts between *segname1* and *segname2*.  If *segname2*
    is defined, Probe outputs the interfacial contacts
    (1->2 and 2->1) between the two segments only.  Otherwise
    Probe outputs all intrasegment contacts of segment *segname1*.
    Only one atom-atom contact per unique pair is reported, in
    order of hb (H-bond) then cc (close contact).  Contacts
    labeled so (small overlap), bo (bad overlap), wc (wide contact)
    are not reported.  Metal contacts are not reported accurately, so
    the default behavior is to exlude them. For metal coordination,
    see :func:`~.pdbheader.parse_metal_contacts`.

    Parameters
    ----------
    pdb_file : str
        path to pdb file, preferably a biological unit
    segname1 : str
    segname2 : str, optional
    probe_sel_criteria : str, optional
        filter Probe output by selection
    outdir : str, optional
        path to output directory for Probe txt file

    Returns
    -------
    pandas.DataFrame
        Probe contact info for unique atom-atom pairs
    """
    cmd = _make_cmd(segname1, pdb_file, segname2,
                   probe_sel_criteria, outdir)
    probe_dict = defaultdict(list)
    probe_data = []
    with os.popen(cmd) as probefile:
        for line in probefile:
            line_data = _parse_probe_line(line, segname1, segname2)
            probe_dict[line_data[1:]].append(line_data[0])
    for info, interactions in probe_dict.items():
        if 'bo' in interactions:
            continue
        elif 'so' in interactions:
            continue
        elif 'hb' in interactions:
            interaction = 'hb'
        elif 'cc' in interactions:
            interaction = 'cc'
        elif 'wc' in interactions:
            continue
        data = [interaction]
        data.extend(info)
        probe_data.append(data)
    probe_df = pd.DataFrame(probe_data, columns=probe_col_names)
    return probe_df.astype(dtype=probe_dtype_dict)

