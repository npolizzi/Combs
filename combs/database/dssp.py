from .stride import _make_segid_dict
import pandas as pd


__all__ = ['parse_dssp']


#dtypes of dataframe columns in :func:`parse_dssp`
dssp_dtype_dict = {'segid': 'category',
                     'chain': 'category',
                     'resnum': int,
                     'dssp': 'category',
                     'phi': float,
                     'psi': float,
                     'sasa': int}


# dataframe columns in :func:`parse_dssp`
dssp_col_names = ['segid', 'chain',
                    'resnum', 'dssp',
                    'phi', 'psi', 'sasa']


def _parse_dssp_line(line, seg_dict):
    """Note that seg_dict gets altered via
    :func:`collections.deque.popleft()`

    Parameters
    ----------
    line
        line from a file open
    seg_dict: defaultdict
        output of :func:`_make_segid_dict`
    """
    CHAIN = line[11]
    RESN = int(line[5:10])
    SS = line[16]
    if SS == ' ':
        SS = 'C'
    SEGID = seg_dict[(RESN, CHAIN)].popleft()
    PHI = float(line[103:109])
    PSI = float(line[109:115])
    AREA = int(line[34:38])
    return (SEGID, CHAIN, RESN,
            SS, PHI, PSI, AREA)


def parse_dssp(dssp_file, pdb):
    """Used to generate :class:`~pandas.DataFrame`
    containing dssp info of a PDB.

    Parameters
    ----------
    dssp_file : str
        path to dssp file
    pdb : :class:`~prody.Atomic`
        corresponding pdb object

    Returns
    -------
    :class:`~pandas.DataFrame`
        Dataframe containing columns from
        *dssp_col_names*
    """
    seg_dict = _make_segid_dict(pdb)
    dssp_data = []
    with open(dssp_file, 'r') as dsspfile:
        for line in dsspfile:
            if line.startswith('  #  RESIDUE'):
                break
        for line in dsspfile:
            if line[13] == '!':
                continue
            line_data = _parse_dssp_line(line, seg_dict)
            dssp_data.append(line_data)
    dssp_df = pd.DataFrame(dssp_data, columns=dssp_col_names)
    return dssp_df.astype(dtype=dssp_dtype_dict)



