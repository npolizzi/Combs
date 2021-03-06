from collections import defaultdict
from collections import deque
import pandas as pd


__all__ = ['parse_stride']


#dtypes of dataframe columns in :func:`parse_stride`
stride_dtype_dict = {'segid': 'category',
                     'chain': 'category',
                     'resnum': int,
                     'stride': 'category',
                     'phi': float,
                     'psi': float,
                     'sasa': float}


# dataframe columns in :func:`parse_stride`
stride_col_names = ['segid', 'chain',
                    'resnum', 'stride',
                    'phi', 'psi', 'sasa']


def _make_segid_dict(pdb):
    """
    Parameters
    ----------
    pdb : :class:`prody.Atomic`
    """
    ca = pdb.select('name CA')
    segnames = ca.getSegnames()
    resnums = ca.getResnums()
    chids = ca.getChids()
    seg_dict = defaultdict(deque)
    for rn, ch, segn in zip(resnums, chids, segnames):
        seg_dict[(rn, ch)].append(segn)
    return seg_dict


def _parse_stride_line(line, seg_dict):
    """Note that seg_dict gets altered via
    :func:`collections.deque.popleft()`

    Parameters
    ----------
    line
        line from a file open
    seg_dict: defaultdict
        output of :func:`_make_segid_dict`
    """
    CHAIN = line[9:11].strip()
    RESN = int(line[11:16])
    SS = line[24]
    SEGID = seg_dict[(RESN, CHAIN)].popleft()
    PHI = float(line[42:49])
    PSI = float(line[52:59])
    AREA = float(line[64:69])
    return (SEGID, CHAIN, RESN,
            SS, PHI, PSI, AREA)


def parse_stride(stride_file, pdb):
    """Used to generate :class:`~pandas.DataFrame`
    containing stride info of a PDB.

    Parameters
    ----------
    stride_file : str
        path to stride file
    pdb : :class:`~prody.Atomic`
        corresponding pdb object

    Returns
    -------
    :class:`~pandas.DataFrame`
        Dataframe containing columns from
        *stride_col_names*
    """
    seg_dict = _make_segid_dict(pdb)
    stride_data = []
    with open(stride_file, 'r') as stridefile:
        for line in stridefile:
            if not line.startswith('ASG '):
                continue
            line_data = _parse_stride_line(line, seg_dict)
            stride_data.append(line_data)
    stride_df = pd.DataFrame(stride_data, columns=stride_col_names)
    return stride_df.astype(dtype=stride_dtype_dict)


