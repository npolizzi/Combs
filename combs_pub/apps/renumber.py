import prody as pr
import numpy as np

__all__ = ['renumber_chids_resnums']


def renumber_chids_resnums(obj, new_chid):
    """Assumes the resindices of a selection are monotonically increasing by 1"""
    try:
        sele = obj.ifg_frag
    except:
        sele = obj.sele

    sele_copy = sele.copy()
    sele_copy.setResnums(sele.getResindices() - np.min(obj.resindex) + 10)
    sele_copy.setChids([new_chid]*len(sele))

    return sele_copy