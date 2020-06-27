import prody as pr

__all__ = ['calc_vandarotamer', 'calc_vandarotamer_fmt']


dist = {'GLY': ['HA3', 'A1'],
        'ALA': ['CB', 'A1'],
        'VAL': ['CB', 'A1'],
        'LEU': ['CB', 'A1'],
        'ILE': ['CB', 'A1'],
        'PHE': ['CB', 'A1'],
        'MET': ['CB', 'A1'],
        'TRP': ['CB', 'A1'],
        'TYR': ['CB', 'A1'],
        'PRO': ['CB', 'A1'],
        'CYS': ['CB', 'A1'],
        'HIS': ['CB', 'A1'],
        'SER': ['CB', 'A1'],
        'THR': ['CB', 'A1'],
        'ASN': ['CB', 'A1'],
        'GLN': ['CB', 'A1'],
        'LYS': ['CB', 'A1'],
        'ARG': ['CB', 'A1'],
        'GLU': ['CB', 'A1'],
        'ASP': ['CB', 'A1'],
        'MSE': ['CB', 'A1'],
        }

ang1 = {'GLY': ['CA', 'HA3', 'A1'],
        'ALA': ['CA', 'CB', 'A1'],
        'VAL': ['CA', 'CB', 'A1'],
        'LEU': ['CA', 'CB', 'A1'],
        'ILE': ['CA', 'CB', 'A1'],
        'PHE': ['CA', 'CB', 'A1'],
        'MET': ['CA', 'CB', 'A1'],
        'TRP': ['CA', 'CB', 'A1'],
        'TYR': ['CA', 'CB', 'A1'],
        'PRO': ['CA', 'CB', 'A1'],
        'CYS': ['CA', 'CB', 'A1'],
        'HIS': ['CA', 'CB', 'A1'],
        'SER': ['CA', 'CB', 'A1'],
        'THR': ['CA', 'CB', 'A1'],
        'ASN': ['CA', 'CB', 'A1'],
        'GLN': ['CA', 'CB', 'A1'],
        'LYS': ['CA', 'CB', 'A1'],
        'ARG': ['CA', 'CB', 'A1'],
        'GLU': ['CA', 'CB', 'A1'],
        'ASP': ['CA', 'CB', 'A1'],
        'MSE': ['CA', 'CB', 'A1'],
        }

dih1 = {'GLY': ['N', 'CA', 'HA3', 'A1'],
        'ALA': ['N', 'CA', 'CB', 'A1'],
        'VAL': ['N', 'CA', 'CB', 'A1'],
        'LEU': ['N', 'CA', 'CB', 'A1'],
        'ILE': ['N', 'CA', 'CB', 'A1'],
        'PHE': ['N', 'CA', 'CB', 'A1'],
        'MET': ['N', 'CA', 'CB', 'A1'],
        'TRP': ['N', 'CA', 'CB', 'A1'],
        'TYR': ['N', 'CA', 'CB', 'A1'],
        'PRO': ['N', 'CA', 'CB', 'A1'],
        'CYS': ['N', 'CA', 'CB', 'A1'],
        'HIS': ['N', 'CA', 'CB', 'A1'],
        'SER': ['N', 'CA', 'CB', 'A1'],
        'THR': ['N', 'CA', 'CB', 'A1'],
        'ASN': ['N', 'CA', 'CB', 'A1'],
        'GLN': ['N', 'CA', 'CB', 'A1'],
        'LYS': ['N', 'CA', 'CB', 'A1'],
        'ARG': ['N', 'CA', 'CB', 'A1'],
        'GLU': ['N', 'CA', 'CB', 'A1'],
        'ASP': ['N', 'CA', 'CB', 'A1'],
        'MSE': ['N', 'CA', 'CB', 'A1'],
        }


ang2 = {'GLY': ['HA3', 'A1', 'A2'],
        'ALA': [ 'CB', 'A1', 'A2'],
        'VAL': [ 'CB', 'A1', 'A2'],
        'LEU': [ 'CB', 'A1', 'A2'],
        'ILE': [ 'CB', 'A1', 'A2'],
        'PHE': [ 'CB', 'A1', 'A2'],
        'MET': [ 'CB', 'A1', 'A2'],
        'TRP': [ 'CB', 'A1', 'A2'],
        'TYR': [ 'CB', 'A1', 'A2'],
        'PRO': [ 'CB', 'A1', 'A2'],
        'CYS': [ 'CB', 'A1', 'A2'],
        'HIS': [ 'CB', 'A1', 'A2'],
        'SER': [ 'CB', 'A1', 'A2'],
        'THR': [ 'CB', 'A1', 'A2'],
        'ASN': [ 'CB', 'A1', 'A2'],
        'GLN': [ 'CB', 'A1', 'A2'],
        'LYS': [ 'CB', 'A1', 'A2'],
        'ARG': [ 'CB', 'A1', 'A2'],
        'GLU': [ 'CB', 'A1', 'A2'],
        'ASP': [ 'CB', 'A1', 'A2'],
        'MSE': [ 'CB', 'A1', 'A2'],
        }

dih2 = {'GLY': ['CA', 'HA3', 'A1', 'A2'],
        'ALA': [ 'CA', 'CB', 'A1', 'A2'],
        'VAL': [ 'CA', 'CB', 'A1', 'A2'],
        'LEU': [ 'CA', 'CB', 'A1', 'A2'],
        'ILE': [ 'CA', 'CB', 'A1', 'A2'],
        'PHE': [ 'CA', 'CB', 'A1', 'A2'],
        'MET': [ 'CA', 'CB', 'A1', 'A2'],
        'TRP': [ 'CA', 'CB', 'A1', 'A2'],
        'TYR': [ 'CA', 'CB', 'A1', 'A2'],
        'PRO': [ 'CA', 'CB', 'A1', 'A2'],
        'CYS': [ 'CA', 'CB', 'A1', 'A2'],
        'HIS': [ 'CA', 'CB', 'A1', 'A2'],
        'SER': [ 'CA', 'CB', 'A1', 'A2'],
        'THR': [ 'CA', 'CB', 'A1', 'A2'],
        'ASN': [ 'CA', 'CB', 'A1', 'A2'],
        'GLN': [ 'CA', 'CB', 'A1', 'A2'],
        'LYS': [ 'CA', 'CB', 'A1', 'A2'],
        'ARG': [ 'CA', 'CB', 'A1', 'A2'],
        'GLU': [ 'CA', 'CB', 'A1', 'A2'],
        'ASP': [ 'CA', 'CB', 'A1', 'A2'],
        'MSE': [ 'CA', 'CB', 'A1', 'A2'],
        }


dih3 = {'GLY': ['HA3', 'A1', 'A2', 'A3'],
        'ALA': ['CB', 'A1', 'A2', 'A3'],
        'VAL': ['CB', 'A1', 'A2', 'A3'],
        'LEU': ['CB', 'A1', 'A2', 'A3'],
        'ILE': ['CB', 'A1', 'A2', 'A3'],
        'PHE': ['CB', 'A1', 'A2', 'A3'],
        'MET': ['CB', 'A1', 'A2', 'A3'],
        'TRP': ['CB', 'A1', 'A2', 'A3'],
        'TYR': ['CB', 'A1', 'A2', 'A3'],
        'PRO': ['CB', 'A1', 'A2', 'A3'],
        'CYS': ['CB', 'A1', 'A2', 'A3'],
        'HIS': ['CB', 'A1', 'A2', 'A3'],
        'SER': ['CB', 'A1', 'A2', 'A3'],
        'THR': ['CB', 'A1', 'A2', 'A3'],
        'ASN': ['CB', 'A1', 'A2', 'A3'],
        'GLN': ['CB', 'A1', 'A2', 'A3'],
        'LYS': ['CB', 'A1', 'A2', 'A3'],
        'ARG': ['CB', 'A1', 'A2', 'A3'],
        'GLU': ['CB', 'A1', 'A2', 'A3'],
        'ASP': ['CB', 'A1', 'A2', 'A3'],
        'MSE': ['CB', 'A1', 'A2', 'A3'],
        }


def get_dist(vdm, ifg, comb):
    return pr.calcDistance(vdm.ires_sele.select('name ' + dist[vdm.resname][0]),
                           ifg.sele.select('name ' + comb.vandarotamer_dict['A1']))[0]

def get_ang1(vdm, ifg, comb):
    return pr.calcAngle(vdm.ires_sele.select('name ' + ang1[vdm.resname][0]),
                        vdm.ires_sele.select('name ' + ang1[vdm.resname][1]),
                        ifg.sele.select('name ' + comb.vandarotamer_dict['A1']))[0]

def get_dih1(vdm, ifg, comb):
    return pr.calcDihedral(vdm.ires_sele.select('name ' + dih1[vdm.resname][0]),
                           vdm.ires_sele.select('name ' + dih1[vdm.resname][1]),
                           vdm.ires_sele.select('name ' + dih1[vdm.resname][2]),
                           ifg.sele.select('name ' + comb.vandarotamer_dict['A1']))[0]

def get_ang2(vdm, ifg, comb):
    return pr.calcAngle(vdm.ires_sele.select('name ' + ang2[vdm.resname][0]),
                        ifg.sele.select('name ' + comb.vandarotamer_dict['A1']),
                        ifg.sele.select('name ' + comb.vandarotamer_dict['A2']))[0]

def get_dih2(vdm, ifg, comb):
    return pr.calcDihedral(vdm.ires_sele.select('name ' + dih2[vdm.resname][0]),
                           vdm.ires_sele.select('name ' + dih2[vdm.resname][1]),
                           ifg.sele.select('name ' + comb.vandarotamer_dict['A1']),
                           ifg.sele.select('name ' + comb.vandarotamer_dict['A2']))[0]

def get_dih3(vdm, ifg, comb):
    return pr.calcDihedral(vdm.ires_sele.select('name ' + dih3[vdm.resname][0]),
                           ifg.sele.select('name ' + comb.vandarotamer_dict['A1']),
                           ifg.sele.select('name ' + comb.vandarotamer_dict['A2']),
                           ifg.sele.select('name ' + comb.vandarotamer_dict['A3']))[0]


def calc_vandarotamer(vdm, ifg, comb):
    ''''''
    return (get_dist(vdm, ifg, comb), get_ang1(vdm, ifg, comb), get_dih1(vdm, ifg, comb),
            get_ang2(vdm, ifg, comb), get_dih2(vdm, ifg, comb), get_dih3(vdm, ifg, comb))

def calc_vandarotamer_fmt(vdm, ifg, comb):
    ''''''
    return ' '.join(['{0:.2f}'.format(i) for i in [get_dist(vdm, ifg, comb), get_ang1(vdm, ifg, comb),
                                          get_dih1(vdm, ifg, comb), get_ang2(vdm, ifg, comb),
                                          get_dih2(vdm, ifg, comb), get_dih3(vdm, ifg, comb)]])

