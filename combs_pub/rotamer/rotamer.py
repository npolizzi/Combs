import prody as pr

__all__ = ['get_chi', 'calc_rotamer']

#### ADAPTED FROM LENNA PETERSON https://gist.github.com/lennax/0f5f65ddbfa278713f58 #####


def get_chi(residue_sele, atoms_list):
    '''Take parsedvdm and a list of 4 atoms and calculate dihedral in prody
    Helper function for rotamer()'''

    prody_list = [residue_sele.select('name %s' % name) for name in atoms_list]
    return pr.calcDihedral(prody_list[0], prody_list[1], prody_list[2], prody_list[3])



chi_dict = dict(
                chi1=dict(
                    ARG=['N', 'CA', 'CB', 'CG'],
                    ASN=['N', 'CA', 'CB', 'CG'],
                    ASP=['N', 'CA', 'CB', 'CG'],
                    CYS=['N', 'CA', 'CB', 'SG'],
                    GLN=['N', 'CA', 'CB', 'CG'],
                    GLU=['N', 'CA', 'CB', 'CG'],
                    HIS=['N', 'CA', 'CB', 'CG'],
                    ILE=['N', 'CA', 'CB', 'CG1'],
                    LEU=['N', 'CA', 'CB', 'CG'],
                    LYS=['N', 'CA', 'CB', 'CG'],
                    MET=['N', 'CA', 'CB', 'CG'],
                    PHE=['N', 'CA', 'CB', 'CG'],
                    PRO=['N', 'CA', 'CB', 'CG'],
                    SER=['N', 'CA', 'CB', 'OG'],
                    THR=['N', 'CA', 'CB', 'OG1'],
                    TRP=['N', 'CA', 'CB', 'CG'],
                    TYR=['N', 'CA', 'CB', 'CG'],
                    VAL=['N', 'CA', 'CB', 'CG1'],
                        ),
                chi2=dict(
                    ARG=['CA', 'CB', 'CG', 'CD'],
                    ASN=['CA', 'CB', 'CG', 'OD1'],
                    ASP=['CA', 'CB', 'CG', 'OD1'],
                    GLN=['CA', 'CB', 'CG', 'CD'],
                    GLU=['CA', 'CB', 'CG', 'CD'],
                    HIS=['CA', 'CB', 'CG', 'ND1'],
                    ILE=['CA', 'CB', 'CG1', 'CD1'],
                    LEU=['CA', 'CB', 'CG', 'CD1'],
                    LYS=['CA', 'CB', 'CG', 'CD'],
                    MET=['CA', 'CB', 'CG', 'SD'],
                    PHE=['CA', 'CB', 'CG', 'CD1'],
                    PRO=['CA', 'CB', 'CG', 'CD'],
                    TRP=['CA', 'CB', 'CG', 'CD1'],
                    TYR=['CA', 'CB', 'CG', 'CD1'],
                        ),
                chi3=dict(
                    ARG=['CB', 'CG', 'CD', 'NE'],
                    GLN=['CB', 'CG', 'CD', 'OE1'],
                    GLU=['CB', 'CG', 'CD', 'OE1'],
                    LYS=['CB', 'CG', 'CD', 'CE'],
                    MET=['CB', 'CG', 'SD', 'CE'],
                        ),
                chi4=dict(
                    ARG=['CG', 'CD', 'NE', 'CZ'],
                    LYS=['CG', 'CD', 'CE', 'NZ'],
                        ),
                chi5=dict(
                    ARG=['CD', 'NE', 'CZ', 'NH1'],
                        )
                )

alt_chi_dict = dict(
                    chi1=dict(VAL=['N', 'CA', 'CB', 'CG2']),
                    chi2=dict(
                                ASP=['CA', 'CB', 'CG', 'OD2'],
                                LEU=['CA', 'CB', 'CG', 'CD2'],
                                PHE=['CA', 'CB', 'CG', 'CD2'],
                                TYR=['CA', 'CB', 'CG', 'CD2'],
                                ),
                    )


def calc_rotamer(prody_pdb, resnum, chid, segment):
    '''Calculates dihedrals for all the chi angles in the vdm residue (vdmires).
    Returns nested list of all the chi angles for the vdm ires. Empty list for ALA, GLY, and vdms that 
    fail the 'try' statement. If successful, ex of nested list is: 
    [[chi1, altchi1], [chi2], [chi3], [chi4] ] '''

    resi_sele = prody_pdb.select('segment %s and chain %s and resnum `%s` and (not element H) and (not name C) and (not name O)'
                                 % (segment, chid, resnum))
    restype = resi_sele.getResnames()[0]
    if restype == 'ALA' or restype == 'GLY':
        return []

    chi_list = []
    # format is nested list, ex:  [[chi1, altchi1], [chi2], [chi3], [chi4] ]
    for chi in ['chi1', 'chi2', 'chi3', 'chi4']:
        try:
            ls = []
            dihedral = get_chi(resi_sele, chi_dict[chi][restype])
            ls.append(dihedral)
            try:
                dihedral = get_chi(resi_sele, alt_chi_dict[chi][restype])
                ls.append(dihedral)
            except:
                pass  # if there are no alt chis
            chi_list.append(ls)
        except:
           pass  # if there are no chi3's, chi4's, etc
    return chi_list



