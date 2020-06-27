__all__ = ['Vandermer']

import prody as pr
import numpy as np
import gzip
import freesasa
from ..rotamer.rotamer import calc_rotamer
from ..apps.renumber import renumber_chids_resnums
from ..apps.constants import one_letter_code
from ..apps import vandarotamer as vdr


class Vandermer:
    def __init__(self, ifg, parsed_pdb):
        try:
            self.resnum = ifg.contact_resnums.pop()  # i residue
            self.chid = ifg.contact_chids.pop()
            self.segment = ifg.contact_segments.pop()
            self.resindex = ifg.contact_resindices.pop()  # i residue
            self.contact_info = ' '.join('(' + c[0] + ' ' + c[1] + ' ' + '{0:.2f}'.format(c[2]) + ')'
                                         for c in ifg.contact_info_protein.pop())
        except IndexError:
            print('No more vdMs for this iFG.')
        self.sele = self.get_vdm_sele(parsed_pdb)
        self.ires_sele = self.get_vdm_ires_sele(parsed_pdb)
        self.resname = np.unique(self.sele.select('resnum `' + str(self.resnum) + '`').getResnames())[0]
        self.resindices, self._ind = np.unique(self.sele.getResindices(), return_index=True)
        self.resnums = self.sele.getResnums()[self._ind]
        self.length = len(self.resindices)
        self.atom_names = ' '.join((ifg.contact_atoms_protein & self.ires_sele).getNames())
        self.sequence = self.get_sequence()
        self.rotamer = None
        self.ifg_contacts = ' '.join(ifg.contact_dict[(self.resnum, self.chid, self.segment)])
        self.ifg_contact_pairs = ' '.join(['(' + ' '.join(p) + ')' for p in ifg.contact_pair_dict[(self.resnum, self.chid, self.segment)]])
        self.rel_resnums = ''.join(str(ri - self.resindex) for ri in self.resindices)
        self.sec_struct_phi_psi_frag = None
        self.sec_struct_dssp = None
        self.sec_struct_phi_psi = None
        self.min_hull_dist_cb_ca = None
        self.residue_sasa = None
        self.dssp_sasa = None
        self.sasa_3A_probe = None
        self.sasa_4A_probe = None
        self.sasa_5A_probe = None
        self.contact_atoms_sele = ifg.contact_atoms_protein & self.sele
        self.contact_number = None
        self.contact_atom_names = None
        self.contact_resnums = None
        self.contact_resnames = None
        self.per_res_contact_number = None
        self.probe_hbonds = []
        self.hbond_atom_names = []
        self.hbond_ifg_atom_names = []
        self.hbond_vdm_atom_names = []
        self.hbond_resnames = []
        self.hbond_rel_resnums = []  # relative residue numbering, i.e. i-4, i+3, etc.
        self.hbond_resnums = []
        self.hbond_number = None
        self.hbond_angle = []
        self.hbond_dist_acc_hyd = []
        self.hbond_dist_heavy = []
        self.ca_hbond_atom_names = []
        self.ca_hbond_ifg_atom_names = []
        self.ca_hbond_vdm_atom_names = []
        self.ca_hbond_resnames = []
        self.ca_hbond_resnums = []
        self.ca_hbond_rel_resnums = []
        self.ca_hbond_angle = []
        self.ca_hbond_dist_acc_hyd = []
        self.ca_hbond_dist_heavy = []
        self.ca_hbond_number = None
        self.bb_cb_atom_ind = self.get_bb_cb_atom_indices()
        self.vandarotamer = None

    def get_bb_cb_atom_indices(self):
        sele = self.sele.select('protein and (backbone or name CB) and resindex `' + str(self.resindex)
                                                          + '` and not element H D')
        if sele is not None:
            return sele.getIndices()
        else:
            return None

    def get_sequence(self):
        seq_list = []
        for rn in self.sele.getResnames()[self._ind]:
            if rn in one_letter_code.keys():
                seq_list.append(one_letter_code[rn])
            else:
                seq_list.append('X')
        return ''.join(seq_list)

    def get_min_hull_dist_cb_ca(self, parsed_pdb):
        sel = parsed_pdb.prody_pdb.select('segment ' + self.segment + ' and chain ' + self.chid
                                          + ' and resnum ' + str(self.resnum))
        if set(sel.getResnames()).pop() == 'GLY':
            pnt = sel.select('name CA').getCoords().flatten()
        elif len(sel) == 1:
            pnt = sel.getCoords().flatten()
        else:
            pnt = sel.select('name CB').getCoords().flatten()
        self.min_hull_dist_cb_ca = '{0:.2f}'.format(parsed_pdb.alphahull.get_pnt_distance(pnt))

    def get_contact(self):
        _resnums, _ind = np.unique(self.contact_atoms_sele.getResnums(), return_index=True)
        _sort_ind = np.argsort(_resnums)
        self.contact_resnums = ''.join(str(rn) for rn in (_resnums[_sort_ind] - self.resnum))
        _contact_number = [len(self.contact_atoms_sele.select('resnum `' + str(rn) + '`'))
                           for rn in _resnums[_sort_ind]]
        self.contact_number = np.sum(_contact_number)
        self.per_res_contact_number = ' '.join(str(num) for num in _contact_number)
        self.contact_resnames = ' '.join(self.contact_atoms_sele.getResnames()[_ind])
        _contact_atom_names = ['(' + ' '.join(self.contact_atoms_sele.select('resnum `' + str(rn) + '`').getNames())
                               + ')' for rn in _resnums[_sort_ind]]
        self.contact_atom_names = ' '.join(_contact_atom_names)

    def get_vdm_ires_sele(self, parsed_pdb):
        return parsed_pdb.prody_pdb.select('protein and segment ' + self.segment + ' and chain ' + self.chid
                                           + ' and resnum `' + str(self.resnum) + '`')

    def get_vdm_sele(self, parsed_pdb):
        """Selects vdm fragment based on connectivity of atoms (as opposed to +- residue numbering, which can be
        misleading for irregularly numbered pdbs).  'Bonded 15' means up to 5 residues away from the i residue."""
        resinds = np.unique(parsed_pdb.prody_pdb.select('backbone and bonded 15 to (backbone and resnum `'
                                                        + str(self.resnum) + '` and chain ' + self.chid
                                                        + ' and segment ' + self.segment
                                                        + ')').getResindices())
        return parsed_pdb.prody_pdb.select('protein and resindex ' + ' '.join(str(ri) for ri in resinds))

    def get_sec_struct_dssp(self, parsed_pdb):
        bb_resnums = self.sele.getResnums()[self._ind]
        bb_chids = self.sele.getChids()[self._ind]
        bb_dssp = []
        for rn, ch in zip(bb_resnums, bb_chids):
            try:
                bb_dssp.append(parsed_pdb.dssp_ss[(rn, ch)])
            except:
                bb_dssp.append('-')
        self.sec_struct_dssp = ''.join(bb_dssp)

    def get_sec_struct_phipsi(self, parsed_pdb):
        phipsi = []
        for resn in self.sele.getResnums()[self._ind]:
            try:
                phi = pr.calcDihedral(parsed_pdb.prody_pdb.select(
                    'name C and bonded 1 to (resnum ' + str(resn)
                    + ' segment ' + self.segment + ' chain ' + self.chid + ' name N)'),
                    parsed_pdb.prody_pdb.select(
                        'resnum ' + str(resn)
                        + ' segment ' + self.segment + ' chain ' + self.chid + ' name N'),
                    parsed_pdb.prody_pdb.select(
                        'resnum ' + str(resn)
                        + ' segment ' + self.segment + ' chain ' + self.chid + ' name CA'),
                    parsed_pdb.prody_pdb.select(
                        'resnum ' + str(resn)
                        + ' segment ' + self.segment + ' chain ' + self.chid + ' name C'))[0]
            except:
                phi = None
                # traceback.print_exc()
            try:
                psi = pr.calcDihedral(
                    parsed_pdb.prody_pdb.select(
                        'resnum ' + str(resn)
                        + ' segment ' + self.segment + ' chain ' + self.chid + ' name N'),
                    parsed_pdb.prody_pdb.select(
                        'resnum ' + str(resn)
                        + ' segment ' + self.segment + ' chain ' + self.chid + ' name CA'),
                    parsed_pdb.prody_pdb.select(
                        'resnum ' + str(resn)
                        + ' segment ' + self.segment + ' chain ' + self.chid + ' name C'),
                    parsed_pdb.prody_pdb.select(
                        'name N and bonded 1 to (resnum ' + str(resn)
                        + ' segment ' + self.segment + ' chain ' + self.chid + ' name C)'))[0]
            except:
                psi = None

            if phi is not None and psi is not None:
                phipsi.append('(' + '{0:.2f}'.format(phi) + ' ' + '{0:.2f}'.format(psi) + ')')
                if resn == self.resnum:
                    self.sec_struct_phi_psi = '{0:.2f}'.format(phi) + ' ' + '{0:.2f}'.format(psi)
            elif phi is None and psi is not None:
                phipsi.append('(' + 'None' + ' ' + '{0:.2f}'.format(psi) + ')')
                if resn == self.resnum:
                    self.sec_struct_phi_psi = 'None ' + '{0:.2f}'.format(psi)
            elif phi is not None and psi is None:
                phipsi.append('(' + '{0:.2f}'.format(phi) + ' ' + 'None' + ')')
                if resn == self.resnum:
                    self.sec_struct_phi_psi = '{0:.2f}'.format(phi) + ' None'
            else:
                phipsi.append('(None None)')
        self.sec_struct_phi_psi_frag = ' '.join(phipsi)

    def get_sec_struct(self, parsed_pdb):
        self.get_sec_struct_dssp(parsed_pdb)
        self.get_sec_struct_phipsi(parsed_pdb)

    @staticmethod
    def remove_from_list(_list, numbers):
        for elem in numbers:
            _list[:] = [x for x in _list if x != elem]
        return _list

    def get_probe_hbonds(self, ifg):
        self.probe_hbonds = ' : '.join([' '.join(str(k) for k in h) for h in ifg.probe_hbonds
                                        if all([h[2] == self.segment, h[3] == self.chid, h[4] == self.resnum])])

    def get_hbond(self, ifg):
        _hbond_atom_names = []
        _hbond_angle = []
        _hbond_resnames = []
        _hbond_rel_resnums = []
        _hbond_resnums = []
        _hbond_dist_acc_hyd = []
        _hbond_dist_heavy = []
        _hbond_ifg_atom_names =[]
        _hbond_vdm_atom_names =[]
        for i, hbond in enumerate(ifg.hbond_resnums):
            for resnum, resind in zip(self.resnums, self.resindices):
                if resnum in self.remove_from_list(hbond.copy(), ifg.resnum):
                    an = ifg.hbond_atom_names[i]
                    _hbond_atom_names.append('(' + ' '.join(an) + ')')
                    _hbond_ifg_atom_names.append('(' + ' '.join([an[j] for j, x in enumerate(hbond)
                                                                 if x in ifg.resnum]) + ')')
                    _hbond_vdm_atom_names.append('(' + ' '.join([an[j] for j, x in enumerate(hbond)
                                                                 if x not in ifg.resnum]) + ')')
                    _hbond_angle.append('{0:.2f}'.format(ifg.hbond_angle[i]))
                    _hbond_resnames.append('(' + ' '.join(ifg.hbond_resnames[i]) + ')')
                    _hbond_rel_resnums.append(resind-self.resindex)
                    _hbond_resnums.append('(' + ' '.join(str(rn) for rn in ifg.hbond_resnums[i]) + ')')
                    _hbond_dist_acc_hyd.append('{0:.2f}'.format(ifg.hbond_dist_acc_hyd[i]))
                    _hbond_dist_heavy.append('{0:.2f}'.format(ifg.hbond_dist_heavy[i]))
        _sort_ind = np.argsort(_hbond_rel_resnums).tolist()
        _hbond_atom_names = [_hbond_atom_names[i] for i in _sort_ind]
        _hbond_ifg_atom_names = [_hbond_ifg_atom_names[i] for i in _sort_ind]
        _hbond_vdm_atom_names = [_hbond_vdm_atom_names[i] for i in _sort_ind]
        _hbond_angle = [_hbond_angle[i] for i in _sort_ind]
        _hbond_resnames = [_hbond_resnames[i] for i in _sort_ind]
        _hbond_resnums = [_hbond_resnums[i] for i in _sort_ind]
        _hbond_rel_resnums = [_hbond_rel_resnums[i] for i in _sort_ind]
        _hbond_dist_acc_hyd = [_hbond_dist_acc_hyd[i] for i in _sort_ind]
        _hbond_dist_heavy = [_hbond_dist_heavy[i] for i in _sort_ind]
        self.hbond_atom_names = ' '.join(_hbond_atom_names)
        self.hbond_ifg_atom_names = ' '.join(_hbond_ifg_atom_names)
        self.hbond_vdm_atom_names = ' '.join(_hbond_vdm_atom_names)
        self.hbond_angle = ' '.join(_hbond_angle)
        self.hbond_resnames = ' '.join(_hbond_resnames)
        self.hbond_resnums = ' '.join(_hbond_resnums)
        self.hbond_rel_resnums = ''.join(str(rn) for rn in _hbond_rel_resnums)
        self.hbond_dist_acc_hyd = ' '.join(_hbond_dist_acc_hyd)
        self.hbond_dist_heavy = ' '.join(_hbond_dist_heavy)
        self.hbond_number = len(_hbond_resnums)

    def get_ca_hbond(self, ifg):
        _ca_hbond_atom_names = []
        _ca_hbond_ifg_atom_names = []
        _ca_hbond_vdm_atom_names = []
        _ca_hbond_angle = []
        _ca_hbond_resnames = []
        _ca_hbond_rel_resnums = []
        _ca_hbond_resnums = []
        _ca_hbond_dist_acc_hyd = []
        _ca_hbond_dist_heavy = []
        for i, hbond in enumerate(ifg.ca_hbond_resnums):
            for resnum, resind in zip(self.resnums, self.resindices):
                if resnum in self.remove_from_list(hbond.copy(), ifg.resnum):
                    an = ifg.ca_hbond_atom_names[i]
                    _ca_hbond_atom_names.append('(' + ' '.join(an) + ')')
                    _ca_hbond_ifg_atom_names.append('(' + ' '.join([an[j] for j, x in enumerate(hbond)
                                                                 if x in ifg.resnum]) + ')')
                    _ca_hbond_vdm_atom_names.append('(' + ' '.join([an[j] for j, x in enumerate(hbond)
                                                                 if x not in ifg.resnum]) + ')')
                    _ca_hbond_angle.append('{0:.2f}'.format(ifg.ca_hbond_angle[i]))
                    _ca_hbond_resnames.append('(' + ' '.join(ifg.ca_hbond_resnames[i]) + ')')
                    _ca_hbond_resnums.append('(' + ' '.join(str(rn) for rn in ifg.ca_hbond_resnums[i]) + ')')
                    _ca_hbond_rel_resnums.append(resind-self.resindex)
                    _ca_hbond_dist_acc_hyd.append('{0:.2f}'.format(ifg.ca_hbond_dist_acc_hyd[i]))
                    _ca_hbond_dist_heavy.append('{0:.2f}'.format(ifg.ca_hbond_dist_heavy[i]))
        _sort_ind = np.argsort(_ca_hbond_rel_resnums).tolist()
        _ca_hbond_atom_names = [_ca_hbond_atom_names[i] for i in _sort_ind]
        _ca_hbond_ifg_atom_names = [_ca_hbond_ifg_atom_names[i] for i in _sort_ind]
        _ca_hbond_vdm_atom_names = [_ca_hbond_vdm_atom_names[i] for i in _sort_ind]
        _ca_hbond_angle = [_ca_hbond_angle[i] for i in _sort_ind]
        _ca_hbond_resnames = [_ca_hbond_resnames[i] for i in _sort_ind]
        _ca_hbond_resnums = [_ca_hbond_resnums[i] for i in _sort_ind]
        _ca_hbond_rel_resnums = [_ca_hbond_rel_resnums[i] for i in _sort_ind]
        _ca_hbond_dist_acc_hyd = [_ca_hbond_dist_acc_hyd[i] for i in _sort_ind]
        _ca_hbond_dist_heavy = [_ca_hbond_dist_heavy[i] for i in _sort_ind]
        self.ca_hbond_atom_names = ' '.join(_ca_hbond_atom_names)
        self.ca_hbond_ifg_atom_names = ' '.join(_ca_hbond_ifg_atom_names)
        self.ca_hbond_vdm_atom_names = ' '.join(_ca_hbond_vdm_atom_names)
        self.ca_hbond_angle = ' '.join(_ca_hbond_angle)
        self.ca_hbond_resnames = ' '.join(_ca_hbond_resnames)
        self.ca_hbond_resnums = ' '.join(_ca_hbond_resnums)
        self.ca_hbond_rel_resnums = ''.join(str(rn) for rn in _ca_hbond_rel_resnums)
        self.ca_hbond_dist_acc_hyd = ' '.join(_ca_hbond_dist_acc_hyd)
        self.ca_hbond_dist_heavy = ' '.join(_ca_hbond_dist_heavy)
        self.ca_hbond_number = len(_ca_hbond_resnums)

    def get_rotamer(self, parsed_pdb):
        _rotamer = calc_rotamer(parsed_pdb.prody_pdb, self.resnum, self.chid, self.segment)
        newrot = []
        for item in _rotamer:
            newitem = ['{0:.2f}'.format(float(chi)) for chi in item]
            newrot.append('(' + ' '.join(newitem) + ')')
        self.rotamer = ' '.join(newrot)

    def calc_large_probe_sasa(self, parsed_pdb, fs_result):
        if self.bb_cb_atom_ind.any():
            return '{0:.2f}'.format(sum(fs_result.atomArea(i) for i in np.where(np.in1d(parsed_pdb.prody_pdb_bb_cb_atom_ind,
                                                                   self.bb_cb_atom_ind))[0]))

    def get_dssp_sasa(self, parsed_pdb):
        self.dssp_sasa = parsed_pdb.dssp_sasa[(self.resnum, self.chid)]

    def calc_sasa(self, parsed_pdb):
        """Calculates the per atom solvent accessible surface area of the iFG and the sasa of the residue containing
        the iFG.  Needs FreeSASA module to be imported.  Takes as argument an instance of ParsedPDB class, which
        contains the iFG.  Right now this function isn't optimized, in the sense that the iFG atoms must be in the
        same residue.  Need better general way to select iFG atoms...

        parsed_pdb: an instance of class ParsedPDB having attributes .contacts, .fs_struct, .fs_result, .dssp,
        .prody_pdb
        """

        assert isinstance(parsed_pdb.fs_struct,
                          freesasa.Structure), 'parsed_pdb object must have attribute freesasa structure obj'
        assert isinstance(parsed_pdb.fs_result,
                          freesasa.Result), 'parsed_pdb object must have attribute freesasa result obj'

        selections = freesasa.selectArea(('vdm_residue, chain ' + self.chid + ' and resi '
                                          + str(self.resnum),),
                                         parsed_pdb.fs_struct, parsed_pdb.fs_result)

        self.residue_sasa = '{0:.2f}'.format(selections['vdm_residue'])
        self.sasa_3A_probe = self.calc_large_probe_sasa(parsed_pdb, parsed_pdb.fs_result_cb_3A)
        self.sasa_4A_probe = self.calc_large_probe_sasa(parsed_pdb, parsed_pdb.fs_result_cb_4A)
        self.sasa_5A_probe = self.calc_large_probe_sasa(parsed_pdb, parsed_pdb.fs_result_cb_5A)

    def get_vandarotamer(self, ifg, comb):
        """vandarotamer is formated as (distance, angle1, dihedral1, angle2, dihedral2, dihedral3)"""
        try:
            self.vandarotamer = vdr.calc_vandarotamer_fmt(self, ifg, comb)
        except:
            self.vandarotamer = 'None'
    def get_info(self, ifg, parsed_pdb, comb):
        self.get_contact()
        if comb.probe_path:
            self.get_probe_hbonds(ifg)
        else:
            self.get_hbond(ifg)
        self.get_ca_hbond(ifg)
        self.get_sec_struct(parsed_pdb)
        self.get_rotamer(parsed_pdb)
        self.get_vandarotamer(ifg, comb)
        self.get_dssp_sasa(parsed_pdb)
        self.calc_sasa(parsed_pdb)
        self.get_min_hull_dist_cb_ca(parsed_pdb)

    def send_info(self, ifg, parsed_pdb, comb):
        self.get_info(ifg, parsed_pdb, comb)
        _vdm_pdb_info = [ifg.count,
                         ifg.vdm_count,
                         self.resname,
                         self.resnum,
                         self.resindex,
                         self.chid,
                         self.segment,
                         self.atom_names,
                         self.sequence,
                         self.length,
                         self.rel_resnums,
                         self.sec_struct_dssp,
                         self.sec_struct_phi_psi,
                         self.sec_struct_phi_psi_frag,
                         self.rotamer,
                         self.vandarotamer
                         ]

        _vdm_sasa_info = [ifg.count,
                          ifg.vdm_count,
                          self.min_hull_dist_cb_ca,
                          self.residue_sasa,
                          self.dssp_sasa,
                          self.sasa_3A_probe,
                          self.sasa_4A_probe,
                          self.sasa_5A_probe
                          ]

        _ifg_contact_vdm = [ifg.count,
                            ifg.vdm_count,
                            self.contact_number,
                            self.per_res_contact_number,
                            self.contact_atom_names,
                            self.contact_resnames,
                            self.contact_resnums,
                            self.contact_info,
                            self.ifg_contacts,
                            self.ifg_contact_pairs
                            ]

        if comb.probe_path:
            _ifg_hbond_vdm = [ifg.count,
                              ifg.vdm_count,
                              self.probe_hbonds]
            if any(_ifg_hbond_vdm[2:]):
                parsed_pdb._ifg_hbond_vdm.append(_ifg_hbond_vdm)
        else:
            _ifg_hbond_vdm = [ifg.count,
                              ifg.vdm_count,
                              self.hbond_number,
                              self.hbond_atom_names,
                              self.hbond_resnames,
                              self.hbond_resnums,
                              self.hbond_rel_resnums,
                              self.hbond_ifg_atom_names,
                              self.hbond_vdm_atom_names,
                              self.hbond_angle,
                              self.hbond_dist_acc_hyd,
                              self.hbond_dist_heavy,
                              ]

            _ifg_ca_hbond_vdm = [ifg.count,
                                 ifg.vdm_count,
                                 self.ca_hbond_number,
                                 self.ca_hbond_atom_names,
                                 self.ca_hbond_resnames,
                                 self.ca_hbond_resnums,
                                 self.ca_hbond_rel_resnums,
                                 self.ca_hbond_ifg_atom_names,
                                 self.ca_hbond_vdm_atom_names,
                                 self.ca_hbond_angle,
                                 self.ca_hbond_dist_acc_hyd,
                                 self.ca_hbond_dist_heavy,
                                 ]
            if any(_ifg_hbond_vdm[2:]):
                parsed_pdb._ifg_hbond_vdm.append(_ifg_hbond_vdm)
            if any(_ifg_ca_hbond_vdm[2:]):
                parsed_pdb._ifg_ca_hbond_vdm.append(_ifg_ca_hbond_vdm)

        parsed_pdb._vdm_pdb_info.append(_vdm_pdb_info)
        parsed_pdb._ifg_contact_vdm.append(_ifg_contact_vdm)
        parsed_pdb._vdm_sasa_info.append(_vdm_sasa_info)

        ifg.vdm_count += 1

    def print_pdb(self, ifg, parsed_pdb, comb):
        vdm_renum = renumber_chids_resnums(self, 'X')
        ifg_renum = renumber_chids_resnums(ifg, 'Y')
        filename = comb.output_dir_pdb + 'iFG_' + str(ifg.count) + '_vdM_' + str(ifg.vdm_count) \
                   + '_' + comb.file_tag + '.pdb.gz'
        with gzip.open(filename, 'wt') as pdbfile:
            pr.writePDBStream(pdbfile, vdm_renum, **{'secondary': False})
            pr.writePDBStream(pdbfile, ifg_renum, **{'secondary': False})
            if ifg.contact_atoms_water:
                pr.writePDBStream(pdbfile, ifg.contact_atoms_water, **{'secondary': False})
            if ifg.contact_atoms_metal:
                pr.writePDBStream(pdbfile, ifg.contact_atoms_metal, **{'secondary': False})
            if ifg.contact_atoms_ligand:
                pr.writePDBStream(pdbfile, parsed_pdb.prody_pdb.select('resindex ' + ' '.join(str(ri)
                                          for ri in np.unique(ifg.contact_atoms_ligand.getResindices()))),
                                  **{'secondary': False})













