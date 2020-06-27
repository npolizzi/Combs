__all__ = ['IntFG']


import prody as pr
import numpy as np
import freesasa
from ..rotamer.rotamer import calc_rotamer
from ..apps.constants import one_letter_code, resnames_aa_20, resnames_aa_20_join
import itertools
import collections
import os
import traceback
from functools import reduce


class IntFG:
    """A class for a prody selection of an interacting functional group (IntFG).  An instance of IntFG should store 
    the selection info of the iFG and contain methods to find the neighboring atoms within a certain distance. It 
    should also contain methods that find the unique residue names and chainIDs of those neighboring atoms, for use
    later in the creation of interactamer and vandermer objects.  It should also contain methods that calculate and 
    store info about the iFG such as burial and total number of Hbonds, etc.
    """

    def __init__(self, parsed_pdb, comb):
        """instance of class IntFG has attributes including selection names, neighboring atoms, neighborhood density 
        of atoms, etc."""
        self.sele = parsed_pdb.possible_ifgs.pop()
        self.resindex, self._ind = np.unique(self.sele.getResindices(), return_index=True)
        self.resname = self.sele.getResnames()[self._ind]
        self.resnum = self.sele.getResnums()[self._ind]
        self.atom_names = {resname: self.sele.select('resindex ' + str(resindex)).getNames() for resname, resindex in
                           zip(self.resname, self.resindex)}
        self.chid = np.unique(self.sele.getChids())[0]
        self.center_coords = pr.calcCenter(self.sele)
        self.vdm_count = 1
        self.count = comb.ifg_count
        self.sasa = None
        self.residue_sasa = None
        self.dssp_sasa = None
        self.sasa_3A_probe = None
        self.sasa_4A_probe = None
        self.sasa_5A_probe = None
        self.contact_atoms_all = None
        self.contact_atoms_protein = None
        self.contact_resnums = None
        self.contact_chids = None
        self.contact_resindices = None
        self.contact_segments = None
        self.contact_atoms_water = None
        self.contact_atoms_ligand = None
        self.contact_atoms_metal = None
        self.contact_info_water = []
        self.contact_info_ligand = []
        self.contact_info_metal = []
        self.contact_info_protein = []
        self.contact_dict = collections.defaultdict(set)
        self.contact_pair_dict = collections.defaultdict(list)
        self.probe_hbonds = []
        self.rotamer = None
        self.min_hull_dist_ifg = None
        self.min_hull_dist_cb_ca = None
        self.cbeta_density = None
        self.heavy_atom_density_5A = None
        self.heavy_atom_density_10A = None
        if comb.ifg_seq_str != 'element':
            self.ifg_frag = parsed_pdb.prody_pdb.select('segment A and chain ' + self.chid + ' and resnum `' + str(np.min(self.resnum)-1)
                                                    + 'to' + str(np.max(self.resnum)+1) + '`')
        else:
            self.ifg_frag = self.sele
        self.frag_length = len(np.unique(self.ifg_frag.getResindices()))
        if comb.ifg_seq_str != 'element':
            self.sequence = ''.join(one_letter_code[rn] for rn in self.resname)
        else:
            self.sequence = ''
        self.sec_struct_dssp = None
        self.sec_struct_phi_psi = None
        self.contact_number_water = None
        self.per_res_contact_number_water = None
        self.contact_atom_names_water = None
        self.contact_resnames_water = None
        self.contact_resnums_water = None
        self.contact_number_ligand = None
        self.per_res_contact_number_ligand = None
        self.contact_atom_names_ligand = None
        self.contact_resnames_ligand = None
        self.contact_resnums_ligand = None
        self.contact_number_metal = None
        self.per_res_contact_number_metal = None
        self.contact_atom_names_metal = None
        self.contact_resnames_metal = None
        self.contact_resnums_metal = None
        self.hbond_atom_names = []
        self.hbond_resnames = []
        self.hbond_resnums = []
        self.hbond_angle = []
        self.hbond_dist_acc_hyd = []
        self.hbond_dist_heavy = []
        self.hbond_atom_names_water = []
        self.hbond_number_water = []
        self.hbond_resnames_water = []
        self.hbond_resnums_water = []
        self.hbond_angle_water = []
        self.hbond_dist_acc_hyd_water = []
        self.hbond_dist_heavy_water = []
        self.hbond_number_ligand = []
        self.hbond_atom_names_ligand = []
        self.hbond_resnames_ligand = []
        self.hbond_resnums_ligand = []
        self.hbond_angle_ligand = []
        self.hbond_dist_acc_hyd_ligand = []
        self.hbond_dist_heavy_ligand = []
        self.ca_hbond_atom_names = []
        self.ca_hbond_resnames = []
        self.ca_hbond_resnums = []
        self.ca_hbond_angle = []
        self.ca_hbond_dist_acc_hyd = []
        self.ca_hbond_dist_heavy = []
        self.bb_cb_atom_ind = self.get_bb_cb_atom_indices(parsed_pdb)

    def get_bb_cb_atom_indices(self, parsed_pdb):
        sele = parsed_pdb.prody_pdb.select('protein and (backbone or name CB) and resindex '
                                           + ' '.join(str(ri) for ri in self.resindex) + ' and not element H D')
        if sele is not None:
            return sele.getIndices()
        else:
            return None

    def find_contact_atoms(self, parsed_pdb, comb, radius1=3.5, radius2=4.8):
        """Takes a prody contacts object and a radius (integer) and finds the neighboring atoms of the iFG.  The
        atoms are saved as all, protein, waters, ligands, and metals. The atoms do not include atoms of the
        iFG residue itself.

        parsed_pdb: an instance of class ParsedPDB having attributes .contacts, .fs_struct, .fs_result, .dssp
        radius: an integer
        """

        if comb.probe_path:
            try:
                scratch_dir = comb.scratch or comb.output_dir_csv
                resnum = np.unique(self.sele.getResnums())[0]
                chain = np.unique(self.sele.getChids())[0]
                resname = np.unique(self.sele.getResnames())[0]
                ifg_dict = collections.defaultdict(list)

                selfsele = '"seg___A chain' + chain + ' ' + str(resnum) + ' ' + comb.probe_ifgatoms[resname] + '"'
                targsele = '"seg___A not (' + str(resnum) + ' chain' + chain + ')"'
                # Probe output abbr: wc: wide contact, cc: close contact, so: small overlap, bo: bad overlap, hb: H-bonds
                with os.popen(comb.probe_path + 'probe -quiet -MC -docho -condense -unformated -once '
                                      + selfsele + ' ' + targsele + ' ' + comb.input_dir_pdb
                                      + parsed_pdb.pdb_acc_code + 'H.pdb') as infile:
                    for line in infile:
                        spl = line.split(':')[1:]
                        probe_type = spl[1]
                        ifg_name = spl[2][10:15].strip()
                        ifg_atomtype = spl[12]
                        vdm_chid = spl[3][:2].strip()
                        vdm_resnum = int(spl[3][2:6])
                        vdm_resname = spl[3][6:10].strip()
                        if vdm_resname == 'HOH':
                            vdm_name = 'O'
                        else:
                            vdm_name = spl[3][10:15].strip()
                        vdm_atomtype = spl[13]
                        pr_score = spl[11]
                        ifg_dict['contacts'].append(
                            (ifg_name, ifg_atomtype, 'A', vdm_chid, vdm_resnum, vdm_resname, vdm_name, vdm_atomtype, pr_score))
                        self.contact_dict[(vdm_resnum, vdm_chid, 'A')].add(ifg_name)
                        self.contact_pair_dict[(vdm_resnum, vdm_chid, 'A')].append((ifg_name, vdm_name, spl[1], pr_score))
                        if probe_type == 'hb':
                            ifg_dict['hbonds'].append(
                                (ifg_name, ifg_atomtype, 'A', vdm_chid, vdm_resnum, vdm_resname, vdm_name, vdm_atomtype))

                for seg in parsed_pdb.segnames[1:]:
                    s = '____'
                    s = s[:-len(seg):] + seg
                    selfsele = '"seg___A chain' + chain + ' ' + str(resnum) + ' ' + comb.probe_ifgatoms[resname] + '"'
                    targsele = '"seg' + s + '"'
                    with os.popen(comb.probe_path + 'probe -quiet -MC -docho -condense -unformated -once '
                                          + selfsele + ' ' + targsele + ' ' + comb.input_dir_pdb
                                          + parsed_pdb.pdb_acc_code + 'H.pdb') as infile:
                        for line in infile:
                            spl = line.split(':')[1:]
                            probe_type = spl[1]
                            ifg_name = spl[2][10:15].strip()
                            ifg_atomtype = spl[12]
                            vdm_chid = spl[3][:2].strip()
                            vdm_resnum = int(spl[3][2:6])
                            vdm_resname = spl[3][6:10].strip()
                            if vdm_resname == 'HOH':
                                vdm_name = 'O'
                            else:
                                vdm_name = spl[3][10:15].strip()
                            vdm_atomtype = spl[13]
                            pr_score = spl[11]
                            self.contact_dict[(vdm_resnum, vdm_chid, seg)].add(ifg_name)
                            self.contact_pair_dict[(vdm_resnum, vdm_chid, seg)].append((ifg_name, vdm_name, spl[1], pr_score))
                            ifg_dict['contacts'].append(
                                (ifg_name, ifg_atomtype, seg, vdm_chid, vdm_resnum, vdm_resname, vdm_name, vdm_atomtype,
                                 pr_score))
                            if probe_type == 'hb':
                                ifg_dict['hbonds'].append(
                                    (
                                    ifg_name, ifg_atomtype, seg, vdm_chid, vdm_resnum, vdm_resname, vdm_name, vdm_atomtype))

                contact_atom_seles = [parsed_pdb.prody_pdb.select('segment ' + c[2]
                                                        + ' chain ' + c[3]
                                                        + ' resnum ' + str(c[4])
                                                        + ' name ' + c[6]) for c in ifg_dict['contacts']]
                self.probe_hbonds = ifg_dict['hbonds']

                if contact_atom_seles != []:
                    self.contact_atoms_all = reduce(lambda a, b: a | b, contact_atom_seles)
                    nbrs = pr.findNeighbors(self.sele, 4.8, self.contact_atoms_all)
                    nbrs_info = [(nbr[1], nbr[1].getResindex(), nbr[1].getResnum(), nbr[1].getChid(), nbr[0].getName(),
                                  nbr[1].getName(), nbr[2]) for nbr in nbrs]
                    nbrs_info_full = sorted(nbrs_info, key=lambda x: x[1])
                    contact_names_dist = {}
                    resind_resnum_chids = []
                    for resind_resnum_chid, contact_group in itertools.groupby(nbrs_info_full, key=lambda x: x[1:4]):
                        contact_names_dist[resind_resnum_chid[0]] = [c[4:] for c in contact_group]
                        resind_resnum_chids.append(resind_resnum_chid)
                else:
                    self.contact_atoms_all = None
            except Exception:
                traceback.print_exc()
                self.contact_atoms_all = None

        else:
            nbrs = pr.findNeighbors(self.sele, radius2,
                                    parsed_pdb.prody_pdb.select('not element H D and not resindex '
                                                                + ' '.join(str(ri) for ri in self.resindex)))

            if nbrs:
                nbrs_info = [(nbr[1], nbr[1].getResindex(), nbr[1].getResnum(), nbr[1].getChid(), nbr[0].getName(),
                             nbr[1].getName(), nbr[2]) for nbr in nbrs]
                nbrs_info_3p5 = [c for c in nbrs_info if c[-1] <= radius1]
                nbrs_info_4p8 = [c for c in nbrs_info if c[-1] > radius1 and c[-2][0] == 'C']
                nbrs_info_full = sorted(nbrs_info_3p5 + nbrs_info_4p8, key=lambda x: x[1])

                if nbrs_info_full:
                    contact_names_dist = {}
                    resind_resnum_chids = []
                    for resind_resnum_chid, contact_group in itertools.groupby(nbrs_info_full, key=lambda x: x[1:4]):
                        contact_names_dist[resind_resnum_chid[0]] = [c[4:] for c in contact_group]
                        resind_resnum_chids.append(resind_resnum_chid)
                    self.contact_atoms_all = self.get_contact_atoms(nbrs_info_full)

        # selects the subsets of atoms
        if self.contact_atoms_all is not None:
            self.contact_atoms_water = self.contact_atoms_all.select('water')
            if self.contact_atoms_water is not None:
                self.contact_info_water = ' '.join('(' + b[0] + ' ' + b[1] + ' ' + '{0:.2f}'.format(b[2]) + ')'
                                           for c in [contact_names_dist[ri]
                                           for ri in np.unique(self.contact_atoms_water.getResindices())]
                                                   for b in c)

            self.contact_atoms_ligand = self.contact_atoms_all.select('hetero and not (water or element NA MG CA K '
                                                                       'FE ZN CO CU NI MN MO V)')
            if self.contact_atoms_ligand is not None:
                self.contact_info_ligand = ' '.join('(' + b[0] + ' ' + b[1] + ' ' + '{0:.2f}'.format(b[2]) + ')'
                                           for c in [contact_names_dist[ri]
                                           for ri in np.unique(self.contact_atoms_ligand.getResindices())]
                                                    for b in c)

            self.contact_atoms_metal = self.contact_atoms_all.select('element NA MG CA K FE ZN CO CU NI MN MO V')
            if self.contact_atoms_metal is not None:
                self.contact_info_metal = ' '.join('(' + b[0] + ' ' + b[1] + ' ' + '{0:.2f}'.format(b[2]) + ')'
                                           for c in [contact_names_dist[ri]
                                           for ri in np.unique(self.contact_atoms_metal.getResindices())]
                                                   for b in c)

            self.contact_atoms_protein = self.contact_atoms_all.select('resname ' + resnames_aa_20_join)
            if self.contact_atoms_protein is not None:
                _resin, _ind = np.unique(self.contact_atoms_protein.getResindices(), return_index=True)
                self.contact_resnums = self.contact_atoms_protein.getResnums()[_ind].tolist()
                self.contact_chids = self.contact_atoms_protein.getChids()[_ind].tolist()
                self.contact_resindices = self.contact_atoms_protein.getResindices()[_ind].tolist()
                self.contact_segments = self.contact_atoms_protein.getSegnames()[_ind].tolist()
                self.contact_info_protein = [contact_names_dist[ri] for ri in self.contact_resindices]
                comb.ifg_count += 1

    @staticmethod
    def get_contact_atoms(nbrs_info_full):
        contact_atoms = nbrs_info_full[0][0]
        for nbr in nbrs_info_full[1:]:
            contact_atoms = contact_atoms | nbr[0]
        return contact_atoms

    def get_min_hull_dist_cb_ca(self, parsed_pdb):
        sel = parsed_pdb.prody_pdb.select('segment A and chain ' + self.chid
                                          + ' and resnum ' + str(self.resnum[0]))
        try:
            if set(sel.getResnames()).pop() == 'GLY':
                pnt = sel.select('name CA').getCoords().flatten()
            elif len(sel) == 1:
                pnt = sel.getCoords().flatten()
            else:
                pnt = sel.select('name CB').getCoords().flatten()
        except:
            pnt = sel.select('name FE').getCoords().flatten()
        self.min_hull_dist_cb_ca = '{0:.2f}'.format(parsed_pdb.alphahull.get_pnt_distance(pnt))

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

        if len(self.resnum) == 1:
            selections = freesasa.selectArea(('ifg_atoms, chain ' + self.chid + ' and resi ' + str(self.resnum[0])
                                              + ' and name ' + '+'.join(self.atom_names[self.resname[0]]),
                                              'ifg_residue, chain ' + self.chid + ' and resi ' + str(self.resnum[0])),
                                             parsed_pdb.fs_struct, parsed_pdb.fs_result)
        else:
            selections = freesasa.selectArea(('ifg_atoms, chain ' + self.chid + ' and ((resi ' + str(self.resnum[0])
                                              + ' and name ' + '+'.join(self.atom_names[self.resname[0]])
                                              + ') or (resi ' + str(self.resnum[1]) + ' and name '
                                              + '+'.join(self.atom_names[self.resname[1]]) + '))',
                                              'ifg_residue, chain ' + self.chid + ' and resi '
                                              + '+'.join(str(rn) for rn in self.resnum)), parsed_pdb.fs_struct,
                                             parsed_pdb.fs_result)

        self.sasa = '{0:.2f}'.format(selections['ifg_atoms'])
        self.residue_sasa = '{0:.2f}'.format(selections['ifg_residue'])
        self.sasa_3A_probe = self.calc_large_probe_sasa(parsed_pdb, parsed_pdb.fs_result_cb_3A)
        self.sasa_4A_probe = self.calc_large_probe_sasa(parsed_pdb, parsed_pdb.fs_result_cb_4A)
        self.sasa_5A_probe = self.calc_large_probe_sasa(parsed_pdb, parsed_pdb.fs_result_cb_5A)

    def calc_large_probe_sasa(self, parsed_pdb, fs_result):
        if self.bb_cb_atom_ind is not None:
            if self.bb_cb_atom_ind.any():
                return '{0:.2f}'.format(sum(fs_result.atomArea(i) for i in np.where(np.in1d(parsed_pdb.prody_pdb_bb_cb_atom_ind,
                                                                       self.bb_cb_atom_ind))[0]))

    def calc_atom_density(self, parsed_pdb, radius, atom_sel_str):
        """"""
        sele = parsed_pdb.prody_pdb.select('(' + atom_sel_str + ' and not resindex ' +
                                               ' '.join(str(ri) for ri in self.resindex) + ') within ' + str(radius)
                                               + ' of center', center=self.center_coords)
        return len(sele) if sele is not None else 0

    def get_atom_densities(self, parsed_pdb):
        self.cbeta_density = self.calc_atom_density(parsed_pdb, 10, 'name CB')
        self.heavy_atom_density_5A = self.calc_atom_density(parsed_pdb, 5, '(not element H D)')
        self.heavy_atom_density_10A = self.calc_atom_density(parsed_pdb, 10, '(not element H D)')

    def get_contact_water(self):
        if self.contact_atoms_water is not None:
            _resnums, _ind = np.unique(self.contact_atoms_water.getResnums(), return_index=True)
            _sort_ind = np.argsort(_resnums)
            self.contact_resnums_water = ' '.join(str(rn) for rn in (_resnums[_sort_ind]))
            _contact_number = [len(self.contact_atoms_water.select('resnum `' + str(rn) + '`'))
                               for rn in _resnums[_sort_ind]]
            self.contact_number_water = np.sum(_contact_number)
            self.per_res_contact_number_water = ' '.join(str(num) for num in _contact_number)
            self.contact_resnames_water = ' '.join(self.contact_atoms_water.getResnames()[_ind])
            _contact_atom_names = ['(' + ' '.join(self.contact_atoms_water.select('resnum `' + str(rn)
                                                                                  + '`').getNames()) + ')'
                                   for rn in _resnums[_sort_ind]]
            self.contact_atom_names_water = ' '.join(_contact_atom_names)
    
    def get_contact_ligand(self):
        if self.contact_atoms_ligand is not None:
            _resnums, _ind = np.unique(self.contact_atoms_ligand.getResnums(), return_index=True)
            _sort_ind = np.argsort(_resnums)
            self.contact_resnums_ligand = ' '.join(str(rn) for rn in (_resnums[_sort_ind]))
            _contact_number = [len(self.contact_atoms_ligand.select('resnum `' + str(rn) + '`'))
                               for rn in _resnums[_sort_ind]]
            self.contact_number_ligand = np.sum(_contact_number)
            self.per_res_contact_number_ligand = ' '.join(str(num) for num in _contact_number)
            self.contact_resnames_ligand = ' '.join(self.contact_atoms_ligand.getResnames()[_ind])
            _contact_atom_names = ['(' + ' '.join(self.contact_atoms_ligand.select('resnum `' + str(rn)
                                                                                   + '`').getNames()) + ')'
                                   for rn in _resnums[_sort_ind]]
            self.contact_atom_names_ligand = ' '.join(_contact_atom_names)
    
    def get_contact_metal(self):
        if self.contact_atoms_metal is not None:
            _resnums, _ind = np.unique(self.contact_atoms_metal.getResnums(), return_index=True)
            _sort_ind = np.argsort(_resnums)
            self.contact_resnums_metal = ' '.join(str(rn) for rn in (_resnums[_sort_ind]))
            _contact_number = [len(self.contact_atoms_metal.select('resnum `' + str(rn) + '`'))
                               for rn in _resnums[_sort_ind]]
            self.contact_number_metal = np.sum(_contact_number)
            self.per_res_contact_number_metal = ' '.join(str(num) for num in _contact_number)
            self.contact_resnames_metal = ' '.join(self.contact_atoms_metal.getResnames()[_ind])
            _contact_atom_names = ['(' + ' '.join(self.contact_atoms_metal.select('resnum `' + str(rn)
                                                                                  + '`').getNames()) + ')'
                                   for rn in _resnums[_sort_ind]]
            self.contact_atom_names_metal = ' '.join(_contact_atom_names)
        
    def get_contact(self):
        self.get_contact_ligand()
        self.get_contact_water()
        self.get_contact_metal()

    def iter_triplets_acc(self, parsed_pdb):
        NO_selection = self.sele.select('element O or (resname HIS and sidechain and element N)')
        if NO_selection is not None:
            for acceptor in NO_selection.iterAtoms():
                hyds = parsed_pdb.contacts(2.5, acceptor.getCoords()).select('element H D')
                if hyds is not None:
                    if hyds & ~self.sele is not None:
                        hyds = hyds & ~self.sele
                        for hyd in hyds.iterAtoms():
                            donors = parsed_pdb.contacts(1.1, hyd.getCoords()).select('element O N')
                            if donors is not None:
                                for donor in donors.iterAtoms():
                                    yield (acceptor, hyd, donor)
                waters = parsed_pdb.contacts(3.2, acceptor.getCoords()).select('water')
                if waters is not None:
                    if waters & ~self.sele is not None:
                        waters = waters & ~self.sele
                        for water in waters.iterAtoms():
                            yield (acceptor, water, water)

    def iter_triplets_don(self, parsed_pdb):
        NO_selection = self.sele.select('element O N')
        if NO_selection is not None:
            for donor in NO_selection.iterAtoms():
                hyds = parsed_pdb.contacts(1.1, donor.getCoords()).select('element H D')
                if hyds is not None:
                    for hyd in hyds.iterAtoms():
                        acceptors = parsed_pdb.contacts(2.5, hyd.getCoords()).select('element O N') & ~self.sele
                        if acceptors is not None:
                            for acceptor in acceptors.iterAtoms():
                                if acceptor.getElement() == 'N':
                                    if parsed_pdb.contacts(1.1, acceptor.getCoords()).select('element H D') is None:
                                        yield (acceptor, hyd, donor)
                                else:
                                    yield (acceptor, hyd, donor)

    def iter_hbonds_acc(self, parsed_pdb):
        for hbond in self.iter_triplets_acc(parsed_pdb):
            if hbond[2].getResname() == 'HOH':
                hyds = parsed_pdb.contacts(1.1, hbond[0].getCoords()).select('element H D')
                if hyds is not None:
                    angle_tests = []
                    for hyd in hyds.iterAtoms():
                        angle = pr.calcAngle(hbond[0], hyd, hbond[2])
                        if angle > 90:
                            angle_tests.append(False)
                        else:
                            angle_tests.append(True)
                    if all(angle_tests):
                        distance_acc_hyd = 99
                        distance_heavy = pr.calcDistance(hbond[0], hbond[2])
                        angle = 360
                        yield hbond, angle, distance_acc_hyd, distance_heavy
                else:
                    distance_acc_hyd = 99
                    distance_heavy = pr.calcDistance(hbond[0], hbond[2])
                    angle = 360
                    yield hbond, angle, distance_acc_hyd, distance_heavy
            else:
                angle = pr.calcAngle(hbond[0], hbond[1], hbond[2])
                if angle > 90:
                    distance_acc_hyd = pr.calcDistance(hbond[0], hbond[1])
                    distance_heavy = pr.calcDistance(hbond[0], hbond[2])
                    yield hbond, angle, distance_acc_hyd, distance_heavy

    def iter_hbonds_don(self, parsed_pdb):
        for hbond in self.iter_triplets_don(parsed_pdb):
            angle = pr.calcAngle(hbond[0], hbond[1], hbond[2])
            if angle > 90:
                distance_acc_hyd = pr.calcDistance(hbond[0], hbond[1])
                distance_heavy = pr.calcDistance(hbond[0], hbond[2])
                yield hbond, angle, distance_acc_hyd, distance_heavy

    def iter_hbonds(self, parsed_pdb):
        for hbond, angle, distance_acc_hyd, distance_heavy in self.iter_hbonds_acc(parsed_pdb):
            yield hbond, angle, distance_acc_hyd, distance_heavy
        for hbond, angle, distance_acc_hyd, distance_heavy in self.iter_hbonds_don(parsed_pdb):
            yield hbond, angle, distance_acc_hyd, distance_heavy

    def find_hbonds(self, parsed_pdb):
        for hbond, angle, distance_acc_hyd, distance_heavy in self.iter_hbonds(parsed_pdb):
            self.hbond_atom_names.append(list(map(lambda x: x.getName(), hbond)))
            self.hbond_resnames.append(list(map(lambda x: x.getResname(), hbond)))
            self.hbond_resnums.append(list(map(lambda x: x.getResnum(), hbond)))
            self.hbond_angle.append(angle)
            self.hbond_dist_acc_hyd.append(distance_acc_hyd)
            self.hbond_dist_heavy.append(distance_heavy)

    def partition_hbonds_water(self):
        _hbond_atom_names_water = []
        _hbond_resnames_water = []
        _hbond_resnums_water = []
        _hbond_angle_water = []
        _hbond_dist_acc_hyd_water = []
        _hbond_dist_heavy_water = []
        for i, resnames in enumerate(self.hbond_resnames):
            if 'HOH' in resnames:
                _hbond_atom_names_water.append('(' + ' '.join(self.hbond_atom_names[i]) + ')')
                _hbond_resnames_water.append('(' + ' '.join(self.hbond_resnames[i]) + ')')
                _hbond_resnums_water.append('(' + ' '.join(str(rn) for rn in self.hbond_resnums[i]) + ')')
                _hbond_angle_water.append('{0:.2f}'.format(self.hbond_angle[i]))
                _hbond_dist_acc_hyd_water.append('{0:.2f}'.format(self.hbond_dist_acc_hyd[i]))
                _hbond_dist_heavy_water.append('{0:.2f}'.format(self.hbond_dist_heavy[i]))
        if _hbond_atom_names_water:
            self.hbond_atom_names_water = ' '.join(_hbond_atom_names_water)
            self.hbond_resnames_water = ' '.join(_hbond_resnames_water)
            self.hbond_resnums_water = ' '.join(_hbond_resnums_water)
            self.hbond_angle_water = ' '.join(_hbond_angle_water)
            self.hbond_dist_acc_hyd_water = ' '.join(_hbond_dist_acc_hyd_water)
            self.hbond_dist_heavy_water = ' '.join(_hbond_dist_heavy_water)
            self.hbond_number_water = len(_hbond_resnums_water)

    def partition_hbonds_ligand(self):
        _hbond_atom_names_ligand = []
        _hbond_resnames_ligand = []
        _hbond_resnums_ligand = []
        _hbond_angle_ligand = []
        _hbond_dist_acc_hyd_ligand = []
        _hbond_dist_heavy_ligand = []
        _not_ligand = ['HOH']
        _not_ligand.extend(one_letter_code.keys())
        for i, resnames in enumerate(self.hbond_resnames):
            if not set(_not_ligand).intersection(set(resnames)):
                _hbond_atom_names_ligand.append('(' + ' '.join(self.hbond_atom_names[i]) + ')')
                _hbond_resnames_ligand.append('(' + ' '.join(self.hbond_resnames[i]) + ')')
                _hbond_resnums_ligand.append('(' + ' '.join(str(rn) for rn in self.hbond_resnums[i]) + ')')
                _hbond_angle_ligand.append('{0:.2f}'.format(self.hbond_angle[i]))
                _hbond_dist_acc_hyd_ligand.append('{0:.2f}'.format(self.hbond_dist_acc_hyd[i]))
                _hbond_dist_heavy_ligand.append('{0:.2f}'.format(self.hbond_dist_heavy[i]))
        if _hbond_atom_names_ligand:
            self.hbond_atom_names_ligand = ' '.join(_hbond_atom_names_ligand)
            self.hbond_resnames_ligand = ' '.join(_hbond_resnames_ligand)
            self.hbond_resnums_ligand = ' '.join(_hbond_resnums_ligand)
            self.hbond_angle_ligand = ' '.join(_hbond_angle_ligand)
            self.hbond_dist_acc_hyd_ligand = ' '.join(_hbond_dist_acc_hyd_ligand)
            self.hbond_dist_heavy_ligand = ' '.join(_hbond_dist_heavy_ligand)
            self.hbond_number_ligand = len(_hbond_resnums_ligand)

    def get_probe_hbonds_lig(self):
        self.probe_hbonds_lig = ' : '.join([' '.join(str(k) for k in h) for h in self.probe_hbonds
                                            if (h[5] != 'HOH' and h[5] not in resnames_aa_20)])

    def get_probe_hbonds_hoh(self):
        self.probe_hbonds_hoh = ' : '.join([' '.join(str(k) for k in h) for h in self.probe_hbonds if h[5] == 'HOH'])

    def get_hbonds(self, parsed_pdb, comb):
        if comb.probe_path:
            self.get_probe_hbonds_lig()
            self.get_probe_hbonds_hoh()
        else:
            self.find_hbonds(parsed_pdb)
            self.partition_hbonds_water()
            self.partition_hbonds_ligand()

    def iter_ca_triplets_acc(self, parsed_pdb):
        NO_selection = self.sele.select('element O or (resname HIS and sidechain and element N)')
        if NO_selection is not None:
            for acceptor in NO_selection.iterAtoms():
                hyds = parsed_pdb.contacts(2.7, acceptor.getCoords()).select('element H D') # see Horowitz JBC 2012
                if hyds is not None:
                    if hyds & ~self.sele is not None:
                        hyds = hyds & ~self.sele
                        for hyd in hyds.iterAtoms():
                            donors = parsed_pdb.contacts(1.1, hyd.getCoords()).select('name CA')
                            if donors is not None:
                                for donor in donors.iterAtoms():
                                    yield (acceptor, hyd, donor)

    def iter_ca_triplets_don(self, parsed_pdb):
        NO_selection = self.sele.select('name CA')
        if NO_selection is not None:
            for donor in NO_selection.iterAtoms():
                hyds = parsed_pdb.contacts(1.1, donor.getCoords()).select('element H D')
                if hyds is not None:
                    for hyd in hyds.iterAtoms():
                        acceptors = parsed_pdb.contacts(2.7, hyd.getCoords()).select('element O N') & ~self.sele
                        if acceptors is not None:
                            for acceptor in acceptors.iterAtoms():
                                yield (acceptor, hyd, donor)

    def iter_ca_hbonds_acc(self, parsed_pdb):
        for hbond in self.iter_ca_triplets_acc(parsed_pdb):
            angle = pr.calcAngle(hbond[0], hbond[1], hbond[2])
            if angle > 90:
                distance_acc_hyd = pr.calcDistance(hbond[0], hbond[1])
                distance_heavy = pr.calcDistance(hbond[0], hbond[2])
                yield hbond, angle, distance_acc_hyd, distance_heavy

    def iter_ca_hbonds_don(self, parsed_pdb):
        for hbond in self.iter_ca_triplets_don(parsed_pdb):
            angle = pr.calcAngle(hbond[0], hbond[1], hbond[2])
            if angle > 90:
                distance_acc_hyd = pr.calcDistance(hbond[0], hbond[1])
                distance_heavy = pr.calcDistance(hbond[0], hbond[2])
                yield hbond, angle, distance_acc_hyd, distance_heavy

    def iter_ca_hbonds(self, parsed_pdb):
        for hbond, angle, distance_acc_hyd, distance_heavy in self.iter_ca_hbonds_acc(parsed_pdb):
            yield hbond, angle, distance_acc_hyd, distance_heavy
        for hbond, angle, distance_acc_hyd, distance_heavy in self.iter_ca_hbonds_don(parsed_pdb):
            yield hbond, angle, distance_acc_hyd, distance_heavy

    def get_ca_hbonds(self, parsed_pdb, comb):
        if comb.probe_path:
            pass
        else:
            for hbond, angle, distance_acc_hyd, distance_heavy in self.iter_ca_hbonds(parsed_pdb):
                self.ca_hbond_atom_names.append(list(map(lambda x: x.getName(), hbond)))
                self.ca_hbond_resnames.append(list(map(lambda x: x.getResname(), hbond)))
                self.ca_hbond_resnums.append(list(map(lambda x: x.getResnum(), hbond)))
                self.ca_hbond_angle.append(angle)
                self.ca_hbond_dist_acc_hyd.append(distance_acc_hyd)
                self.ca_hbond_dist_heavy.append(distance_heavy)

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
                    + ' segment A chain ' + self.chid + ' name N)'),
                    parsed_pdb.prody_pdb.select(
                        'resnum ' + str(resn)
                        + ' segment A chain ' + self.chid + ' name N'),
                    parsed_pdb.prody_pdb.select(
                        'resnum ' + str(resn)
                        + ' segment A chain ' + self.chid + ' name CA'),
                    parsed_pdb.prody_pdb.select(
                        'resnum ' + str(resn)
                        + ' segment A chain ' + self.chid + ' name C'))[0]
            except:
                phi = None
            try:
                psi = pr.calcDihedral(
                    parsed_pdb.prody_pdb.select(
                        'resnum ' + str(resn)
                        + ' segment A chain ' + self.chid + ' name N'),
                    parsed_pdb.prody_pdb.select(
                        'resnum ' + str(resn)
                        + ' segment A chain ' + self.chid + ' name CA'),
                    parsed_pdb.prody_pdb.select(
                        'resnum ' + str(resn)
                        + ' segment A chain ' + self.chid + ' name C'),
                    parsed_pdb.prody_pdb.select(
                        'name N and bonded 1 to (resnum ' + str(resn)
                        + ' segment A chain ' + self.chid + ' name C)'))[0]
            except:
                psi = None
            if phi is not None and psi is not None:
                phipsi.append('(' + '{0:.2f}'.format(phi) + ' ' + '{0:.2f}'.format(psi) + ')')
            elif phi is None and psi is not None:
                phipsi.append('(' + 'None' + ' ' + '{0:.2f}'.format(psi) + ')')
            elif phi is not None and psi is None:
                phipsi.append('(' + '{0:.2f}'.format(phi) + ' ' + 'None' + ')')
            else:
                phipsi.append('(None None)')
        self.sec_struct_phi_psi = ' '.join(phipsi)

    def get_sec_struct(self, parsed_pdb):
        self.get_sec_struct_dssp(parsed_pdb)
        self.get_sec_struct_phipsi(parsed_pdb)

    def get_rotamer(self, parsed_pdb):
        _rotamer = calc_rotamer(parsed_pdb.prody_pdb, self.resnum[0], self.chid, 'A')
        newrot = []
        for item in _rotamer:
            newitem = ['{0:.2f}'.format(float(chi)) for chi in item]
            newrot.append('(' + ' '.join(newitem) + ')')
        self.rotamer = ' '.join(newrot)

    def get_dssp_sasa(self, parsed_pdb):
        try:
            bb_resnums = self.sele.getResnums()[self._ind]
            bb_chids = self.sele.getChids()[self._ind]
            bb_dssp = [parsed_pdb.dssp_sasa[(rn, ch)] for rn, ch in zip(bb_resnums, bb_chids)]
            self.dssp_sasa = ' '.join(bb_dssp)
        except:
            self.dssp_sasa = '-'

    def get_info(self, parsed_pdb):
        self.get_min_hull_dist_cb_ca(parsed_pdb)
        self.calc_sasa(parsed_pdb)
        self.get_atom_densities(parsed_pdb)
        self.get_contact()
        self.get_sec_struct(parsed_pdb)
        self.get_rotamer(parsed_pdb)
        self.get_dssp_sasa(parsed_pdb)

    def send_info(self, parsed_pdb, comb):
        self.get_info(parsed_pdb)
        _ifg_pdb_info = [self.count,
                         parsed_pdb.pdb_acc_code,
                         ' '.join(rn for rn in self.resname),
                         ' '.join(str(rn) for rn in self.resnum),
                         ' '.join(str(ri) for ri in self.resindex),
                         self.chid,
                         'A',
                         ' '.join(key + ': ' + ' '.join(val) for key, val in self.atom_names.items()),
                         self.sequence,
                         self.frag_length,
                         self.sec_struct_dssp,
                         self.sec_struct_phi_psi,
                         self.rotamer
                         ]

        _ifg_atom_density = [self.count,
                             self.min_hull_dist_cb_ca,
                             self.cbeta_density,
                             self.heavy_atom_density_5A,
                             self.heavy_atom_density_10A,
                             self.sasa,
                             self.residue_sasa,
                             self.dssp_sasa,
                             self.sasa_3A_probe,
                             self.sasa_4A_probe,
                             self.sasa_5A_probe
                             ]

        _ifg_contact_water = [self.count,
                              self.contact_number_water,
                              self.per_res_contact_number_water,
                              self.contact_atom_names_water,
                              self.contact_resnames_water,
                              self.contact_resnums_water,
                              self.contact_info_water
                              ]

        _ifg_contact_ligand = [self.count,
                               self.contact_number_ligand,
                               self.per_res_contact_number_ligand,
                               self.contact_atom_names_ligand,
                               self.contact_resnames_ligand,
                               self.contact_resnums_ligand,
                               self.contact_info_ligand
                               ]

        _ifg_contact_metal = [self.count,
                              self.contact_number_metal,
                              self.per_res_contact_number_metal,
                              self.contact_atom_names_metal,
                              self.contact_resnames_metal,
                              self.contact_resnums_metal,
                              self.contact_info_metal
                              ]

        if comb.probe_path:
            _ifg_hbond_water = [self.count,
                                self.probe_hbonds_hoh]

            _ifg_hbond_ligand = [self.count,
                                 self.probe_hbonds_lig]

        else:
            _ifg_hbond_water = [self.count,
                                self.hbond_number_water,
                                self.hbond_atom_names_water,
                                self.hbond_resnames_water,
                                self.hbond_resnums_water,
                                self.hbond_angle_water,
                                self.hbond_dist_acc_hyd_water,
                                self.hbond_dist_heavy_water,
                                ]

            _ifg_hbond_ligand = [self.count,
                                 self.hbond_number_ligand,
                                 self.hbond_atom_names_ligand,
                                 self.hbond_resnames_ligand,
                                 self.hbond_resnums_ligand,
                                 self.hbond_angle_ligand,
                                 self.hbond_dist_acc_hyd_ligand,
                                 self.hbond_dist_heavy_ligand,
                                 ]

        parsed_pdb._ifg_pdb_info.append(_ifg_pdb_info)
        parsed_pdb._ifg_atom_density.append(_ifg_atom_density)
        if any(_ifg_contact_water[1:]):
            parsed_pdb._ifg_contact_water.append(_ifg_contact_water)
        if any(_ifg_contact_ligand[1:]):
            parsed_pdb._ifg_contact_ligand.append(_ifg_contact_ligand)
        if any(_ifg_contact_metal[1:]):
            parsed_pdb._ifg_contact_metal.append(_ifg_contact_metal)
        if any(_ifg_hbond_water[1:]):
            parsed_pdb._ifg_hbond_water.append(_ifg_hbond_water)
        if any(_ifg_hbond_ligand[1:]):
            parsed_pdb._ifg_hbond_ligand.append(_ifg_hbond_ligand)



















