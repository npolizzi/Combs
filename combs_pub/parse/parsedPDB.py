__all__ = ['ParsedPDB']

import prody as pr
import numpy as np
import os
import freesasa
from ..apps.transformation import get_rot_trans
from ..apps.convex_hull import AlphaHull


class ParsedPDB:
    """A class of protein objects parsed via prody, with the following attributes:
    
    contacts: prody contact object describing distances between atoms of the pdb
    fs_struct: freesasa structure for solvent-exposed surface area (sasa) calcs
    fs_result: freesasa result for sasa structure
    dssp: dssp results of secondary structure assignments of the pdb
    prody_pdb: prody parsed pdb object
    possible_ifgs: a list of prody selections of all possible iFGs of the pdb object
    pdb_acc_code: pdb accession code
    pdb_chain: chain to consider for iFG selections
    
    Class ParsedPDB has the following methods:
    write_csv:  writes the data described below to csv file
    find_ifgs: finds iFGs
    
    The following are lists to be printed to csv file:
    
    ifg_vdm_data_pdb_info: includes residue names, numbers, chain ids, vdm length, and secondary structure info
    ifg_data_valence_total: includes valence info of iFG in context of entire protein, with breakdown by atom type.  
    ifg_data_valence_vdm_fragment: includes valence info of iFG in context of vandermer fragment, with breakdown by 
        atom type
    ifg_data_valence_vdm_iresidue: includes valence info of iFG in context of vandermer central residue 
        (ie, the i residue), with breakdown by atom type
    ifg_data_hbond_total:  
    ifg_data_hbond_vdm_fragment:
    ifg_data_hbond_vdm_iresidue:
    ifg_data_atom_density:
    
    """

    def __init__(self, comb, pdb_acc_code, chain, **kwargs):
        """ :comb: arg: instance of cls Comb with attributes pdbchain_dict, ifg_selection_info
        :pdb_acc_code: type: str: 4 character pdb accession code
        :param kwargs: 
            path_to_pdb
            path_to_dssp 
        """
        #search for acc code in input_dir_pdb from comb object.
        assert isinstance(pdb_acc_code, str), 'PDB accession code needs to be a string'
        pdb_file = [file.name for file in os.scandir(comb.input_dir_pdb) if pdb_acc_code in file.name]
        try:
            if pdb_file:
                pdb_file = pdb_file[0]
                self.prody_pdb = pr.parsePDB(comb.input_dir_pdb + pdb_file, altloc='A', model=1)
            elif 'path_to_pdb' in kwargs:
                self.prody_pdb = pr.parsePDB(kwargs.get('path_to_pdb'), altloc='A', model=1)
            else:
                try:
                    os.mkdir(comb.input_dir_pdb + 'raw')
                    os.mkdir(comb.input_dir_pdb + 'reduce')
                except:
                    pass
                pr.fetchPDB(pdb_acc_code, compressed=False, folder=comb.input_dir_pdb + 'raw')
                os.system(comb.path_to_reduce + comb.reduce + ' -FLIP -Quiet -DB ' + comb.path_to_reduce
                          + 'reduce_wwPDB_het_dict.txt ' + comb.input_dir_pdb + 'raw/' + pdb_acc_code.lower()
                          + '.pdb > ' + comb.input_dir_pdb + 'reduce/' + pdb_acc_code.lower() + 'H.pdb')
                self.prody_pdb = pr.parsePDB(comb.input_dir_pdb + 'reduce/' + pdb_acc_code.lower() + 'H.pdb',
                                             altloc='A', model=1)
        except NameError:
            raise NameError('ParsePDB instance needs a pdb file path or a valid pdb accession code.')

        self.pdb_acc_code = pdb_acc_code.lower()
        self.pdb_chain = chain
        if len(self.prody_pdb) == len(self.prody_pdb.select('icode _')) \
                and self.prody_pdb.select('protein and chain ' + self.pdb_chain) is not None:
            self.contacts = pr.Contacts(self.prody_pdb)
            self.set_bonds()

            if pdb_file:
                self.fs_struct = freesasa.Structure(comb.input_dir_pdb + pdb_file)
            elif 'path_to_pdb' in kwargs:
                self.fs_struct = freesasa.Structure(kwargs.get('path_to_pdb'))
            else:
                path = comb.input_dir_pdb + 'reduce/'
                self.fs_struct = freesasa.Structure(path + next(file.name for file in os.scandir(path) if
                                                                 self.pdb_acc_code in file.name))

            self.fs_result = freesasa.calc(self.fs_struct)

            self.fs_result_cb_3A = self.freesasa_cb(probe_radius=3)
            self.fs_result_cb_4A = self.freesasa_cb(probe_radius=4)
            self.fs_result_cb_5A = self.freesasa_cb(probe_radius=5)
            self.prody_pdb_bb_cb_atom_ind = self.prody_pdb.select('protein and (backbone or name CB) '
                                                                  'and not element H D').getIndices()

            dssp_file = [file.name for file in os.scandir(comb.input_dir_dssp) if pdb_acc_code in file.name]
            if dssp_file:
                dssp_file = dssp_file[0]
                self.parse_dssp(dssp_file, comb)

            if comb.query_path:
                self.possible_ifgs = self.find_possible_ifgs_rmsd(comb, rmsd_threshold=comb.rmsd_threshold)
            else:
                self.possible_ifgs = self.find_possible_ifgs(comb)
        else:
            self.possible_ifgs = None

        self.alphahull = self.set_alphahull()
        self.segnames = sorted(np.unique(self.prody_pdb.getSegnames()))
        self._ifg_pdb_info = []
        self._ifg_atom_density = []
        self._ifg_contact_water = []
        self._ifg_contact_ligand = []
        self._ifg_contact_metal = []
        # vdM specific:
        self._vdm_pdb_info = []
        self._vdm_sasa_info = []
        self._ifg_contact_vdm = []
        self._ifg_hbond_vdm = []
        self._ifg_hbond_water = []
        self._ifg_hbond_ligand = []
        self._ifg_ca_hbond_vdm = []

    def parse_dssp(self, dssp_file, comb):
        dssp_ss = {}
        dssp_sasa = {}
        with open(comb.input_dir_dssp + dssp_file, 'r') as infile:
            for line in infile:
                if line.startswith('  #  RESIDUE'):
                    break
            for line in infile:
                if line[13] == '!':
                    continue
                resnum = int(line[5:10])
                chain = line[11]
                if line[16] != ' ':
                    dssp_ss[(resnum, chain)] = line[16]
                else:
                    dssp_ss[(resnum, chain)] = '-'
                dssp_sasa[(resnum, chain)] = line[34:38].strip()
        self.dssp_sasa = dssp_sasa
        self.dssp_ss = dssp_ss

    def set_alphahull(self):
        segnames = ' '.join(set(self.prody_pdb.getSegnames()))
        pdb_alpha = self.prody_pdb.select('segment ' + segnames)
        ahull = AlphaHull(alpha=8)
        ahull.set_coords(pdb_alpha)
        ahull.calc_hull()
        return ahull

    def set_bonds(self):
        """Sets backbone bonds of chain based on proximity of atoms, used for vdM fragment selection."""
        # This needs to be for the whole protein because vdMs can reach across chains.
        bb_sel = self.prody_pdb.select('protein and name N C CA')
        dm = pr.buildDistMatrix(bb_sel)
        ind = np.where((np.tril(dm) < 1.7) & (np.tril(dm) > 0))
        atom_ind = bb_sel.getIndices()
        self.prody_pdb.setBonds([(atom_ind[i], atom_ind[j]) for i, j in zip(ind[0], ind[1])])

    def find_possible_ifgs(self, comb):
        """uses iFG definitions in comb object to select iFGs in the parsed protein object that have all atoms 
        and occupancies = 1.
        """
        possible_ifgs = []
        if comb.ifg_seq_str == 'element':
            poss_ifg_sel = self.prody_pdb.select(comb.ifg_sele_dict[1]['prody_selection'])
            if poss_ifg_sel is not None:
                for atom in poss_ifg_sel.iterAtoms():
                    if self.prody_pdb.select('segment A and chain ' + self.pdb_chain + ' within '
                                                     + str(comb.radius1) + ' of atom', atom=atom):
                        possible_ifgs.append(atom.select('all'))
                        comb.total_possible_ifgs += 1
            return possible_ifgs

        elif comb.num_res_ifg == 1:
            poss_ifg_sel = self.prody_pdb.select(
                'segment A and chain ' + self.pdb_chain + ' resname "' + list(comb.ifg_sele_dict[1].keys())[0] + '"')
            if poss_ifg_sel is not None:
                ifg_resindices, indices = np.unique(poss_ifg_sel.getResindices(), return_index=True)
                ifg_resnames = poss_ifg_sel.getResnames()[indices]

                for ifg_resindex, ifg_resname in zip(ifg_resindices, ifg_resnames):
                        ifg_selection = self.prody_pdb.select('resindex ' + str(ifg_resindex) + ' and name '
                                                              + comb.ifg_sele_dict[1][ifg_resname])
                        if ifg_selection is not None:
                            num_atoms = len(ifg_selection)
                            if num_atoms == len(comb.ifg_sele_dict[1][ifg_resname].split()):
                                if all(ifg_selection.getResnums() > 0):
                                    possible_ifgs.append(ifg_selection)
                comb.total_possible_ifgs += len(possible_ifgs)
            return possible_ifgs
        else:
            poss_ifg_sel = self.prody_pdb.select('segment A and chain ' + self.pdb_chain + ' sequence "' + comb.ifg_seq_str + '"')
            if poss_ifg_sel is not None:
                ifg_resindices_cat_list, indices = np.unique(poss_ifg_sel.getResindices(), return_index=True)
                ifg_resnames_cat_list = poss_ifg_sel.getResnames()[indices]
                ifg_resindex_pairs = [ifg_resindices_cat_list[i:i + 2] for i in range(0, len(ifg_resindices_cat_list), 2)]
                ifg_resname_pairs = [ifg_resnames_cat_list[i:i + 2] for i in range(0, len(ifg_resnames_cat_list), 2)]
                for ifg_resindex_pair, ifg_resname_pair in zip(ifg_resindex_pairs, ifg_resname_pairs):
                    resind1, resind2 = ifg_resindex_pair
                    resname1, resname2 = ifg_resname_pair
                    try:
                        ifg_selection = self.prody_pdb.select('(resindex ' + str(resind1) + ' and name '
                                                          + comb.ifg_sele_dict[1][resname1]+')'
                                                          + ' or (resindex ' + str(resind2) + ' and name '
                                                          + comb.ifg_sele_dict[2][resname2]+')')
                    except KeyError:
                        print('Non-canonical residue in iFG, skipping.')
                        ifg_selection = None
                    if ifg_selection is not None:
                        num_atoms = len(ifg_selection)
                        names = comb.ifg_sele_dict[1][resname1].split()
                        names.extend(comb.ifg_sele_dict[2][resname2].split())
                        if num_atoms == len(names):
                            if all(ifg_selection.getResnums() > 0):
                                possible_ifgs.append(ifg_selection)
                comb.total_possible_ifgs += len(possible_ifgs)
            return possible_ifgs

    def find_possible_ifgs_rmsd(self, comb, rmsd_threshold=1.0):
        """uses iFG definitions in comb object to select iFGs in the parsed protein object that have all atoms
        and occupancies = 1.
        """
        possible_ifgs = []
        if comb.num_res_ifg_query == 1:
            poss_ifg_sel = self.prody_pdb.select('segment A and chain ' + self.pdb_chain + ' sequence "' + comb.ifg_seq_str_query + '"')
            if poss_ifg_sel is not None:
                ifg_resindices, indices = np.unique(poss_ifg_sel.getResindices(), return_index=True)
                ifg_resnames = poss_ifg_sel.getResnames()[indices]

                for ifg_resindex, ifg_resname in zip(ifg_resindices, ifg_resnames):
                        ifg_selection = self.prody_pdb.select('resindex ' + str(ifg_resindex) + ' and name '
                                                              + comb.ifg_sele_dict_query[1][ifg_resname])
                        if ifg_selection is not None:
                            num_atoms = len(ifg_selection)
                            if num_atoms == len(comb.ifg_sele_dict_query[1][ifg_resname].split()):
                                if all(ifg_selection.getResnums() > 0):
                                    possible_ifgs.append(ifg_selection)
                comb.total_possible_ifgs += len(possible_ifgs)
        else:
            poss_ifg_sel = self.prody_pdb.select('segment A and chain ' + self.pdb_chain + ' sequence "' + comb.ifg_seq_str_query + '"')
            if poss_ifg_sel is not None:
                ifg_resindices_cat_list, indices = np.unique(poss_ifg_sel.getResindices(), return_index=True)
                ifg_resnames_cat_list = poss_ifg_sel.getResnames()[indices]
                ifg_resindex_pairs = [ifg_resindices_cat_list[i:i + 2] for i in range(0, len(ifg_resindices_cat_list), 2)]
                ifg_resname_pairs = [ifg_resnames_cat_list[i:i + 2] for i in range(0, len(ifg_resnames_cat_list), 2)]
                for ifg_resindex_pair, ifg_resname_pair in zip(ifg_resindex_pairs, ifg_resname_pairs):
                    resind1, resind2 = ifg_resindex_pair
                    resname1, resname2 = ifg_resname_pair
                    try:
                        ifg_selection = self.prody_pdb.select('(resindex ' + str(resind1) + ' and name '
                                                          + comb.ifg_sele_dict_query[1][resname1]+')'
                                                          + ' or (resindex ' + str(resind2) + ' and name '
                                                          + comb.ifg_sele_dict_query[2][resname2]+')')
                    except KeyError:
                        print('Non-canonical residue in iFG, skipping.')
                        ifg_selection = None
                    if ifg_selection is not None:
                        num_atoms = len(ifg_selection)
                        names = comb.ifg_sele_dict_query[1][resname1].split()
                        names.extend(comb.ifg_sele_dict_query[2][resname2].split())
                        if num_atoms == len(names):
                            if all(ifg_selection.getResnums() > 0):
                                possible_ifgs.append(ifg_selection)
                comb.total_possible_ifgs += len(possible_ifgs)

        passed_possible_ifgs = []
        for pifg in possible_ifgs:
            com = pr.calcCenter(pifg.select('name ' + ' '.join(comb.query_names[0])))
            q2_sel = self.prody_pdb.select('name ' + ' '.join(comb.query_names[1]) + ' within '
                                            + str(comb.query_distance) + ' of center', center=com)
            if q2_sel is not None:
                resinds_query2s = np.unique(q2_sel.getResindices())
                q_sel1_coords = [pifg.select('name ' + n).getCoords()[0] for n in comb.query_names[0]]
                for resind in resinds_query2s:
                    q_sel = self.prody_pdb.select('name ' + ' '.join(comb.query_names[1]) + ' and resindex ' + str(resind))
                    if len(q_sel) ==  len(comb.query_names[1]):
                        q_sel2_coords = [q_sel.select('name ' + n).getCoords()[0] for n in comb.query_names[1]]
                        pifg_coords = np.vstack((q_sel1_coords, q_sel2_coords))
                        for coords in comb.query_coords:
                            R, m_com, t_com = get_rot_trans(coords, pifg_coords)
                            coords_transformed = np.dot((coords - m_com), R) + t_com
                            rmsd = pr.calcRMSD(coords_transformed, pifg_coords)
                            if rmsd <= rmsd_threshold:
                                passed_possible_ifgs.append(q_sel)  # This only takes the query2 selection as the iFG.
                                break
        return passed_possible_ifgs


    def freesasa_cb(self, probe_radius=1.4):
        cb_sele = self.prody_pdb.select('protein and (backbone or name CB) and not element H D')
        coords = list(x for y in cb_sele.getCoords() for x in y)
        radii = list(freesasa.Classifier().radius(x, y) for x, y in zip(cb_sele.getResnames(), cb_sele.getNames()))
        return freesasa.calcCoord(coords, radii, freesasa.Parameters({'probe-radius': probe_radius}))

    def write_csv(self, comb):
        comb._csvwriter_ifg_pdb_info.writerows(self._ifg_pdb_info)
        comb._csvwriter_ifg_atom_density.writerows(self._ifg_atom_density)
        comb._csvwriter_vdm_pdb_info.writerows(self._vdm_pdb_info)
        comb._csvwriter_vdm_sasa_info.writerows(self._vdm_sasa_info)
        comb._csvwriter_ifg_contact_vdm.writerows(self._ifg_contact_vdm)
        comb._csvwriter_ifg_contact_water.writerows(self._ifg_contact_water)
        comb._csvwriter_ifg_contact_ligand.writerows(self._ifg_contact_ligand)
        comb._csvwriter_ifg_contact_metal.writerows(self._ifg_contact_metal)
        comb._csvwriter_ifg_hbond_vdm.writerows(self._ifg_hbond_vdm)
        comb._csvwriter_ifg_hbond_water.writerows(self._ifg_hbond_water)
        comb._csvwriter_ifg_hbond_ligand.writerows(self._ifg_hbond_ligand)
        if comb.probe_path == []:
            comb._csvwriter_ifg_ca_hbond_vdm.writerows(self._ifg_ca_hbond_vdm)





