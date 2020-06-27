__all__ = ['Comb']

import os
import re
from contextlib import ContextDecorator
import csv
from collections import defaultdict
from ..apps.constants import one_letter_code
import prody as pr
import numpy as np
from scipy.spatial.distance import cdist


class Comb(ContextDecorator):
    """
    :arg ifg_seq_str (optional): a regular expression (re) of type [re1] or [re1][re2] that defines a sequence with  
        total characters = 1 or 2. This should be defined if more than 1 amino acid is part of the iFG selection. 
        examples:
            ifg_seq_str = '[QN]' is a one-amino-acid selection of each ASN and GLN in the protein. 
            ifg_seq_str = '.P' will select all 2 residue sequences with an amino acid
                preceding a PRO. 
        If more than one re is required, it should be in list format.
        example:
            ifg_seq_str = ['[AL][WY]', '[VI]F'] will select all 2 residue sequences with ALA or LEU preceding
                a TRP or TYR, and will also selection all 2 residue sequences with VAL and ILE preceding a PHE.  
    :type ifg_seq_str: str or list of str
    :arg ifg_sele_dict: a dictionary with keys as numbers and values as dictionaries with keys as resnames and values 
        as atom names for the corresponding residue in the sequence selection string
         examples: 
            ifg_sele_dict = {1: {'ASN':'CB CG OD1 ND2', 'GLN':'CG CD OE1 NE2'}} 
                This ifg_sele_dict corresponds to the ifg_seq_str: '[QN]'.  As here, in the case where the ifg_seq_str 
                contains only one character, the ifg_sele_dict can be defined as:
                    ifg_sele_dict = {'ASN':'CB CG OD1 ND2', 'GLN':'CG CD OE1 NE2'}
            ifg_sele_dict = {1: {'ANY':'C O CA'}, 2: {'PRO':'N'}} where 'ANY' is the abbreviation for any amino acid.  
                This ifg_sele_dict corresponds to the ifg_seq_str: '.P',
                where 'ANY' is the abbreviation for any amino acid.  
                This ifg_sele_dict corresponds to the ifg_seq_str: ['[AL][WY]', '[VI]F'].
            ifg_sele_dict = {1: {'ALA': 'C O CA', 'LEU': 'C O CA'}, 2: {'TRP': 'N', 'TYR': 'N'}, 3: {'VAL': 'C CA CB',
                                                                    'ILE': 'C CA CB'}, 4: {'PHE': 'CG'}}                                                                        
    
    :arg path_to_pdb_chain_file: path to a txt file that has 5 letter pdb accession and unique chain on each line
    :type path_to_pdb_chain_file: str
    :arg file_tag: user-defined tag on the pdb and csv files that get printed
    :type file_tag: str
    :arg input_dir_pdb: path to input directory for pdbs to be combed.
    :type input_dir_pdb: str
    :arg output_dir_pdb: path to output directory for vandermer pdbs. Default is current working directory.
    :type output_dir_pdb: str
    :arg output_dir_csv: path to output directory for comb csv database files. Default is current working 
        directory.
    :type output_dir_csv: str
    :arg ifg_count: the unique number assigned to each iFG. The n vanderMers have numbers 1 through n for each iFG.
    :type increment: int
    :attribute pdb_chains: each unique chain associated with a pdb to be searched through to find iFGs. This is a built
    attribute that requires the path_to_pdb_chain_file and needs no other user input.
    :type pdb_chains: dict with keys=pdb accession code, values=unique chain
    :arg add_non_canonical: dictionary with keys as three letter resname strings, values as a one-letter string.
    :type add_non_canonical: dictionary
    :arg path_to_reduce: path to the reduce program
    :arg reduce: name of reduce executable
    
    """

    def __init__(self, ifg_sele_dict, **kwargs):
        """param kwargs: """
        _cwd = os.getcwd()
        if kwargs.get('add_non_canonical') is not None: # I think this takes care of issue with finding iFGs in ParsePDB
            #doesn't fix unless there is a valid one letter code!
            add_dict = kwargs.get('add_non_canonical')
            for three_let, one_let in add_dict.items():
                one_letter_code[three_let] = one_let

        path_to_pdb_chain_file = kwargs.get('path_to_pdb_chain_file', _cwd)
        ifg_seq_str = kwargs.get('ifg_seq_str')
        if ifg_seq_str is None: #this is only valid for a one-residue ifg_sele_dict such as
                                # {'ASN':'CB CG OD1 ND2', 'GLN':'CG CD OE1 NE2'}
            self.ifg_seq_str = '[' + ''.join([one_letter_code[three_letter_code] for three_letter_code in
                                        ifg_sele_dict.keys()]) + ']'
        else:
            self.ifg_seq_str = ifg_seq_str
        self.num_res_ifg = len(re.sub('\[.*\]', 'c',  self.ifg_seq_str))
        if any([isinstance(val, dict) for val in ifg_sele_dict.values()]):
            self.ifg_sele_dict = ifg_sele_dict
        else:
            self.ifg_sele_dict = {1: ifg_sele_dict}

        for dict_ in self.ifg_sele_dict.values():
            if 'ANY' in dict_.keys():
                for key in one_letter_code.keys():
                    dict_[key] = dict_['ANY']
        try:
            self.pdbs_chains = self.make_pdb_chain_dict(path_to_pdb_chain_file)
        except:
            self.pdbs_chains = None
        self.file_tag = kwargs.get('file_tag', 'comb')
        self.ifg_count = 1
        self.total_possible_ifgs = 0
        self.input_dir_pdb = kwargs.get('input_dir_pdb', _cwd)
        if self.input_dir_pdb[-1] != '/':
            self.input_dir_pdb += '/'
        self.input_dir_dssp = kwargs.get('input_dir_dssp', _cwd)
        if self.input_dir_dssp[-1] != '/':
            self.input_dir_dssp += '/'
        self.output_dir_pdb = kwargs.get('output_dir_pdb', _cwd)
        if self.output_dir_pdb[-1] != '/':
            self.output_dir_pdb += '/'
        self.output_dir_csv = kwargs.get('output_dir_csv', _cwd)
        if self.output_dir_csv[-1] != '/':
            self.output_dir_csv += '/'
        self.path_to_reduce = kwargs.get('path_to_reduce', _cwd)
        if self.path_to_reduce[-1] != '/':
            self.path_to_reduce += '/'
        self.probe_path = kwargs.get('probe_path', None)
        if self.probe_path is not None and self.probe_path[-1] != '/':
            self.probe_path += '/'
        self.reduce = kwargs.get('reduce', 'reduce')
        self.vandarotamer_dict = kwargs.get('vandarotamer_dict', None)
        self.radius1 = kwargs.get('radius1', 3.5)
        self.radius2 = kwargs.get('radius2', 4.8)
        self.probe_ifgatoms = kwargs.get('probe_ifgatoms', None)
        self.query_path = kwargs.get('query_path', None)
        self.scratch = kwargs.get('scratch', None)
        if self.query_path:
            self.query_coords = []
            self.query_cyclic = kwargs.get('query_cyclic', [False, False])
            self.query_lig_corr = kwargs.get('query_lig_corr', [None, None])
            self.ifg_seq_str_query = kwargs.get('ifg_seq_str_query', [])
            self.num_res_ifg_query = len(re.sub('\[.*\]', 'c', self.ifg_seq_str_query))
            self.query_names = kwargs.get('query_names', None)
            self.ifg_sele_dict_query = kwargs.get('ifg_sele_dict_query', None)
            if self.ifg_sele_dict_query:
                for dict_ in self.ifg_sele_dict_query.values():
                    if 'ANY' in dict_.keys():
                        for key in one_letter_code.keys():
                            dict_[key] = dict_['ANY']
            self.rmsd_threshold = kwargs.get('rmsd_threshold', None)
            self.query = None
            self.query_coords = None
            self.query = pr.parsePDB(self.query_path)
            self.make_query_coords()

    @staticmethod
    def make_pdb_chain_dict(path_to_pdb_chain_file_):
        with open(path_to_pdb_chain_file_) as infile:
            pdb_chain_dict = defaultdict(list)
            for line in infile:
                try:
                    pdb = line[0:4].lower()
                    chain = line[4]
                    pdb_chain_dict[pdb].append(chain)
                except Exception:
                    print('This pdb was not included from txt file: ', line)
        return pdb_chain_dict

    def __enter__(self):
        fieldnames_ifg_pdb = ['iFG_count', 'pdb', 'resname', 'resnum', 'resindex', 'chid', 'segi', 'atom_names', 'sequence',
                              'frag_length', 'sec_struct_dssp', 'sec_struct_phi_psi', 'rotamer']
        self._csvfile_ifg_pdb_info, self._csvwriter_ifg_pdb_info = self.init_csv('ifg_pdb_info', fieldnames_ifg_pdb)

        fieldnames_vdm_pdb = ['iFG_count', 'vdM_count', 'resname', 'resnum', 'resindex', 'chid', 'segi', 'atom_names',
                              'sequence', 'frag_length', 'rel_resnums', 'sec_struct_dssp', 'sec_struct_phi_psi',
                              'sec_struct_phi_psi_frag', 'rotamer', 'vandarotamer']
        self._csvfile_vdm_pdb_info, self._csvwriter_vdm_pdb_info = self.init_csv('vdm_pdb_info', fieldnames_vdm_pdb)

        fieldnames_ifg_contact_vdm = ['iFG_count', 'vdM_count', 'number_contacts', 'number_contacts_per_res',
                                      'atom_names', 'resnames', 'resnums', 'dist_info', 'ifg_probe_contacts',
                                      'probe_contact_pairs']
        self._csvfile_ifg_contact_vdm, self._csvwriter_ifg_contact_vdm = \
            self.init_csv('ifg_contact_vdm', fieldnames_ifg_contact_vdm)

        fieldnames_ifg_contact_water = ['iFG_count', 'number_contacts', 'number_contacts_per_res',
                                      'atom_names', 'resnames', 'resnums', 'dist_info']
        self._csvfile_ifg_contact_water, self._csvwriter_ifg_contact_water = \
            self.init_csv('ifg_contact_water', fieldnames_ifg_contact_water)

        fieldnames_ifg_contact_ligand = ['iFG_count', 'number_contacts', 'number_contacts_per_res',
                                        'atom_names', 'resnames', 'resnums', 'dist_info']
        self._csvfile_ifg_contact_ligand, self._csvwriter_ifg_contact_ligand = \
            self.init_csv('ifg_contact_ligand', fieldnames_ifg_contact_ligand)

        fieldnames_ifg_contact_metal = ['iFG_count', 'number_contacts', 'number_contacts_per_res',
                                        'atom_names', 'resnames', 'resnums', 'dist_info']
        self._csvfile_ifg_contact_metal, self._csvwriter_ifg_contact_metal = \
            self.init_csv('ifg_contact_metal', fieldnames_ifg_contact_metal)


        if self.probe_path:
            fieldnames_ifg_hbond_vdm = ['iFG_count', 'vdM_count', 'probe_hbond_info']
            self._csvfile_ifg_hbond_vdm, self._csvwriter_ifg_hbond_vdm = \
                self.init_csv('ifg_hbond_vdm', fieldnames_ifg_hbond_vdm)

            fieldnames_ifg_hbond_water = ['iFG_count', 'probe_hbond_info']
            self._csvfile_ifg_hbond_water, self._csvwriter_ifg_hbond_water = \
                self.init_csv('ifg_hbond_water', fieldnames_ifg_hbond_water)

            fieldnames_ifg_hbond_ligand = ['iFG_count', 'probe_hbond_info']
            self._csvfile_ifg_hbond_ligand, self._csvwriter_ifg_hbond_ligand = \
                self.init_csv('ifg_hbond_ligand', fieldnames_ifg_hbond_ligand)

        else:
            fieldnames_ifg_hbond_vdm = ['iFG_count', 'vdM_count', 'number_hbonds', 'atom_names', 'resnames', 'resnums',
                                        'rel_resnums', 'iFG_atom_names', 'vdM_atom_names', 'angle', 'distance_acc_hyd',
                                        'distance_heavy']
            self._csvfile_ifg_hbond_vdm, self._csvwriter_ifg_hbond_vdm = \
                self.init_csv('ifg_hbond_vdm', fieldnames_ifg_hbond_vdm)

            fieldnames_ifg_hbond_water = ['iFG_count', 'number_hbonds', 'atom_names', 'resnames', 'resnums',
                                        'angle', 'distance_acc_hyd', 'distance_heavy']
            self._csvfile_ifg_hbond_water, self._csvwriter_ifg_hbond_water = \
                self.init_csv('ifg_hbond_water', fieldnames_ifg_hbond_water)

            fieldnames_ifg_hbond_ligand = ['iFG_count', 'number_hbonds', 'atom_names', 'resnames', 'resnums',
                                          'angle', 'distance_acc_hyd', 'distance_heavy']
            self._csvfile_ifg_hbond_ligand, self._csvwriter_ifg_hbond_ligand = \
                self.init_csv('ifg_hbond_ligand', fieldnames_ifg_hbond_ligand)

            fieldnames_ifg_ca_hbond_vdm = ['iFG_count', 'vdM_count', 'number_hbonds', 'atom_names', 'resnames', 'resnums',
                                           'rel_resnums', 'iFG_atom_names', 'vdM_atom_names', 'angle', 'distance_acc_hyd',
                                           'distance_heavy']
            self._csvfile_ifg_ca_hbond_vdm, self._csvwriter_ifg_ca_hbond_vdm = \
                self.init_csv('ifg_ca_hbond_vdm', fieldnames_ifg_ca_hbond_vdm)

        fieldnames_ifg_atom_density = ['iFG_count', 'min_hull_dist_CB_CA', 'cBeta_density',
                                       'heavy_atom_density_5A', 'heavy_atom_density_10A',
                                       'iFG_sasa', 'iFG_sasa_full_residue', 'iFG_sasa_dssp_full_residue',
                                       'iFG_sasa_CB_3A_probe', 'iFG_sasa_CB_4A_probe', 'iFG_sasa_CB_5A_probe']
        self._csvfile_ifg_atom_density, self._csvwriter_ifg_atom_density = \
            self.init_csv('ifg_atom_density', fieldnames_ifg_atom_density)

        fieldnames_vdm_sasa_info = ['iFG_count', 'vdM_count', 'min_hull_dist_CB_CA', 'vdM_sasa_full_residue',
                                    'vdM_sasa_dssp_full_residue',
                                       'vdM_sasa_CB_3A_probe', 'vdM_sasa_CB_4A_probe', 'vdM_sasa_CB_5A_probe']
        self._csvfile_vdm_sasa_info, self._csvwriter_vdm_sasa_info = \
            self.init_csv('vdm_sasa_info', fieldnames_vdm_sasa_info)

        return self

    def init_csv(self, filename, _fieldnames):
        csvfile = open(self.output_dir_csv + self.file_tag + '_' + filename + '.csv', 'w')
        csvwriter = csv.writer(csvfile)
        csvwriter.writerow(_fieldnames)
        return csvfile, csvwriter

    def write_poss_ifg_csv(self, filename, _fieldnames):
        file, writer = self.init_csv(filename, _fieldnames)
        writer.writerow([self.total_possible_ifgs])
        file.close()

    def make_query_coords(self):
        q1_coords = [self.query.select('name ' + n).getCoords()[0] for n in self.query_lig_corr[0]]
        if self.query_cyclic[0]:
            len_coords = len(q1_coords)
            q_sel1_coords = [[q1_coords[i - j] for i in range(len_coords)] for j in range(len_coords)]
        else:
            q_sel1_coords = [q1_coords]

        q2_coords = [self.query.select('name ' + n).getCoords()[0] for n in self.query_lig_corr[1]]
        if self.query_cyclic[1]:
            len_coords = len(q2_coords)
            q_sel2_coords = [[q2_coords[i - j] for i in range(len_coords)] for j in range(len_coords)]
        else:
            q_sel2_coords = [q2_coords]

        com = pr.calcCenter(self.query.select('name ' + ' '.join(self.query_lig_corr[0])))
        self.query_distance = np.max(cdist([com], q_sel2_coords[0])) + self.rmsd_threshold

        superpose_list = []
        for q1 in q_sel1_coords:
            for q2 in q_sel2_coords:
                superpose_list.append(np.vstack((q1, q2)))
        self.query_coords = superpose_list

    def __exit__(self, *exc):
        self._csvfile_ifg_pdb_info.close()
        self._csvfile_vdm_pdb_info.close()
        self._csvfile_ifg_contact_vdm.close()
        self._csvfile_ifg_contact_water.close()
        self._csvfile_ifg_contact_ligand.close()
        self._csvfile_ifg_contact_metal.close()
        self._csvfile_ifg_hbond_vdm.close()
        self._csvfile_ifg_hbond_water.close()
        self._csvfile_ifg_hbond_ligand.close()
        self._csvfile_ifg_atom_density.close()
        return False



