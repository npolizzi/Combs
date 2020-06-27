import sys
sys.path.append('~/combs/src/')
import combs
import os

try:
    output_dir_pdb = 'output_directory_for_vdM_pdbs'
    os.makedirs(output_dir_pdb)
except:
    pass

try:
    output_dir_csv = 'output_directory_for_csv_file'
    os.makedirs(output_dir_csv)
except:
    pass

ifg_dict = {'ASN': 'CB CG ND2 OD1'}
probe_ifg_dict = {'ASN': '(atomHD21,atomHD22,atom_ND2,atom_HB,atom_OD1,atom_CB_,atom_CG_)'}
vandarotamer_dict = {'A1': 'OD1', 'A2': 'CG', 'A3': 'ND2'}
kwargs = {'file_tag': 'asparagine',
          'input_dir_pdb': 'path_to_pdb_directory',
          'input_dir_dssp': 'path_to_dssp_directory',
          'output_dir_pdb': output_dir_pdb,
          'output_dir_csv': output_dir_csv,
          'path_to_pdb_chain_file': 'path_to_text_file_of_nonredundant_PDB_chain',
          'probe_path': 'path_to_probe_program',
          'input_dir_all_ala': 'path_to_allAla_pdb_directory',
	      'path_to_reduce': 'path_to_reduce_program',
	      'reduce': 'reduce.3.23.130521.macosx',
          'probe_ifgatoms': probe_ifg_dict,
          'vandarotamer_dict': vandarotamer_dict
          }
combs.run_comb(ifg_dict, **kwargs)
