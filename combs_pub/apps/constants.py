__all__ = ['one_letter_code', 'resnames_aa_20', 'resnames_aa_20_join']

import collections

one_letter_code = {'CYS': 'C', 'ASP': 'D', 'SER': 'S', 'GLN': 'Q', 'LYS': 'K',
                   'ILE': 'I', 'PRO': 'P', 'THR': 'T', 'PHE': 'F', 'ASN': 'N',
                   'GLY': 'G', 'HIS': 'H', 'LEU': 'L', 'ARG': 'R', 'TRP': 'W',
                   'ALA': 'A', 'VAL': 'V', 'GLU': 'E', 'TYR': 'Y', 'MET': 'M',
                   'MSE': 'm', 'ANY': '.', 'FE': 'fe', 'ZN': 'zn', 'HEM': 'h'}

resnames_aa_20 = ['CYS', 'ASP', 'SER', 'GLN', 'LYS',
                   'ILE', 'PRO', 'THR', 'PHE', 'ASN',
                   'GLY', 'HIS', 'LEU', 'ARG', 'TRP',
                   'ALA', 'VAL', 'GLU', 'TYR', 'MET',
                   'MSE']

resnames_aa_20_join = 'CYS ASP SER GLN LYS ILE PRO THR PHE ASN GLY ' \
                      'HIS LEU ARG TRP ALA VAL GLU TYR MET MSE'

interactamer_atoms = collections.defaultdict(dict)
interactamer_atoms['HIS']['Delta'] = ['ND1', 'CG', 'CB']
interactamer_atoms['HIS']['Epsilon'] = ['NE2', 'CD2', 'CE1']
interactamer_atoms['LYS']['Amino'] = ['CE', 'CD', 'NZ']
interactamer_atoms['ASP']['Carboxylate'] = ['CG', 'OD1', 'OD2']
interactamer_atoms['GLN']['Carboxamide'] = ['CD', 'OE1', 'NE2']
interactamer_atoms['GLU']['Carboxylate'] = ['CD', 'OE1', 'OE2']
interactamer_atoms['ASN']['Carboxamide'] = ['CG', 'OD1', 'ND2']
interactamer_atoms['ALA']['Methyl'] = ['C', 'CA', 'CB']
interactamer_atoms['ARG']['Guano'] = ['CZ', 'NH2', 'NH1']
interactamer_atoms['THR']['Alcohol'] = ['OG1', 'CB', 'CG2']
interactamer_atoms['GLY']['MainChain'] = ['CA', 'N', 'C']
interactamer_atoms['TYR']['PhenolOH'] = ['CZ', 'CE1', 'OH']
interactamer_atoms['SER']['Alcohol'] = ['CB', 'OG', 'CA']
interactamer_atoms['TRP']['IndoleNH'] = ['NE1', 'CD1', 'CE2']


dict_ = {'LYS': ['CE', 'CD', 'NZ'],
          'ASP': ['CG', 'OD1', 'OD2'],
          'PHE': ['CZ', 'CE1', 'CE2'],
          'ASN': ['CG', 'OD1', 'ND2'],
          'GLN': ['CD', 'OE1', 'NE2'],
          'ALA': ['C', 'CA', 'CB'],
          'ARG': ['CZ', 'NH2', 'NH1'],
          'THR': ['OG1', 'CB', 'CG2'],
          'GLY': ['CA', 'N', 'C'],
          'TYR': ['CZ', 'CE1', 'OH'],
          'LEU': ['CG', 'CD1', 'CD2'],
          'VAL': ['CB', 'CG1', 'CG2'],
          'GLU': ['CD', 'OE1', 'OE2'],
          'PRO': ['CB', 'CG', 'CD'],
          'SER': ['CB', 'OG', 'CA'],
          'CYS': ['CB', 'SG', 'CA'],
          'MET': ['SD', 'CG', 'CE'],
          'TRP': ['NE1', 'CD1', 'CE2'],
          'ILE': ['CG1', 'CD1', 'CG2'],
          }


flip_names = {'PHE': [('CE1', 'CE2'), ('CD1', 'CD2')],
              'ASP': [('OD1', 'OD2')],
              'GLU': [('OE1', 'OE2')],
              'ARG': [('NH1', 'NH2')],
              'TYR': [('CE1', 'CE2'), ('CD1', 'CD2')]
              }

flip_residues = ['PHE', 'ASP', 'GLU', 'ARG', 'TYR']

flip_sets = [{'OD1', 'OD2'}, {'CE1', 'CE2'}, {'NH2', 'NH1'}, {'OE1', 'OE2'}, {'CD1', 'CD2'}]

bb_type_dict = {'N_CA': ['N', 'H', 'CA'], 'C_O': ['C', 'O', 'CA'], 'SC': ['CA', 'N', 'C'],
                'PHI_PSI': ['CA', 'N', 'C']}

residue_sc_names = {'ALA': ['CB'], 'CYS': ['CB', 'SG'], 'ASP': ['CB', 'CG', 'OD1', 'OD2'],
                    'ASN': ['CB', 'CG', 'OD1', 'ND2'], 'VAL': ['CB', 'CG1', 'CG2'],
                    'GLU': ['CB', 'CG', 'CD', 'OE1', 'OE2'], 'LEU': ['CB', 'CG', 'CD1', 'CD2'],
                    'HIS': ['CB', 'CG', 'ND1', 'CE1', 'NE2', 'CD2'],
                    'ILE': ['CB', 'CG2', 'CG1', 'CD1'], 'MET': ['CB', 'CG', 'SD', 'CE'],
                    'TRP': ['CB', 'CG', 'CD1', 'NE1', 'CE2', 'CD2', 'CE3', 'CZ3', 'CH2', 'CZ2'],
                    'SER': ['CB', 'OG'], 'LYS': ['CB', 'CG', 'CD', 'CE', 'NZ'],
                    'PHE': ['CB', 'CG', 'CD1', 'CD2', 'CE1', 'CE2', 'CZ'], 'PRO': ['CB', 'CG', 'CD'],
                    'GLY': [], 'THR': ['CB', 'OG1', 'CG2'],
                    'TYR': ['CB', 'CG', 'CD1', 'CD2', 'CE1', 'CE2', 'CZ', 'OH'],
                    'GLN': ['CB', 'CG', 'CD', 'OE1', 'NE2'],
                    'ARG': ['CB', 'CG', 'CD', 'NE', 'CZ', 'NH1', 'NH2']}


