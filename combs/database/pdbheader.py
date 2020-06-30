"""Functions for parsing REMARK 620 for metal coordination
and LINK for non-canonical linkages.  Probe does not behave
reliably for metal contacts, so these header sections are
substitutes for Probe output."""
import pandas as pd


#dtypes of dataframe columns in :func:`parse_metal_contacts`
metal_dtype_dict = {'interaction': 'category',
                    'resname1': 'category',
                    'chain1': 'category',
                    'resnum1': int,
                    'name1': 'category',
                    'resname2': 'category',
                    'chain2': 'category',
                    'resnum2': int,
                    'name2': 'category'}


# dataframe columns in :func:`parse_metal_contacts`
metal_col_names = list(metal_dtype_dict.keys())[1:]


def parse_620_block(stream):
    """"""
    metal_data = []
    stream.readline()
    metal_line = stream.readline().strip().split()
    metal_info = metal_line[2:]
    stream.readline()
    for line in stream:
        if line[:12] == 'REMARK 620 N':
            break
        contact_line = line.strip().split()
        contact_info = contact_line[3:7]
        metal_data.append(metal_info + contact_info)
    return metal_data


def parse_metal_contacts(pdb_file):
    """
    
    Parameters
    ----------
    pdb_file

    Returns
    -------

    """
    with open(pdb_file, 'r') as infile:
        metal_data = []
        for line in infile:
            if line[:10] == 'REMARK 620':
                break
        for line in infile:
            if line.strip() == 'REMARK 620':
                metal_data.extend(parse_620_block(infile))
    metal_df = pd.DataFrame(metal_data, columns=metal_col_names)
    metal_df['interaction'] = 'm'
    return metal_df.astype(dtype=metal_dtype_dict)


#dtypes of dataframe columns in :func:`parse_link`
link_dtype_dict = {'name1': 'category',
                   'atloc1': 'category',
                   'resname1': 'category',
                   'chain1': 'category',
                   'resnum1': int,
                   'icode1': 'category',
                   'name2': 'category',
                   'altloc2': 'category',
                   'resname2': 'category',
                   'chain2': 'category',
                   'resnum2': int,
                   'icode2': 'category',
                   'distance': float}


# dataframe columns in :func:`parse_link`
link_col_names = list(link_dtype_dict.keys())


def _parse_link_line(line):
    """"""
    ATOM1 = line[12:16].strip()
    ALTLOC1 = line[16]
    RESNAME1 = line[17:20].strip()
    CHAIN1 = line[21]
    RESNUM1 = line[22:26].strip()
    ICODE1 = line[26]
    ATOM2 = line[42:46].strip()
    ALTLOC2 = line[46]
    RESNAME2 = line[47:50].strip()
    CHAIN2 = line[51]
    RESNUM2 = line[52:56].strip()
    ICODE2 = line[56]
    DISTANCE = line[73:78].strip()
    return (ATOM1, ALTLOC1, RESNAME1, CHAIN1,
            RESNUM1, ICODE1, ATOM2, ALTLOC2,
            RESNAME2, CHAIN2, RESNUM2, ICODE2,
            DISTANCE)


def parse_link(pdb_file):
    with open(pdb_file, 'r') as infile:
        link_data = []
        for line in infile:
            if line[:4] != 'LINK':
                continue
            else:
                link_data.append(_parse_link_line(line))
    link_df = pd.DataFrame(link_data, columns=link_col_names)
    return link_df.astype(dtype=link_dtype_dict)