import sys
sys.path.append('/Users/npolizzi/Projects/design/Combs/')
import combs
import gzip
import pickle


def main():
    path = '/Users/npolizzi/Desktop/test/bios/'
    pdb_file = path + '3bca_H_1.pdb'
    with gzip.open(path + 'pdb.gz', 'rb') as infile:
        pdb, d = pickle.load(infile)
    segnames = ['B','D'] #list(sorted(set(pdb.ca.getSegnames())))
    probe_df1 = combs.database.probe.parse_probe(pdb_file, segnames[0], segnames[1])
    probe_df2 = combs.database.probe.parse_probe(pdb_file, segnames[0])
    cols = probe_df2.columns
    probe_df2_ = probe_df2[cols[1:]].drop_duplicates()
    print()


if __name__ == "__main__":
    main()