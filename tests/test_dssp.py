import sys
sys.path.append('/Users/npolizzi/Projects/design/Combs/')
import combs
import gzip
import pickle


def main():
    path = '/Users/npolizzi/Desktop/test/bios/'
    with gzip.open(path + 'pdb.gz', 'rb') as infile:
        pdb, d = pickle.load(infile)
    dssp = combs.database.dssp.parse_dssp(path + '3bca_H_1.dssp', pdb)
    print()

if __name__ == "__main__":
    main()