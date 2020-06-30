import sys
sys.path.append('/Users/npolizzi/Projects/design/Combs/')
import combs


def main():
    path = '/Users/npolizzi/Desktop/test/'
    pdb = '1mft.pdb'
    metals_df = combs.database.pdbheader.parse_metal_contacts(path + pdb)
    link_df = combs.database.pdbheader.parse_link(path + pdb)
    print()

if __name__ == "__main__":
    main()