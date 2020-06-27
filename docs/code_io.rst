CODE I/O
++++++++

Input (common usage)
---------------------

    * *protein file*
        :type: pdb, pdb.gz

        - backbone only
        - backbone with sidechains

    * *ligand file*
        :type: pdb, mol2

        - fragment names to map vdMs (automated vs manual)

    * *names of residues to map vdMs*
        :type: str, txt

        - residue, chain, segment id

    * *options*
        - RMSD cutoff for chemical groups
        - kinds of vdMs to load at each position
            * aa types
            * chemical group fragment types
            * H-bond only vdMs
            * vdM cluster score cutoff
            * vdM secondary structure score cutoff
            * limit vdMs by rotamer
                - use input rotamer only
                - use particular rotamer (phenix naming
                  convention)
            * topN cutoff for outputted vdMs.  This likely
              won't increase speed of algorithm though
        - output shells
            * first_shell
            * second_shell, etc.
        - design constraints
        - different kinds of scoring fns?
        - score only existing interactions (e.g. Tyr8 interaction
          with His20, checking for Tyr vdM of imidazole and His vdM
          of phenol.)

Need a way to automate fragment generation of ligand so that fragments
are generated to map onto existing vdM chemical groups.  RDkit might be
of use here.  Some fragment generators are BRICS and RECAP.


Output (common usage)
---------------------

    * pdb with best vdMs (w/ or w/o networks) and ligand location
        :type: pdb, pdb.gz

    * score file
        :type: txt, xlsx, dataframe (pkl)

        - list of allowed vdMs at each position with score
        - overall score of pose

    * aggregate pdb with allowed vdMs
        :type: pdb, pdb.gz

        - output cluster centroids only
            * cluster RMSD threshold

    * aggregate pdb with allowed ligand positions
        :type: pdb, pdb.gz

        - output cluster centroids only
            * cluster RMSD threshold

