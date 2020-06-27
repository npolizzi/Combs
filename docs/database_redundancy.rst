Database Redundancy
++++++++++++++++++++

There are several issues to consider concerning redundancy
when generating vdMs.  Redundancy in the PDB is typically
treated at the chain level.  Sequences of protein chains are
clustered at some homology level after pairwise alignments.
vdMs may consist of 100s of members in the best cases; average
cluster sizes may be in the 10s.  So dealing with redundancy
becomes very important in order to distinguish signal from noise.
Some chains in the PDB are crystallized as fusion proteins with
common proteins to aid in crystallization, expression, purification.
For example, GFP, MBP, cytb562 are commonly found fused to protein
chains by short linkers.  This complicates handling redundancy at the
chain level.  One would need a complete list of all fusion proteins
and then check every chain to see if it contains one of these.
While this is possible to some extent, it is not a perfect method.
Another complication would be to consider all the protein-protein
interfaces that would be missed if only considering one
representative of protein A, if protein A is also found in
structures bound to protein B, C, D, etc.  A wealth of information
is thrown out from structures AB, AC, AD.  No approach will be
perfect, but some might be more general than others. Consider
also the creation of vdMs from bound ligands in co-crystal
structures.  If two crystal structures, say proteins A and B,
contain the same bound ligand, and sequences of A and B are 99%
similar to each other, then A or B will be omitted in a
non-redundant dataset.  But what if the sequences of A and B only
differed at the binding site, which is the area of importance for
vdM generation anyway?  We should consider a method for removing
redundancy that handles this case.

I propose the following method:
    1. Collect all crystal structures with resolution <= 2.8 A and Robs <= 0.3.
    2. Rank these structures in the PDB by 1/resolution - Robs.
    3. Generate all biounits for a given structure, starting from the best-ranked structures.  Also generate DSSP, phenix.ramalyze, phenix.rotalyze, and phenix.real_space_correlation, sasa files for the protein.
    4. Within each biounit, search for the "ligand" and all residues contacting the ligand.  If the contacting residue and the "ligand" have occupancy 1, 1 < b factor <= 40, rscc > 0.7, sigma(2Fo-dFc) > 1.1, no clashes, "and all backbone atoms modeled, achieved indirectly by ensuring that phi, psi, omega, and tau were defined for each residue" (criteria from Richardson ultimate rotamer library, which would exclude terminal amino acids).
    5. A. If the ligand is part of a protein (e.g. carboxamide of ASN), obtain a stretch of sequence that is +- 7 residues of the contacting residue; obtain a similar stretch of sequence flanking the ligand.
       B. If the ligand is not part of a protein (e.g. a bound sugar), then obtain a stretch of sequence that is +- 12 residues of the contacting residue.  (I will admit that the choice of sequence length here is a bit arbitrary, which makes me uncomfortable.)
    6. Concatenate a string of the sequences, linked by a "-" between contacting residue seq and iFG seq.  Represent any missing N or C term residues with an underscore "_".
    7. Query the concatenated string in a dictionary (hash table).  If the string exists in the dictionary, the interaction is already represented in a better-resolution structure and the query is considered redundant and discarded.  This assumes that the matches are structurally similar.  One could do a superposition at this step to ensure this is the case.  If the structures are different, keep the query and the unique structures would have to be stored for further queries.  If one residue is different in a 25 residue query, it would be kept, amounting to a redundancy of 25/25 = 96%.  Or for 29/30 of 96.7%.  The data structure could be a dictionary that holds sequences as keys and a list of coordinate arrays as values.  Coordinates could be Calpha or Calpha plus sidechain/ligand coords.

Can get rid of redundant structures by clustering sequences by chain at 100% identity, then creating biounits for each structure within that cluster, then comparing all biounits.  Take as representative structure that with highest 1/res - Robs.




