These are example scripts of how COMBS was used to design ABLE and LABLE proteins
to bind the ligand apixaban.  These scripts show asparagine as an example.
Carboxamide was comprised of ASN and GLN residues.  C=O was comprised of
Ala and Gly backbone.  The workflow in general was to create a vdM database (asparagine_comb.py),
cluster the vdMs (run_cluster.py), make relative vdMs by aligning precisely on backbone (make_rel_vdms_hbond_asn.py),
making representative vdMs for sampling (make_reps.py), superposing apixaban onto these representative vdMs
(superpose_ligand.py), and sampling these representative vdMs and ligand-superimposed vdMs on
template backbones (run_comb.py) to discover binding poses.  Absolute paths were removed from the scripts
and replaced with placeholder strings.