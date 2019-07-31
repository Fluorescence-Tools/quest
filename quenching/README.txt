1) Go to the folder of the program in the command line (by clicking on shell.bat)
2) run: "python determine_qy.py xxxxxx" xxxx are the parameter
3) mandatory parameters: the pdb-file, the chain id, the amino acid numbers

Example
-------

python determine_qy.py -f 3q5d_fixed.pdb -c " " -p 11 401
                                 ^           ^     ^^^^^^
                          pdb file path    chain   LP resid


To get help on the parameters, run:
python determine_qy.py -h




Additionally, there is a helper script which replaces the resname of a given residue with "ALA". This might be usefull if you want to exclude one of the quenchers.

python hide_quencher.py     123        3q5d_fixed.pdb   out.pdb
                             ^              ^             ^
                     resid to exclude    source pdb    modified pdb path
