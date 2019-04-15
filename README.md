Module name: scg4py (Systematic Coarse-Graining for Python)

Author: Saeed Mortezazadeh

Description:
scg4py is written in python language to generate coarse-grained (CG) potentials 
of the system of interest through systematic coarse-graining approach. This module 
is a collection of tools to implement all coarse-graining steps including mapping 
the atomistic trajectory to the CG trajectory, generating the CG topology file, 
calculating distribution functions and refining the CG potentials through Iterative 
Boltzmann Inversion (IBI) and Inverse Monte Carlo (IMC) methods. In the following 
paper, we applied this module to study the lamellar and inverse hexagonal formation
of lipid phases at different conditions. If you find this module useful, please cite our paper:

* _Implicit Solvent Systematic Coarse-Graining of Dioleoylphosphatidylethanolamine Lipids: 
from the Inverted Hexagonal to the Bilayer Structure_,
Saeed Mortezazadeh, Yousef Jamali, Hossein Naderi-Manesh, and Alexander P.Lyubartsev, 
(https://doi.org/10.1371/journal.pone.0214673)

For using this code you need python 3 and the following prerequisite modules: 
numpy, scipy, matplotlib, and mdtraj for parsing the ’dcd’ and ’xtc’ trajectory 
file formats. For ease of use, you can append the path of scg4py code to PYHTONPATH
in the .bashrc file.

export PYTHONPATH="/path/to/scg4py/:$PYTHONPATH"
