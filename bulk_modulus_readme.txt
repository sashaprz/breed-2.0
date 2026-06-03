
Want bulk modulus. physically, this is the curvature of the energy volume curve at its minimum 
(add yt vid explaining bulk vs youngs modulus)
So i use a cheap standin for DFT (actually computing the effects of each of the electrons within the system)
to compute energy of the lattice at a series of volumes, then i git those points to a curve shape (Birch-
Murnaghan), then read off the curvature of the fit. 
Then optionally, I swap energy E for free energy F to get temperature dependence - this is called QHA. 
There's a front end to handle disorder. 

MLIP (MACE, SevenNet, MatterSim) is a neural network trained on millions of DFT calculations. it learned an
approximation to a function that maps a set of atomic positions, a cell, and the chemical species to the system's
total energy. From one run, you get total energy E, the force on every atom, and the stress tensor (how energy 
changes when you stretch, compress, or shear the lattice). 
MLIP is basically DFT but WAY faster (its trained on DFT data), and it works on novel structures.
-> use forces to relax structures, energy to build EOS, the stress for independent cross check, and forces-under-
displacement to get phonons. 

1. turn disordered structure into definite atoms
input crustal structure (from materials project, or my own generation). a structure can have fractional occupancy: 
this site has Li 50% of the time. This is a statistical statement, as the crystal is huge and only some sites
have Li there. But MLIP needs a real atom at every site. 
If the structure is ordered (or is very close to being ordered and can be rounded) then this is skipped. 
If it's actually disordered then: 
1. representation: build a supercell large enough to turn the fractional occupancy into a full occupancy. Ex if 
50% occupancy, double the current cell and fill only one of the 2 sites. 
2. configuration: you have to decide which of the 2 (or more) sites gets filled, and which is left empty. There are
different ways to do this but I am ranking candidates with a cheap proxy (Eswalkd electrostatic energy)
and using the lowest energy option. 

Eswald: assign each atom a charge (as if it were ionized) and then compute the electrostatic interactions
ie the total forces those charged atoms exert on each other. The dominant mechanism is that like charges
will repel, so that would be higher energy than having like charges spaced throughout the crystal. 
So this is a cheap proxy for finding "lowest energy" option. Good enough, not super accurate. 

2. find the MLIP's equilibrium volume 
<insert the energy vs volume graph and show minimum i want to find)>
(the volume that's lowest energy and thus most favourable).i need to find the actual minimum because curvature
is only bulk modulus when measured at the minimum, not partly up a slopw.
There are 2 things i can change: 1) the size of the cell 2) where the atoms sit inside the cell. 
first i relax both of these at once. i basically run an optimizer until the largest force on any atom 
drops below a threshold. 
It is important to relax with the same MLIP i will later scan with, even if i have an already-relaxed structure
from Materials porject. this is important because variations in method would shift th eequilibrium volumes
i get, which would give me a biased curvature and thus not an accurate prediction for bulk modulus. 

3. sample the curve
pick a grid of target volumes around equilibrium volume (9 points spanning +/-5%) biased towards expansion. 
For each target volume you do 2 things: 
1. scale the cell isotropicaly (multiply the cell matrix by f^(1/3) so the volume scales by exactly f. scaling 
uniformly gives bulk modulus, not some combination of bulk and shear moduli. 
2. at the strained volume, you relax internal positions while the cell volume is held fixed. you are finding
lowest energy achieveable at the clamped volume. 

4. fit the equation of state and find bulk modulus
now i have a set of (V, E) points that trace the minimum energy valley. i COULD take a numerical second derivative
of three close points, which is the formula for bulk modulus, but that would be extremely noise sensitive. So, 
instead I use equation of states (EOS). Birch-Murnaghan is derived from finite strain elasticity theory, so it 
has the physically correct shape over a wide window. I fit my 9 points to it - there are 4 free parameters: 
equilibrium energy, equilibrium volume, equilibrium bulk modulus, and pressure derivative of bulk modulus. 
then i can get exact curvature off the global shape rather than from local differences. Then, I can 
also use the fitted coefficients to check against my previous numbers to see if my fit was correct. 

























