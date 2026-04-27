cat << 'SCRIPT' > /home/sashaprz/run_bvse.py
from bvlain import Lain
from pymatgen.core import Structure
from pymatgen.analysis.bond_valence import BVAnalyzer
import os, csv, warnings, traceback
warnings.filterwarnings('ignore')

oxi_rules = {
    'Li':1,'Na':1,'K':1,'Rb':1,'Cs':1,'Ag':1,'Cu':1,
    'Ba':2,'Ca':2,'Sr':2,'Mg':2,'Zn':2,'Cd':2,
    'Al':3,'La':3,'Y':3,'Gd':3,'Sc':3,'B':3,'In':3,'Ga':3,'Fe':3,'Cr':3,'Er':3,'Nd':3,'Sm':3,'Dy':3,'Ho':3,'Yb':3,'Lu':3,'Pr':3,'Ce':3,'Eu':3,'Tb':3,
    'Si':4,'Ge':4,'Sn':4,'Ti':4,'Zr':4,'Hf':4,'Mn':4,
    'P':5,'Nb':5,'Ta':5,'V':5,'Sb':5,'Bi':5,
    'W':6,'Mo':6,
    'O':-2,'S':-2,'Se':-2,'Te':-2,
    'F':-1,'Cl':-1,'Br':-1,'I':-1,
    'N':-3,'H':1
}

cif_dir = '/mnt/c/Users/Sasha/repos/genetic_algo/new_ionic_cond_predictor/BREED_2/cifs'
out_file = '/mnt/c/Users/Sasha/repos/genetic_algo/new_ionic_cond_predictor/BREED_2/bvse_features.csv'

bva = BVAnalyzer()
results = []
success = 0
fail = 0

cifs = sorted([f for f in os.listdir(cif_dir) if f.endswith('.cif')])
total = len(cifs)

for i, f in enumerate(cifs):
    print(f'[{i+1}/{total}] {f}', end=' ... ', flush=True)
    
    try:
        st = Structure.from_file(os.path.join(cif_dir, f))
    except Exception as e:
        print(f'FAIL (read): {e}')
        fail += 1
        continue
    
    # Try auto oxidation states first
    decorated = False
    try:
        st = bva.get_oxi_state_decorated_structure(st)
        decorated = True
    except:
        # Try rule-based
        elems = set(str(e) for e in st.composition.elements)
        if elems.issubset(oxi_rules.keys()):
            oxi_map = {e: oxi_rules[e] for e in elems}
            total_charge = sum(st.composition[e] * oxi_map[e] for e in elems)
            if abs(total_charge) < 0.5:
                st.add_oxidation_state_by_element(oxi_map)
                decorated = True
    
    if not decorated:
        print('FAIL (oxi states)')
        fail += 1
        continue
    
    # Run BVSE
    try:
        calc = Lain(verbose=False)
        calc.read_structure(st, oxi_check=False)
        params = {'mobile_ion': 'Li1+', 'r_cut': 10.0, 'resolution': 0.2, 'k': 100}
        _ = calc.bvse_distribution(**params)
        energies = calc.percolation_barriers(encut=5.0)
        
        row = {
            'cif_id': f.replace('.cif', ''),
            'barrier_1d': energies.get('1D percolation barrier', None),
            'barrier_2d': energies.get('2D percolation barrier', None),
            'barrier_3d': energies.get('3D percolation barrier', None),
        }
        results.append(row)
        print(f'OK (1D={row["barrier_1d"]:.3f}, 3D={row["barrier_3d"]:.3f})')
        success += 1
    except Exception as e:
        print(f'FAIL (bvse): {e}')
        fail += 1

# Write CSV
with open(out_file, 'w', newline='') as csvfile:
    writer = csv.DictWriter(csvfile, fieldnames=['cif_id', 'barrier_1d', 'barrier_2d', 'barrier_3d'])
    writer.writeheader()
    writer.writerows(results)

print(f'\nDone. Success: {success}, Failed: {fail}')
print(f'Results saved to {out_file}')
SCRIPT

echo "Script written. Run with:"
echo "python3 /home/sashaprz/run_bvse.py"