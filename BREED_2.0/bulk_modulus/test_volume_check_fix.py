#!/usr/bin/env python3
"""Fast unit check of the per-atom volume-disagreement fix (no MLIP needed).

A zero-force/zero-stress fake calculator makes relax_structure converge in one
step, so we exercise only the mp_volume vs mp_n_atoms comparison branch:

  * with mp_n_atoms -> per-atom comparison -> NO spurious warning on a supercell,
  * without mp_n_atoms -> legacy raw comparison -> still warns (back-compat).
"""

import warnings
warnings.filterwarnings("ignore")

import numpy as np
from ase.build import bulk
from ase.calculators.calculator import Calculator, all_changes

from physics_bulk_modulus import relax_structure, ensure_supercell, BulkModulusResult


class FakeCalc(Calculator):
    """Returns 0 energy / forces / stress for any geometry -> instant relax."""
    implemented_properties = ["energy", "forces", "stress"]

    def calculate(self, atoms=None, properties=None, system_changes=all_changes):
        super().calculate(atoms, properties, system_changes)
        self.results["energy"] = 0.0
        self.results["forces"] = np.zeros((len(atoms), 3))
        self.results["stress"] = np.zeros(6)


def has_disagree_warning(res):
    return any("disagrees with MP" in w for w in res.warnings)


def main():
    calc = FakeCalc()
    # 2-atom Si primitive -> supercelled to meet the 12 A receptive field.
    prim = bulk("Si", "diamond", a=5.43)
    sc = ensure_supercell(prim, 12.0)
    mp_vol_per_atom = prim.get_volume() / len(prim)
    mp_volume = prim.get_volume()       # MP "primitive" volume (2 atoms)
    mp_n_atoms = len(prim)              # 2
    print(f"primitive: {len(prim)} atoms, V={prim.get_volume():.2f} A^3 "
          f"(V/atom={mp_vol_per_atom:.3f})")
    print(f"supercell: {len(sc)} atoms, V={sc.get_volume():.2f} A^3 "
          f"(V/atom={sc.get_volume()/len(sc):.3f})")

    results = []

    # 1) New path: per-atom comparison, no spurious warning.
    res = BulkModulusResult("fix-test")
    relax_structure(sc, calc, fmax=0.02, mp_volume=mp_volume,
                    mp_n_atoms=mp_n_atoms, res=res, resymmetrize=False)
    cond = not has_disagree_warning(res)
    results.append(cond)
    print(f"  [{'PASS' if cond else 'FAIL'}] per-atom (mp_n_atoms given): "
          f"no spurious volume warning  (warnings={res.warnings})")

    # 2) Legacy path: no mp_n_atoms -> raw comparison still warns on a supercell.
    res2 = BulkModulusResult("legacy-test")
    relax_structure(sc, calc, fmax=0.02, mp_volume=mp_volume,
                    mp_n_atoms=None, res=res2, resymmetrize=False)
    cond2 = has_disagree_warning(res2)
    results.append(cond2)
    print(f"  [{'PASS' if cond2 else 'FAIL'}] legacy (no mp_n_atoms): raw "
          f"comparison still warns (back-compat preserved)")

    # 3) Per-atom correctly DOES warn on a genuine density mismatch.
    res3 = BulkModulusResult("real-mismatch")
    relax_structure(sc, calc, fmax=0.02, mp_volume=mp_volume * 0.5,
                    mp_n_atoms=mp_n_atoms, res=res3, resymmetrize=False)
    cond3 = has_disagree_warning(res3)
    results.append(cond3)
    print(f"  [{'PASS' if cond3 else 'FAIL'}] per-atom flags a REAL 50% density "
          f"mismatch (diagnostic still works)")

    n = sum(results)
    print(f"\n  RESULT: {n}/{len(results)} checks passed")
    return 0 if n == len(results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
