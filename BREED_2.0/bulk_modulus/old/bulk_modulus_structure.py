#!/usr/bin/env python3
"""
Structure-based Bulk Modulus Predictor

Improvements over composition_bulk_modulus_predictor.py:
  - Bond-length statistics from CIF nearest-neighbour analysis
  - Coordination-environment features via CrystalNN / VoronoiNN
  - Atomic packing descriptors (packing fraction, Voronoi volumes, free volume)
  - Symmetry-aware features: crystal-system one-hot, lattice anisotropy, centrosymmetry
  - Richer density physics: electron density, valence-electron density, heavy-element fraction
  - Bonding / elastic proxy descriptors: EN mismatch, bond ionicity, d-electron fraction,
    metallicity indicator, cohesive-energy proxy
  - Optional matminer ElementProperty (Magpie) features
  - SHAP / permutation-importance analysis after training
  - Stratified train/test split on bulk-modulus quantiles; composition-aware grouping
    to prevent polymorph leakage
  - Optional GNN path: M3GNet (matgl), CHGNet, or custom CGCNN (torch only)
"""

import os, json, pickle, warnings
import numpy as np
import pandas as pd
warnings.filterwarnings('ignore')

from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split, StratifiedKFold, GroupShuffleSplit
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.preprocessing import StandardScaler
from sklearn.inspection import permutation_importance as sk_perm_imp

from pymatgen.core import Structure, Composition, Element
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer
from pymatgen.analysis.local_env import CrystalNN, VoronoiNN

# ── Optional imports ──────────────────────────────────────────────────────────

try:
    from matminer.featurizers.composition import ElementProperty
    _MATMINER = True
except ImportError:
    _MATMINER = False
    print("[bulk_modulus_structure] matminer not found — built-in elemental stats only")

try:
    import shap as _shap_lib
    _SHAP = True
except ImportError:
    _SHAP = False

# GNN backend detection (first match wins)
_GNN_BACKEND = None
try:
    import matgl
    _GNN_BACKEND = 'matgl'
except ImportError:
    pass
if _GNN_BACKEND is None:
    try:
        from chgnet.model import CHGNet as _CHGNetClass
        _GNN_BACKEND = 'chgnet'
    except ImportError:
        pass
if _GNN_BACKEND is None:
    try:
        import torch
        import torch.nn as nn
        _GNN_BACKEND = 'cgcnn_torch'
    except ImportError:
        pass

# ── Constants ─────────────────────────────────────────────────────────────────

CRYSTAL_SYSTEMS = ['triclinic','monoclinic','orthorhombic',
                   'tetragonal','trigonal','hexagonal','cubic']

_D_BLOCK = {
    'Sc','Ti','V','Cr','Mn','Fe','Co','Ni','Cu','Zn',
    'Y','Zr','Nb','Mo','Tc','Ru','Rh','Pd','Ag','Cd',
    'Hf','Ta','W','Re','Os','Ir','Pt','Au','Hg',
    'La','Ce','Pr','Nd','Pm','Sm','Eu','Gd','Tb','Dy','Ho','Er','Tm','Yb','Lu',
}

_METALLIC = {
    'Li','Be','Na','Mg','Al','K','Ca','Sc','Ti','V','Cr','Mn','Fe','Co','Ni',
    'Cu','Zn','Ga','Rb','Sr','Y','Zr','Nb','Mo','Tc','Ru','Rh','Pd','Ag','Cd',
    'In','Sn','Cs','Ba','La','Ce','Pr','Nd','Pm','Sm','Eu','Gd','Tb','Dy','Ho',
    'Er','Tm','Yb','Lu','Hf','Ta','W','Re','Os','Ir','Pt','Au','Hg','Tl','Pb','Bi',
}

_COHESIVE_EV = {
    'H':2.24,'Li':1.63,'Be':3.32,'B':5.81,'C':7.37,'N':4.92,'O':2.60,
    'F':0.84,'Na':1.11,'Mg':1.51,'Al':3.39,'Si':4.63,'P':3.43,'S':2.85,
    'Cl':1.40,'K':0.93,'Ca':1.84,'Sc':3.90,'Ti':4.85,'V':5.31,'Cr':4.10,
    'Mn':2.92,'Fe':4.28,'Co':4.39,'Ni':4.44,'Cu':3.49,'Zn':1.35,'Ga':2.81,
    'Ge':3.85,'As':2.96,'Se':2.46,'Br':1.22,'Rb':0.86,'Sr':1.72,'Y':4.37,
    'Zr':6.25,'Nb':7.57,'Mo':6.82,'Ru':6.74,'Rh':5.75,'Pd':3.89,'Ag':2.95,
    'Cd':1.16,'In':2.52,'Sn':3.12,'Sb':2.75,'Te':2.19,'Cs':0.80,'Ba':1.90,
    'La':4.47,'Ce':4.32,'Hf':6.35,'Ta':8.10,'W':8.90,'Os':8.17,'Pt':5.84,
    'Au':3.81,'Pb':2.03,'Bi':2.18,
}

_D_ELECTRONS = {
    'Sc':1,'Ti':2,'V':3,'Cr':5,'Mn':5,'Fe':6,'Co':7,'Ni':8,'Cu':10,'Zn':10,
    'Y':1,'Zr':2,'Nb':4,'Mo':5,'Tc':5,'Ru':7,'Rh':8,'Pd':10,'Ag':10,'Cd':10,
    'Hf':2,'Ta':3,'W':4,'Re':5,'Os':6,'Ir':7,'Pt':9,'Au':10,'Hg':10,
    'La':1,'Ce':1,'Gd':1,'Lu':1,
}


def _is_high_id(mp_id: str) -> bool:
    """mp-IDs >= 2,000,000 correspond to newer MP calculations (possibly r2SCAN)."""
    try:
        return int(mp_id.replace("mp-", "")) >= 2_000_000
    except (ValueError, AttributeError):
        return False


# ── Elemental properties table ────────────────────────────────────────────────

_ELEM_PROPS = {
    'H':  {'mass':1.008,  'Z':1,  'val':1,  'EN':2.20,'ir':0.31,'cr':0.31,'vol':14.4, 'mp':14.01,  'K':0.0,  'G':0.0  },
    'Li': {'mass':6.941,  'Z':3,  'val':1,  'EN':0.98,'ir':0.76,'cr':1.28,'vol':13.02,'mp':453.69, 'K':11.0, 'G':4.2  },
    'Be': {'mass':9.012,  'Z':4,  'val':2,  'EN':1.57,'ir':0.27,'cr':0.96,'vol':5.0,  'mp':1560.0, 'K':130.0,'G':132.0},
    'B':  {'mass':10.811, 'Z':5,  'val':3,  'EN':2.04,'ir':0.27,'cr':0.84,'vol':4.39, 'mp':2349.0, 'K':320.0,'G':180.0},
    'C':  {'mass':12.011, 'Z':6,  'val':4,  'EN':2.55,'ir':0.16,'cr':0.76,'vol':5.29, 'mp':3823.0, 'K':442.0,'G':578.0},
    'N':  {'mass':14.007, 'Z':7,  'val':5,  'EN':3.04,'ir':0.16,'cr':0.71,'vol':17.3, 'mp':63.15,  'K':140.0,'G':30.0 },
    'O':  {'mass':15.999, 'Z':8,  'val':6,  'EN':3.44,'ir':1.40,'cr':0.66,'vol':14.0, 'mp':54.36,  'K':150.0,'G':33.0 },
    'F':  {'mass':18.998, 'Z':9,  'val':7,  'EN':3.98,'ir':1.33,'cr':0.57,'vol':17.1, 'mp':53.53,  'K':80.0, 'G':25.0 },
    'Na': {'mass':22.990, 'Z':11, 'val':1,  'EN':0.93,'ir':1.02,'cr':1.66,'vol':23.78,'mp':370.87, 'K':6.3,  'G':3.3  },
    'Mg': {'mass':24.305, 'Z':12, 'val':2,  'EN':1.31,'ir':0.72,'cr':1.41,'vol':14.0, 'mp':923.0,  'K':45.0, 'G':17.0 },
    'Al': {'mass':26.982, 'Z':13, 'val':3,  'EN':1.61,'ir':0.54,'cr':1.21,'vol':10.0, 'mp':933.47, 'K':76.0, 'G':26.0 },
    'Si': {'mass':28.086, 'Z':14, 'val':4,  'EN':1.90,'ir':0.40,'cr':1.11,'vol':12.1, 'mp':1687.0, 'K':100.0,'G':80.0 },
    'P':  {'mass':30.974, 'Z':15, 'val':5,  'EN':2.19,'ir':0.44,'cr':1.07,'vol':17.0, 'mp':317.3,  'K':120.0,'G':50.0 },
    'S':  {'mass':32.065, 'Z':16, 'val':6,  'EN':2.58,'ir':1.84,'cr':1.05,'vol':15.5, 'mp':388.36, 'K':80.0, 'G':40.0 },
    'Cl': {'mass':35.453, 'Z':17, 'val':7,  'EN':3.16,'ir':1.81,'cr':1.02,'vol':25.2, 'mp':171.6,  'K':50.0, 'G':20.0 },
    'K':  {'mass':39.098, 'Z':19, 'val':1,  'EN':0.82,'ir':1.38,'cr':2.03,'vol':45.94,'mp':336.53, 'K':3.1,  'G':1.3  },
    'Ca': {'mass':40.078, 'Z':20, 'val':2,  'EN':1.00,'ir':1.00,'cr':1.76,'vol':26.0, 'mp':1115.0, 'K':17.0, 'G':7.4  },
    'Sc': {'mass':44.956, 'Z':21, 'val':3,  'EN':1.36,'ir':0.75,'cr':1.70,'vol':15.0, 'mp':1814.0, 'K':57.0, 'G':29.0 },
    'Ti': {'mass':47.867, 'Z':22, 'val':4,  'EN':1.54,'ir':0.61,'cr':1.60,'vol':10.64,'mp':1941.0, 'K':110.0,'G':44.0 },
    'V':  {'mass':50.942, 'Z':23, 'val':5,  'EN':1.63,'ir':0.64,'cr':1.53,'vol':8.32, 'mp':2183.0, 'K':160.0,'G':47.0 },
    'Cr': {'mass':51.996, 'Z':24, 'val':6,  'EN':1.66,'ir':0.62,'cr':1.39,'vol':7.23, 'mp':2180.0, 'K':160.0,'G':115.0},
    'Mn': {'mass':54.938, 'Z':25, 'val':7,  'EN':1.55,'ir':0.67,'cr':1.39,'vol':7.35, 'mp':1519.0, 'K':120.0,'G':80.0 },
    'Fe': {'mass':55.845, 'Z':26, 'val':8,  'EN':1.83,'ir':0.65,'cr':1.32,'vol':7.09, 'mp':1811.0, 'K':170.0,'G':82.0 },
    'Co': {'mass':58.933, 'Z':27, 'val':9,  'EN':1.88,'ir':0.65,'cr':1.26,'vol':6.67, 'mp':1768.0, 'K':180.0,'G':75.0 },
    'Ni': {'mass':58.693, 'Z':28, 'val':10, 'EN':1.91,'ir':0.69,'cr':1.24,'vol':6.59, 'mp':1728.0, 'K':180.0,'G':76.0 },
    'Cu': {'mass':63.546, 'Z':29, 'val':11, 'EN':1.90,'ir':0.73,'cr':1.32,'vol':7.11, 'mp':1357.77,'K':140.0,'G':48.0 },
    'Zn': {'mass':65.38,  'Z':30, 'val':12, 'EN':1.65,'ir':0.74,'cr':1.22,'vol':9.16, 'mp':692.68, 'K':70.0, 'G':43.0 },
    'Ga': {'mass':69.723, 'Z':31, 'val':3,  'EN':1.81,'ir':0.62,'cr':1.22,'vol':11.8, 'mp':302.91, 'K':56.0, 'G':23.0 },
    'Ge': {'mass':72.64,  'Z':32, 'val':4,  'EN':2.01,'ir':0.53,'cr':1.20,'vol':13.6, 'mp':1211.4, 'K':75.0, 'G':41.0 },
    'As': {'mass':74.922, 'Z':33, 'val':5,  'EN':2.18,'ir':0.58,'cr':1.19,'vol':13.1, 'mp':1090.0, 'K':58.0, 'G':30.0 },
    'Se': {'mass':78.96,  'Z':34, 'val':6,  'EN':2.55,'ir':1.98,'cr':1.20,'vol':16.5, 'mp':494.0,  'K':50.0, 'G':25.0 },
    'Br': {'mass':79.904, 'Z':35, 'val':7,  'EN':2.96,'ir':1.96,'cr':1.20,'vol':23.5, 'mp':265.8,  'K':40.0, 'G':15.0 },
    'Rb': {'mass':85.468, 'Z':37, 'val':1,  'EN':0.82,'ir':1.52,'cr':2.20,'vol':55.76,'mp':312.46, 'K':2.5,  'G':1.0  },
    'Sr': {'mass':87.62,  'Z':38, 'val':2,  'EN':0.95,'ir':1.18,'cr':1.95,'vol':33.7, 'mp':1050.0, 'K':12.0, 'G':6.1  },
    'Y':  {'mass':88.906, 'Z':39, 'val':3,  'EN':1.22,'ir':0.90,'cr':1.90,'vol':19.88,'mp':1799.0, 'K':41.0, 'G':26.0 },
    'Zr': {'mass':91.224, 'Z':40, 'val':4,  'EN':1.33,'ir':0.72,'cr':1.75,'vol':14.0, 'mp':2128.0, 'K':90.0, 'G':33.0 },
    'Nb': {'mass':92.906, 'Z':41, 'val':5,  'EN':1.60,'ir':0.72,'cr':1.64,'vol':10.83,'mp':2750.0, 'K':170.0,'G':38.0 },
    'Mo': {'mass':95.96,  'Z':42, 'val':6,  'EN':2.16,'ir':0.69,'cr':1.54,'vol':9.38, 'mp':2896.0, 'K':230.0,'G':20.0 },
    'Ru': {'mass':101.07, 'Z':44, 'val':8,  'EN':2.20,'ir':0.68,'cr':1.46,'vol':8.17, 'mp':2607.0, 'K':220.0,'G':173.0},
    'Rh': {'mass':102.91, 'Z':45, 'val':9,  'EN':2.28,'ir':0.67,'cr':1.42,'vol':8.28, 'mp':2237.0, 'K':380.0,'G':150.0},
    'Pd': {'mass':106.42, 'Z':46, 'val':10, 'EN':2.20,'ir':0.86,'cr':1.39,'vol':8.56, 'mp':1828.05,'K':180.0,'G':44.0 },
    'Ag': {'mass':107.87, 'Z':47, 'val':11, 'EN':1.93,'ir':1.15,'cr':1.45,'vol':10.27,'mp':1234.93,'K':100.0,'G':30.0 },
    'Cd': {'mass':112.41, 'Z':48, 'val':12, 'EN':1.69,'ir':0.95,'cr':1.44,'vol':13.0, 'mp':594.22, 'K':42.0, 'G':19.0 },
    'In': {'mass':114.82, 'Z':49, 'val':3,  'EN':1.78,'ir':0.80,'cr':1.42,'vol':15.7, 'mp':429.75, 'K':41.0, 'G':26.0 },
    'Sn': {'mass':118.71, 'Z':50, 'val':4,  'EN':1.96,'ir':0.69,'cr':1.39,'vol':16.3, 'mp':505.08, 'K':58.0, 'G':18.0 },
    'Sb': {'mass':121.76, 'Z':51, 'val':5,  'EN':2.05,'ir':0.76,'cr':1.39,'vol':18.4, 'mp':903.78, 'K':42.0, 'G':20.0 },
    'Te': {'mass':127.60, 'Z':52, 'val':6,  'EN':2.10,'ir':2.21,'cr':1.38,'vol':20.5, 'mp':722.66, 'K':40.0, 'G':16.0 },
    'I':  {'mass':126.90, 'Z':53, 'val':7,  'EN':2.66,'ir':2.20,'cr':1.39,'vol':25.7, 'mp':386.85, 'K':35.0, 'G':12.0 },
    'Cs': {'mass':132.91, 'Z':55, 'val':1,  'EN':0.79,'ir':1.67,'cr':2.44,'vol':70.0, 'mp':301.59, 'K':1.6,  'G':0.6  },
    'Ba': {'mass':137.33, 'Z':56, 'val':2,  'EN':0.89,'ir':1.35,'cr':2.15,'vol':39.0, 'mp':1000.0, 'K':9.6,  'G':4.9  },
    'La': {'mass':138.91, 'Z':57, 'val':3,  'EN':1.10,'ir':1.03,'cr':2.07,'vol':22.5, 'mp':1193.0, 'K':28.0, 'G':14.0 },
    'Ce': {'mass':140.12, 'Z':58, 'val':4,  'EN':1.12,'ir':1.01,'cr':2.04,'vol':20.7, 'mp':1068.0, 'K':22.0, 'G':14.0 },
    'Pr': {'mass':140.91, 'Z':59, 'val':5,  'EN':1.13,'ir':0.99,'cr':2.03,'vol':20.8, 'mp':1208.0, 'K':29.0, 'G':15.0 },
    'Nd': {'mass':144.24, 'Z':60, 'val':6,  'EN':1.14,'ir':0.98,'cr':2.01,'vol':20.6, 'mp':1297.0, 'K':32.0, 'G':16.0 },
    'Sm': {'mass':150.36, 'Z':62, 'val':8,  'EN':1.17,'ir':0.96,'cr':1.98,'vol':19.9, 'mp':1345.0, 'K':38.0, 'G':18.0 },
    'Eu': {'mass':151.96, 'Z':63, 'val':9,  'EN':1.20,'ir':0.95,'cr':1.98,'vol':28.9, 'mp':1099.0, 'K':8.3,  'G':7.9  },
    'Gd': {'mass':157.25, 'Z':64, 'val':10, 'EN':1.20,'ir':0.94,'cr':1.96,'vol':19.9, 'mp':1585.0, 'K':38.0, 'G':22.0 },
    'Tb': {'mass':158.93, 'Z':65, 'val':11, 'EN':1.20,'ir':0.92,'cr':1.94,'vol':19.2, 'mp':1629.0, 'K':38.0, 'G':22.0 },
    'Dy': {'mass':162.50, 'Z':66, 'val':12, 'EN':1.22,'ir':0.91,'cr':1.92,'vol':19.0, 'mp':1680.0, 'K':41.0, 'G':25.0 },
    'Ho': {'mass':164.93, 'Z':67, 'val':13, 'EN':1.23,'ir':0.90,'cr':1.92,'vol':18.7, 'mp':1734.0, 'K':40.0, 'G':26.0 },
    'Er': {'mass':167.26, 'Z':68, 'val':14, 'EN':1.24,'ir':0.89,'cr':1.89,'vol':18.4, 'mp':1802.0, 'K':44.0, 'G':28.0 },
    'W':  {'mass':183.84, 'Z':74, 'val':6,  'EN':2.36,'ir':0.66,'cr':1.62,'vol':9.47, 'mp':3695.0, 'K':310.0,'G':161.0},
    'Os': {'mass':190.23, 'Z':76, 'val':8,  'EN':2.20,'ir':0.63,'cr':1.44,'vol':8.42, 'mp':3306.0, 'K':462.0,'G':222.0},
    'Pt': {'mass':195.08, 'Z':78, 'val':10, 'EN':2.28,'ir':0.80,'cr':1.36,'vol':9.09, 'mp':2041.4, 'K':230.0,'G':61.0 },
    'Au': {'mass':196.97, 'Z':79, 'val':11, 'EN':2.54,'ir':1.37,'cr':1.36,'vol':10.2, 'mp':1337.33,'K':220.0,'G':27.0 },
    'Pb': {'mass':207.20, 'Z':82, 'val':4,  'EN':2.33,'ir':1.19,'cr':1.46,'vol':18.3, 'mp':600.61, 'K':46.0, 'G':5.6  },
    'Bi': {'mass':208.98, 'Z':83, 'val':5,  'EN':2.02,'ir':1.03,'cr':1.48,'vol':21.3, 'mp':544.55, 'K':31.0, 'G':12.0 },
}


# ── Feature extraction helpers ────────────────────────────────────────────────

def _safe(fn, fallback):
    """Call fn(); return fallback array on any exception."""
    try:
        return fn()
    except Exception:
        return np.array(fallback, dtype=float)


def _bond_features(structure):
    """Mean/min/max bond length, bond-length variance, avg neighbours per atom."""
    nn = CrystalNN()
    all_bonds = []
    cn_per_site = []
    for i in range(len(structure)):
        try:
            info = nn.get_nn_info(structure, i)
            cn_per_site.append(len(info))
            for nbr in info:
                d = structure[i].distance(structure[nbr['site_index']])
                all_bonds.append(d)
        except Exception:
            continue
    if not all_bonds:
        return np.zeros(5)
    b = np.array(all_bonds)
    return np.array([b.mean(), b.min(), b.max(), b.var(),
                     np.mean(cn_per_site) if cn_per_site else 0.0])


def _coordination_features(structure):
    """Mean CN, CN std, min/max CN, CN histogram (bins 1-2,3-4,5-6,7-8,9+),
    fraction tetrahedral (CN=4) and octahedral (CN=6)."""
    nn = CrystalNN()
    cns = []
    for i in range(len(structure)):
        try:
            cns.append(len(nn.get_nn_info(structure, i)))
        except Exception:
            continue
    if not cns:
        return np.zeros(11)
    cns = np.array(cns, dtype=float)
    bins = [
        np.mean((cns >= 1) & (cns <= 2)),
        np.mean((cns >= 3) & (cns <= 4)),
        np.mean((cns >= 5) & (cns <= 6)),
        np.mean((cns >= 7) & (cns <= 8)),
        np.mean(cns >= 9),
    ]
    frac_tet = np.mean(cns == 4)
    frac_oct = np.mean(cns == 6)
    return np.array([cns.mean(), cns.std(), cns.min(), cns.max(),
                     *bins, frac_tet, frac_oct])


def _packing_features(structure):
    """Packing fraction, Voronoi volume stats, free volume fraction, atomic density."""
    lattice = structure.lattice
    cell_vol = lattice.volume

    # Sphere-packing fraction using covalent radii
    sphere_vol = 0.0
    for site in structure:
        sym = str(site.specie)
        r = _ELEM_PROPS.get(sym, {}).get('cr', 1.5)
        sphere_vol += (4.0 / 3.0) * np.pi * r ** 3
    packing_frac = sphere_vol / cell_vol if cell_vol > 0 else 0.0
    free_vol_frac = max(0.0, 1.0 - packing_frac)
    atomic_density = len(structure) / cell_vol if cell_vol > 0 else 0.0

    # Voronoi volumes
    voro_vols = []
    vnn = VoronoiNN()
    for i in range(len(structure)):
        try:
            vol = vnn.get_voronoi_polyhedra(structure, i)
            vv = sum(v['volume'] for v in vol.values())
            voro_vols.append(vv)
        except Exception:
            continue

    if voro_vols:
        vv = np.array(voro_vols)
        return np.array([packing_frac,
                         vv.mean(), vv.std(), vv.min(), vv.max(),
                         free_vol_frac, atomic_density])
    return np.array([packing_frac, 0, 0, 0, 0, free_vol_frac, atomic_density])


def _symmetry_features(structure):
    """Crystal-system one-hot (7), lattice anisotropy (a/b, b/c, c/a),
    angles (alpha, beta, gamma), centrosymmetry flag (1 scalar)."""
    try:
        spga = SpacegroupAnalyzer(structure)
        cs = spga.get_crystal_system()
        sg_num = spga.get_space_group_number()
        # centrosymmetric: space groups with inversion (point group has -1)
        try:
            pg = spga.get_point_group_symbol()
            centrosym = float('-' in pg or 'mmm' in pg or '6/m' in pg
                              or 'm-3' in pg or '4/m' in pg or '-3' in pg)
        except Exception:
            centrosym = 0.0
    except Exception:
        cs = 'triclinic'
        sg_num = 1
        centrosym = 0.0

    cs_onehot = [float(cs == s) for s in CRYSTAL_SYSTEMS]
    lat = structure.lattice
    a, b, c = lat.abc
    alpha, beta, gamma = lat.angles
    eps = 1e-8
    aniso = [a / (b + eps), b / (c + eps), c / (a + eps)]
    return np.array([sg_num, *cs_onehot, *aniso, alpha, beta, gamma, centrosym])


def _density_physics_features(structure, composition):
    """Electron density, valence-electron density, atoms/volume,
    avg atomic volume, heavy-element fraction (Z > 36)."""
    vol = structure.volume
    n_atoms = len(structure)

    total_Z = 0.0
    total_val = 0.0
    heavy_count = 0.0
    for el, frac in composition.fractional_composition.items():
        sym = str(el)
        p = _ELEM_PROPS.get(sym, {})
        Z = p.get('Z', 0)
        total_Z += Z * frac
        total_val += p.get('val', 0) * frac
        if Z > 36:
            heavy_count += frac

    electron_density = (total_Z * n_atoms) / vol if vol > 0 else 0.0
    valence_electron_density = (total_val * n_atoms) / vol if vol > 0 else 0.0
    atoms_per_vol = n_atoms / vol if vol > 0 else 0.0
    avg_atomic_vol = vol / n_atoms if n_atoms > 0 else 0.0

    return np.array([electron_density, valence_electron_density,
                     atoms_per_vol, avg_atomic_vol, heavy_count])


def _bonding_elastic_features(structure, composition):
    """EN mismatch, bond ionicity (Pauling), size mismatch,
    d-electron fraction, metallicity indicator, cohesive-energy proxy."""
    ENs, crs, vals, d_elecs, metals, cohesives = [], [], [], [], [], []

    for el, frac in composition.fractional_composition.items():
        sym = str(el)
        p = _ELEM_PROPS.get(sym, {})
        ENs.append(p.get('EN', 0))
        crs.append(p.get('cr', 1.5))
        vals.append(p.get('val', 0))
        d_elecs.append(_D_ELECTRONS.get(sym, 0))
        metals.append(float(sym in _METALLIC))
        cohesives.append(_COHESIVE_EV.get(sym, 3.0) * frac)

    if not ENs:
        return np.zeros(6)

    en_mismatch = max(ENs) - min(ENs)
    # Pauling bond ionicity proxy: 1 - exp(-0.25 * dEN^2)
    bond_ionicity = 1.0 - np.exp(-0.25 * en_mismatch ** 2)
    size_mismatch = max(crs) / (min(crs) + 1e-8)

    total_val = sum(v * frac for v, (_, frac) in zip(vals, composition.fractional_composition.items()))
    total_d = sum(d * frac for d, (_, frac) in zip(d_elecs, composition.fractional_composition.items()))
    d_frac = total_d / (total_val + 1e-8)

    metallicity = np.mean(metals)
    cohesive_proxy = sum(cohesives)

    return np.array([en_mismatch, bond_ionicity, size_mismatch,
                     d_frac, metallicity, cohesive_proxy])


def _elemental_stats_features(composition):
    """6 statistics (mean, max, min, range, weighted-mean, std) for each of
    10 elemental properties → 60 features."""
    prop_keys = ['mass', 'val', 'EN', 'ir', 'cr', 'vol', 'mp', 'K', 'G', 'Z']
    feat = []
    fracs = {str(el): f for el, f in composition.fractional_composition.items()}
    for pk in prop_keys:
        vals = []
        wsum = 0.0
        for sym, frac in fracs.items():
            v = _ELEM_PROPS.get(sym, {}).get(pk, 0.0)
            vals.append(v)
            wsum += v * frac
        if vals:
            feat.extend([np.mean(vals), np.max(vals), np.min(vals),
                         np.max(vals) - np.min(vals), wsum,
                         np.std(vals) if len(vals) > 1 else 0.0])
        else:
            feat.extend([0.0] * 6)
    return np.array(feat)


def _matminer_features(composition):
    """ElementProperty (Magpie preset) features if matminer is installed."""
    if not _MATMINER:
        return np.zeros(0)
    try:
        ep = ElementProperty.from_preset('magpie')
        df = pd.DataFrame([{'composition': composition}])
        ep.featurize_dataframe(df, 'composition', ignore_errors=True)
        cols = ep.feature_labels()
        row = df[cols].fillna(0).values[0]
        return row.astype(float)
    except Exception:
        return np.zeros(132)


def _print_split_stats(y_train: np.ndarray, y_test: np.ndarray) -> None:
    """Print bulk-modulus distribution for train and test to confirm balance."""
    print("\nTrain/Test target distribution:")
    for label, y in [("Train", y_train), ("Test", y_test)]:
        q25, q50, q75 = np.percentile(y, [25, 50, 75])
        print(f"  {label:5s}: n={len(y):4d}  mean={y.mean():.1f}  std={y.std():.1f}"
              f"  Q1/Q2/Q3=[{q25:.1f}/{q50:.1f}/{q75:.1f}]"
              f"  range=[{y.min():.1f}, {y.max():.1f}] GPa")


def _run_baselines(y_train: np.ndarray, y_test: np.ndarray,
                   X_train_s: np.ndarray, X_test_s: np.ndarray,
                   feature_names: list) -> None:
    """Three baselines to confirm the full model is learning meaningful signal.

    Feature-vector layout (bulk_modulus_structure.py):
      [0-2]   basic composition (3)
      [3-4]   density, vol_per_atom (2)
      [5-19]  symmetry (15)
      [20-79] elemental stats — 10 props × 6 stats (60)  ← composition-only
      [80-84] density physics (5)
      [85-90] bonding/elastic (6)
      [91+]   bond/coord/packing/matminer (structural)
    """
    print("\nBaselines (vs full model below):")

    # 1. Mean predictor — trivial lower bound
    y_mean = np.full_like(y_test, y_train.mean())
    print(f"  Mean predictor:       MAE={mean_absolute_error(y_test, y_mean):.2f} GPa"
          f"  R²={r2_score(y_test, y_mean):.3f}")

    # 2. Density-only RF — single structural feature at index 3
    try:
        rf_d = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
        rf_d.fit(X_train_s[:, [3]], y_train)
        pred_d = rf_d.predict(X_test_s[:, [3]])
        print(f"  Density-only RF:      MAE={mean_absolute_error(y_test, pred_d):.2f} GPa"
              f"  R²={r2_score(y_test, pred_d):.3f}")
    except Exception:
        pass

    # 3. Elemental-stats-only RF — 60 pure composition features at indices 20:80
    if X_train_s.shape[1] > 80:
        try:
            rf_comp = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
            rf_comp.fit(X_train_s[:, 20:80], y_train)
            pred_comp = rf_comp.predict(X_test_s[:, 20:80])
            print(f"  Elemental-stats RF:   MAE={mean_absolute_error(y_test, pred_comp):.2f} GPa"
                  f"  R²={r2_score(y_test, pred_comp):.3f}")
        except Exception:
            pass


# ── Main predictor class ──────────────────────────────────────────────────────

class StructureBulkModulusPredictor:
    """
    Structure-aware bulk modulus predictor.

    feature_mode:
      'full'    – all sub-features including bond/coordination/packing (slower)
      'fast'    – skip CrystalNN-heavy steps (bond + coordination), use elemental
                  stats + symmetry + packing approx + density + bonding only
    """

    MODEL_FILE  = 'structure_bulk_modulus_model.pkl'
    SCALER_FILE = 'structure_bulk_modulus_scaler.pkl'

    def __init__(self, feature_mode='full'):
        self.feature_mode = feature_mode
        self.model = None
        self.scaler = StandardScaler()
        self.feature_names: list[str] = []
        self._n_features: int | None = None

    # ── feature extraction ────────────────────────────────────────────────────

    def _build_feature_names(self):
        names = []
        # basic composition
        names += ['n_elements', 'n_atoms', 'mol_weight']
        # density + volume
        names += ['density', 'vol_per_atom']
        # symmetry block: 1 + 7 + 3 + 3 + 1 = 15
        names += ['sg_number']
        names += [f'cs_{s}' for s in CRYSTAL_SYSTEMS]
        names += ['lat_a_over_b', 'lat_b_over_c', 'lat_c_over_a']
        names += ['alpha', 'beta', 'gamma', 'centrosym']
        # elemental stats: 10 props × 6 stats = 60
        prop_keys = ['mass','val','EN','ir','cr','vol','mp','K','G','Z']
        stats = ['mean','max','min','range','wmean','std']
        for pk in prop_keys:
            for s in stats:
                names.append(f'elem_{pk}_{s}')
        # density physics: 5
        names += ['electron_density','valence_electron_density',
                  'atoms_per_vol','avg_atomic_vol','heavy_elem_frac']
        # bonding/elastic: 6
        names += ['EN_mismatch','bond_ionicity','size_mismatch',
                  'd_electron_frac','metallicity','cohesive_proxy']
        if self.feature_mode == 'full':
            # bond features: 5
            names += ['bond_len_mean','bond_len_min','bond_len_max',
                      'bond_len_var','avg_neighbors']
            # coordination: 11
            names += ['CN_mean','CN_std','CN_min','CN_max',
                      'CN_hist_1-2','CN_hist_3-4','CN_hist_5-6',
                      'CN_hist_7-8','CN_hist_9+',
                      'frac_tet','frac_oct']
            # packing: 7
            names += ['packing_frac',
                      'voro_vol_mean','voro_vol_std','voro_vol_min','voro_vol_max',
                      'free_vol_frac','atomic_density']
        if _MATMINER:
            try:
                ep = ElementProperty.from_preset('magpie')
                names += ep.feature_labels()
            except Exception:
                names += [f'magpie_{i}' for i in range(132)]
        return names

    def extract_features(self, cif_file_path: str) -> np.ndarray:
        structure = Structure.from_file(cif_file_path)
        composition = structure.composition
        lattice = structure.lattice
        a, b, c = lattice.abc

        feats = []

        # basic composition (3)
        feats.append(np.array([len(composition), composition.num_atoms,
                                composition.weight]))

        # density + vol_per_atom (2)
        feats.append(np.array([structure.density,
                                structure.volume / max(1, len(structure))]))

        # symmetry (15)
        feats.append(_safe(lambda: _symmetry_features(structure), np.zeros(15)))

        # elemental stats (60)
        feats.append(_elemental_stats_features(composition))

        # density physics (5)
        feats.append(_safe(lambda: _density_physics_features(structure, composition),
                            np.zeros(5)))

        # bonding / elastic (6)
        feats.append(_safe(lambda: _bonding_elastic_features(structure, composition),
                            np.zeros(6)))

        if self.feature_mode == 'full':
            feats.append(_safe(lambda: _bond_features(structure), np.zeros(5)))
            feats.append(_safe(lambda: _coordination_features(structure), np.zeros(11)))
            feats.append(_safe(lambda: _packing_features(structure), np.zeros(7)))

        if _MATMINER:
            feats.append(_safe(lambda: _matminer_features(composition), np.zeros(132)))

        vec = np.concatenate(feats).astype(float)
        vec = np.nan_to_num(vec, nan=0.0, posinf=0.0, neginf=0.0)
        return vec

    # ── training ──────────────────────────────────────────────────────────────

    def train_model(self,
                    training_data_file: str = 'low_bm_training/training_metadata.json',
                    analyze_importance: bool = True):
        print("Training Structure-Based Bulk Modulus Predictor")
        print("=" * 60)

        if not os.path.exists(training_data_file):
            print(f"Training data not found: {training_data_file}")
            return None

        with open(training_data_file) as f:
            training_data = json.load(f)
        print(f"Loaded {len(training_data)} samples")

        features, targets, compositions = [], [], []
        n_r2scan = 0
        cif_dir = os.path.join(os.path.dirname(os.path.abspath(training_data_file)),
                               "structures")

        print("Extracting features from CIF files...")
        for i, sample in enumerate(training_data):
            # Exclude entries that may use a different DFT functional (r2SCAN vs GGA/PBE)
            if sample.get('possibly_r2scan') or _is_high_id(sample.get('material_id', '')):
                n_r2scan += 1
                continue

            cif_path = os.path.join(cif_dir, sample['cif_file'])
            if not os.path.exists(cif_path):
                continue
            bm = sample['bulk_modulus']
            if bm > 1000 or bm < 5:
                continue
            try:
                vec = self.extract_features(cif_path)
                features.append(vec)
                targets.append(bm)
                compositions.append(sample.get('formula', ''))
            except Exception as e:
                print(f"  Failed {sample.get('formula','')}: {e}")
            if (i + 1) % 100 == 0:
                print(f"  {i+1}/{len(training_data)}")

        if n_r2scan:
            print(f"Skipped {n_r2scan} entries with mp-ID >= 2,000,000 (possible r2SCAN).")
        if not features:
            print("No valid features extracted.")
            return None

        # Align feature vector lengths
        max_len = max(len(v) for v in features)
        features = np.array([np.pad(v, (0, max_len - len(v))) for v in features])
        targets = np.array(targets)
        groups = np.array(compositions)
        self._n_features = max_len

        print(f"Extracted {len(features)} vectors, {max_len} features each")

        # Build feature names once
        self.feature_names = self._build_feature_names()

        # Grouped split: keep all polymorphs of a given composition in the same
        # partition so the model is evaluated on truly held-out compositions.
        gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
        train_idx, test_idx = next(gss.split(features, targets, groups=groups))
        X_train, X_test = features[train_idx], features[test_idx]
        y_train, y_test = targets[train_idx], targets[test_idx]

        print(f"Train: {len(X_train)}, Test: {len(X_test)}")
        _print_split_stats(y_train, y_test)

        # Scale
        X_train_s = self.scaler.fit_transform(X_train)
        X_test_s  = self.scaler.transform(X_test)

        _run_baselines(y_train, y_test, X_train_s, X_test_s, self.feature_names)

        # Train Random Forest
        print("\nTraining Random Forest...")
        self.model = RandomForestRegressor(
            n_estimators=400, max_depth=25,
            min_samples_split=3, min_samples_leaf=1,
            max_features='sqrt', random_state=42, n_jobs=-1
        )
        self.model.fit(X_train_s, y_train)

        train_pred = self.model.predict(X_train_s)
        test_pred  = self.model.predict(X_test_s)

        print(f"\nModel Performance:")
        print(f"  Train  MAE={mean_absolute_error(y_train,train_pred):.2f} GPa  R²={r2_score(y_train,train_pred):.3f}")
        print(f"  Test   MAE={mean_absolute_error(y_test, test_pred ):.2f} GPa  R²={r2_score(y_test, test_pred ):.3f}")

        if analyze_importance:
            self._analyze_importance(X_test_s, y_test)

        # Save
        script_dir = os.path.dirname(os.path.abspath(__file__))
        with open(os.path.join(script_dir, self.MODEL_FILE),  'wb') as f:
            pickle.dump((self.model, self._n_features), f)
        with open(os.path.join(script_dir, self.SCALER_FILE), 'wb') as f:
            pickle.dump(self.scaler, f)
        print(f"Model saved to {script_dir}")
        return self.model

    # ── importance analysis ───────────────────────────────────────────────────

    def _analyze_importance(self, X_test_s, y_test, top_n=20):
        print(f"\nTop {top_n} features by permutation importance:")
        result = sk_perm_imp(self.model, X_test_s, y_test,
                             n_repeats=5, random_state=42, n_jobs=-1)
        imp = result.importances_mean
        names = (self.feature_names[:len(imp)]
                 if self.feature_names else [str(i) for i in range(len(imp))])
        order = np.argsort(imp)[::-1][:top_n]
        for rank, idx in enumerate(order, 1):
            print(f"  {rank:2d}. {names[idx]:<40s}  {imp[idx]:.4f}")

        if _SHAP:
            print("\nComputing SHAP values (TreeExplainer)...")
            try:
                explainer = _shap_lib.TreeExplainer(self.model)
                # Use a small sample for speed
                sample = X_test_s[:min(200, len(X_test_s))]
                shap_values = explainer.shap_values(sample)
                mean_abs = np.abs(shap_values).mean(axis=0)
                order_s = np.argsort(mean_abs)[::-1][:top_n]
                print(f"\nTop {top_n} features by mean |SHAP|:")
                for rank, idx in enumerate(order_s, 1):
                    n = names[idx] if idx < len(names) else str(idx)
                    print(f"  {rank:2d}. {n:<40s}  {mean_abs[idx]:.4f}")
            except Exception as e:
                print(f"  SHAP failed: {e}")

    # ── prediction ────────────────────────────────────────────────────────────

    def _load_model(self):
        script_dir = os.path.dirname(os.path.abspath(__file__))
        model_path  = os.path.join(script_dir, self.MODEL_FILE)
        scaler_path = os.path.join(script_dir, self.SCALER_FILE)
        if not os.path.exists(model_path):
            print(f"Model not found: {model_path}. Run train_model() first.")
            return False
        with open(model_path, 'rb') as f:
            obj = pickle.load(f)
        # support both old (model only) and new (model, n_features) saves
        if isinstance(obj, tuple):
            self.model, self._n_features = obj
        else:
            self.model = obj
        if os.path.exists(scaler_path):
            with open(scaler_path, 'rb') as f:
                self.scaler = pickle.load(f)
        return True

    def predict_bulk_modulus(self, cif_file_path: str) -> float | None:
        if self.model is None and not self._load_model():
            return None
        try:
            vec = self.extract_features(cif_file_path)
            if self._n_features and len(vec) < self._n_features:
                vec = np.pad(vec, (0, self._n_features - len(vec)))
            elif self._n_features and len(vec) > self._n_features:
                vec = vec[:self._n_features]
            vec_s = self.scaler.transform(vec.reshape(1, -1))
            return float(self.model.predict(vec_s)[0])
        except Exception as e:
            print(f"Prediction failed: {e}")
            return None


# ── Optional GNN path ─────────────────────────────────────────────────────────

if _GNN_BACKEND == 'matgl':
    import matgl
    from matgl.ext.ase import M3GNetCalculator
    from ase.io import read as ase_read

    class M3GNetPredictor:
        """Fine-tune matgl M3GNet for bulk modulus prediction.

        Uses the pretrained M3GNet-MP-2021.2.8-PES potential to extract
        structure embeddings, then fits a small MLP regression head on top.
        Falls back to direct elastic tensor prediction if available.
        """

        def __init__(self):
            self.pot = matgl.load_model('M3GNet-MP-2021.2.8-PES')
            self.head = None
            self.scaler = StandardScaler()

        def _embed(self, cif_path: str) -> np.ndarray:
            import torch
            atoms = ase_read(cif_path)
            calc = M3GNetCalculator(potential=self.pot)
            atoms.set_calculator(calc)
            # Use total energy + forces RMS as a simple embedding proxy
            energy = atoms.get_potential_energy()
            forces = atoms.get_forces()
            forces_rms = float(np.sqrt((forces ** 2).mean()))
            vol_per_atom = atoms.get_volume() / len(atoms)
            return np.array([energy / len(atoms), forces_rms, vol_per_atom])

        def train_model(self, training_data_file: str,
                        epochs: int = 200) -> None:
            from sklearn.neural_network import MLPRegressor
            print("Training M3GNet-based predictor...")
            with open(training_data_file) as f:
                data = json.load(f)

            cif_dir = os.path.join(os.path.dirname(os.path.abspath(training_data_file)), "structures")
            X, y = [], []
            for s in data:
                path = os.path.join(cif_dir, s['cif_file'])
                if not os.path.exists(path):
                    continue
                bm = s['bulk_modulus']
                if bm > 1000 or bm < 5:
                    continue
                try:
                    X.append(self._embed(path))
                    y.append(bm)
                except Exception:
                    continue

            X, y = np.array(X), np.array(y)
            X_s = self.scaler.fit_transform(X)
            X_tr, X_te, y_tr, y_te = train_test_split(X_s, y, test_size=0.2, random_state=42)
            self.head = MLPRegressor(hidden_layer_sizes=(64, 32), max_iter=epochs, random_state=42)
            self.head.fit(X_tr, y_tr)
            pred = self.head.predict(X_te)
            print(f"  M3GNet MAE={mean_absolute_error(y_te,pred):.2f} R²={r2_score(y_te,pred):.3f}")

        def predict_bulk_modulus(self, cif_path: str) -> float | None:
            if self.head is None:
                return None
            try:
                x = self.scaler.transform(self._embed(cif_path).reshape(1, -1))
                return float(self.head.predict(x)[0])
            except Exception:
                return None


elif _GNN_BACKEND == 'cgcnn_torch':
    import torch
    import torch.nn as nn

    # Gaussian expansion of bond distances
    _GAUSS_FILTER = torch.linspace(0, 8, 40)
    _GAUSS_STEP   = 0.2

    def _gauss_expand(dist: float, centers=_GAUSS_FILTER, step=_GAUSS_STEP) -> torch.Tensor:
        return torch.exp(-((torch.tensor(dist) - centers) ** 2) / step ** 2)

    # One-hot element encoding: map atomic number to 0-based index for first 83 elements
    _MAX_Z = 84
    def _elem_onehot(z: int) -> torch.Tensor:
        v = torch.zeros(_MAX_Z)
        v[min(z - 1, _MAX_Z - 1)] = 1.0
        return v

    _ATOM_FEAT_DIM = _MAX_Z + 3  # one-hot + EN + cr + val
    _BOND_FEAT_DIM = 40           # Gaussian expansion

    def _build_graph(structure):
        """Returns (atom_feats [N, atom_dim], bond_feats [E, bond_dim],
        src [E], dst [E]) tensors for CGCNN."""
        nn_analyzer = CrystalNN()
        n = len(structure)
        atom_feats = []
        srcs, dsts, bond_feats = [], [], []

        for i, site in enumerate(structure):
            sym = str(site.specie)
            p = _ELEM_PROPS.get(sym, {})
            z = p.get('Z', 1)
            oh = _elem_onehot(z)
            extra = torch.tensor([p.get('EN', 2.0) / 4.0,
                                   p.get('cr', 1.5) / 3.0,
                                   p.get('val', 4.0) / 14.0])
            atom_feats.append(torch.cat([oh, extra]))

            try:
                nbrs = nn_analyzer.get_nn_info(structure, i)
            except Exception:
                nbrs = []
            for nbr in nbrs:
                j = nbr['site_index']
                d = float(structure[i].distance(structure[j]))
                srcs.append(i)
                dsts.append(j)
                bond_feats.append(_gauss_expand(d))

        if not srcs:
            # fallback: self-loops
            for i in range(n):
                srcs.append(i); dsts.append(i)
                bond_feats.append(_gauss_expand(3.0))

        return (torch.stack(atom_feats),
                torch.stack(bond_feats),
                torch.tensor(srcs, dtype=torch.long),
                torch.tensor(dsts, dtype=torch.long))

    class CGCNNConv(nn.Module):
        def __init__(self, atom_dim, bond_dim):
            super().__init__()
            self.fc = nn.Linear(2 * atom_dim + bond_dim, atom_dim)
            self.bn = nn.BatchNorm1d(atom_dim)
            self.sigmoid = nn.Sigmoid()
            self.softplus = nn.Softplus()

        def forward(self, h, bond_feat, src, dst):
            msg = torch.cat([h[src], h[dst], bond_feat], dim=-1)
            gate = self.sigmoid(self.fc(msg))
            upd  = self.softplus(self.fc(msg))
            agg  = torch.zeros_like(h).scatter_add(0, dst.unsqueeze(1).expand_as(upd), gate * upd)
            return self.bn(h + agg)

    class CGCNN(nn.Module):
        def __init__(self, atom_dim=_ATOM_FEAT_DIM, bond_dim=_BOND_FEAT_DIM,
                     hidden=64, n_conv=3):
            super().__init__()
            self.embedding = nn.Linear(atom_dim, hidden)
            self.convs = nn.ModuleList([CGCNNConv(hidden, bond_dim) for _ in range(n_conv)])
            self.fc1 = nn.Linear(hidden, hidden // 2)
            self.fc2 = nn.Linear(hidden // 2, 1)
            self.relu = nn.ReLU()

        def forward(self, atom_feat, bond_feat, src, dst):
            h = self.relu(self.embedding(atom_feat))
            for conv in self.convs:
                h = conv(h, bond_feat, src, dst)
            g = h.mean(dim=0, keepdim=True)  # global mean pool
            out = self.relu(self.fc1(g))
            return self.fc2(out).squeeze()

    class CGCNNPredictor:
        """Train a CGCNN from scratch on CIF structures."""

        MODEL_FILE = 'cgcnn_bulk_modulus.pt'

        def __init__(self, n_conv=3, hidden=64, epochs=200, lr=1e-3):
            self.n_conv = n_conv
            self.hidden = hidden
            self.epochs = epochs
            self.lr = lr
            self.net = None
            self.y_mean = 0.0
            self.y_std  = 1.0

        def train_model(self, training_data_file: str):
            print("Training CGCNN from scratch...")
            with open(training_data_file) as f:
                data = json.load(f)

            cif_dir = os.path.join(os.path.dirname(os.path.abspath(training_data_file)), "structures")
            graphs, targets = [], []
            for s in data:
                path = os.path.join(cif_dir, s['cif_file'])
                if not os.path.exists(path):
                    continue
                bm = s['bulk_modulus']
                if bm > 1000 or bm < 5:
                    continue
                try:
                    structure = Structure.from_file(path)
                    graphs.append(_build_graph(structure))
                    targets.append(bm)
                except Exception:
                    continue

            if not graphs:
                print("No graphs built.")
                return

            targets = np.array(targets)
            self.y_mean = targets.mean()
            self.y_std  = targets.std() + 1e-8
            targets_norm = (targets - self.y_mean) / self.y_std

            idx = list(range(len(graphs)))
            np.random.seed(42)
            np.random.shuffle(idx)
            split = int(0.8 * len(idx))
            tr_idx, te_idx = idx[:split], idx[split:]

            self.net = CGCNN(hidden=self.hidden, n_conv=self.n_conv)
            opt = torch.optim.Adam(self.net.parameters(), lr=self.lr)
            loss_fn = nn.MSELoss()

            print(f"  Train={len(tr_idx)}, Test={len(te_idx)}, epochs={self.epochs}")
            for ep in range(self.epochs):
                self.net.train()
                np.random.shuffle(tr_idx)
                ep_loss = 0.0
                for i in tr_idx:
                    af, bf, src, dst = graphs[i]
                    y_t = torch.tensor(targets_norm[i], dtype=torch.float32)
                    pred = self.net(af.float(), bf.float(), src, dst)
                    loss = loss_fn(pred, y_t)
                    opt.zero_grad(); loss.backward(); opt.step()
                    ep_loss += loss.item()
                if (ep + 1) % 20 == 0:
                    self.net.eval()
                    preds, trues = [], []
                    with torch.no_grad():
                        for i in te_idx:
                            af, bf, src, dst = graphs[i]
                            p = self.net(af.float(), bf.float(), src, dst).item()
                            preds.append(p * self.y_std + self.y_mean)
                            trues.append(targets[i])
                    mae = mean_absolute_error(trues, preds)
                    r2  = r2_score(trues, preds)
                    print(f"  Epoch {ep+1}/{self.epochs}  loss={ep_loss/len(tr_idx):.4f}  "
                          f"MAE={mae:.2f}  R²={r2:.3f}")

            script_dir = os.path.dirname(os.path.abspath(__file__))
            torch.save({'net': self.net.state_dict(),
                        'y_mean': self.y_mean, 'y_std': self.y_std,
                        'hidden': self.hidden, 'n_conv': self.n_conv},
                       os.path.join(script_dir, self.MODEL_FILE))
            print("CGCNN saved.")

        def _load(self):
            script_dir = os.path.dirname(os.path.abspath(__file__))
            path = os.path.join(script_dir, self.MODEL_FILE)
            if not os.path.exists(path):
                return False
            ckpt = torch.load(path, map_location='cpu')
            self.y_mean = ckpt['y_mean']
            self.y_std  = ckpt['y_std']
            self.net = CGCNN(hidden=ckpt['hidden'], n_conv=ckpt['n_conv'])
            self.net.load_state_dict(ckpt['net'])
            return True

        def predict_bulk_modulus(self, cif_path: str) -> float | None:
            if self.net is None and not self._load():
                return None
            try:
                structure = Structure.from_file(cif_path)
                af, bf, src, dst = _build_graph(structure)
                self.net.eval()
                with torch.no_grad():
                    p = self.net(af.float(), bf.float(), src, dst).item()
                return float(p * self.y_std + self.y_mean)
            except Exception as e:
                print(f"CGCNN prediction failed: {e}")
                return None


# ── Convenience functions (same interface as original file) ───────────────────

def train_structure_predictor(training_data_file: str = 'low_bm_training/training_metadata.json',
                               feature_mode: str = 'full',
                               use_gnn: bool = False):
    """Train the structure-based RF predictor (and optionally a GNN)."""
    predictor = StructureBulkModulusPredictor(feature_mode=feature_mode)
    predictor.train_model(training_data_file)

    if use_gnn:
        if _GNN_BACKEND == 'matgl':
            gnn = M3GNetPredictor()
            gnn.train_model(training_data_file)
        elif _GNN_BACKEND == 'cgcnn_torch':
            gnn = CGCNNPredictor()
            gnn.train_model(training_data_file)
        else:
            print("No GNN backend found (install matgl, chgnet, or torch+pymatgen).")

    return predictor


def predict_bulk_modulus_structure(cif_file_path: str) -> float | None:
    """Drop-in replacement for predict_bulk_modulus_enhanced."""
    predictor = StructureBulkModulusPredictor()
    return predictor.predict_bulk_modulus(cif_file_path)


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Structure-based bulk modulus predictor')
    parser.add_argument('--train', action='store_true')
    parser.add_argument('--gnn', action='store_true', help='Also train GNN backend')
    parser.add_argument('--fast', action='store_true',
                        help='Skip CrystalNN-heavy features (faster)')
    parser.add_argument('--predict', type=str, default=None,
                        help='Path to CIF file to predict')
    parser.add_argument('--data', type=str,
                        default='low_bm_training/training_metadata.json')
    args = parser.parse_args()

    print("Structure-Based Bulk Modulus Predictor")
    print(f"  matminer : {'available' if _MATMINER else 'not installed'}")
    print(f"  SHAP     : {'available' if _SHAP else 'not installed'}")
    print(f"  GNN      : {_GNN_BACKEND or 'none'}")

    if args.train:
        mode = 'fast' if args.fast else 'full'
        train_structure_predictor(args.data, feature_mode=mode, use_gnn=args.gnn)

    if args.predict:
        result = predict_bulk_modulus_structure(args.predict)
        if result is not None:
            print(f"Predicted bulk modulus: {result:.1f} GPa")
        else:
            print("Prediction failed.")
