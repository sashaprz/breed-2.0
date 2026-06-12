"""
Dataset + DataLoader helpers for the improved CGCNN pipeline.

Why not just use `cgcnn.data.CIFData` / `get_train_val_test_loader`?
- CIFData reads a single hardcoded `id_prop.csv` (one target column) and shuffles
  internally, and the stock loader does a *random contiguous* split. We need:
    * targets that differ by task (is_metal for the classifier, band_gap for the
      regressor) over the SAME cif directory,
    * an explicit train/val/test id list coming from a composition-grouped split
      (improvement #2), with no internal reshuffle,
    * optional graph caching to disk for fast multi-seed ensemble training.

So this module reuses the *graph-building math* from `cgcnn.data` (the
occupancy-weighted atom features, neighbour search, Gaussian distance expansion,
and `collate_pool`) but drives it from an explicit (id, target) list.

`build_graph` is a standalone copy of `CIFData.__getitem__`'s graph construction so
it can also be used by the fetcher (to validate structures) and by `cache_graphs.py`.
"""
from __future__ import annotations

import warnings
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

# Reused verbatim from the vendored CGCNN code.
from cgcnn.data import (
    AtomCustomJSONInitializer,
    GaussianDistance,
    collate_pool,
)

import config


def build_graph(cif_path, ari: AtomCustomJSONInitializer, gdf: GaussianDistance,
                max_num_nbr: int = config.MAX_NUM_NBR, radius: float = config.RADIUS):
    """Build (atom_fea, nbr_fea, nbr_fea_idx) for one CIF.

    Mirrors cgcnn.data.CIFData.__getitem__ (occupancy-weighted site features,
    distance-sorted neighbours, Gaussian-expanded bond distances). Raises on
    structures that cannot form a valid graph -- callers catch to drop them.
    """
    from pymatgen.core.structure import Structure

    crystal = Structure.from_file(str(cif_path))

    # occupancy-weighted atom features per site
    atom_fea_list = []
    feat_dim = ari.get_atom_fea(1).shape
    for site in crystal:
        site_feat = np.zeros(feat_dim)
        total_occ = 0.0
        for sp in site.species:
            if sp.number <= 0:
                continue
            site_feat += ari.get_atom_fea(sp.number) * site.species[sp]
            total_occ += site.species[sp]
        if total_occ > 0:
            site_feat /= total_occ
        else:
            warnings.warn(f"No valid species at a site in {cif_path}; zero feature.")
        atom_fea_list.append(site_feat)
    atom_fea = torch.Tensor(np.vstack(atom_fea_list))

    # neighbours
    all_nbrs = crystal.get_all_neighbors(radius, include_index=True)
    all_nbrs = [sorted(nbrs, key=lambda x: x[1]) for nbrs in all_nbrs]

    nbr_fea_idx, nbr_fea = [], []
    for nbr in all_nbrs:
        if len(nbr) < max_num_nbr:
            warnings.warn(f"{cif_path} has fewer than {max_num_nbr} neighbours.")
            nbr_fea_idx.append([x[2] for x in nbr] + [0] * (max_num_nbr - len(nbr)))
            nbr_fea.append([x[1] for x in nbr] + [radius + 1.0] * (max_num_nbr - len(nbr)))
        else:
            nbr_fea_idx.append([x[2] for x in nbr[:max_num_nbr]])
            nbr_fea.append([x[1] for x in nbr[:max_num_nbr]])

    nbr_fea_idx = torch.LongTensor(nbr_fea_idx)
    nbr_fea = torch.Tensor(gdf.expand(np.array(nbr_fea)))
    return atom_fea, nbr_fea, nbr_fea_idx


class GraphDataset(Dataset):
    """Crystal-graph dataset driven by an explicit (id, target) list.

    Parameters
    ----------
    id_target_pairs : list of (material_id, float target)
    cif_dir         : directory holding <material_id>.cif
    use_cache       : if True, load precomputed graphs from config.GRAPH_CACHE_DIR
                      when present (written by cache_graphs.py), else build on the fly.
    """
    def __init__(self, id_target_pairs, cif_dir=config.CIF_DIR, use_cache=True):
        self.pairs = list(id_target_pairs)
        self.cif_dir = Path(cif_dir)
        self.use_cache = use_cache
        self.ari = AtomCustomJSONInitializer(str(config.ATOM_INIT_FILE))
        self.gdf = GaussianDistance(dmin=config.GAUSS_DMIN, dmax=config.RADIUS,
                                    step=config.GAUSS_STEP)

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        mid, target = self.pairs[idx]
        cache = config.GRAPH_CACHE_DIR / f"{mid}.pt"
        if self.use_cache and cache.exists():
            atom_fea, nbr_fea, nbr_fea_idx = torch.load(cache)
        else:
            atom_fea, nbr_fea, nbr_fea_idx = build_graph(
                self.cif_dir / f"{mid}.cif", self.ari, self.gdf)
        return (atom_fea, nbr_fea, nbr_fea_idx), torch.Tensor([float(target)]), mid

    def sample_graph_dims(self):
        """(orig_atom_fea_len, nbr_fea_len) -- needed to build the model."""
        (atom_fea, nbr_fea, _), _, _ = self[0]
        return atom_fea.shape[-1], nbr_fea.shape[-1]


def make_loader(id_target_pairs, batch_size=config.BATCH_SIZE, shuffle=True,
                num_workers=config.WORKERS, use_cache=True):
    """Build a DataLoader over an explicit (id, target) list."""
    dataset = GraphDataset(id_target_pairs, use_cache=use_cache)
    loader = DataLoader(
        dataset, batch_size=batch_size, shuffle=shuffle,
        num_workers=num_workers, collate_fn=collate_pool,
        pin_memory=(config.DEVICE.type == "cuda"),
    )
    return dataset, loader
