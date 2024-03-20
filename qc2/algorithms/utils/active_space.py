from dataclasses import dataclass
import numpy as np


def get_active_space_idx(
        nao, nelectron,
        n_active_orbitals,
        n_active_electrons
):
    """Calculates active space indices given active orbitals and electrons."""
    # Set active space parameters
    nelecore = sum(nelectron) - sum(n_active_electrons)
    if nelecore % 2 == 1:
        raise ValueError('odd number of core electrons')

    occ_idx = np.arange(nelecore // 2)
    act_idx = (occ_idx[-1] + 1 + np.arange(n_active_orbitals)
               if len(occ_idx) > 0
               else np.arange(n_active_orbitals))
    virt_idx = np.arange(act_idx[-1]+1, nao)

    return occ_idx, act_idx, virt_idx


@dataclass
class ActiveSpace:
    num_active_electrons: tuple[int, int]
    num_active_spatial_orbitals: int
