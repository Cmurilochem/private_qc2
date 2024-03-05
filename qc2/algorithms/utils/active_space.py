from dataclasses import dataclass


@dataclass
class ActiveSpace:
    num_active_electrons: tuple[int, int]
    num_active_spatial_orbitals: int
