"""Example of a VQE calc using Qiskit-Nature and PySCF-ASE calculator fcidump.

Standard restricted calculation => H2 example.

Notes:
    Requires the installation of qc2, ase, qiskit and h5py.
"""

from ase.build import molecule

import qiskit_nature
from qiskit_nature.second_q.mappers import BravyiKitaevMapper
from qiskit_algorithms.optimizers import SLSQP
from qiskit.primitives import Estimator

from qc2.ase import PySCF
from qc2.data import qc2Data
from qc2.algorithms.qiskit import VQE
from qc2.algorithms.utils import ActiveSpace

# Avoid using the deprecated `PauliSumOp` object
qiskit_nature.settings.use_pauli_sum_op = False


# set Atoms object
mol = molecule("H2")

# file to save data
fcidump_file = "h2_ase_pyscf_qiskit.fcidump"

# init the hdf5 file
qc2data = qc2Data(fcidump_file, mol, schema="fcidump")

# specify the qchem calculator
qc2data.molecule.calc = PySCF()  # default => RHF/STO-3G

# run calculation and save qchem data in the hdf5 file
qc2data.run()


# set up VQE calc
qc2data.algorithm = VQE(
    active_space=ActiveSpace(
        num_active_electrons=(1, 1), num_active_spatial_orbitals=2
    ),
    optimizer=SLSQP(),
    estimator=Estimator(),
    mapper=BravyiKitaevMapper(),
)

# run the calc
result = qc2data.algorithm.run()

# define activate space
n_active_electrons = (1, 1)  # => (n_alpha, n_beta)
n_active_spatial_orbitals = 2

# define the type of fermionic-to-qubit transformation
mapper = BravyiKitaevMapper()
