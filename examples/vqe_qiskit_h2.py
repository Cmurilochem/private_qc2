"""Example of a VQE calc using Qiskit-Nature and DIRAC-ASE calculator.

Standard restricted calculation => H2 example.

Notes:
    Requires the installation of qc2, ase, qiskit and h5py.
"""

import subprocess
from ase.build import molecule

import qiskit_nature
from qiskit_algorithms.optimizers import SLSQP
from qiskit.primitives import Estimator

from qc2.ase import DIRAC
from qc2.data import qc2Data
from qc2.algorithms.qiskit import VQE

# Avoid using the deprecated `PauliSumOp` object
qiskit_nature.settings.use_pauli_sum_op = False


def clean_up_DIRAC_files():
    """Remove DIRAC calculation outputs."""
    command = "rm dirac* MDCINT* MRCONEE* FCIDUMP* AOMOMAT* FCI*"
    subprocess.run(command, shell=True, capture_output=True)


# set Atoms object
mol = molecule("H2")

# file to save data
hdf5_file = "h2_dirac_qiskit.hdf5"

# init the hdf5 file
qc2data = qc2Data(hdf5_file, mol)

# specify the qchem calculator
qc2data.molecule.calc = DIRAC()  # default => RHF/STO-3G

# run calculation and save qchem data in the hdf5 file
qc2data.run()

# set up VQE calc
qc2data.algorithm = VQE(
    optimizer=SLSQP(),
    estimator=Estimator(),
)

# run vqe
qc2Data.algorithm.run()

clean_up_DIRAC_files()
