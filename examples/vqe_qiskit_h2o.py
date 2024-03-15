"""Example of a VQE calc using Qiskit-Nature and PYSCF-ASE calculator.

Standard restricted calculation => H2O example.

Notes:
    Requires the installation of qc2, ase, qiskit and h5py.
"""

from ase.build import molecule

from qiskit.primitives import Estimator
from qiskit_algorithms.optimizers import SLSQP

import qiskit_nature
from qiskit_nature.second_q.circuit.library import UCC
from qiskit_nature.second_q.mappers import JordanWignerMapper

from qc2.ase import PySCF
from qc2.data import qc2Data
from qc2.algorithms.qiskit import VQE

# Avoid using the deprecated `PauliSumOp` object
qiskit_nature.settings.use_pauli_sum_op = False

# set Atoms object
mol = molecule("H2O")

# file to save data
hdf5_file = "h2o_pyscf_qiskit.hdf5"

# init the hdf5 file
qc2data = qc2Data(hdf5_file, mol)

# specify the qchem calculator
qc2data.molecule.calc = PySCF()  # default => RHF/STO-3G

# run calculation and save qchem data in the hdf5 file
qc2data.run()

# set up VQE calc
ansatz = UCC(4, (2, 2), "sd", JordanWignerMapper())
estimator = Estimator()
opt = SLSQP()
qc2data.algorithm = VQE(ansatz=ansatz, estimator=estimator, optimizer=opt)

# run the qc calc
qc2data.algorithm.run()
