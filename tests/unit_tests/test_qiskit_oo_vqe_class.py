import os
import pytest
import numpy as np

from ase.build import molecule

from qc2.data import qc2Data
from qc2.ase import PySCF
from qc2.algorithms.utils import ActiveSpace
from qc2.algorithms.qiskit import oo_VQE


@pytest.fixture
def qc2data():
    """Fixture to set up qc2Data instance."""
    # Create a temporary file for testing
    tmp_filename = str('test_qc2data.h5')

    # Create an ASE Atoms instance for testing
    mol = molecule('H2')

    # Create the qc2Data instance
    qc2_data = qc2Data(tmp_filename, mol, schema='qcschema')
    qc2_data.molecule.calc = PySCF()
    yield qc2_data

    # Clean up the temporary file after the tests
    os.remove(tmp_filename)


@pytest.fixture
def oo_vqe(qc2data):
    """Fixture to set up oo_VQE instance."""
    qc2data.run()
    active_space = ActiveSpace(
        num_active_electrons=(1, 1),
        num_active_spatial_orbitals=2
    )
    oo_vqe_instance = oo_VQE(qc2data, active_space=active_space)
    yield oo_vqe_instance


def test_initialization(oo_vqe):
    """Test if you can initialize the class."""
    assert isinstance(oo_vqe, oo_VQE)


def test_theta_kappa_optimization(oo_vqe):
    """Test optimization workflow."""
    # optimize all params
    energy_l, theta_l, kappa_l = oo_vqe.run()
    # calculate energies with best params
    opt_theta, energy_opt_theta = oo_vqe._circuit_optimization(
        oo_vqe.circuit_params,
        oo_vqe.orbital_params
    )
    energy_from_params = oo_vqe._get_energy_from_parameters(
        oo_vqe.circuit_params,
        oo_vqe.orbital_params
    )

    assert all(isinstance(num, float) for num in energy_l)
    assert all(isinstance(term, np.ndarray) for term in theta_l)
    assert all(isinstance(term, list) for term in kappa_l)
    assert all(num != 0 for num in oo_vqe.orbital_params)
    assert oo_vqe.energy == pytest.approx(-1.1373015, 1e-6)
    assert energy_opt_theta == pytest.approx(oo_vqe.energy, 1e-6)
    assert energy_from_params == pytest.approx(oo_vqe.energy, 1e-6)
    assert opt_theta == pytest.approx(oo_vqe.circuit_params, 1e-6)
