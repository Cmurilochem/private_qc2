"""Module defining VQE algorithm for Qiskit-Nature."""

from typing import List, Tuple, Dict
from qiskit_nature.second_q.mappers import JordanWignerMapper
from qiskit_nature.second_q.circuit.library import HartreeFock, UCC
from qiskit_nature.second_q.mappers import QubitMapper
from qiskit_algorithms.minimum_eigensolvers import VQE as vqe_solver
from qiskit_algorithms.minimum_eigensolvers import VQEResult
from qiskit.primitives import Estimator
from qiskit.circuit import QuantumCircuit
from qiskit_algorithms.optimizers import SLSQP
from qc2.algorithms.base import VQEBASE
from qc2.algorithms.utils import ActiveSpace


class VQE(VQEBASE):
    """Main class for VQE"""

    def __init__(
        self,
        qc2data=None,
        ansatz=None,
        active_space=None,
        mapper=None,
        estimator=None,
        optimizer=None,
        reference_state=None,
        init_params=None,
        verbose=0,
    ):
        """Initiate the class

        Args:
            qc2data (data container): qc2 data container with scf informatioms
            active_space (ActiveSpace, optional): Description of the active sapce. Defaults to ActiveSpace((2, 2), 2).
            mapper (qiskit_nature.second_q.mappers, optional): Method used to map the qubits. Defaults to JordanWignerMapper().
        """

        super().__init__(qc2data, "qiskit")

        # init active space and mapper
        self.active_space = (
            ActiveSpace((2, 2), 2) if active_space is None else active_space
        )
        self.mapper = JordanWignerMapper() if mapper is None else mapper

        # init circuit
        self.estimator = Estimator() if estimator is None else estimator
        self.optimizer = SLSQP() if optimizer is None else optimizer
        self.reference_state = (
            self._get_default_reference(self.active_space, self.mapper)
            if reference_state is None
            else reference_state
        )
        self.ansatz = (
            self._get_default_ansatz(
                self.active_space, self.mapper, self.reference_state
            )
            if ansatz is None
            else ansatz
        )
        self.params = (
            self._get_default_init_params(self.ansatz.num_parameters)
            if init_params is None
            else init_params
        )

        # init algorithm-specific attributes
        self.verbose = verbose
        self.energy = None

    @staticmethod
    def _get_default_reference(
        active_space: ActiveSpace, mapper: QubitMapper
    ) -> QuantumCircuit:
        """Set up the default reference state circuit

        Args:
            active_space (ActiveSpace): description of the active space
            mapper (mapper): mapper class instance

        Returns:
            QuantumCircuit: hartree fock circuit
        """
        return HartreeFock(
            active_space.num_active_spatial_orbitals,
            active_space.num_active_electrons,
            mapper,
        )

    @staticmethod
    def _get_default_ansatz(
        active_space: ActiveSpace, mapper: QubitMapper, reference_state: QuantumCircuit
    ) -> UCC:
        """Set up the default UCC ansatz from a HF reference

        Args:
            active_space (ActiveSpace): description of the active space
            mapper (mapper): mapper class instance
            reference_state (QuantumCircuit): reference state circuit

        Returns:
            QuantumCircuit: circuit ansatz
        """

        return UCC(
            num_spatial_orbitals=active_space.num_active_spatial_orbitals,
            num_particles=active_space.num_active_electrons,
            qubit_mapper=mapper,
            initial_state=reference_state,
            excitations="sd",
        )

    @staticmethod
    def _get_default_init_params(nparams: List) -> List:
        """Set up the init para,s

        Args:
            nparams (List): default values

        Returns:
            List : List of params values
        """
        return [0.0] * nparams

    def run(self, *args, **kwargs) -> Tuple[VQEResult, Dict]:
        """Run the algo"""
        # create Hamiltonian
        self._init_qubit_hamiltonian()

        # create a simple callback to print intermediate results
        intermediate_info = {"nfev": [], "parameters": [], "energy": [], "metadata": []}

        def callback(nfev, parameters, energy, metadata):
            intermediate_info["nfev"].append(nfev)
            intermediate_info["parameters"].append(parameters)
            intermediate_info["energy"].append(energy + self.e_core)
            intermediate_info["metadata"].append(metadata)
            if self.verbose is not None:
                if nfev % 2 == 0:
                    print(
                        f"iter = {intermediate_info['nfev'][-1]:03}, "
                        f"energy = {intermediate_info['energy'][-1]:.12f} Ha"
                    )

        # instantiate the solver
        solver = vqe_solver(
            self.estimator,
            self.ansatz,
            self.optimizer,
            callback=callback,
            *args,
            **kwargs,
        )
        solver.initial_point = self.params

        # call the minimizer and save final results
        result = solver.compute_minimum_eigenvalue(self.qubit_op)
        self.params = intermediate_info["parameters"][-1]
        self.energy = intermediate_info["energy"][-1]

        print("=== QISKIT VQE RESULTS ===")
        print("* Electronic ground state " f"energy (Hartree): {result.eigenvalue}")
        print(f"* Inactive core energy (Hartree): {self.e_core}")
        print(f">>> Total ground state energy (Hartree): {self.energy}\n")

        return result, intermediate_info
