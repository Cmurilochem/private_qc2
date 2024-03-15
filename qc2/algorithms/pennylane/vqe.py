"""Module defining VQE algorithm for PennyLane."""

from typing import List, Tuple, Callable
from qiskit_nature.second_q.mappers import JordanWignerMapper
import pennylane as qml
from pennylane import numpy as np
from pennylane.qnode import QNode
from pennylane.operation import Operator
from qc2.algorithms.utils import ActiveSpace
from qc2.algorithms.base import VQEBASE


class VQE(VQEBASE):
    """Main class for VQE"""

    def __init__(
        self,
        qc2data=None,
        ansatz=None,
        active_space=None,
        mapper=None,
        device=None,
        optimizer=None,
        reference_state=None,
        init_params=None,
        max_iterations=50,
        conv_tol=1e-7,
        verbose=0,
    ):
        """Initiate the class

        Args:
            qc2data (data container): qc2 data container with scf informatioms
            active_space (ActiveSpace, optional): Description of the active sapce. Defaults to ActiveSpace((2, 2), 2).
            mapper (qiskit_nature.second_q.mappers, optional): Method used to map the qubits. Defaults to JordanWignerMapper().
        """
        super().__init__(qc2data, "pennylane")

        # use the provided ansatz
        if ansatz is not None:
            self.ansatz = ansatz
            if any(
                param is not None
                for param in [active_space, mapper, device, reference_state]
            ):
                raise ValueError(
                    "active_space, mapper, device and reference_state arguments must be None when passing an ansatz"
                )

        # create a default ansatz
        else:
            # init active space and mapper
            self.active_space = (
                ActiveSpace((1, 1), 2) if active_space is None else active_space
            )

            # init circuit
            self.device = "default.qubit" if device is None else device
            self.mapper = JordanWignerMapper() if mapper is None else mapper
            self.qubits = 2 * self.active_space.num_active_spatial_orbitals
            self.electrons = sum(self.active_space.num_active_electrons)

            self.reference_state = self._get_default_reference(
                self.qubits, self.electrons
            )

            self.ansatz = self._get_default_ansatz(
                self.qubits, self.electrons, self.reference_state
            )

        # init optimizer
        self.optimizer = (
            qml.GradientDescentOptimizer(stepsize=0.5)
            if optimizer is None
            else optimizer
        )

        # init params ansatz
        self.params = (
            self._get_default_init_param(self.qubits, self.electrons)
            if init_params is None
            else init_params
        )

        # init algorithm-specific attributes
        self.max_iterations = max_iterations
        self.conv_tol = conv_tol
        self.verbose = verbose
        self.circuit = None
        self.energy = None

    @staticmethod
    def _get_default_reference(qubits: int, electrons: int) -> np.ndarray:
        """Set up default reference state

        Args:
            qubits (_type_): _description_
            electrons (_type_): _description_

        Returns:
            _type_: _description_
        """
        return qml.qchem.hf_state(electrons, qubits)

    @staticmethod
    def _get_default_ansatz(
        qubits: int, electrons: int, reference_state: np.ndarray
    ) -> Callable:
        """Set up default ansatz

        Args:
            qubits (_type_): _description_
            electrons (_type_): _description_
            reference_state (_type_): _description_

        Returns:
            _type_: _description_
        """

        # Generate single and double excitations
        singles, doubles = qml.qchem.excitations(electrons, qubits)

        # Map excitations to the wires the UCCSD circuit will act on
        s_wires, d_wires = qml.qchem.excitations_to_wires(singles, doubles)

        # Return a function that applies the UCCSD ansatz
        def ansatz(params):
            qml.UCCSD(
                params,
                wires=range(qubits),
                s_wires=s_wires,
                d_wires=d_wires,
                init_state=reference_state,
            )

        return ansatz

    @staticmethod
    def _get_default_init_param(qubits: int, electrons: int) -> np.ndarray:
        # Generate single and double excitations
        singles, doubles = qml.qchem.excitations(electrons, qubits)
        return np.zeros(len(singles) + len(doubles))

    @staticmethod
    def _build_circuit(
        dev: str,
        qubits: int,
        ansatz: Callable,
        qubit_op: Operator,
        device_args=None,
        device_kwargs=None,
        qnode_args=None,
        qnode_kwargs=None,
    ) -> QNode:
        """Build and return a quantum circuit."""
        # Set default values if None
        device_args = device_args if device_args is not None else []
        device_kwargs = device_kwargs if device_kwargs is not None else {}
        qnode_args = qnode_args if qnode_args is not None else []
        qnode_kwargs = qnode_kwargs if qnode_kwargs is not None else {}

        # Define the device
        device = qml.device(dev, wires=qubits, *device_args, **device_kwargs)

        # Define the QNode and call the ansatz function within it
        @qml.qnode(device, *qnode_args, **qnode_kwargs)
        def circuit(params):
            ansatz(params)
            return qml.expval(qubit_op)

        return circuit

    def run(self, *args, **kwargs) -> Tuple[List, List]:
        """Run the algo"""
        print(">>> Optimizing circuit parameters...")

        # create Hamiltonian
        self._init_qubit_hamiltonian()

        # build circuit after building the qubit hamiltonian
        self.circuit = self._build_circuit(
            self.device, self.qubits, self.ansatz, self.qubit_op, *args, **kwargs
        )

        theta = self.params
        energy_l = []
        theta_l = []

        # Optimize the circuit parameters and compute the energy
        for n in range(self.max_iterations):
            theta, corr_energy = self.optimizer.step_and_cost(self.circuit, theta)
            energy = corr_energy + self.e_core
            energy_l.append(energy)
            theta_l.append(theta)

            if self.verbose is not None:
                if n % 2 == 0:
                    print(f"iter = {n:03}, energy = {energy_l[-1]:.12f} Ha")

            if n > 1:
                if abs(energy_l[-1] - energy_l[-2]) < self.conv_tol:
                    # save final parameters
                    if self.verbose is not None:
                        self.params = theta_l[-1]
                        self.energy = energy_l[-1]
                        print("optimization finished.\n")
                        print("=== PENNYLANE VQE RESULTS ===")
                        print(
                            "* Electronic ground state "
                            f"energy (Hartree): {corr_energy:.12f}"
                        )
                        print(
                            "* Inactive core " f"energy (Hartree): {self.e_core:.12f}"
                        )
                        print(
                            ">>> Total ground state "
                            f"energy (Hartree): {self.energy:.12f}\n"
                        )
                    break
        # in case of non-convergence
        else:
            raise RuntimeError(
                "Optimization did not converge within the maximum iterations."
                " Consider increasing 'max_iterations' attribute or"
                " setting a different 'optimizer'."
            )

        return energy_l, theta_l
