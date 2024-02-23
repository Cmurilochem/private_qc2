from qiskit_nature.second_q.mappers import JordanWignerMapper
import pennylane as qml
from pennylane import numpy as np
from ..utils.active_sapce import ActiveSpace
from ..base.vqe_base import VQEBASE


class VQE(VQEBASE):
    """Main class for VQE"""

    def __init__(self, qc2data=None, **kwargs):
        """Initiate the class

        Args:
            qc2data (data container): qc2 data container with scf informatioms
            active_space (ActiveSpace, optional): Description of the active sapce. Defaults to ActiveSpace((2, 2), 2).
            mapper (qiskit_nature.second_q.mappers, optional): Method used to map the qubits. Defaults to JordanWignerMapper().
        """
        super().__init__(qc2data, "pennylane")

        # init active space and mapper
        self.active_space = (
            ActiveSpace((2, 2), 2)
            if "active_space" not in kwargs
            else kwargs["active_space"]
        )
        self.mapper = (
            JordanWignerMapper() if "mapper" not in kwargs else kwargs["mapper"]
        )

        # create Hamiltonian
        self._init_qubit_hamiltonian(self, self.format, self.active_space, self.mapper)

        self.qubits = 2 * self.active_space.num_active_spatial_orbitals
        self.electrons = sum(self.active_space.num_active_electrons)

        self.optimizer = (
            qml.GradientDescentOptimizer(stepsize=0.5)
            if "optimizer" not in kwargs
            else kwargs["optimizer"]
        )
        self.reference_state = (
            self._get_default_reference(self.active_space, self.mapper)
            if "reference_state" not in kwargs
            else kwargs["reference_state"]
        )
        self.circuit = (
            self._get_default_ansatz(
                self.qubits, self.electrons, self.reference_state, self.qubit_op
            )
            if "ansatz" not in kwargs
            else kwargs["ansatz"]
        )
        self.params = (
            self._get_default_init_param(self.qubits, self.electrons)
            if "init_param" not in kwargs
            else kwargs["init_param"]
        )

    @staticmethod
    def _get_default_reference(qubits, electrons):
        """Set up default reference state

        Args:
            qubits (_type_): _description_
            electrons (_type_): _description_

        Returns:
            _type_: _description_
        """
        return qml.qchem.hf_state(electrons, qubits)

    @staticmethod
    def _get_default_ansatz(qubits, electrons, reference_state, qubit_op):
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

        # Define the device
        dev = qml.device("default.qubit", wires=qubits)

        # Define the qnode
        @qml.qnode(dev)
        def circuit(params):
            qml.UCCSD(params, range(qubits), s_wires, d_wires, reference_state)
            return qml.expval(qubit_op)

        return circuit

    @staticmethod
    def _get_default_init_param(qubits, electrons):
        # Generate single and double excitations
        singles, doubles = qml.qchem.excitations(electrons, qubits)
        return np.zeros(len(singles) + len(doubles))

    def run(self, niter=21):
        """Run the algo"""

        # Optimize the circuit parameters and compute the energy
        for _ in range(niter):
            params, energy = self.optimizer.step_and_cost(self.circuit, self.params)

        print("=== PENNYLANE VQE RESULTS ===")
        print(f"* Electronic ground state energy (Hartree): {energy}")
        print(f"* Inactive core energy (Hartree): {self.e_core}")
        print(f">>> Total ground state energy (Hartree): {energy+self.e_core}\n")

        # print(f"+++ Final parameters:{params}")
