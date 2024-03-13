from qiskit_nature.second_q.mappers import JordanWignerMapper
import pennylane as qml
from pennylane import numpy as np
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
        optimizer=None,
        reference_state=None,
        init_params=None,
    ):
        """Initiate the class

        Args:
            qc2data (data container): qc2 data container with scf informatioms
            active_space (ActiveSpace, optional): Description of the active sapce. Defaults to ActiveSpace((2, 2), 2).
            mapper (qiskit_nature.second_q.mappers, optional): Method used to map the qubits. Defaults to JordanWignerMapper().
        """
        super().__init__(qc2data, "pennylane")

        # init active space and mapper
        self.active_space = (
            ActiveSpace((2, 2), 2) if active_space is None else active_space
        )
        self.mapper = JordanWignerMapper() if mapper is None else mapper

        self.qubits = 2 * self.active_space.num_active_spatial_orbitals
        self.electrons = sum(self.active_space.num_active_electrons)

        self.optimizer = (
            qml.GradientDescentOptimizer(stepsize=0.5)
            if optimizer is None
            else optimizer
        )
        self.reference_state = (
            self._get_default_reference(self.qubits, self.electrons)
            if reference_state is None
            else reference_state
        )
        self.ansatz = (
            self._get_default_ansatz(
                self.qubits, self.electrons, self.reference_state
            )
            if ansatz is None
            else ansatz
        )
        self.params = (
            self._get_default_init_param(self.qubits, self.electrons)
            if init_params is None
            else init_params
        )
        self.circuit = None
        self.result = None

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
    def _get_default_ansatz(qubits, electrons, reference_state):
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
                params, wires=range(qubits), s_wires=s_wires,
                d_wires=d_wires, init_state=reference_state
            )
        return ansatz

    @staticmethod
    def _get_default_init_param(qubits, electrons):
        # Generate single and double excitations
        singles, doubles = qml.qchem.excitations(electrons, qubits)
        return np.zeros(len(singles) + len(doubles))

    @staticmethod
    def _build_circuit(dev, qubits, ansatz, qubit_op):
        """Build and return a quantum circuit."""
        # Define the device
        device = qml.device(dev, wires=qubits)

        # Define the QNode and call the ansatz function within it
        @qml.qnode(device)
        def circuit(params):
            ansatz(params)
            return qml.expval(qubit_op)

        return circuit

    def run(self, niter=21):
        """Run the algo"""

        # create Hamiltonian
        self._init_qubit_hamiltonian()

        # build circuit after having qubit hamiltonian
        self.circuit = self._build_circuit(
            "default.qubit", self.qubits, self.ansatz, self.qubit_op
        )

        # Optimize the circuit parameters and compute the energy
        for _ in range(niter):
            self.params, self.result = self.optimizer.step_and_cost(self.circuit, self.params)
            #if n % 2 == 0:
            #    print("step = {:},  E = {:.8f} Ha".format(n, self.result))

        print("=== PENNYLANE VQE RESULTS ===")
        print(f"* Electronic ground state energy (Hartree): {self.result}")
        print(f"* Inactive core energy (Hartree): {self.e_core}")
        print(f">>> Total ground state energy (Hartree): {self.result+self.e_core}\n")

        # print(f"+++ Final parameters:{params}")
