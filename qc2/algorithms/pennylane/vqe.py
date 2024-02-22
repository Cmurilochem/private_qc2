from qiskit_nature.second_q.mappers import JordanWignerMapper
import pennylane as qml
from pennylane import numpy as np
from ..utils.active_sapce import ActiveSpace
from ..base.vqe_base import VQEBASE


class VQE(VQEBASE):
    """Main class for VQE"""

    def __init__(
        self, qc2data, active_space=ActiveSpace((2, 2), 2), mapper=JordanWignerMapper()
    ):
        """Initiate the class

        Args:
            qc2data (data container): qc2 data container with scf informatioms
            active_space (ActiveSpace, optional): Description of the active sapce. Defaults to ActiveSpace((2, 2), 2).
            mapper (qiskit_nature.second_q.mappers, optional): Method used to map the qubits. Defaults to JordanWignerMapper().
        """
        super().__init_(qc2data, "pennylane", active_space, mapper)

    def run(self, **kwargs):

        def _get_default_reference(qubits, electrons):
            return qml.qchem.hf_state(electrons, qubits)

        def _get_default_ansatz(qubits, electrons, reference_state):

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
                return qml.expval(self.qubit_op)

            return circuit

        def _get_default_init_param(qubits, electrons):
            # Generate single and double excitations
            singles, doubles = qml.qchem.excitations(electrons, qubits)
            return np.zeros(len(singles) + len(doubles))

        niter_default = 21
        qubits = 2 * self.active_space.num_active_spatial_orbitals
        electrons = sum(self.active_space.num_active_electrons)

        optimizer = (
            qml.GradientDescentOptimizer(stepsize=0.5)
            if "optimizer" not in kwargs
            else kwargs["optimizer"]
        )
        reference_state = (
            _get_default_reference(self.active_space, self.mapper)
            if "reference_state" not in kwargs
            else kwargs["reference_state"]
        )
        circuit = (
            _get_default_ansatz(qubits, electrons, reference_state)
            if "ansatz" not in kwargs
            else kwargs["ansatz"]
        )
        params = (
            _get_default_init_param(qubits, electrons)
            if "init_param" not in kwargs
            else kwargs["init_param"]
        )
        niter = niter_default if "niter" not in kwargs else kwargs["niter"]

        # Optimize the circuit parameters and compute the energy
        for _ in range(niter):
            params, energy = optimizer.step_and_cost(circuit, params)

        print("=== PENNYLANE VQE RESULTS ===")
        print(f"* Electronic ground state energy (Hartree): {energy}")
        print(f"* Inactive core energy (Hartree): {self.e_core}")
        print(f">>> Total ground state energy (Hartree): {energy+self.e_core}\n")

        # print(f"+++ Final parameters:{params}")
