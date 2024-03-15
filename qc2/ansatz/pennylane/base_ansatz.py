from qiskit_nature.second_q.mappers import JordanWignerMapper
import pennylane as qml
from ...algorithms.utils import ActiveSpace


class BaseAnsatzPennylane:

    def __init__(self, active_space, mapper):
        self.active_space = active_space
        self.mapper = mapper

        self.qubits = 2 * self.active_space.num_active_spatial_orbitals
        self.electrons = sum(self.active_space.num_active_electrons)

    def get_reference_state(self):
        """Set up default reference state

        Args:
            qubits (_type_): _description_
            electrons (_type_): _description_

        Returns:
            _type_: _description_
        """
        return qml.qchem.hf_state(self.qubits, self.electrons)
