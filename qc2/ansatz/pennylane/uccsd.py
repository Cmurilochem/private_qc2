from qiskit_nature.second_q.mappers import JordanWignerMapper
import pennylane as qml
from .base_ansatz import BaseAnsatzPennylane
from ...algorithms.utils import ActiveSpace


class UCCSD(BaseAnsatzPennylane):

    def __init__(self, active_space=None, mapper=JordanWignerMapper()):
        super().__init__(active_space, mapper)
