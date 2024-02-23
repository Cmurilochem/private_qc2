from qiskit_nature.second_q.mappers import JordanWignerMapper
from qiskit_nature.second_q.circuit.library import HartreeFock, UCC
from qiskit_algorithms.minimum_eigensolvers import VQE as vqe_solver
from qiskit.primitives import Estimator
from qiskit_algorithms.optimizers import SLSQP
from ..base.vqe_base import VQEBASE
from ..utils.active_sapce import ActiveSpace


class VQE(VQEBASE):
    """Main class for VQE"""

    def __init__(self, qc2data=None, **kwargs):
        """Initiate the class

        Args:
            qc2data (data container): qc2 data container with scf informatioms
            active_space (ActiveSpace, optional): Description of the active sapce. Defaults to ActiveSpace((2, 2), 2).
            mapper (qiskit_nature.second_q.mappers, optional): Method used to map the qubits. Defaults to JordanWignerMapper().
        """

        super().__init__(qc2data, "qiskit")

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

        # init circuit
        self.estimator = (
            Estimator() if "estimator" not in kwargs else kwargs["estimator"]
        )
        self.optimizer = SLSQP() if "optimizer" not in kwargs else kwargs["optimizer"]
        self.reference_state = (
            self._get_default_reference(self.active_space, self.mapper)
            if "reference_state" not in kwargs
            else kwargs["reference_state"]
        )
        self.ansatz = (
            self._get_default_ansatz(
                self.active_space, self.mapper, self.reference_state
            )
            if "ansatz" not in kwargs
            else kwargs["ansatz"]
        )
        self.params = (
            self._get_default_init_params(self.ansatz.num_parameters)
            if "init_params" not in kwargs
            else kwargs["init_params"]
        )

    @staticmethod
    def _get_default_reference(active_space, mapper):
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
    def _get_default_ansatz(active_space, mapper, reference_state):
        """Set up the default UCC ansatz from a HF reference

        Args:
            active_space (ActiveSpace): description of the active space
            mapper (mapper): mapper class instance
            reference_state (QuantumCircuit): reference state circuit

        Returns:
            QuantumCircuit: circuit ansaz
        """

        return UCC(
            num_spatial_orbitals=active_space.num_active_spatial_orbitals,
            num_particles=active_space.num_active_electrons,
            qubit_mapper=mapper,
            initial_state=reference_state,
            excitations="sdtq",
        )

    @staticmethod
    def _get_default_init_params(nparams):
        """Set up the init para,s

        Args:
            nparams (np.ndarray): default values

        Returns:
            List : List of params values
        """
        return [0.0] * nparams

    def run(self):
        """Run the algo"""

        solver = vqe_solver(self.estimator, self.ansatz, self.optimizer)
        solver.initial_point = self.params
        result = vqe_solver.compute_minimum_eigenvalue(self.qubit_op)

        print("=== QISKIT VQE RESULTS ===")
        print(f"* Electronic ground state energy (Hartree): {result.eigenvalue}")
        print(f"* Inactive core energy (Hartree): {self.e_core}")
        print(
            f">>> Total ground state energy (Hartree): {result.eigenvalue+self.e_core}\n"
        )

        return result
