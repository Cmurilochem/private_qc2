from qiskit_nature.second_q.mappers import JordanWignerMapper
from qiskit_nature.second_q.circuit.library import HartreeFock, UCC
from qiskit_algorithms.minimum_eigensolvers import VQE as vqe_solver
from qiskit.primitives import Estimator
from qiskit_algorithms.optimizers import SLSQP
from ..base.vqe_base import VQEBASE
from ..utils.active_space import ActiveSpace


class VQE(VQEBASE):
    """Main class for VQE"""

    def __init__(
        self,
        qc2data=None,
        ansatz=None,
        estimator=None,
        optimizer=None,
        init_params=None,
    ):
        """Initiate the class

        Args:
            qc2data (data container): qc2 data container with scf informatioms
            active_space (ActiveSpace, optional): Description of the active sapce. Defaults to ActiveSpace((2, 2), 2).
            mapper (qiskit_nature.second_q.mappers, optional): Method used to map the qubits. Defaults to JordanWignerMapper().
        """

        super().__init__(qc2data, "qiskit")

        # init qiskit solver utils
        self.estimator = Estimator() if estimator is None else estimator
        self.optimizer = SLSQP() if optimizer is None else optimizer

        # load the ansatz
        if ansatz is not None:
            self.ansatz = ansatz
            self.mapper = ansatz.qubit_mapper
            self.active_space = ActiveSpace(
                ansatz.num_particles, ansatz.num_spatial_orbitals
            )

        # or create the default one
        else:
            self.active_space = ActiveSpace((1, 1), 2)
            self.mapper = JordanWignerMapper()
            self.ansatz = self._get_default_ansatz(self.active_space, self.mapper)

        # init parameters
        self.params = (
            self._get_default_init_params(self.ansatz.num_parameters)
            if init_params is None
            else init_params
        )

    @staticmethod
    def _get_default_ansatz(active_space, mapper):
        """Set up the default UCC ansatz from a HF reference

        Args:
            active_space (ActiveSpace): description of the active space
            mapper (mapper): mapper class instance

        Returns:
            QuantumCircuit: circuit ansatz
        """
        # reference HF state
        reference_state = HartreeFock(
            active_space.num_active_spatial_orbitals,
            active_space.num_active_electrons,
            mapper,
        )

        return UCC(
            num_spatial_orbitals=active_space.num_active_spatial_orbitals,
            num_particles=active_space.num_active_electrons,
            qubit_mapper=mapper,
            initial_state=reference_state,
            excitations="sd",
        )

    @staticmethod
    def _get_default_init_params(nparams):
        """Set up the init params

        Args:
            nparams (np.ndarray): default values

        Returns:
            List : List of params values
        """
        return [0.0] * nparams

    def run(self):
        """Run the algo"""

        # create Hamiltonian
        self._init_qubit_hamiltonian()

        solver = vqe_solver(self.estimator, self.ansatz, self.optimizer)
        solver.initial_point = self.params
        result = solver.compute_minimum_eigenvalue(self.qubit_op)

        print("=== QISKIT VQE RESULTS ===")
        print(f"* Electronic ground state energy (Hartree): {result.eigenvalue}")
        print(f"* Inactive core energy (Hartree): {self.e_core}")
        print(
            f">>> Total ground state energy (Hartree): {result.eigenvalue+self.e_core}\n"
        )

        return result
