from qiskit_nature.second_q.mappers import JordanWignerMapper
from qiskit_nature.second_q.circuit.library import HartreeFock, UCC
from qiskit_algorithms.minimum_eigensolvers import VQE as vqe_solver
from qiskit.primitives import Estimator
from qiskit_algorithms.optimizers import SLSQP
from ..base.vqe_base import VQEBASE
from ..utils.active_sapce import ActiveSpace


class VQE(VQEBASE):
    """Main class for VQE"""

    def __init__(
        self,
        qc2data,
        active_space=ActiveSpace((2, 2), 2),
        mapper=JordanWignerMapper(),
    ):
        """Initiate the class

        Args:
            qc2data (data container): qc2 data container with scf informatioms
            active_space (ActiveSpace, optional): Description of the active sapce. Defaults to ActiveSpace((2, 2), 2).
            mapper (qiskit_nature.second_q.mappers, optional): Method used to map the qubits. Defaults to JordanWignerMapper().
        """

        super().__init__(qc2data, "qiskit", active_space=active_space, mapper=mapper)

    def run(self, **kwargs):

        def _get_default_reference(active_space, mapper):
            return HartreeFock(
                active_space.num_active_spatial_orbitals,
                active_space.num_active_electrons,
                mapper,
            )

        def _get_default_ansatz(active_space, mapper, reference_state):

            return UCC(
                num_spatial_orbitals=active_space.num_active_spatial_orbitals,
                num_particles=active_space.num_active_electrons,
                qubit_mapper=mapper,
                initial_state=reference_state,
                excitations="sdtq",
            )

        def _get_default_init_params(nparams):
            return [0.0] * nparams

        estimator = Estimator() if "estimator" not in kwargs else kwargs["estimator"]
        optimizer = SLSQP() if "optimizer" not in kwargs else kwargs["optimizer"]
        reference_state = (
            _get_default_reference(self.active_space, self.mapper)
            if "reference_state" not in kwargs
            else kwargs["reference_state"]
        )
        ansatz = (
            _get_default_ansatz(self.active_space, self.mapper, reference_state)
            if "ansatz" not in kwargs
            else kwargs["ansatz"]
        )
        params = (
            _get_default_init_params(self.ansatz.num_parameters)
            if "init_params" not in kwargs
            else kwargs["init_params"]
        )

        solver = vqe_solver(estimator, ansatz, optimizer)
        solver.initial_point = params
        result = vqe_solver.compute_minimum_eigenvalue(self.qubit_op)

        print("=== QISKIT VQE RESULTS ===")
        print(f"* Electronic ground state energy (Hartree): {result.eigenvalue}")
        print(f"* Inactive core energy (Hartree): {self.e_core}")
        print(
            f">>> Total ground state energy (Hartree): {result.eigenvalue+self.e_core}\n"
        )

        return result
