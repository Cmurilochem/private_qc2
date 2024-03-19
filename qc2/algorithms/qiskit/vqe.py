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
    """
    Main class for the VQE algorithm with Qiskit-Nature.

    This class initializes and executes the VQE algorithm using specified
    quantum components like ansatz, optimizer, and estimator.

    Attributes:
        active_space (ActiveSpace): Describes the active space for quantum
            simulation. Defaults to ``ActiveSpace((2, 2), 2)``.
        mapper (QubitMapper): Strategy for fermionic-to-qubit mapping.
            Defaults to :class:`qiskit.JordanWignerMapper`.
        estimator (BaseEstimator): Method for estimating the
            expectation value. Defaults to :class:`qiskit.Estimator`
        optimizer (qiskit.Optmizer): Optimization routine for circuit
            variational parameters. Defaults to
            :class:`qiskit_algorithms.SLSQP`.
        reference_state (QuantumCircuit): Reference state for the VQE
            algorithm. Defaults to :class:`qiskit.HartreeFock`.
        ansatz (UCC): The ansatz for the VQE algorithm.
            Defaults to :class:`qiskit.UCCSD`.
        params (List): List of VQE circuit parameters. It gets updated
            during the optimization process. Defaults to a list with
            entries of zero.
        verbose (int): Verbosity level. Defaults to 0.
        energy (float): Calculated energy value after running VQE.
            Defaults to None.
    """

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
        verbose=0
    ):
        """Initializes the VQE class.

        Args:
            qc2data (qc2Data): An instance of :class:`~qc2.data.qc2Data`.
            ansatz (UCC): The ansatz for the VQE algorithm.
                Defaults to :class:`qiskit.UCCSD`.
            active_space (ActiveSpace): Describes the active space for quantum
                simulation. Defaults to ``ActiveSpace((2, 2), 2)``.
            mapper (QubitMapper): Strategy for fermionic-to-qubit mapping.
                Defaults to :class:`qiskit.JordanWignerMapper`.
            estimator (BaseEstimator): Method for estimating the
                expectation value. Defaults to :class:`qiskit.Estimator`
            optimizer (qiskit.Optmizer): Optimization routine for circuit
                variational parameters. Defaults to
                :class:`qiskit_algorithms.SLSQP`.
            reference_state (QuantumCircuit): Reference state for the VQE
                algorithm. Defaults to :class:`qiskit.HartreeFock`.
            init_params (List): List of VQE circuit parameters.
                Defaults to a list with entries of zero.
            verbose (int): Verbosity level. Defaults to 0.

        **Example**

        >>> from ase.build import molecule
        >>> from qc2.ase import PySCF
        >>> from qc2.data import qc2Data
        >>> from qc2.algorithms.qiskit import VQE
        >>> from qc2.algorithms.utils import ActiveSpace
        >>>
        >>> mol = molecule('H2O')
        >>>
        >>> hdf5_file = 'h2o.hdf5'
        >>> qc2data = qc2Data(hdf5_file, mol, schema='qcschema')
        >>> qc2data.molecule.calc = PySCF()
        >>> qc2data.run()
        >>> qc2data.algorithm = VQE(
        ...     active_space=ActiveSpace(
        ...         num_active_electrons=(2, 2),
        ...         num_active_spatial_orbitals=4
        ...     ),
        ...     optimizer=SLSQP(),
        ...     estimator=Estimator(),
        ... )
        >>> result, intermediate_info = qc2data.algorithm.run()
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
        """Set up the default reference state circuit based on Hartree Fock.

        Args:
            active_space (ActiveSpace): description of the active space.
            mapper (mapper): mapper class instance.

        Returns:
            QuantumCircuit: Hartree Fock circuit as the reference state.
        """
        return HartreeFock(
            active_space.num_active_spatial_orbitals,
            active_space.num_active_electrons,
            mapper,
        )

    @staticmethod
    def _get_default_ansatz(
        active_space: ActiveSpace,
        mapper: QubitMapper,
        reference_state: QuantumCircuit
    ) -> UCC:
        """Set up the default UCC ansatz from a Hartree Fock reference state.

        Args:
            active_space (ActiveSpace): Description of the active space.
            mapper (QubitMapper): Mapper class instance.
            reference_state (QuantumCircuit): Reference state circuit.

        Returns:
            UCC: UCC ansatz quantum circuit.
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
        """Generates a list of initial circuit parameters for the ansatz.

        Args:
            nparams (int): Number of parameters in the ansatz.

        Returns:
            List[float]: List of initial parameter values (all zeros).
        """
        return [0.0] * nparams

    def run(self, *args, **kwargs) -> Tuple[VQEResult, Dict]:
        """Executes the VQE algorithm.

        Args:
            *args: Variable length argument list to be passed to
                the :class:`qiskit_algorithm.VQE` class.
            **kwargs: Arbitrary keyword arguments to be passed to
                the :class:`qiskit_algorithm.VQE` class.

        Returns:
            Tuple[VQEResult, Dict]:
                The VQE result and a dictionary with intermediate information.

        **Example**

        >>> from ase.build import molecule
        >>> from qc2.ase import PySCF
        >>> from qc2.data import qc2Data
        >>> from qc2.algorithms.qiskit import VQE
        >>> from qc2.algorithms.utils import ActiveSpace
        >>>
        >>> mol = molecule('H2O')
        >>>
        >>> hdf5_file = 'h2o.hdf5'
        >>> qc2data = qc2Data(hdf5_file, mol, schema='qcschema')
        >>> qc2data.molecule.calc = PySCF()
        >>> qc2data.run()
        >>> qc2data.algorithm = VQE(
        ...     active_space=ActiveSpace(
        ...         num_active_electrons=(2, 2),
        ...         num_active_spatial_orbitals=4
        ...     ),
        ...     optimizer=SLSQP(),
        ...     estimator=Estimator(),
        ... )
        >>> result, intermediate_info = qc2data.algorithm.run()
        """
        # create Hamiltonian
        self._init_qubit_hamiltonian()

        # create a simple callback to print intermediate results
        intermediate_info = {
            "nfev": [],
            "parameters": [],
            "energy": [],
            "metadata": []
        }

        def callback(nfev, parameters, energy, metadata):
            intermediate_info['nfev'].append(nfev)
            intermediate_info['parameters'].append(parameters)
            intermediate_info['energy'].append(energy+self.e_core)
            intermediate_info['metadata'].append(metadata)
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
            **kwargs
        )
        solver.initial_point = self.params

        # call the minimizer and save final results
        result = solver.compute_minimum_eigenvalue(self.qubit_op)
        self.params = intermediate_info['parameters'][-1]
        self.energy = intermediate_info['energy'][-1]

        print("=== QISKIT VQE RESULTS ===")
        print("* Electronic ground state "
              f"energy (Hartree): {result.eigenvalue}")
        print(f"* Inactive core energy (Hartree): {self.e_core}")
        print(
            f">>> Total ground state energy (Hartree): {self.energy}\n"
        )

        return result, intermediate_info
