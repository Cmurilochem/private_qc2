"""Module defining orbital optimization class util."""
from typing import List, Tuple, Optional, Union
import numpy as np
from scipy.linalg import expm
from scipy.optimize import minimize
from pennylane.operation import Operator
from qiskit.quantum_info import SparsePauliOp
from qiskit_nature.second_q.mappers import QubitMapper, JordanWignerMapper
from qiskit_nature.second_q.problems import ElectronicBasis
from qiskit_nature.second_q.operators.tensor_ordering import to_chemist_ordering
from qc2.data import qc2Data
from qc2.algorithms.utils import (
    ActiveSpace,
    vector_to_skew_symmetric,
    skew_symmetric_to_vector,
    get_active_space_idx,
    reshape_2,
    get_non_redundant_indices
)


class OrbitalOptimization():
    """
    A class to perform orbital optimization for quantum chemical systems.

    This class is responsible for setting up and managing the data related to
    orbital optimization part of the oo-VQE algorithm.

    Attributes:
        qc2data (qc2Data): An instance of :class:`~qc2.data.qc2Data`.
        schema_dataclass (QCSchema): An instance of :class:`QCSchema`.
        es_problem (ElectronicStructureProblem): The electronic structure
            problem in AO basis as processed from the schema.
        n_electrons (Tuple[int, int]): Number of alpha and beta electrons.
        nao (int): Number of spatial orbitals.
        n_active_orbitals (int): Number of active orbitals to consider.
        n_active_electrons (Tuple[int, int]): Number of active electrons.
        occ_idx (list): Indices of occupied molecular orbitals.
        act_idx (list): Indices of active molecular orbitals.
        virt_idx (list): Indices of virtual molecular orbitals.
        params_idx (list): Indices for non-redundant orbital rotations.
        n_kappa (int): Dimension of the kappa vector for orbital rotations.
        mapper (QubitMapper): The fermionic-to-qubit mapping algorithm.
    """
    def __init__(
                self,
                qc2data: qc2Data,
                active_space: ActiveSpace,
                freeze_active: bool = False,
                mapper: QubitMapper = JordanWignerMapper(),
                format: str = "qiskit"
    ) -> None:
        """
        Initializes the OrbitalOptimization class.

        Args:
            qc2data (qc2Data): An instance of qc2Data containing quantum
                chemistry information.
            active_space (ActiveSpace): Instance of
                :class:`~qc2.algorithms.utils.ActiveSpace`
                containing the description of the active space.
            freeze_active (bool, optional): If True, active orbitals will be
                frozen in the optimization process. Defaults to False.
            mapper (QubitMapper, optional): An instance of :class:`QubitMapper`
                to be used in mapping orbitals to qubits.
                Defaults to JordanWignerMapper.
            format (str, optional): Which quantum backend we want to use.
                Defaults to "qiskit".

        **Example**

        >>> from ase.build import molecule
        >>> from qc2.ase import PySCF
        >>> from qc2.data import qc2Data
        >>>
        >>> mol = molecule('H2O')
        >>>
        >>> hdf5_file = 'h2o.hdf5'
        >>> qc2data = qc2Data(hdf5_file, mol, schema='qcschema')
        >>> oo_problem = OrbitalOptimization(
        ...     qc2data=qc2data,
        ...     active_space=ActiveSpace(
        ...         num_active_electrons=(2, 2),
        ...         num_active_spatial_orbitals=4
        ...     ),
        ...     mapper=JordanWignerMapper(),
        ...     format="qiskit"
        ... )
        """
        # molecule related attributes
        self.qc2data = qc2data
        self.schema_dataclass = self.qc2data.read_schema()
        self.es_problem = self.qc2data.process_schema(
            basis=ElectronicBasis.AO
        )
        self.n_electrons = (
            self.es_problem.num_alpha,
            self.es_problem.num_beta
        )
        self.nao = self.es_problem.num_spatial_orbitals

        # active space parameters
        self.n_active_orbitals = active_space.num_active_spatial_orbitals
        self.n_active_electrons = active_space.num_active_electrons

        # calculate active space params
        (self.occ_idx,
         self.act_idx,
         self.virt_idx) = get_active_space_idx(
            self.nao, self.n_electrons,
            self.n_active_orbitals,
            self.n_active_electrons
         )

        # calculate non-redundant orbital rotations
        self.params_idx = get_non_redundant_indices(
            self.occ_idx, self.act_idx,
            self.virt_idx, freeze_active
        )

        # set dimension of the kappa vector
        self.n_kappa = len(self.params_idx)

        # set fermionic-to-qubit mapper
        self.mapper = mapper
        self.format = format

    def get_transformed_qubit_hamiltonian(
            self,
            kappa: List
    ) -> Tuple[float, Union[SparsePauliOp, Operator]]:
        """Sets up the qubit Hamiltonian in the transformed MO basis.

        Args:
            kappa (List): Orbital rotation parameters vector.

        Returns:
            Tuple[float, SparsePauliOp]:
                - core_energy (float): The core energy after active space
                  and MO transformation.
                - qubit_op (Union[SparsePauliOp, Operator]):
                  If the format is ``qiskit``, it returns a
                  :class:`SparsePauliOp` representing the
                  tranformed qubit Hamiltonian in the qiskit format.
                  If the format is ``pennylane``, it returns a
                  :class:`Operator` instance representing the
                  qubit Hamiltonian in the PennyLane format.
        """
        (k_matrix_transform_a,
         k_matrix_transform_b) = self.get_transformed_mos(kappa)

        # get rotated qubit hamiltonian in MO basis
        core_energy, qubit_op = self.qc2data.get_qubit_hamiltonian(
            self.n_active_electrons,
            self.n_active_orbitals,
            self.mapper,
            format=self.format,
            transform=True,
            initial_es_problem=self.es_problem,
            matrix_transform_a=k_matrix_transform_a,
            matrix_transform_b=k_matrix_transform_b,
            initial_basis='atomic',
            final_basis='molecular'
        )
        return core_energy, qubit_op

    def orbital_optimization(
            self,
            one_rdm: np.ndarray,
            two_rdm: np.ndarray,
            kappa_init: Optional[List] = None,
    ):
        """Optimize orbital parameters."""
        def objective_function(kappa):
            return self.get_energy_from_kappa(kappa, one_rdm, two_rdm)

        def gradient(kappa):
            return self.get_analytic_gradients(kappa, one_rdm, two_rdm)

        if kappa_init is None:
            kappa_init = [0.0] * self.n_kappa

        # perform the optimization
        result = minimize(objective_function, kappa_init, method='SLSQP', jac=gradient)

        # check for any error
        if not result.success:
            msg = result.message
            raise RuntimeError(f"Orbital optimization failed. {msg}.")

        # optimized kappa
        kappa_optimized = result.x

        return kappa_optimized, objective_function(kappa_optimized)

    def get_analytic_gradients(
            self,
            kappa: List,
            rdm1: np.ndarray,
            rdm2: np.ndarray
    ) -> np.ndarray:
        """
        Calculates analytic gradients for orbital optimization.

        This method calculates the first derivative of the energy
        with respect to orbital rotation parameters.

        Args:
            kappa (List): Orbital rotation parameters vector.
            rdm1 (np.array): One-electron reduced density matrix.
            rdm2 (np.array): Two-electron reduced density matrix.

        Returns:
            np.array: A vector of len(kappa) containing analytic gradients.

        Notes:
            Based on the method outlined in
            [1] P. E. M. Siegbahn, J. Almlof, A. Heiberg, and B. O. Roos
            "The complete active space SCF (CASSCF) method in a Newton-Raphson
            formulation with application to the HNO molecule"
            J. Chem. Phys. 74, 2384-2396 (1981)
        """
        # get transformed MO coefficients
        mo_coeff_a, mo_coeff_b = self.get_transformed_mos(kappa)

        # calculate full-space one- and two-electron integrals
        (_, one_electron_integrals,
         two_electron_integrals) = self._get_full_space_integrals(
             mo_coeff_a, mo_coeff_b
        )

        # calculate fock matrix
        fock_matrix = self.get_fock_matrix(
            one_electron_integrals[0],
            two_electron_integrals[0],
            rdm1, rdm2
        )

        # calculate the analytic gradients
        # eq.(10) of [1]
        gradient = 2 * (fock_matrix - np.transpose(fock_matrix))

        # convert the gradient skew-symmetric matrix to a vector
        return self._kappa_matrix_to_vector(gradient)

    def get_fock_matrix(
            self,
            one_electron_integrals: np.array,
            two_electron_integrals: np.array,
            rdm1: np.array,
            rdm2: np.array
    ):
        """
        Constructs the Fock matrix for orbital optimization.

        Args:
            one_electron_integrals (np.array): One-electron integrals.
            two_electron_integrals (np.array): Two-electron integrals.
            rdm1 (np.array): One-electron reduced density matrix.
            rdm2 (np.array): Two-electron reduced density matrix.

        Returns:
            np.array: The Fock matrix.

        Notes:
            Based on the method outlined in
            [1] P. E. M. Siegbahn, J. Almlof, A. Heiberg, and B. O. Roos
            "The complete active space SCF (CASSCF) method in a Newton-Raphson
            formulation with application to the HNO molecule"
            J. Chem. Phys. 74, 2384-2396 (1981)
        """
        # convert two-electron integrals to chemistry notation
        two_electron_integrals = to_chemist_ordering(two_electron_integrals)

        # prepare rdms
        d_rdm1 = rdm1.real
        p_rdm2 = rdm2.real/2.0

        n_mos = self.nao
        f = list(range(n_mos))
        oc = self.occ_idx.tolist()
        ac = self.act_idx.tolist()

        # initiate fock matrix
        fock_matrix = np.zeros((n_mos, n_mos))

        # calculate the inactive part; eq.(15a) of [1]
        f_inactive = one_electron_integrals.copy()
        f_inactive_two_e_term_1 = 2 * np.einsum(
            "ijkk->ij", two_electron_integrals[np.ix_(f, f, oc, oc)]
        )
        f_inactive_two_e_term_2 = np.einsum(
            "ikjk->ij", two_electron_integrals[np.ix_(f, oc, f, oc)]
        )
        f_inactive += (f_inactive_two_e_term_1 - f_inactive_two_e_term_2)

        # calculate the active part; eq.(15b) of [1]
        f_active_term_1 = np.einsum(
            "tu,pqtu->pq", d_rdm1, two_electron_integrals[np.ix_(f, f, ac, ac)]
        )
        f_active_term_2 = 0.5 * np.einsum(
            "tu,ptqu->pq", d_rdm1, two_electron_integrals[np.ix_(f, ac, f, ac)]
        )
        f_active = f_active_term_1 - f_active_term_2

        # calculating final part
        idxs = np.ix_(oc, f)

        # eq.(14a) of [1]
        fock_matrix[idxs] = 2 * (f_active[idxs] + f_inactive[idxs])

        # eq.(14b) of [1]
        f_act_term1 = np.einsum("tu,qu->tq", d_rdm1, f_inactive[np.ix_(f, ac)])
        f_act_term2 = 2 * np.einsum(
            "tuvx,quvx->tq", p_rdm2, two_electron_integrals[np.ix_(f, ac, ac, ac)]
        )
        fock_matrix[np.ix_(ac, f)] += (f_act_term1 + f_act_term2)

        return fock_matrix

    def get_energy_from_kappa(
            self,
            kappa: List,
            one_rdm: np.ndarray,
            two_rdm: np.ndarray
    ) -> float:
        """Gets total energy after transforming the MOs with kappa.

        Args:
            kappa (List): Orbital rotation parameters vector.
            rdm1 (np.array): One-electron reduced density matrix.
            rdm2 (np.array): Two-electron reduced density matrix.

        Returns:
            float: Total ground-state energy.
        """
        mo_coeff_a, mo_coeff_b = self.get_transformed_mos(kappa)
        return self.get_energy_from_mo_coeffs(
            mo_coeff_a, mo_coeff_b, one_rdm, two_rdm
        )

    def get_energy_from_mo_coeffs(
            self,
            mo_coeff_a: np.ndarray,
            mo_coeff_b: Optional[np.ndarray],
            one_rdm: np.ndarray,
            two_rdm: np.ndarray
    ) -> float:
        """Get energy given one- and two-particle reduced density matrices.

        Args:
            mo_coeff_a (np.array): Alpha MO coefficients vector.
            mo_coeff_b (np.array, optional): Beta MO coefficients vector.
            rdm1 (np.array): One-electron reduced density matrix.
            rdm2 (np.array): Two-electron reduced density matrix.

        Returns:
            float: Total ground-state energy.
        """
        # get active space integrals
        (core_energy,
         one_electron_integrals,
         two_electron_integrals) = self._get_activate_space_integrals(
             mo_coeff_a, mo_coeff_b
         )

        # for restricted cases only?
        return sum(
            (core_energy,
             np.einsum("pq, pq", one_electron_integrals[0], one_rdm),
             0.5 * np.einsum("pqrs, pqrs", two_electron_integrals[0], two_rdm))
        ).real

    def get_transformed_mos(self, kappa: List) -> Tuple[
        np.ndarray, np.ndarray
    ]:
        """Transforms MO coefficients with orbital rotation parameters.

        Args:
            kappa (List): Orbital rotation parameters vector.

        Returns:
            Tuple[np.ndarray, np.ndarray]: A tuple containing the transformed
                MO coefficients.
        """
        if len(kappa) != self.n_kappa:
            raise ValueError(
                "Incorrect dimension for the initial orbital parameter list."
                f"Dimention should be {self.n_kappa}"
            )
        # get MO coefficients
        mo_coeff_a, mo_coeff_b = self._get_mo_coeffs()

        # calculate rotation matrix given the kappa parameters
        k_matrix_a = self._get_rotation_matrix(kappa)
        # restricted case constraint
        k_matrix_b = k_matrix_a

        mo_coeff_transformed_a = mo_coeff_a @ k_matrix_a
        mo_coeff_transformed_b = None
        if mo_coeff_b is not None:
            mo_coeff_transformed_b = mo_coeff_b @ k_matrix_b
        return mo_coeff_transformed_a, mo_coeff_transformed_b

    def _get_rotation_matrix(self, kappa: List) -> np.ndarray:
        """Creates rotation matrix from kappa parameters."""
        kappa_matrix = self._kappa_vector_to_matrix(kappa)
        return expm(-kappa_matrix)

    def _kappa_vector_to_matrix(self, kappa: List) -> np.ndarray:
        """Generates skew-symm. matrix from orbital rotation parameters."""
        kappa_total_vector = np.zeros(self.nao * (self.nao - 1) // 2)
        kappa_total_vector[np.array(self.params_idx)] = kappa
        return vector_to_skew_symmetric(kappa_total_vector)

    def _kappa_matrix_to_vector(self, kappa_matrix):
        """Generate orbital rotation parameters from a skew-symmetric matrix"""
        kappa_total_vector = skew_symmetric_to_vector(kappa_matrix)
        return kappa_total_vector[self.params_idx]

    def _get_mo_coeffs(self) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """Extracts molecular orbital coefficients."""
        if self.qc2data._schema != "qcschema":
            raise ValueError(
                "Orbital optimization requires 'qcschema', "
                f"but found '{self.qc2data._schema}'"
            )
        nmo = self.schema_dataclass.properties.calcinfo_nmo
        nao = len(self.schema_dataclass.wavefunction.scf_orbitals_a) // nmo
        mo_coeff_a = reshape_2(
            self.schema_dataclass.wavefunction.scf_orbitals_a, nao, nmo
        )
        mo_coeff_b = None
        if self.schema_dataclass.wavefunction.scf_orbitals_b is not None:
            mo_coeff_b = reshape_2(
                self.schema_dataclass.wavefunction.scf_orbitals_b, nao, nmo
            )
        return mo_coeff_a, mo_coeff_b

    def _get_activate_space_integrals(
            self,
            mo_coeff_a: np.ndarray,
            mo_coeff_b: Optional[np.ndarray]
    ) -> Tuple[float, List, List]:
        """Extracts activate space integrals in MO basis."""
        (active_space_es_problem,
         core_energy, _) = self.qc2data.get_fermionic_hamiltonian(
            self.n_active_electrons,
            self.n_active_orbitals,
            transform=True,
            initial_es_problem=self.es_problem,
            matrix_transform_a=mo_coeff_a,
            matrix_transform_b=mo_coeff_b,
            initial_basis='atomic',
            final_basis='molecular'
        )

        alpha = active_space_es_problem.hamiltonian.electronic_integrals.alpha
        beta = active_space_es_problem.hamiltonian.electronic_integrals.beta
        beta_alpha = (
            active_space_es_problem.hamiltonian.electronic_integrals.beta_alpha
        )

        one_electron_integrals = [alpha['+-'].array, beta['+-'].array]
        two_electron_integrals = [
            alpha['++--'].array, beta_alpha['++--'].array, beta['++--'].array
        ]
        return core_energy, one_electron_integrals, two_electron_integrals

    def _get_full_space_integrals(
            self,
            mo_coeff_a: np.ndarray,
            mo_coeff_b: Optional[np.ndarray]
    ) -> Tuple[List, List]:
        """Extracts full space one- and two-electron integrals in MO basis."""
        (_, hamiltonian_MO_basis) = self.qc2data.get_transformed_hamiltonian(
            initial_es_problem=self.es_problem,
            matrix_transform_a=mo_coeff_a,
            matrix_transform_b=mo_coeff_b,
            initial_basis='atomic',
            final_basis='molecular'
        )

        alpha = hamiltonian_MO_basis.electronic_integrals.alpha
        beta = hamiltonian_MO_basis.electronic_integrals.beta
        beta_alpha = hamiltonian_MO_basis.electronic_integrals.beta_alpha

        one_electron_integrals = [alpha['+-'].array, beta['+-'].array]
        two_electron_integrals = [
            alpha['++--'].array, beta_alpha['++--'].array, beta['++--'].array
        ]
        core_energy = hamiltonian_MO_basis.nuclear_repulsion_energy
        return core_energy, one_electron_integrals, two_electron_integrals
