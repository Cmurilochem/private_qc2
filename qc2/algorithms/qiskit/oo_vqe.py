""""Module defining oo VQE algorithm."""
from typing import List, Tuple
import itertools as itt
import numpy as np
from qiskit_nature.second_q.operators import FermionicOp
from qc2.algorithms.qiskit.vqe import VQE
from qc2.algorithms.utils import OrbitalOptimization


class oo_VQE(VQE):
    """Main class for orbital-optimized VQE.

    This class extends the VQE class to include orbital optimization. It
    supports customized ansatzes, active space definitions, qubit mapping
    strategies, estimation methods, and optimization routines. Orbital 
    optimization is performed alongside VQE parameter optimization.

    Attributes:
        freeze_active (bool): If True, freezes the active
            space during optimization.
        orbital_params (Any): Initial parameters for orbital optimization.
        circuit_params (Any): Parameters for the VQE circuit,
            inherited from VQE class.
        oo_problem (Any): The orbital optimization problem definition,
            initially None.
        max_iterations (int): Maximum number of iterations for the optimizer.
        conv_tol (float): Convergence tolerance for the optimization.
        verbose (int): Verbosity level.
        result (Any): Stores the result of the VQE computation, initially None.
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
        init_circuit_params=None,
        init_orbital_params=None,
        freeze_active=False,
        max_iterations=50,
        conv_tol=1e-7,
        verbose=0
    ):
        """Initializes the oo-VQE class.

        Args:
            qc2data (Any): An instance of :class:`~qc2.data.qc2Data`.
            ansatz (Any): The ansatz for the VQE algorithm.
            active_space (Any): Definition of the active space.
            mapper (Any): Strategy for fermionic-to-qubit mapping.
            estimator (Any): Method for estimating the expectation value.
            optimizer (Any): Optimization routine for variational parameters.
            reference_state (Any): Reference state for the VQE algorithm.
            init_circuit_params (Any): Initial parameters for the VQE circuit.
            init_orbital_params (Any): Initial parameters for the
                orbital optimization part.
            freeze_active (bool): If True, active space is frozen
                during orbital optimization.
            max_iterations (int): Maximum number of iterations in optimization.
            conv_tol (float): Convergence tolerance for optimization.
            verbose (int): Level of verbosity.
        """
        super().__init__(
            qc2data,
            ansatz,
            active_space,
            mapper,
            estimator,
            optimizer,
            reference_state,
            init_circuit_params
        )
        self.freeze_active = freeze_active
        self.orbital_params = init_orbital_params
        self.circuit_params = self.params
        self.oo_problem = None
        self.max_iterations = max_iterations
        self.conv_tol = conv_tol
        self.verbose = verbose
        self.result = None

    def _get_rdms(
            self,
            theta: List,
            sum_spin=True
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Get 1- and 2-RDMs.

        Args:
            theta (List): circuit parameters with which
                to calculate RDMs.
            sum_spin (bool): If True, the spin-summed 1-RDM and 2-RDM will be
                returned. If False, the full 1-RDM and 2-RDM will be returned.
                Defaults to True.

        Returns:
            Tuple[np.ndarray, np.ndarray]: 1- and 2-RDMs.
        """
        if len(theta) != self.ansatz.num_parameters:
            raise ValueError("Incorrect dimension for amplitude list.")

        # initialize the RDM arrays
        n_mol_orbitals = self.active_space.num_active_spatial_orbitals
        n_spin_orbitals = self.active_space.num_active_spatial_orbitals * 2
        rdm1_spin = np.zeros((n_spin_orbitals,) * 2, dtype=complex)
        rdm2_spin = np.zeros((n_spin_orbitals,) * 4, dtype=complex)

        # get the fermionic hamiltonian
        _, _, fermionic_op = self.qc2data.get_fermionic_hamiltonian(
            self.active_space.num_active_electrons,
            self.active_space.num_active_spatial_orbitals
        )

        # run over the hamiltonian terms and calculate expectation values
        for key, _ in fermionic_op.terms():
            # assign indices depending on one- or two-body term
            length = len(key)
            if length == 2:
                iele, jele = (int(ele[1]) for ele in tuple(key[0:2]))
            elif length == 4:
                iele, jele, kele, lele = (int(ele[1]) for ele in tuple(key[0:4]))

            # get fermionic and qubit representation of each term
            fermionic_ham_temp = FermionicOp.from_terms([(key, 1.0)])
            qubit_ham_temp = self.mapper.map(
                fermionic_ham_temp, register_length=n_spin_orbitals
            )
            # calculate expectation values
            energy_temp = self.estimator.run(
                self.ansatz, qubit_ham_temp, theta
            ).result().values

            # put the values in np arrays (differentiate 1- and 2-RDM)
            if length == 2:
                rdm1_spin[iele, jele] = energy_temp
            elif length == 4:
                rdm2_spin[iele, lele, jele, kele] = energy_temp

        if sum_spin:
            # get spin-free RDMs
            rdm1_np = np.zeros((n_mol_orbitals,) * 2, dtype=np.complex128)
            rdm2_np = np.zeros((n_mol_orbitals,) * 4, dtype=np.complex128)

            # construct spin-summed 1-RDM
            mod = n_spin_orbitals // 2
            for i, j in itt.product(range(n_spin_orbitals), repeat=2):
                # use i//2 if electrons are organized as a,b,..a,b (pennylane)
                rdm1_np[i % mod, j % mod] += rdm1_spin[i, j]

            # construct spin-summed 2-RDM
            for i, j, k, l in itt.product(range(n_spin_orbitals), repeat=4):
                rdm2_np[
                    i % mod, j % mod, k % mod, l % mod
                ] += rdm2_spin[i, j, k, l]

            return rdm1_np, rdm2_np

        return rdm1_spin, rdm2_spin

    def _get_energy_from_parameters(
            self,
            theta: List,
            kappa: List
    ) -> float:
        """Get total energy given circuit and orbital parameters."""
        mo_coeff_a, mo_coeff_b = self.oo_problem.get_transformed_mos(kappa)
        one_rdm, two_rdm = self._get_rdms(theta)
        return self.oo_problem.get_energy_from_mo_coeffs(
            mo_coeff_a, mo_coeff_b, one_rdm, two_rdm
        )

    def _circuit_optimization(
            self,
            theta: List,
            kappa: List
    ) -> Tuple[List, float]:
        """Get total energy and best circuit parameters for a given kappa."""
        def objective_function(theta):
            (core_energy,
             qubit_op) = self.oo_problem.get_transformed_qubit_hamiltonian(
                 kappa
             )
            job = self.estimator.run(self.ansatz, qubit_op, theta)
            cost = job.result().values + core_energy
            return cost

        # optimize theta with kappa fixed
        circuit_optimization_result = self.optimizer.minimize(
            fun=objective_function, x0=theta
        )
        theta_optimized = circuit_optimization_result.x

        return theta_optimized, objective_function(theta_optimized)

    def run(self) -> Tuple[List, List, List]:
        """Optimize both the circuit and orbital parameters."""
        print(">>> Optimizing circuit and orbital parameters...")

        # instantiate oo class
        self.oo_problem = OrbitalOptimization(
            self.qc2data,
            self.active_space,
            self.freeze_active,
            self.mapper,
            "qiskit"
        )

        # set initial parameters
        self.orbital_params = (
            self._get_default_init_params(self.oo_problem.n_kappa)
            if self.orbital_params is None
            else self.orbital_params
        )
        theta = self.circuit_params
        kappa = self.orbital_params

        theta_l = []
        kappa_l = []
        energy_l = []

        # get initial energy from initial circuit params
        energy_init = self._get_energy_from_parameters(theta, kappa)
        if self.verbose is not None:
            print(f"iter = 000, energy = {energy_init:.12f}")
            energy_l.append(energy_init)

        for n in range(self.max_iterations):
            # optimize circuit parameters with fixed kappa
            theta, _ = self._circuit_optimization(theta, kappa)

            # optimize orbital parameters with fixed theta from previous run
            rdm1, rdm2 = self._get_rdms(theta)
            kappa, _ = self.oo_problem.orbital_optimization(rdm1, rdm2, kappa)

            # calculate final energy with all optimized parameters
            energy = self._get_energy_from_parameters(theta, kappa)

            theta_l.append(theta)
            kappa_l.append(kappa)
            energy_l.append(energy)

            if self.verbose is not None:
                print(f"iter = {n+1:03}, energy = {energy:.12f}")
            if n > 1:
                if abs(energy_l[-1] - energy_l[-2]) < self.conv_tol:
                    # save final parameters
                    self.circuit_params = theta_l[-1]
                    self.orbital_params = kappa_l[-1]
                    self.result = energy_l[-1]
                    if self.verbose is not None:
                        print("optimization finished.\n")
                        print("=== QISKIT oo-VQE RESULTS ===")
                        print("* Total ground state "
                              f"energy (Hartree): {self.result:.12f}")
                    break

        return energy_l, theta_l, kappa_l