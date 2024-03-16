""""Module defining oo VQE algorithm for PennyLane."""
from typing import List, Tuple
import itertools as itt
import pennylane as qml
from pennylane import numpy as np
from qiskit_nature.second_q.operators import FermionicOp
from qc2.algorithms.pennylane import VQE
from qc2.algorithms.utils import OrbitalOptimization
from qc2.pennylane.convert import _qiskit_nature_to_pennylane


class oo_VQE(VQE):
    """Main class for orbital-optimized VQE with Pennylane.

    Attributes:
        freeze_active (bool): If True, freezes the active
            space during optimization.
        orbital_params (List): Initial parameters for orbital optimization.
        circuit_params (List): Parameters for the VQE circuit,
            inherited from VQE class.
        oo_problem (OrbitalOptimization): The orbital optimization problem
            definition, initially None.
        max_iterations (int): Maximum number of iterations for the optimizer.
        conv_tol (float): Convergence tolerance for the optimization.
        verbose (int): Verbosity level.
        energy (float): Stores the result of the VQE computation,
            initially None.
    """
    def __init__(
        self,
        qc2data=None,
        ansatz=None,
        active_space=None,
        mapper=None,
        device=None,
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
            qc2data (qc2Data, optional): An instance of :class:`~qc2.data.qc2Data`.
            ansatz (Any): The ansatz for the VQE algorithm.
            active_space (Any): Definition of the active space.
            mapper (Any): Strategy for fermionic-to-qubit mapping.
            device (Any): Device for estimating the expectation value.
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
            device,
            optimizer,
            reference_state,
            init_circuit_params,
            max_iterations,
            conv_tol,
            verbose
        )
        self.freeze_active = freeze_active
        self.orbital_params = init_orbital_params
        self.circuit_params = self.params
        self.oo_problem = None

    @staticmethod
    def _get_default_init_orbital_params(n_kappa: List) -> List:
        """Set up the init orbital rotation parameters.

        Args:
            n_kappa (List): number of orbital rotation parameters.

        Returns:
            List : List of params values
        """
        return [0.0] * n_kappa

    def _get_rdms(
            self,
            theta: List,
            sum_spin=True,
            *args, **kwargs
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
        if len(theta) != len(self.params):
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
            qubit_ham_temp_qiskit = self.mapper.map(
                fermionic_ham_temp, register_length=n_spin_orbitals
            )

            # convert qiskit SparsePauliOp to pennylane Operator
            coefficients, operators = _qiskit_nature_to_pennylane(qubit_ham_temp_qiskit)
            qubit_ham_temp = sum(c * op for c, op in zip(coefficients, operators))

            # calculate expectation values
            circuit = VQE._build_circuit(
                self.device,
                self.qubits,
                self.ansatz,
                qubit_ham_temp,
                *args, **kwargs
            )
            energy_temp = circuit(theta)

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
            kappa: List,
            *args, **kwargs
    ) -> float:
        """Get total energy given circuit and orbital parameters."""
        mo_coeff_a, mo_coeff_b = self.oo_problem.get_transformed_mos(kappa)
        one_rdm, two_rdm = self._get_rdms(theta, *args, **kwargs)
        return self.oo_problem.get_energy_from_mo_coeffs(
            mo_coeff_a, mo_coeff_b, one_rdm, two_rdm
        )

    def _circuit_optimization(
            self,
            theta: List,
            kappa: List,
            *args, **kwargs
    ) -> Tuple[List, float]:
        """Get total energy and best circuit parameters for a given kappa."""
        energy_l = []
        theta_l = []

        # get qubit Hamiltonian for given orbital rotation parameters
        (core_energy,
         qubit_op) = self.oo_problem.get_transformed_qubit_hamiltonian(kappa)

        # build up the pennylane circuit
        circuit = VQE._build_circuit(
            self.device,
            self.qubits,
            self.ansatz,
            qubit_op,
            *args, **kwargs
        )

        # Optimize the circuit parameters and compute the energy
        circ_params = theta
        for n in range(self.max_iterations):
            circ_params, corr_energy = self.optimizer.step_and_cost(
                circuit, circ_params
            )
            energy = corr_energy + core_energy
            energy_l.append(energy)
            theta_l.append(circ_params)

            if n > 1:
                if abs(energy_l[-1] - energy_l[-2]) < self.conv_tol:
                    theta_optimized = theta_l[-1]
                    energy_optimized = energy_l[-1]
                    break
        # in case of non-convergence
        else:
            raise RuntimeError(
                "Circuit optimization step did not converge."
                " Consider increasing 'max_iterations' attribute or"
                " setting a different 'optimizer'."
            )

        return theta_optimized, energy_optimized

    def run(self, *args, **kwargs) -> Tuple[List, List, List]:
        """Optimize both the circuit and orbital parameters."""
        print(">>> Optimizing circuit and orbital parameters...")

        # instantiate oo class
        self.oo_problem = OrbitalOptimization(
            self.qc2data,
            self.active_space,
            self.freeze_active,
            self.mapper,
            "pennylane"
        )

        # set initial parameters
        self.orbital_params = (
            self._get_default_init_orbital_params(self.oo_problem.n_kappa)
            if self.orbital_params is None
            else self.orbital_params
        )
        theta = self.circuit_params
        kappa = self.orbital_params

        theta_l = []
        kappa_l = []
        energy_l = []

        # get initial energy from initial circuit params
        energy_init = self._get_energy_from_parameters(
            theta, kappa, *args, **kwargs
        )
        if self.verbose is not None:
            print(f"iter = 000, energy = {energy_init:.12f} Ha")
            energy_l.append(energy_init)

        for n in range(self.max_iterations):
            # optimize circuit parameters with fixed kappa
            theta, _ = self._circuit_optimization(
                theta, kappa, *args, **kwargs
            )

            # optimize orbital parameters with fixed theta from previous run
            rdm1, rdm2 = self._get_rdms(theta, *args, **kwargs)
            kappa, _ = self.oo_problem.orbital_optimization(rdm1, rdm2, kappa)

            # calculate final energy with all optimized parameters
            energy = self._get_energy_from_parameters(
                theta, kappa, *args, **kwargs
            )

            theta_l.append(theta)
            kappa_l.append(kappa)
            energy_l.append(energy)

            if self.verbose is not None:
                print(f"iter = {n+1:03}, energy = {energy:.12f} Ha")
            if n > 1:
                if abs(energy_l[-1] - energy_l[-2]) < self.conv_tol:
                    # save final parameters
                    self.circuit_params = theta_l[-1]
                    self.orbital_params = kappa_l[-1]
                    self.energy = energy_l[-1]
                    if self.verbose is not None:
                        print("optimization finished.\n")
                        print("=== PENNYLANE oo-VQE RESULTS ===")
                        print("* Total ground state "
                              f"energy (Hartree): {self.energy:.12f}")
                    break
        # in case of non-convergence
        else:
            raise RuntimeError(
                "Optimization did not converge within the maximum iterations."
                " Consider increasing 'max_iterations' attribute or"
                " setting a different 'optimizer'."
            )

        return energy_l, theta_l, kappa_l
