from dataclasses import dataclass 
from qiskit_nature.second_q.mappers import JordanWignerMapper
from qiskit_nature.second_q.circuit.library import HartreeFock, UCC
from qiskit_algorithms.minimum_eigensolvers import VQE as vqe_solver
from qiskit.primitives import Estimator
from qiskit_algorithms.optimizers import SLSQP
import pennylane as qml
from pennylane import numpy as np

@dataclass
class ActiveSpace:
    num_active_electrons: tuple[int ,int]
    num_active_spatial_orbitals: int 


class VQE():
   """Main VQE class.

    This class orchestrates VQE calculation.

    Attributes:
        _schema (str): Format in which to save qchem data.
            Options are ``qcschema`` or ``fcidump``.
            Defaults to ``qcschema``.

        _filename (str): The path to the HDF5 or fcidump file used
            to save/read qchem data.

        _molecule (Atoms): Attribute representing the
            molecular structure as an ASE :class:`ase.atoms.Atoms` instance.
    """

    def __init__(self, 
                qc2data, 
                active_space = ActiveSpace((2,2), 2),
                mapper = JordanWignerMapper(),
                format='qiskit'
    ):
      
        self.active_space = active_space
        self.mapper = mapper 
        self.forma = format
        self.e_core, self.qubit_op = qc2data.get_qubit_hamiltonian(active_space.num_active_electrons,
                                                         active_space.num_active_spatial_orbitals,
                                                         mapper, format=format)
        

    def run(self, **kwargs):
        run_map={'qiskit':self._run_qiskit, 'pennylane': self._run_pennylane}
        return run_map[self.format](**kwargs)


    def _run_qiskit(self, **kwargs):

        def _get_default_reference(active_space, mapper):
            return HartreeFock( active_space.num_active_spatial_orbitals,
                                active_space.num_active_electrons,
                                mapper)

        def _get_default_ansatz(active_space, mapper, reference_state): 

            return UCC(
                num_spatial_orbitals=active_space.num_active_spatial_orbitals,
                num_particles=active_space.num_active_electrons,
                qubit_mapper=mapper,
                initial_state=reference_state,
                excitations='sdtq'
            )
        
        def _get_default_init_params(nparams):
            return [0.0] * nparams

        estimator = Estimator() if 'estimator' not in kwargs else kwargs['estimator']
        optimizer = SLSQP() if 'optimizer' not in kwargs else kwargs['optimizer']
        reference_state = _get_default_reference(self.active_space, self.mapper) if 'reference_state' not in kwargs else kwargs['reference_state']
        ansatz = _get_default_ansatz(self.active_space, self.mapper, reference_state) if 'ansatz' not in kwargs else kwargs['ansatz']
        params = _get_default_init_params(self.ansatz.num_parameters) if 'init_params' not in kwargs else kwargs['init_params']

        solver = vqe_solver(estimator, ansatz, optimizer)
        solver.initial_point = params
        result = vqe_solver.compute_minimum_eigenvalue(self.qubit_op)

        print("=== QISKIT VQE RESULTS ===")
        print(f"* Electronic ground state energy (Hartree): {result.eigenvalue}")
        print(f"* Inactive core energy (Hartree): {self.e_core}")
        print(f">>> Total ground state energy (Hartree): {result.eigenvalue+self.e_core}\n")

        return result

    def _run_pennylane(self, **kwargs):

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

        optimizer = qml.GradientDescentOptimizer(stepsize=0.5) if 'optimizer' not in kwargs else kwargs['optimizer']
        reference_state = _get_default_reference(self.active_space, self.mapper) if 'reference_state' not in kwargs else kwargs['reference_state']
        circuit = _get_default_ansatz(qubits, electrons, reference_state) if 'ansatz' not in kwargs else kwargs['ansatz']
        params = _get_default_init_param(qubits, electrons) if "init_param" not in kwargs else kwargs["init_param"]
        niter = niter_default if 'niter' not in kwargs else kwargs['niter'] 

        # Optimize the circuit parameters and compute the energy
        for _ in range(niter):
            params, energy = optimizer.step_and_cost(circuit, params)

        print("=== PENNYLANE VQE RESULTS ===")
        print(f"* Electronic ground state energy (Hartree): {energy}")
        print(f"* Inactive core energy (Hartree): {self.e_core}")
        print(f">>> Total ground state energy (Hartree): {energy+self.e_core}\n")

        # print(f"+++ Final parameters:{params}")
