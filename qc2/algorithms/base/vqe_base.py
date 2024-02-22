class VQEBASE:
    """Base class for VQE"""

    def __init__(
        self,
        qc2data,
        format,
        active_space,
        mapper,
    ):
        """Initiate the class

        Args:
            qc2data (data container): qc2 data container with scf informatioms
            active_space (ActiveSpace, optional): Description of the active sapce. Defaults to ActiveSpace((2, 2), 2).
            mapper (qiskit_nature.second_q.mappers, optional): Method used to map the qubits. Defaults to JordanWignerMapper().
            format (str, optional): Which quantum backend we want to use. Defaults to "qiskit".
        """

        self.active_space = active_space
        self.mapper = mapper
        self.forma = format
        self.e_core, self.qubit_op = qc2data.get_qubit_hamiltonian(
            active_space.num_active_electrons,
            active_space.num_active_spatial_orbitals,
            mapper,
            format=format,
        )
