from qiskit_nature.second_q.circuit.library import UCC as UCCBASE
from qiskit_nature.second_q.mappers import JordanWignerMapper
from qiskit_nature.second_q.circuit.library import HartreeFock
from ...algorithms.utils import ActiveSpace


def get_reference_state(active_space, mapper):
    # reference HF state
    return HartreeFock(
        active_space.num_active_spatial_orbitals,
        active_space.num_active_electrons,
        mapper,
    )


class UCC(UCCBASE):

    def __init__(
        self,
        active_space=None,
        mapper=None,
        initial_state=None,
        excitations="sd",
        reps=1,
    ):
        """_summary_

        Args:
            active_space (_type_, optional): _description_. Defaults to None.
            mapper (_type_, optional): _description_. Defaults to None.
            initial_state (_type_, optional): _description_. Defaults to None.
            excitations (str, optional): _description_. Defaults to "sd".
            reps (int, optional): _description_. Defaults to 1.
        """

        # check
        if mapper is None:
            mapper = JordanWignerMapper()
        if active_space is None:
            active_space = ActiveSpace((1, 1), 2)
        if initial_state is None:
            initial_state = get_reference_state(active_space, mapper)

        super().__init__(
            num_spatial_orbitals=active_space.num_active_spatial_orbitals,
            num_particles=active_space.num_active_electrons,
            qubit_mapper=mapper,
            initial_state=initial_state,
            excitations=excitations,
            reps=reps,
        )
