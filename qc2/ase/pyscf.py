"""This module defines an ASE interface to PySCF.

https://pyscf.org/ => Official documentation
https://github.com/pyscf/pyscf => GitHub page

Note: Adapted from
https://github.com/pyscf/pyscf/blob/master/pyscf/pbc/tools/pyscf_ase.py
&
https://github.com/pyscf/pyscf/issues/624
"""
from typing import Union, Optional, List, Dict, Any, Tuple
import warnings
import copy
import re
import os
import numpy as np
import h5py

from ase import Atoms
from ase.calculators.calculator import Calculator, all_changes
from ase.units import Ha, Bohr
from ase.calculators.calculator import InputError
from ase.calculators.calculator import CalculatorSetupError

from pyscf import gto, scf, dft, lib
from pyscf.scf.chkfile import dump_scf
from pyscf import __version__ as pyscf_version

from .qc2_ase_base_class import BaseQc2ASECalculator


def ase_atoms_to_pyscf(ase_atoms: Atoms) -> List[List[Union[str, np.ndarray]]]:
    """Converts ASE atoms to PySCF atom.

    Args:
        ase_atoms (Atoms): ASE Atoms object.

    Returns:
        List[List[Union[str, np.ndarray]]]: PySCF atom.
    """
    return [[ase_atoms.get_chemical_symbols()[i], ase_atoms.get_positions()[i]]
            for i in range(len(ase_atoms.get_positions()))]


class PySCF(Calculator, BaseQc2ASECalculator):
    """An ASE calculator for the PySCF quantum chemistry package.

    Args:
        Calculator (Calculator): Base-class for all ASE calculators.
        BaseQc2ASECalculator (BaseQc2ASECalculator): Base class for
            ase calculartors in qc2.

    Raises:
        InputError: If attributes other than
            'method', 'xc', 'basis', 'multiplicity',
            'charge', 'relativistic', 'cart', 'scf_addons', 
            'verbose', and 'output' are input as Calculator.
        CalculatorSetupError: If abinitio methods other than
            'scf.RHF', 'scf.UHF', 'scf.ROHF',
            'dft.RKS', 'dft.UKS', and 'dft.ROKS' are selected.

    Example of a typical ASE-PySCF input:

    >>> from ase import Atoms
    >>> from ase.build import molecule
    >>> from qc2.ase.pyscf import PySCF
    >>>
    >>> molecule = Atoms(...) or molecule = molecule('...')
    >>> molecule.calc = PySCF(method='dft.RKS',
                            xc='b3lyp',
                            basis='6-31g*',
                            charge=0,
                            multiplicity=1,
                            verbose=0,
                            cart=False,
                            relativistic=False,
                            scf_addons='frac_occ',
                            output='output_file_name_with_no_extension')
    >>> energy = molecule.get_potential_energy()
    >>> gradient = molecule.get_forces()

    where (currently)

    method = 'scf.RHF'
           = 'scf.UHF'
           = 'scf.ROHF'
           = 'dft.RKS'
           = 'dft.UKS'
           = 'dft.ROKS'
    
    Notes: 
        - Scalar relativistic corrections can be added with
            'relativistic = True' keyword. If selected,
            the scf object will be decorated by x2c() method, e.g.,
            mf = scf.RHF(mol).x2c().
            'relativistic' is False by default.
        - pyscf.scf.addons functions can also be included, e.g.:
            if scf_addons='frac_occ' keyword is added, then
            mf = scf.addons.frac_occ(mf).
            'scf_addons' is None by default.
    """
    implemented_properties: List[str] = ['energy', 'forces']

    default_parameters: Dict[str, Any] = {'method': 'scf.HF',
                                          'basis': 'sto-3g',
                                          'xc': 'b3lyp',
                                          'multiplicity': 1,
                                          'charge': 0,
                                          'relativistic': False,
                                          'cart': False,
                                          'scf_addons': None,
                                          'output': None,
                                          'verbose': 0}

    def __init__(self,
                 restart: Optional[bool] = None,
                 ignore_bad_restart: Optional[bool] = False,
                 label: Optional[str] = 'PySCF',
                 atoms: Optional[Atoms] = None,
                 command: Optional[str] = None,
                 directory: str = '.',
                 **kwargs) -> None:
        """ASE-PySCF Class Constructor to initialize the object.

        Notes:
            Basic implementation based on the class Psi4(Calculator);
            see, e.g., ase/ase/calculators/psi4.py.

        Args:
            restart (bool, optional): Prefix for restart file.
                May contain a directory. Defaults to None: don't restart.
            ignore_bad_restart (bool, optional): Deprecated and will
                stop working in the future. Defaults to False.
            label (str, optional): Calculator name. Defaults to 'PySCF'.
            atoms (Atoms, optional): Atoms object to which the calculator
                will be attached. When restarting, atoms will get its
                positions and unit-cell updated from file. Defaults to None.
            command (str, optional): Command used to start calculation.
                Defaults to None.
            directory (str, optional): Working directory in which
                to perform calculations. Defaults to '.'.
        """
        # initializing base class Calculator.
        # see ase/ase/calculators/calculator.py.
        Calculator.__init__(self, restart=restart,
                            ignore_bad_restart=ignore_bad_restart, label=label,
                            atoms=atoms, command=command, directory=directory,
                            **kwargs)
        """Transforms **kwargs into a dictionary with calculation parameters.

        Starting with (attr1=value1, attr2=value2, ...)
            it creates self.parameters['attr1']=value1, and so on.
        """

        # Check self.parameters input keys and values
        self.check_pyscf_attributes()

        self.mol = None
        self.mf = None

        # initialize qc2 base class for ASE calculators.
        # see .qc2_ase_base_class.py
        BaseQc2ASECalculator.__init__(self)

    def check_pyscf_attributes(self) -> None:
        """Checks for any missing and/or mispelling PySCF input attribute.

        Notes:
            it can also be used to eventually set specific
            environment variables, ios, etc.
        """
        recognized_attributes: List[str] = [
            'ignore_bad_restart', 'command', 'method',
            'xc', 'basis', 'multiplicity', 'charge',
            'relativistic', 'cart', 'scf_addons',
            'output', 'verbose', 'kpts', 'nbands', 'smearing'
            ]

        # self.parameters gathers all PYSCF input options in a dictionary.
        # it is defined by class Calculator(BaseCalculator)
        # see ase/ase/calculators/calculator.py
        for key, value in self.parameters.items():
            # check attribute name
            if key not in recognized_attributes:
                raise InputError('Attribute', key,
                                 ' not recognized. Please check input.')

        # dealing with lack of multiplicity and charge info
        if 'multiplicity' not in self.parameters.keys():
            self.parameters['multiplicity'] = 1
            warnings.warn('Multiplicity not provided.'
                          'Assuming default singlet.')

        if 'charge' not in self.parameters.keys():
            self.parameters['charge'] = 0
            warnings.warn('Charge not provided. Assuming default zero.')

        # verbose sets the amount of pyscf printing
        # verbose = 0 prints no info
        if 'verbose' not in self.parameters.keys():
            self.parameters['verbose'] = 0

        # scalar relativistic corrections
        if 'relativistic' not in self.parameters.keys():
            self.parameters['relativistic'] = False

        # cartesian vs spherical basis functions
        if 'cart' not in self.parameters.keys():
            self.parameters['cart'] = False

        # cartesian vs spherical basis functions
        if 'scf_addons' not in self.parameters.keys():
            self.parameters['scf_addons'] = None

        # outputfile
        if 'output' not in self.parameters.keys():
            self.parameters['output'] = None

        # dealing with some ASE specific inputs
        if 'kpts' in self.parameters.keys():
            raise InputError('This ASE-PySCF interface does not yet implement'
                             ' periodic calculations, and thus does not'
                             ' accept k-points as parameters. '
                             'Please remove this argument.')

        if 'nbands' in self.parameters.keys():
            raise InputError('This ASE-PySCF interface '
                             'does not support the keyword "nbands".')

        if 'smearing' in self.parameters.keys():
            raise InputError('Finite temperature DFT is not currently'
                             ' implemented in this PySCF-ASE interface,'
                             ' thus a smearing argument cannot be utilized.'
                             ' Please remove this argument.')

    def calculate(self, atoms: Optional[Atoms] = None,
                  properties: List[str] = ['energy'],
                  system_changes: List[str] = all_changes) -> None:
        """This method is the core responsible for the actual calculation.

        Note: Implementation based on Calculator.calculate() method.
            see also ase/ase/calculators/calculator.py.

        Args:
            atoms (Atoms, optional): Atoms object to which
                the calculator is attached. Defaults to None.
            properties (list[str], optional): List of what properties
                need to be calculated. Defaults to ['energy'].
            system_changes (list[str], optional): List of what has changed
                since last calculation. Can be any combination
                of these six: 'positions', 'numbers', 'cell', 'pbc',
                'initial_charges' and 'initial_magmoms'.
                Defaults to all_changes.

        Raises:
            CalculatorSetupError: If a proper geometry is not provided.
            CalculatorSetupError: If abinitio methods other than
            'scf.RHF', 'scf.UHF', 'scf.ROHF', 'dft.RKS', 'dft.UKS',
                and 'dft.ROKS' are selected.
        """
        # setting up self.atoms attribute from base class Calculator.calculate.
        # this is extracted from the atoms Atoms object.
        Calculator.calculate(self, atoms=atoms)

        # checking that self.atoms has been properly initiated/updated
        if self.atoms is None:
            raise CalculatorSetupError('An Atoms object must be provided to '
                                       'perform a calculation.')

        # the spin variable corresponds to 2S instead of 2S+1
        spin_2s = self.parameters['multiplicity'] - 1

        # if requested, set pyscf output file
        outputfile = None
        if self.parameters['output'] is not None:
            outputfile = self.parameters['output'] + ".out"
            self.parameters['verbose'] = 5

        # passing geometry and other definitions
        self.mol = gto.M(atom=ase_atoms_to_pyscf(self.atoms),
                         basis=self.parameters['basis'],
                         charge=self.parameters['charge'],
                         spin=spin_2s,
                         verbose=self.parameters['verbose'],
                         cart=self.parameters['cart'],
                         output=outputfile)

        # make dictionary of implemented methods
        implemented_methods = {'scf.HF': scf.HF,
                               'scf.RHF': scf.RHF,
                               'scf.UHF': scf.UHF,
                               'scf.ROHF': scf.ROHF,
                               'dft.KS': dft.KS,
                               'dft.RKS': dft.RKS,
                               'dft.UKS': dft.UKS,
                               'dft.ROKS': dft.ROKS}

        # define mf object 
        if self.parameters['method'] in implemented_methods:
            self.mf = implemented_methods[self.parameters['method']](self.mol)
        else:
            raise CalculatorSetupError('Method not yet implemented. '
                                       'Current PySCF-ASE calculator '
                                       'only allows for',
                                       implemented_methods,
                                       'wave functions.'
                                       ' Please check input method.')

        if 'dft' in self.parameters['method']:
            self.mf.xc = self.parameters['xc']

        # add scalar relativistic corrections
        if self.parameters['relativistic']:
            self.mf = self.mf.x2c()

        # add decorations into scf object using scf.addons module
        if self.parameters['scf_addons']:
            # get the name of the function to call
            func_name = self.parameters['scf_addons']
            # get the function object from the scf.addons module
            func = getattr(scf.addons, func_name)
            self.mf = func(self.mf)

        # self.mf.chkfile =  self.parameters['output'] + ".chk"

        # calculating energy in eV
        energy = self.mf.kernel() * Ha
        self.results['energy'] = energy

        # calculating forces
        if 'forces' in properties:
            gf = self.mf.nuc_grad_method()
            gf.verbose = self.parameters['verbose']
            if 'dft' in self.parameters['method']:
                gf.grid_response = True
            # analytic gradienta in eV/Angstrom
            forces = -1.0 * gf.kernel() * (Ha / Bohr)
            totalforces = []
            totalforces.extend(forces)
            totalforces = np.array(totalforces)
            self.results['forces'] = totalforces

    def save(self, hdf5_filename: str) -> None:
        """Dumps qchem data to a HDF5.

        Args:
            hdf5_filename (str): HDF5 file to save the data to.

        Notes:
            HDF5 files are written following the QCSchema.

        Returns:
            None

        Example:
        >>> from ase.build import molecule
        >>> from qc2.ase.pyscf import PySCF
        >>>
        >>> molecule = molecule('H2')
        >>> molecule.calc = PySCF()  # => RHF/STO-3G
        >>> molecule.calc.get_potential_energy()
        >>> molecule.calc.save('h2.h5')
        """
        # calculate 1- and 2-electron integrals in MO basis
        integrals = self.get_integrals()
        e_core = integrals[0]
        one_body_coefficients_a = integrals[1]
        one_body_coefficients_b = integrals[2]
        two_body_coefficients_aa = integrals[3]
        two_body_coefficients_bb = integrals[4]
        two_body_coefficients_ab = integrals[5]
        two_body_coefficients_ba = integrals[6]

        # open the HDF5 file in write mode
        file = h5py.File(hdf5_filename, "w")

        # set up general definitions for the QCSchema
        # 1 => general initial attributes
        file.attrs['driver'] = "energy"
        file.attrs['schema_name'] = 'qcschema_molecule'
        file.attrs['schema_version'] = pyscf_version
        file.attrs['return_result'] = self.mf.e_tot
        file.attrs['success'] = True

        # 2 => molecule group 
        molecule = file.require_group("molecule")
        molecule.attrs['symbols'] = [self.mol.atom_pure_symbol(i)
                                     for i in range(self.mol.natm)]
        molecule.attrs['geometry'] = self.mol.atom_coords(
            unit="Bohr").ravel().tolist()
        molecule.attrs['molecular_charge'] = self.mol.charge
        molecule.attrs['molecular_multiplicity'] = self.mol.spin + 1
        molecule.attrs['atomic_numbers'] = [atom[0] for atom in self.mol._atm]
        molecule.attrs['schema_name'] = "qcschema_molecule"
        molecule.attrs['schema_version'] = pyscf_version

        # 3 => properties group
        properties = file.require_group("properties")
        properties.attrs['calcinfo_nbasis'] = self.mol.nbas
        properties.attrs['calcinfo_nmo'] = self.mol.nao
        properties.attrs['calcinfo_nalpha'] = self.mol.nelec[0]
        properties.attrs['calcinfo_nbeta'] = self.mol.nelec[1]
        properties.attrs['calcinfo_natom'] = self.mol.natm
        properties.attrs['nuclear_repulsion_energy'] = e_core
        properties.attrs['return_energy'] = self.mf.e_tot

        # 4 => model group
        model = file.require_group("model")
        model.attrs['basis'] = self.mol.basis
        model.attrs['method'] = self.mf._method_name()

        # 5 => provenance group
        provenance = file.require_group("provenance")
        provenance.attrs['creator'] = self.name
        provenance.attrs['version'] = pyscf_version
        provenance.attrs['routine'] = f"ASE-{self.__class__.__name__}.save()"

        # 6 => keywords group
        file.require_group("keywords")

        # 7 => wavefunction group
        wavefunction = file.require_group("wavefunction")
        wavefunction.attrs['basis'] = self.mol.basis

        # tolerance to consider number zero.
        EQ_TOLERANCE = 1e-8

        # truncate numbers lower than EQ_TOLERANCE
        one_body_coefficients_a[
            np.absolute(one_body_coefficients_a) < EQ_TOLERANCE] = 0.
        one_body_coefficients_b[
            np.absolute(one_body_coefficients_b) < EQ_TOLERANCE] = 0.
        two_body_coefficients_aa[
            np.absolute(two_body_coefficients_aa) < EQ_TOLERANCE] = 0.
        two_body_coefficients_bb[
            np.absolute(two_body_coefficients_bb) < EQ_TOLERANCE] = 0.
        two_body_coefficients_ba[
            np.absolute(two_body_coefficients_ba) < EQ_TOLERANCE] = 0.
        two_body_coefficients_ab[
            np.absolute(two_body_coefficients_ab) < EQ_TOLERANCE] = 0.

        # 1-body coeffs in QCSchema format
        wavefunction.create_dataset("scf_fock_mo_a",
                                    data=one_body_coefficients_a.flatten())
        wavefunction.create_dataset("scf_fock_mo_b",
                                    data=one_body_coefficients_b.flatten())

        # 2-body coeffs in QCSchema format
        wavefunction.create_dataset("scf_eri_mo_aa",
                                    data=two_body_coefficients_aa.flatten())
        wavefunction.create_dataset("scf_eri_mo_bb",
                                    data=two_body_coefficients_bb.flatten())
        wavefunction.create_dataset("scf_eri_mo_ba",
                                    data=two_body_coefficients_ba.flatten())
        wavefunction.create_dataset("scf_eri_mo_ab",
                                    data=two_body_coefficients_ab.flatten())

        # mo coefficients in AO basis
        alpha_coeff, beta_coeff = self._expand_mo_object(
            self.mf.mo_coeff, array_dimension=3
        )
        if beta_coeff is None:
            beta_coeff = alpha_coeff
        wavefunction.create_dataset("scf_orbitals_a",
                                    data=alpha_coeff.flatten())
        wavefunction.create_dataset("scf_orbitals_b",
                                    data=beta_coeff.flatten())
        # scf mo energies
        alpha_mo, beta_mo = self._expand_mo_object(
            self.mf.mo_energy, array_dimension=2
        )
        if beta_mo is None:
            beta_mo = alpha_mo
        wavefunction.create_dataset("scf_eigenvalues_a", data=alpha_mo.flatten())
        wavefunction.create_dataset("scf_eigenvalues_b", data=beta_mo.flatten())
        # ROSE localized orbitals?
        wavefunction.create_dataset("localized_orbitals_a", data='')
        wavefunction.create_dataset("localized_orbitals_b", data='')

        file.close()

    def load(self, hdf5_filename: str) -> None:
        """Loads electronic structure data from a HDF5 file.

        Example:
        >>> from ase.build import molecule
        >>> from qc2.ase.pyscf import PySCF
        >>>
        >>> molecule = molecule('H2')
        >>> molecule.calc = PySCF()     # => RHF/STO-3G
        >>> molecule.calc.load('h2.h5') # => instead of 'molecule.calc.get_potential_energy()'
        """
        BaseQc2ASECalculator.load(self, hdf5_filename)

    def get_integrals(self) -> Tuple[float, np.ndarray,
                                     np.ndarray, np.ndarray,
                                     np.ndarray, np.ndarray, np.ndarray]:
        """Retrieves 1- & 2-body integrals in MO basis from PySCF routines.
        
        Returns:
            A tuple containing the following:
                - e_core (float): The nuclear repulsion energy.
                - one_body_int_a & one_body_int_b: Numpy arrays containing
                    alpha and beta components of the one-body integrals.
                - two_body_int_aa, two_body_int_bb, two_body_int_ab
                    & two_body_int_ba: Numpy arrays containing
                    alpha-alpha, beta-beta, alpha-beta & beta-alpha
                    components of the two-body integrals.
        """
        # get nuclear repulsion energy
        e_core = self.mf.energy_nuc()

        # define alpha and beta MO coeffients
        alpha_coeff, beta_coeff = self._expand_mo_object(
            self.mf.mo_coeff, array_dimension=3
        )
        # fock matrix in AO basis
        hcore = self.mf.get_hcore()
        # calculate alpha and beta one-electron integrals in MO basis
        one_body_int_a = np.dot(np.dot(alpha_coeff.T, hcore), alpha_coeff)
        if beta_coeff is None:
            one_body_int_b = one_body_int_a
        else:
            one_body_int_b = np.dot(np.dot(beta_coeff.T, hcore), beta_coeff)

        # define two-electron integrals in AO basis
        eri_nosym_ao = self.mol.intor("int2e", aosym=1)

        # calculate alpha-alpha, beta-beta, beta-alpha, alpha-beta
        # two-electron integrals in MO basis
        einsum_ao_to_mo = "pqrs,pi,qj,rk,sl->ijkl"
        two_body_int_aa = np.einsum(einsum_ao_to_mo,
                                    eri_nosym_ao,
                                    alpha_coeff,
                                    alpha_coeff,
                                    alpha_coeff,
                                    alpha_coeff)
        if beta_coeff is None:
            two_body_int_bb = two_body_int_aa
            two_body_int_ba = two_body_int_aa
            two_body_int_ab = two_body_int_aa
        else:
            two_body_int_bb = np.einsum(einsum_ao_to_mo,
                                        eri_nosym_ao,
                                        beta_coeff,
                                        beta_coeff,
                                        beta_coeff,
                                        beta_coeff)
            two_body_int_ba = np.einsum(einsum_ao_to_mo,
                                        eri_nosym_ao,
                                        beta_coeff,
                                        beta_coeff,
                                        alpha_coeff,
                                        alpha_coeff)
            two_body_int_ab = np.einsum(einsum_ao_to_mo,
                                        eri_nosym_ao,
                                        alpha_coeff,
                                        alpha_coeff,
                                        beta_coeff,
                                        beta_coeff)

        return (e_core, one_body_int_a, one_body_int_b, two_body_int_aa,
                two_body_int_bb, two_body_int_ab, two_body_int_ba)

    def _expand_mo_object(
        self,
        mo_object: Union[Tuple[Optional[np.ndarray], Optional[np.ndarray]],
                         np.ndarray],
        array_dimension: int = 2,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Expands the mo object into alpha- and beta-spin components.

        Notes:
            Adapted from Qiskit-Nature pyscfdriver.py.

        Args:
            mo_object: the molecular orbital object to expand.
            array_dimension:  This argument specifies the dimension of the
                numpy array (if a tuple is not encountered). Making this
                configurable permits this function to be used to expand both,
                MO coefficients (3D array) and MO energies (2D array).

        Returns:
            The (alpha, beta) tuple of MO data.
        """
        if isinstance(mo_object, tuple):
            return mo_object

        if len(mo_object.shape) == array_dimension:
            return mo_object[0], mo_object[1]

        return mo_object, None


# all methods below are still under development......

    def dump_ibos_from_rose_to_chkfile(self,
                                       input_file: str = "ibo.pyscf",
                                       output_file: str = "ibo.chk") -> None:
        """Saves calculated ROSE IBOs to a checkpoint file.

        Args:
            input_file (str): Input file containing ROSE IBO data.
                Defaults to "ibo.pyscf".
            output_file (str): Output checkpoint file.
                Defaults to "ibo.chk".

        Raises:
            FileNotFoundError: If the input file does not exist.

        Notes:
            This method reads the ROSE IBO data from the input file and
            saves it in the specified output file in a checkpoint format.
            The data is stored in a PySCF compatible format.
        """
        # reading info from ROSE "ibo.pyscf" output
        if os.path.exists(input_file):
            with open(input_file, "r") as f:
                nao = read_int("Number of basis functions", f)
                alpha_energies = read_real_list("Alpha Orbital Energies", f)
                alpha_IBO = read_real_list("Alpha MO coefficients", f)
                beta_energies = read_real_list("Beta Orbital Energies", f)
                beta_IBO = read_real_list("Beta MO coefficients", f)
        else:
            raise FileNotFoundError("Cannot open", input_file,
                                    "Run ROSE before calling this method.")

        # start preparing the data
        mol = self.mol
        mf = self.mf
        method_name = self.mf._method_name()

        ibo_wfn = copy.copy(mf)

        nmo = len(alpha_energies)
        if ('UHF' not in method_name) and ('UKS' not in method_name):
            alpha_IBO_coeff = np.zeros((len(mf.mo_coeff), nao), dtype=float)
            beta_IBO_coeff = np.zeros((len(mf.mo_coeff), nao), dtype=float)
        else:
            alpha_IBO_coeff = np.zeros((len(mf.mo_coeff[0]), nao), dtype=float)
            beta_IBO_coeff = np.zeros((len(mf.mo_coeff[1]), nao), dtype=float)   

        ij = 0
        for i in range(nmo):
            for j in range(nao):
                alpha_IBO_coeff[j, i] = alpha_IBO[ij]
                if ('UHF' in method_name) or ('UKS' in method_name):
                    beta_IBO_coeff[j, i] = beta_IBO[ij]
                ij += 1

        if ('UHF' not in method_name) and ('UKS' not in method_name):
            alpha_energy = np.zeros(len(mf.mo_energy), dtype=float)
            alpha_energy[:len(alpha_energies)] = alpha_energies
            ibo_wfn.mo_energy = alpha_energy
            ibo_wfn.mo_coeff = alpha_IBO_coeff
        else:
            alpha_energy = np.zeros(len(mf.mo_energy[0]), dtype=float)
            alpha_energy[:len(alpha_energies)] = alpha_energies
            beta_energy = np.zeros(len(mf.mo_energy[1]), dtype=float)
            beta_energy[:len(beta_energies)] = beta_energies
            ibo_wfn.mo_energy[0] = alpha_energy
            ibo_wfn.mo_energy[1] = beta_energy
            ibo_wfn.mo_coeff[0] = alpha_IBO_coeff
            ibo_wfn.mo_coeff[1] = beta_IBO_coeff

        e_tot = self.mf.e_tot
        dump_scf(mol, output_file,
             e_tot, ibo_wfn.mo_energy, ibo_wfn.mo_coeff, ibo_wfn.mo_occ)

    def dump_mo_input_file_for_rose(self, output_file: str) -> None:
        """Writes molecular orbitals input file for ROSE.

        Args:
            output_file (str, optional): Output file name.

        Raises:
            IOError: If there is an error while writing the file.

        Notes:
            This method extracts relevant information from the PySCF
            "dumpers" and writes an input file to be read by ROSE.
        """
        # extracting relevant info from pyscf "dumpers"
        mol_data = self.get_mol_data()
        basis_data = self.get_basis_set_data()
        mo_data = self.get_mo_data()
        ei_data = self.get_integrals_data()

        method_name = self.mf._method_name()

        # start writing input file to be read by ROSE
        with open(output_file, "w") as f:
            f.write("Generated by ASE-PySCF calculator\n")
            write_int(f, "Number of atoms", mol_data['natom'])
            write_int(f, "Charge", mol_data['charge'])
            write_int(f, "Multiplicity", mol_data['multiplicity'])
            write_int(f, "Number of electrons", mol_data['nelec'])
            write_int(f, "Number of alpha electrons", mol_data['nalpha'])
            write_int(f, "Number of beta electrons", mol_data['nbeta'])
            write_int(f, "Number of basis functions", mol_data['nao'])
            write_int_list(f, "Atomic numbers", mol_data['Zint'])
            write_singlep_list(f, "Nuclear charges", mol_data['Zreal'])
            write_doublep_list(f, "Current cartesian coordinates",
                               mol_data['cart_coord'])
            write_int(f, "Number of primitive shells",
                      basis_data['nshells'])
            write_int(f, "Pure/Cartesian d shells", basis_data['pureamd'])
            write_int(f, "Pure/Cartesian f shells", basis_data['pureamf'])
            write_int_list(f, "Shell types", basis_data['orb_momentum'])
            write_int_list(f, "Number of primitives per shell",
                           basis_data['nprim_shell'])
            write_int_list(f, "Shell to atom map",
                           basis_data['shell_atom_map'])
            write_singlep_list(f, "Primitive exponents",
                               basis_data['prim_exp'])
            write_singlep_list(f, "Contraction coefficients",
                               basis_data['contr_coeffs'])
            write_doublep_list(f, "Coordinates of each shell",
                               basis_data['coord_shell'])
            f.write("{:43}R{:27.15e}\n".format("Total Energy",
                                               mo_data['scf_e']))
            write_doublep_list(f, "Alpha Orbital Energies",
                               mo_data['alpha_energies'])
            write_doublep_list(f, "Alpha MO coefficients",
                               mo_data['alpha_MO'])

            if ('UHF' in method_name) or ('UKS' in method_name):
                write_doublep_list(f, "Beta Orbital Energies",
                                   mo_data['beta_energies'])
                write_doublep_list(f, "Beta MO coefficients",
                                   mo_data['beta_MO'])

            f.write("{:43}R{:27.15e}\n".format(
                "Core Energy", ei_data['E_core']))
            write_doublep_list(f, "One electron integrals",
                               ei_data['one_body_int'])

    def get_mol_data(self) -> Dict[str, Any]:
        """Retrieves molecular data from the PySCF object.

        Returns:
            dict: A dictionary containing the molecular data.

        Notes:
            This method retrieves various molecular data from the PySCF object
            and saves it in a dictionary format. If an output file name
            is passed to the PySCF calculator, this method also adds these
            info into the file.
        """
        charge = self.mol.charge
        natom  = self.mol.natm
        nelec  = self.mol.nelectron
        nalpha = (nelec + self.mol.spin)//2
        nbeta  = (nelec - self.mol.spin)//2
        nao    = self.mol.nao_cart()
        nshells= self.mol.nbas
        multiplicity = self.mol.spin + 1

        cart_coord = []
        for i in range(natom):
            cart_coord.append(self.mol.atom_coords().tolist()[i][0])
            cart_coord.append(self.mol.atom_coords().tolist()[i][1])
            cart_coord.append(self.mol.atom_coords().tolist()[i][2])

        Zint = []
        for i in range(len(self.mol._atm)):
            Zint.append(self.mol._atm[i][0])
        Zreal = Zint

        mol_data = {'charge': charge,
                    'natom': natom,
                    'nelec': nelec,
                    'nalpha': nalpha,
                    'nbeta': nbeta,
                    'nao': nao,
                    'nshells': nshells,
                    'multiplicity': multiplicity,
                    'cart_coord': cart_coord,
                    'Zint': Zint,
                    'Zreal': Zreal}

        if self.mol.output:
            filename = self.mol.output.split(".")
            output_filename = filename[0] + '.chk'
            lib.chkfile.save(output_filename, 'mol_data', mol_data)
        return mol_data

    def get_basis_set_data(self) -> Dict[str, Any]:
        """Retrieves basis set data from the PySCF object.

        Returns:
            dict: A dictionary containing the basis set data.

        Notes:
            This method retrieves the basis set data from the PySCF object
            and saves it in a dictionary format. If an output file name
            is passed to the PySCF calculator, this method also adds these
            info into the file.
    """
        natom  = self.mol.natm
        nshells= self.mol.nbas

        shell_atom_map = []
        orb_momentum = []
        contract_coeff = []
        for i in range(len(self.mol._bas)):
            shell_atom_map.append(self.mol._bas[i][0] + 1)
            orb_momentum.append(self.mol._bas[i][1])
            contract_coeff.append(self.mol._bas[i][3])

        nprim_shell = []
        coord_shell = []
        for i in range(nshells):
            nprim_shell.append(1)
            coord_shell.append(self.mol.atom_coords().tolist()[shell_atom_map[i] - 1][0])
            coord_shell.append(self.mol.atom_coords().tolist()[shell_atom_map[i] - 1][1])
            coord_shell.append(self.mol.atom_coords().tolist()[shell_atom_map[i] - 1][2])

        prim_exp = []
        for i in range(natom):
            atom_type = self.atoms.symbols[i]
            primitives_exp = self.mol._basis[atom_type]
            for j in range(len(primitives_exp)):
                for k in range(1,len(primitives_exp[0])):
                    prim_exp.append(primitives_exp[j][k][0])

        if self.mol.cart:
            pureamd = 1
            pureamf = 1
        else:
            pureamd = 0
            pureamf = 0

        basis_data = {'nshells': nshells,
                      'pureamd': pureamd,
                      'pureamf': pureamf,
                      'orb_momentum': orb_momentum,
                      'nprim_shell': nprim_shell,
                      'shell_atom_map': shell_atom_map,
                      'prim_exp': prim_exp,
                      'contr_coeffs': [1]*len(prim_exp),
                      'coord_shell': coord_shell}
        
        if self.mol.output:
            filename = self.mol.output.split(".")
            output_filename = filename[0] + '.chk'
            lib.chkfile.save(output_filename, 'basis_data', basis_data)
        return basis_data

    def get_mo_data(self) -> Dict[str, Any]:
        """Retrieves molecular orbital data from the PySCF object.

        Returns:
            dict: A dictionary containing the molecular orbital data.

        Notes:
            This method retrieves the molecular orbital data from the PySCF object
            and saves it in a dictionary format. If an output file name
            is passed to the PySCF calculator, this method also adds these info
            into the file.
        """
        scf_e = self.mf.e_tot

        alpha_MO = []
        alpha_energies = []
        beta_MO = []
        beta_energies = []

        method_name = self.mf._method_name()

        if ('UHF' not in method_name) and ('UKS' not in method_name):
            alpha_coeff = self.mf.mo_coeff.copy()
            alpha_energies = self.mf.mo_energy.copy()
        else:
            alpha_coeff = self.mf.mo_coeff[0].copy()
            beta_coeff = self.mf.mo_coeff[1].copy()
            alpha_energies = self.mf.mo_energy[0].copy()
            beta_energies = self.mf.mo_energy[1].copy()

        for i in range(alpha_coeff.shape[1]):
            for j in range(alpha_coeff.shape[0]):
                alpha_MO.append(alpha_coeff[j][i])

        if ('UHF' in method_name) or ('UKS' in method_name):
            for i in range(beta_coeff.shape[1]):
                for j in range(beta_coeff.shape[0]):
                    beta_MO.append(beta_coeff[j][i])

        mo_data = {'scf_e': scf_e,
                   'alpha_energies': alpha_energies,
                   'alpha_MO': alpha_MO,
                   'beta_energies': beta_energies,
                   'beta_MO': beta_MO}
        
        if self.mol.output:
            filename = self.mol.output.split(".")
            output_filename = filename[0] + '.chk'
            lib.chkfile.save(output_filename, 'mo_data', mo_data)
        return mo_data

    def get_integrals_data(self) -> Dict[str, Any]:
        """Retrieves one-electron integrals from the PySCF object.

        Returns:
            dict: A dictionary containing the one-electron integrals.

        Notes:
            This method extracts the one-electron integrals from
            the PySCF object and saves them in a dictionary format.
            If an output file name is passed to the PySCF calculator,
            this method also adds these info into the file.
    """
        scf_e = self.mf.e_tot

        method_name = self.mf._method_name()

        if ('UHF' not in method_name) and ('UKS' not in method_name):
            alpha_coeff = self.mf.mo_coeff.copy()
        else:
            alpha_coeff = self.mf.mo_coeff[0].copy()
            beta_coeff = self.mf.mo_coeff[1].copy()

        E_core = scf_e - self.mf.energy_elec()[0]
        one_body_int = []
        if ('UHF' not in method_name) and ('UKS' not in method_name):
            h1e = alpha_coeff.T.dot(self.mf.get_hcore()).dot(alpha_coeff)
            h1e = self.mf.mo_coeff.T.dot(self.mf.get_hcore()).dot(self.mf.mo_coeff)
            for i in range(1, h1e.shape[0]+1):
                for j in range(1, i+1):
                    one_body_int.append(h1e[i-1, j-1])
        else:
            h1e_alpha = alpha_coeff.T.dot(self.mf.get_hcore()).dot(alpha_coeff)
            h1e_beta = beta_coeff.T.dot(self.mf.get_hcore()).dot(beta_coeff)
            h1e_alpha = self.mf.mo_coeff[0].T.dot(self.mf.get_hcore()).dot(self.mf.mo_coeff[0])
            h1e_beta = self.mf.mo_coeff[1].T.dot(self.mf.get_hcore()).dot(self.mf.mo_coeff[1])
            for i in range(1, h1e_alpha.shape[0]+1):
                for j in range(1, i+1):
                    one_body_int.append(h1e_alpha[i-1, j-1])
                    one_body_int.append(h1e_beta[i-1, j-1])

        ei_data = {}
        ei_data = {'E_core': E_core,
                   'one_body_int': one_body_int}
        
        if self.mol.output:
            filename = self.mol.output.split(".")
            output_filename = filename[0] + '.chk'
            lib.chkfile.save(output_filename, 'ei_data', ei_data)
        return ei_data


def write_int(file, text: str, var: int) -> None:
    """Writes an integer value to a file in a specific format.

    Args:
        f (file object): The file object to write to.
        text (str): A string of text to precede the integer value.
        var (int): The integer value to be written to the file.

    Returns:
        None
    """
    file.write("{:43}I{:17d}\n".format(text, var))


def write_int_list(file, text: str, var: List[int]) -> None:
    """Writes a list of integers to a file in a specific format.

    Args:
        f (file object): The file object to write to.
        text (str): A string of text to precede the list of integers.
        var (list): The list of integers to be written to the file.

    Returns:
        None
    """
    file.write("{:43}{:3} N={:12d}\n".format(text, "I", len(var)))
    dim = 0
    buff = 6
    if (len(var) < 6):
        buff = len(var)
    for i in range((len(var)-1)//6+1):
        for j in range(buff):
            file.write("{:12d}".format(var[dim+j]))
        file.write("\n")
        dim = dim + 6
        if (len(var) - dim) < 6:
            buff = len(var) - dim


def write_singlep_list(file, text: str, var: List[float]) -> None:
    """Writes a list of single-precision floating-point values to a file object.

    Args:
        file (file object): A file object to write to.
        text (str): The text to be written before the list.
        var (list): The list of single-precision floating-point
            values to write.

    Returns:
        None
    """
    file.write("{:43}{:3} N={:12d}\n".format(text, "R", len(var)))
    dim = 0
    buff = 5
    if (len(var) < 5):
        buff = len(var)
    for i in range((len(var)-1)//5+1):
        for j in range(buff):
            file.write("{:16.8e}".format(var[dim+j]))
        file.write("\n")
        dim = dim + 5
        if (len(var) - dim) < 5:
            buff = len(var) - dim


def write_doublep_list(file, text: str, var: List[float]) -> None:
    """Writes a list of double precision floating point numbers to a file.

    Args:
        f (file object): the file to write the data to
        text (str): a label or description for the data
        var (list): a list of double precision floating point
            numbers to write to file

    Returns:
        None
    """
    file.write("{:43}{:3} N={:12d}\n".format(text, "R", len(var)))
    dim = 0
    buff = 5
    if (len(var) < 5):
        buff = len(var)
    for i in range((len(var)-1)//5+1):
        for j in range(buff):
            file.write("{:24.16e}".format(var[dim+j]))
        file.write("\n")
        dim = dim + 5
        if (len(var) - dim) < 5:
            buff = len(var) - dim


def read_int(text:str, file) -> int:
    """Reads an integer value from a text file.

    Args:
        text (str): The text to search for in the file.
        f (file object): The file object to read from.

    Returns:
        int: The integer value found in the file.
    """
    for line in file:
        if re.search(text, line):
            var = int(line.rsplit(None, 1)[-1])
            return var


def read_real(text: str, file) -> float:
    """Reads a floating-point value from a text file.

    Args:
        text (str): The text to search for in the file.
        f (file): The file object to read from.

    Returns:
        float: The floating-point value found in the file.
    """
    for line in file:
        if re.search(text, line):
            var = float(line.rsplit(None, 1)[-1])
            return var


def read_real_list(text: str, file) -> List[float]:
    """Reads a list of floating-point values from a text file.

    Args:
        text (str): The text to search for in the file.
        file (file object): The file object to read from.

    Returns:
        list: A list of floating-point values found in the file.
    """
    for line in file:
        if re.search(text, line):
            n = int(line.rsplit(None, 1)[-1])
            var = []
            for i in range((n-1)//5+1):
                line = next(file)
                for j in line.split():
                    var += [float(j)]
            return var