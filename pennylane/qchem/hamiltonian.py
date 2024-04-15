# Copyright 2018-2021 Xanadu Quantum Technologies Inc.

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
This module contains the functions needed for computing the molecular Hamiltonian.
"""
# pylint: disable= too-many-branches, too-many-arguments, too-many-locals, too-many-nested-blocks
import pennylane as qml

from .hartree_fock import nuclear_energy, scf
from .observable_hf import fermionic_observable, qubit_observable

from pennylane.operation import active_new_opmath
from pennylane.pauli.utils import simplify


def electron_integrals(mol, core=None, active=None):
    r"""Return a function that computes the one- and two-electron integrals in the molecular orbital
    basis.

    The one- and two-electron integrals are required to construct a molecular Hamiltonian in the
    second-quantized form

    .. math::

        H = \sum_{pq} h_{pq} c_p^{\dagger} c_q + \frac{1}{2} \sum_{pqrs} h_{pqrs} c_p^{\dagger} c_q^{\dagger} c_r c_s,

    where :math:`c^{\dagger}` and :math:`c` are the creation and annihilation operators,
    respectively, and :math:`h_{pq}` and :math:`h_{pqrs}` are the one- and two-electron integrals.
    These integrals can be computed by integrating over molecular orbitals :math:`\phi` as

    .. math::

        h_{pq} = \int \phi_p(r)^* \left ( -\frac{\nabla_r^2}{2} - \sum_i \frac{Z_i}{|r-R_i|} \right )  \phi_q(r) dr,

    and

    .. math::

        h_{pqrs} = \int \frac{\phi_p(r_1)^* \phi_q(r_2)^* \phi_r(r_2) \phi_s(r_1)}{|r_1 - r_2|} dr_1 dr_2.

    The molecular orbitals are constructed as a linear combination of atomic orbitals as

    .. math::

        \phi_i = \sum_{\nu}c_{\nu}^i \chi_{\nu}.

    The one- and two-electron integrals can be written in the molecular orbital basis as

    .. math::

        h_{pq} = \sum_{\mu \nu} C_{p \mu} h_{\mu \nu} C_{\nu q},

    and

    .. math::

        h_{pqrs} = \sum_{\mu \nu \rho \sigma} C_{p \mu} C_{q \nu} h_{\mu \nu \rho \sigma} C_{\rho r} C_{\sigma s}.

    The :math:`h_{\mu \nu}` and :math:`h_{\mu \nu \rho \sigma}` terms refer to the elements of the
    core matrix and the electron repulsion tensor, respectively, and :math:`C` is the molecular
    orbital expansion coefficient matrix.

    Args:
        mol (~qchem.molecule.Molecule): the molecule object
        core (list[int]): indices of the core orbitals
        active (list[int]): indices of the active orbitals

    Returns:
        function: function that computes the core constant and the one- and two-electron integrals

    **Example**

    >>> symbols  = ['H', 'H']
    >>> geometry = np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 1.0]], requires_grad = False)
    >>> alpha = np.array([[3.42525091, 0.62391373, 0.1688554],
    >>>                   [3.42525091, 0.62391373, 0.1688554]], requires_grad=True)
    >>> mol = qml.qchem.Molecule(symbols, geometry, alpha=alpha)
    >>> args = [alpha]
    >>> electron_integrals(mol)(*args)
    (1.0,
     array([[-1.3902192695e+00,  0.0000000000e+00],
            [-4.4408920985e-16, -2.9165331336e-01]]),
     array([[[[ 7.1443907755e-01, -2.7755575616e-17],
              [ 5.5511151231e-17,  1.7024144301e-01]],
             [[ 5.5511151231e-17,  1.7024144301e-01],
              [ 7.0185315353e-01,  6.6613381478e-16]]],
            [[[-1.3877787808e-16,  7.0185315353e-01],
              [ 1.7024144301e-01,  2.2204460493e-16]],
             [[ 1.7024144301e-01, -4.4408920985e-16],
              [ 6.6613381478e-16,  7.3883668974e-01]]]]))
    """

    def _electron_integrals(*args):
        r"""Compute the one- and two-electron integrals in the molecular orbital basis.

        Args:
            *args (array[array[float]]): initial values of the differentiable parameters

        Returns:
            tuple[array[float]]: 1D tuple containing core constant, one- and two-electron integrals
        """
        _, coeffs, _, h_core, repulsion_tensor = scf(mol)(*args)
        one = qml.math.einsum("qr,rs,st->qt", coeffs.T, h_core, coeffs)
        two = qml.math.swapaxes(
            qml.math.einsum(
                "ab,cd,bdeg,ef,gh->acfh", coeffs.T, coeffs.T, repulsion_tensor, coeffs, coeffs
            ),
            1,
            3,
        )
        core_constant = nuclear_energy(mol.nuclear_charges, mol.coordinates)(*args)

        if core is None and active is None:
            return core_constant, one, two

        for i in core:
            core_constant = core_constant + 2 * one[i][i]
            for j in core:
                core_constant = core_constant + 2 * two[i][j][j][i] - two[i][j][i][j]

        for p in active:
            for q in active:
                for i in core:
                    o = qml.math.zeros(one.shape)
                    o[p, q] = 1.0
                    one = one + (2 * two[i][p][q][i] - two[i][p][i][q]) * o

        one = one[qml.math.ix_(active, active)]
        two = two[qml.math.ix_(active, active, active, active)]

        return core_constant, one, two

    return _electron_integrals


def fermionic_hamiltonian(mol, cutoff=1.0e-12, core=None, active=None):
    r"""Return a function that computes the fermionic Hamiltonian.

    Args:
        mol (~qchem.molecule.Molecule): the molecule object
        cutoff (float): cutoff value for discarding the negligible electronic integrals
        core (list[int]): indices of the core orbitals
        active (list[int]): indices of the active orbitals

    Returns:
        function: function that computes the fermionic hamiltonian

    **Example**

    >>> symbols  = ['H', 'H']
    >>> geometry = np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 1.0]], requires_grad = False)
    >>> alpha = np.array([[3.42525091, 0.62391373, 0.1688554],
    >>>                   [3.42525091, 0.62391373, 0.1688554]], requires_grad=True)
    >>> mol = qml.qchem.Molecule(symbols, geometry, alpha=alpha)
    >>> args = [alpha]
    >>> h = fermionic_hamiltonian(mol)(*args)
    """

    def _fermionic_hamiltonian(*args):
        r"""Compute the fermionic hamiltonian.

        Args:
            *args (array[array[float]]): initial values of the differentiable parameters

        Returns:
            FermiSentence: fermionic Hamiltonian
        """

        core_constant, one, two = electron_integrals(mol, core, active)(*args)

        return fermionic_observable(core_constant, one, two, cutoff)

    return _fermionic_hamiltonian


def diff_hamiltonian(mol, cutoff=1.0e-12, core=None, active=None):
    r"""Return a function that computes the qubit Hamiltonian.

    Args:
        mol (~qchem.molecule.Molecule): the molecule object
        cutoff (float): cutoff value for discarding the negligible electronic integrals
        core (list[int]): indices of the core orbitals
        active (list[int]): indices of the active orbitals

    Returns:
        function: function that computes the qubit hamiltonian

    **Example**

    >>> symbols  = ['H', 'H']
    >>> geometry = np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 1.0]], requires_grad = False)
    >>> alpha = np.array([[3.42525091, 0.62391373, 0.1688554],
    >>>                   [3.42525091, 0.62391373, 0.1688554]], requires_grad=True)
    >>> mol = qml.qchem.Molecule(symbols, geometry, alpha=alpha)
    >>> args = [alpha]
    >>> h = diff_hamiltonian(mol)(*args)
    >>> h.coeffs
    array([ 0.29817879+0.j,  0.20813365+0.j,  0.20813365+0.j,
             0.17860977+0.j,  0.04256036+0.j, -0.04256036+0.j,
            -0.04256036+0.j,  0.04256036+0.j, -0.34724873+0.j,
             0.13290293+0.j, -0.34724873+0.j,  0.17546329+0.j,
             0.17546329+0.j,  0.13290293+0.j,  0.18470917+0.j])
    """

    def _molecular_hamiltonian(*args):
        r"""Compute the qubit hamiltonian.

        Args:
            *args (array[array[float]]): initial values of the differentiable parameters

        Returns:
            Hamiltonian: the qubit Hamiltonian
        """

        h_ferm = fermionic_hamiltonian(mol, cutoff, core, active)(*args)

        return qubit_observable(h_ferm)

    return _molecular_hamiltonian


def molecular_hamiltonian(
    symbols,
    coordinates,
    name="molecule",
    charge=0,
    mult=1,
    basis="sto-3g",
    method="dhf",
    active_electrons=None,
    active_orbitals=None,
    mapping="jordan_wigner",
    outpath=".",
    wires=None,
    alpha=None,
    coeff=None,
    args=None,
    load_data=False,
    convert_tol=1e012,
):  # pylint:disable=too-many-arguments, too-many-statements
    r"""Generate the qubit Hamiltonian of a molecule.

    This function drives the construction of the second-quantized electronic Hamiltonian
    of a molecule and its transformation to the basis of Pauli matrices.

    The net charge of the molecule can be given to simulate cationic/anionic systems. Also, the
    spin multiplicity can be input to determine the number of unpaired electrons occupying the HF
    orbitals as illustrated in the left panel of the figure below.

    The basis of Gaussian-type *atomic* orbitals used to represent the *molecular* orbitals can be
    specified to go beyond the minimum basis approximation.

    An active space can be defined for a given number of *active electrons* occupying a reduced set
    of *active orbitals* as sketched in the right panel of the figure below.

    |

    .. figure:: ../../_static/qchem/fig_mult_active_space.png
        :align: center
        :width: 90%

    |

    Args:
        symbols (list[str]): symbols of the atomic species in the molecule
        coordinates (array[float]): atomic positions in Cartesian coordinates.
            The atomic coordinates must be in atomic units and can be given as either a 1D array of
            size ``3*N``, or a 2D array of shape ``(N, 3)`` where ``N`` is the number of atoms.
        name (str): name of the molecule
        charge (int): Net charge of the molecule. If not specified a neutral system is assumed.
        mult (int): Spin multiplicity :math:`\mathrm{mult}=N_\mathrm{unpaired} + 1`
            for :math:`N_\mathrm{unpaired}` unpaired electrons occupying the HF orbitals.
            Possible values of ``mult`` are :math:`1, 2, 3, \ldots`. If not specified,
            a closed-shell HF state is assumed.
        basis (str): atomic basis set used to represent the molecular orbitals
        method (str): Quantum chemistry method used to solve the
            mean field electronic structure problem. Available options are ``method="dhf"``
            to specify the built-in differentiable Hartree-Fock solver, ``method="pyscf"`` to use
            the PySCF package (requires ``pyscf`` to be installed), or ``method="openfermion"`` to
            use the OpenFermion-PySCF plugin (this requires ``openfermionpyscf`` to be installed).
        active_electrons (int): Number of active electrons. If not specified, all electrons
            are considered to be active.
        active_orbitals (int): Number of active orbitals. If not specified, all orbitals
            are considered to be active.
        mapping (str): transformation used to map the fermionic Hamiltonian to the qubit Hamiltonian
        outpath (str): path to the directory containing output files
        wires (Wires, list, tuple, dict): Custom wire mapping for connecting to Pennylane ansatz.
            For types ``Wires``/``list``/``tuple``, each item in the iterable represents a wire label
            corresponding to the qubit number equal to its index.
            For type dict, only int-keyed dict (for qubit-to-wire conversion) is accepted for
            partial mapping. If None, will use identity map.
        alpha (array[float]): exponents of the primitive Gaussian functions
        coeff (array[float]): coefficients of the contracted Gaussian functions
        args (array[array[float]]): initial values of the differentiable parameters
        load_data (bool): flag to load data from the basis-set-exchange library
        convert_tol (float): Tolerance in `machine epsilon <https://numpy.org/doc/stable/reference/generated/numpy.real_if_close.html>`_
            for the imaginary part of the Hamiltonian coefficients created by openfermion.
            Coefficients with imaginary part less than 2.22e-16*tol are considered to be real.

    Returns:
        tuple[pennylane.Hamiltonian, int]: the fermionic-to-qubit transformed Hamiltonian
        and the number of qubits

    **Example**

    >>> symbols, coordinates = (['H', 'H'], np.array([0., 0., -0.66140414, 0., 0., 0.66140414]))
    >>> H, qubits = molecular_hamiltonian(symbols, coordinates)
    >>> print(qubits)
    4
    >>> print(H)
    (-0.04207897647782188) [I0]
    + (0.17771287465139934) [Z0]
    + (0.1777128746513993) [Z1]
    + (-0.24274280513140484) [Z2]
    + (-0.24274280513140484) [Z3]
    + (0.17059738328801055) [Z0 Z1]
    + (0.04475014401535161) [Y0 X1 X2 Y3]
    + (-0.04475014401535161) [Y0 Y1 X2 X3]
    + (-0.04475014401535161) [X0 X1 Y2 Y3]
    + (0.04475014401535161) [X0 Y1 Y2 X3]
    + (0.12293305056183801) [Z0 Z2]
    + (0.1676831945771896) [Z0 Z3]
    + (0.1676831945771896) [Z1 Z2]
    + (0.12293305056183801) [Z1 Z3]
    + (0.176276408043196) [Z2 Z3]
    """

    if method not in ["dhf", "pyscf", "openfermion"]:
        raise ValueError("Only 'dhf', 'pyscf' and 'openfermion' backends are supported.")

    if len(coordinates) == len(symbols) * 3:
        geometry_dhf = qml.numpy.array(coordinates.reshape(len(symbols), 3))
        geometry_hf = coordinates
    elif len(coordinates) == len(symbols):
        geometry_dhf = qml.numpy.array(coordinates)
        geometry_hf = coordinates.flatten()

    wires_map = None

    if wires:
        wires_new = qml.qchem.convert._process_wires(wires)
        wires_map = dict(zip(range(len(wires_new)), list(wires_new.labels)))

    if method == "dhf":

        if mapping != "jordan_wigner":
            raise ValueError(
                "Only 'jordan_wigner' mapping is supported for the differentiable workflow."
            )
        if mult != 1:
            raise ValueError(
                "Openshell systems are not supported for the differentiable workflow. Use "
                "`method = 'pyscf'` or change the charge or spin multiplicity of the molecule."
            )
        if args is None and isinstance(geometry_dhf, qml.numpy.tensor):
            geometry_dhf.requires_grad = False
        mol = qml.qchem.Molecule(
            symbols,
            geometry_dhf,
            charge=charge,
            mult=mult,
            basis_name=basis,
            load_data=load_data,
            alpha=alpha,
            coeff=coeff,
        )
        core, active = qml.qchem.active_space(
            mol.n_electrons, mol.n_orbitals, mult, active_electrons, active_orbitals
        )

        requires_grad = args is not None
        h = (
            qml.qchem.diff_hamiltonian(mol, core=core, active=active)(*args)
            if requires_grad
            else qml.qchem.diff_hamiltonian(mol, core=core, active=active)()
        )

        if active_new_opmath():
            h_as_ps = qml.pauli.pauli_sentence(h)
            coeffs = qml.numpy.real(list(h_as_ps.values()), requires_grad=requires_grad)

            h_as_ps = qml.pauli.PauliSentence(dict(zip(h_as_ps.keys(), coeffs)))
            h = (
                qml.s_prod(0, qml.Identity(h.wires[0]))
                if len(h_as_ps) == 0
                else h_as_ps.operation()
            )
        else:
            coeffs = qml.numpy.real(h.coeffs, requires_grad=requires_grad)
            h = qml.Hamiltonian(coeffs, h.ops)

        if wires:
            h = qml.map_wires(h, wires_map)
        return h, 2 * len(active)

    if method == "pyscf":
        core_constant, one_mo, two_mo = qml.qchem.openfermion_obs._pyscf_integrals(
            symbols, geometry_hf, charge, mult, basis, active_electrons, active_orbitals
        )

        hf = qml.qchem.fermionic_observable(core_constant, one_mo, two_mo)

        if active_new_opmath():
            h_pl = qml.jordan_wigner(hf, wire_map=wires_map, tol=1.0e-10).simplify()

        else:
            h_pl = qml.jordan_wigner(hf, ps=True, wire_map=wires_map, tol=1.0e-10).hamiltonian()
            h_pl = simplify(h_pl)

        return h_pl, len(h_pl.wires)

    h_pl = qml.qchem.openfermion_obs._openfermion_hamiltonian(
        symbols,
        coordinates,
        name,
        charge,
        mult,
        basis,
        active_electrons,
        active_orbitals,
        mapping,
        outpath,
        wires,
        convert_tol,
    )

    return h_pl, len(h_pl.wires)
