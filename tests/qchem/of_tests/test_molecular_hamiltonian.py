# Copyright 2018-2023 Xanadu Quantum Technologies Inc.

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
Unit tests for molecular Hamiltonians.
"""
# pylint: disable=too-many-arguments, protected-access
import pytest

import pennylane as qml
from pennylane import Identity, PauliX, PauliY, PauliZ
from pennylane import numpy as np
from pennylane import qchem
from pennylane.operation import active_new_opmath

test_symbols = ["C", "C", "N", "H", "H", "H", "H", "H"]
test_coordinates = np.array(
    [
        0.68219113,
        -0.85415621,
        -1.04123909,
        -1.34926445,
        0.23621577,
        0.61794044,
        1.29068294,
        0.25133357,
        1.40784596,
        0.83525895,
        -2.88939124,
        -1.16974047,
        1.26989596,
        0.19275206,
        -2.69852891,
        -2.57758643,
        -1.05824663,
        1.61949529,
        -2.17129532,
        2.04090421,
        0.11338357,
        2.06547065,
        2.00877887,
        1.20186581,
    ]
)


@pytest.mark.parametrize(
    (
        "charge",
        "mult",
        "package",
        "nact_els",
        "nact_orbs",
        "mapping",
    ),
    [
        (0, 1, "pyscf", 2, 2, "jordan_WIGNER"),
        (1, 2, "pyscf", 3, 4, "BRAVYI_kitaev"),
        (-1, 2, "pyscf", 1, 2, "jordan_WIGNER"),
        (2, 1, "pyscf", 2, 2, "BRAVYI_kitaev"),
    ],
)
@pytest.mark.usefixtures("skip_if_no_openfermion_support", "use_legacy_and_new_opmath")
def test_building_hamiltonian(
    charge,
    mult,
    package,
    nact_els,
    nact_orbs,
    mapping,
    tmpdir,
):
    r"""Test that the generated Hamiltonian `built_hamiltonian` is an instance of the PennyLane
    Hamiltonian class and the correctness of the total number of qubits required to run the
    quantum simulation. The latter is tested for different values of the molecule's charge and
    for active spaces with different size"""

    args = (test_symbols, test_coordinates)
    kwargs = {
        "charge": charge,
        "mult": mult,
        "method": package,
        "active_electrons": nact_els,
        "active_orbitals": nact_orbs,
        "mapping": mapping,
        "outpath": tmpdir.strpath,
    }

    built_hamiltonian, qubits = qchem.molecular_hamiltonian(*args, **kwargs)

    if active_new_opmath():
        assert not isinstance(built_hamiltonian, qml.Hamiltonian)
    else:
        assert isinstance(built_hamiltonian, qml.Hamiltonian)
    assert qubits == 2 * nact_orbs


@pytest.mark.parametrize(
    (
        "charge",
        "mult",
        "package",
        "nact_els",
        "nact_orbs",
        "mapping",
    ),
    [
        (0, 1, "pyscf", 2, 2, "jordan_WIGNER"),
        (2, 1, "pyscf", 2, 2, "BRAVYI_kitaev"),
    ],
)
@pytest.mark.usefixtures("skip_if_no_openfermion_support", "use_legacy_and_new_opmath")
def test_building_hamiltonian_molecule_class(
    charge,
    mult,
    package,
    nact_els,
    nact_orbs,
    mapping,
    tmpdir,
):
    r"""Test that the generated Hamiltonian `built_hamiltonian` using the molecule class, is an
    instance of the PennyLane Hamiltonian class and the correctness of the total number of qubits
    required to run the quantum simulation. The latter is tested for different values of the
    molecule's charge and for active spaces with different size"""

    args = qchem.Molecule(test_symbols, test_coordinates, charge=charge, mult=mult)
    kwargs = {
        "method": package,
        "active_electrons": nact_els,
        "active_orbitals": nact_orbs,
        "mapping": mapping,
        "outpath": tmpdir.strpath,
    }

    built_hamiltonian, qubits = qchem.molecular_hamiltonian(args, **kwargs)

    if active_new_opmath():
        assert not isinstance(built_hamiltonian, qml.Hamiltonian)
    else:
        assert isinstance(built_hamiltonian, qml.Hamiltonian)
    assert qubits == 2 * nact_orbs


@pytest.mark.parametrize(
    ("symbols", "geometry", "h_ref_data"),
    [
        (
            ["H", "H"],
            np.array([0.0, 0.0, 0.0, 0.0, 0.0, 1.0]),
            # computed with OpenFermion; data reordered
            # h_mol = molecule.get_molecular_hamiltonian()
            # h_f = openfermion.transforms.get_fermion_operator(h_mol)
            # h_q = openfermion.transforms.jordan_wigner(h_f)
            # h_pl = qchem.convert_observable(h_q, wires=[0, 1, 2, 3], tol=(5e-5))
            (
                np.array(
                    [
                        0.2981788017,
                        0.2081336485,
                        0.2081336485,
                        0.1786097698,
                        0.042560361,
                        -0.042560361,
                        -0.042560361,
                        0.042560361,
                        -0.3472487379,
                        0.1329029281,
                        -0.3472487379,
                        0.175463289,
                        0.175463289,
                        0.1329029281,
                        0.1847091733,
                    ]
                ),
                [
                    Identity(wires=[0]),
                    PauliZ(wires=[0]),
                    PauliZ(wires=[1]),
                    PauliZ(wires=[0]) @ PauliZ(wires=[1]),
                    PauliY(wires=[0]) @ PauliX(wires=[1]) @ PauliX(wires=[2]) @ PauliY(wires=[3]),
                    PauliY(wires=[0]) @ PauliY(wires=[1]) @ PauliX(wires=[2]) @ PauliX(wires=[3]),
                    PauliX(wires=[0]) @ PauliX(wires=[1]) @ PauliY(wires=[2]) @ PauliY(wires=[3]),
                    PauliX(wires=[0]) @ PauliY(wires=[1]) @ PauliY(wires=[2]) @ PauliX(wires=[3]),
                    PauliZ(wires=[2]),
                    PauliZ(wires=[0]) @ PauliZ(wires=[2]),
                    PauliZ(wires=[3]),
                    PauliZ(wires=[0]) @ PauliZ(wires=[3]),
                    PauliZ(wires=[1]) @ PauliZ(wires=[2]),
                    PauliZ(wires=[1]) @ PauliZ(wires=[3]),
                    PauliZ(wires=[2]) @ PauliZ(wires=[3]),
                ],
            ),
        ),
        (
            ["H", "H"],
            np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 1.0]]),
            # computed with OpenFermion; data reordered
            # h_mol = molecule.get_molecular_hamiltonian()
            # h_f = openfermion.transforms.get_fermion_operator(h_mol)
            # h_q = openfermion.transforms.jordan_wigner(h_f)
            # h_pl = qchem.convert_observable(h_q, wires=[0, 1, 2, 3], tol=(5e-5))
            (
                np.array(
                    [
                        0.2981788017,
                        0.2081336485,
                        0.2081336485,
                        0.1786097698,
                        0.042560361,
                        -0.042560361,
                        -0.042560361,
                        0.042560361,
                        -0.3472487379,
                        0.1329029281,
                        -0.3472487379,
                        0.175463289,
                        0.175463289,
                        0.1329029281,
                        0.1847091733,
                    ]
                ),
                [
                    Identity(wires=[0]),
                    PauliZ(wires=[0]),
                    PauliZ(wires=[1]),
                    PauliZ(wires=[0]) @ PauliZ(wires=[1]),
                    PauliY(wires=[0]) @ PauliX(wires=[1]) @ PauliX(wires=[2]) @ PauliY(wires=[3]),
                    PauliY(wires=[0]) @ PauliY(wires=[1]) @ PauliX(wires=[2]) @ PauliX(wires=[3]),
                    PauliX(wires=[0]) @ PauliX(wires=[1]) @ PauliY(wires=[2]) @ PauliY(wires=[3]),
                    PauliX(wires=[0]) @ PauliY(wires=[1]) @ PauliY(wires=[2]) @ PauliX(wires=[3]),
                    PauliZ(wires=[2]),
                    PauliZ(wires=[0]) @ PauliZ(wires=[2]),
                    PauliZ(wires=[3]),
                    PauliZ(wires=[0]) @ PauliZ(wires=[3]),
                    PauliZ(wires=[1]) @ PauliZ(wires=[2]),
                    PauliZ(wires=[1]) @ PauliZ(wires=[3]),
                    PauliZ(wires=[2]) @ PauliZ(wires=[3]),
                ],
            ),
        ),
    ],
)
@pytest.mark.usefixtures("use_legacy_and_new_opmath")
def test_differentiable_hamiltonian(symbols, geometry, h_ref_data):
    r"""Test that molecular_hamiltonian returns the correct Hamiltonian with the differentiable
    backend."""

    geometry.requires_grad = True
    args = [geometry.reshape(2, 3)]
    h_args = qchem.molecular_hamiltonian(symbols, geometry, method="dhf", args=args)[0]

    geometry.requires_grad = False
    h_noargs = qchem.molecular_hamiltonian(symbols, geometry, method="dhf")[0]

    ops = [
        qml.operation.Tensor(*op) if isinstance(op, qml.ops.Prod) else op
        for op in map(qml.simplify, h_ref_data[1])
    ]
    h_ref = qml.Hamiltonian(h_ref_data[0], ops)

    h_ref_coeffs, h_ref_ops = h_ref.terms()
    h_args_coeffs, h_args_ops = h_args.terms()
    h_noargs_coeffs, h_noargs_ops = h_noargs.terms()

    assert all(coeff.requires_grad is True for coeff in h_args_coeffs)
    assert all(coeff.requires_grad is False for coeff in h_noargs_coeffs)

    assert np.allclose(np.sort(h_args_coeffs), np.sort(h_ref_coeffs))
    assert qml.Hamiltonian(np.ones(len(h_args_coeffs)), h_args_ops).compare(
        qml.Hamiltonian(np.ones(len(h_ref_coeffs)), h_ref_ops)
    )

    assert np.allclose(np.sort(h_noargs_coeffs), np.sort(h_ref_coeffs))
    assert qml.Hamiltonian(np.ones(len(h_noargs_coeffs)), h_noargs_ops).compare(
        qml.Hamiltonian(np.ones(len(h_ref_coeffs)), h_ref_ops)
    )


@pytest.mark.parametrize(
    ("symbols", "geometry", "h_ref_data"),
    [
        (
            ["H", "H"],
            np.array([0.0, 0.0, 0.0, 0.0, 0.0, 1.0]),
            # computed with OpenFermion; data reordered
            # h_mol = molecule.get_molecular_hamiltonian()
            # h_f = openfermion.transforms.get_fermion_operator(h_mol)
            # h_q = openfermion.transforms.jordan_wigner(h_f)
            # h_pl = qchem.convert_observable(h_q, wires=[0, 1, 2, 3], tol=(5e-5))
            (
                np.array(
                    [
                        0.2981788017,
                        0.2081336485,
                        0.2081336485,
                        0.1786097698,
                        0.042560361,
                        -0.042560361,
                        -0.042560361,
                        0.042560361,
                        -0.3472487379,
                        0.1329029281,
                        -0.3472487379,
                        0.175463289,
                        0.175463289,
                        0.1329029281,
                        0.1847091733,
                    ]
                ),
                [
                    Identity(wires=[0]),
                    PauliZ(wires=[0]),
                    PauliZ(wires=[1]),
                    PauliZ(wires=[0]) @ PauliZ(wires=[1]),
                    PauliY(wires=[0]) @ PauliX(wires=[1]) @ PauliX(wires=[2]) @ PauliY(wires=[3]),
                    PauliY(wires=[0]) @ PauliY(wires=[1]) @ PauliX(wires=[2]) @ PauliX(wires=[3]),
                    PauliX(wires=[0]) @ PauliX(wires=[1]) @ PauliY(wires=[2]) @ PauliY(wires=[3]),
                    PauliX(wires=[0]) @ PauliY(wires=[1]) @ PauliY(wires=[2]) @ PauliX(wires=[3]),
                    PauliZ(wires=[2]),
                    PauliZ(wires=[0]) @ PauliZ(wires=[2]),
                    PauliZ(wires=[3]),
                    PauliZ(wires=[0]) @ PauliZ(wires=[3]),
                    PauliZ(wires=[1]) @ PauliZ(wires=[2]),
                    PauliZ(wires=[1]) @ PauliZ(wires=[3]),
                    PauliZ(wires=[2]) @ PauliZ(wires=[3]),
                ],
            ),
        ),
        (
            ["H", "H"],
            np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 1.0]]),
            # computed with OpenFermion; data reordered
            # h_mol = molecule.get_molecular_hamiltonian()
            # h_f = openfermion.transforms.get_fermion_operator(h_mol)
            # h_q = openfermion.transforms.jordan_wigner(h_f)
            # h_pl = qchem.convert_observable(h_q, wires=[0, 1, 2, 3], tol=(5e-5))
            (
                np.array(
                    [
                        0.2981788017,
                        0.2081336485,
                        0.2081336485,
                        0.1786097698,
                        0.042560361,
                        -0.042560361,
                        -0.042560361,
                        0.042560361,
                        -0.3472487379,
                        0.1329029281,
                        -0.3472487379,
                        0.175463289,
                        0.175463289,
                        0.1329029281,
                        0.1847091733,
                    ]
                ),
                [
                    Identity(wires=[0]),
                    PauliZ(wires=[0]),
                    PauliZ(wires=[1]),
                    PauliZ(wires=[0]) @ PauliZ(wires=[1]),
                    PauliY(wires=[0]) @ PauliX(wires=[1]) @ PauliX(wires=[2]) @ PauliY(wires=[3]),
                    PauliY(wires=[0]) @ PauliY(wires=[1]) @ PauliX(wires=[2]) @ PauliX(wires=[3]),
                    PauliX(wires=[0]) @ PauliX(wires=[1]) @ PauliY(wires=[2]) @ PauliY(wires=[3]),
                    PauliX(wires=[0]) @ PauliY(wires=[1]) @ PauliY(wires=[2]) @ PauliX(wires=[3]),
                    PauliZ(wires=[2]),
                    PauliZ(wires=[0]) @ PauliZ(wires=[2]),
                    PauliZ(wires=[3]),
                    PauliZ(wires=[0]) @ PauliZ(wires=[3]),
                    PauliZ(wires=[1]) @ PauliZ(wires=[2]),
                    PauliZ(wires=[1]) @ PauliZ(wires=[3]),
                    PauliZ(wires=[2]) @ PauliZ(wires=[3]),
                ],
            ),
        ),
    ],
)
@pytest.mark.usefixtures("use_legacy_and_new_opmath")
def test_differentiable_hamiltonian_molecule_class(symbols, geometry, h_ref_data):
    r"""Test that molecular_hamiltonian generated using the molecule class
    returns the correct Hamiltonian with the differentiable backend."""

    geometry.requires_grad = True
    args = [geometry.reshape(2, 3)]
    molecule = qchem.Molecule(symbols, geometry)
    h_args = qchem.molecular_hamiltonian(molecule, method="dhf", args=args)[0]

    geometry.requires_grad = False
    molecule = qchem.Molecule(symbols, geometry)
    h_noargs = qchem.molecular_hamiltonian(molecule, method="dhf")[0]

    ops = [
        qml.operation.Tensor(*op) if isinstance(op, qml.ops.Prod) else op
        for op in map(qml.simplify, h_ref_data[1])
    ]
    h_ref = qml.Hamiltonian(h_ref_data[0], ops)

    h_ref_coeffs, h_ref_ops = h_ref.terms()
    h_args_coeffs, h_args_ops = h_args.terms()
    h_noargs_coeffs, h_noargs_ops = h_noargs.terms()

    assert all(coeff.requires_grad is True for coeff in h_args_coeffs)
    assert all(coeff.requires_grad is False for coeff in h_noargs_coeffs)

    assert np.allclose(np.sort(h_args_coeffs), np.sort(h_ref_coeffs))
    assert qml.Hamiltonian(np.ones(len(h_args_coeffs)), h_args_ops).compare(
        qml.Hamiltonian(np.ones(len(h_ref_coeffs)), h_ref_ops)
    )

    assert np.allclose(np.sort(h_noargs_coeffs), np.sort(h_ref_coeffs))
    assert qml.Hamiltonian(np.ones(len(h_noargs_coeffs)), h_noargs_ops).compare(
        qml.Hamiltonian(np.ones(len(h_ref_coeffs)), h_ref_ops)
    )


@pytest.mark.usefixtures("use_legacy_and_new_opmath")
@pytest.mark.parametrize(
    ("symbols", "geometry", "method", "wiremap"),
    [
        (
            ["H", "H"],
            np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 2.0]]),
            "pyscf",
            ["a", "b", "c", "d"],
        ),
        (
            ["H", "H"],
            np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 2.0]]),
            "pyscf",
            [0, "z", 3, "ancilla"],
        ),
    ],
)
@pytest.mark.usefixtures("skip_if_no_openfermion_support")
def test_custom_wiremap_hamiltonian_pyscf(symbols, geometry, method, wiremap, tmpdir):
    r"""Test that the generated Hamiltonian has the correct wire labels given by a custom wiremap."""

    hamiltonian, _ = qchem.molecular_hamiltonian(
        symbols=symbols,
        coordinates=geometry,
        method=method,
        wires=wiremap,
        outpath=tmpdir.strpath,
    )

    assert set(hamiltonian.wires) == set(wiremap)


@pytest.mark.parametrize(
    ("symbols", "geometry", "method", "wiremap"),
    [
        (
            ["H", "H"],
            np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 2.0]]),
            "pyscf",
            ["a", "b", "c", "d"],
        ),
        (
            ["H", "H"],
            np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 2.0]]),
            "pyscf",
            [0, "z", 3, "ancilla"],
        ),
    ],
)
@pytest.mark.usefixtures("skip_if_no_openfermion_support")
def test_custom_wiremap_hamiltonian_pyscf_molecule_class(
    symbols, geometry, method, wiremap, tmpdir
):
    r"""Test that the generated Hamiltonian has the correct wire labels given by a custom wiremap."""

    molecule = qchem.Molecule(symbols, geometry)
    hamiltonian, _ = qchem.molecular_hamiltonian(
        molecule,
        method=method,
        wires=wiremap,
        outpath=tmpdir.strpath,
    )

    assert set(hamiltonian.wires) == set(wiremap)


@pytest.mark.usefixtures("use_legacy_and_new_opmath")
@pytest.mark.parametrize(
    ("symbols", "geometry", "wiremap", "args"),
    [
        (
            ["H", "H"],
            np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 2.0]]),
            [0, "z", 3, "ancilla"],
            None,
        ),
        (
            ["H", "H"],
            np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 2.0]]),
            [0, "z", 3, "ancilla"],
            [np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 2.0]])],
        ),
    ],
)
def test_custom_wiremap_hamiltonian_dhf(symbols, geometry, wiremap, args, tmpdir):
    r"""Test that the generated Hamiltonian has the correct wire labels given by a custom wiremap."""

    wiremap_dict = dict(zip(range(len(wiremap)), wiremap))

    hamiltonian_ref, _ = qchem.molecular_hamiltonian(
        symbols=symbols,
        coordinates=geometry,
        args=args,
        outpath=tmpdir.strpath,
    )

    hamiltonian, _ = qchem.molecular_hamiltonian(
        symbols=symbols,
        coordinates=geometry,
        wires=wiremap,
        args=args,
        outpath=tmpdir.strpath,
    )

    wiremap_calc = dict(zip(list(hamiltonian_ref.wires), list(hamiltonian.wires)))

    assert wiremap_calc == wiremap_dict


@pytest.mark.usefixtures("use_legacy_and_new_opmath")
@pytest.mark.parametrize(
    ("symbols", "geometry", "wiremap", "args"),
    [
        (
            ["H", "H"],
            np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 2.0]]),
            [0, "z", 3, "ancilla"],
            None,
        ),
        (
            ["H", "H"],
            np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 2.0]]),
            [0, "z", 3, "ancilla"],
            [np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 2.0]])],
        ),
    ],
)
def test_custom_wiremap_hamiltonian_dhf_molecule_class(symbols, geometry, wiremap, args, tmpdir):
    r"""Test that the generated Hamiltonian has the correct wire labels given by a custom wiremap."""

    wiremap_dict = dict(zip(range(len(wiremap)), wiremap))

    molecule = qchem.Molecule(symbols, geometry)
    hamiltonian_ref, _ = qchem.molecular_hamiltonian(
        molecule,
        args=args,
        outpath=tmpdir.strpath,
    )

    hamiltonian, _ = qchem.molecular_hamiltonian(
        molecule,
        wires=wiremap,
        args=args,
        outpath=tmpdir.strpath,
    )

    wiremap_calc = dict(zip(list(hamiltonian_ref.wires), list(hamiltonian.wires)))

    assert wiremap_calc == wiremap_dict


file_content = """\
2
in Angstrom
H          0.00000        0.00000       -0.35000
H          0.00000        0.00000        0.35000
"""


def test_mol_hamiltonian_with_read_structure(tmpdir):
    """Test that the pipeline of using molecular_hamiltonian with
    read_structure executes without errors."""
    f_name = "h2.xyz"
    filename = tmpdir.join(f_name)

    with open(filename, "w") as f:
        f.write(file_content)

    symbols, coordinates = qchem.read_structure(str(filename), outpath=tmpdir)
    H, num_qubits = qchem.molecular_hamiltonian(symbols, coordinates)
    assert len(H.terms()) == 2
    assert num_qubits == 4


def test_mol_hamiltonian_with_read_structure_molecule_class(tmpdir):
    """Test that the pipeline of using molecular_hamiltonian with
    read_structure executes without errors."""
    f_name = "h2.xyz"
    filename = tmpdir.join(f_name)

    with open(filename, "w") as f:
        f.write(file_content)

    symbols, coordinates = qchem.read_structure(str(filename), outpath=tmpdir)

    molecule = qchem.Molecule(symbols, coordinates)
    H, num_qubits = qchem.molecular_hamiltonian(molecule)
    assert len(H.terms()) == 2
    assert num_qubits == 4


@pytest.mark.parametrize(
    ("symbols", "geometry"),
    [
        (
            ["H", "H"],
            np.array([0.0, 0.0, 0.0, 0.0, 0.0, 1.0]),
        ),
    ],
)
def test_diff_hamiltonian_error(symbols, geometry):
    r"""Test that molecular_hamiltonian raises an error with unsupported mapping."""

    with pytest.raises(ValueError, match="Only 'jordan_wigner' mapping is supported"):
        qchem.molecular_hamiltonian(symbols, geometry, method="dhf", mapping="bravyi_kitaev")

    with pytest.raises(
        ValueError, match="Only 'dhf', 'pyscf' and 'openfermion' backends are supported"
    ):
        qchem.molecular_hamiltonian(symbols, geometry, method="psi4")

    with pytest.raises(ValueError, match="Openshell systems are not supported"):
        qchem.molecular_hamiltonian(symbols, geometry, mult=3)


@pytest.mark.parametrize(
    ("symbols", "geometry"),
    [
        (
            ["H", "H"],
            np.array([0.0, 0.0, 0.0, 0.0, 0.0, 1.0]),
        ),
    ],
)
def test_diff_hamiltonian_error_molecule_class(symbols, geometry):
    r"""Test that molecular_hamiltonian raises an error with unsupported mapping."""

    with pytest.raises(ValueError, match="Only 'jordan_wigner' mapping is supported"):
        qchem.molecular_hamiltonian(symbols, geometry, method="dhf", mapping="bravyi_kitaev")

    molecule = qchem.Molecule(symbols, geometry)
    with pytest.raises(
        ValueError, match="Only 'dhf', 'pyscf' and 'openfermion' backends are supported"
    ):
        qchem.molecular_hamiltonian(molecule, method="psi4")


@pytest.mark.parametrize(
    ("symbols", "geometry", "method", "args"),
    [
        (
            ["H", "H"],
            np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 2.0]]),
            "pyscf",
            None,
        ),
        (
            ["H", "H"],
            np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 2.0]]),
            "dhf",
            None,
        ),
        (
            ["H", "H"],
            np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 2.0]]),
            "dhf",
            [np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 2.0]])],
        ),
    ],
)
@pytest.mark.usefixtures("skip_if_no_openfermion_support", "use_legacy_and_new_opmath")
def test_real_hamiltonian(symbols, geometry, method, args, tmpdir):
    r"""Test that the generated Hamiltonian has real coefficients."""

    hamiltonian, _ = qchem.molecular_hamiltonian(
        symbols=symbols,
        coordinates=geometry,
        method=method,
        args=args,
        outpath=tmpdir.strpath,
    )

    assert np.isrealobj(hamiltonian.terms()[0])


@pytest.mark.parametrize(
    ("symbols", "geometry", "method", "args"),
    [
        (
            ["H", "H"],
            np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 2.0]]),
            "pyscf",
            None,
        ),
        (
            ["H", "H"],
            np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 2.0]]),
            "dhf",
            None,
        ),
        (
            ["H", "H"],
            np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 2.0]]),
            "dhf",
            [np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 2.0]])],
        ),
    ],
)
@pytest.mark.usefixtures("skip_if_no_openfermion_support", "use_legacy_and_new_opmath")
def test_real_hamiltonian_molecule_class(symbols, geometry, method, args, tmpdir):
    r"""Test that the generated Hamiltonian has real coefficients."""

    molecule = qchem.Molecule(symbols, geometry)
    hamiltonian, _ = qchem.molecular_hamiltonian(
        molecule,
        method=method,
        args=args,
        outpath=tmpdir.strpath,
    )

    assert np.isrealobj(hamiltonian.terms()[0])


@pytest.mark.parametrize(
    ("symbols", "geometry", "core_ref", "one_ref", "two_ref"),
    [
        (
            ["H", "H"],
            np.array([0.0, 0.0, 0.0, 0.0, 0.0, 2.0]),
            np.array([0.5]),
            np.array([[-1.08269537e00, 1.88626892e-13], [1.88848936e-13, -6.04947784e-01]]),
            np.array(
                [
                    [
                        [[6.16219836e-01, -1.93289829e-13], [-1.93373095e-13, 2.00522469e-01]],
                        [[-1.93345340e-13, 2.00522469e-01], [6.13198399e-01, -1.86684002e-13]],
                    ],
                    [
                        [[-1.93289829e-13, 6.13198399e-01], [2.00522469e-01, -1.86572979e-13]],
                        [[2.00522469e-01, -1.86961557e-13], [-1.86684002e-13, 6.43874664e-01]],
                    ],
                ]
            ),
        ),
    ],
)
@pytest.mark.usefixtures("skip_if_no_openfermion_support")
def test_pyscf_integrals(symbols, geometry, core_ref, one_ref, two_ref):
    r"""Test that _pyscf_integrals returns correct integrals."""

    core, one, two = qchem.openfermion_obs._pyscf_integrals(symbols, geometry)

    assert np.allclose(core, core_ref)
    assert np.allclose(one, one_ref)
    assert np.allclose(two, two_ref)
