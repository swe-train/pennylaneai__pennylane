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
"""Simulate a quantum script."""
# pylint: disable=protected-access
from typing import Sequence
from numpy.random import default_rng, binomial

import pennylane as qml
from pennylane.typing import Result
from pennylane.wires import Wires

from .initialize_state import create_initial_state
from .apply_operation import apply_operation
from .measure import measure
from .sampling import measure_with_samples


INTERFACE_TO_LIKE = {
    # map interfaces known by autoray to themselves
    None: None,
    "numpy": "numpy",
    "autograd": "autograd",
    "jax": "jax",
    "torch": "torch",
    "tensorflow": "tensorflow",
    # map non-standard interfaces to those known by autoray
    "auto": None,
    "scipy": "numpy",
    "jax-jit": "jax",
    "jax-python": "jax",
    "JAX": "jax",
    "pytorch": "torch",
    "tf": "tensorflow",
    "tensorflow-autograph": "tensorflow",
    "tf-autograph": "tensorflow",
}


def _valid_flex_int(s):
    """Returns True if s is a non-negative integer."""
    return isinstance(s, int) and s >= 0


def _valid_flex_tuple(s):
    """Returns True if s is a tuple of the form (shots, copies)."""
    return (
        isinstance(s, tuple)
        and len(s) == 2
        and _valid_flex_int(s[0])
        and isinstance(s[1], int)
        and s[1] > 0
    )


class _FlexShots(qml.measurements.Shots):
    """Shots class that allows zero shots."""

    total_shots = None
    """The total number of shots to be executed."""

    shot_vector = None
    """The tuple of :class:`~ShotCopies` to be executed. Each element is of the form ``(shots, copies)``."""

    _SHOT_ERROR = ValueError(
        "Shots must be a single positive integer, a tuple pair of the form (shots, copies), or a sequence of these types."
    )

    _frozen = False

    # pylint: disable=super-init-not-called
    def __init__(self, shots=None):
        if shots is None:
            self.total_shots = None
            self.shot_vector = ()
        elif isinstance(shots, int):
            if shots < 0:
                raise self._SHOT_ERROR
            self.total_shots = shots
            self.shot_vector = (qml.measurements.ShotCopies(shots, 1),)
        elif isinstance(shots, Sequence):
            if not all(_valid_flex_int(s) or _valid_flex_tuple(s) for s in shots):
                raise self._SHOT_ERROR
            self.__all_tuple_init__([s if isinstance(s, tuple) else (s, 1) for s in shots])
        elif isinstance(shots, self.__class__):
            return  # self already _is_ shots as defined by __new__
        else:
            raise self._SHOT_ERROR

        self._frozen = True


def expand_state_over_wires(state, state_wires, all_wires, is_state_batched):
    """
    Expand and re-order a state given some initial and target wire orders, setting
    all additional wires to the 0 state.

    Args:
        state (~pennylane.typing.TensorLike): The state to re-order and expand
        state_wires (.Wires): The wire order of the inputted state
        all_wires (.Wires): The desired wire order
        is_state_batched (bool): Whether the state has a batch dimension or not

    Returns:
        TensorLike: The state in the new desired size and order
    """
    pad_width = 2 ** len(all_wires) - 2 ** len(state_wires)
    pad = (pad_width, 0) if qml.math.get_interface(state) == "torch" else (0, pad_width)
    shape = (2,) * len(all_wires)
    if is_state_batched:
        pad = ((0, 0), pad)
        batch_size = qml.math.shape(state)[0]
        shape = (batch_size,) + shape
        state = qml.math.reshape(state, (batch_size, -1))
    else:
        pad = (pad,)
        state = qml.math.flatten(state)

    state = qml.math.pad(state, pad, mode="constant")
    state = qml.math.reshape(state, shape)

    # re-order
    new_wire_order = Wires.unique_wires([all_wires, state_wires]) + state_wires
    desired_axes = [new_wire_order.index(w) for w in all_wires]
    if is_state_batched:
        desired_axes = [0] + [i + 1 for i in desired_axes]
    return qml.math.transpose(state, desired_axes)


def _postselection_postprocess(state, is_state_batched, shots):
    """Update state after projector is applied."""
    if is_state_batched:
        raise ValueError(
            "Cannot postselect on circuits with broadcasting. Use the "
            "qml.transforms.broadcast_expand transform to split a broadcasted "
            "tape into multiple non-broadcasted tapes before executing if "
            "postselection is used."
        )

    if qml.math.is_abstract(state):
        return state, shots, False

    norm = qml.math.norm(state)
    if not qml.math.is_abstract(norm) and qml.math.isclose(float(norm), 0.0):
        new_state = qml.math.cast_like(
            qml.math.full(qml.math.shape(state), qml.numpy.NaN, like=qml.math.get_interface(state)),
            state,
        )
        is_nan = True

    else:
        new_state = state / norm
        is_nan = False

        # defer_measurements will raise an error with batched shots or broadcasting so we can
        # assume that both the state and shots are unbatched.
        if shots:
            # Clip the number of shots using a binomial distribution using the probability of
            # measuring the postselected state.
            postselected_shots = (
                [binomial(s, float(norm)) for s in shots]
                if not qml.math.is_abstract(norm)
                else shots
            )

            # _FlexShots is used here since the binomial distribution could result in zero
            # valid samples
            shots = _FlexShots(postselected_shots)

    return new_state, shots, is_nan


def get_final_state(circuit, debugger=None, interface=None):
    """
    Get the final state that results from executing the given quantum script.

    This is an internal function that will be called by the successor to ``default.qubit``.

    Args:
        circuit (.QuantumScript): The single circuit to simulate
        debugger (._Debugger): The debugger to use
        interface (str): The machine learning interface to create the initial state with

    Returns:
        Tuple[TensorLike, bool]: A tuple containing the final state of the quantum script and
            whether the state has a batch dimension.

    """
    circuit = circuit.map_to_standard_wires()

    prep = None
    if len(circuit) > 0 and isinstance(circuit[0], qml.operation.StatePrepBase):
        prep = circuit[0]

    state = create_initial_state(circuit.op_wires, prep, like=INTERFACE_TO_LIKE[interface])

    # initial state is batched only if the state preparation (if it exists) is batched
    is_state_batched = bool(prep and prep.batch_size is not None)
    for op in circuit.operations[bool(prep) :]:
        state = apply_operation(op, state, is_state_batched=is_state_batched, debugger=debugger)

        # Handle postselection on mid-circuit measurements
        if isinstance(op, qml.Projector):
            state, circuit._shots, state_nan = _postselection_postprocess(
                state, is_state_batched, circuit.shots
            )
            if state_nan:
                return state, is_state_batched

        # new state is batched if i) the old state is batched, or ii) the new op adds a batch dim
        is_state_batched = is_state_batched or op.batch_size is not None

    if set(circuit.op_wires) < set(circuit.wires):
        state = expand_state_over_wires(
            state,
            Wires(range(len(circuit.op_wires))),
            Wires(range(circuit.num_wires)),
            is_state_batched,
        )

    return state, is_state_batched


def _get_single_nan_res(measurements, batch_size, interface, shots):
    """Helper to get NaN results for one item in a shot vector."""

    res = []

    for m in measurements:
        if isinstance(m, qml.measurements.SampleMP):
            res.append(qml.math.asarray([], like=interface))
            continue
        if isinstance(m, qml.measurements.CountsMP):
            res.append({})
            continue

        shape = m.shape(qml.device("default.qubit", wires=m.wires), qml.measurements.Shots(shots))
        if batch_size is not None:
            shape = (batch_size,) + shape

        if shape == ():
            out = qml.math.asarray(qml.numpy.NaN, like=interface)
        else:
            out = qml.math.full(shape, qml.numpy.NaN, like=interface)

        res.append(out)

    res = tuple(res)

    if len(res) == 1:
        res = res[0]

    return res


def _measure_nan_state(circuit, state, is_state_batched):
    """Helper function for creating NaN results with the expected shape."""
    batch_size = qml.math.shape(state)[0] if is_state_batched else None
    interface = qml.math.get_interface(state)

    if circuit.shots.has_partitioned_shots:
        res = tuple(
            _get_single_nan_res(circuit.measurements, batch_size, interface, s)
            for s in circuit.shots
        )
    else:
        res = _get_single_nan_res(
            circuit.measurements, batch_size, interface, circuit.shots.total_shots
        )

    return res


def measure_final_state(circuit, state, is_state_batched, rng=None, prng_key=None) -> Result:
    """
    Perform the measurements required by the circuit on the provided state.

    This is an internal function that will be called by the successor to ``default.qubit``.

    Args:
        circuit (.QuantumScript): The single circuit to simulate
        state (TensorLike): The state to perform measurement on
        is_state_batched (bool): Whether the state has a batch dimension or not.
        rng (Union[None, int, array_like[int], SeedSequence, BitGenerator, Generator]): A
            seed-like parameter matching that of ``seed`` for ``numpy.random.default_rng``.
            If no value is provided, a default RNG will be used.
        prng_key (Optional[jax.random.PRNGKey]): An optional ``jax.random.PRNGKey``. This is
            the key to the JAX pseudo random number generator. Only for simulation using JAX.
            If None, the default ``sample_state`` function and a ``numpy.random.default_rng``
            will be for sampling.

    Returns:
        Tuple[TensorLike]: The measurement results
    """

    circuit = circuit.map_to_standard_wires()

    if qml.math.any(qml.math.isnan(state)):
        return _measure_nan_state(circuit, state, is_state_batched)

    if not circuit.shots:
        # analytic case

        if len(circuit.measurements) == 1:
            return measure(circuit.measurements[0], state, is_state_batched=is_state_batched)

        return tuple(
            measure(mp, state, is_state_batched=is_state_batched) for mp in circuit.measurements
        )

    # finite-shot case

    rng = default_rng(rng)
    results = measure_with_samples(
        circuit.measurements,
        state,
        shots=circuit.shots,
        is_state_batched=is_state_batched,
        rng=rng,
        prng_key=prng_key,
    )

    if len(circuit.measurements) == 1:
        if circuit.shots.has_partitioned_shots:
            return tuple(res[0] for res in results)

        return results[0]

    return results


def simulate(
    circuit: qml.tape.QuantumScript, rng=None, prng_key=None, debugger=None, interface=None
) -> Result:
    """Simulate a single quantum script.

    This is an internal function that will be called by the successor to ``default.qubit``.

    Args:
        circuit (QuantumTape): The single circuit to simulate
        rng (Union[None, int, array_like[int], SeedSequence, BitGenerator, Generator]): A
            seed-like parameter matching that of ``seed`` for ``numpy.random.default_rng``.
            If no value is provided, a default RNG will be used.
        prng_key (Optional[jax.random.PRNGKey]): An optional ``jax.random.PRNGKey``. This is
            the key to the JAX pseudo random number generator. If None, a random key will be
            generated. Only for simulation using JAX.
        debugger (_Debugger): The debugger to use
        interface (str): The machine learning interface to create the initial state with

    Returns:
        tuple(TensorLike): The results of the simulation

    Note that this function can return measurements for non-commuting observables simultaneously.

    This function assumes that all operations provide matrices.

    >>> qs = qml.tape.QuantumScript([qml.RX(1.2, wires=0)], [qml.expval(qml.PauliZ(0)), qml.probs(wires=(0,1))])
    >>> simulate(qs)
    (0.36235775447667357,
    tensor([0.68117888, 0.        , 0.31882112, 0.        ], requires_grad=True))

    """
    state, is_state_batched = get_final_state(circuit, debugger=debugger, interface=interface)
    return measure_final_state(circuit, state, is_state_batched, rng=rng, prng_key=prng_key)
