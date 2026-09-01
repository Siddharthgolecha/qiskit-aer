# This code is part of Qiskit.
#
# (C) Copyright IBM 2018, 2024.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.
"""
Integration tests for mid-circuit measurement (MCM) support.

These tests validate that a measurement appearing before the last operation
in a circuit simulates correctly, that results are available for classical
feedback via if_test, and that existing final-only-measurement optimisations
are not regressed.
"""

import unittest

from ddt import ddt, data

from qiskit.circuit import QuantumCircuit, QuantumRegister, ClassicalRegister, Measure
from qiskit_aer import AerSimulator
from qiskit_aer.noise import NoiseModel
from qiskit_aer.noise.errors import ReadoutError, pauli_error
from qiskit_aer.primitives import SamplerV2
from test.terra.backends.simulator_test_case import SimulatorTestCase, supported_methods

# Methods that support mid-circuit measurement and control-flow.
SUPPORTED_METHODS = [
    "statevector",
    "density_matrix",
    "matrix_product_state",
]

# A tighter subset for tests that use if_test (stabilizer does not support
# arbitrary controlled gates inside if_else bodies in all cases).
FEEDBACK_METHODS = [
    "statevector",
    "density_matrix",
    "matrix_product_state",
]


@ddt
class TestMidCircuitMeasurement(SimulatorTestCase):
    """Tests for mid-circuit measurement as a first-class feature."""

    # ------------------------------------------------------------------
    # Test 1: Basic mid-circuit measure — no classical feedback
    # ------------------------------------------------------------------
    @supported_methods(SUPPORTED_METHODS)
    def test_basic_mid_circuit_measure(self, method, device):
        """Measure qubit 0 mid-circuit; unconditional X on qubit 1; final measure_all.

        Expected: qubit 1 is always |1⟩ (flipped by X), qubit 0 is 0 or 1
        with approximately equal probability (H before MCM).
        """
        shots = 2000
        qreg = QuantumRegister(2, "q")
        creg = ClassicalRegister(2, "c")
        circ = QuantumCircuit(qreg, creg)

        circ.h(0)
        # Mid-circuit measurement into creg[0] — no downstream conditional
        circ.measure(0, 0)
        # Unconditional operation on qubit 1 after the MCM
        circ.x(1)
        # Final measurements
        circ.measure(0, 0)
        circ.measure(1, 1)

        backend = self.backend(method=method, device=device)
        result = backend.run(circ, shots=shots).result()
        self.assertSuccess(result)

        counts = result.get_counts()
        # qubit 1 (creg[1]) must always be 1 because of the unconditional X.
        # Counts keys are "c1 c0" ordered MSB→LSB: "10" or "11"
        for bitstring in counts:
            self.assertEqual(
                bitstring[0],
                "1",
                msg=f"qubit 1 should always be 1 after X gate, got '{bitstring}'",
            )
        # Both outcomes for qubit 0 should be present (H gate → ~50/50)
        self.assertGreater(len(counts), 1, msg="Expected both |10⟩ and |11⟩ outcomes")

    # ------------------------------------------------------------------
    # Test 2: Classical feedback — if_test conditioned on MCM result
    # ------------------------------------------------------------------
    @supported_methods(FEEDBACK_METHODS)
    def test_mid_circuit_measure_classical_feedback(self, method, device):
        """Measure qubit 0 mid-circuit; use if_test to flip qubit 1 iff result=1.

        Final state: qubit 1 == qubit 0 in every shot (qubit 0 teleported to qubit 1).
        """
        shots = 2000
        qreg = QuantumRegister(2, "q")
        creg = ClassicalRegister(1, "c")
        meas = ClassicalRegister(2, "m")
        circ = QuantumCircuit(qreg, creg, meas)

        circ.h(0)
        circ.measure(0, creg[0])  # MCM → creg[0]
        with circ.if_test((creg, 1)):
            circ.x(1)
        circ.measure(0, meas[0])
        circ.measure(1, meas[1])

        backend = self.backend(method=method, device=device)
        result = backend.run(circ, shots=shots).result()
        self.assertSuccess(result)

        counts = result.get_counts()
        # creg is the first register; meas is the second.
        # get_counts returns "meas[1] meas[0] creg[0]" (registers ordered MSB→LSB).
        # qubit 1 (meas[1]) should equal qubit 0 (meas[0]) in every shot.
        for bitstring, count in counts.items():
            # Format: "meas1 meas0 creg0" with a space between registers
            parts = bitstring.split()
            # parts[0] = meas register ("meas1 meas0"), parts[1] = creg ("creg0")
            meas_bits = parts[0]  # e.g. "00" or "11"
            self.assertEqual(
                meas_bits[0],
                meas_bits[1],
                msg=(
                    f"qubit 1 should equal qubit 0 after conditional X; "
                    f"got bitstring '{bitstring}' ({count} shots)"
                ),
            )

    # ------------------------------------------------------------------
    # Test 3: MCM then reset — post-reset state is fresh |0⟩
    # ------------------------------------------------------------------
    @supported_methods(FEEDBACK_METHODS)
    def test_mid_circuit_measure_then_reset(self, method, device):
        """Put qubit 0 in |1⟩, MCM (always reads 1), reset, apply H, final measure.

        After reset the qubit is in |0⟩ regardless of the MCM result, so H
        produces a 50/50 distribution on the final measurement.
        """
        shots = 2000
        qreg = QuantumRegister(1, "q")
        creg = ClassicalRegister(1, "c")
        circ = QuantumCircuit(qreg, creg)

        circ.x(0)           # |0⟩ → |1⟩
        circ.measure(0, 0)  # MCM — should always read 1
        circ.reset(0)       # back to |0⟩
        circ.h(0)           # equal superposition
        circ.measure(0, 0)  # final — should be 50/50

        backend = self.backend(method=method, device=device)
        result = backend.run(circ, shots=shots).result()
        self.assertSuccess(result)

        counts = result.get_counts()
        # Both "0" and "1" must appear after the H + measure sequence.
        self.assertIn("0", counts, msg="Expected |0⟩ outcome after reset+H")
        self.assertIn("1", counts, msg="Expected |1⟩ outcome after reset+H")
        # Rough balance check: neither outcome should be less than 30% of shots.
        for outcome, cnt in counts.items():
            self.assertGreater(
                cnt,
                0.30 * shots,
                msg=f"Outcome '{outcome}' count {cnt} is unexpectedly low for a 50/50 distribution",
            )

    # ------------------------------------------------------------------
    # Test 4: Deterministic MCM + conditional X → final always |0⟩
    # ------------------------------------------------------------------
    @supported_methods(FEEDBACK_METHODS)
    def test_mid_circuit_measure_deterministic(self, method, device):
        """Put qubit 0 in |1⟩, MCM reads 1, conditional X undoes it.

        Final measurement must always yield 0.
        """
        shots = 200
        qreg = QuantumRegister(1, "q")
        creg = ClassicalRegister(1, "c")
        circ = QuantumCircuit(qreg, creg)

        circ.x(0)           # prepare |1⟩
        circ.measure(0, 0)  # MCM — deterministically 1
        with circ.if_test((creg, 1)):
            circ.x(0)       # undo the |1⟩ → back to |0⟩
        circ.measure(0, 0)  # final — always 0

        backend = self.backend(method=method, device=device)
        result = backend.run(circ, shots=shots).result()
        self.assertSuccess(result)

        counts = result.get_counts()
        self.assertEqual(
            counts,
            {"0": shots},
            msg=f"All shots should yield '0' after deterministic MCM+correction; got {counts}",
        )

    # ------------------------------------------------------------------
    # Test 5: Final-only measurement still uses can_sample optimisation
    # ------------------------------------------------------------------
    @data("statevector", "density_matrix")
    def test_mid_circuit_measure_does_not_break_final_sampling(self, method):
        """A circuit with only final measurements must still set measure_sampling=True.

        This regression test confirms that the is_conditional scoping change in
        the compiler does not accidentally disable the can_sample optimisation
        for circuits that have no MCM at all.
        """
        shots = 500
        circ = QuantumCircuit(2, 2)
        circ.h(0)
        circ.cx(0, 1)
        circ.measure([0, 1], [0, 1])

        backend = AerSimulator(method=method, seed_simulator=42)
        result = backend.run(circ, shots=shots).result()
        self.assertSuccess(result)

        # The measure_sampling metadata key should be present and True, indicating
        # the fast sampling path was taken.
        for res in result.results:
            self.assertIn(
                "measure_sampling",
                res.metadata,
                msg="measure_sampling key missing from metadata",
            )
            self.assertTrue(
                res.metadata["measure_sampling"],
                msg="measure_sampling should be True for a final-only measurement circuit",
            )

    # ------------------------------------------------------------------
    # Test 6: MCM circuit with readout noise — simulation must complete
    # ------------------------------------------------------------------
    @supported_methods(FEEDBACK_METHODS)
    def test_mid_circuit_measure_with_readout_noise(self, method, device):
        """MCM circuit with a ReadoutError on the noise model must not crash.

        We use a lenient delta to accommodate the noise; the goal is to confirm
        that the simulator accepts the circuit and produces a result object with
        the success flag set, rather than checking precise counts.
        """
        shots = 1000
        # Modest readout error: 5% on |0⟩, 5% on |1⟩
        readout_error = ReadoutError([[0.95, 0.05], [0.05, 0.95]])
        noise_model = NoiseModel()
        noise_model.add_all_qubit_readout_error(readout_error)

        qreg = QuantumRegister(1, "q")
        creg = ClassicalRegister(1, "c")
        circ = QuantumCircuit(qreg, creg)
        circ.x(0)
        circ.measure(0, 0)  # MCM
        with circ.if_test((creg, 1)):
            circ.x(0)       # conditional correction
        circ.measure(0, 0)  # final

        backend = self.backend(method=method, device=device, noise_model=noise_model)
        result = backend.run(circ, shots=shots).result()
        self.assertSuccess(result)

        counts = result.get_counts()
        # Without noise all shots would be "0"; with readout noise some "1"s appear.
        # Just confirm the dominant outcome is "0" (>70% given 5% error).
        zero_fraction = counts.get("0", 0) / shots
        self.assertGreater(
            zero_fraction,
            0.70,
            msg=f"Expected '0' to dominate with 5% readout noise; got counts={counts}",
        )

    # ------------------------------------------------------------------
    # Test 7: SamplerV2 end-to-end — deterministic MCM + conditional X
    # ------------------------------------------------------------------
    def test_mid_circuit_measure_sampler_v2(self):
        """Deterministic MCM circuit through SamplerV2 must return all shots at '0'.

        This exercises the full primitive path and confirms that SamplerV2
        correctly handles circuits where a measure op precedes further gate ops.
        """
        shots = 200
        qreg = QuantumRegister(1, "q")
        creg = ClassicalRegister(1, "c")
        circ = QuantumCircuit(qreg, creg)

        circ.x(0)           # prepare |1⟩
        circ.measure(0, 0)  # MCM — always 1
        with circ.if_test((creg, 1)):
            circ.x(0)       # correct back to |0⟩
        circ.measure(0, 0)  # final — always 0

        sampler = SamplerV2(default_shots=shots, seed=42)
        job = sampler.run([circ])
        result = job.result()

        pub_result = result[0]
        # The DataBin has one entry per classical register.
        # creg name is "c"; bit values are packed into a BitArray.
        bit_array = pub_result.data.c
        # get_counts() returns a dict of bitstring → count
        counts = bit_array.get_counts()
        self.assertEqual(
            counts,
            {"0": shots},
            msg=f"SamplerV2: all shots should yield '0' after MCM+correction; got {counts}",
        )


    # ------------------------------------------------------------------
    # Test 8: Labeled MCM error shifts only its target (per-measurement noise)
    # ------------------------------------------------------------------
    @supported_methods(SUPPORTED_METHODS)
    def test_labeled_mcm_error_targets_only_labeled_measurement(self, method, device):
        """A QuantumError on "measure_flip" applies only before the labeled MCM.

        q0: Measure(label="flip") followed by x(0) — forces sequential MCM path.
            With an X error before the measure, |0⟩ is flipped to |1⟩ before
            measurement, so c0 always reads 1.
        q1: unlabeled measure — no error, always reads 0 (qubit stays |0⟩).

        Expected counts with noise: {"01": shots}  (c1c0, MSB first → "01")
        Expected counts without noise: {"00": shots}
        """
        shots = 200

        def make_circuit():
            qreg = QuantumRegister(2, "q")
            creg = ClassicalRegister(2, "c")
            circ = QuantumCircuit(qreg, creg)
            circ.append(Measure(label="flip"), [0], [0])  # labeled MCM on q0
            circ.x(0)                                      # same-qubit op → sequential path
            circ.measure(1, 1)                             # unlabeled final measure on q1
            return circ

        # ---- with noise ----
        nm = NoiseModel()
        nm.add_quantum_error(pauli_error([("X", 1.0)]), "measure_flip", [0])
        circ = make_circuit()
        backend = self.backend(method=method, device=device, noise_model=nm)
        result = backend.run(circ, shots=shots).result()
        self.assertSuccess(result)
        counts = result.get_counts()
        self.assertEqual(
            counts,
            {"01": shots},
            msg=f"With X-before-measure on q0, expected {{'01': {shots}}}; got {counts}",
        )

        # ---- without noise ----
        circ2 = make_circuit()
        backend2 = self.backend(method=method, device=device)
        result2 = backend2.run(circ2, shots=shots).result()
        self.assertSuccess(result2)
        counts2 = result2.get_counts()
        self.assertEqual(
            counts2,
            {"00": shots},
            msg=f"Without noise, expected {{'00': {shots}}}; got {counts2}",
        )

    # ------------------------------------------------------------------
    # Test 9: Labeled FINAL measure — error applied on fast sampling path
    # ------------------------------------------------------------------
    @data("density_matrix")
    def test_labeled_final_measure_error_on_sampling_path(self, method):
        """A QuantumError on a labeled FINAL measure fires even on the fast path.

        Single qubit in |0⟩, Measure(label="flip") is the only (final) measure.
        An X error before the measure flips |0⟩ → |1⟩, so the result is always "1".
        measure_sampling must still be True, proving the fast path was taken.
        """
        shots = 200
        qreg = QuantumRegister(1, "q")
        creg = ClassicalRegister(1, "c")
        circ = QuantumCircuit(qreg, creg)
        circ.append(Measure(label="flip"), [0], [0])

        nm = NoiseModel()
        nm.add_all_qubit_quantum_error(pauli_error([("X", 1.0)]), "measure_flip")

        backend = AerSimulator(method=method, seed_simulator=42, noise_model=nm)
        result = backend.run(circ, shots=shots).result()
        self.assertSuccess(result)

        counts = result.get_counts()
        self.assertEqual(
            counts,
            {"1": shots},
            msg=f"X-before-measure should always yield '1'; got {counts}",
        )
        self.assertTrue(
            result.results[0].metadata.get("measure_sampling"),
            msg="measure_sampling should be True for a single labeled final measure",
        )

    # ------------------------------------------------------------------
    # Test 10: Two labels route to two distinct errors independently
    # ------------------------------------------------------------------
    @supported_methods(SUPPORTED_METHODS)
    def test_two_labels_route_to_independent_errors(self, method, device):
        """Two labeled measurements receive independent per-label QuantumErrors.

        q0: Measure(label="flipA") with an X error → always reads 1 (|0⟩→|1⟩).
        q1: Measure(label="flipB") with a Z error → no-op on |0⟩, still reads 0.

        Expected counts: {"01": shots}  (c1=0, c0=1 → "01" MSB-first)
        """
        shots = 200
        qreg = QuantumRegister(2, "q")
        creg = ClassicalRegister(2, "c")
        circ = QuantumCircuit(qreg, creg)
        circ.append(Measure(label="flipA"), [0], [0])
        circ.append(Measure(label="flipB"), [1], [1])

        nm = NoiseModel()
        nm.add_all_qubit_quantum_error(pauli_error([("X", 1.0)]), "measure_flipA")
        nm.add_all_qubit_quantum_error(pauli_error([("Z", 1.0)]), "measure_flipB")

        backend = self.backend(method=method, device=device, noise_model=nm)
        result = backend.run(circ, shots=shots).result()
        self.assertSuccess(result)

        counts = result.get_counts()
        self.assertEqual(
            counts,
            {"01": shots},
            msg=(
                f"X on q0 should give c0=1, Z on q1 should be no-op giving c1=0; "
                f"expected {{'01': {shots}}}, got {counts}"
            ),
        )


if __name__ == "__main__":
    unittest.main()
