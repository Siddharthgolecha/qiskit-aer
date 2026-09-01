# Mid-Circuit Measurement — Implementation Plan

## Top-Level Overview

**Goal:** Complete mid-circuit measurement (MCM) support in qiskit-aer so that a Qiskit
`QuantumCircuit` containing one or more `measure` instructions before the final operation
layer simulates correctly, with measurement results available for classical feedback (conditional
gates via `if_test` / classical expressions), and with comprehensive regression tests ensuring
no existing behaviour is broken.

**What "mid-circuit measurement" means here:**
A measurement that occurs before the last operation in a circuit, whose result is stored in a
classical register and may be used to condition subsequent quantum gates (classical feedback).
This is distinct from (but enabled by the same infrastructure as) control-flow circuits —
it is the fundamental primitive that makes `if_test`, `while_loop`, etc. meaningful.

**What is already done (branch state):**
- `OpType::measure` already stores results to both `memory` and `registers` fields.
- `Circuit::can_sample` is correctly set to `false` whenever a conditional or mid-circuit
  measure is detected, so ops are executed sequentially in the correct order.
- `AerCompiler._assemble_op` already calls `aer_circ.measure(qubits, clbits, clbits, name)`
  when `is_conditional=True`, routing measurement results into the register file for classical
  feedback.
- Classical expression infrastructure (`CExpr`, `VarExpr`, `BinaryExpr`, etc.) is fully bound
  via pybind11 and used by `aer_compiler.py` to translate Qiskit 2.0-style `condition_expr`.
- Shot-branching infrastructure (`shot_branching.hpp`, `multi_state_executor.hpp`) exists and
  works for `statevector` and `density_matrix` methods.
- Control-flow tests (`test_control_flow.py`) cover `if_test`, `if_else`, `for_loop`,
  `while_loop`, `switch_case` — all of which imply mid-circuit measurements.

**What is still missing:**
1. `is_conditional` detection in the compiler is circuit-wide but not precisely scoped —
   it forces *all* measures to write registers even when no conditional gate follows.
   This makes register allocation unnecessarily broad and can misfire.
2. The `SamplerV2` primitive does not have an explicit correctness check for mid-circuit
   measurement circuits; since it routes through `AerSimulator` directly it is likely
   fine but needs a targeted integration test to confirm.
3. There is no dedicated test file for mid-circuit measurement as a first-class feature.
   Existing tests cover it only indirectly through control-flow and shot-branching tests.
4. *(Lower priority — deferred)* `aerbackend.py` does not expose a `dynamic_reprate_enabled`
   capability flag on its `Target`; this is a transpiler-level concern and does not block
   simulation correctness.
5. *(Deferred to follow-up)* Noise model has no explicit hook for mid-circuit measurement
   noise applied at the intermediate point; readout error handling needs a separate audit.

**Scope of this plan (three active sub-tasks):**
- **Sub-Task 1** — Fix `is_conditional` scoping in the compiler (correctness).
- **Sub-Task 2** — Verify/fix `SamplerV2` end-to-end path for MCM.
- **Sub-Task 3** — Comprehensive MCM test suite.

Sub-Tasks 4 and 5 are explicitly out of scope for this iteration.

---

## Sub-Tasks

---

### Sub-Task 1 — Fix `is_conditional` scoping in `aer_compiler.py`

**Status:** `[x] done`

**Intent:**
Currently `is_conditional` is set to `True` if *any* instruction in the circuit has a
`condition` or `condition_expr`.  When `True`, *every* `measure` call in
`_assemble_op` writes `registers=clbits` even for instructions that have no downstream
conditional gate.  This wastes register slots and inflates `num_registers`.  The correct
behaviour is: a measurement should write to registers only when at least one subsequent
gate is conditioned on those classical bits.

**Expected Outcomes:**
- Circuits that have both conditioned and unconditioned measurements compile correctly.
- `num_registers` for circuits without conditionals is 0 (current behaviour preserved).
- Circuits with mid-circuit measurements + downstream `if_test` blocks simulate correctly
  and produce the right counts.

**Todo List:**
1. In [`qiskit_aer/backends/aer_compiler.py`](qiskit_aer/backends/aer_compiler.py) inside
   `_assemble_circuit`, change `is_conditional` from a single boolean to a set of classical
   bit indices that are actually read by a conditional gate somewhere in the flattened circuit.
2. In `_assemble_op`, when handling `"measure"`, pass `registers=clbits` only if any of the
   target `clbits` appear in that set; otherwise pass `registers=[]`.
3. Add a helper function `_conditional_clbits(circuit)` that returns the set of classical
   bit indices that feed at least one conditional operation in the circuit data (including
   inside inlined control-flow blocks).
4. Ensure backward compatibility: the existing `bfunc`-based path for old-style `c_if`
   uses separate register slots and is unaffected by this change.

**Relevant Context:**
- [`qiskit_aer/backends/aer_compiler.py:693`](qiskit_aer/backends/aer_compiler.py:693) —
  current `is_conditional` computation.
- [`qiskit_aer/backends/aer_compiler.py:940`](qiskit_aer/backends/aer_compiler.py:940) —
  where `is_conditional` gates `registers` argument of `measure`.
- [`qiskit_aer/backends/wrappers/aer_circuit_binding.hpp:193`](qiskit_aer/backends/wrappers/aer_circuit_binding.hpp:193) —
  binding signature for `Circuit::measure(qubits, memory, registers, name)`.

---

### Sub-Task 2 — Verify and fix `SamplerV2` primitive for mid-circuit measurements

**Status:** `[x] done — no code change required`

**Finding:** `SamplerV2._run_pubs` routes directly through `AerSimulator` with `memory=True`
and reads raw bit-strings from the result.  `_analyze_circuit` only inspects
`circuit.cregs` (not measurement ops), so it is MCM-safe.  No pre-processing
step is broken for mid-circuit measurement circuits.

**Intent:**
`SamplerV2` routes directly through `AerSimulator` and reads classical memory from the
backend result, so it is architecturally compatible with MCM.  However, no integration
test currently confirms this path works end-to-end for circuits that have measurements
before the final layer.  This sub-task locates the `SamplerV2` implementation, audits
whether any pre-processing step (e.g. circuit key caching, final-measurement extraction)
silently breaks for MCM circuits, and adds any small fixes required.

**Expected Outcomes:**
- Running a mid-circuit measurement circuit through `SamplerV2` returns the correct
  quasi-probability distribution with classical register values populated from the
  final measurement layer only.
- If `SamplerV2` performs any `final_measurement_mapping`-style pre-processing, that
  logic is guarded against MCM circuits.
- Circuits without mid-circuit measurements are unaffected.

**Todo List:**
1. Locate the `SamplerV2` implementation file(s) in `qiskit_aer/primitives/`.
2. Audit any circuit pre-processing that inspects instruction order or extracts
   measurement mappings — check it is safe for circuits where a `measure` op appears
   before a non-trivial quantum op on the same qubit.
3. If a pre-processing step is unsafe, add a guard that bypasses it when mid-circuit
   measurements are detected (check for any `measure` whose qubit still has subsequent
   non-barrier ops).
4. Confirm the result object's `memory` array is correctly populated for MCM circuits
   and that the quasi-distribution reflects only the *final* measurement outcomes.

**Relevant Context:**
- `SamplerV2` lives in `qiskit_aer/primitives/` — locate with `glob("**/sampler*.py")`.
- The deprecated `Sampler` (`sampler.py`) has the problematic `final_measurement_mapping`
  at line 52; check whether `SamplerV2` reuses this helper.
- `AerSimulator` result memory layout: one bit-string per shot, ordered by classical
  register index.

---

### Sub-Task 3 — Write mid-circuit measurement test suite

**Status:** `[x] done`

**File created:** `test/terra/backends/aer_simulator/test_mid_circuit_measurement.py`

**Intent:**
Create a dedicated test file that validates mid-circuit measurement as a first-class feature
across the supported simulation methods.  Tests are deliberately independent of the
control-flow and shot-branching test files so that MCM regressions surface clearly.
Coverage: basic MCM counts, classical feedback via `if_test`, MCM + reset, deterministic
MCM, regression for `can_sample=True` path, readout noise with MCM, and a `SamplerV2`
end-to-end test (which also validates Sub-Task 2 findings).

**Expected Outcomes:**
- New file `test/terra/backends/aer_simulator/test_mid_circuit_measurement.py` exists and
  all tests pass.
- Tests run on `statevector`, `density_matrix`, and `matrix_product_state` methods.
- No existing test in any other test file is broken by changes from Sub-Tasks 1 and 2.

**Todo List:**
1. Create `test/terra/backends/aer_simulator/test_mid_circuit_measurement.py` following
   the structure of [`test/terra/backends/aer_simulator/test_control_flow.py`](test/terra/backends/aer_simulator/test_control_flow.py).
2. Add `test_basic_mid_circuit_measure` — apply H on qubit 0, measure qubit 0 to clbit 0,
   apply unconditional X on qubit 1, measure all at end.  Verify counts always have
   qubit 1 = 1 and qubit 0 = 0 or 1 with equal probability.
3. Add `test_mid_circuit_measure_classical_feedback` — measure qubit 0 mid-circuit, use
   `if_test((creg, 1))` to conditionally flip qubit 1.  Verify that qubit 1 mirrors
   qubit 0 in the final counts across `statevector`, `density_matrix`,
   `matrix_product_state`.
4. Add `test_mid_circuit_measure_then_reset` — put qubit 0 in |1⟩, measure mid-circuit
   (should always read 1), reset qubit 0, apply H, final measure.  Distribution on
   qubit 0 should be 50/50 after reset.
5. Add `test_mid_circuit_measure_deterministic` — put qubit 0 in |1⟩, measure mid-circuit
   into creg, use `if_test((creg, 1))` to apply X on qubit 0, final measure.  All shots
   should yield qubit 0 = 0 (X undid the |1⟩).
6. Add `test_mid_circuit_measure_does_not_break_final_sampling` — a circuit with only
   final measurements must still have `measure_sampling=True` in result metadata,
   confirming the `can_sample` optimisation path is unaffected by Sub-Task 1 changes.
7. Add `test_mid_circuit_measure_with_readout_noise` — attach a `ReadoutError` to the
   noise model and confirm the simulation completes and produces counts within a
   statistical delta (verifies noise does not crash on MCM circuits).
8. Add `test_mid_circuit_measure_sampler_v2` — build the same deterministic MCM circuit
   from step 5 and run it via `SamplerV2`; verify the returned `BitArray` / quasi-
   distribution has all shots at the expected bitstring.

**Relevant Context:**
- Structure pattern: [`test/terra/backends/aer_simulator/test_control_flow.py`](test/terra/backends/aer_simulator/test_control_flow.py).
- Base class: [`test/terra/backends/simulator_test_case.py`](test/terra/backends/simulator_test_case.py).
- `SamplerV2` lives in `qiskit_aer/primitives/` (locate exact file via glob).
- `ref_measure` / `ref_reset` helpers: [`test/terra/reference/`](test/terra/reference/).

---

### Sub-Task 4 (Deferred) — Expose `dynamic_reprate_enabled` capability flag

**Status:** `[ ] pending — lower priority, not in scope for this iteration`

Backend transpiler flag work. See original plan notes above.

---

### Sub-Task 5 (Deferred) — Mid-circuit readout noise audit

**Status:** `[ ] pending — follow-up, not in scope for this iteration`

Noise model insertion-point audit for mid-circuit `ReadoutError`. See original plan notes above.
