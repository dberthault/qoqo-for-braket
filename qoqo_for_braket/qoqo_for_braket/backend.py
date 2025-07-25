# Copyright © 2023 HQS Quantum Simulations GmbH.
#
# Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except
# in compliance with the License. You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software distributed under the License
# is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express
# or implied. See the License for the specific language governing permissions and limitations under
# the License.

"""Provides the BraketBackend class."""

import json
import os
import shutil
import tempfile
import numpy as np
import qoqo_qasm

from typing import Any, Dict, List, Optional, Tuple, Union, cast
from braket.aws import AwsDevice, AwsQuantumJob, AwsQuantumTask, AwsQuantumTaskBatch
from braket.aws.aws_session import AwsSession
from braket.circuits import Circuit as BraketCircuit
from braket.devices import LocalSimulator
from braket.ir import openqasm
from braket.jobs.local import LocalQuantumJob
from qoqo import Circuit, QuantumProgram
from qoqo import operations as ops  # type: ignore
from qoqo.measurements import ClassicalRegister  # type:ignore
from qiskit.providers import BackendV2
from qiskit.transpiler import CouplingMap, Target, InstructionProperties, Layout, PassManager
from qiskit import QuantumCircuit, transpile, QuantumRegister, ClassicalRegister
from qiskit.transpiler.passes import (
    VF2Layout,  # noise-aware perfect-layout search
    SabreSwap,  # heuristic router that inserts SWAPs
    FullAncillaAllocation,
    ApplyLayout,
)
from qiskit.circuit.library import (
    IGate,
    RGate,
    RXGate,
    RYGate,
    RZGate,
    SXGate,
    XGate,
    CXGate,
    CZGate,
    Measure,
    HGate,
    YGate,
    ZGate,
    TGate,
    TdgGate,
    SGate,
    SdgGate,
    SwapGate,
)
from qiskit.providers.options import Options
from qiskit.qasm2 import dumps

from qoqo_for_braket_devices.devices import StandardizedDevice  # type: ignore
from qoqo_for_braket.interface import (
    ionq_verbatim_interface,
    iqm_verbatim_interface,
    oqc_verbatim_interface,
    rigetti_verbatim_interface,
)
from qoqo_for_braket.post_processing import _post_process_circuit_result
from qoqo_for_braket.queued_results import QueuedCircuitRun, QueuedHybridRun, QueuedProgramRun

LOCAL_SIMULATORS_LIST: List[str] = ["braket_sv", "braket_dm", "braket_ahs"]
REMOTE_SIMULATORS_LIST: List[str] = [
    "arn:aws:braket:::device/quantum-simulator/amazon/sv1",
    "arn:aws:braket:::device/quantum-simulator/amazon/tn1",
    "arn:aws:braket:::device/quantum-simulator/amazon/dm1",
]


def from_qoqo_to_qiskit(qoqo_circuit):
    num_qubits = qoqo_circuit.number_of_qubits()
    qreg = QuantumRegister(num_qubits, "q")
    cl_registers = []
    clreg_map = {}

    for definition in qoqo_circuit.definitions():
        if definition.hqslang() == "DefinitionBit":
            name = definition.name()
            creg = ClassicalRegister(definition.length(), name)
            cl_registers.append(creg)
            clreg_map[name] = creg

    qc = QuantumCircuit(qreg, *cl_registers)

    for op in qoqo_circuit.operations():
        if op.hqslang() == "RotateXY":
            theta, phi = float(op.theta()), float(op.phi())
            qubit = op.qubit()
            qc.r(theta, phi, qreg[qubit])
        elif op.hqslang() == "ControlledPauliZ":
            control, target = op.control(), op.target()
            qc.cz(qreg[control], qreg[target])
        elif op.hqslang() == "MeasureQubit":
            qubit = op.qubit()
            clreg_name = op.readout()
            cbit_idx = op.readout_index()
            qc.measure(qreg[qubit], clreg_map[clreg_name][cbit_idx])
    return qc


class BraketBackend:
    """Qoqo backend execution qoqo objects on AWS braket.

    Args:
        device: The AWS device the circuit should run on.
                provided as aws arn or for local devices
                starting with "local:" as for the braket QuantumJob
        aws_session: An optional braket AwsSession. If set to None
                     AwsSession will be created automatically.
        verbatim_mode: Only use native gates for real devices and block
                       recompilation by devices
        batch_mode: Run circuits in batch mode when running measurements.
                    Does not work when circuits define different numbers of shots.
    """

    def __init__(
        self,
        device: Optional[str] = None,
        aws_session: Optional[AwsSession] = None,
        verbatim_mode: bool = False,
        batch_mode: bool = False,
        use_hybrid_jobs: bool = True,
    ) -> None:
        """Initialise the BraketBackend class.

        Args:
            device: Optional ARN of the Braket device to use. If none is provided, the \
                    default LocalSimulator will be used.
            aws_session: Optional AwsSession to use. If none is provided, a new one will be created
            verbatim_mode: Whether to use verbatim boxes to avoid recompilation
            batch_mode: Run circuits in batch mode when running measurements. \
                    Does not work when circuits define different numbers of shots.
            use_hybrid_jobs: Uses hybrid jobs to run measurements and register measurements.

        """
        self.aws_session = aws_session
        self.device = "braket_sv" if device is None else device
        self.verbatim_mode = verbatim_mode

        self.__use_actual_hardware = False
        self.__force_rigetti_verbatim = False
        self.__force_ionq_verbatim = False
        self.__force_iqm_verbatim = False
        self.__force_oqc_verbatim = False
        self.__max_circuit_length = 100
        self.__max_number_shots = 100
        self.batch_mode = batch_mode
        self.use_hybrid_jobs = use_hybrid_jobs

    def _create_config(self) -> Dict[str, Any]:
        return {
            "device": self.device,
            "verbatim_mode": self.verbatim_mode,
            "max_shots": self.__max_number_shots,
            "max_circuit_length": self.__max_circuit_length,
            "use_actual_hardware": self.__use_actual_hardware,
            "force_rigetti_verbatim": self.__force_rigetti_verbatim,
            "force_oqc_verbatim": self.__force_oqc_verbatim,
            "force_ionq_verbatim": self.__force_ionq_verbatim,
            "force_iqm_verbatim": self.__force_iqm_verbatim,
            "batch_mode": self.batch_mode,
        }

    def _load_config(self, config: Dict[str, Any]) -> None:
        self.device = config["device"]
        self.verbatim_mode = config["verbatim_mode"]
        self.__max_number_shots = config["max_shots"]
        self.__max_circuit_length = config["max_circuit_length"]
        self.__use_actual_hardware = config["use_actual_hardware"]
        self.__force_rigetti_verbatim = config["force_rigetti_verbatim"]
        self.__force_oqc_verbatim = config["force_oqc_verbatim"]
        self.__force_ionq_verbatim = config["force_ionq_verbatim"]
        self.__force_iqm_verbatim = config["force_iqm_verbatim"]
        self.batch_mode = config["batch_mode"]

    def allow_use_actual_hardware(self) -> None:
        """Allow the use of actual hardware - will cost money."""
        self.__use_actual_hardware = True

    def disallow_use_actual_hardware(self) -> None:
        """Disallow the use of actual hardware."""
        self.__use_actual_hardware = False

    def force_rigetti_verbatim(self) -> None:
        """Force the use of rigetti verbatim. Mostly used for testing purposes."""
        self.__force_rigetti_verbatim = True
        self.verbatim_mode = True

    def force_ionq_verbatim(self) -> None:
        """Force the use of ionq verbatim. Mostly used for testing purposes."""
        self.__force_ionq_verbatim = True
        self.verbatim_mode = True

    def force_iqm_verbatim(self) -> None:
        """Force the use of iqm verbatim. Mostly used for testing purposes."""
        self.__force_iqm_verbatim = True
        self.verbatim_mode = True

    def force_oqc_verbatim(self) -> None:
        """Force the use of oqc verbatim. Mostly used for testing purposes."""
        self.__force_oqc_verbatim = True
        self.verbatim_mode = True

    def change_max_shots(self, shots: int) -> None:
        """Change the maximum number of shots allowed.

        Args:
            shots: new maximum allowed number of shots
        """
        self.__max_number_shots = shots

    def change_max_circuit_length(self, length: int) -> None:
        """Change the maximum circuit length allowed.

        Args:
            length: new maximum allowed length of circuit
        """
        self.__max_circuit_length = length

    def __create_device(self) -> Union[LocalSimulator, AwsDevice]:
        """Creates the device and returns it.

        Returns:
            The instanciated device (either an AwsDevice or a LocalSimulator)

        Raises:
            ValueError: Device specified isn't allowed. You can allow it by calling the
                        `allow_use_actual_hardware` function, but please be aware that
                        this may incur significant monetary charges.
        """
        if self.device.startswith("local:") or self.device in LOCAL_SIMULATORS_LIST:
            device = LocalSimulator(self.device)
        elif self.device in REMOTE_SIMULATORS_LIST:
            device = AwsDevice(self.device)
        else:
            if self.__use_actual_hardware:
                # allow list simulator devices of AWS e.g. state vector simulator
                device = AwsDevice(self.device)
            else:
                raise ValueError(
                    "Device specified isn't allowed. You can allow it by calling the "
                    + "`allow_use_actual_hardware` function, but please be aware that "
                    + "this may incur significant monetary charges."
                )
        return device

    def _fetch_device_calibrations(self) -> Dict:
        """Fetches the device calibrations from the AWS backend.

        Returns:
            Dict: The device calibrations.
        """
        device = self.__create_device()
        if not isinstance(device, AwsDevice):
            raise ValueError(f"Device {device} does not support fetching calibrations.")

        props = device.properties.dict()

        num_qubits = props.get("paradigm").get("qubitCount")
        connectivity = props.get("paradigm", {}).get("connectivity", {})

        # Check if fully connected
        if connectivity.get("fullyConnected"):
            coupling_map = [[i, j] for i in range(num_qubits) for j in range(i + 1, num_qubits)]
        else:
            connectivity_graph = connectivity.get("connectivityGraph", {})
            coupling_map = []
            for qubit_str, neighbors in connectivity_graph.items():
                qubit = int(qubit_str) - 1
                for neighbor_str in neighbors:
                    neighbor = int(neighbor_str) - 1
                    coupling_map.append([qubit, neighbor])

        basis_gates = props.get("paradigm", {}).get("nativeGateSet", [])

        stand_dict = props.get("standardized")
        json_str = json.dumps(stand_dict)
        standardized_device = StandardizedDevice.from_json(json_str)

        return {
            "num_qubits": num_qubits,
            "noise_infos": standardized_device,
            "coupling_map": coupling_map,
            "basis_gates": basis_gates,
        }

    def remap_circuit(
        self,
        circuit: Circuit,
        initial_layout: Layout | None = None,
    ) -> Tuple[Circuit, Layout]:
        """Remap the circuit qubits for the device.

        Args:
            circuit (Circuit): The qoqo Circuit to remap.
            initial_layout (Layout): If set it will map the circuit to this layout
                instead of using the backend.

        Returns:
            Circuit: The remapped qoqo Circuit.
            Layout: The layout of the circuit after remapping
        """
        calibration_dict = self._fetch_device_calibrations()

        # Create the backend with basic configuration

        qasm_backend = qoqo_qasm.QasmBackend()
        qiskit_circuit = from_qoqo_to_qiskit(circuit)
        if initial_layout:
            remapped_circuit = transpile(
                qiskit_circuit,
                optimization_level=0,
                initial_layout=initial_layout,
            )
        else:
            qiskit_backend = QiskitBraketBackend(calibration_dict=calibration_dict)
            pm = PassManager(
                [
                    VF2Layout(
                        coupling_map=qiskit_backend.coupling_map,
                        target=qiskit_backend.target,
                    ),  # Adaptive, error-aware mapping
                    FullAncillaAllocation(qiskit_backend.target),
                    ApplyLayout(),
                    SabreSwap(qiskit_backend.coupling_map),  # Routing
                ]
            )
            remapped_circuit = pm.run(qiskit_circuit)
        final_layout = remapped_circuit.layout.final_virtual_layout(filter_ancillas=True)
        transpiled_qasm = dumps(remapped_circuit)

        remapped_qoqo_circuit = qasm_backend.qasm_str_to_circuit(transpiled_qasm)

        return remapped_qoqo_circuit, final_layout

    # runs a circuit internally and can be used to produce sync and async results
    def _run_circuit(
        self,
        circuit: Circuit,
    ) -> Tuple[AwsQuantumTask, Dict[str, Any], Circuit]:
        """Simulate a Circuit on a AWS backend.

        The default number of shots for the simulation is 100.
        Any kind of Measurement instruction only works as intended if
        it is the last instruction in the Circuit.
        Currently only one simulation is performed, meaning different measurements on different
        registers are not supported.

        Args:
            circuit (Circuit): the Circuit to simulate.

        Returns:
            (AwsQuantumTask, {readout, output_registers, output_lengths}, input_bit_circuit)

        Raises:
            ValueError: Circuit contains multiple ways to set the number of measurements
        """
        (
            output_bit_register_dict,
            output_float_register_dict,
            output_complex_register_dict,
            output_bit_register_lengths,
            output_float_register_lengths,
            output_complex_register_lengths,
        ) = self._set_up_registers(circuit)
        (task_specification, shots, readout, input_bit_circuit) = self._prepare_circuit_for_run(
            circuit
        )
        return (
            self.__create_device().run(task_specification, shots=shots),
            {
                "readout_name": readout,
                "output_registers": (
                    output_bit_register_dict,
                    output_float_register_dict,
                    output_complex_register_dict,
                ),
                "output_register_lengths": (
                    output_bit_register_lengths,
                    output_float_register_lengths,
                    output_complex_register_lengths,
                ),
            },
            input_bit_circuit,
        )

    # runs a circuit internally and can be used to produce sync and async results
    def _run_circuits_batch(
        self,
        circuits: List[Circuit],
    ) -> Tuple[AwsQuantumTaskBatch, List[Dict[str, Any]], Circuit]:
        """Run a list of Circuits on a AWS backend in batch mode.

        The default number of shots for the simulation is 100.
        Any kind of Measurement instruction only works as intended if
        it is the last instruction in the Circuit.
        Currently only one simulation is performed, meaning different measurements on different
        registers are not supported.

        Args:
            circuits (List[Circuit]): the Circuits to simulate.

        Returns:
            (AwsQuantumTaskBatch, {readout, output_registers}, input_bit_circuit)

        Raises:
            ValueError: Circuit contains multiple ways to set the number of measurements
        """
        task_specifications: List[BraketCircuit] = []
        shots_list = []
        metadata = []
        for circuit in circuits:
            (task_specification, shots, readout, input_bit_circuit) = (
                self._prepare_circuit_for_run(circuit)
            )
            (
                output_bit_register_dict,
                output_float_register_dict,
                output_complex_register_dict,
                output_bit_register_lengths,
                output_float_register_lengths,
                output_complex_register_lengths,
            ) = self._set_up_registers(circuit)
            task_specifications.append(task_specification)
            shots_list.append(shots)
            metadata.append(
                {
                    "readout_name": readout,
                    "output_registers": (
                        output_bit_register_dict,
                        output_float_register_dict,
                        output_complex_register_dict,
                    ),
                    "output_register_lengths": (
                        output_bit_register_lengths,
                        output_float_register_lengths,
                        output_complex_register_lengths,
                    ),
                }
            )
        unique_shots = np.unique(shots_list)
        if len(unique_shots) > 1:
            raise ValueError("Circuits contains multiple ways to set the number of measurements")
        else:
            shots = unique_shots[0]
        return (
            self.__create_device().run_batch(task_specifications, shots=int(shots)),
            metadata,
            input_bit_circuit,
        )

    def _prepare_circuit_for_run(
        self, circuit: Circuit
    ) -> Tuple[BraketCircuit, int, str, Circuit]:
        """Prepares a braket circuit for running on braket.

        Args:
            circuit (Circuit): The qoqo Circuit that should be run.

        Returns:
            (BraketCircuit, int, str, Circuit): The braket circuit, the number of shots,
            the readout and the InputBit circuit.
        """
        measurement_vector: List[ops.Operation] = [
            item
            for sublist in [
                circuit.filter_by_tag("PragmaSetNumberOfMeasurements"),
                circuit.filter_by_tag("PragmaRepeatedMeasurement"),
            ]
            for item in sublist
        ]
        measure_qubit_vector: List[ops.Operation] = circuit.filter_by_tag("MeasureQubit")
        if len(measurement_vector) > 1:
            raise ValueError("Circuit contains multiple ways to set the number of measurements")

        shots = measurement_vector[0].number_measurements() if measurement_vector else 100
        if measurement_vector:
            readout = measurement_vector[0].readout()
        elif measure_qubit_vector:
            readout = measure_qubit_vector[0].readout()
        else:
            readout = "ro"

        input_bit_circuit = Circuit()

        tmp_circuit = Circuit()
        for c in circuit:
            if c.hqslang() == "InputBit":
                input_bit_circuit += c
            else:
                tmp_circuit += c

        circuit = tmp_circuit

        if not self.verbatim_mode:
            qasm_backend = qoqo_qasm.QasmBackend("q", "3.0Braket")
            qasm_string = qasm_backend.circuit_to_qasm_str(circuit)
            task_specification = openqasm.Program(source=qasm_string)

        elif "Aspen" in self.device or self.__force_rigetti_verbatim:
            task_specification = rigetti_verbatim_interface.call_circuit(circuit)
        elif "ionq" in self.device or self.__force_ionq_verbatim:
            task_specification = ionq_verbatim_interface.call_circuit(circuit)
        elif "Garnet" in self.device or self.__force_iqm_verbatim:
            task_specification = iqm_verbatim_interface.call_circuit(circuit)

        elif "Lucy" in self.device or self.__force_oqc_verbatim:
            task_specification = oqc_verbatim_interface.call_circuit(circuit)

        if (
            self.__use_actual_hardware
            or self.__force_ionq_verbatim
            or self.__force_iqm_verbatim
            or self.__force_oqc_verbatim
            or self.__force_rigetti_verbatim
        ):
            if shots > self.__max_number_shots:
                raise ValueError(
                    "Number of shots specified exceeds the number of shots allowed for hardware"
                )
            if len(circuit) > self.__max_circuit_length:
                raise ValueError(
                    "Circuit generated is longer that the max circuit length allowed for hardware"
                )
        return (task_specification, shots, readout, input_bit_circuit)

    def _set_up_registers(self, circuit: Circuit) -> Tuple[
        Dict[str, List[List[bool]]],
        Dict[str, List[List[float]]],
        Dict[str, List[List[complex]]],
        Dict[str, int],
        Dict[str, int],
        Dict[str, int],
    ]:
        """Sets up the output registers for a circuit running on braket.

        Args:
            circuit (Circuit): The qoqo Circuit for which to prepare the registers.

        Returns:
            (Dict[str, List[List[bool]]], Dict[str, List[List[float]]],
             Dict[str, List[List[complex]]]): The output bit register, float
                                              register and complex register.
        """
        output_bit_register_dict: Dict[str, List[List[bool]]] = {}
        output_float_register_dict: Dict[str, List[List[float]]] = {}
        output_complex_register_dict: Dict[str, List[List[complex]]] = {}
        output_bit_register_lengths: Dict[str, int] = {}
        output_float_register_lengths: Dict[str, int] = {}
        output_complex_register_lengths: Dict[str, int] = {}

        for bit_def in circuit.filter_by_tag("DefinitionBit"):
            # if bit_def.is_output():
            output_bit_register_dict[bit_def.name()] = []
            output_bit_register_lengths[bit_def.name()] = bit_def.length()

        for float_def in circuit.filter_by_tag("DefinitionFloat"):
            # if float_def.is_output():
            output_float_register_dict[float_def.name()] = cast(List[List[float]], [])
            output_float_register_lengths[float_def.name()] = float_def.length()

        for complex_def in circuit.filter_by_tag("DefinitionComplex"):
            # if complex_def.is_output():
            output_complex_register_dict[complex_def.name()] = cast(List[List[complex]], [])
            output_complex_register_lengths[complex_def.name()] = complex_def.length()

        return (
            output_bit_register_dict,
            output_float_register_dict,
            output_complex_register_dict,
            output_bit_register_lengths,
            output_float_register_lengths,
            output_complex_register_lengths,
        )

    def run_circuit(
        self,
        circuit: Circuit,
    ) -> Tuple[
        Dict[str, List[List[bool]]],
        Dict[str, List[List[float]]],
        Dict[str, List[List[complex]]],
    ]:
        """Simulate a Circuit on a AWS backend.

        The default number of shots for the simulation is 100.
        Any kind of Measurement instruction only works as intended if
        it is the last instruction in the Circuit.
        Currently only one simulation is performed, meaning different measurements on different
        registers are not supported.

        Args:
            circuit (Circuit): the Circuit to simulate.

        Returns:
            Tuple[Dict[str, List[List[bool]]],
                  Dict[str, List[List[float]]],
                  Dict[str, List[List[complex]]]]: bit, float and complex registers dictionaries.
        """
        (quantum_task, metadata, input_bit_circuit) = self._run_circuit(circuit)
        results = quantum_task.result()
        (
            output_bit_register_dict,
            output_float_register_dict,
            output_complex_register_dict,
        ) = _post_process_circuit_result(results, metadata, input_bit_circuit)

        return (output_bit_register_dict, output_float_register_dict, output_complex_register_dict)

    def run_circuits_batch(self, circuits: List[Circuit]) -> Tuple[
        Dict[str, List[List[bool]]],
        Dict[str, List[List[float]]],
        Dict[str, List[List[complex]]],
    ]:
        """Run a list of Circuits on a AWS backend in batch mode.

        The default number of shots for the simulation is 100.
        Any kind of Measurement instruction only works as intended if
        it is the last instruction in the Circuit.
        Currently only one simulation is performed, meaning different measurements on different
        registers are not supported.

        Args:
            circuits (List[Circuit]): the Circuit to simulate.

        Returns:
            Tuple[Dict[str, List[List[bool]]],
                  Dict[str, List[List[float]]],
                  Dict[str, List[List[complex]]]]: bit, float and complex registers dictionaries.
        """
        (quantum_task_batch, batch_metadata, input_bit_circuit) = self._run_circuits_batch(
            circuits
        )
        bool_register_dict: Dict[str, List[List[bool]]] = {}
        float_register_dict: Dict[str, List[List[float]]] = {}
        complex_register_dict: Dict[str, List[List[complex]]] = {}
        for quantum_task, metadata in zip(quantum_task_batch.results(), batch_metadata):
            results = quantum_task
            (
                tmp_bool_register_dict,
                tmp_float_register_dict,
                tmp_complex_register_dict,
            ) = _post_process_circuit_result(results, metadata, input_bit_circuit)
            for key, value_bools in tmp_bool_register_dict.items():
                if key in bool_register_dict:
                    bool_register_dict[key].extend(value_bools)
                else:
                    bool_register_dict[key] = value_bools
            for key, value_floats in tmp_float_register_dict.items():
                if key in float_register_dict:
                    float_register_dict[key].extend(value_floats)
                else:
                    float_register_dict[key] = value_floats
            for key, value_complexes in tmp_complex_register_dict.items():
                if key in complex_register_dict:
                    complex_register_dict[key].extend(value_complexes)
                else:
                    complex_register_dict[key] = value_complexes
        return (bool_register_dict, float_register_dict, complex_register_dict)

    def run_measurement_registers(self, measurement: Any) -> Tuple[
        Dict[str, List[List[bool]]],
        Dict[str, List[List[float]]],
        Dict[str, List[List[complex]]],
    ]:
        """Run all circuits of a measurement with the AWS Braket backend.

        Args:
            measurement: The measurement that is run.

        Returns:
            Tuple[Dict[str, List[List[bool]]],
                  Dict[str, List[List[float]]],
                  Dict[str, List[List[complex]]]]
        """
        constant_circuit = measurement.constant_circuit()
        output_bit_register_dict: Dict[str, List[List[bool]]] = {}
        output_float_register_dict: Dict[str, List[List[float]]] = {}
        output_complex_register_dict: Dict[str, List[List[complex]]] = {}
        run_circuits = []
        for circuit in measurement.circuits():
            if constant_circuit is None:
                run_circuits.append(circuit)
            else:
                run_circuits.append(constant_circuit + circuit)
        if self.batch_mode:
            (
                output_bit_register_dict,
                output_float_register_dict,
                output_complex_register_dict,
            ) = self.run_circuits_batch(run_circuits)
        else:
            for run_circuit in run_circuits:
                (
                    tmp_bit_register_dict,
                    tmp_float_register_dict,
                    tmp_complex_register_dict,
                ) = self.run_circuit(run_circuit)
                for key, value_bools in tmp_bit_register_dict.items():
                    if key in output_bit_register_dict:
                        output_bit_register_dict[key].extend(value_bools)
                    else:
                        output_bit_register_dict[key] = value_bools
                for key, value_floats in tmp_float_register_dict.items():
                    if key in output_float_register_dict:
                        output_float_register_dict[key].extend(value_floats)
                    else:
                        output_float_register_dict[key] = value_floats
                for key, value_complexes in tmp_complex_register_dict.items():
                    if key in output_complex_register_dict:
                        output_complex_register_dict[key].extend(value_complexes)
                    else:
                        output_complex_register_dict[key] = value_complexes

        return (
            output_bit_register_dict,
            output_float_register_dict,
            output_complex_register_dict,
        )

    def run_measurement_registers_hybrid(self, measurement: Any) -> Union[
        Tuple[
            Dict[str, List[List[bool]]],
            Dict[str, List[List[float]]],
            Dict[str, List[List[complex]]],
        ],
        Dict[str, float],
    ]:
        """Run all circuits of a measurement with the AWS Braket backend using hybrid jobs.

        Using hybrid jobs allows us to naturally group the circuits from a measurement.

        Args:
            measurement: The measurement that is run.

        Returns:
            Union[
                Tuple[
                    Dict[str, List[List[bool]]],
                    Dict[str, List[List[float]]],
                    Dict[str, List[List[complex]]],
                ],
                Dict[str, float],
            ]
        """
        job = self._run_measurement_registers_hybrid(measurement)
        with tempfile.TemporaryDirectory() as tmpdir:
            jobname = job.name
            job.download_result(extract_to=tmpdir)
            if isinstance(job, AwsQuantumJob):
                with open(os.path.join(os.path.join(tmpdir, jobname), "output.json")) as f:
                    outputs = json.load(f)
            elif isinstance(job, LocalQuantumJob):
                with open(os.path.join(os.path.join(os.getcwd(), jobname), "output.json")) as f:
                    outputs = json.load(f)
                shutil.rmtree(os.path.join(os.getcwd(), jobname))
            if not isinstance(measurement, ClassicalRegister):
                outputs = measurement.evaluate(outputs[0], outputs[1], outputs[2])

        return outputs

    def run_measurement_registers_hybrid_queued(self, measurement: Any) -> QueuedHybridRun:
        """Run all circuits of a measurement with the AWS Braket backend using hybrid jobs.

        Using hybrid jobs allows us to naturally group the circuits from a measurement.

        Args:
            measurement: The measurement that is run.

        Returns:
            QueuedHybridRun
        """
        job = self._run_measurement_registers_hybrid(measurement, wait_until_complete=False)
        return QueuedHybridRun(self.aws_session, job, job.metadata(), measurement)

    def _run_measurement_registers_hybrid(
        self, measurement: Any, wait_until_complete: bool = True
    ) -> AwsQuantumJob:
        """Run all circuits of a measurement with the AWS Braket backend using hybrid jobs.

        Using hybrid jobs allows us to naturally group the circuits from a measurement.

        Args:
            measurement: The measurement that is run.
            wait_until_complete: Whether to wait for the job to complete.\
                Should be False when using queued runs.

        Returns:
            AwsQuantumJob
        """
        # get path of this file
        file_path = os.path.dirname(os.path.realpath(__file__))
        measurement_json = measurement.to_json()
        helper_file_path = os.path.join(file_path, "qoqo_hybrid_helper.py")
        # create named temporary directory with tempfile
        os.mkdir("_tmp_hybrid_helper")
        shutil.copyfile(
            helper_file_path, os.path.join("_tmp_hybrid_helper", "qoqo_hybrid_helper.py")
        )
        requirement_lines = ["qoqo >= 1.11\n", "qoqo-for-braket >= 0.5"]
        with open(os.path.join("_tmp_hybrid_helper", "requirements.txt"), "w") as f:
            # write each line from requirement_lines to separate lines in file
            f.writelines(requirement_lines)
        with open(".tmp_measurement_input.json", "w") as f:
            f.write(measurement_json)
        with open(".tmp_config_input.json", "w") as f:
            json.dump(self._create_config(), f)
        device_for_run = self.__create_device()
        if isinstance(device_for_run, LocalSimulator):
            job = LocalQuantumJob.create(
                device=self.device,
                source_module="_tmp_hybrid_helper",
                entry_point="_tmp_hybrid_helper.qoqo_hybrid_helper:run_measurement_register",
                input_data={
                    "measurement": ".tmp_measurement_input.json",
                    "config": ".tmp_config_input.json",
                },
            )
        elif isinstance(device_for_run, AwsDevice):
            job = AwsQuantumJob.create(
                device=device_for_run._arn,
                source_module="_tmp_hybrid_helper",
                entry_point="_tmp_hybrid_helper.qoqo_hybrid_helper:run_measurement_register",
                wait_until_complete=wait_until_complete,
                input_data={
                    "measurement": ".tmp_measurement_input.json",
                    "config": ".tmp_config_input.json",
                },
            )
        shutil.rmtree("_tmp_hybrid_helper")
        os.remove(".tmp_measurement_input.json")
        os.remove(".tmp_config_input.json")
        return job

    def run_measurement(self, measurement: Any) -> Optional[Dict[str, float]]:
        """Run a circuit with the AWS Braket backend.

        Args:
            measurement: The measurement that is run.

        Returns:
            Optional[Dict[str, float]]
        """
        (
            output_bit_register_dict,
            output_float_register_dict,
            output_complex_register_dict,
        ) = self.run_measurement_registers(measurement)

        return measurement.evaluate(
            output_bit_register_dict,
            output_float_register_dict,
            output_complex_register_dict,
        )

    def run_program(
        self, program: QuantumProgram, params_values: Union[List[float], List[List[float]]]
    ) -> Optional[
        List[
            Union[
                Tuple[
                    Dict[str, List[List[bool]]],
                    Dict[str, List[List[float]]],
                    Dict[str, List[List[complex]]],
                ],
                Dict[str, float],
            ]
        ]
    ]:
        """Run a qoqo quantum program on a AWS backend multiple times.

        It can handle QuantumProgram instances containing any kind of measurement. The list of
        lists of parameters will be used to call `program.run(self, params)` or
        `program.run_registers(self, params)` as many times as the number of sublists.
        The return type will change accordingly.

        If no parameters values are provided, a normal call `program.run(self, [])` call
        will be executed.

        Args:
            program (QuantumProgram): the qoqo quantum program to run.
            params_values (Union[List[float], List[List[float]]]): the parameters values to pass
            to the quantum program.

        Returns:
            Optional[
                List[
                    Union[
                        Tuple[
                            Dict[str, List[List[bool]]],
                            Dict[str, List[List[float]]],
                            Dict[str, List[List[complex]]],
                        ],
                        Dict[str, float],
                    ]
                ]
            ]: list of dictionaries (or tuples of dictionaries) containing the
                run results.
        """
        returned_results = []

        if isinstance(program.measurement(), ClassicalRegister):
            if not params_values:
                returned_results.append(program.run_registers(self, []))
            if isinstance(params_values[0], list):
                for params in params_values:
                    returned_results.append(program.run_registers(self, params))
            else:
                return program.run_registers(self, params_values)
        else:
            if not params_values:
                returned_results.append(program.run(self, []))
            if isinstance(params_values[0], list):
                for params in params_values:
                    returned_results.append(program.run(self, params))
            else:
                return program.run(self, params_values)

        return returned_results

    def run_circuit_queued(self, circuit: Circuit) -> QueuedCircuitRun:
        """Run a Circuit on a AWS backend and return a queued Job Result.

        The default number of shots for the simulation is 100.
        Any kind of Measurement instruction only works as intended if
        it are the last instructions in the Circuit.
        Currently only one simulation is performed, meaning different measurements on different
        registers are not supported.

        Args:
            circuit (Circuit): the Circuit to simulate.

        Returns:
            QueuedCircuitRun
        """
        (quantum_task, metadata, _input_bit_circuit) = self._run_circuit(circuit)
        return QueuedCircuitRun(self.aws_session, quantum_task, metadata)

    def run_measurement_queued(self, measurement: Any) -> QueuedProgramRun:
        """Run a qoqo measurement on a AWS backend and return a queued Job Result.

        The default number of shots for the simulation is 100.
        Any kind of Measurement instruction only works as intended if
        it are the last instructions in the Circuit.
        Currently only one simulation is performed, meaning different measurements on different
        registers are not supported.

        Args:
            measurement (qoqo.measurement): the measurement to simulate.

        Returns:
            QueuedProgramRun
        """
        queued_circuits = []
        constant_circuit = measurement.constant_circuit()
        for circuit in measurement.circuits():
            if constant_circuit is None:
                run_circuit = circuit
            else:
                run_circuit = constant_circuit + circuit

            queued_circuits.append(self.run_circuit_queued(run_circuit))
        return QueuedProgramRun(measurement, queued_circuits)

    def run_program_queued(
        self, program: QuantumProgram, params_values: List[List[float]], hybrid: bool = False
    ) -> List[Union[QueuedProgramRun, QueuedHybridRun]]:
        """Run a qoqo quantum program on a AWS backend multiple times return a list of queued Jobs.

        This effectively performs the same operations as `run_program` but returns
        queued results.

        The hybrid parameter can specify whether to run the program in hybrid mode or not.

        Args:
            program (QuantumProgram): the qoqo quantum program to run.
            params_values (List[List[float]]): the parameters values to pass to the quantum
                program.
            hybrid (bool): whether to run the program in hybrid mode.

        Raises:
            ValueError: incorrect length of params_values compared to program's input
                parameter names.

        Returns:
            List[Union[QueuedProgramRun, QueuedHybridRun]]
        """
        queued_runs: List[Union[QueuedProgramRun, QueuedHybridRun]] = []
        input_parameter_names = program.input_parameter_names()

        if not params_values:
            if hybrid:
                queued_runs.append(
                    self.run_measurement_registers_hybrid_queued(program.measurement())
                )
            else:
                queued_runs.append(self.run_measurement_queued(program.measurement()))
        for params in params_values:
            if len(params) != len(input_parameter_names):
                raise ValueError(
                    f"Wrong number of parameters {len(input_parameter_names)} parameters"
                    f" expected {len(params)} parameters given."
                )
            substituted_parameters = dict(zip(input_parameter_names, params))
            measurement = program.measurement().substitute_parameters(substituted_parameters)
            if hybrid:
                queued_runs.append(self.run_measurement_registers_hybrid_queued(measurement))
            else:
                queued_runs.append(self.run_measurement_queued(measurement))

        return queued_runs


_GATE_MAP_STR = {
    "id": "id",
    "prx": "r",
    "rx": "rx",
    "ry": "ry",
    "rz": "rz",
    "sx": "sx",
    "x": "x",
    "cx": "cx",
    "cz": "cz",
    "h": "h",
    "y": "y",
    "z": "z",
    "t": "t",
    "tdg": "tdg",
    "s": "s",
    "sdg": "sdg",
    "swap": "swap",
}

_GATE_MAP = {
    "id": IGate,
    "r": RGate,
    "rx": RXGate,
    "ry": RYGate,
    "rz": RZGate,
    "sx": SXGate,
    "x": XGate,
    "cx": CXGate,
    "cz": CZGate,
    "h": HGate,
    "y": YGate,
    "z": ZGate,
    "t": TGate,
    "tdg": TdgGate,
    "s": SGate,
    "sdg": SdgGate,
    "swap": SwapGate,
}


class QiskitBraketBackend(BackendV2):
    def __init__(self, calibration_dict):
        super().__init__(name="QiskitBraketBackend")
        self._num_qubits = calibration_dict["num_qubits"]
        self._coupling_map = CouplingMap(calibration_dict["coupling_map"])
        self._basis_gates = [
            _GATE_MAP_STR[gate]
            for gate in calibration_dict["basis_gates"]
            if _GATE_MAP_STR.get(gate)
        ]

        # Process calibration data
        self.noise_infos = calibration_dict["noise_infos"]
        self.single_qubit_errors = {}
        self.measurement_errors = {}
        self.two_qubit_errors = {}
        self.t1_times = {}
        self.t2_times = {}
        self.t1_errors = {}
        self.t2_errors = {}

        self._process_calibration_data()
        self._target = self._build_target()

    def _process_calibration_data(self):
        # Single qubit properties
        for qubit_id_str, qubit_prop in self.noise_infos.one_qubit_properties.items():
            qubit_id = int(qubit_id_str) - 1
            # Direct attribute access for T1/T2 properties
            t1_prop = qubit_prop.t1  # t1_prop.value, t1_prop.standard_error
            t2_prop = qubit_prop.t2  # t2_prop.value, t2_prop.standard_error

            self.t1_times[qubit_id] = t1_prop.value
            self.t1_errors[qubit_id] = t1_prop.standard_error
            self.t2_times[qubit_id] = t2_prop.value
            self.t2_errors[qubit_id] = t2_prop.standard_error

            # Extract fidelities
            fidelity_measurements = qubit_prop.one_qubit_fidelity
            single_qubit_fidelity = 0.99  # Default fallback
            readout_fidelity = 0.95  # Default fallback

            for fidelity_measurement in fidelity_measurements:
                # Assume direct attribute access and string attribute for fidelity type name!
                fidelity_type_name = fidelity_measurement.fidelity_type.name
                if fidelity_type_name == "SIMULTANEOUS_RANDOMIZED_BENCHMARKING":
                    single_qubit_fidelity = fidelity_measurement.fidelity
                elif fidelity_type_name == "READOUT":
                    readout_fidelity = fidelity_measurement.fidelity

            self.single_qubit_errors[qubit_id] = 1 - single_qubit_fidelity
            self.measurement_errors[qubit_id] = 1 - readout_fidelity

        # Two qubit properties
        for qubit_pair_str, two_qubit_prop in self.noise_infos.two_qubit_properties.items():
            qubit_pair = [int(q) - 1 for q in qubit_pair_str.split("-")]
            gate_fidelities = two_qubit_prop.two_qubit_gate_fidelity
            for gate_fidelity in gate_fidelities:
                error = 1 - gate_fidelity.fidelity
                if qubit_pair[0] not in self.two_qubit_errors:
                    self.two_qubit_errors[qubit_pair[0]] = {}
                if qubit_pair[1] not in self.two_qubit_errors:
                    self.two_qubit_errors[qubit_pair[1]] = {}
                self.two_qubit_errors[qubit_pair[0]][qubit_pair[1]] = error
                self.two_qubit_errors[qubit_pair[1]][qubit_pair[0]] = error

    def _build_target(self):
        target = Target(num_qubits=self._num_qubits)
        all_qubits = list(range(self._num_qubits))

        # Add measurement with readout errors
        target.add_instruction(
            Measure(),
            {
                (q,): InstructionProperties(error=self.measurement_errors.get(q, 0.05))
                for q in range(self._num_qubits)
            },
            name="measure",
        )

        for gate_name in self._basis_gates:
            gate_cls = _GATE_MAP[gate_name]
            if gate_name in ("cx", "swap", "cz"):
                pair_dict = {
                    tuple(pair): InstructionProperties(
                        error=self.two_qubit_errors.get(pair[0], {}).get(pair[1], 0.01)
                    )
                    for pair in self._coupling_map.get_edges()
                }
                target.add_instruction(gate_cls(), pair_dict, name=gate_name)
            elif gate_name in ("rx", "ry", "rz"):
                prop_dict = {
                    (q,): InstructionProperties(error=self.single_qubit_errors.get(q, 0.01))
                    for q in all_qubits
                }
                target.add_instruction(gate_cls(1), prop_dict, name=gate_name)
            elif gate_name == "r":
                prop_dict = {
                    (q,): InstructionProperties(error=self.single_qubit_errors.get(q, 0.01))
                    for q in all_qubits
                }
                target.add_instruction(gate_cls(0, 0), prop_dict, name=gate_name)
            else:
                prop_dict = {
                    (q,): InstructionProperties(error=self.single_qubit_errors.get(q, 0.01))
                    for q in all_qubits
                }
                target.add_instruction(gate_cls(), prop_dict, name=gate_name)

        return target

    @property
    def num_qubits(self):
        return self._num_qubits

    @property
    def coupling_map(self):
        return self._coupling_map

    @property
    def operation_names(self):
        return self._basis_gates

    @property
    def target(self):
        return self._target

    @property
    def max_circuits(self):
        return 1000

    @staticmethod
    def _default_options():
        return Options()

    def run(self, circuits, **kwargs):
        raise NotImplementedError("This backend is for transpilation only")

    # T1/T2 inspection helpers
    def get_t1_time(self, qubit):
        return self.t1_times.get(qubit)

    def get_t2_time(self, qubit):
        return self.t2_times.get(qubit)

    def get_t1_error(self, qubit):
        return self.t1_errors.get(qubit)

    def get_t2_error(self, qubit):
        return self.t2_errors.get(qubit)
