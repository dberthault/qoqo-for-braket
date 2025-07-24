// Copyright © 2025 HQS Quantum Simulations GmbH. All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except
// in compliance with the License. You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software distributed under the
// License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either
// express or implied. See the License for the specific language governing permissions and
// limitations under the License.

use pyo3::prelude::*;
use std::collections::HashMap;

// Import the original structs from another crate
use roqoqo_for_braket_devices::devices::{
    BraketSchemaHeader, FidelityMeasurement, FidelityType, OneQubitProperty, StandardizedDevice,
    TimeProperty, TwoQubitGateFidelity, TwoQubitProperty,
};

// PyO3 wrapper structs - these contain the original structs
#[pyclass(name = "BraketSchemaHeader")]
#[derive(Debug, Clone)]
pub struct BraketSchemaHeaderWrapper {
    pub internal: BraketSchemaHeader,
}

#[pymethods]
impl BraketSchemaHeaderWrapper {
    #[new]
    pub fn new(name: String, version: String) -> Self {
        Self {
            internal: BraketSchemaHeader { name, version },
        }
    }

    #[getter]
    pub fn name(&self) -> String {
        self.internal.name.clone()
    }

    #[setter]
    pub fn set_name(&mut self, name: String) {
        self.internal.name = name;
    }

    #[getter]
    pub fn version(&self) -> String {
        self.internal.version.clone()
    }

    #[setter]
    pub fn set_version(&mut self, version: String) {
        self.internal.version = version;
    }
}

impl From<BraketSchemaHeader> for BraketSchemaHeaderWrapper {
    fn from(internal: BraketSchemaHeader) -> Self {
        Self { internal }
    }
}

#[pyclass(name = "TimeProperty")]
#[derive(Debug, Clone)]
pub struct TimePropertyWrapper {
    pub internal: TimeProperty,
}

#[pymethods]
impl TimePropertyWrapper {
    #[new]
    pub fn new(value: f64, standard_error: f64, unit: String) -> Self {
        Self {
            internal: TimeProperty {
                value,
                standard_error,
                unit,
            },
        }
    }

    #[getter]
    pub fn value(&self) -> f64 {
        self.internal.value
    }

    #[setter]
    pub fn set_value(&mut self, value: f64) {
        self.internal.value = value;
    }

    #[getter]
    pub fn standard_error(&self) -> f64 {
        self.internal.standard_error
    }

    #[setter]
    pub fn set_standard_error(&mut self, standard_error: f64) {
        self.internal.standard_error = standard_error;
    }

    #[getter]
    pub fn unit(&self) -> String {
        self.internal.unit.clone()
    }

    #[setter]
    pub fn set_unit(&mut self, unit: String) {
        self.internal.unit = unit;
    }
}

impl From<TimeProperty> for TimePropertyWrapper {
    fn from(internal: TimeProperty) -> Self {
        Self { internal }
    }
}

#[pyclass(name = "FidelityType")]
#[derive(Debug, Clone)]
pub struct FidelityTypeWrapper {
    pub internal: FidelityType,
}

#[pymethods]
impl FidelityTypeWrapper {
    #[new]
    pub fn new(name: String, description: Option<String>) -> Self {
        Self {
            internal: FidelityType { name, description },
        }
    }

    #[getter]
    pub fn name(&self) -> String {
        self.internal.name.clone()
    }

    #[setter]
    pub fn set_name(&mut self, name: String) {
        self.internal.name = name;
    }

    #[getter]
    pub fn description(&self) -> Option<String> {
        self.internal.description.clone()
    }

    #[setter]
    pub fn set_description(&mut self, description: Option<String>) {
        self.internal.description = description;
    }
}

impl From<FidelityType> for FidelityTypeWrapper {
    fn from(internal: FidelityType) -> Self {
        Self { internal }
    }
}

#[pyclass(name = "FidelityMeasurement")]
#[derive(Debug, Clone)]
pub struct FidelityMeasurementWrapper {
    pub internal: FidelityMeasurement,
}

#[pymethods]
impl FidelityMeasurementWrapper {
    #[new]
    pub fn new(
        fidelity_type: FidelityTypeWrapper,
        fidelity: f64,
        standard_error: Option<f64>,
    ) -> Self {
        Self {
            internal: FidelityMeasurement {
                fidelity_type: fidelity_type.internal,
                fidelity,
                standard_error,
            },
        }
    }

    #[getter]
    pub fn fidelity_type(&self) -> FidelityTypeWrapper {
        FidelityTypeWrapper::from(self.internal.fidelity_type.clone())
    }

    #[setter]
    pub fn set_fidelity_type(&mut self, fidelity_type: FidelityTypeWrapper) {
        self.internal.fidelity_type = fidelity_type.internal;
    }

    #[getter]
    pub fn fidelity(&self) -> f64 {
        self.internal.fidelity
    }

    #[setter]
    pub fn set_fidelity(&mut self, fidelity: f64) {
        self.internal.fidelity = fidelity;
    }

    #[getter]
    pub fn standard_error(&self) -> Option<f64> {
        self.internal.standard_error
    }

    #[setter]
    pub fn set_standard_error(&mut self, standard_error: Option<f64>) {
        self.internal.standard_error = standard_error;
    }
}

impl From<FidelityMeasurement> for FidelityMeasurementWrapper {
    fn from(internal: FidelityMeasurement) -> Self {
        Self { internal }
    }
}

#[pyclass(name = "OneQubitProperty")]
#[derive(Debug, Clone)]
pub struct OneQubitPropertyWrapper {
    pub internal: OneQubitProperty,
}

#[pymethods]
impl OneQubitPropertyWrapper {
    #[new]
    pub fn new(
        t1: TimePropertyWrapper,
        t2: TimePropertyWrapper,
        one_qubit_fidelity: Vec<FidelityMeasurementWrapper>,
    ) -> Self {
        Self {
            internal: OneQubitProperty {
                t1: t1.internal,
                t2: t2.internal,
                one_qubit_fidelity: one_qubit_fidelity.into_iter().map(|f| f.internal).collect(),
            },
        }
    }

    #[getter]
    pub fn t1(&self) -> TimePropertyWrapper {
        TimePropertyWrapper::from(self.internal.t1.clone())
    }

    #[setter]
    pub fn set_t1(&mut self, t1: TimePropertyWrapper) {
        self.internal.t1 = t1.internal;
    }

    #[getter]
    pub fn t2(&self) -> TimePropertyWrapper {
        TimePropertyWrapper::from(self.internal.t2.clone())
    }

    #[setter]
    pub fn set_t2(&mut self, t2: TimePropertyWrapper) {
        self.internal.t2 = t2.internal;
    }

    #[getter]
    pub fn one_qubit_fidelity(&self) -> Vec<FidelityMeasurementWrapper> {
        self.internal
            .one_qubit_fidelity
            .iter()
            .map(|f| FidelityMeasurementWrapper::from(f.clone()))
            .collect()
    }

    #[setter]
    pub fn set_one_qubit_fidelity(&mut self, one_qubit_fidelity: Vec<FidelityMeasurementWrapper>) {
        self.internal.one_qubit_fidelity =
            one_qubit_fidelity.into_iter().map(|f| f.internal).collect();
    }
}

impl From<OneQubitProperty> for OneQubitPropertyWrapper {
    fn from(internal: OneQubitProperty) -> Self {
        Self { internal }
    }
}

#[pyclass(name = "TwoQubitGateFidelity")]
#[derive(Debug, Clone)]
pub struct TwoQubitGateFidelityWrapper {
    pub internal: TwoQubitGateFidelity,
}

#[pymethods]
impl TwoQubitGateFidelityWrapper {
    #[new]
    pub fn new(
        gate_name: String,
        fidelity: f64,
        standard_error: f64,
        fidelity_type: FidelityTypeWrapper,
        direction: Option<String>,
    ) -> Self {
        Self {
            internal: TwoQubitGateFidelity {
                direction,
                gate_name,
                fidelity,
                standard_error,
                fidelity_type: fidelity_type.internal,
            },
        }
    }

    #[getter]
    pub fn direction(&self) -> Option<String> {
        self.internal.direction.clone()
    }

    #[setter]
    pub fn set_direction(&mut self, direction: Option<String>) {
        self.internal.direction = direction;
    }

    #[getter]
    pub fn gate_name(&self) -> String {
        self.internal.gate_name.clone()
    }

    #[setter]
    pub fn set_gate_name(&mut self, gate_name: String) {
        self.internal.gate_name = gate_name;
    }

    #[getter]
    pub fn fidelity(&self) -> f64 {
        self.internal.fidelity
    }

    #[setter]
    pub fn set_fidelity(&mut self, fidelity: f64) {
        self.internal.fidelity = fidelity;
    }

    #[getter]
    pub fn standard_error(&self) -> f64 {
        self.internal.standard_error
    }

    #[setter]
    pub fn set_standard_error(&mut self, standard_error: f64) {
        self.internal.standard_error = standard_error;
    }

    #[getter]
    pub fn fidelity_type(&self) -> FidelityTypeWrapper {
        FidelityTypeWrapper::from(self.internal.fidelity_type.clone())
    }

    #[setter]
    pub fn set_fidelity_type(&mut self, fidelity_type: FidelityTypeWrapper) {
        self.internal.fidelity_type = fidelity_type.internal;
    }
}

impl From<TwoQubitGateFidelity> for TwoQubitGateFidelityWrapper {
    fn from(internal: TwoQubitGateFidelity) -> Self {
        Self { internal }
    }
}

#[pyclass(name = "TwoQubitProperty")]
#[derive(Debug, Clone)]
pub struct TwoQubitPropertyWrapper {
    pub internal: TwoQubitProperty,
}

#[pymethods]
impl TwoQubitPropertyWrapper {
    #[new]
    pub fn new(two_qubit_gate_fidelity: Vec<TwoQubitGateFidelityWrapper>) -> Self {
        Self {
            internal: TwoQubitProperty {
                two_qubit_gate_fidelity: two_qubit_gate_fidelity
                    .into_iter()
                    .map(|f| f.internal)
                    .collect(),
            },
        }
    }

    #[getter]
    pub fn two_qubit_gate_fidelity(&self) -> Vec<TwoQubitGateFidelityWrapper> {
        self.internal
            .two_qubit_gate_fidelity
            .iter()
            .map(|f| TwoQubitGateFidelityWrapper::from(f.clone()))
            .collect()
    }

    #[setter]
    pub fn set_two_qubit_gate_fidelity(
        &mut self,
        two_qubit_gate_fidelity: Vec<TwoQubitGateFidelityWrapper>,
    ) {
        self.internal.two_qubit_gate_fidelity = two_qubit_gate_fidelity
            .into_iter()
            .map(|f| f.internal)
            .collect();
    }
}

impl From<TwoQubitProperty> for TwoQubitPropertyWrapper {
    fn from(internal: TwoQubitProperty) -> Self {
        Self { internal }
    }
}

#[pyclass(name = "StandardizedDevice")]
#[derive(Debug, Clone)]
pub struct StandardizedDeviceWrapper {
    pub internal: StandardizedDevice,
}

#[pymethods]
impl StandardizedDeviceWrapper {
    #[new]
    pub fn new(
        braket_schema_header: BraketSchemaHeaderWrapper,
        one_qubit_properties: HashMap<String, OneQubitPropertyWrapper>,
        two_qubit_properties: HashMap<String, TwoQubitPropertyWrapper>,
    ) -> Self {
        Self {
            internal: StandardizedDevice {
                braket_schema_header: braket_schema_header.internal,
                one_qubit_properties: one_qubit_properties
                    .into_iter()
                    .map(|(k, v)| (k, v.internal))
                    .collect(),
                two_qubit_properties: two_qubit_properties
                    .into_iter()
                    .map(|(k, v)| (k, v.internal))
                    .collect(),
            },
        }
    }

    #[staticmethod]
    pub fn from_json(json_str: &str) -> PyResult<Self> {
        let standardized: StandardizedDevice = serde_json::from_str(json_str).map_err(|e| {
            pyo3::exceptions::PyValueError::new_err(format!("JSON parse error: {e}"))
        })?;

        Ok(Self {
            internal: standardized,
        })
    }

    #[getter]
    pub fn braket_schema_header(&self) -> BraketSchemaHeaderWrapper {
        BraketSchemaHeaderWrapper::from(self.internal.braket_schema_header.clone())
    }

    #[setter]
    pub fn set_braket_schema_header(&mut self, braket_schema_header: BraketSchemaHeaderWrapper) {
        self.internal.braket_schema_header = braket_schema_header.internal;
    }

    #[getter]
    pub fn one_qubit_properties(&self) -> HashMap<String, OneQubitPropertyWrapper> {
        self.internal
            .one_qubit_properties
            .iter()
            .map(|(k, v)| (k.clone(), OneQubitPropertyWrapper::from(v.clone())))
            .collect()
    }

    #[setter]
    pub fn set_one_qubit_properties(
        &mut self,
        one_qubit_properties: HashMap<String, OneQubitPropertyWrapper>,
    ) {
        self.internal.one_qubit_properties = one_qubit_properties
            .into_iter()
            .map(|(k, v)| (k, v.internal))
            .collect();
    }

    #[getter]
    pub fn two_qubit_properties(&self) -> HashMap<String, TwoQubitPropertyWrapper> {
        self.internal
            .two_qubit_properties
            .iter()
            .map(|(k, v)| (k.clone(), TwoQubitPropertyWrapper::from(v.clone())))
            .collect()
    }

    #[setter]
    pub fn set_two_qubit_properties(
        &mut self,
        two_qubit_properties: HashMap<String, TwoQubitPropertyWrapper>,
    ) {
        self.internal.two_qubit_properties = two_qubit_properties
            .into_iter()
            .map(|(k, v)| (k, v.internal))
            .collect();
    }

    pub fn get_one_qubit_property(&self, qubit_id: &str) -> Option<OneQubitPropertyWrapper> {
        self.internal
            .one_qubit_properties
            .get(qubit_id)
            .map(|prop| OneQubitPropertyWrapper::from(prop.clone()))
    }

    pub fn get_two_qubit_property(&self, qubit_pair: &str) -> Option<TwoQubitPropertyWrapper> {
        self.internal
            .two_qubit_properties
            .get(qubit_pair)
            .map(|prop| TwoQubitPropertyWrapper::from(prop.clone()))
    }

    pub fn get_one_qubit_ids(&self) -> Vec<String> {
        self.internal.one_qubit_properties.keys().cloned().collect()
    }

    pub fn get_two_qubit_pairs(&self) -> Vec<String> {
        self.internal.two_qubit_properties.keys().cloned().collect()
    }
}

impl From<StandardizedDevice> for StandardizedDeviceWrapper {
    fn from(internal: StandardizedDevice) -> Self {
        Self { internal }
    }
}
