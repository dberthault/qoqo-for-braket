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

use serde::{Deserialize, Serialize};
use std::collections::HashMap;

#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct StandardizedDevice {
    #[serde(rename = "braketSchemaHeader")]
    pub braket_schema_header: BraketSchemaHeader,
    #[serde(rename = "oneQubitProperties")]
    pub one_qubit_properties: HashMap<String, OneQubitProperty>,
    #[serde(rename = "twoQubitProperties")]
    pub two_qubit_properties: HashMap<String, TwoQubitProperty>,
}

#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct BraketSchemaHeader {
    pub name: String,
    pub version: String,
}

#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct OneQubitProperty {
    #[serde(rename = "T1")]
    pub t1: TimeProperty,
    #[serde(rename = "T2")]
    pub t2: TimeProperty,
    #[serde(rename = "oneQubitFidelity")]
    pub one_qubit_fidelity: Vec<FidelityMeasurement>,
}

#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct TimeProperty {
    pub value: f64,
    #[serde(rename = "standardError")]
    pub standard_error: f64,
    pub unit: String,
}

#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct FidelityMeasurement {
    #[serde(rename = "fidelityType")]
    pub fidelity_type: FidelityType,
    pub fidelity: f64,
    #[serde(rename = "standardError")]
    pub standard_error: Option<f64>,
}

#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct FidelityType {
    pub name: String,
    pub description: Option<String>,
}

#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct TwoQubitProperty {
    #[serde(rename = "twoQubitGateFidelity")]
    pub two_qubit_gate_fidelity: Vec<TwoQubitGateFidelity>,
}

#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct TwoQubitGateFidelity {
    pub direction: Option<String>,
    #[serde(rename = "gateName")]
    pub gate_name: String,
    pub fidelity: f64,
    #[serde(rename = "standardError")]
    pub standard_error: f64,
    #[serde(rename = "fidelityType")]
    pub fidelity_type: FidelityType,
}
