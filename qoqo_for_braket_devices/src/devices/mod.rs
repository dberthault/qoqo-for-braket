// Copyright © 2023 HQS Quantum Simulations GmbH. All Rights Reserved.
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

mod ionq_aria1;
pub use ionq_aria1::*;

mod ionq_harmony;
pub use ionq_harmony::*;

mod oqc_lucy;
pub use oqc_lucy::*;

mod rigetti_aspenm3;
pub use rigetti_aspenm3::*;

mod standardize_device;
pub use standardize_device::*;

use qoqo_iqm::GarnetDeviceWrapper;

use pyo3::prelude::*;

/// AWS Devices
#[pymodule]
pub fn aws_devices(_py: Python, m: &Bound<PyModule>) -> PyResult<()> {
    m.add_class::<IonQAria1DeviceWrapper>()?;
    m.add_class::<IonQHarmonyDeviceWrapper>()?;
    m.add_class::<OQCLucyDeviceWrapper>()?;
    m.add_class::<RigettiAspenM3DeviceWrapper>()?;
    m.add_class::<GarnetDeviceWrapper>()?;
    m.add_class::<BraketSchemaHeaderWrapper>()?;
    m.add_class::<TimePropertyWrapper>()?;
    m.add_class::<FidelityTypeWrapper>()?;
    m.add_class::<FidelityMeasurementWrapper>()?;
    m.add_class::<OneQubitPropertyWrapper>()?;
    m.add_class::<TwoQubitGateFidelityWrapper>()?;
    m.add_class::<TwoQubitPropertyWrapper>()?;
    m.add_class::<StandardizedDeviceWrapper>()?;
    Ok(())
}
