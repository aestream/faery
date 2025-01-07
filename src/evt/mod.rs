mod common;
mod decoder;
mod encoder;

use crate::types;

use pyo3::prelude::*;

impl From<decoder::Error> for PyErr {
    fn from(error: decoder::Error) -> Self {
        PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(error.to_string())
    }
}

impl From<encoder::Error> for PyErr {
    fn from(error: encoder::Error) -> Self {
        PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(error.to_string())
    }
}

impl From<common::Error> for PyErr {
    fn from(error: common::Error) -> Self {
        PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(error.to_string())
    }
}

#[pyclass]
pub struct Decoder {
    inner: Option<decoder::Decoder>,
}

#[pymethods]
impl Decoder {
    #[new]
    #[pyo3(signature = (path, dimensions_fallback, version_fallback))]
    fn new(
        path: &pyo3::Bound<'_, pyo3::types::PyAny>,
        dimensions_fallback: Option<(u16, u16)>,
        version_fallback: Option<String>,
    ) -> PyResult<Self> {
        Python::with_gil(|python| -> PyResult<Self> {
            Ok(Decoder {
                inner: Some(decoder::Decoder::new(
                    types::python_path_to_string(python, path)?,
                    dimensions_fallback,
                    version_fallback
                        .map(|version| common::Version::from_string(&version))
                        .transpose()?,
                )?),
            })
        })
    }

    #[getter]
    fn version(&self) -> PyResult<String> {
        match self.inner {
            Some(ref decoder) => Ok(decoder.version().to_string().to_owned()),
            None => Err(pyo3::exceptions::PyException::new_err(
                "called version after __exit__",
            )),
        }
    }

    #[getter]
    fn dimensions(&self) -> PyResult<(u16, u16)> {
        match self.inner {
            Some(ref decoder) => Ok(decoder.dimensions),
            None => Err(pyo3::exceptions::PyException::new_err(
                "called dimensions after __exit__",
            )),
        }
    }

    fn __enter__(slf: Py<Self>) -> Py<Self> {
        slf
    }

    #[pyo3(signature = (_exception_type, _value, _traceback))]
    fn __exit__(
        &mut self,
        _exception_type: Option<PyObject>,
        _value: Option<PyObject>,
        _traceback: Option<PyObject>,
    ) -> PyResult<bool> {
        if self.inner.is_none() {
            return Err(pyo3::exceptions::PyException::new_err(
                "multiple calls to __exit__",
            ));
        }
        let _ = self.inner.take();
        Ok(false)
    }

    fn __iter__(shell: PyRefMut<Self>) -> PyResult<Py<Decoder>> {
        Ok(shell.into())
    }

    fn __next__(mut shell: PyRefMut<Self>) -> PyResult<Option<PyObject>> {
        let packet = match shell.inner {
            Some(ref mut decoder) => match decoder.next() {
                Ok(result) => match result {
                    Some(result) => result,
                    None => return Ok(None),
                },
                Err(result) => return Err(result.into()),
            },
            None => {
                return Err(pyo3::exceptions::PyException::new_err(
                    "used decoder after __exit__",
                ))
            }
        };
        Python::with_gil(|python| -> PyResult<Option<PyObject>> {
            let python_packet = pyo3::types::PyDict::new_bound(python);
            if !packet.0.is_empty() {
                let length = packet.0.len() as numpy::npyffi::npy_intp;
                let array = types::ArrayType::Dvs.new_array(python, length);
                python_packet.set_item("events", unsafe {
                    for index in 0..length {
                        let event_cell = types::array_at(python, array, index);
                        std::ptr::copy(
                            &packet.0[index as usize]
                                as *const neuromorphic_types::DvsEvent<u64, u16, u16>
                                as *const u8,
                            event_cell,
                            std::mem::size_of::<neuromorphic_types::DvsEvent<u64, u16, u16>>(),
                        );
                    }
                    PyObject::from_owned_ptr(python, array as *mut pyo3::ffi::PyObject)
                })?;
            }
            if !packet.1.is_empty() {
                let length = packet.1.len() as numpy::npyffi::npy_intp;
                let array = types::ArrayType::EvtTrigger.new_array(python, length);
                python_packet.set_item("triggers", unsafe {
                    for index in 0..length {
                        let trigger_cell = types::array_at(python, array, index);
                        let trigger = packet.1[index as usize];
                        let mut trigger_array = [0u8; 10];
                        trigger_array[0..8].copy_from_slice(&trigger.t.to_le_bytes());
                        trigger_array[8] = trigger.id;
                        trigger_array[9] = match trigger.polarity {
                            neuromorphic_types::TriggerPolarity::Falling => 0,
                            neuromorphic_types::TriggerPolarity::Rising => 1,
                        };
                        std::ptr::copy(trigger_array.as_ptr(), trigger_cell, trigger_array.len());
                    }
                    PyObject::from_owned_ptr(python, array as *mut pyo3::ffi::PyObject)
                })?;
            }
            Ok(Some(python_packet.into()))
        })
    }
}

#[pyclass]
pub struct Encoder {
    inner: Option<encoder::Encoder>,
}

#[pymethods]
impl Encoder {
    #[new]
    fn new(
        path: &pyo3::Bound<'_, pyo3::types::PyAny>,
        version: &str,
        zero_t0: bool,
        dimensions: (u16, u16),
    ) -> PyResult<Self> {
        Python::with_gil(|python| -> PyResult<Self> {
            Ok(Encoder {
                inner: Some(encoder::Encoder::new(
                    types::python_path_to_string(python, path)?,
                    common::Version::from_string(version)?,
                    zero_t0,
                    dimensions,
                )?),
            })
        })
    }

    fn __enter__(slf: Py<Self>) -> Py<Self> {
        slf
    }

    #[pyo3(signature = (_exception_type, _value, _traceback))]
    fn __exit__(
        &mut self,
        _exception_type: Option<PyObject>,
        _value: Option<PyObject>,
        _traceback: Option<PyObject>,
    ) -> PyResult<bool> {
        if self.inner.is_none() {
            return Err(pyo3::exceptions::PyException::new_err(
                "multiple calls to __exit__",
            ));
        }
        let _ = self.inner.take();
        Ok(false)
    }

    fn t0(&mut self) -> PyResult<Option<u64>> {
        match &self.inner {
            Some(encoder) => Ok(encoder.t0()),
            None => Err(pyo3::exceptions::PyException::new_err(
                "t0 called after __exit__",
            )),
        }
    }

    fn write(&mut self, packet: &pyo3::Bound<'_, pyo3::types::PyDict>) -> PyResult<()> {
        Python::with_gil(|python| -> PyResult<()> {
            match self.inner.as_mut() {
                Some(encoder) => {
                    let mut events_and_length = None;
                    let mut triggers_and_length = None;
                    for (key, value) in packet.iter() {
                        let name: String = key.extract()?;
                        match name.as_str() {
                            "events" => {
                                events_and_length = Some(types::check_array(
                                    python,
                                    types::ArrayType::Dvs,
                                    &value,
                                )?);
                            }
                            "triggers" => {
                                triggers_and_length = Some(types::check_array(
                                    python,
                                    types::ArrayType::Dvs,
                                    &value,
                                )?);
                            }
                            name => {
                                return Err(PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(
                                    format!("unexpected dict key \"{name}\""),
                                ));
                            }
                        }
                    }
                    let has_events = events_and_length.map_or(false, |(_, length)| length > 0);
                    let has_triggers = triggers_and_length.map_or(false, |(_, length)| length > 0);
                    if has_events && has_triggers {
                        let (events, events_length) = events_and_length.unwrap();
                        let (triggers, triggers_length) = events_and_length.unwrap();
                        let mut event_index = 0;
                        let mut trigger_index = 0;
                        unsafe {
                            let mut event_cell: *mut neuromorphic_types::DvsEvent<u64, u16, u16> =
                                types::array_at(python, events, event_index);
                            let mut trigger_cell: *mut neuromorphic_types::TriggerEvent<u64, u8> =
                                types::array_at(python, triggers, trigger_index);
                            loop {
                                if event_index < events_length {
                                    if trigger_index < triggers_length {
                                        if (*trigger_cell).t < (*event_cell).t {
                                            encoder.write_trigger_event(*trigger_cell)?;
                                            trigger_index += 1;
                                            if trigger_index < triggers_length {
                                                trigger_cell = types::array_at(
                                                    python,
                                                    triggers,
                                                    trigger_index,
                                                );
                                            }
                                        } else {
                                            encoder.write_dvs_event(*event_cell)?;
                                            event_index += 1;
                                            if event_index < events_length {
                                                event_cell =
                                                    types::array_at(python, events, event_index);
                                            }
                                        }
                                    } else {
                                        encoder.write_dvs_event(*event_cell)?;
                                        event_index += 1;
                                        if event_index < events_length {
                                            event_cell =
                                                types::array_at(python, events, event_index);
                                        }
                                    }
                                } else {
                                    if trigger_index < triggers_length {
                                        encoder.write_trigger_event(*trigger_cell)?;
                                        trigger_index += 1;
                                        if trigger_index < triggers_length {
                                            trigger_cell =
                                                types::array_at(python, triggers, trigger_index);
                                        }
                                    } else {
                                        break;
                                    }
                                }
                            }
                        }
                    } else if has_events {
                        let (events, events_length) = events_and_length.unwrap();
                        unsafe {
                            for index in 0..events_length {
                                let event_cell = types::array_at(python, events, index);
                                encoder.write_dvs_event(*event_cell)?;
                            }
                        }
                    } else if has_triggers {
                        let (triggers, triggers_length) = events_and_length.unwrap();
                        unsafe {
                            for index in 0..triggers_length {
                                let trigger_cell = types::array_at(python, triggers, index);
                                encoder.write_trigger_event(*trigger_cell)?;
                            }
                        }
                    }
                    Ok(())
                }
                None => Err(pyo3::exceptions::PyException::new_err(
                    "write called after __exit__",
                )),
            }
        })
    }
}
