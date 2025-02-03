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

#[pyclass]
pub struct Decoder {
    inner: Option<decoder::Decoder>,
}

#[pymethods]
impl Decoder {
    #[new]
    fn new(path: &pyo3::Bound<'_, pyo3::types::PyAny>, t0: u64) -> PyResult<Self> {
        Python::with_gil(|python| -> PyResult<Self> {
            Ok(Decoder {
                inner: Some(decoder::Decoder::new(
                    types::python_path_to_string(python, path)?,
                    t0,
                )?),
            })
        })
    }

    #[getter]
    fn version(&self) -> PyResult<String> {
        match self.inner {
            Some(ref decoder) => Ok({
                let version = decoder.version();
                format!("{}.{}.{}", version[0], version[1], version[2])
            }),
            None => Err(pyo3::exceptions::PyException::new_err(
                "called version after __exit__",
            )),
        }
    }

    #[getter]
    fn event_type(&self) -> PyResult<String> {
        match self.inner {
            Some(ref decoder) => Ok(match decoder.event_type {
                common::Type::Generic => "generic",
                common::Type::Dvs => "dvs",
                common::Type::Atis => "atis",
                common::Type::Color => "color",
            }
            .to_owned()),
            None => Err(pyo3::exceptions::PyException::new_err(
                "called event_type after __exit__",
            )),
        }
    }

    #[getter]
    fn dimensions(&self) -> PyResult<Option<(u16, u16)>> {
        match self.inner {
            Some(ref decoder) => Ok(decoder.dimensions()),
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
                    "called __next__ after __exit__",
                ))
            }
        };
        Python::with_gil(|python| -> PyResult<Option<PyObject>> {
            Ok(Some(match packet {
                decoder::Packet::Generic(events) => {
                    let length = events.len() as numpy::npyffi::npy_intp;
                    let array = types::ArrayType::EsGeneric.new_array(python, length);
                    unsafe {
                        for index in 0..length {
                            let event_cell = types::array_at(python, array, index);
                            let event = &events[index as usize];
                            let mut event_array = [0u8; 8 + std::mem::size_of::<usize>()];
                            event_array[0..8].copy_from_slice(&event.t.to_le_bytes());
                            let pybytes = pyo3::ffi::PyBytes_FromStringAndSize(
                                event.bytes.as_ptr() as *const core::ffi::c_char,
                                event.bytes.len() as pyo3::ffi::Py_ssize_t,
                            );
                            event_array[8..8 + std::mem::size_of::<usize>()]
                                .copy_from_slice(&(pybytes as usize).to_ne_bytes());
                            std::ptr::copy(event_array.as_ptr(), event_cell, event_array.len());
                        }
                        PyObject::from_owned_ptr(python, array as *mut pyo3::ffi::PyObject)
                    }
                }
                decoder::Packet::Dvs(events) => {
                    let length = events.len() as numpy::npyffi::npy_intp;
                    let array = types::ArrayType::Dvs.new_array(python, length);
                    unsafe {
                        for index in 0..length {
                            let event_cell = types::array_at(python, array, index);
                            std::ptr::copy(
                                &events[index as usize]
                                    as *const neuromorphic_types::DvsEvent<u64, u16, u16>
                                    as *const u8,
                                event_cell,
                                std::mem::size_of::<neuromorphic_types::DvsEvent<u64, u16, u16>>(),
                            );
                        }
                        PyObject::from_owned_ptr(python, array as *mut pyo3::ffi::PyObject)
                    }
                }
                decoder::Packet::Atis(events) => {
                    let length = events.len() as numpy::npyffi::npy_intp;
                    let array = types::ArrayType::EsAtis.new_array(python, length);
                    unsafe {
                        for index in 0..length {
                            let event_cell = types::array_at(python, array, index);
                            let event = events[index as usize];
                            let mut event_array = [0u8; 14];
                            event_array[0..8].copy_from_slice(&event.t.to_le_bytes());
                            event_array[8..10].copy_from_slice(&event.x.to_le_bytes());
                            event_array[10..12].copy_from_slice(&event.y.to_le_bytes());
                            match event.polarity {
                                neuromorphic_types::AtisPolarity::Off => {
                                    event_array[12] = 0;
                                    event_array[13] = 0;
                                }
                                neuromorphic_types::AtisPolarity::On => {
                                    event_array[12] = 0;
                                    event_array[13] = 1;
                                }
                                neuromorphic_types::AtisPolarity::ExposureStart => {
                                    event_array[12] = 1;
                                    event_array[13] = 0;
                                }
                                neuromorphic_types::AtisPolarity::ExposureEnd => {
                                    event_array[12] = 1;
                                    event_array[13] = 1;
                                }
                            }
                            std::ptr::copy(event_array.as_ptr(), event_cell, event_array.len());
                        }
                        PyObject::from_owned_ptr(python, array as *mut pyo3::ffi::PyObject)
                    }
                }
                decoder::Packet::Color(events) => {
                    let length = events.len() as numpy::npyffi::npy_intp;
                    let array = types::ArrayType::EsColor.new_array(python, length);
                    unsafe {
                        for index in 0..length {
                            let event_cell = types::array_at(python, array, index);
                            std::ptr::copy(
                                &events[index as usize] as *const common::ColorEvent as *const u8,
                                event_cell,
                                std::mem::size_of::<common::ColorEvent>(),
                            );
                        }
                        PyObject::from_owned_ptr(python, array as *mut pyo3::ffi::PyObject)
                    }
                }
            }))
        })
    }
}

#[pyclass]
pub struct Encoder {
    inner: Option<encoder::Encoder>,
}

const CHECK_ERROR: &str =
    "the array's \"bytes\" field must contain bytes objects (PyBytes_Check failed)";

const CONVERT_ERROR: &str =
    "the array's \"bytes\" field must contain bytes objects (PyBytes_AsStringAndSize failed)";

fn atis_payload_error(exposure: u8, polarity: u8) -> String {
    format!(
        "the exposure and the polarity must be 0 or 1 (got exposure={} and polarity={})",
        exposure, polarity,
    )
}

#[pymethods]
impl Encoder {
    #[new]
    #[pyo3(signature = (path, event_type, zero_t0, dimensions))]
    fn new(
        path: &pyo3::Bound<'_, pyo3::types::PyAny>,
        event_type: &str,
        zero_t0: bool,
        dimensions: Option<(u16, u16)>,
    ) -> PyResult<Self> {
        Python::with_gil(|python| -> PyResult<Self> {
            Ok(Encoder {
                inner: Some(encoder::Encoder::new(
                    types::python_path_to_string(python, path)?,
                    zero_t0,
                    encoder::EncoderType::new(event_type, dimensions)?,
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

    fn write(&mut self, events: &pyo3::Bound<'_, pyo3::types::PyAny>) -> PyResult<()> {
        Python::with_gil(|python| -> PyResult<()> {
            match self.inner.as_mut() {
                Some(encoder) => match encoder {
                    encoder::Encoder::Generic(encoder) => {
                        let (array, length) =
                            types::check_array(python, types::ArrayType::EsGeneric, events)?;
                        unsafe {
                            for index in 0..length {
                                let event_cell = types::array_at(python, array, index);
                                let mut event_array = [0u8; 8 + std::mem::size_of::<usize>()];
                                std::ptr::copy(
                                    event_cell,
                                    event_array.as_mut_ptr(),
                                    event_array.len(),
                                );
                                encoder.write(common::GenericEvent {
                                    t: u64::from_le_bytes(
                                        event_array[0..8].try_into().expect("8 bytes"),
                                    ),
                                    bytes: {
                                        let pointer = usize::from_ne_bytes(
                                            event_array[8..8 + std::mem::size_of::<usize>()]
                                                .try_into()
                                                .expect("std::mem::size_of::<usize>() bytes"),
                                        )
                                            as *mut pyo3::ffi::PyObject;
                                        if pyo3::ffi::PyBytes_Check(pointer) == 0 {
                                            return Err(PyErr::new::<
                                                pyo3::exceptions::PyRuntimeError,
                                                _,
                                            >(
                                                CHECK_ERROR.to_owned()
                                            ));
                                        }
                                        let mut data: *mut core::ffi::c_char = std::ptr::null_mut();
                                        let mut length: pyo3::ffi::Py_ssize_t = 0;
                                        if pyo3::ffi::PyBytes_AsStringAndSize(
                                            pointer,
                                            &mut data as *mut *mut core::ffi::c_char,
                                            &mut length as *mut pyo3::ffi::Py_ssize_t,
                                        ) < 0
                                        {
                                            return Err(PyErr::new::<
                                                pyo3::exceptions::PyRuntimeError,
                                                _,
                                            >(
                                                CONVERT_ERROR.to_owned()
                                            ));
                                        }
                                        std::slice::from_raw_parts(
                                            data as *const u8,
                                            length as usize,
                                        )
                                    },
                                })?;
                            }
                        }
                        Ok(())
                    }
                    encoder::Encoder::Dvs(encoder) => {
                        let (array, length) =
                            types::check_array(python, types::ArrayType::Dvs, events)?;
                        unsafe {
                            for index in 0..length {
                                let event_cell = types::array_at(python, array, index);
                                encoder.write(*event_cell)?;
                            }
                        }
                        Ok(())
                    }
                    encoder::Encoder::Atis(encoder) => {
                        let (array, length) =
                            types::check_array(python, types::ArrayType::EsAtis, events)?;
                        unsafe {
                            for index in 0..length {
                                let event_cell = types::array_at(python, array, index);
                                let mut event_array = [0u8; 14];
                                std::ptr::copy(
                                    event_cell,
                                    event_array.as_mut_ptr(),
                                    event_array.len(),
                                );
                                encoder.write(neuromorphic_types::AtisEvent {
                                    t: u64::from_le_bytes(
                                        event_array[0..8].try_into().expect("8 bytes"),
                                    ),
                                    x: u16::from_le_bytes(
                                        event_array[8..10].try_into().expect("2 bytes"),
                                    ),
                                    y: u16::from_le_bytes(
                                        event_array[10..12].try_into().expect("2 bytes"),
                                    ),
                                    polarity: {
                                        if event_array[12] == 0 && event_array[13] == 0 {
                                            neuromorphic_types::AtisPolarity::Off
                                        } else if event_array[12] == 0 && event_array[13] == 1 {
                                            neuromorphic_types::AtisPolarity::On
                                        } else if event_array[12] == 1 && event_array[13] == 0 {
                                            neuromorphic_types::AtisPolarity::ExposureStart
                                        } else if event_array[12] == 1 && event_array[13] == 1 {
                                            neuromorphic_types::AtisPolarity::ExposureEnd
                                        } else {
                                            return Err(PyErr::new::<
                                                pyo3::exceptions::PyRuntimeError,
                                                _,
                                            >(
                                                atis_payload_error(
                                                    event_array[12],
                                                    event_array[13],
                                                ),
                                            ));
                                        }
                                    },
                                })?;
                            }
                        }

                        Ok(())
                    }
                    encoder::Encoder::Color(encoder) => {
                        let (array, length) =
                            types::check_array(python, types::ArrayType::EsColor, events)?;
                        unsafe {
                            for index in 0..length {
                                let event_cell = types::array_at(python, array, index);
                                encoder.write(*event_cell)?;
                            }
                        }
                        Ok(())
                    }
                },
                None => Err(pyo3::exceptions::PyException::new_err(
                    "write called after __exit__",
                )),
            }
        })
    }
}
