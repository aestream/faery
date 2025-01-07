mod common;
mod decoder;
mod encoder;

use crate::types;

use pyo3::prelude::*;

impl From<common::Error> for PyErr {
    fn from(error: common::Error) -> Self {
        PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(error.to_string())
    }
}

impl From<common::TypeError> for PyErr {
    fn from(error: common::TypeError) -> Self {
        PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(error.to_string())
    }
}

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

impl From<encoder::PacketError> for PyErr {
    fn from(error: encoder::PacketError) -> Self {
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
    fn event_type(&self) -> PyResult<String> {
        match self.inner {
            Some(ref decoder) => Ok(decoder.event_type.to_string().to_owned()),
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
            let length = packet.len() as numpy::npyffi::npy_intp;
            let array = types::ArrayType::Dat.new_array(python, length);
            unsafe {
                for index in 0..length {
                    let event_cell = types::array_at(python, array, index);
                    std::ptr::copy(
                        &packet[index as usize] as *const common::Event as *const u8,
                        event_cell,
                        std::mem::size_of::<common::Event>(),
                    );
                }
                Ok(Some(PyObject::from_owned_ptr(
                    python,
                    array as *mut pyo3::ffi::PyObject,
                )))
            }
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
    #[pyo3(signature = (path, version, event_type, zero_t0, dimensions))]
    fn new(
        path: &pyo3::Bound<'_, pyo3::types::PyAny>,
        version: &str,
        event_type: &str,
        zero_t0: bool,
        dimensions: Option<(u16, u16)>,
    ) -> PyResult<Self> {
        Python::with_gil(|python| -> PyResult<Self> {
            Ok(Encoder {
                inner: Some(encoder::Encoder::new(
                    types::python_path_to_string(python, path)?,
                    common::Version::from_string(version)?,
                    zero_t0,
                    common::Type::new(event_type, dimensions)?,
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

    fn write(&mut self, packet: &pyo3::Bound<'_, pyo3::types::PyAny>) -> PyResult<()> {
        Python::with_gil(|python| -> PyResult<()> {
            match self.inner.as_mut() {
                Some(encoder) => {
                    let (array, length) =
                        types::check_array(python, types::ArrayType::Dat, packet)?;
                    unsafe {
                        for index in 0..length {
                            let event_cell = types::array_at(python, array, index);
                            encoder.write(*event_cell)?;
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
