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
    array_type: types::ArrayType,
    closed: bool,
}

#[pymethods]
impl Decoder {
    #[new]
    /// With `as_events=True`, packets use faery's event dtype (t, x, y, on)
    /// instead of the raw DAT layout (t, x, y, payload): the payload is written
    /// as 0/1 during decoding, so no conversion is needed afterwards.
    #[pyo3(signature = (path, dimensions_fallback, version_fallback, as_events=false))]
    fn new(
        path: &pyo3::Bound<'_, pyo3::types::PyAny>,
        dimensions_fallback: Option<(u16, u16)>,
        version_fallback: Option<String>,
        as_events: bool,
    ) -> PyResult<Self> {
        Ok(Decoder {
            inner: Some(decoder::Decoder::new(
                types::python_path_to_string(path)?,
                dimensions_fallback,
                version_fallback
                    .map(|version| common::Version::from_string(&version))
                    .transpose()?,
                as_events,
            )?),
            array_type: if as_events {
                types::ArrayType::Dvs
            } else {
                types::ArrayType::Dat
            },
            closed: false,
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

    /// (first_t, last_t) in microseconds, t0 included, or None for an empty
    /// file. Reads only the timestamp words. Consumes the decoder: call it on a
    /// fresh one, not one being iterated.
    fn time_range(&mut self, python: Python<'_>) -> PyResult<Option<(u64, u64)>> {
        match self.inner.take() {
            Some(decoder) => Ok(python.detach(|| decoder.time_range())?),
            None => Err(pyo3::exceptions::PyException::new_err(
                "called time_range after __exit__ or after time_range",
            )),
        }
    }

    fn __enter__(slf: Py<Self>) -> Py<Self> {
        slf
    }

    #[pyo3(signature = (_exception_type, _value, _traceback))]
    fn __exit__(
        &mut self,
        _exception_type: Option<Py<PyAny>>,
        _value: Option<Py<PyAny>>,
        _traceback: Option<Py<PyAny>>,
    ) -> PyResult<bool> {
        if self.closed {
            return Err(pyo3::exceptions::PyException::new_err(
                "multiple calls to __exit__",
            ));
        }
        // time_range() may already have consumed the inner decoder.
        self.closed = true;
        let _ = self.inner.take();
        Ok(false)
    }

    fn __iter__(shell: PyRefMut<Self>) -> PyResult<Py<Decoder>> {
        Ok(shell.into())
    }

    fn __next__(mut shell: PyRefMut<Self>) -> PyResult<Option<Py<PyAny>>> {
        let array_type = shell.array_type;
        let packet = match shell.inner {
            Some(ref mut decoder) => match decoder.next()? {
                Some(result) => result,
                None => return Ok(None),
            },
            None => {
                return Err(pyo3::exceptions::PyException::new_err(
                    "called __next__ after __exit__",
                ))
            }
        };
        Python::attach(|python| -> PyResult<Option<Py<PyAny>>> {
            let length = packet.len() as numpy::npyffi::npy_intp;
            let array = array_type.new_array(python, length);
            if array.is_null() {
                return Err(PyErr::fetch(python));
            }
            unsafe {
                // Both dtypes are packed 13-byte records laid out like
                // common::Event, and a new array is C-contiguous, so the whole
                // packet is one copy rather than a PyArray_GetPtr call per event.
                let record_size =
                    numpy::npyffi::PyDataType_ELSIZE(python, (*array).descr) as usize;
                if record_size != std::mem::size_of::<common::Event>() {
                    pyo3::ffi::Py_DECREF(array as *mut pyo3::ffi::PyObject);
                    return Err(PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(format!(
                        "unexpected DAT record size {record_size} (expected {})",
                        std::mem::size_of::<common::Event>()
                    )));
                }
                std::ptr::copy_nonoverlapping(
                    packet.as_ptr() as *const u8,
                    (*array).data as *mut u8,
                    packet.len() * std::mem::size_of::<common::Event>(),
                );
                Ok(Some(
                    pyo3::Bound::from_owned_ptr(python, array as *mut pyo3::ffi::PyObject).unbind(),
                ))
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
        Ok(Encoder {
            inner: Some(encoder::Encoder::new(
                types::python_path_to_string(path)?,
                common::Version::from_string(version)?,
                zero_t0,
                common::Type::new(event_type, dimensions)?,
            )?),
        })
    }

    fn __enter__(slf: Py<Self>) -> Py<Self> {
        slf
    }

    #[pyo3(signature = (_exception_type, _value, _traceback))]
    fn __exit__(
        &mut self,
        _exception_type: Option<Py<PyAny>>,
        _value: Option<Py<PyAny>>,
        _traceback: Option<Py<PyAny>>,
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
        Python::attach(|python| -> PyResult<()> {
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
