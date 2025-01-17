mod decoder;
mod encoder;

use crate::types;

use pyo3::prelude::*;

impl From<decoder::Error> for PyErr {
    fn from(error: decoder::Error) -> Self {
        PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(error.to_string())
    }
}

impl From<decoder::ReadError> for PyErr {
    fn from(error: decoder::ReadError) -> Self {
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
    fn new(
        path: &pyo3::Bound<'_, pyo3::types::PyAny>,
        dimensions: (u16, u16),
        has_header: bool,
        separator: u8,
        t_index: usize,
        x_index: usize,
        y_index: usize,
        on_index: usize,
        t_scale: f64,
        t0: u64,
        on_value: Vec<u8>,
        off_value: Vec<u8>,
        skip_errors: bool,
    ) -> PyResult<Self> {
        Python::with_gil(|python| -> PyResult<Self> {
            Ok(Decoder {
                inner: Some(decoder::Decoder::new(
                    if path.is_none() {
                        decoder::Input::stdin()
                    } else {
                        decoder::Input::File(std::io::BufReader::new(std::fs::File::open(
                            types::python_path_to_string(python, path)?,
                        )?))
                    },
                    decoder::Properties::new(
                        dimensions, has_header, separator, t_index, x_index, y_index, on_index,
                        t_scale, t0, on_value, off_value,
                    )?,
                    skip_errors,
                )?),
            })
        })
    }

    #[getter]
    fn dimensions(&self) -> PyResult<(u16, u16)> {
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
        let events = match shell.inner {
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
    fn new(
        path: &pyo3::Bound<'_, pyo3::types::PyAny>,
        separator: u8,
        header: bool,
        dimensions: (u16, u16),
    ) -> PyResult<Self> {
        Python::with_gil(|python| -> PyResult<Self> {
            Ok(Encoder {
                inner: Some(encoder::Encoder::new(
                    if path.is_none() {
                        encoder::Output::stdout()
                    } else {
                        encoder::Output::File(std::io::BufWriter::new(std::fs::File::create(
                            types::python_path_to_string(python, path)?,
                        )?))
                    },
                    separator,
                    header,
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

    fn write(&mut self, events: &pyo3::Bound<'_, pyo3::types::PyAny>) -> PyResult<()> {
        Python::with_gil(|python| -> PyResult<()> {
            match self.inner.as_mut() {
                Some(encoder) => {
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
                None => Err(pyo3::exceptions::PyException::new_err(
                    "write called after __exit__",
                )),
            }
        })
    }
}
