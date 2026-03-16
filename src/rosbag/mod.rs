mod decoder;

use crate::types;

use pyo3::prelude::*;

impl From<decoder::Error> for PyErr {
    fn from(error: decoder::Error) -> Self {
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
    #[pyo3(signature = (path))]
    fn new(path: &pyo3::Bound<'_, pyo3::types::PyAny>) -> PyResult<Self> {
        Ok(Decoder {
            inner: Some(decoder::Decoder::new(types::python_path_to_string(path)?)?),
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

    /*
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
    */
}
