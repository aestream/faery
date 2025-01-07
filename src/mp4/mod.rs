mod mp4;
mod x264;

use numpy::PyArrayMethods;
use numpy::PyUntypedArrayMethods;
use pyo3::prelude::*;

use crate::types;

impl From<x264::Error> for PyErr {
    fn from(error: x264::Error) -> Self {
        PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(error.to_string())
    }
}

impl From<mp4::Error> for PyErr {
    fn from(error: mp4::Error) -> Self {
        PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(error.to_string())
    }
}

#[pyclass]
pub struct Encoder {
    inner: Option<mp4::Encoder<std::io::BufWriter<std::fs::File>>>,
}

#[pymethods]
impl Encoder {
    #[new]
    fn new(
        path: &pyo3::Bound<'_, pyo3::types::PyAny>,
        dimensions: (u16, u16),
        frame_rate: f64,
        crf: f32,
        preset: &str,
        tune: &str,
        profile: &str,
    ) -> PyResult<Self> {
        Python::with_gil(|python| -> PyResult<Self> {
            Ok(Encoder {
                inner: Some(mp4::Encoder::from_parameters_and_path(
                    x264::Parameters {
                        preset: x264::Preset::from_string(preset)?,
                        tune: x264::Tune::from_string(tune)?,
                        profile: x264::Profile::from_string(profile)?,
                        crf,
                        width: dimensions.0,
                        height: dimensions.1,
                        use_opencl: false,
                        frame_rate,
                    },
                    types::python_path_to_string(python, path)?,
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
        match self.inner.as_mut() {
            Some(inner) => {
                let _ = inner.finalize();
            }
            None => {
                return Err(pyo3::exceptions::PyException::new_err(
                    "multiple calls to __exit__",
                ));
            }
        }
        let _ = self.inner.take();
        Ok(false)
    }

    fn write(&mut self, frame: &pyo3::Bound<'_, numpy::PyArray3<u8>>) -> PyResult<()> {
        if !frame.is_contiguous() {
            return Err(pyo3::exceptions::PyAttributeError::new_err(format!(
                "the frame's memory must be contiguous"
            )));
        }
        let readonly_frame = frame.readonly();
        let array = readonly_frame.as_array();
        let dimensions = array.dim();
        if dimensions.2 != 3 && dimensions.2 != 4 {
            return Err(pyo3::exceptions::PyAttributeError::new_err(format!(
                "expected an array whose last dimension is 3 (RGB) or 4 (RGBA) (got a {} x {} x {} array)",
                dimensions.0, dimensions.1, dimensions.2,
            )));
        }

        match self.inner.as_mut() {
            Some(encoder) => {
                if dimensions.2 == 3 {
                    encoder.push_rgb(x264::RGBFrame::new(
                        dimensions.1 as u16,
                        dimensions.0 as u16,
                        // unsafe: the array is contiguous
                        unsafe {
                            std::slice::from_raw_parts(
                                array.as_ptr(),
                                dimensions.0 * dimensions.1 * 3,
                            )
                        },
                    )?)?;
                } else {
                    assert_eq!(dimensions.2, 4, "the last dimension is 4");
                    encoder.push_rgba(x264::RGBAFrame::new(
                        dimensions.1 as u16,
                        dimensions.0 as u16,
                        // unsafe: the array is contiguous
                        unsafe {
                            std::slice::from_raw_parts(
                                array.as_ptr() as *const u32,
                                dimensions.0 * dimensions.1,
                            )
                        },
                    )?)?;
                }
                Ok(())
            }
            None => Err(pyo3::exceptions::PyException::new_err(
                "write called after __exit__",
            )),
        }
    }
}
