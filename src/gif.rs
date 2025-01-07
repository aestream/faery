use numpy::PyArrayMethods;
use numpy::PyUntypedArrayMethods;
use pyo3::prelude::*;

use crate::types;

struct Inner {
    frame_index: usize,
    frame_rate: f64,
    collector: Option<gifski::Collector>,
    writing_thread: Option<std::thread::JoinHandle<()>>,
    error: std::sync::Arc<std::sync::Mutex<Option<gifski::Error>>>,
}

#[pyclass]
pub struct Encoder {
    inner: Option<Inner>,
}

#[pymethods]
impl Encoder {
    #[new]
    fn new(
        path: &pyo3::Bound<'_, pyo3::types::PyAny>,
        dimensions: (u16, u16),
        frame_rate: f64,
        quality: u8,
        fast: bool,
    ) -> PyResult<Self> {
        let path = Python::with_gil(|python| -> PyResult<String> {
            types::python_path_to_string(python, path)
        })?;
        let (collector, writer) = gifski::new(gifski::Settings {
            width: Some(dimensions.0 as u32),
            height: Some(dimensions.1 as u32),
            quality,
            fast,
            repeat: gifski::Repeat::Infinite,
        })
        .map_err(|error| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(error.to_string()))?;
        let file = std::fs::File::create(path).map_err(|error| {
            PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(error.to_string())
        })?;
        let error = std::sync::Arc::new(std::sync::Mutex::new(None));
        let thread_error = error.clone();
        Ok(Encoder {
            inner: Some(Inner {
                frame_index: 0,
                frame_rate,
                collector: Some(collector),
                writing_thread: Some(std::thread::spawn(move || {
                    if let Err(error) = writer.write(file, &mut gifski::progress::NoProgress {}) {
                        thread_error
                            .lock()
                            .expect("thread_error is not poisoned")
                            .replace(error);
                    }
                })),
                error,
            }),
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
                let _ = inner.collector.take();
                if let Some(writing_thread) = inner.writing_thread.take() {
                    let _ = writing_thread.join();
                }
                if let Some(error) = inner.error.lock().expect("error is not poisoned").take() {
                    return Err(pyo3::exceptions::PyException::new_err(error.to_string()));
                }
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
            Some(inner) => {
                if let Some(error) = inner.error.lock().expect("error is not poisoned").take() {
                    return Err(pyo3::exceptions::PyException::new_err(error.to_string()));
                }
                match inner.collector.as_mut() {
                    Some(collector) => {
                        let array_slice = array.as_slice().expect("the frame is contiguous");
                        collector
                            .add_frame_rgba(
                                inner.frame_index,
                                gifski::collector::ImgVec::new(
                                    if dimensions.2 == 3 {
                                        array_slice
                                            .chunks_exact(3)
                                            .map(|chunk| gifski::collector::RGBA8 {
                                                r: chunk[0],
                                                g: chunk[1],
                                                b: chunk[2],
                                                a: 0xFFu8,
                                            })
                                            .collect()
                                    } else {
                                        array_slice
                                            .chunks_exact(4)
                                            .map(|chunk| gifski::collector::RGBA8 {
                                                r: chunk[0],
                                                g: chunk[1],
                                                b: chunk[2],
                                                a: chunk[3],
                                            })
                                            .collect()
                                    },
                                    dimensions.1,
                                    dimensions.0,
                                ),
                                inner.frame_index as f64 * (1.0 / inner.frame_rate),
                            )
                            .map_err(|error| {
                                PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(error.to_string())
                            })
                    }
                    None => Err(pyo3::exceptions::PyException::new_err("collector is None")),
                }
            }
            None => Err(pyo3::exceptions::PyException::new_err(
                "write called after __exit__",
            )),
        }
    }
}
