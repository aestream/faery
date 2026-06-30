use pyo3::prelude::*;

use crate::types;

trait Rasterize: numpy::Element + Copy + Default {
    fn inc(value: &mut Self);
}

impl Rasterize for u16 {
    #[inline(always)]
    fn inc(value: &mut Self) {
        *value = value.saturating_add(1);
    }
}

impl Rasterize for u32 {
    #[inline(always)]
    fn inc(value: &mut Self) {
        *value = value.saturating_add(1);
    }
}

impl Rasterize for f32 {
    #[inline(always)]
    fn inc(value: &mut Self) {
        *value += 1.0;
    }
}

#[pyfunction]
#[pyo3(signature = (events, width, height, dtype="u16"))]
pub fn rasterize_to_frame(
    events: &pyo3::Bound<'_, pyo3::types::PyAny>,
    width: u16,
    height: u16,
    dtype: &str,
) -> PyResult<PyObject> {
    Python::with_gil(|python| -> PyResult<PyObject> {
        let (array, length) = types::check_array(python, types::ArrayType::Dvs, events)?;
        match dtype {
            "u16" => rasterize_typed::<u16>(python, array, length, width, height),
            "u32" => rasterize_typed::<u32>(python, array, length, width, height),
            "f32" => rasterize_typed::<f32>(python, array, length, width, height),
            other => Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(format!(
                "unsupported dtype \"{}\" (expected \"u16\", \"u32\", or \"f32\")",
                other
            ))),
        }
    })
}

fn rasterize_typed<T: Rasterize>(
    python: Python<'_>,
    array: *mut numpy::npyffi::PyArrayObject,
    length: numpy::npyffi::npy_intp,
    width: u16,
    height: u16,
) -> PyResult<PyObject> {
    if width == 0 || height == 0 {
        return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
            "width and height must be greater than zero",
        ));
    }
    let mut dimensions = [
        2 as numpy::npyffi::npy_intp,
        height as numpy::npyffi::npy_intp,
        width as numpy::npyffi::npy_intp,
    ];
    let mut zero_index = [0 as numpy::npyffi::npy_intp; 3];
    let plane_stride = width as usize * height as usize;
    let total_elements = 2 * plane_stride;
    unsafe {
        let frame = numpy::PY_ARRAY_API.PyArray_Empty(
            python,
            3,
            dimensions.as_mut_ptr(),
            T::get_dtype(python).into_ptr() as *mut numpy::npyffi::PyArray_Descr,
            0,
        ) as *mut numpy::npyffi::PyArrayObject;
        if frame.is_null() {
            return Err(PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(
                "PyArray_Empty returned null",
            ));
        }
        let data = numpy::PY_ARRAY_API.PyArray_GetPtr(python, frame, zero_index.as_mut_ptr())
            as *mut T;
        std::ptr::write_bytes(data as *mut u8, 0, total_elements * std::mem::size_of::<T>());
        // Walk the structured array by pointer arithmetic instead of calling
        // PyArray_GetPtr per event — saves a function call per event in the hot loop.
        let base = (*array).data as *const u8;
        let stride = *((*array).strides) as usize;
        for index in 0..length {
            let event = base.add(index as usize * stride)
                as *const neuromorphic_types::PolarityEvent<u64, u16, u16>;
            let x = std::ptr::read_unaligned(std::ptr::addr_of!((*event).x));
            let y = std::ptr::read_unaligned(std::ptr::addr_of!((*event).y));
            let polarity = std::ptr::read_unaligned(std::ptr::addr_of!((*event).polarity));
            if x >= width {
                pyo3::ffi::Py_DECREF(frame as *mut pyo3::ffi::PyObject);
                return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(format!(
                    "event x ({}) is out of bounds (width = {})",
                    x, width
                )));
            }
            if y >= height {
                pyo3::ffi::Py_DECREF(frame as *mut pyo3::ffi::PyObject);
                return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(format!(
                    "event y ({}) is out of bounds (height = {})",
                    y, height
                )));
            }
            let p = polarity as usize;
            let offset = p * plane_stride + (y as usize) * (width as usize) + (x as usize);
            T::inc(&mut *data.add(offset));
        }
        Ok(PyObject::from_owned_ptr(
            python,
            frame as *mut pyo3::ffi::PyObject,
        ))
    }
}
