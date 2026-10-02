use pyo3::prelude::*;

use crate::types;

trait Rasterize: numpy::Element + Copy + Default {
    fn inc(value: &mut Self);
}

impl Rasterize for u16 {
    #[inline(always)]
    fn inc(value: &mut Self) {
        // A branch rather than saturating_add: saturation almost never
        // happens, so the branch is free, whereas the branchless select
        // LLVM emits for saturating_add made the u16 scatter 2.4x slower.
        if *value != u16::MAX {
            *value += 1;
        }
    }
}

impl Rasterize for u32 {
    #[inline(always)]
    fn inc(value: &mut Self) {
        // See u16.
        if *value != u32::MAX {
            *value += 1;
        }
    }
}

impl Rasterize for f32 {
    #[inline(always)]
    fn inc(value: &mut Self) {
        *value += 1.0;
    }
}

#[pyfunction]
#[pyo3(signature = (events, width, height, dtype="u16", out=None))]
pub fn rasterize_to_frame(
    events: &pyo3::Bound<'_, pyo3::types::PyAny>,
    width: u16,
    height: u16,
    dtype: &str,
    out: Option<&pyo3::Bound<'_, pyo3::types::PyAny>>,
) -> PyResult<Py<PyAny>> {
    let python = events.py();
    let (array, length) = types::check_array(python, types::ArrayType::Dvs, events)?;
    match dtype {
        "u16" => rasterize_typed::<u16>(python, array, length, width, height, out),
        "u32" => rasterize_typed::<u32>(python, array, length, width, height, out),
        "f32" => rasterize_typed::<f32>(python, array, length, width, height, out),
        other => Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(format!(
            "unsupported dtype \"{}\" (expected \"u16\", \"u32\", or \"f32\")",
            other
        ))),
    }
}

/// Calls `f(index, linear)` for every event, where `linear` is
/// `p * height * width + y * width + x` (p = 0 for OFF, 1 for ON).
///
/// # Safety
/// `array` and `length` must come from `types::check_array` with `ArrayType::Dvs`.
unsafe fn for_each_event(
    array: *mut numpy::npyffi::PyArrayObject,
    length: numpy::npyffi::npy_intp,
    width: u16,
    height: u16,
    mut f: impl FnMut(usize, usize),
) -> PyResult<()> {
    let plane_stride = width as usize * height as usize;
    // Walk the structured array by pointer arithmetic instead of calling
    // PyArray_GetPtr per event. The stride is signed: views such as
    // events[::-1] have negative strides.
    let base = (*array).data as *const u8;
    let stride = *((*array).strides) as isize;
    for index in 0..length as isize {
        let event =
            base.offset(index * stride) as *const neuromorphic_types::PolarityEvent<u64, u16, u16>;
        let x = std::ptr::read_unaligned(std::ptr::addr_of!((*event).x));
        let y = std::ptr::read_unaligned(std::ptr::addr_of!((*event).y));
        // Read the raw byte rather than the Polarity enum: numpy bools can
        // hold any byte (e.g. via .view()), and only 0 and 1 are valid enum
        // values or safe plane indices.
        let polarity = std::ptr::read_unaligned(std::ptr::addr_of!((*event).polarity) as *const u8);
        if x >= width {
            return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(format!(
                "event x ({}) is out of bounds (width = {})",
                x, width
            )));
        }
        if y >= height {
            return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(format!(
                "event y ({}) is out of bounds (height = {})",
                y, height
            )));
        }
        let p = (polarity != 0) as usize;
        f(
            index as usize,
            p * plane_stride + (y as usize) * (width as usize) + (x as usize),
        );
    }
    Ok(())
}

fn check_dimensions(width: u16, height: u16) -> PyResult<()> {
    if width == 0 || height == 0 {
        return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
            "width and height must be greater than zero",
        ));
    }
    Ok(())
}

/// Allocates a numpy array, zeroed or not, and takes ownership of it.
unsafe fn new_array<'py, T: numpy::Element>(
    python: Python<'py>,
    dimensions: &mut [numpy::npyffi::npy_intp],
    zeroed: bool,
) -> PyResult<(pyo3::Bound<'py, PyAny>, *mut T)> {
    // PyArray_Empty steals the descriptor reference.
    let descriptor = T::get_dtype(python).into_ptr() as *mut numpy::npyffi::PyArray_Descr;
    let array = numpy::PY_ARRAY_API.PyArray_Empty(
        python,
        dimensions.len() as i32,
        dimensions.as_mut_ptr(),
        descriptor,
        0,
    );
    if array.is_null() {
        return Err(PyErr::fetch(python));
    }
    // Owning the array here releases it on every early return.
    let array = pyo3::Bound::from_owned_ptr(python, array);
    let array_object = array.as_ptr() as *mut numpy::npyffi::PyArrayObject;
    let data = (*array_object).data as *mut T;
    if zeroed {
        // Not PyArray_Zeros: for multi-MB frames calloc returns fresh pages
        // that fault in one by one during the random scatter (2x slower per
        // packet in benchmarks/test_handoff.py). memset on malloc'd memory
        // reuses pages that are already mapped.
        let elements: numpy::npyffi::npy_intp = dimensions.iter().product();
        std::ptr::write_bytes(data, 0, elements as usize);
    }
    Ok((array, data))
}

fn rasterize_typed<T: Rasterize>(
    python: Python<'_>,
    array: *mut numpy::npyffi::PyArrayObject,
    length: numpy::npyffi::npy_intp,
    width: u16,
    height: u16,
    out: Option<&pyo3::Bound<'_, pyo3::types::PyAny>>,
) -> PyResult<Py<PyAny>> {
    check_dimensions(width, height)?;
    if let Some(out) = out {
        return rasterize_into::<T>(array, length, width, height, out);
    }
    let mut dimensions = [
        2 as numpy::npyffi::npy_intp,
        height as numpy::npyffi::npy_intp,
        width as numpy::npyffi::npy_intp,
    ];
    unsafe {
        let (frame, data) = new_array::<T>(python, &mut dimensions, true)?;
        for_each_event(array, length, width, height, |_, linear| {
            T::inc(&mut *data.add(linear));
        })?;
        Ok(frame.unbind())
    }
}

/// Zeroes `out` (a writeable, C-contiguous (2, height, width) array of
/// dtype T) and rasterizes into it. Reusing `out` skips the per-packet
/// allocation, and lets the caller place the frame in pinned memory.
fn rasterize_into<T: Rasterize>(
    array: *mut numpy::npyffi::PyArrayObject,
    length: numpy::npyffi::npy_intp,
    width: u16,
    height: u16,
    out: &pyo3::Bound<'_, pyo3::types::PyAny>,
) -> PyResult<Py<PyAny>> {
    use numpy::{PyArrayMethods, PyUntypedArrayMethods};
    let invalid = || {
        PyErr::new::<pyo3::exceptions::PyValueError, _>(format!(
            "out must be a writeable, C-contiguous numpy array with shape (2, {}, {}) and dtype {}",
            height,
            width,
            T::get_dtype(out.py()).str().map(|s| s.to_string()).unwrap_or_default()
        ))
    };
    let frame = out
        .cast::<numpy::PyArray3<T>>()
        .map_err(|_| invalid())?;
    // as_slice_mut accepts Fortran order too; the flat indices assume C order.
    if frame.shape() != [2, height as usize, width as usize] || !frame.is_c_contiguous() {
        return Err(invalid());
    }
    let mut frame_readwrite = frame.try_readwrite().map_err(|_| invalid())?;
    let data = frame_readwrite.as_slice_mut().map_err(|_| invalid())?;
    data.fill(T::default());
    let data = data.as_mut_ptr();
    unsafe {
        for_each_event(array, length, width, height, |_, linear| {
            T::inc(&mut *data.add(linear));
        })?;
    }
    drop(frame_readwrite);
    Ok(out.clone().unbind())
}

/// Returns a 1-D int32 array with one flat index per event into a
/// (2, height, width) frame: p * height * width + y * width + x.
///
/// A consumer rebuilds the frame with a single scatter, e.g.
/// `torch.zeros(2 * height * width).index_add_(0, indices, ones).view(2, height, width)`.
#[pyfunction]
pub fn linear_indices(
    events: &pyo3::Bound<'_, pyo3::types::PyAny>,
    width: u16,
    height: u16,
) -> PyResult<Py<PyAny>> {
    check_dimensions(width, height)?;
    if 2 * width as u64 * height as u64 > i32::MAX as u64 {
        return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(format!(
            "2 x {} x {} frame elements do not fit in int32 indices",
            width, height
        )));
    }
    let python = events.py();
    let (array, length) = types::check_array(python, types::ArrayType::Dvs, events)?;
    let mut dimensions = [length];
    unsafe {
        let (indices, data) = new_array::<i32>(python, &mut dimensions, false)?;
        for_each_event(array, length, width, height, |index, linear| {
            *data.add(index) = linear as i32;
        })?;
        Ok(indices.unbind())
    }
}
