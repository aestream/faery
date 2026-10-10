use pyo3::prelude::*;

use crate::types;

trait Rasterize: numpy::Element + Copy + Default + Send {
    fn inc(value: &mut Self);
}

impl Rasterize for u8 {
    #[inline(always)]
    fn inc(value: &mut Self) {
        // See u16.
        if *value != u8::MAX {
            *value += 1;
        }
    }
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
        "u8" => rasterize_typed::<u8>(python, array, length, width, height, out),
        "u16" => rasterize_typed::<u16>(python, array, length, width, height, out),
        "u32" => rasterize_typed::<u32>(python, array, length, width, height, out),
        "f32" => rasterize_typed::<f32>(python, array, length, width, height, out),
        other => Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(format!(
            "unsupported dtype \"{}\" (expected \"u8\", \"u16\", \"u32\", or \"f32\")",
            other
        ))),
    }
}

/// A structured DVS event array, reduced to what the walk needs.
///
/// The fields are plain values (no Python references), so the walk can run
/// with the GIL released. The caller keeps the numpy array alive meanwhile.
#[derive(Clone, Copy)]
struct Events {
    base: *const u8,
    stride: isize,
    length: isize,
}

// Safety: the pointer is only read during a walk, while the caller holds a
// reference to the array.
unsafe impl Send for Events {}

impl Events {
    /// # Safety
    /// `array` and `length` must come from `types::check_array` with `ArrayType::Dvs`.
    unsafe fn new(array: *mut numpy::npyffi::PyArrayObject, length: numpy::npyffi::npy_intp) -> Self {
        Events {
            base: (*array).data as *const u8,
            // Signed: views such as events[::-1] have negative strides.
            stride: *((*array).strides) as isize,
            length: length as isize,
        }
    }
}

/// A raw output pointer that can cross into `Python::detach`.
#[derive(Clone, Copy)]
struct SendPtr<T>(*mut T);

// Safety: each array behind a SendPtr is written by one walk at a time.
unsafe impl<T> Send for SendPtr<T> {}

impl<T> SendPtr<T> {
    // A method rather than `.0`: closures capture disjoint fields, and
    // capturing `.0` would capture the bare (non-Send) pointer.
    #[inline(always)]
    fn get(self) -> *mut T {
        self.0
    }
}

/// Calls `f(index, linear)` for every event, where `linear` is
/// `p * height * width + y * width + x` (p = 0 for OFF, 1 for ON).
///
/// Touches no Python objects, so callers run it inside `Python::detach`:
/// another thread (e.g. a consumer feeding a GPU) runs meanwhile.
///
/// # Safety
/// `events` must describe a live DVS array (see `Events::new`).
unsafe fn for_each_event(
    events: Events,
    width: u16,
    height: u16,
    mut f: impl FnMut(usize, usize),
) -> PyResult<()> {
    let plane_stride = width as usize * height as usize;
    // Walk the structured array by pointer arithmetic instead of calling
    // PyArray_GetPtr per event.
    for index in 0..events.length {
        let event = events.base.offset(index * events.stride)
            as *const neuromorphic_types::PolarityEvent<u64, u16, u16>;
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
        let elements = dimensions.iter().product::<numpy::npyffi::npy_intp>() as usize;
        let target = SendPtr(data);
        python.detach(move || std::ptr::write_bytes(target.get(), 0, elements));
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
        let events = Events::new(array, length);
        let data = SendPtr(data);
        python.detach(move || {
            for_each_event(events, width, height, |_, linear| {
                T::inc(&mut *data.get().add(linear));
            })
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
    let events = unsafe { Events::new(array, length) };
    // frame_readwrite stays borrowed across the detach, so no other
    // rust-numpy borrow of `out` can start while the walk writes it.
    out.py().detach(move || {
        data.fill(T::default());
        let data = SendPtr(data.as_mut_ptr());
        unsafe {
            for_each_event(events, width, height, |_, linear| {
                T::inc(&mut *data.get().add(linear));
            })
        }
    })?;
    drop(frame_readwrite);
    Ok(out.clone().unbind())
}

/// Returns a 1-D int32 array with one flat index per event into a
/// (2, height, width) frame: p * height * width + y * width + x.
///
/// A consumer rebuilds the frame with a single scatter, e.g.
/// `torch.zeros(2 * height * width).index_add_(0, indices, ones).view(2, height, width)`.
///
/// With `out` (a writeable, C-contiguous 1-D int32 array with room for every
/// event), the indices are written to its start and `out[:len(events)]` is
/// returned instead of a new array.
///
/// `frame` adds `frame * 2 * height * width` to every index: the indices then
/// point into frame `frame` of a flattened `(frames, 2, height, width)` stack,
/// so one scatter can build several frames.
#[pyfunction]
#[pyo3(signature = (events, width, height, out=None, frame=0))]
pub fn linear_indices(
    events: &pyo3::Bound<'_, pyo3::types::PyAny>,
    width: u16,
    height: u16,
    out: Option<&pyo3::Bound<'_, pyo3::types::PyAny>>,
    frame: u32,
) -> PyResult<Py<PyAny>> {
    check_dimensions(width, height)?;
    let frame_size = 2 * width as u64 * height as u64;
    if (frame as u64 + 1) * frame_size > i32::MAX as u64 + 1 {
        return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(format!(
            "{} frames of 2 x {} x {} elements do not fit in int32 indices",
            frame as u64 + 1,
            width,
            height
        )));
    }
    let base = (frame as u64 * frame_size) as usize;
    let python = events.py();
    let (array, length) = types::check_array(python, types::ArrayType::Dvs, events)?;
    if let Some(out) = out {
        return linear_indices_into(array, length, width, height, base, out);
    }
    let mut dimensions = [length];
    unsafe {
        let (indices, data) = new_array::<i32>(python, &mut dimensions, false)?;
        let events = Events::new(array, length);
        let data = SendPtr(data);
        python.detach(move || {
            for_each_event(events, width, height, |index, linear| {
                *data.get().add(index) = (base + linear) as i32;
            })
        })?;
        Ok(indices.unbind())
    }
}

fn linear_indices_into(
    array: *mut numpy::npyffi::PyArrayObject,
    length: numpy::npyffi::npy_intp,
    width: u16,
    height: u16,
    base: usize,
    out: &pyo3::Bound<'_, pyo3::types::PyAny>,
) -> PyResult<Py<PyAny>> {
    use numpy::{PyArrayMethods, PyUntypedArrayMethods};
    let invalid = || {
        PyErr::new::<pyo3::exceptions::PyValueError, _>(format!(
            "out must be a writeable, C-contiguous 1-D numpy array with dtype int32 and at least {} elements",
            length
        ))
    };
    let indices = out
        .cast::<numpy::PyArray1<i32>>()
        .map_err(|_| invalid())?;
    if indices.len() < length as usize || !indices.is_c_contiguous() {
        return Err(invalid());
    }
    let mut indices_readwrite = indices.try_readwrite().map_err(|_| invalid())?;
    let data = indices_readwrite.as_slice_mut().map_err(|_| invalid())?;
    let events = unsafe { Events::new(array, length) };
    out.py().detach(move || {
        let data = SendPtr(data.as_mut_ptr());
        unsafe {
            for_each_event(events, width, height, |index, linear| {
                *data.get().add(index) = (base + linear) as i32;
            })
        }
    })?;
    drop(indices_readwrite);
    Ok(out
        .get_item(pyo3::types::PySlice::new(out.py(), 0, length as isize, 1))?
        .unbind())
}
