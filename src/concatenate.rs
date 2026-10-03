use numpy::PyArrayDescrMethods;
use pyo3::prelude::*;

/// One part, reduced to plain values so the copy can run with the GIL
/// released. The caller keeps a reference to the numpy array meanwhile.
#[derive(Clone, Copy)]
struct Part {
    base: *const u8,
    stride: isize,
    length: usize,
}

// Safety: parts are only read during the copy, while `concatenate` holds a
// reference to each array.
unsafe impl Send for Part {}

#[derive(Clone, Copy)]
struct Target(*mut u8);

// Safety: the output array is new, so the copy is its only user.
unsafe impl Send for Target {}

impl Target {
    // A method rather than `.0`: closures capture disjoint fields, and
    // capturing `.0` would capture the bare (non-Send) pointer.
    #[inline(always)]
    fn get(self) -> *mut u8 {
        self.0
    }
}

/// Concatenates one-dimensional numpy arrays of the structured `dtype` into a
/// new contiguous array.
///
/// numpy before 2.5 copies structured records field by field (about 8x
/// slower than memcpy); this copies contiguous parts with one memcpy each,
/// with the GIL released.
///
/// Returns None (the caller falls back to numpy.concatenate) if a part is not
/// a one-dimensional numpy array with a dtype equivalent to `dtype`, or if
/// `dtype` holds Python objects.
#[pyfunction]
pub fn concatenate<'py>(
    parts: &pyo3::Bound<'py, PyAny>,
    dtype: &pyo3::Bound<'py, PyAny>,
) -> PyResult<Option<pyo3::Bound<'py, PyAny>>> {
    let python = parts.py();
    let mut descriptor: *mut numpy::npyffi::PyArray_Descr = std::ptr::null_mut();
    if unsafe {
        numpy::PY_ARRAY_API.PyArray_DescrConverter(python, dtype.as_ptr(), &mut descriptor)
    } < 0
    {
        return Err(PyErr::fetch(python));
    }
    // Owning the descriptor here releases it on every early return.
    let descriptor_object =
        unsafe { pyo3::Bound::from_owned_ptr(python, descriptor as *mut pyo3::ffi::PyObject) };
    if descriptor_object.cast::<numpy::PyArrayDescr>()?.has_object() {
        return Ok(None);
    }
    let item_size = unsafe { numpy::npyffi::PyDataType_ELSIZE(python, descriptor) } as usize;

    // Holding a reference to every part keeps them alive while the GIL is
    // released, even if another thread changes `parts` meanwhile.
    let mut arrays = Vec::new();
    let mut views = Vec::new();
    let mut total: usize = 0;
    for part in parts.try_iter()? {
        let part = part?;
        if unsafe { numpy::npyffi::array::PyArray_Check(python, part.as_ptr()) } == 0 {
            return Ok(None);
        }
        let array = part.as_ptr() as *mut numpy::npyffi::PyArrayObject;
        if unsafe { (*array).nd } != 1
            || unsafe {
                numpy::PY_ARRAY_API.PyArray_EquivTypes(python, (*array).descr, descriptor)
            } == 0
            || unsafe { numpy::npyffi::PyDataType_ELSIZE(python, (*array).descr) } as usize
                != item_size
        {
            return Ok(None);
        }
        let length = unsafe { *(*array).dimensions } as usize;
        views.push(Part {
            base: unsafe { (*array).data } as *const u8,
            // Signed: views such as events[::-1] have negative strides.
            stride: unsafe { *(*array).strides } as isize,
            length,
        });
        total += length;
        arrays.push(part);
    }

    // PyArray_NewFromDescr steals the descriptor reference.
    let mut dimension = total as numpy::npyffi::npy_intp;
    let output = unsafe {
        numpy::PY_ARRAY_API.PyArray_NewFromDescr(
            python,
            numpy::npyffi::get_type_object(python, numpy::npyffi::NpyTypes::PyArray_Type),
            descriptor_object.clone().into_ptr() as *mut numpy::npyffi::PyArray_Descr,
            1,
            &mut dimension,
            std::ptr::null_mut(),
            std::ptr::null_mut(),
            0,
            std::ptr::null_mut(),
        )
    };
    if output.is_null() {
        return Err(PyErr::fetch(python));
    }
    let output = unsafe { pyo3::Bound::from_owned_ptr(python, output) };
    let target = Target(unsafe { (*(output.as_ptr() as *mut numpy::npyffi::PyArrayObject)).data }
        as *mut u8);
    python.detach(move || {
        let mut offset = 0;
        for part in views {
            let destination = unsafe { target.get().add(offset * item_size) };
            if part.stride == item_size as isize {
                unsafe {
                    std::ptr::copy_nonoverlapping(part.base, destination, part.length * item_size)
                };
            } else {
                for index in 0..part.length {
                    unsafe {
                        std::ptr::copy_nonoverlapping(
                            part.base.offset(index as isize * part.stride),
                            destination.add(index * item_size),
                            item_size,
                        )
                    };
                }
            }
            offset += part.length;
        }
    });
    drop(arrays);
    Ok(Some(output))
}
