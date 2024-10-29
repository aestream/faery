use numpy::prelude::*;
use numpy::Element;
use pyo3::prelude::*;

pub fn python_path_to_string(
    python: Python,
    path: &pyo3::Bound<'_, pyo3::types::PyAny>,
) -> PyResult<String> {
    if let Ok(result) = path.downcast::<pyo3::types::PyString>() {
        return Ok(result.to_string());
    }
    if let Ok(result) = path.downcast::<pyo3::types::PyBytes>() {
        return Ok(result.to_string());
    }
    let fspath_result = path.to_object(python).call_method0(python, "__fspath__")?;
    {
        let fspath_as_string: Result<
            &pyo3::Bound<'_, pyo3::types::PyString>,
            pyo3::DowncastError<'_, '_>,
        > = fspath_result.downcast_bound(python);
        if let Ok(result) = fspath_as_string {
            return Ok(result.to_string());
        }
    }
    let fspath_as_bytes: &pyo3::Bound<'_, pyo3::types::PyBytes> = fspath_result
        .downcast_bound(python)
        .map_err(|__fspath__| pyo3::exceptions::PyTypeError::new_err("path must be a string, bytes, or an object with an __fspath__ method (such as pathlib.Path"))?;
    Ok(fspath_as_bytes.to_string())
}

#[derive(thiserror::Error, Debug)]
pub enum CheckArrayError {
    #[error("the object is not a numpy array")]
    PyArrayCheck,

    #[error("expected a one-dimensional array (got a {0} array)")]
    Dimensions(String),

    #[error("the array is not structured (https://numpy.org/doc/stable/user/basics.rec.html)")]
    NotStructured,

    #[error("the array must have a field \"{0}\"")]
    MissingField(String),

    #[error("the field \"{name}\" must have the type \"{expected_type}\" (got \"{actual_type}\")")]
    Field {
        name: String,
        expected_type: String,
        actual_type: String,
    },

    #[error(
        "the field \"{name}\" must have the offset \"{expected_offset}\" (got \"{actual_offset}\")"
    )]
    FieldOffset {
        name: String,
        expected_offset: core::ffi::c_long,
        actual_offset: core::ffi::c_long,
    },

    #[error("the array has extra fields (expected {expected}, got {actual})")]
    ExtraFields { expected: String, actual: String },
}

impl Into<PyErr> for CheckArrayError {
    fn into(self) -> PyErr {
        PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(self.to_string())
    }
}

pub fn check_array(
    python: Python,
    array_type: ArrayType,
    object: &pyo3::Bound<'_, pyo3::types::PyAny>,
) -> PyResult<(*mut numpy::npyffi::PyArrayObject, numpy::npyffi::npy_intp)> {
    if unsafe { numpy::npyffi::array::PyArray_Check(python, object.as_ptr()) } == 0 {
        return Err(CheckArrayError::PyArrayCheck.into());
    }
    let array = object.as_ptr() as *mut numpy::npyffi::PyArrayObject;
    let dimensions_length = unsafe { (*array).nd };
    if dimensions_length != 1 {
        let mut dimensions = String::new();
        for dimension in 0..dimensions_length {
            use std::fmt::Write;
            write!(
                dimensions,
                "{}{}",
                unsafe { *((*array).dimensions.offset(dimension as isize)) },
                if dimension < dimensions_length - 1 {
                    "x"
                } else {
                    ""
                }
            )
            .expect("write! did not fail");
        }
        return Err(CheckArrayError::Dimensions(dimensions).into());
    }
    let fields = unsafe { numpy::npyffi::PyDataType_FIELDS(python, (*array).descr) };
    if unsafe { pyo3::ffi::PyMapping_Check(fields) } == 0 {
        return Err(CheckArrayError::NotStructured.into());
    }
    let expected_fields = array_type.fields();
    let mut expected_offset = 0;
    for expected_field in expected_fields.iter() {
        let actual_field = unsafe {
            pyo3::ffi::PyMapping_GetItemString(
                fields,
                expected_field.null_terminated_name.as_ptr() as *const core::ffi::c_char,
            )
        };
        if actual_field.is_null() {
            return Err(CheckArrayError::MissingField(expected_field.name()).into());
        }
        let actual_description = unsafe { pyo3::ffi::PyTuple_GetItem(actual_field, 0) }
            as *mut numpy::npyffi::PyArray_Descr;
        let expected_description = expected_field.dtype(python);
        unsafe {
            (*expected_description).byteorder = b'<' as core::ffi::c_char;
        }
        if unsafe {
            numpy::PY_ARRAY_API.PyArray_EquivTypes(python, expected_description, actual_description)
        } == 0
            || unsafe { (*expected_description).byteorder != (*actual_description).byteorder }
        {
            let error = CheckArrayError::Field {
                name: expected_field.name(),
                expected_type: simple_description_to_string(python, expected_description),
                actual_type: simple_description_to_string(python, actual_description),
            };
            unsafe { pyo3::ffi::Py_DECREF(actual_field) };
            return Err(error.into());
        }
        let actual_offset =
            unsafe { pyo3::ffi::PyLong_AsLong(pyo3::ffi::PyTuple_GetItem(actual_field, 1)) };
        if actual_offset != expected_offset {
            unsafe { pyo3::ffi::Py_DECREF(actual_field) };
            return Err(CheckArrayError::FieldOffset {
                name: expected_field.name(),
                actual_offset,
                expected_offset,
            }
            .into());
        }
        expected_offset += expected_field.size() as core::ffi::c_long;
        unsafe { pyo3::ffi::Py_DECREF(actual_field) };
    }
    let expected_fields_length = expected_fields.len();
    let actual_names = unsafe { numpy::npyffi::PyDataType_NAMES(python, (*array).descr) };
    let actual_names_length = unsafe { pyo3::ffi::PyTuple_GET_SIZE(actual_names) };
    if actual_names_length != expected_fields_length as pyo3::ffi::Py_ssize_t {
        use std::fmt::Write;
        let mut expected = "[".to_owned();
        for (index, expected_field) in expected_fields.iter().enumerate() {
            write!(
                &mut expected,
                "\"{}\"{}",
                &expected_field.null_terminated_name
                    [0..expected_field.null_terminated_name.len() - 1],
                if index == expected_fields_length - 1 {
                    ""
                } else {
                    ", "
                }
            )
            .unwrap();
        }
        write!(&mut expected, "]").unwrap();
        let mut actual = "[".to_owned();
        for index in 0..actual_names_length {
            let mut length: pyo3::ffi::Py_ssize_t = 0;
            let data = unsafe {
                pyo3::ffi::PyUnicode_AsUTF8AndSize(
                    pyo3::ffi::PyTuple_GET_ITEM(actual_names, index),
                    &mut length as *mut pyo3::ffi::Py_ssize_t,
                )
            } as *const u8;
            write!(
                &mut actual,
                "\"{}\"{}",
                std::str::from_utf8(unsafe { std::slice::from_raw_parts(data, length as usize) })
                    .expect("pyo3::ffi::PyUnicode_AsUTF8AndSize returned valid UTF8 bytes"),
                if index == actual_names_length - 1 {
                    ""
                } else {
                    ", "
                }
            )
            .unwrap();
        }
        write!(&mut actual, "]").unwrap();
        return Err(CheckArrayError::ExtraFields { expected, actual }.into());
    }
    Ok((array, unsafe { *((*array).dimensions) }))
}

fn simple_description_to_string(
    python: Python,
    description: *mut numpy::npyffi::PyArray_Descr,
) -> String {
    format!(
        "{}{}{}",
        unsafe { (*description).byteorder } as u8 as char,
        unsafe { (*description).type_ } as u8 as char,
        unsafe { numpy::npyffi::PyDataType_ELSIZE(python, description) }
    )
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ArrayType {
    Dvs,
    AedatImu,
    AedatTrigger,
    Dat,
    EsGeneric,
    EsAtis,
    EsColor,
    EvtTrigger,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FieldType {
    Empty,
    Bool,
    F32,
    U8,
    U16,
    U64,
    Object,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Field {
    pub null_terminated_name: &'static str,
    pub title: Option<&'static str>,
    pub field_type: FieldType,
}

impl Field {
    pub const fn new(
        null_terminated_name: &'static str,
        title: Option<&'static str>,
        field_type: FieldType,
    ) -> Self {
        Self {
            null_terminated_name,
            title,
            field_type,
        }
    }

    pub const fn size(&self) -> usize {
        match self.field_type {
            FieldType::Empty => 0,
            FieldType::Bool => 1,
            FieldType::F32 => 4,
            FieldType::U8 => 1,
            FieldType::U16 => 2,
            FieldType::U64 => 8,
            FieldType::Object => std::mem::size_of::<usize>(),
        }
    }

    pub fn name(&self) -> String {
        self.null_terminated_name[0..self.null_terminated_name.len() - 1].to_owned()
    }

    pub fn num(&self, python: Python) -> core::ffi::c_int {
        match self.field_type {
            FieldType::Empty => panic!("Field::num called on an empty field"),
            FieldType::Bool => bool::get_dtype_bound(python).num(),
            FieldType::F32 => f32::get_dtype_bound(python).num(),
            FieldType::U8 => u8::get_dtype_bound(python).num(),
            FieldType::U16 => u16::get_dtype_bound(python).num(),
            FieldType::U64 => u64::get_dtype_bound(python).num(),
            FieldType::Object => numpy::PyArrayDescr::object_bound(python).num(),
        }
    }

    pub fn dtype(&self, python: Python) -> *mut numpy::npyffi::PyArray_Descr {
        let dtype = unsafe { numpy::PY_ARRAY_API.PyArray_DescrFromType(python, self.num(python)) };
        if dtype.is_null() {
            panic!("PyArray_DescrFromType failed");
        }
        dtype
    }
}

const EMPTY: Field = Field {
    null_terminated_name: "\0",
    title: None,
    field_type: FieldType::Empty,
};

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Fields([Field; 11]);

impl ArrayType {
    pub const fn fields(self) -> Fields {
        Fields(match self {
            ArrayType::Dvs => [
                Field::new("t\0", None, FieldType::U64),
                Field::new("x\0", None, FieldType::U16),
                Field::new("y\0", None, FieldType::U16),
                Field::new("on\0", Some("p"), FieldType::Bool),
                EMPTY,
                EMPTY,
                EMPTY,
                EMPTY,
                EMPTY,
                EMPTY,
                EMPTY,
            ],
            ArrayType::AedatImu => [
                Field::new("t\0", None, FieldType::U64),
                Field::new("temperature\0", None, FieldType::F32),
                Field::new("accelerometer_x\0", None, FieldType::F32),
                Field::new("accelerometer_y\0", None, FieldType::F32),
                Field::new("accelerometer_z\0", None, FieldType::F32),
                Field::new("gyroscope_x\0", None, FieldType::F32),
                Field::new("gyroscope_y\0", None, FieldType::F32),
                Field::new("gyroscope_z\0", None, FieldType::F32),
                Field::new("magnetometer_x\0", None, FieldType::F32),
                Field::new("magnetometer_y\0", None, FieldType::F32),
                Field::new("magnetometer_z\0", None, FieldType::F32),
            ],
            ArrayType::AedatTrigger => [
                Field::new("t\0", None, FieldType::U64),
                Field::new("source\0", None, FieldType::U8),
                EMPTY,
                EMPTY,
                EMPTY,
                EMPTY,
                EMPTY,
                EMPTY,
                EMPTY,
                EMPTY,
                EMPTY,
            ],
            ArrayType::Dat => [
                Field::new("t\0", None, FieldType::U64),
                Field::new("x\0", None, FieldType::U16),
                Field::new("y\0", None, FieldType::U16),
                Field::new("payload\0", None, FieldType::U8),
                EMPTY,
                EMPTY,
                EMPTY,
                EMPTY,
                EMPTY,
                EMPTY,
                EMPTY,
            ],
            ArrayType::EsGeneric => [
                Field::new("t\0", None, FieldType::U64),
                Field::new("bytes\0", None, FieldType::Object),
                EMPTY,
                EMPTY,
                EMPTY,
                EMPTY,
                EMPTY,
                EMPTY,
                EMPTY,
                EMPTY,
                EMPTY,
            ],
            ArrayType::EsAtis => [
                Field::new("t\0", None, FieldType::U64),
                Field::new("x\0", None, FieldType::U16),
                Field::new("y\0", None, FieldType::U16),
                Field::new("exposure\0", Some("e"), FieldType::Bool),
                Field::new("polarity\0", Some("p"), FieldType::Bool),
                EMPTY,
                EMPTY,
                EMPTY,
                EMPTY,
                EMPTY,
                EMPTY,
            ],
            ArrayType::EsColor => [
                Field::new("t\0", None, FieldType::U64),
                Field::new("x\0", None, FieldType::U16),
                Field::new("y\0", None, FieldType::U16),
                Field::new("r\0", None, FieldType::Bool),
                Field::new("g\0", None, FieldType::Bool),
                Field::new("b\0", None, FieldType::Bool),
                EMPTY,
                EMPTY,
                EMPTY,
                EMPTY,
                EMPTY,
            ],
            ArrayType::EvtTrigger => [
                Field::new("t\0", None, FieldType::U64),
                Field::new("source\0", None, FieldType::U8),
                Field::new("rising\0", None, FieldType::Bool),
                EMPTY,
                EMPTY,
                EMPTY,
                EMPTY,
                EMPTY,
                EMPTY,
                EMPTY,
                EMPTY,
            ],
        })
    }

    #[allow(unused)]
    pub fn dtype(self, python: Python) -> *mut numpy::npyffi::PyArray_Descr {
        self.fields().dtype(python)
    }

    pub fn new_array(
        self,
        python: Python,
        length: numpy::npyffi::npy_intp,
    ) -> *mut numpy::npyffi::PyArrayObject {
        self.fields().new_array(python, length)
    }
}

pub struct FieldIterator<'a> {
    fields: &'a Fields,
    index: usize,
    length: usize,
}

impl<'a> Iterator for FieldIterator<'a> {
    type Item = Field;

    fn next(&mut self) -> Option<Self::Item> {
        if self.index < self.length {
            self.index += 1;
            Some(self.fields.0[self.index - 1])
        } else {
            None
        }
    }
}

impl Fields {
    pub const fn len(&self) -> usize {
        let mut index = 0;
        while index < self.0.len() {
            if matches!(self.0[index].field_type, FieldType::Empty) {
                return index;
            }
            index += 1;
        }
        index
    }

    pub fn iter(&self) -> FieldIterator {
        FieldIterator {
            fields: self,
            index: 0,
            length: self.len(),
        }
    }

    pub fn dtype(&self, python: Python) -> *mut numpy::npyffi::PyArray_Descr {
        unsafe {
            let dtype_as_list = pyo3::ffi::PyList_New(self.len() as pyo3::ffi::Py_ssize_t);
            for (index, field) in self.iter().enumerate() {
                set_dtype_as_list_field(
                    python,
                    dtype_as_list,
                    index,
                    field.null_terminated_name,
                    field.title,
                    field.num(python),
                );
            }
            let mut dtype: *mut numpy::npyffi::PyArray_Descr = std::ptr::null_mut();
            if numpy::PY_ARRAY_API.PyArray_DescrConverter(python, dtype_as_list, &mut dtype) < 0 {
                panic!("PyArray_DescrConverter failed");
            }
            pyo3::ffi::Py_DECREF(dtype_as_list);
            dtype
        }
    }

    pub fn new_array(
        &self,
        python: Python,
        mut length: numpy::npyffi::npy_intp,
    ) -> *mut numpy::npyffi::PyArrayObject {
        let dtype = self.dtype(python);
        unsafe {
            numpy::PY_ARRAY_API.PyArray_NewFromDescr(
                python,
                numpy::PY_ARRAY_API
                    .get_type_object(python, numpy::npyffi::array::NpyTypes::PyArray_Type),
                dtype,
                1_i32,
                &mut length,
                std::ptr::null_mut(),
                std::ptr::null_mut(),
                0_i32,
                std::ptr::null_mut(),
            ) as *mut numpy::npyffi::PyArrayObject
        }
    }
}

unsafe fn set_dtype_as_list_field(
    python: pyo3::Python,
    list: *mut pyo3::ffi::PyObject,
    index: usize,
    null_terminated_name: &str,
    title: Option<&str>,
    numpy_type: core::ffi::c_int,
) {
    let tuple = pyo3::ffi::PyTuple_New(2);
    if pyo3::ffi::PyTuple_SetItem(
        tuple,
        0 as pyo3::ffi::Py_ssize_t,
        match title {
            Some(title) => {
                let tuple = pyo3::ffi::PyTuple_New(2);
                if pyo3::ffi::PyTuple_SetItem(
                    tuple,
                    0 as pyo3::ffi::Py_ssize_t,
                    pyo3::ffi::PyUnicode_FromStringAndSize(
                        title.as_ptr() as *const core::ffi::c_char,
                        title.len() as pyo3::ffi::Py_ssize_t,
                    ),
                ) < 0
                {
                    panic!("PyTuple_SetItem 1 failed");
                }
                if pyo3::ffi::PyTuple_SetItem(
                    tuple,
                    1 as pyo3::ffi::Py_ssize_t,
                    pyo3::ffi::PyUnicode_FromStringAndSize(
                        null_terminated_name.as_ptr() as *const core::ffi::c_char,
                        (null_terminated_name.len() - 1) as pyo3::ffi::Py_ssize_t,
                    ),
                ) < 0
                {
                    panic!("PyTuple_SetItem 0 failed");
                }
                tuple
            }
            None => pyo3::ffi::PyUnicode_FromStringAndSize(
                null_terminated_name.as_ptr() as *const core::ffi::c_char,
                (null_terminated_name.len() - 1) as pyo3::ffi::Py_ssize_t,
            ),
        },
    ) < 0
    {
        panic!("PyTuple_SetItem 0 failed");
    }
    if pyo3::ffi::PyTuple_SetItem(
        tuple,
        1 as pyo3::ffi::Py_ssize_t,
        numpy::PY_ARRAY_API.PyArray_TypeObjectFromType(python, numpy_type),
    ) < 0
    {
        panic!("PyTuple_SetItem 1 failed");
    }
    if pyo3::ffi::PyList_SetItem(list, index as pyo3::ffi::Py_ssize_t, tuple) < 0 {
        panic!("PyList_SetItem failed");
    }
}

#[inline(always)]
pub unsafe fn array_at<T>(
    python: pyo3::Python,
    array: *mut numpy::npyffi::PyArrayObject,
    mut index: numpy::npyffi::npy_intp,
) -> *mut T {
    numpy::PY_ARRAY_API.PyArray_GetPtr(python, array, &mut index as *mut numpy::npyffi::npy_intp)
        as *mut T
}
