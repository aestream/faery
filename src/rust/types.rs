use numpy::prelude::*;
use numpy::Element;
use pyo3::prelude::*;


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
          ]
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