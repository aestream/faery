use numpy::Element;
use pyo3::prelude::*;

use crate::font;

#[pyfunction]
pub fn render(svg_string: &str) -> PyResult<PyObject> {
    let fontdb = std::sync::Arc::new({
        let mut fontdb = resvg::usvg::fontdb::Database::new();
        let mut font = Vec::new();
        font.resize(font::ROBOTO_MONO_REGULAR.len(), 0);
        font.copy_from_slice(font::ROBOTO_MONO_REGULAR);
        fontdb.load_font_data(font);
        fontdb
    });
    let tree = resvg::usvg::Tree::from_str(
        svg_string,
        &resvg::usvg::Options {
            resources_dir: None,
            dpi: 96.0,
            font_family: "Roboto Mono".to_owned(),
            font_size: 12.0,
            languages: vec!["en".to_string()],
            shape_rendering: resvg::usvg::ShapeRendering::default(),
            text_rendering: resvg::usvg::TextRendering::default(),
            image_rendering: resvg::usvg::ImageRendering::default(),
            default_size: resvg::usvg::Size::from_wh(100.0, 100.0).unwrap(),
            image_href_resolver: resvg::usvg::ImageHrefResolver::default(),
            font_resolver: resvg::usvg::FontResolver::default(),
            fontdb,
            style_sheet: None,
        },
    )
    .map_err(|error| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(error.to_string()))?;
    let size = tree.size();
    let width = size.width().ceil() as u32;
    if width == 0 {
        return Err(PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(
            "the SVG's width is 0",
        ));
    }
    let height = size.height().ceil() as u32;
    if height == 0 {
        return Err(PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(
            "the SVG's height is 0",
        ));
    }
    let mut pixmap =
        resvg::tiny_skia::Pixmap::new(width, height).expect("width and height are not zero");
    resvg::render(
        &tree,
        resvg::tiny_skia::Transform::identity(),
        &mut pixmap.as_mut(),
    );
    Ok(Python::with_gil(|python| {
        let mut dimensions = [
            height as numpy::npyffi::npy_intp,
            width as numpy::npyffi::npy_intp,
            4,
        ];
        let mut index = [
            0 as numpy::npyffi::npy_intp,
            0 as numpy::npyffi::npy_intp,
            0 as numpy::npyffi::npy_intp,
        ];
        unsafe {
            let array = numpy::PY_ARRAY_API.PyArray_Empty(
                python,
                3,
                dimensions.as_mut_ptr(),
                u8::get_dtype_bound(python).into_ptr() as *mut numpy::npyffi::PyArray_Descr,
                0,
            ) as *mut numpy::npyffi::PyArrayObject;
            std::ptr::copy_nonoverlapping(
                pixmap.data().as_ptr(),
                numpy::PY_ARRAY_API.PyArray_GetPtr(python, array, index.as_mut_ptr()) as *mut u8,
                width as usize * height as usize * 4,
            );
            PyObject::from_owned_ptr(python, array as *mut pyo3::ffi::PyObject)
        }
    }))
}
