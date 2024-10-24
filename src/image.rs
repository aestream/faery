use image::ImageEncoder;
use numpy::PyArrayMethods;
use numpy::PyUntypedArrayMethods;
use pyo3::prelude::*;

#[pyfunction]
pub fn encode(
    frame: &pyo3::Bound<'_, numpy::PyArray3<u8>>,
    compression_level: &str,
) -> PyResult<PyObject> {
    if !frame.is_contiguous() {
        return Err(pyo3::exceptions::PyAttributeError::new_err(format!(
            "the frame's memory must be contiguous"
        )));
    }
    let compression_type  = match compression_level {
        "default" => image::codecs::png::CompressionType::Default,
        "fast" => image::codecs::png::CompressionType::Fast,
        "best" => image::codecs::png::CompressionType::Best,
        compression_level => return Err(pyo3::exceptions::PyAttributeError::new_err(format!(
            "unknown compression level \"{compression_level}\" (expected \"default\", \"fast\", or \"best\")"
        ))),
    };
    let readonly_frame = frame.readonly();
    let array = readonly_frame.as_array();
    let dimensions = array.dim();
    if dimensions.2 != 3 && dimensions.2 != 4 {
        return Err(pyo3::exceptions::PyAttributeError::new_err(format!(
            "expected an array whose last dimension is 3 (RGB) or 4 (RGBA) (got a {} x {} x {} array)",
            dimensions.0, dimensions.1, dimensions.2,
        )));
    }
    let mut buffer: Vec<u8> = Vec::new();
    {
        let encoder = image::codecs::png::PngEncoder::new_with_quality(
            &mut buffer,
            compression_type,
            if matches!(
                compression_type,
                image::codecs::png::CompressionType::Default
            ) || matches!(compression_type, image::codecs::png::CompressionType::Best)
            {
                image::codecs::png::FilterType::Adaptive
            } else {
                image::codecs::png::FilterType::NoFilter
            },
        );
        encoder
            .write_image(
                // unsafe: the array is contiguous
                unsafe {
                    std::slice::from_raw_parts(
                        array.as_ptr(),
                        dimensions.0 * dimensions.1 * dimensions.2,
                    )
                },
                dimensions.1 as u32,
                dimensions.0 as u32,
                if dimensions.2 == 3 {
                    image::ExtendedColorType::Rgb8
                } else {
                    image::ExtendedColorType::Rgba8
                },
            )
            .map_err(|error| pyo3::exceptions::PyRuntimeError::new_err(format!("{error:?}")))?;
    }
    Ok(Python::with_gil(|python| {
        pyo3::types::PyBytes::new_bound(python, &buffer).into()
    }))
}

#[inline(always)]
fn alpha_compose_over(
    color_a: u8,
    alpha_a0: u8,
    alpha_a1: u8,
    color_b: u8,
    alpha_b: u8,
) -> (u8, u8) {
    let color_a = color_a as u32;
    let alpha_a_2 = alpha_a0 as u32 * alpha_a1 as u32;
    let color_b = color_b as u32;
    let alpha_b = alpha_b as u32;
    let alpha_output = 255 * alpha_a_2 + (255 * 255 - alpha_a_2) * alpha_b;
    (
        ((255 * color_a * alpha_a_2 + (255 * 255 - alpha_a_2) * color_b * alpha_b) / alpha_output)
            as u8,
        (alpha_output / (255 * 255)) as u8,
    )
}

fn write_text(
    mut array: numpy::ndarray::ArrayBase<
        numpy::ndarray::ViewRepr<&mut u8>,
        numpy::ndarray::Dim<[usize; 3]>,
    >,
    dimensions: (usize, usize, usize),
    text: &str,
    mut x_offset: i32,
    y_offset: i32,
    scale: i32,
    color: (u8, u8, u8, u8),
    font: &fontdue::Font,
) {
    for character in text.chars() {
        let (metrics, bitmap) = font.rasterize(character, scale as f32);
        let xmin = x_offset as i32 + metrics.xmin;
        let ymin = y_offset as i32 - metrics.ymin + scale - metrics.height as i32;
        for y in 0..metrics.height as i32 {
            let output_y = y + ymin;
            if output_y >= 0 && output_y < dimensions.0 as i32 {
                for x in 0..metrics.width as i32 {
                    let output_x = x + xmin;
                    if output_x >= 0 && output_x < dimensions.1 as i32 {
                        let alpha = bitmap[x as usize + y as usize * metrics.width as usize];
                        if dimensions.2 == 3 {
                            let mut rgb = array.slice_mut(numpy::ndarray::prelude::s![
                                output_y,
                                output_x,
                                ..
                            ]);
                            // unsafe: {output_y, output_x, 0..3} are in bounds
                            unsafe {
                                *rgb.uget_mut(0) = alpha_compose_over(
                                    color.0,
                                    alpha,
                                    color.3,
                                    *rgb.uget_mut(0),
                                    255,
                                )
                                .0;
                                *rgb.uget_mut(1) = alpha_compose_over(
                                    color.1,
                                    alpha,
                                    color.3,
                                    *rgb.uget_mut(1),
                                    255,
                                )
                                .0;
                                *rgb.uget_mut(2) = alpha_compose_over(
                                    color.2,
                                    alpha,
                                    color.3,
                                    *rgb.uget_mut(2),
                                    255,
                                )
                                .0;
                            }
                        } else {
                            let mut rgba = array.slice_mut(numpy::ndarray::prelude::s![
                                output_y,
                                output_x,
                                ..
                            ]);
                            // unsafe: {output_y, output_x, 0..4} are in bounds
                            unsafe {
                                *rgba.uget_mut(0) = alpha_compose_over(
                                    color.0,
                                    alpha,
                                    color.3,
                                    *rgba.uget_mut(0),
                                    *rgba.uget_mut(3),
                                )
                                .0;
                                *rgba.uget_mut(1) = alpha_compose_over(
                                    color.1,
                                    alpha,
                                    color.3,
                                    *rgba.uget_mut(1),
                                    *rgba.uget_mut(3),
                                )
                                .0;
                                *rgba.uget_mut(2) = alpha_compose_over(
                                    color.2,
                                    alpha,
                                    color.3,
                                    *rgba.uget_mut(2),
                                    *rgba.uget_mut(3),
                                )
                                .0;
                                *rgba.uget_mut(3) = ((255 * (alpha as u32) * (color.3 as u32)
                                    + (255 * 255 - (alpha as u32) * (color.3 as u32))
                                        * (*rgba.uget_mut(3) as u32))
                                    / (255 * 255))
                                    as u8;
                            }
                        }
                    }
                }
            }
        }
        x_offset += metrics.advance_width as i32;
    }
}

static SCALES_AND_FONTS: pyo3::sync::GILProtected<std::cell::RefCell<Vec<(i32, fontdue::Font)>>> =
    pyo3::sync::GILProtected::new(std::cell::RefCell::new(Vec::new()));

#[pyfunction]
pub fn annotate(
    frame: &pyo3::Bound<'_, numpy::PyArray3<u8>>,
    text: &str,
    x_offset: i32,
    y_offset: i32,
    scale: i32,
    color: (u8, u8, u8, u8),
) -> PyResult<()> {
    if !frame.is_contiguous() {
        return Err(pyo3::exceptions::PyAttributeError::new_err(format!(
            "the frame's memory must be contiguous"
        )));
    }
    let mut readwrite_frame = frame.readwrite();
    let array = readwrite_frame.as_array_mut();
    let dimensions = array.dim();
    if dimensions.2 != 3 && dimensions.2 != 4 {
        return Err(pyo3::exceptions::PyAttributeError::new_err(format!(
            "expected an array whose last dimension is 3 (RGB) or 4 (RGBA) (got a {} x {} x {} array)",
            dimensions.0, dimensions.1, dimensions.2,
        )));
    }
    Python::with_gil(|python| {
        let mut scales_and_fonts = SCALES_AND_FONTS.get(python).borrow_mut();
        for (font_scale, font) in scales_and_fonts.iter() {
            if *font_scale == scale {
                write_text(
                    array, dimensions, text, x_offset, y_offset, scale, color, font,
                );
                return;
            }
        }
        let font = fontdue::Font::from_bytes(
            include_bytes!("RobotoMono-Regular.ttf") as &[u8],
            fontdue::FontSettings {
                collection_index: 0,
                scale: scale as f32,
                load_substitutions: true,
            },
        )
        .expect("loading RobotoMono-Regular.ttf did not fail");
        write_text(
            array, dimensions, text, x_offset, y_offset, scale, color, &font,
        );
        scales_and_fonts.push((scale, font));
    });
    Ok(())
}
