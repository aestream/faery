use image::ImageEncoder;
use numpy::PyArrayDescrMethods;
use numpy::PyArrayMethods;
use numpy::PyUntypedArrayMethods;
use numpy::ToPyArray;
use pyo3::prelude::*;

use crate::font;

#[pyfunction]
pub fn decode(bytes: &[u8]) -> PyResult<PyObject> {
    let decoder = image::codecs::png::PngDecoder::new(std::io::Cursor::new(bytes))
        .map_err(|error| pyo3::exceptions::PyException::new_err(format!("{error}")))?;
    let image = image::DynamicImage::from_decoder(decoder)
        .map_err(|error| pyo3::exceptions::PyException::new_err(format!("{error}")))?
        .to_rgba8();
    let (width, height) = image.dimensions();
    let array =
        numpy::ndarray::ArrayView3::<u8>::from_shape((height as usize, width as usize, 4), &image)
            .map_err(|error| pyo3::exceptions::PyException::new_err(format!("{error}")))?;
    Ok(Python::with_gil(|python| {
        array.to_pyarray_bound(python).into()
    }))
}

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
    size: i32,
    color: (u8, u8, u8, u8),
    font: &fontdue::Font,
) {
    for character in text.chars() {
        let (metrics, bitmap) = font.rasterize(character, size as f32);
        let xmin = x_offset as i32 + metrics.xmin;
        let ymin = y_offset as i32 - metrics.ymin + size - metrics.height as i32;
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
    x: i32,
    y: i32,
    size: i32,
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
            if *font_scale == size {
                write_text(array, dimensions, text, x, y, size, color, font);
                return;
            }
        }
        let font = fontdue::Font::from_bytes(
            font::ROBOTO_MONO_REGULAR as &[u8],
            fontdue::FontSettings {
                collection_index: 0,
                scale: size as f32,
                load_substitutions: true,
            },
        )
        .expect("loading RobotoMono-Regular.ttf did not fail");
        write_text(array, dimensions, text, x, y, size, color, &font);
        scales_and_fonts.push((size, font));
    });
    Ok(())
}

fn parse_filter(string: &str) -> PyResult<image::imageops::FilterType> {
    match string {
        "nearest" => Ok(image::imageops::FilterType::Nearest),
        "triangle" => Ok(image::imageops::FilterType::Triangle),
        "catmull_rom" => Ok(image::imageops::FilterType::CatmullRom),
        "gaussian" => Ok(image::imageops::FilterType::Gaussian),
        "lanczos3" => Ok(image::imageops::FilterType::Lanczos3),
        string => Err(pyo3::exceptions::PyAttributeError::new_err(format!(
            "unknwon filter type \"{string}\""
        ))),
    }
}

enum ImageType {
    U8,
    F64,
}

#[pyfunction]
pub fn resize(
    frame: &pyo3::Bound<'_, numpy::PyUntypedArray>,
    new_dimensions: (u16, u16),
    sampling_filter: &str,
) -> PyResult<PyObject> {
    if !frame.is_contiguous() {
        return Err(pyo3::exceptions::PyAttributeError::new_err(format!(
            "the frame's memory must be contiguous"
        )));
    }
    let image_type = Python::with_gil(|python| {
        if frame.dtype().is_equiv_to(&numpy::dtype_bound::<u8>(python)) {
            Some(ImageType::U8)
        } else if frame
            .dtype()
            .is_equiv_to(&numpy::dtype_bound::<f64>(python))
        {
            Some(ImageType::F64)
        } else {
            None
        }
    });
    match image_type {
        Some(ImageType::U8) => {
            let frame: &pyo3::Bound<'_, numpy::PyArray3<u8>> = frame.downcast()?;
            let readonly_frame = frame.readonly();
            let array_dimensions = readonly_frame.as_array().dim();
            if array_dimensions.2 == 3 {
                let image = image::ImageBuffer::<image::Rgb<u8>, &[u8]>::from_raw(
                    array_dimensions.1 as u32,
                    array_dimensions.0 as u32,
                    readonly_frame.as_slice().expect("the frame is contiguous"),
                )
                .expect("from_raw does not need to allocate");
                let resized_image = image::imageops::resize(
                    &image,
                    new_dimensions.0 as u32,
                    new_dimensions.1 as u32,
                    parse_filter(sampling_filter)?,
                );
                let array = numpy::ndarray::ArrayView3::<u8>::from_shape(
                    (new_dimensions.1 as usize, new_dimensions.0 as usize, 3),
                    &resized_image,
                )
                .map_err(|error| pyo3::exceptions::PyException::new_err(format!("{error}")))?;
                Ok(Python::with_gil(|python| {
                    array.to_pyarray_bound(python).into()
                }))
            } else if array_dimensions.2 == 4 {
                let image = image::ImageBuffer::<image::Rgba<u8>, &[u8]>::from_raw(
                    array_dimensions.1 as u32,
                    array_dimensions.0 as u32,
                    readonly_frame.as_slice().expect("the frame is contiguous"),
                )
                .expect("from_raw does not need to allocate");
                let resized_image = image::imageops::resize(
                    &image,
                    new_dimensions.0 as u32,
                    new_dimensions.1 as u32,
                    parse_filter(sampling_filter)?,
                );
                let array = numpy::ndarray::ArrayView3::<u8>::from_shape(
                    (new_dimensions.1 as usize, new_dimensions.0 as usize, 4),
                    &resized_image,
                )
                .map_err(|error| pyo3::exceptions::PyException::new_err(format!("{error}")))?;
                Ok(Python::with_gil(|python| {
                    array.to_pyarray_bound(python).into()
                }))
            } else {
                Err(pyo3::exceptions::PyAttributeError::new_err(format!(
                    "expected an array whose last dimension is 3 (RGB) or 4 (RGBA) (got a {} x {} x {} array)",
                    array_dimensions.0, array_dimensions.1, array_dimensions.2,
                )))
            }
        }
        Some(ImageType::F64) => {
            let frame: &pyo3::Bound<'_, numpy::PyArray3<f64>> = frame.downcast()?;
            if !frame.is_contiguous() {
                return Err(pyo3::exceptions::PyAttributeError::new_err(format!(
                    "the frame's memory must be contiguous"
                )));
            }
            let readonly_frame = frame.readonly();
            let array_dimensions = readonly_frame.as_array().dim();
            if array_dimensions.2 == 2 {
                let image = image::ImageBuffer::<image::LumaA<f64>, &[f64]>::from_raw(
                    array_dimensions.1 as u32,
                    array_dimensions.0 as u32,
                    readonly_frame.as_slice().expect("the frame is contiguous"),
                )
                .expect("from_raw does not need to allocate");
                let resized_image = image::imageops::resize(
                    &image,
                    new_dimensions.0 as u32,
                    new_dimensions.1 as u32,
                    parse_filter(sampling_filter)?,
                );
                let array = numpy::ndarray::ArrayView3::<f64>::from_shape(
                    (new_dimensions.1 as usize, new_dimensions.0 as usize, 2),
                    &resized_image,
                )
                .map_err(|error| pyo3::exceptions::PyException::new_err(format!("{error}")))?;
                Ok(Python::with_gil(|python| {
                    array.to_pyarray_bound(python).into()
                }))
            } else {
                Err(pyo3::exceptions::PyAttributeError::new_err(format!(
                    "expected an array whose last dimension is 2 (got a {} x {} x {} array)",
                    array_dimensions.0, array_dimensions.1, array_dimensions.2,
                )))
            }
        }
        None => Err(pyo3::exceptions::PyAttributeError::new_err(format!(
            "unsupported image type (the dtype must be numpy.uint8 or numpy.float64)"
        ))),
    }
}

#[pyfunction]
pub fn overlay(
    frame: &pyo3::Bound<'_, numpy::PyArray3<u8>>,
    overlay: &pyo3::Bound<'_, numpy::PyArray3<u8>>,
    x: i32,
    y: i32,
    new_dimensions: (u16, u16),
    sampling_filter: &str,
) -> PyResult<()> {
    if !frame.is_contiguous() {
        return Err(pyo3::exceptions::PyAttributeError::new_err(format!(
            "the frame's memory must be contiguous"
        )));
    }
    if !overlay.is_contiguous() {
        return Err(pyo3::exceptions::PyAttributeError::new_err(format!(
            "the overlay's memory must be contiguous"
        )));
    }
    let mut readwrite_frame = frame.readwrite();
    let frame_dimensions = readwrite_frame.as_array_mut().dim();
    let readonly_overlay = overlay.readonly();
    let overlay_dimensions = readonly_overlay.as_array().dim();
    let sampling_filter = parse_filter(sampling_filter)?;
    if overlay_dimensions.2 == 3 {
        let overlay_image = image::ImageBuffer::<image::Rgb<u8>, &[u8]>::from_raw(
            overlay_dimensions.1 as u32,
            overlay_dimensions.0 as u32,
            readonly_overlay
                .as_slice()
                .expect("the overlay is contiguous"),
        )
        .expect("from_raw does not need to allocate");
        if frame_dimensions.2 == 3 {
            let mut frame_image = image::ImageBuffer::<image::Rgb<u8>, &mut [u8]>::from_raw(
                frame_dimensions.1 as u32,
                frame_dimensions.0 as u32,
                readwrite_frame
                    .as_slice_mut()
                    .expect("the frame is contiguous"),
            )
            .expect("from_raw does not need to allocate");
            if overlay_image.width() == new_dimensions.0 as u32
                && overlay_image.height() == new_dimensions.1 as u32
                && matches!(sampling_filter, image::imageops::FilterType::Nearest)
            {
                // RGB over RGB without resizing
                image::imageops::overlay(&mut frame_image, &overlay_image, x as i64, y as i64);
            } else {
                // RGB over RGB with resizing
                let overlay_image = image::imageops::resize(
                    &overlay_image,
                    new_dimensions.0 as u32,
                    new_dimensions.1 as u32,
                    sampling_filter,
                );
                image::imageops::overlay(&mut frame_image, &overlay_image, x as i64, y as i64);
            }
            Ok(())
        } else if frame_dimensions.2 == 4 {
            // RGB over RGBA, always resize since we need to allocate to use to_rgba8
            let mut frame_image = image::ImageBuffer::<image::Rgba<u8>, &mut [u8]>::from_raw(
                frame_dimensions.1 as u32,
                frame_dimensions.0 as u32,
                readwrite_frame
                    .as_slice_mut()
                    .expect("the frame is contiguous"),
            )
            .expect("from_raw does not need to allocate");
            let overlay_image = image::imageops::resize(
                &overlay_image,
                new_dimensions.0 as u32,
                new_dimensions.1 as u32,
                sampling_filter,
            );
            let overlay_image = image::DynamicImage::ImageRgb8(overlay_image).to_rgba8();
            image::imageops::overlay(&mut frame_image, &overlay_image, x as i64, y as i64);
            Ok(())
        } else {
            Err(pyo3::exceptions::PyAttributeError::new_err(format!(
                "expected a frame array whose last dimension is 3 (RGB) or 4 (RGBA) (got a {} x {} x {} array)",
                frame_dimensions.0, frame_dimensions.1, frame_dimensions.2,
            )))
        }
    } else if overlay_dimensions.2 == 4 {
        let overlay_image = image::ImageBuffer::<image::Rgba<u8>, &[u8]>::from_raw(
            overlay_dimensions.1 as u32,
            overlay_dimensions.0 as u32,
            readonly_overlay
                .as_slice()
                .expect("the overlay is contiguous"),
        )
        .expect("from_raw does not need to allocate");
        if frame_dimensions.2 == 3 {
            // RGBA over RGB, always resize since we need to allocate to use to_rgb8
            let mut frame_image = image::ImageBuffer::<image::Rgb<u8>, &mut [u8]>::from_raw(
                frame_dimensions.1 as u32,
                frame_dimensions.0 as u32,
                readwrite_frame
                    .as_slice_mut()
                    .expect("the frame is contiguous"),
            )
            .expect("from_raw does not need to allocate");
            let overlay_image = image::imageops::resize(
                &overlay_image,
                new_dimensions.0 as u32,
                new_dimensions.1 as u32,
                sampling_filter,
            );
            let overlay_image = image::DynamicImage::ImageRgba8(overlay_image).to_rgb8();
            image::imageops::overlay(&mut frame_image, &overlay_image, x as i64, y as i64);
            Ok(())
        } else if frame_dimensions.2 == 4 {
            let mut frame_image = image::ImageBuffer::<image::Rgba<u8>, &mut [u8]>::from_raw(
                frame_dimensions.1 as u32,
                frame_dimensions.0 as u32,
                readwrite_frame
                    .as_slice_mut()
                    .expect("the frame is contiguous"),
            )
            .expect("from_raw does not need to allocate");
            if overlay_image.width() == new_dimensions.0 as u32
                && overlay_image.height() == new_dimensions.1 as u32
                && matches!(sampling_filter, image::imageops::FilterType::Nearest)
            {
                // RGBA over RGBA without resizing
                image::imageops::overlay(&mut frame_image, &overlay_image, x as i64, y as i64);
            } else {
                // RGBA over RGBA with resizing
                let overlay_image = image::imageops::resize(
                    &overlay_image,
                    new_dimensions.0 as u32,
                    new_dimensions.1 as u32,
                    sampling_filter,
                );
                image::imageops::overlay(&mut frame_image, &overlay_image, x as i64, y as i64);
            }
            Ok(())
        } else {
            Err(pyo3::exceptions::PyAttributeError::new_err(format!(
                "expected a frame array whose last dimension is 3 (RGB) or 4 (RGBA) (got a {} x {} x {} array)",
                frame_dimensions.0, frame_dimensions.1, frame_dimensions.2,
            )))
        }
    } else {
        Err(pyo3::exceptions::PyAttributeError::new_err(format!(
            "expected an overlay array whose last dimension is 3 (RGB) or 4 (RGBA) (got a {} x {} x {} array)",
            overlay_dimensions.0, overlay_dimensions.1, overlay_dimensions.2,
        )))
    }
}
