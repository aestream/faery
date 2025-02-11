use numpy::Element;
use numpy::PyArrayMethods;
use pyo3::prelude::*;

use crate::types;
use crate::utilities;

#[derive(Clone)]
enum Function {
    // colormap @ exp(-Δt / 𝜏)
    // (implementation) colormap[round(α * exp(Δt * υ))]
    Exponential {
        upsilon: f32,
        alpha: f32,
        colormap: Vec<u32>,
        default: u32,
        ts: Vec<u64>,
    },
    ExponentialDiverging {
        upsilon: f32,
        alphas: (f32, f32),
        colormaps: (Vec<u32>, Vec<u32>),
        default: u32,
        ts_and_polarities: Vec<(u64, neuromorphic_types::DvsPolarity)>,
    },
    // colormap @ (1 - Δt / (2𝜏))
    // (implementation) reversed_colormap[min(round(Δt * υ), α)]
    //
    // For linear decays, we use 2𝜏 to get the same overall "illuminance"
    // per event as the other decays.
    // Ee approximate the illuminance as the integral of the decay
    // from 0 (event time) to infinity.
    Linear {
        upsilon: f32,
        alpha: usize,
        reversed_colormap: Vec<u32>,
        default: u32,
        ts: Vec<u64>,
    },
    LinearDiverging {
        upsilons: (f32, f32),
        alphas: (usize, usize),
        reversed_colormaps: (Vec<u32>, Vec<u32>),
        default: u32,
        ts_and_polarities: Vec<(u64, neuromorphic_types::DvsPolarity)>,
    },
    // Δt ≤ 𝜏 ? colormap @ 1.0 : colormap @ 0.0
    Window {
        tau: u64,
        colors: (u32, u32),
        ts: Vec<u64>,
    },
    WindowDiverging {
        tau: u64,
        colors: (u32, u32, u32),
        ts_and_polarities: Vec<(u64, neuromorphic_types::DvsPolarity)>,
    },

    // activity = activity * exp(-Δt / 𝜏) + 1
    // min = sorted_activities[minimum_clip * len(sorted_activities)]
    // max = sorted_activities[maximum_clip * len(sorted_activities)]
    // ρ = γ < 0 ? 2 + γ² : -2 - γ²
    // σ = √(ρ² - 4)
    // χ = (activity.clip(min, max) - min) / (max - min) ∈ [0, 1]
    // ψ = (χ * (σ + ρ) / (χ * 2σ + ρ - σ)) ∈ [0, 1]
    // colormap @ (ψ * (len(colormap) - 1))
    //
    // (implementation)
    // activity = activity * exp(Δt * υ) + 1
    // χ = (activity.clip(min, max) - min) / (max - min)
    // colormap[round(χ / (α * χ + β))]
    Cumulative {
        upsilon: f32,
        alpha: f32,
        beta: f32,
        colormap: Vec<u32>,
        ts_and_activities: Vec<(u64, f32)>,
        minimum_clip: f32,
        maximum_clip: f32,
    },
    CumulativeDiverging {
        upsilon: f32,
        alphas: (f32, f32),
        betas: (f32, f32),
        colormaps: (Vec<u32>, Vec<u32>),
        ts_and_activities: (Vec<(u64, f32)>, Vec<(u64, f32)>),
        minimum_clip: f32,
        maximum_clip: f32,
    },
}

struct Inner {
    dimensions: (u16, u16),
    function: Function,
    previous_t: u64,
}

#[pyclass]
pub struct Renderer {
    inner: Option<Inner>,
}

fn component_f64_to_u8(component: f64) -> u8 {
    let value = (component * 255.0).round();
    if value <= 0.0 {
        0
    } else if value >= 255.0 {
        255
    } else {
        value as u8
    }
}

fn parse_sequential_colormap(
    array: numpy::ndarray::ArrayBase<
        numpy::ndarray::ViewRepr<&f64>,
        numpy::ndarray::Dim<[usize; 2]>,
    >,
) -> PyResult<Vec<u32>> {
    let dimensions = array.dim();
    let mut colormap = Vec::with_capacity(dimensions.0);
    for index in 0..dimensions.0 {
        colormap.push(u32::from_ne_bytes([
            component_f64_to_u8(
                *array
                    .get((index, 0))
                    .expect("the array's last dimension is 4"),
            ),
            component_f64_to_u8(
                *array
                    .get((index, 1))
                    .expect("the array's last dimension is 4"),
            ),
            component_f64_to_u8(
                *array
                    .get((index, 2))
                    .expect("the array's last dimension is 4"),
            ),
            component_f64_to_u8(
                *array
                    .get((index, 3))
                    .expect("the array's last dimension is 4"),
            ),
        ]));
    }
    Ok(colormap)
}

fn parse_diverging_colormap(
    array: numpy::ndarray::ArrayBase<
        numpy::ndarray::ViewRepr<&f64>,
        numpy::ndarray::Dim<[usize; 2]>,
    >,
) -> PyResult<(Vec<u32>, Vec<u32>)> {
    let dimensions = array.dim();
    let (mut off_colormap, mut on_colormap) = if dimensions.0 % 2 == 0 {
        (
            Vec::with_capacity(dimensions.0 / 2 + 1),
            Vec::with_capacity(dimensions.0 / 2),
        )
    } else {
        (
            Vec::with_capacity(dimensions.0 / 2 + 1),
            Vec::with_capacity(dimensions.0 / 2 + 1),
        )
    };
    for index in (0..=dimensions.0 / 2).rev() {
        off_colormap.push(u32::from_ne_bytes([
            component_f64_to_u8(
                *array
                    .get((index, 0))
                    .expect("the array's last dimension is 4"),
            ),
            component_f64_to_u8(
                *array
                    .get((index, 1))
                    .expect("the array's last dimension is 4"),
            ),
            component_f64_to_u8(
                *array
                    .get((index, 2))
                    .expect("the array's last dimension is 4"),
            ),
            component_f64_to_u8(
                *array
                    .get((index, 3))
                    .expect("the array's last dimension is 4"),
            ),
        ]));
    }
    for index in dimensions.0 / 2..dimensions.0 {
        on_colormap.push(u32::from_ne_bytes([
            component_f64_to_u8(
                *array
                    .get((index, 0))
                    .expect("the array's last dimension is 4"),
            ),
            component_f64_to_u8(
                *array
                    .get((index, 1))
                    .expect("the array's last dimension is 4"),
            ),
            component_f64_to_u8(
                *array
                    .get((index, 2))
                    .expect("the array's last dimension is 4"),
            ),
            component_f64_to_u8(
                *array
                    .get((index, 3))
                    .expect("the array's last dimension is 4"),
            ),
        ]));
    }
    Ok((off_colormap, on_colormap))
}

fn clipped_minimum_maximum(
    mut sorted_unbounded_frame: Vec<f32>,
    minimum_clip: f32,
    maximum_clip: f32,
) -> (f32, f32) {
    if sorted_unbounded_frame.is_empty() {
        (0.0f32, 1.0f32)
    } else {
        sorted_unbounded_frame
            .sort_by(|a, b| a.partial_cmp(b).expect("activities are finite numbers"));
        let mut minimum = if minimum_clip == 0.0 {
            0.0
        } else {
            sorted_unbounded_frame
                [((sorted_unbounded_frame.len() - 1) as f32 * minimum_clip).round() as usize]
        };
        let mut maximum = sorted_unbounded_frame
            [((sorted_unbounded_frame.len() - 1) as f32 * maximum_clip).round() as usize];
        if minimum < maximum {
            (minimum, maximum)
        } else {
            minimum = if maximum_clip == 0.0 {
                0.0
            } else {
                sorted_unbounded_frame[0]
            };
            maximum = sorted_unbounded_frame[sorted_unbounded_frame.len() - 1];
            if minimum < maximum {
                (minimum, maximum)
            } else {
                (minimum, minimum + 1.0)
            }
        }
    }
}

#[pymethods]
impl Renderer {
    #[new]
    fn new(
        dimensions: (u16, u16),
        decay: &str,
        tau: u64,
        minimum_clip: f32,
        maximum_clip: f32,
        gamma: f32,
        colormap_type: &str,
        colormap_rgba: &pyo3::Bound<'_, numpy::PyArray2<f64>>,
    ) -> PyResult<Self> {
        let readonly_colormap = colormap_rgba.readonly();
        let colormap_array = readonly_colormap.as_array();
        let colormap_dimensions = colormap_array.dim();
        if colormap_dimensions.0 == 0 {
            return Err(pyo3::exceptions::PyAttributeError::new_err(format!(
                "colormap_rgba must be an array with at least one row (got a {} x {} array)",
                colormap_dimensions.0, colormap_dimensions.1,
            )));
        }
        if colormap_dimensions.1 != 4 {
            return Err(pyo3::exceptions::PyAttributeError::new_err(format!(
                "colormap_rgba must be an array whose last dimension is 4 (RGBA) (got a {} x {} array)",
                colormap_dimensions.0, colormap_dimensions.1,
            )));
        }
        Ok(Renderer {
            inner: Some(Inner {
                dimensions,
                function: match decay {
                    "exponential" => match colormap_type {
                        "sequential" => {
                            let colormap = parse_sequential_colormap(colormap_array)?;
                            let default = *colormap.first().expect("the colormap is not empty");
                            Function::Exponential {
                            upsilon: (-1.0 / tau as f64) as f32,
                            alpha: (colormap.len() - 1) as f32,
                            colormap,
                            default,
                            ts: vec![u64::MAX; dimensions.0 as usize * dimensions.1 as usize],
                        }
                        },
                        "diverging" | "cyclic" => {
                            let (off_colormap, on_colormap) = parse_diverging_colormap(colormap_array)?;
                            let default = *off_colormap.first().expect("the colormap is not empty");
                            Function::ExponentialDiverging {
                                upsilon: (-1.0 / tau as f64) as f32,
                                alphas: (
                                    (off_colormap.len() - 1) as f32,
                                    (on_colormap.len() - 1) as f32,
                                ),
                                colormaps: (off_colormap, on_colormap),
                                default,
                                ts_and_polarities: vec![(u64::MAX, neuromorphic_types::DvsPolarity::Off); dimensions.0 as usize * dimensions.1 as usize],
                            }
                        },
                        colormap_type => return Err(pyo3::exceptions::PyAttributeError::new_err(format!(
                            "unknown colormap type \"{colormap_type}\" (expected \"sequential\", \"diverging\", or \"cyclic\")"
                        ))),
                    },
                    "linear" => match colormap_type {
                        "sequential" => {
                            let mut colormap = parse_sequential_colormap(colormap_array)?;
                            let default = *colormap.first().expect("the colormap is not empty");
                            colormap.reverse();
                            Function::Linear {
                                upsilon: ((colormap.len() - 1) as f64 / (2 * tau) as f64) as f32,
                                alpha: colormap.len() - 1,
                                reversed_colormap: colormap,
                                default,
                                ts: vec![u64::MAX; dimensions.0 as usize * dimensions.1 as usize],
                            }
                        },
                        "diverging" | "cyclic" => {
                            let (mut off_colormap, mut on_colormap) = parse_diverging_colormap(colormap_array)?;
                            let default = *off_colormap.first().expect("the colormap is not empty");
                            off_colormap.reverse();
                            on_colormap.reverse();
                            Function::LinearDiverging { upsilons: (
                                ((off_colormap.len() - 1) as f64 / (2 * tau) as f64) as f32,
                                ((on_colormap.len() - 1) as f64 / (2 * tau) as f64) as f32,
                            ), alphas: (
                                off_colormap.len() - 1,
                                on_colormap.len() - 1,
                            ), reversed_colormaps: (off_colormap, on_colormap), default, ts_and_polarities: vec![(u64::MAX, neuromorphic_types::DvsPolarity::Off); dimensions.0 as usize * dimensions.1 as usize] }
                        },
                        colormap_type => return Err(pyo3::exceptions::PyAttributeError::new_err(format!(
                            "unknown colormap type \"{colormap_type}\" (expected \"sequential\", \"diverging\", or \"cyclic\")"
                        ))),
                    },
                    "window" => match colormap_type {
                        "sequential" => {
                            let colormap = parse_sequential_colormap(colormap_array)?;
                            Function::Window {
                                tau,
                                colors: (*colormap.first().expect("the colormap is not empty"), *colormap.last().expect("the colormap is not empty")),
                                ts: vec![u64::MAX; dimensions.0 as usize * dimensions.1 as usize],
                            }
                        },
                        "diverging" | "cyclic" => {
                            let (off_colormap, on_colormap) = parse_diverging_colormap(colormap_array)?;
                            Function::WindowDiverging {
                                tau,
                                colors: (*off_colormap.first().expect("the colormap is not empty"), *off_colormap.last().expect("the colormap is not empty"), *on_colormap.last().expect("the colormap is not empty")),
                                ts_and_polarities: vec![(u64::MAX, neuromorphic_types::DvsPolarity::Off); dimensions.0 as usize * dimensions.1 as usize],
                            }
                        },
                        colormap_type => return Err(pyo3::exceptions::PyAttributeError::new_err(format!(
                            "unknown colormap type \"{colormap_type}\" (expected \"sequential\", \"diverging\", or \"cyclic\")"
                        ))),
                    },
                    "cumulative" => {
                        if minimum_clip < 0.0 || minimum_clip >= maximum_clip || maximum_clip > 1.0 {
                            return Err(pyo3::exceptions::PyAttributeError::new_err(format!(
                                "minimum_clip must be smaller than maximum_clip and they must be in the range [0.0, 0.1] (got {} and {})",
                                minimum_clip,
                                maximum_clip,
                            )));
                        }
                        let rho = if gamma < 0.0 {
                            2.0 + gamma.powi(2)
                        } else {
                            -2.0 - gamma.powi(2)
                        };
                        let sigma = (rho.powi(2) - 4.0).sqrt();
                        match colormap_type {
                            "sequential" => {
                                let colormap = parse_sequential_colormap(colormap_array)?;
                                Function::Cumulative {
                                    upsilon: (-1.0 / tau as f64) as f32,
                                    alpha: 2.0 * sigma / ((colormap.len() - 1) as f32 * (rho + sigma)),
                                    beta: (rho - sigma) / ((colormap.len() - 1) as f32 * (rho + sigma)),
                                    colormap,
                                    ts_and_activities: vec![(0, 0.0); dimensions.0 as usize * dimensions.1 as usize],
                                    minimum_clip,
                                    maximum_clip,
                                }
                            },
                            "diverging" | "cyclic" => {
                                let (off_colormap, on_colormap) = parse_diverging_colormap(colormap_array)?;
                                Function::CumulativeDiverging {
                                    upsilon: (-1.0 / tau as f64) as f32,
                                    alphas: (
                                        2.0 * sigma / ((off_colormap.len() - 1) as f32 * (rho + sigma)),
                                        2.0 * sigma / ((on_colormap.len() - 1) as f32 * (rho + sigma)),
                                    ),
                                    betas: (
                                        (rho - sigma) / ((off_colormap.len() - 1) as f32 * (rho + sigma)),
                                        (rho - sigma) / ((on_colormap.len() - 1) as f32 * (rho + sigma)),
                                    ),
                                    colormaps: (off_colormap, on_colormap),
                                    ts_and_activities: (
                                        vec![(0, 0.0); dimensions.0 as usize * dimensions.1 as usize],
                                        vec![(0, 0.0); dimensions.0 as usize * dimensions.1 as usize],
                                    ),
                                    minimum_clip,
                                    maximum_clip,
                                }
                            },
                            colormap_type => return Err(pyo3::exceptions::PyAttributeError::new_err(format!(
                                "unknown colormap type \"{colormap_type}\" (expected \"sequential\", \"diverging\", or \"cyclic\")"
                            ))),
                        }
                    },
                    decay => return Err(pyo3::exceptions::PyAttributeError::new_err(format!(
                        "unknown decay \"{decay}\" (expected \"exponential\", \"linear\", or \"window\")"
                    ))),
                },
                previous_t: 0,
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
        if self.inner.is_none() {
            return Err(pyo3::exceptions::PyException::new_err(
                "multiple calls to __exit__",
            ));
        }
        let _ = self.inner.take();
        Ok(false)
    }

    fn render(
        &mut self,
        events: &pyo3::Bound<'_, pyo3::types::PyAny>,
        render_t: u64,
    ) -> PyResult<PyObject> {
        Python::with_gil(|python| -> PyResult<PyObject> {
            match self.inner.as_mut() {
                Some(renderer) => {
                    let (array, length) =
                        types::check_array(python, types::ArrayType::Dvs, events)?;
                    match &mut renderer.function {
                        Function::Exponential { ts, .. }
                        | Function::Linear { ts, .. }
                        | Function::Window { ts, .. } => {
                            for index in 0..length {
                                let (t, x, y) = unsafe {
                                    let event_cell: *mut neuromorphic_types::DvsEvent<
                                        u64,
                                        u16,
                                        u16,
                                    > = types::array_at(python, array, index);
                                    ((*event_cell).t, (*event_cell).x, (*event_cell).y)
                                };
                                if t < renderer.previous_t {
                                    return Err(utilities::WriteError::NonMonotonic {
                                        previous_t: renderer.previous_t,
                                        t,
                                    }
                                    .into());
                                }
                                if x >= renderer.dimensions.0 {
                                    return Err(utilities::WriteError::XOverflow {
                                        x,
                                        width: renderer.dimensions.0,
                                    }
                                    .into());
                                }
                                if y >= renderer.dimensions.1 {
                                    return Err(utilities::WriteError::YOverflow {
                                        y,
                                        height: renderer.dimensions.1,
                                    }
                                    .into());
                                }
                                ts[(y as usize * renderer.dimensions.0 as usize) + x as usize] = t;
                            }
                        }
                        Function::ExponentialDiverging {
                            ts_and_polarities, ..
                        }
                        | Function::LinearDiverging {
                            ts_and_polarities, ..
                        }
                        | Function::WindowDiverging {
                            ts_and_polarities, ..
                        } => {
                            for index in 0..length {
                                let (t, x, y, polarity) = unsafe {
                                    let event_cell: *mut neuromorphic_types::DvsEvent<
                                        u64,
                                        u16,
                                        u16,
                                    > = types::array_at(python, array, index);
                                    (
                                        (*event_cell).t,
                                        (*event_cell).x,
                                        (*event_cell).y,
                                        (*event_cell).polarity,
                                    )
                                };
                                if t < renderer.previous_t {
                                    return Err(utilities::WriteError::NonMonotonic {
                                        previous_t: renderer.previous_t,
                                        t,
                                    }
                                    .into());
                                }
                                if x >= renderer.dimensions.0 {
                                    return Err(utilities::WriteError::XOverflow {
                                        x,
                                        width: renderer.dimensions.0,
                                    }
                                    .into());
                                }
                                if y >= renderer.dimensions.1 {
                                    return Err(utilities::WriteError::YOverflow {
                                        y,
                                        height: renderer.dimensions.1,
                                    }
                                    .into());
                                }
                                ts_and_polarities
                                    [(y as usize * renderer.dimensions.0 as usize) + x as usize] =
                                    (t, polarity);
                            }
                        }
                        Function::Cumulative {
                            upsilon,
                            ts_and_activities,
                            ..
                        } => {
                            for index in 0..length {
                                let (t, x, y) = unsafe {
                                    let event_cell: *mut neuromorphic_types::DvsEvent<
                                        u64,
                                        u16,
                                        u16,
                                    > = types::array_at(python, array, index);
                                    ((*event_cell).t, (*event_cell).x, (*event_cell).y)
                                };
                                if t < renderer.previous_t {
                                    return Err(utilities::WriteError::NonMonotonic {
                                        previous_t: renderer.previous_t,
                                        t,
                                    }
                                    .into());
                                }
                                if x >= renderer.dimensions.0 {
                                    return Err(utilities::WriteError::XOverflow {
                                        x,
                                        width: renderer.dimensions.0,
                                    }
                                    .into());
                                }
                                if y >= renderer.dimensions.1 {
                                    return Err(utilities::WriteError::YOverflow {
                                        y,
                                        height: renderer.dimensions.1,
                                    }
                                    .into());
                                }
                                let (previous_t, activity) = &mut ts_and_activities
                                    [(y as usize * renderer.dimensions.0 as usize) + x as usize];
                                *activity =
                                    *activity * ((t - *previous_t) as f32 * *upsilon).exp() + 1.0;
                                *previous_t = t;
                            }
                        }
                        Function::CumulativeDiverging {
                            upsilon,
                            ts_and_activities,
                            ..
                        } => {
                            for index in 0..length {
                                let (t, x, y, polarity) = unsafe {
                                    let event_cell: *mut neuromorphic_types::DvsEvent<
                                        u64,
                                        u16,
                                        u16,
                                    > = types::array_at(python, array, index);
                                    (
                                        (*event_cell).t,
                                        (*event_cell).x,
                                        (*event_cell).y,
                                        (*event_cell).polarity,
                                    )
                                };
                                if t < renderer.previous_t {
                                    return Err(utilities::WriteError::NonMonotonic {
                                        previous_t: renderer.previous_t,
                                        t,
                                    }
                                    .into());
                                }
                                if x >= renderer.dimensions.0 {
                                    return Err(utilities::WriteError::XOverflow {
                                        x,
                                        width: renderer.dimensions.0,
                                    }
                                    .into());
                                }
                                if y >= renderer.dimensions.1 {
                                    return Err(utilities::WriteError::YOverflow {
                                        y,
                                        height: renderer.dimensions.1,
                                    }
                                    .into());
                                }
                                match polarity {
                                    neuromorphic_types::DvsPolarity::Off => {
                                        let (previous_t, activity) = &mut ts_and_activities.0[(y
                                            as usize
                                            * renderer.dimensions.0 as usize)
                                            + x as usize];
                                        *activity = *activity
                                            * ((t - *previous_t) as f32 * *upsilon).exp()
                                            + 1.0;
                                        *previous_t = t;
                                    }
                                    neuromorphic_types::DvsPolarity::On => {
                                        let (previous_t, activity) = &mut ts_and_activities.1[(y
                                            as usize
                                            * renderer.dimensions.0 as usize)
                                            + x as usize];
                                        *activity = *activity
                                            * ((t - *previous_t) as f32 * *upsilon).exp()
                                            + 1.0;
                                        *previous_t = t;
                                    }
                                }
                            }
                        }
                    }
                    if render_t < renderer.previous_t {
                        return Err(pyo3::exceptions::PyValueError::new_err(format!(
                            "render_t cannot be smaller than the last event's timestamp (render_r is {}, the last event timestamp is {}",
                            render_t,
                            renderer.previous_t
                        )));
                    }
                    let array = unsafe {
                        let mut dimensions = [
                            renderer.dimensions.1 as numpy::npyffi::npy_intp,
                            renderer.dimensions.0 as numpy::npyffi::npy_intp,
                            4,
                        ];
                        numpy::PY_ARRAY_API.PyArray_Empty(
                            python,
                            3,
                            dimensions.as_mut_ptr(),
                            u8::get_dtype(python).into_ptr() as *mut numpy::npyffi::PyArray_Descr,
                            0,
                        )
                    } as *mut numpy::npyffi::PyArrayObject;
                    let slice = {
                        let mut index = [
                            0 as numpy::npyffi::npy_intp,
                            0 as numpy::npyffi::npy_intp,
                            0 as numpy::npyffi::npy_intp,
                        ];
                        unsafe {
                            std::slice::from_raw_parts_mut(
                                numpy::PY_ARRAY_API.PyArray_GetPtr(
                                    python,
                                    array,
                                    index.as_mut_ptr(),
                                ) as *mut u32,
                                renderer.dimensions.0 as usize * renderer.dimensions.1 as usize,
                            )
                        }
                    };
                    match &renderer.function {
                        Function::Exponential {
                            upsilon,
                            alpha,
                            colormap,
                            default,
                            ts,
                        } => {
                            for (t, color) in ts.iter().zip(slice.iter_mut()) {
                                if *t == u64::MAX {
                                    *color = *default;
                                } else {
                                    *color = colormap[(alpha
                                        * ((render_t - t) as f32 * upsilon).exp())
                                    .round()
                                        as usize];
                                }
                            }
                        }
                        Function::ExponentialDiverging {
                            upsilon,
                            alphas,
                            colormaps,
                            default,
                            ts_and_polarities,
                        } => {
                            for ((t, polarity), color) in
                                ts_and_polarities.iter().zip(slice.iter_mut())
                            {
                                if *t == u64::MAX {
                                    *color = *default;
                                } else if matches!(polarity, neuromorphic_types::DvsPolarity::Off) {
                                    *color = colormaps.0[(alphas.0
                                        * ((render_t - t) as f32 * upsilon).exp())
                                    .round()
                                        as usize];
                                } else {
                                    *color = colormaps.1[(alphas.1
                                        * ((render_t - t) as f32 * upsilon).exp())
                                    .round()
                                        as usize];
                                }
                            }
                        }
                        Function::Linear {
                            upsilon,
                            alpha,
                            reversed_colormap,
                            default,
                            ts,
                        } => {
                            for (t, color) in ts.iter().zip(slice.iter_mut()) {
                                if *t == u64::MAX {
                                    *color = *default;
                                } else {
                                    *color = reversed_colormap
                                        [(((render_t - t) as f32 * upsilon) as usize).min(*alpha)];
                                }
                            }
                        }
                        Function::LinearDiverging {
                            upsilons,
                            alphas,
                            reversed_colormaps,
                            default,
                            ts_and_polarities,
                        } => {
                            for ((t, polarity), color) in
                                ts_and_polarities.iter().zip(slice.iter_mut())
                            {
                                if *t == u64::MAX {
                                    *color = *default;
                                } else if matches!(polarity, neuromorphic_types::DvsPolarity::Off) {
                                    *color = reversed_colormaps.0[(((render_t - t) as f32
                                        * upsilons.0)
                                        as usize)
                                        .min(alphas.0)];
                                } else {
                                    *color = reversed_colormaps.1[(((render_t - t) as f32
                                        * upsilons.1)
                                        as usize)
                                        .min(alphas.1)];
                                }
                            }
                        }
                        // Δt ≤ 𝜏 ? colormap @ 1.0 : colormap @ 0.0
                        Function::Window { tau, colors, ts } => {
                            for (t, color) in ts.iter().zip(slice.iter_mut()) {
                                if *t == u64::MAX || render_t - t > *tau {
                                    *color = colors.0;
                                } else {
                                    *color = colors.1;
                                }
                            }
                        }
                        Function::WindowDiverging {
                            tau,
                            colors,
                            ts_and_polarities,
                        } => {
                            for ((t, polarity), color) in
                                ts_and_polarities.iter().zip(slice.iter_mut())
                            {
                                if *t == u64::MAX || render_t - t > *tau {
                                    *color = colors.0;
                                } else if matches!(polarity, neuromorphic_types::DvsPolarity::Off) {
                                    *color = colors.1;
                                } else {
                                    *color = colors.2;
                                }
                            }
                        }
                        Function::Cumulative {
                            upsilon,
                            alpha,
                            beta,
                            colormap,
                            ts_and_activities,
                            minimum_clip,
                            maximum_clip,
                        } => {
                            let mut unbounded_frame = vec![
                                0.0;
                                renderer.dimensions.0 as usize
                                    * renderer.dimensions.1 as usize
                            ];
                            for ((t, activity), value) in
                                ts_and_activities.iter().zip(unbounded_frame.iter_mut())
                            {
                                *value = ((render_t - t) as f32 * upsilon).exp() * activity;
                            }
                            let (minimum, maximum) = clipped_minimum_maximum(
                                unbounded_frame
                                    .iter()
                                    .filter_map(
                                        |value| if *value > 0.0 { Some(*value) } else { None },
                                    )
                                    .collect(),
                                *minimum_clip,
                                *maximum_clip,
                            );
                            let scale = 1.0 / (maximum - minimum);
                            for (value, color) in unbounded_frame.iter().zip(slice.iter_mut()) {
                                let chi = (value.clamp(minimum, maximum) - minimum) * scale;
                                *color = colormap[(chi / (*alpha * chi + *beta)).round() as usize];
                            }
                        }
                        Function::CumulativeDiverging {
                            upsilon,
                            alphas,
                            betas,
                            colormaps,
                            ts_and_activities,
                            minimum_clip,
                            maximum_clip,
                        } => {
                            let mut unbounded_frame = vec![
                                0.0;
                                renderer.dimensions.0 as usize
                                    * renderer.dimensions.1 as usize
                            ];
                            for pixel_index in 0..unbounded_frame.len() {
                                let (off_t, mut off_activity) = ts_and_activities.0[pixel_index];
                                off_activity *= ((render_t - off_t) as f32 * upsilon).exp();
                                let (on_t, mut on_activity) = ts_and_activities.1[pixel_index];
                                on_activity *= ((render_t - on_t) as f32 * upsilon).exp();
                                unbounded_frame[pixel_index] = if on_activity > off_activity {
                                    on_activity
                                } else {
                                    -off_activity
                                };
                            }
                            let (minimum, maximum) = clipped_minimum_maximum(
                                unbounded_frame
                                    .iter()
                                    .filter_map(|value| {
                                        if *value != 0.0 {
                                            Some(value.abs())
                                        } else {
                                            None
                                        }
                                    })
                                    .collect(),
                                *minimum_clip,
                                *maximum_clip,
                            );
                            let scale = 1.0 / (maximum - minimum);
                            for (value, color) in unbounded_frame.iter().zip(slice.iter_mut()) {
                                if *value <= 0.0 {
                                    let chi = ((-value).clamp(minimum, maximum) - minimum) * scale;
                                    *color = colormaps.0
                                        [(chi / (alphas.0 * chi + betas.0)).round() as usize];
                                } else {
                                    let chi = (value.clamp(minimum, maximum) - minimum) * scale;
                                    *color = colormaps.1
                                        [(chi / (alphas.1 * chi + betas.1)).round() as usize];
                                }
                            }
                        }
                    }
                    Ok(unsafe {
                        PyObject::from_owned_ptr(python, array as *mut pyo3::ffi::PyObject)
                    })
                }
                None => Err(pyo3::exceptions::PyException::new_err(
                    "write called after __exit__",
                )),
            }
        })
    }
}
