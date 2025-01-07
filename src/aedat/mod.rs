mod common;
mod decoder;
mod encoder;

use crate::types;
use crate::utilities;

use numpy::convert::ToPyArray;
use numpy::ndarray::IntoDimension;
use numpy::prelude::*;
use pyo3::prelude::*;

impl From<decoder::Error> for PyErr {
    fn from(error: decoder::Error) -> Self {
        PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(error.to_string())
    }
}

impl From<decoder::ReadError> for PyErr {
    fn from(error: decoder::ReadError) -> Self {
        PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(error.to_string())
    }
}

impl From<encoder::Error> for PyErr {
    fn from(error: encoder::Error) -> Self {
        PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(error.to_string())
    }
}

impl From<encoder::CompressionError> for PyErr {
    fn from(error: encoder::CompressionError) -> Self {
        PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(error.to_string())
    }
}

impl From<encoder::PacketError> for PyErr {
    fn from(error: encoder::PacketError) -> Self {
        PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(error.to_string())
    }
}

impl From<common::Error> for PyErr {
    fn from(error: common::Error) -> Self {
        PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(error.to_string())
    }
}

impl From<common::DescriptionError> for PyErr {
    fn from(error: common::DescriptionError) -> Self {
        PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(error.to_string())
    }
}

#[pyclass]
#[derive(FromPyObject)]
pub struct Track {
    #[pyo3(get, set)]
    pub id: u32,
    #[pyo3(get, set)]
    pub data_type: String,
    #[pyo3(get, set)]
    pub dimensions: Option<(u16, u16)>,
}

#[pymethods]
impl Track {
    #[new]
    #[pyo3(signature = (id, data_type, dimensions))]
    fn new(id: u32, data_type: String, dimensions: Option<(u16, u16)>) -> Self {
        Self {
            id,
            data_type,
            dimensions,
        }
    }

    fn __repr__(&self) -> String {
        format!(
            "faery.aedat.Track(id={}, data_type=\"{}\", dimensions={})",
            self.id,
            self.data_type,
            match self.dimensions {
                Some(dimensions) => format!("({}, {})", dimensions.0, dimensions.1),
                None => "None".to_owned(),
            }
        )
    }
}

#[pyclass]
pub struct Frame {
    #[pyo3(get)]
    t: u64,
    #[pyo3(get)]
    begin_t: i64,
    #[pyo3(get)]
    end_t: i64,
    #[pyo3(get)]
    exposure_begin_t: i64,
    #[pyo3(get)]
    exposure_end_t: i64,
    #[pyo3(get)]
    format: String,
    #[pyo3(get)]
    offset_x: i16,
    #[pyo3(get)]
    offset_y: i16,
    #[pyo3(get)]
    pixels: PyObject,
}

#[pymethods]
impl Frame {
    fn __repr__(&self) -> String {
        Python::with_gil(|python| -> String {
            format!(
                "faery.aedat.Frame(t={}, begin_t={}, end_t={}, exposure_begin_t={}, exposure_end_t={}, format=\"{}\", offset_x={}, offset_y={}, pixels={})",
                self.t,
                self.begin_t,
                self.end_t,
                self.exposure_begin_t,
                self.exposure_end_t,
                self.format,
                self.offset_x,
                self.offset_y,
                self.pixels.bind(python).repr().map_or_else(
                    |error| error.to_string(),
                    |representation| representation.to_string()
                ),
            )
        })
    }
}

#[pyclass]
pub struct Decoder {
    inner: Option<decoder::Decoder>,
}

#[pymethods]
impl Decoder {
    #[new]
    fn new(path: &pyo3::Bound<'_, pyo3::types::PyAny>) -> PyResult<Self> {
        Python::with_gil(|python| -> PyResult<Self> {
            Ok(Decoder {
                inner: Some(decoder::Decoder::new(types::python_path_to_string(
                    python, path,
                )?)?),
            })
        })
    }

    fn tracks(&self) -> PyResult<Vec<Track>> {
        match self.inner {
            Some(ref decoder) => {
                let mut tracks: Vec<Track> = decoder
                    .id_to_track
                    .iter()
                    .map(|(id, track)| Track {
                        id: *id,
                        data_type: track.to_data_type().to_owned(),
                        dimensions: track.dimensions(),
                    })
                    .collect();
                tracks.sort_by_key(|track| track.id);
                Ok(tracks)
            }
            None => Err(pyo3::exceptions::PyException::new_err(
                "id_to_track called after __exit__",
            )),
        }
    }

    fn description(&self) -> PyResult<&str> {
        match self.inner {
            Some(ref decoder) => Ok(decoder.description()),
            None => Err(pyo3::exceptions::PyException::new_err(
                "document called after __exit__",
            )),
        }
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

    fn __iter__(shell: PyRefMut<Self>) -> PyResult<Py<Decoder>> {
        Ok(shell.into())
    }

    fn __next__(mut shell: PyRefMut<Self>) -> PyResult<Option<(Track, PyObject)>> {
        let packet = match shell.inner {
            Some(ref mut decoder) => match decoder.next() {
                Ok(result) => match result {
                    Some(result) => result,
                    None => return Ok(None),
                },
                Err(result) => return Err(result.into()),
            },
            None => {
                return Err(pyo3::exceptions::PyException::new_err(
                    "__next__ called after __exit__",
                ))
            }
        };
        Python::with_gil(|python| -> PyResult<Option<(Track, PyObject)>> {
            let track = Track {
                id: packet.track_id,
                data_type: packet.track.to_data_type().to_owned(),
                dimensions: packet.track.dimensions(),
            };
            let packet = match packet.track {
                common::Track::Events {
                    dimensions,
                    ref mut previous_t,
                } => {
                    use common::events_generated::size_prefixed_root_as_event_packet;
                    let events = match size_prefixed_root_as_event_packet(packet.buffer) {
                        Ok(result) => match result.elements() {
                            Some(result) => result,
                            None => return Err(decoder::ReadError::EmptyEventsPacket.into()),
                        },
                        Err(_) => return Err(decoder::ReadError::MissingPacketSizePrefix.into()),
                    };
                    let length = events.len() as numpy::npyffi::npy_intp;
                    let array = types::ArrayType::Dvs.new_array(python, length);
                    unsafe {
                        for index in 0..length {
                            let event_cell = types::array_at(python, array, index);
                            let event = events.get(index as usize);
                            let t = event.t().max(*previous_t as i64) as u64;
                            *previous_t = t;
                            let x = event.x();
                            let y = event.y();
                            if x < 0 || x >= dimensions.0 as i16 {
                                return Err(decoder::ReadError::XOverflow {
                                    x,
                                    width: dimensions.0,
                                }
                                .into());
                            }
                            if y < 0 || y >= dimensions.1 as i16 {
                                return Err(decoder::ReadError::YOverflow {
                                    y,
                                    height: dimensions.1,
                                }
                                .into());
                            }
                            let mut event_array = [0u8; 13];
                            event_array[0..8].copy_from_slice(&t.to_le_bytes());
                            event_array[8..10].copy_from_slice(&(x as u16).to_le_bytes());
                            event_array[10..12].copy_from_slice(&(y as u16).to_le_bytes());
                            event_array[12] = if event.on() { 1 } else { 0 };
                            std::ptr::copy(event_array.as_ptr(), event_cell, event_array.len());
                        }
                        PyObject::from_owned_ptr(python, array as *mut pyo3::ffi::PyObject)
                    }
                }
                common::Track::Frame {
                    ref mut previous_t, ..
                } => {
                    let frame =
                        match common::frame_generated::size_prefixed_root_as_frame(packet.buffer) {
                            Ok(result) => result,
                            Err(_) => {
                                return Err(PyErr::from(
                                    decoder::ReadError::MissingPacketSizePrefix,
                                ))
                            }
                        };
                    let t = frame.t().max(*previous_t as i64) as u64;
                    *previous_t = t;
                    Frame {
                        t,
                        begin_t: frame.begin_t(),
                        end_t: frame.end_t(),
                        exposure_begin_t: frame.exposure_begin_t(),
                        exposure_end_t: frame.exposure_end_t(),
                        format: match frame.format() {
                            common::frame_generated::FrameFormat::Gray => "L".to_owned(),
                            common::frame_generated::FrameFormat::Bgr => "RGB".to_owned(),
                            common::frame_generated::FrameFormat::Bgra => "RGBA".to_owned(),
                            _ => return Err(PyErr::from(decoder::ReadError::UnknownFrameFormat)),
                        },
                        offset_x: frame.offset_x(),
                        offset_y: frame.offset_y(),
                        pixels: match frame.format() {
                            common::frame_generated::FrameFormat::Gray => {
                                let dimensions = [frame.height() as usize, frame.width() as usize]
                                    .into_dimension();
                                match frame.pixels() {
                                    Some(result) => result
                                        .bytes()
                                        .to_pyarray_bound(python)
                                        .reshape(dimensions)?
                                        .to_object(python),
                                    None => numpy::array::PyArray2::<u8>::zeros_bound(
                                        python, dimensions, false,
                                    )
                                    .to_object(python),
                                }
                            }
                            common::frame_generated::FrameFormat::Bgr
                            | common::frame_generated::FrameFormat::Bgra => {
                                let channels = if frame.format()
                                    == common::frame_generated::FrameFormat::Bgr
                                {
                                    3_usize
                                } else {
                                    4_usize
                                };
                                let dimensions =
                                    [frame.height() as usize, frame.width() as usize, channels]
                                        .into_dimension();
                                match frame.pixels() {
                                    Some(result) => {
                                        let mut pixels = result.bytes().to_owned();
                                        for index in 0..(pixels.len() / channels) {
                                            pixels.swap(index * channels, index * channels + 2);
                                        }
                                        pixels
                                            .to_pyarray_bound(python)
                                            .reshape(dimensions)?
                                            .to_object(python)
                                    }
                                    None => numpy::array::PyArray3::<u8>::zeros_bound(
                                        python, dimensions, false,
                                    )
                                    .to_object(python),
                                }
                            }
                            _ => return Err(PyErr::from(decoder::ReadError::UnknownFrameFormat)),
                        },
                    }
                    .into_py(python)
                }
                common::Track::Imus { ref mut previous_t } => {
                    let imus = match common::imus_generated::size_prefixed_root_as_imu_packet(
                        packet.buffer,
                    ) {
                        Ok(result) => match result.elements() {
                            Some(result) => result,
                            None => return Err(PyErr::from(decoder::ReadError::EmptyEventsPacket)),
                        },
                        Err(_) => {
                            return Err(PyErr::from(decoder::ReadError::MissingPacketSizePrefix));
                        }
                    };
                    let length = imus.len() as numpy::npyffi::npy_intp;
                    let array = types::ArrayType::AedatImu.new_array(python, length);
                    unsafe {
                        let mut index = 0;
                        for imu in imus {
                            let t = imu.t().max(*previous_t as i64) as u64;
                            *previous_t = t;
                            let imu_cell = types::array_at(python, array, index);
                            let mut imu_array = [0u8; 48];
                            imu_array[0..8].copy_from_slice(&t.to_le_bytes());
                            imu_array[8..12].copy_from_slice(&(imu.temperature()).to_le_bytes());
                            imu_array[12..16]
                                .copy_from_slice(&(imu.accelerometer_x()).to_le_bytes());
                            imu_array[16..20]
                                .copy_from_slice(&(imu.accelerometer_y()).to_le_bytes());
                            imu_array[20..24]
                                .copy_from_slice(&(imu.accelerometer_z()).to_le_bytes());
                            imu_array[24..28].copy_from_slice(&(imu.gyroscope_x()).to_le_bytes());
                            imu_array[28..32].copy_from_slice(&(imu.gyroscope_y()).to_le_bytes());
                            imu_array[32..36].copy_from_slice(&(imu.gyroscope_z()).to_le_bytes());
                            imu_array[36..40]
                                .copy_from_slice(&(imu.magnetometer_x()).to_le_bytes());
                            imu_array[40..44]
                                .copy_from_slice(&(imu.magnetometer_y()).to_le_bytes());
                            imu_array[44..48]
                                .copy_from_slice(&(imu.magnetometer_z()).to_le_bytes());
                            std::ptr::copy(imu_array.as_ptr(), imu_cell, imu_array.len());
                            index += 1;
                        }
                        PyObject::from_owned_ptr(python, array as *mut pyo3::ffi::PyObject)
                    }
                }
                common::Track::Triggers { ref mut previous_t } => {
                    let triggers =
                        match common::triggers_generated::size_prefixed_root_as_trigger_packet(
                            packet.buffer,
                        ) {
                            Ok(result) => match result.elements() {
                                Some(result) => result,
                                None => {
                                    return Err(PyErr::from(decoder::ReadError::EmptyEventsPacket))
                                }
                            },
                            Err(_) => {
                                return Err(PyErr::from(
                                    decoder::ReadError::MissingPacketSizePrefix,
                                ))
                            }
                        };
                    let length = triggers.len() as numpy::npyffi::npy_intp;
                    let array = types::ArrayType::AedatTrigger.new_array(python, length);
                    unsafe {
                        let mut index = 0;
                        for trigger in triggers {
                            let t = trigger.t().max(*previous_t as i64) as u64;
                            *previous_t = t;
                            let trigger_cell = types::array_at(python, array, index);
                            let mut trigger_array = [0u8; 9];
                            trigger_array[0..8].copy_from_slice(&t.to_le_bytes());
                            use common::triggers_generated::TriggerSource;
                            trigger_array[8] = match trigger.source() {
                                TriggerSource::TimestampReset => 0_u8,
                                TriggerSource::ExternalSignalRisingEdge => 1_u8,
                                TriggerSource::ExternalSignalFallingEdge => 2_u8,
                                TriggerSource::ExternalSignalPulse => 3_u8,
                                TriggerSource::ExternalGeneratorRisingEdge => 4_u8,
                                TriggerSource::ExternalGeneratorFallingEdge => 5_u8,
                                TriggerSource::FrameBegin => 6_u8,
                                TriggerSource::FrameEnd => 7_u8,
                                TriggerSource::ExposureBegin => 8_u8,
                                TriggerSource::ExposureEnd => 9_u8,
                                _ => {
                                    return Err(PyErr::from(
                                        decoder::ReadError::UnknownTriggerSource,
                                    ))
                                }
                            };
                            std::ptr::copy(
                                trigger_array.as_ptr(),
                                trigger_cell,
                                trigger_array.len(),
                            );
                            index += 1;
                        }
                        PyObject::from_owned_ptr(python, array as *mut pyo3::ffi::PyObject)
                    }
                }
            };
            Ok(Some((track, packet)))
        })
    }
}

#[pyclass]
pub struct Encoder {
    inner: Option<encoder::Encoder>,
    frame_buffer: Vec<u8>,
}

#[derive(FromPyObject)]
enum DescriptionOrTracks {
    Description(String),
    Tracks(Vec<Track>),
}

#[pymethods]
impl Encoder {
    #[new]
    #[pyo3(signature = (path, description_or_tracks, compression))]
    fn new(
        path: &pyo3::Bound<'_, pyo3::types::PyAny>,
        description_or_tracks: DescriptionOrTracks,
        compression: Option<(String, u8)>,
    ) -> PyResult<Self> {
        Python::with_gil(|python| -> PyResult<Self> {
            Ok(Encoder {
                inner: Some(encoder::Encoder::new(
                    types::python_path_to_string(python, path)?,
                    match &description_or_tracks {
                        DescriptionOrTracks::Description(description) => {
                            encoder::DescriptionOrIdsAndTracks::Description(description.as_str())
                        }
                        DescriptionOrTracks::Tracks(tracks) => {
                            encoder::DescriptionOrIdsAndTracks::IdsAndTracks({
                                let ids_and_tracks: Result<
                                    Vec<(u32, common::Track)>,
                                    common::Error,
                                > = tracks
                                    .iter()
                                    .map(|track| {
                                        common::Track::from_data_type(
                                            &track.data_type,
                                            track.dimensions,
                                        )
                                        .map(|common_track| (track.id, common_track))
                                    })
                                    .collect();
                                ids_and_tracks?
                            })
                        }
                    },
                    encoder::Compression::from_name_and_level(compression)?,
                )?),
                frame_buffer: Vec::new(),
            })
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

    fn write(
        &mut self,
        track_id: u32,
        packet: &pyo3::Bound<'_, pyo3::types::PyAny>,
    ) -> PyResult<()> {
        Python::with_gil(|python| -> PyResult<()> {
            match self.inner.as_mut() {
                Some(encoder) => match encoder.get_track(track_id) {
                    Some(track) => {
                        match track {
                            common::Track::Events {
                                dimensions,
                                ref mut previous_t,
                            } => {
                                let (array, length) =
                                    types::check_array(python, types::ArrayType::Dvs, packet)?;
                                unsafe {
                                    for index in 0..length {
                                        let event_cell: *mut neuromorphic_types::DvsEvent<
                                            u64,
                                            u16,
                                            u16,
                                        > = types::array_at(python, array, index);
                                        let event = *event_cell;
                                        if event.t < *previous_t {
                                            return Err(utilities::WriteError::NonMonotonic {
                                                previous_t: *previous_t,
                                                t: event.t,
                                            }
                                            .into());
                                        }
                                        if event.x >= dimensions.0 {
                                            return Err(utilities::WriteError::XOverflow {
                                                x: event.x,
                                                width: dimensions.0,
                                            }
                                            .into());
                                        }
                                        if event.y >= dimensions.1 {
                                            return Err(utilities::WriteError::YOverflow {
                                                y: event.y,
                                                height: dimensions.1,
                                            }
                                            .into());
                                        }
                                        *previous_t = event.t;
                                    }
                                }
                                encoder.write_events(
                                    track_id,
                                    (0..length).into_iter().map(|index| unsafe {
                                        *types::array_at(python, array, index)
                                    }),
                                )?;
                            }
                            common::Track::Frame {
                                dimensions,
                                ref mut previous_t,
                            } => {
                                let frame_bound: &pyo3::Bound<'_, Frame> = packet.downcast()?;
                                let frame = frame_bound.borrow();
                                if frame.t < *previous_t {
                                    return Err(utilities::WriteError::NonMonotonic {
                                        previous_t: *previous_t,
                                        t: frame.t,
                                    }
                                    .into());
                                }
                                self.frame_buffer.clear();
                                let (frame_format, frame_dimensions) = match frame.format.as_str() {
                                    "L" => {
                                        let array_bound = frame.pixels.downcast_bound::<numpy::PyArray2<u8>>(python)?.readonly();
                                        let array = array_bound.as_array();
                                        let array_dim = array.dim();
                                        if array_dim.1 > dimensions.0 as usize {
                                            return Err(PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(format!(
                                                "the frame width ({}) cannot be larger than the sensor width ({})",
                                                array.dim().1,
                                                dimensions.0
                                            )));
                                        }
                                        if array_dim.0 > dimensions.1 as usize {
                                            return Err(PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(format!(
                                                "the frame height ({}) cannot be larger than the sensor height ({})",
                                                array.dim().0,
                                                dimensions.1
                                            )));
                                        }
                                        self.frame_buffer.reserve(array.len());
                                        for row in array.rows() {
                                            self.frame_buffer.extend(row.iter());
                                        }
                                        (encoder::Format::L, (array_dim.1, array_dim.0))
                                    },
                                    "RGB" | "RGBA" => {
                                        let array_bound = frame.pixels.downcast_bound::<numpy::PyArray3<u8>>(python)?.readonly();
                                        let array = array_bound.as_array();
                                        let array_dim = array.dim();
                                        if array_dim.1 > dimensions.0 as usize {
                                            return Err(PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(format!(
                                                "the frame width ({}) cannot be larger than the sensor width ({})",
                                                array.dim().1,
                                                dimensions.0
                                            )));
                                        }
                                        if array_dim.0 > dimensions.1 as usize {
                                            return Err(PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(format!(
                                                "the frame height ({}) cannot be larger than the sensor height ({})",
                                                array.dim().0,
                                                dimensions.1
                                            )));
                                        }
                                        if frame.format.as_str() == "RGB" {
                                            if array_dim.2 != 3 {
                                                return Err(PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(format!(
                                                    "the frame must have 3 channels (got {})",
                                                    array.dim().2,
                                                )));
                                            }
                                            self.frame_buffer.reserve(array.len());
                                            for subview in array.outer_iter() {
                                                for pixel in subview.rows() {
                                                    self.frame_buffer.extend(&[
                                                        pixel[2],
                                                        pixel[0],
                                                        pixel[1]
                                                    ])
                                                }
                                            }
                                            (encoder::Format::Bgr, (array_dim.1, array_dim.0))
                                        } else {
                                            if array_dim.2 != 4 {
                                                return Err(PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(format!(
                                                    "the frame must have 4 channels (got {})",
                                                    array.dim().2,
                                                )));
                                            }
                                            self.frame_buffer.reserve(array.len());
                                            for subview in array.outer_iter() {
                                                for pixel in subview.rows() {
                                                    self.frame_buffer.extend(&[
                                                        pixel[2],
                                                        pixel[0],
                                                        pixel[1],
                                                        pixel[3],
                                                    ])
                                                }
                                            }
                                            (encoder::Format::Bgra, (array_dim.1, array_dim.0))
                                        }
                                    },
                                    frame_format => return Err(PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(format!(
                                        "unknown format \"{frame_format}\" (expected \"L\", \"RGB\", or \"RGBA\")"
                                    ))),
                                };
                                *previous_t = frame.t;
                                encoder.write_frame(
                                    track_id,
                                    frame.t,
                                    frame.begin_t,
                                    frame.end_t,
                                    frame.exposure_begin_t,
                                    frame.exposure_end_t,
                                    frame_format,
                                    frame_dimensions.0 as i16,
                                    frame_dimensions.1 as i16,
                                    frame.offset_x,
                                    frame.offset_y,
                                    &self.frame_buffer,
                                )?;
                            }
                            common::Track::Imus { ref mut previous_t } => {
                                let (array, length) =
                                    types::check_array(python, types::ArrayType::AedatImu, packet)?;
                                unsafe {
                                    for index in 0..length {
                                        let imu_cell: *mut encoder::Imu =
                                            types::array_at(python, array, index);
                                        let imu = *imu_cell;
                                        if imu.t < *previous_t {
                                            return Err(utilities::WriteError::NonMonotonic {
                                                previous_t: *previous_t,
                                                t: imu.t,
                                            }
                                            .into());
                                        }
                                        *previous_t = imu.t;
                                    }
                                }
                                encoder.write_imus(
                                    track_id,
                                    (0..length).into_iter().map(|index| unsafe {
                                        *types::array_at(python, array, index)
                                    }),
                                )?;
                            }
                            common::Track::Triggers { ref mut previous_t } => {
                                let (array, length) = types::check_array(
                                    python,
                                    types::ArrayType::AedatTrigger,
                                    packet,
                                )?;
                                unsafe {
                                    for index in 0..length {
                                        let trigger_cell: *mut encoder::Trigger =
                                            types::array_at(python, array, index);
                                        let trigger = *trigger_cell;
                                        if trigger.t < *previous_t {
                                            return Err(utilities::WriteError::NonMonotonic {
                                                previous_t: *previous_t,
                                                t: trigger.t,
                                            }
                                            .into());
                                        }
                                        if trigger.source >= 128 {
                                            return Err(utilities::WriteError::TriggerOverflow {
                                                id: trigger.source,
                                                maximum: 128,
                                            }
                                            .into());
                                        }
                                        *previous_t = trigger.t;
                                    }
                                }
                                encoder.write_triggers(
                                    track_id,
                                    (0..length).into_iter().map(|index| unsafe {
                                        *types::array_at(python, array, index)
                                    }),
                                )?;
                            }
                        }
                        Ok(())
                    }
                    None => Err(PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(format!(
                        "unknown track ID {track_id}"
                    ))),
                },
                None => Err(pyo3::exceptions::PyException::new_err(
                    "write called after __exit__",
                )),
            }
        })
    }
}
