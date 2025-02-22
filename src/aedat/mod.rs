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
#[derive(Clone)]
pub struct Track {
    #[pyo3(get, set)]
    pub id: i32,
    #[pyo3(get, set)]
    pub data_type: String,
    #[pyo3(get, set)]
    pub dimensions: Option<(u16, u16)>,
}

#[pymethods]
impl Track {
    #[new]
    #[pyo3(signature = (id, data_type, dimensions))]
    fn new(id: i32, data_type: String, dimensions: Option<(u16, u16)>) -> Self {
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
    start_t: i64,
    #[pyo3(get)]
    end_t: i64,
    #[pyo3(get)]
    exposure_start_t: i64,
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
                "faery.aedat.Frame(t={}, start_t={}, end_t={}, exposure_start_t={}, exposure_end_t={}, format=\"{}\", offset_x={}, offset_y={}, pixels={})",
                self.t,
                self.start_t,
                self.end_t,
                self.exposure_start_t,
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
#[derive(Clone)]
pub struct FileDataDefinition {
    #[pyo3(get)]
    pub byte_offset: i64,
    #[pyo3(get)]
    pub track_id: i32,
    #[pyo3(get)]
    pub size: i32,
    #[pyo3(get)]
    pub elements_count: i64,
    #[pyo3(get)]
    pub start_t: i64,
    #[pyo3(get)]
    pub end_t: i64,
}

#[pymethods]
impl FileDataDefinition {
    fn __repr__(&self) -> String {
        format!(
            "faery.aedat.FileDataDefinition(byte_offset={}, track_id={}, size={}, elements_count={}, start_t={}, end_t={})",
            self.byte_offset,
            self.track_id,
            self.size,
            self.elements_count,
            self.start_t,
            self.end_t,
        )
    }
}

#[pyclass(eq)]
#[derive(Clone, PartialEq, Eq)]
pub enum DescriptionAttribute {
    #[pyo3(name = "string")]
    String(String),
    #[pyo3(name = "int")]
    Int(i32),
    #[pyo3(name = "long")]
    Long(i64),
}

impl From<&common::DescriptionAttribute> for DescriptionAttribute {
    fn from(attribute: &common::DescriptionAttribute) -> Self {
        match attribute {
            common::DescriptionAttribute::String(value) => Self::String(value.clone()),
            common::DescriptionAttribute::Int(value) => Self::Int(*value),
            common::DescriptionAttribute::Long(value) => Self::Long(*value),
        }
    }
}

impl From<DescriptionAttribute> for common::DescriptionAttribute {
    fn from(attribute: DescriptionAttribute) -> Self {
        match attribute {
            DescriptionAttribute::String(value) => Self::String(value),
            DescriptionAttribute::Int(value) => Self::Int(value),
            DescriptionAttribute::Long(value) => Self::Long(value),
        }
    }
}

#[pymethods]
impl DescriptionAttribute {
    #[new]
    fn new(attribute_type: &str, value: &pyo3::Bound<'_, pyo3::types::PyAny>) -> PyResult<Self> {
        match attribute_type {
            "string" => Ok(Self::String(value.downcast::<pyo3::types::PyString>()?.extract()?)),
            "int" => Ok(Self::Int(value.downcast::<pyo3::types::PyInt>()?.extract()?)),
            "long" => Ok(Self::Long(value.downcast::<pyo3::types::PyInt>()?.extract()?)),
            attribute_type => Err(PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(format!(
                "unexpected attribute type \"{attribute_type}\" (expected \"string\", \"int\", or \"long\")"
            )))

        }
    }

    #[getter]
    fn attribute_type(&self) -> String {
        match self {
            DescriptionAttribute::String(_) => "string",
            DescriptionAttribute::Int(_) => "int",
            DescriptionAttribute::Long(_) => "long",
        }
        .to_owned()
    }

    #[getter]
    fn value(&self) -> PyResult<PyObject> {
        Python::with_gil(|python| -> PyResult<PyObject> {
            Ok(match self {
                DescriptionAttribute::String(value) => {
                    value.into_pyobject(python)?.unbind().into_any()
                }
                DescriptionAttribute::Int(value) => {
                    value.into_pyobject(python)?.unbind().into_any()
                }
                DescriptionAttribute::Long(value) => {
                    value.into_pyobject(python)?.unbind().into_any()
                }
            })
        })
    }

    fn __repr__(&self) -> String {
        match self {
            DescriptionAttribute::String(value) => {
                format!("faery.aedat.DescriptionAttribute(attribute_type=\"string\", value=\"{value}\")")
            }
            DescriptionAttribute::Int(value) => {
                format!("faery.aedat.DescriptionAttribute(attribute_type=\"int\", value={value})")
            }
            DescriptionAttribute::Long(value) => {
                format!("faery.aedat.DescriptionAttribute(attribute_type=\"long\", value={value})")
            }
        }
    }
}

#[pyclass(eq)]
#[derive(Clone, PartialEq, Eq)]
pub struct DescriptionNode {
    #[pyo3(get, set)]
    pub name: String,
    #[pyo3(get, set)]
    pub path: String,
    #[pyo3(get, set)]
    pub attributes: std::collections::HashMap<String, DescriptionAttribute>,
    #[pyo3(get, set)]
    pub nodes: Vec<DescriptionNode>,
}

impl From<&common::DescriptionNode> for DescriptionNode {
    fn from(node: &common::DescriptionNode) -> Self {
        Self {
            name: node.name.clone(),
            path: node.path.clone(),
            attributes: node
                .attributes
                .iter()
                .map(|(key, attribute)| (key.clone(), attribute.into()))
                .collect(),
            nodes: node.nodes.iter().map(|node| node.into()).collect(),
        }
    }
}

impl From<DescriptionNode> for common::DescriptionNode {
    fn from(node: DescriptionNode) -> Self {
        Self {
            name: node.name,
            path: node.path,
            attributes: node
                .attributes
                .into_iter()
                .map(|(key, attribute)| (key.clone(), attribute.into()))
                .collect(),
            nodes: node.nodes.into_iter().map(|node| node.into()).collect(),
        }
    }
}

#[pymethods]
impl DescriptionNode {
    #[new]
    fn new(
        name: String,
        path: String,
        attributes: std::collections::HashMap<String, DescriptionAttribute>,
        nodes: Vec<DescriptionNode>,
    ) -> Self {
        Self {
            name,
            path,
            attributes,
            nodes,
        }
    }

    fn __repr__(&self) -> String {
        let mut key_and_attribute: Vec<_> = self.attributes.iter().collect();
        key_and_attribute.sort_by(|a, b| a.0.cmp(b.0));
        format!(
            "faery.aedat.DescriptionNode(name=\"{}\", path=\"{}\", attributes={{{}}}, nodes=[{}])",
            self.name,
            self.path,
            key_and_attribute
                .iter()
                .map(|(key, attribute)| format!("'{}': {}", key, attribute.__repr__()))
                .collect::<Vec<_>>()
                .join(", "),
            self.nodes
                .iter()
                .map(|node| node.__repr__())
                .collect::<Vec<_>>()
                .join(", "),
        )
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
        Ok(Decoder {
            inner: Some(decoder::Decoder::new(types::python_path_to_string(path)?)?),
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

    fn description(&self) -> PyResult<Vec<DescriptionNode>> {
        match self.inner {
            Some(ref decoder) => Ok(decoder
                .description
                .0
                .iter()
                .map(|description| description.into())
                .collect()),
            None => Err(pyo3::exceptions::PyException::new_err(
                "description called after __exit__",
            )),
        }
    }

    fn file_data_definitions(&self) -> PyResult<Vec<FileDataDefinition>> {
        match self.inner {
            Some(ref decoder) => Ok(decoder
                .file_data_definitions
                .iter()
                .map(|file_data_definition| FileDataDefinition {
                    byte_offset: file_data_definition.byte_offset,
                    track_id: file_data_definition.track_id,
                    size: file_data_definition.size,
                    elements_count: file_data_definition.elements_count,
                    start_t: file_data_definition.start_t,
                    end_t: file_data_definition.end_t,
                })
                .collect()),
            None => Err(pyo3::exceptions::PyException::new_err(
                "file_data_definitions called after __exit__",
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
                        start_t: frame.start_t(),
                        end_t: frame.end_t(),
                        exposure_start_t: frame.exposure_start_t(),
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
                                        .to_pyarray(python)
                                        .reshape(dimensions)?
                                        .unbind()
                                        .into_any(),
                                    None => numpy::array::PyArray2::<u8>::zeros(
                                        python, dimensions, false,
                                    )
                                    .unbind()
                                    .into_any(),
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
                                            .to_pyarray(python)
                                            .reshape(dimensions)?
                                            .unbind()
                                            .into_any()
                                    }
                                    None => numpy::array::PyArray3::<u8>::zeros(
                                        python, dimensions, false,
                                    )
                                    .unbind()
                                    .into_any(),
                                }
                            }
                            _ => return Err(PyErr::from(decoder::ReadError::UnknownFrameFormat)),
                        },
                    }
                    .into_pyobject(python)?
                    .unbind()
                    .into_any()
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

#[pymethods]
impl Encoder {
    #[new]
    #[pyo3(signature = (path, description, compression))]
    fn new(
        path: &pyo3::Bound<'_, pyo3::types::PyAny>,
        description: Vec<DescriptionNode>,
        compression: Option<(String, u8)>,
    ) -> PyResult<Self> {
        Ok(Encoder {
            inner: Some(encoder::Encoder::new(
                types::python_path_to_string(path)?,
                common::Description(description.into_iter().map(|node| node.into()).collect()),
                encoder::Compression::from_name_and_level(compression)?,
            )?),
            frame_buffer: Vec::new(),
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
        track_id: i32,
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
                                let mut offset = 0;
                                while offset < length {
                                    encoder.write_events(
                                        track_id,
                                        (offset
                                            ..(offset
                                                + encoder::MAXIMUM_EVENTS_PER_BUFFER as isize)
                                                .min(length))
                                            .into_iter()
                                            .map(|index| unsafe {
                                                *types::array_at(python, array, index)
                                            }),
                                    )?;
                                    offset += encoder::MAXIMUM_EVENTS_PER_BUFFER as isize;
                                }
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
                                    frame.start_t,
                                    frame.end_t,
                                    frame.exposure_start_t,
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
