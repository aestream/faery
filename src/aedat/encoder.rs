use std::io::Seek;
use std::io::Write;

use crate::aedat::common;
use crate::utilities;

pub const MAXIMUM_EVENTS_PER_BUFFER: usize = 1 << 12;
pub const MAXIMUM_IMUS_PER_BUFFER: usize = 1 << 10;
pub const MAXIMUM_TRIGGERS_PER_BUFFER: usize = 1 << 12;

pub struct Encoder {
    file: std::io::BufWriter<std::fs::File>,
    id_to_track: std::collections::HashMap<i32, common::Track>,
    compression: Compression,
    builder_buffer: Option<Vec<u8>>,
    buffer: Vec<u8>,
    file_data_position_offset: u64,
    file_data_position: u64,
    file_data_definitions: Vec<common::FileDataDefinition>,
}

#[derive(thiserror::Error, Debug)]
pub enum Error {
    #[error(transparent)]
    Io(#[from] std::io::Error),

    #[error(transparent)]
    Fmt(#[from] std::fmt::Error),

    #[error(transparent)]
    Description(#[from] common::DescriptionError),
}

#[derive(thiserror::Error, Debug)]
pub enum CompressionError {
    #[error("unknown compression \"{0}\"")]
    Unknown(String),

    #[error(
        "unsupported compression level {value} (\"{name}\" supports levels {minimum} to {maximum})"
    )]
    Level {
        value: u8,
        name: String,
        minimum: u8,
        maximum: u8,
    },
}

#[derive(Clone, Copy, PartialEq, Eq)]
pub enum Compression {
    None,
    Lz4(u8),
    Zstd(u8),
}

impl Compression {
    pub fn from_name_and_level(
        name_and_level: Option<(String, u8)>,
    ) -> Result<Self, CompressionError> {
        match name_and_level {
            Some((name, level)) => match name.as_str() {
                "lz4" => {
                    if level < utilities::LZ4_MINIMUM_LEVEL || level > utilities::LZ4_MAXIMUM_LEVEL
                    {
                        Err(CompressionError::Level {
                            value: level,
                            name: name.to_owned(),
                            minimum: utilities::LZ4_MINIMUM_LEVEL,
                            maximum: utilities::LZ4_MAXIMUM_LEVEL,
                        })
                    } else {
                        Ok(Self::Lz4(level))
                    }
                }
                "zstd" => {
                    if level < utilities::ZSTD_MINIMUM_LEVEL
                        || level > utilities::ZSTD_MAXIMUM_LEVEL
                    {
                        Err(CompressionError::Level {
                            value: level,
                            name: name.to_owned(),
                            minimum: utilities::ZSTD_MINIMUM_LEVEL,
                            maximum: utilities::ZSTD_MAXIMUM_LEVEL,
                        })
                    } else {
                        Ok(Self::Zstd(level))
                    }
                }
                name => Err(CompressionError::Unknown(name.to_owned())),
            },
            None => Ok(Self::None),
        }
    }
}

#[derive(thiserror::Error, Debug)]
pub enum PacketError {
    #[error(transparent)]
    Io(#[from] std::io::Error),

    #[error("too many elements (got {count} but the maximum is {maximum})")]
    ElementsOverflow { count: usize, maximum: usize },

    #[error("start_t ({start_t:?}) must be smaller than or equal to end_t {end_t:?}")]
    FileDataDefinition {
        start_t: Option<u64>,
        end_t: Option<u64>,
    },
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Format {
    L,
    Bgr,
    Bgra,
}

#[repr(C, packed)]
#[derive(Debug, Copy, Clone)]
pub struct Imu {
    pub t: u64,
    pub temperature: f32,
    pub accelerometer_x: f32,
    pub accelerometer_y: f32,
    pub accelerometer_z: f32,
    pub gyroscope_x: f32,
    pub gyroscope_y: f32,
    pub gyroscope_z: f32,
    pub magnetometer_x: f32,
    pub magnetometer_y: f32,
    pub magnetometer_z: f32,
}

#[repr(C, packed)]
#[derive(Debug, Copy, Clone)]
pub struct Trigger {
    pub t: u64,
    pub source: u8,
}

struct PacketInformation {
    track_id: i32,
    elements_count: usize,
    start_t: u64,
    end_t: u64,
}

impl PacketInformation {
    fn new(
        track_id: i32,
        elements_count: usize,
        start_t: Option<u64>,
        end_t: Option<u64>,
    ) -> Result<Self, PacketError> {
        match start_t {
            Some(some_start_t) => match end_t {
                Some(some_end_t) => {
                    if some_start_t > some_end_t {
                        Err(PacketError::FileDataDefinition { start_t, end_t })
                    } else {
                        Ok(Self {
                            track_id,
                            elements_count,
                            start_t: some_start_t,
                            end_t: some_end_t,
                        })
                    }
                }
                None => Err(PacketError::FileDataDefinition { start_t, end_t }),
            },
            None => Err(PacketError::FileDataDefinition { start_t, end_t }),
        }
    }
}

impl Encoder {
    fn write_description(
        file: &mut std::io::BufWriter<std::fs::File>,
        compression: Compression,
        builder: &mut flatbuffers::FlatBufferBuilder,
        description: &str,
    ) -> Result<(u64, u64), std::io::Error> {
        let flatbuffer_description = builder.create_string(description);
        let ioheader = common::io_header_generated::IOHeader::create(
            builder,
            &common::io_header_generated::IOHeaderArgs {
                compression: match compression {
                    Compression::None => common::io_header_generated::Compression::None,
                    Compression::Lz4 { .. } => common::io_header_generated::Compression::Lz4,
                    Compression::Zstd { .. } => common::io_header_generated::Compression::Zstd,
                },
                file_data_position: -1,
                description: Some(flatbuffer_description),
            },
        );
        builder.finish_size_prefixed(
            ioheader,
            Some(common::io_header_generated::IOHEADER_IDENTIFIER),
        );
        let data = builder.finished_data();
        file.write_all(data)?;
        let ioheader =
            unsafe { common::io_header_generated::root_as_ioheader_unchecked(&data[4..]) };
        let offset = 4
            + ioheader._tab.loc()
            + ioheader
                ._tab
                .vtable()
                .get(common::io_header_generated::IOHeader::VT_FILE_DATA_POSITION)
                as usize;
        Ok((offset as u64, data.len() as u64))
    }

    pub fn new<P: AsRef<std::path::Path>>(
        path: P,
        description: common::Description,
        compression: Compression,
    ) -> Result<Self, Error> {
        let mut file = std::io::BufWriter::new(std::fs::File::create(path)?);
        file.write_all(common::MAGIC_NUMBER.as_bytes())?;
        let mut builder = flatbuffers::FlatBufferBuilder::with_capacity(utilities::BUFFER_SIZE);
        builder.force_defaults(true);
        let id_to_track = description.id_to_track()?;
        let (file_data_position_offset, file_data_position) = {
            let (file_data_position_offset, file_data_position) = Self::write_description(
                &mut file,
                compression,
                &mut builder,
                &description.to_xml_string(),
            )?;
            (
                common::MAGIC_NUMBER.len() as u64 + file_data_position_offset,
                common::MAGIC_NUMBER.len() as u64 + file_data_position,
            )
        };
        let (mut builder_buffer, _) = builder.collapse();
        builder_buffer.fill(0);
        Ok(Self {
            file,
            id_to_track,
            compression,
            builder_buffer: Some(builder_buffer),
            file_data_position_offset,
            file_data_position,
            buffer: Vec::new(),
            file_data_definitions: Vec::new(),
        })
    }

    pub fn get_track(&mut self, track_id: i32) -> Option<&mut common::Track> {
        self.id_to_track.get_mut(&track_id)
    }

    fn write_file_data_position(&mut self) -> Result<(), std::io::Error> {
        self.file
            .seek(std::io::SeekFrom::Start(self.file_data_position_offset))?;
        self.file
            .write_all(&(self.file_data_position as i64).to_le_bytes())?;
        self.file.seek(std::io::SeekFrom::End(0))?;
        Ok(())
    }

    fn compress_and_write(
        &mut self,
        data: &[u8],
        packet_information: Option<PacketInformation>,
    ) -> Result<(), PacketError> {
        let byte_offset = self.file_data_position + 8;
        let size = match self.compression {
            Compression::None => {
                if let Some(packet_information) = packet_information.as_ref() {
                    self.file
                        .write_all(&packet_information.track_id.to_le_bytes())?;
                    self.file.write_all(&(data.len() as u32).to_le_bytes())?;
                }
                self.file.write_all(data)?;
                if packet_information.is_some() {
                    self.file_data_position += 8 + data.len() as u64;
                }
                data.len()
            }
            Compression::Lz4(level) => {
                self.buffer.clear();
                let mut encoder = lz4::EncoderBuilder::new()
                    .level(level as u32)
                    .build(&mut self.buffer)?;
                encoder.write_all(data)?;
                encoder.finish().1?;
                if let Some(packet_information) = packet_information.as_ref() {
                    self.file
                        .write_all(&packet_information.track_id.to_le_bytes())?;
                    self.file
                        .write_all(&(self.buffer.len() as u32).to_le_bytes())?;
                }
                self.file.write_all(&self.buffer)?;
                if packet_information.is_some() {
                    self.file_data_position += 8 + self.buffer.len() as u64;
                }
                self.buffer.len()
            }
            Compression::Zstd(level) => {
                self.buffer.clear();
                self.buffer
                    .resize(zstd::zstd_safe::compress_bound(data.len()), 0u8);
                let mut compressor = zstd::bulk::Compressor::new(level as i32)?;
                compressor.compress_to_buffer(data, &mut self.buffer)?;
                if let Some(packet_information) = packet_information.as_ref() {
                    self.file
                        .write_all(&packet_information.track_id.to_le_bytes())?;
                    self.file
                        .write_all(&(self.buffer.len() as u32).to_le_bytes())?;
                }
                self.file.write_all(&self.buffer)?;
                if packet_information.is_some() {
                    self.file_data_position += 8 + self.buffer.len() as u64;
                }
                self.buffer.len()
            }
        };
        if let Some(packet_information) = packet_information {
            self.file_data_definitions.push(common::FileDataDefinition {
                byte_offset: byte_offset as i64,
                track_id: packet_information.track_id,
                size: size as i32,
                elements_count: packet_information.elements_count as i64,
                start_t: packet_information.start_t as i64,
                end_t: packet_information.end_t as i64,
            });
        }
        Ok(())
    }

    fn write_events_with_builder<EventIterator>(
        &mut self,
        track_id: i32,
        events: EventIterator,
        builder: &mut flatbuffers::FlatBufferBuilder,
    ) -> Result<(), PacketError>
    where
        EventIterator: ExactSizeIterator<Item = neuromorphic_types::DvsEvent<u64, u16, u16>>
            + DoubleEndedIterator<Item = neuromorphic_types::DvsEvent<u64, u16, u16>>,
    {
        let count = events.len();
        if count > MAXIMUM_EVENTS_PER_BUFFER {
            return Err(PacketError::ElementsOverflow {
                count,
                maximum: MAXIMUM_EVENTS_PER_BUFFER,
            });
        }
        if count == 0 {
            return Ok(());
        }
        let mut start_t = None;
        let mut end_t = None;

        // create_vector_from_iter calls .rev()
        // the following map function is thus called from last event to first event
        let vector = builder.create_vector_from_iter(events.map(|event| {
            if end_t.is_none() {
                end_t = Some(event.t);
            }
            start_t = Some(event.t);
            common::events_generated::Event::new(
                event.t as i64,
                event.x as i16,
                event.y as i16,
                matches!(event.polarity, neuromorphic_types::Polarity::On),
            )
        }));
        let packet = common::events_generated::EventPacket::create(
            builder,
            &common::events_generated::EventPacketArgs {
                elements: Some(vector),
            },
        );
        builder.finish_size_prefixed(
            packet,
            Some(common::events_generated::EVENT_PACKET_IDENTIFIER),
        );
        self.compress_and_write(
            builder.finished_data(),
            Some(PacketInformation::new(track_id, count, start_t, end_t)?),
        )
    }

    pub fn write_events<EventIterator>(
        &mut self,
        track_id: i32,
        events: EventIterator,
    ) -> Result<(), PacketError>
    where
        EventIterator: ExactSizeIterator<Item = neuromorphic_types::DvsEvent<u64, u16, u16>>
            + DoubleEndedIterator<Item = neuromorphic_types::DvsEvent<u64, u16, u16>>,
    {
        let mut builder = flatbuffers::FlatBufferBuilder::from_vec(
            self.builder_buffer
                .take()
                .expect("builder_buffer is not taken"),
        );
        let result = self.write_events_with_builder(track_id, events, &mut builder);
        let (mut builder_buffer, _) = builder.collapse();
        builder_buffer.fill(0);
        self.builder_buffer.replace(builder_buffer);
        result
    }

    fn write_frame_with_builder(
        &mut self,
        track_id: i32,
        t: u64,
        start_t: i64,
        end_t: i64,
        exposure_start_t: i64,
        exposure_end_t: i64,
        format: Format,
        width: i16,
        height: i16,
        offset_x: i16,
        offset_y: i16,
        pixels: &[u8],
        builder: &mut flatbuffers::FlatBufferBuilder,
    ) -> Result<(), PacketError> {
        let pixels_vector = builder.create_vector(pixels);
        let packet = common::frame_generated::Frame::create(
            builder,
            &common::frame_generated::FrameArgs {
                t: t as i64,
                start_t,
                end_t,
                exposure_start_t,
                exposure_end_t,
                format: match format {
                    Format::L => common::frame_generated::FrameFormat::Gray,
                    Format::Bgr => common::frame_generated::FrameFormat::Bgr,
                    Format::Bgra => common::frame_generated::FrameFormat::Bgra,
                },
                width,
                height,
                offset_x,
                offset_y,
                pixels: Some(pixels_vector),
            },
        );
        builder.finish_size_prefixed(packet, Some(common::frame_generated::FRAME_IDENTIFIER));
        self.compress_and_write(
            builder.finished_data(),
            Some(PacketInformation::new(track_id, 1, Some(t), Some(t))?),
        )
    }

    pub fn write_frame(
        &mut self,
        track_id: i32,
        t: u64,
        start_t: i64,
        end_t: i64,
        exposure_start_t: i64,
        exposure_end_t: i64,
        format: Format,
        width: i16,
        height: i16,
        offset_x: i16,
        offset_y: i16,
        pixels: &[u8],
    ) -> Result<(), PacketError> {
        let mut builder = flatbuffers::FlatBufferBuilder::from_vec(
            self.builder_buffer
                .take()
                .expect("builder_buffer is not taken"),
        );
        let result = self.write_frame_with_builder(
            track_id,
            t,
            start_t,
            end_t,
            exposure_start_t,
            exposure_end_t,
            format,
            width,
            height,
            offset_x,
            offset_y,
            pixels,
            &mut builder,
        );
        let (mut builder_buffer, _) = builder.collapse();
        builder_buffer.fill(0);
        self.builder_buffer.replace(builder_buffer);
        result
    }

    fn write_imus_with_builder<ImuIterator>(
        &mut self,
        track_id: i32,
        imus: ImuIterator,
        builder: &mut flatbuffers::FlatBufferBuilder,
    ) -> Result<(), PacketError>
    where
        ImuIterator: ExactSizeIterator<Item = Imu> + DoubleEndedIterator<Item = Imu>,
    {
        let count = imus.len();
        if count > MAXIMUM_IMUS_PER_BUFFER {
            return Err(PacketError::ElementsOverflow {
                count,
                maximum: MAXIMUM_IMUS_PER_BUFFER,
            });
        }
        if count == 0 {
            return Ok(());
        }
        self.buffer.clear();
        self.buffer.resize(
            count * std::mem::size_of::<flatbuffers::WIPOffset<common::imus_generated::Imu>>(),
            0,
        );
        let imus_offsets = unsafe {
            std::slice::from_raw_parts_mut(
                self.buffer.as_mut_ptr()
                    as *mut flatbuffers::WIPOffset<common::imus_generated::Imu>,
                count,
            )
        };
        let mut start_t = None;
        let mut end_t = None;
        for (index, imu) in imus.enumerate() {
            if start_t.is_none() {
                start_t = Some(imu.t);
            }
            end_t = Some(imu.t);
            imus_offsets[index] = common::imus_generated::Imu::create(
                builder,
                &common::imus_generated::ImuArgs {
                    t: imu.t as i64,
                    temperature: imu.temperature,
                    accelerometer_x: imu.accelerometer_x,
                    accelerometer_y: imu.accelerometer_y,
                    accelerometer_z: imu.accelerometer_z,
                    gyroscope_x: imu.gyroscope_x,
                    gyroscope_y: imu.gyroscope_y,
                    gyroscope_z: imu.gyroscope_z,
                    magnetometer_x: imu.magnetometer_x,
                    magnetometer_y: imu.magnetometer_y,
                    magnetometer_z: imu.magnetometer_z,
                },
            );
        }
        let vector = builder.create_vector(imus_offsets);
        let packet = common::imus_generated::ImuPacket::create(
            builder,
            &common::imus_generated::ImuPacketArgs {
                elements: Some(vector),
            },
        );
        builder.finish_size_prefixed(packet, Some(common::imus_generated::IMU_PACKET_IDENTIFIER));
        self.compress_and_write(
            builder.finished_data(),
            Some(PacketInformation::new(track_id, count, start_t, end_t)?),
        )
    }

    pub fn write_imus<ImuIterator>(
        &mut self,
        track_id: i32,
        imus: ImuIterator,
    ) -> Result<(), PacketError>
    where
        ImuIterator: ExactSizeIterator<Item = Imu> + DoubleEndedIterator<Item = Imu>,
    {
        let mut builder = flatbuffers::FlatBufferBuilder::from_vec(
            self.builder_buffer
                .take()
                .expect("builder_buffer is not taken"),
        );
        let result = self.write_imus_with_builder(track_id, imus, &mut builder);
        let (mut builder_buffer, _) = builder.collapse();
        builder_buffer.fill(0);
        self.builder_buffer.replace(builder_buffer);
        result
    }

    fn write_triggers_with_builder<TriggerIterator>(
        &mut self,
        track_id: i32,
        triggers: TriggerIterator,
        builder: &mut flatbuffers::FlatBufferBuilder,
    ) -> Result<(), PacketError>
    where
        TriggerIterator: ExactSizeIterator<Item = Trigger> + DoubleEndedIterator<Item = Trigger>,
    {
        let count = triggers.len();
        if count > MAXIMUM_TRIGGERS_PER_BUFFER {
            return Err(PacketError::ElementsOverflow {
                count,
                maximum: MAXIMUM_TRIGGERS_PER_BUFFER,
            });
        }
        if count == 0 {
            return Ok(());
        }
        self.buffer.clear();
        self.buffer.resize(
            count
                * std::mem::size_of::<flatbuffers::WIPOffset<common::triggers_generated::Trigger>>(
                ),
            0,
        );
        let triggers_offsets = unsafe {
            std::slice::from_raw_parts_mut(
                self.buffer.as_mut_ptr()
                    as *mut flatbuffers::WIPOffset<common::triggers_generated::Trigger>,
                count,
            )
        };
        let mut start_t = None;
        let mut end_t = None;
        for (index, trigger) in triggers.enumerate() {
            if start_t.is_none() {
                start_t = Some(trigger.t);
            }
            end_t = Some(trigger.t);
            triggers_offsets[index] = common::triggers_generated::Trigger::create(
                builder,
                &common::triggers_generated::TriggerArgs {
                    t: trigger.t as i64,
                    source: common::triggers_generated::TriggerSource(trigger.source as i8),
                },
            );
        }
        let vector = builder.create_vector(triggers_offsets);
        let packet = common::triggers_generated::TriggerPacket::create(
            builder,
            &common::triggers_generated::TriggerPacketArgs {
                elements: Some(vector),
            },
        );
        builder.finish_size_prefixed(
            packet,
            Some(common::triggers_generated::TRIGGER_PACKET_IDENTIFIER),
        );
        self.compress_and_write(
            builder.finished_data(),
            Some(PacketInformation::new(track_id, count, start_t, end_t)?),
        )
    }

    pub fn write_triggers<TriggerIterator>(
        &mut self,
        track_id: i32,
        triggers: TriggerIterator,
    ) -> Result<(), PacketError>
    where
        TriggerIterator: ExactSizeIterator<Item = Trigger> + DoubleEndedIterator<Item = Trigger>,
    {
        let mut builder = flatbuffers::FlatBufferBuilder::from_vec(
            self.builder_buffer
                .take()
                .expect("builder_buffer is not taken"),
        );
        let result = self.write_triggers_with_builder(track_id, triggers, &mut builder);
        let (mut builder_buffer, _) = builder.collapse();
        builder_buffer.fill(0);
        self.builder_buffer.replace(builder_buffer);
        result
    }

    fn write_file_data_definitions_with_builder(
        &mut self,
        builder: &mut flatbuffers::FlatBufferBuilder,
    ) -> Result<(), PacketError> {
        self.buffer.clear();
        self.buffer.resize(
            self.file_data_definitions.len()
                * std::mem::size_of::<
                    flatbuffers::WIPOffset<common::file_data_table_generated::FileDataDefinition>,
                >(),
            0,
        );
        let file_data_definitions_offsets = unsafe {
            std::slice::from_raw_parts_mut(
                self.buffer.as_mut_ptr()
                    as *mut flatbuffers::WIPOffset<
                        common::file_data_table_generated::FileDataDefinition,
                    >,
                self.file_data_definitions.len(),
            )
        };
        for (index, file_data_definition) in self.file_data_definitions.iter().enumerate() {
            file_data_definitions_offsets[index] =
                common::file_data_table_generated::FileDataDefinition::create(
                    builder,
                    &common::file_data_table_generated::FileDataDefinitionArgs {
                        byte_offset: file_data_definition.byte_offset,
                        packet_header: Some(&common::file_data_table_generated::PacketHeader::new(
                            file_data_definition.track_id,
                            file_data_definition.size,
                        )),
                        elements_count: file_data_definition.elements_count,
                        start_t: file_data_definition.start_t,
                        end_t: file_data_definition.end_t,
                    },
                );
        }
        let vector = builder.create_vector(file_data_definitions_offsets);
        let packet = common::file_data_table_generated::FileDataTable::create(
            builder,
            &common::file_data_table_generated::FileDataTableArgs {
                file_data_definitions: Some(vector),
            },
        );
        builder.finish_size_prefixed(
            packet,
            Some(common::file_data_table_generated::FILE_DATA_TABLE_IDENTIFIER),
        );
        self.compress_and_write(builder.finished_data(), None)
    }
}

impl Drop for Encoder {
    fn drop(&mut self) {
        let _ = self.write_file_data_position();
        if self
            .file
            .seek(std::io::SeekFrom::Start(self.file_data_position))
            .is_ok()
        {
            let mut builder = flatbuffers::FlatBufferBuilder::from_vec(
                self.builder_buffer
                    .take()
                    .expect("builder_buffer is not taken"),
            );
            let _ = self.write_file_data_definitions_with_builder(&mut builder);
        }
    }
}
