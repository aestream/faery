use std::io::Seek;
use std::io::Write;

use crate::aedat::common;
use crate::utilities;

pub struct Encoder {
    file: std::io::BufWriter<std::fs::File>,
    id_to_track: std::collections::HashMap<u32, common::Track>,
    compression: Compression,
    builder_buffer: Option<Vec<u8>>,
    buffer: Vec<u8>,
    file_data_position_offset: u64,
    file_data_position: u64,
}

pub enum DescriptionOrIdsAndTracks<'a> {
    Description(&'a str),
    IdsAndTracks(Vec<(u32, common::Track)>),
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

impl Encoder {
    fn write_description(
        file: &mut std::io::BufWriter<std::fs::File>,
        compression: Compression,
        builder: &mut flatbuffers::FlatBufferBuilder,
        description: &str,
    ) -> Result<(u64, u64), std::io::Error> {
        let flatbuffer_description = builder.create_string(description);
        let ioheader = common::ioheader_generated::Ioheader::create(
            builder,
            &common::ioheader_generated::IoheaderArgs {
                compression: match compression {
                    Compression::None => common::ioheader_generated::Compression::None,
                    Compression::Lz4 { .. } => common::ioheader_generated::Compression::Lz4,
                    Compression::Zstd { .. } => common::ioheader_generated::Compression::Zstd,
                },
                file_data_position: -1,
                description: Some(flatbuffer_description),
            },
        );
        builder.finish_size_prefixed(
            ioheader,
            Some(common::ioheader_generated::IOHEADER_IDENTIFIER),
        );
        let data = builder.finished_data();
        file.write_all(builder.finished_data())?;
        let ioheader =
            unsafe { common::ioheader_generated::root_as_ioheader_unchecked(&data[4..]) };
        let offset = 4
            + ioheader._tab.loc()
            + ioheader
                ._tab
                .vtable()
                .get(common::ioheader_generated::Ioheader::VT_FILE_DATA_POSITION)
                as usize;
        Ok((offset as u64, data.len() as u64))
    }

    pub fn new<P: AsRef<std::path::Path>>(
        path: P,
        description_or_id_to_track: DescriptionOrIdsAndTracks,
        compression: Compression,
    ) -> Result<Self, Error> {
        let mut file = std::io::BufWriter::new(std::fs::File::create(path)?);
        file.write_all(common::MAGIC_NUMBER.as_bytes())?;
        let mut builder = flatbuffers::FlatBufferBuilder::with_capacity(utilities::BUFFER_SIZE);
        builder.force_defaults(true);
        let (id_to_track, file_data_position_offset, file_data_position) =
            match description_or_id_to_track {
                DescriptionOrIdsAndTracks::Description(description) => {
                    let (file_data_position_offset, file_data_position) =
                        Self::write_description(&mut file, compression, &mut builder, description)?;
                    (
                        common::description_to_id_to_tracks(description)?,
                        common::MAGIC_NUMBER.len() as u64 + file_data_position_offset,
                        common::MAGIC_NUMBER.len() as u64 + file_data_position,
                    )
                }
                DescriptionOrIdsAndTracks::IdsAndTracks(ids_and_tracks) => {
                    use std::fmt::Write;
                    let mut description = "<dv version=\"2.0\">\n".to_owned();
                    description +=
                        "    <node name=\"outInfo\" path=\"/mainloop/Recorder/outInfo/\">\n";
                    for (id, track) in ids_and_tracks.iter() {
                        write!(
                            description,
                            "        <node name=\"{}\" path=\"/mainloop/Recorder/outInfo/{}/\">\n",
                            id, id
                        )?;
                        write!(
                            description,
                            "            <attr key=\"compression\" type=\"string\">{}</attr>\n",
                            match compression {
                                Compression::None => "NONE",
                                Compression::Lz4 { .. } => "LZ4",
                                Compression::Zstd { .. } => "ZSTD",
                            }
                        )?;
                        match track {
                            common::Track::Events { dimensions, .. } => {
                                write!(description, "            <attr key=\"typeIdentifier\" type=\"string\">EVTS</attr>\n")?;
                                write!(description, "            <node name=\"info\" path=\"/mainloop/Recorder/outInfo/{}/info/\">\n", id)?;
                                write!(
                                    description,
                                    "                <attr key=\"sizeX\" type=\"int\">{}</attr>\n",
                                    dimensions.0
                                )?;
                                write!(
                                    description,
                                    "                <attr key=\"sizeY\" type=\"int\">{}</attr>\n",
                                    dimensions.1
                                )?;
                                description += "            </node>\n";
                            }
                            common::Track::Frame { dimensions, .. } => {
                                write!(description, "            <attr key=\"typeIdentifier\" type=\"string\">FRME</attr>\n")?;
                                write!(description, "            <node name=\"info\" path=\"/mainloop/Recorder/outInfo/{}/info/\">\n", id)?;
                                write!(
                                    description,
                                    "                <attr key=\"sizeX\" type=\"int\">{}</attr>\n",
                                    dimensions.0
                                )?;
                                write!(
                                    description,
                                    "                <attr key=\"sizeY\" type=\"int\">{}</attr>\n",
                                    dimensions.1
                                )?;
                                description += "            </node>\n";
                            }
                            common::Track::Imus { .. } => {
                                write!(description, "            <attr key=\"typeIdentifier\" type=\"string\">IMUS</attr>\n")?;
                            }
                            common::Track::Triggers { .. } => {
                                write!(description, "            <attr key=\"typeIdentifier\" type=\"string\">TRIG</attr>\n")?;
                            }
                        }
                        description += "        </node>\n";
                    }
                    description += "    </node>\n</dv>\n";
                    let (file_data_position_offset, file_data_position) = Self::write_description(
                        &mut file,
                        compression,
                        &mut builder,
                        &description,
                    )?;
                    (
                        common::description_to_id_to_tracks(&description)?,
                        common::MAGIC_NUMBER.len() as u64 + file_data_position_offset,
                        common::MAGIC_NUMBER.len() as u64 + file_data_position,
                    )
                }
            };
        let (builder_buffer, _) = builder.collapse();
        Ok(Self {
            file,
            id_to_track,
            compression,
            builder_buffer: Some(builder_buffer),
            file_data_position_offset,
            file_data_position,
            buffer: Vec::new(),
        })
    }

    pub fn get_track(&mut self, track_id: u32) -> Option<&mut common::Track> {
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

    fn compress_and_write(&mut self, track_id: u32, data: &[u8]) -> Result<(), PacketError> {
        match self.compression {
            Compression::None => {
                self.file.write_all(&track_id.to_le_bytes())?;
                self.file.write_all(&(data.len() as u32).to_le_bytes())?;
                self.file.write_all(data)?;
            }
            Compression::Lz4(level) => {
                self.buffer.clear();
                let mut encoder = lz4::EncoderBuilder::new()
                    .level(level as u32)
                    .build(&mut self.buffer)?;
                encoder.write_all(data)?;
                encoder.finish().1?;
                self.file.write_all(&track_id.to_le_bytes())?;
                self.file
                    .write_all(&(self.buffer.len() as u32).to_le_bytes())?;
                self.file.write_all(&self.buffer)?;
            }
            Compression::Zstd(level) => {
                self.buffer.clear();
                let mut encoder = zstd::stream::Encoder::new(&mut self.buffer, level as i32)?;
                encoder.write_all(data)?;
                encoder.finish()?;
                self.file.write_all(&track_id.to_le_bytes())?;
                self.file
                    .write_all(&(self.buffer.len() as u32).to_le_bytes())?;
                self.file.write_all(&self.buffer)?;
            }
        }
        Ok(())
    }

    fn write_events_with_builder<EventIterator>(
        &mut self,
        track_id: u32,
        events: EventIterator,
        builder: &mut flatbuffers::FlatBufferBuilder,
    ) -> Result<(), PacketError>
    where
        EventIterator: ExactSizeIterator<Item = neuromorphic_types::DvsEvent<u64, u16, u16>>
            + DoubleEndedIterator<Item = neuromorphic_types::DvsEvent<u64, u16, u16>>,
    {
        let vector = builder.create_vector_from_iter(events.map(|event| {
            common::events_generated::Event::new(
                event.t as i64,
                event.x as i16,
                event.y as i16,
                matches!(event.polarity, neuromorphic_types::DvsPolarity::On),
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
        self.compress_and_write(track_id, builder.finished_data())
    }

    pub fn write_events<EventIterator>(
        &mut self,
        track_id: u32,
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
        let (builder_buffer, _) = builder.collapse();
        self.builder_buffer.replace(builder_buffer);
        result
    }

    fn write_frame_with_builder(
        &mut self,
        track_id: u32,
        t: u64,
        begin_t: i64,
        end_t: i64,
        exposure_begin_t: i64,
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
                begin_t,
                end_t,
                exposure_begin_t,
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
        self.compress_and_write(track_id, builder.finished_data())
    }

    pub fn write_frame(
        &mut self,
        track_id: u32,
        t: u64,
        begin_t: i64,
        end_t: i64,
        exposure_begin_t: i64,
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
            begin_t,
            end_t,
            exposure_begin_t,
            exposure_end_t,
            format,
            width,
            height,
            offset_x,
            offset_y,
            pixels,
            &mut builder,
        );
        let (builder_buffer, _) = builder.collapse();
        self.builder_buffer.replace(builder_buffer);
        result
    }

    fn write_imus_with_builder<ImuIterator>(
        &mut self,
        track_id: u32,
        imus: ImuIterator,
        builder: &mut flatbuffers::FlatBufferBuilder,
    ) -> Result<(), PacketError>
    where
        ImuIterator: ExactSizeIterator<Item = Imu> + DoubleEndedIterator<Item = Imu>,
    {
        self.buffer.clear();
        self.buffer.resize(
            imus.len() * std::mem::size_of::<flatbuffers::WIPOffset<common::imus_generated::Imu>>(),
            0,
        );
        let imus_offsets = unsafe {
            std::slice::from_raw_parts_mut(
                self.buffer.as_mut_ptr()
                    as *mut flatbuffers::WIPOffset<common::imus_generated::Imu>,
                imus.len(),
            )
        };
        for (index, imu) in imus.enumerate() {
            let offset = common::imus_generated::Imu::create(
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
            imus_offsets[index] = offset;
        }
        let vector = builder.create_vector(imus_offsets);
        let packet = common::imus_generated::ImuPacket::create(
            builder,
            &common::imus_generated::ImuPacketArgs {
                elements: Some(vector),
            },
        );
        builder.finish_size_prefixed(packet, Some(common::imus_generated::IMU_PACKET_IDENTIFIER));
        self.compress_and_write(track_id, builder.finished_data())
    }

    pub fn write_imus<ImuIterator>(
        &mut self,
        track_id: u32,
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
        let (builder_buffer, _) = builder.collapse();
        self.builder_buffer.replace(builder_buffer);
        result
    }

    fn write_triggers_with_builder<TriggerIterator>(
        &mut self,
        track_id: u32,
        triggers: TriggerIterator,
        builder: &mut flatbuffers::FlatBufferBuilder,
    ) -> Result<(), PacketError>
    where
        TriggerIterator: ExactSizeIterator<Item = Trigger> + DoubleEndedIterator<Item = Trigger>,
    {
        self.buffer.clear();
        self.buffer.resize(
            triggers.len()
                * std::mem::size_of::<flatbuffers::WIPOffset<common::triggers_generated::Trigger>>(
                ),
            0,
        );
        let triggers_offsets = unsafe {
            std::slice::from_raw_parts_mut(
                self.buffer.as_mut_ptr()
                    as *mut flatbuffers::WIPOffset<common::triggers_generated::Trigger>,
                triggers.len(),
            )
        };
        for (index, trigger) in triggers.enumerate() {
            let offset = common::triggers_generated::Trigger::create(
                builder,
                &common::triggers_generated::TriggerArgs {
                    t: trigger.t as i64,
                    source: common::triggers_generated::TriggerSource(trigger.source as i8),
                },
            );
            triggers_offsets[index] = offset;
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
        self.compress_and_write(track_id, builder.finished_data())
    }

    pub fn write_triggers<TriggerIterator>(
        &mut self,
        track_id: u32,
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
        let (builder_buffer, _) = builder.collapse();
        self.builder_buffer.replace(builder_buffer);
        result
    }
}
