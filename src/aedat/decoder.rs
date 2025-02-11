use std::io::Read;
use std::io::Seek;

use crate::aedat::common;

#[derive(thiserror::Error, Debug)]
pub enum Error {
    #[error(transparent)]
    Io(#[from] std::io::Error),

    #[error(transparent)]
    Flatbuffer(#[from] flatbuffers::InvalidFlatbuffer),

    #[error(transparent)]
    Utf8(#[from] std::str::Utf8Error),

    #[error(transparent)]
    Description(#[from] common::DescriptionError),

    #[error("bad magic number (expected \"#!AER-DAT4.0\\r\\n\", got \"{0}\")")]
    MagicNumber(String),

    #[error("empty description")]
    EmptyDescription,

    #[error("unknown compression algorithm")]
    CompressionAlgorithm,
}

pub struct Decoder {
    pub description: common::Description,
    pub id_to_track: std::collections::HashMap<i32, common::Track>,
    pub file_data_definitions: Vec<common::FileDataDefinition>,
    file: std::io::BufReader<std::fs::File>,
    position: i64,
    compression: common::io_header_generated::Compression,
    file_data_position: i64,
    raw_buffer: Vec<u8>,
    buffer: Vec<u8>,
}

impl Decoder {
    pub fn new<Path: AsRef<std::path::Path>>(path: Path) -> Result<Self, Error> {
        let mut file = std::io::BufReader::new(std::fs::File::open(path)?);
        {
            let mut magic_number_buffer = [0; common::MAGIC_NUMBER.len()];
            file.read_exact(&mut magic_number_buffer)?;
            let magic_number = String::from_utf8_lossy(&magic_number_buffer).to_string();
            if magic_number != common::MAGIC_NUMBER {
                return Err(Error::MagicNumber(magic_number));
            }
        }
        let length = {
            let mut bytes = [0; 4];
            file.read_exact(&mut bytes)?;
            u32::from_le_bytes(bytes)
        } as usize;
        let mut raw_buffer = std::vec![0; length as usize];
        file.read_exact(&mut raw_buffer)?;
        let io_header =
            unsafe { common::io_header_generated::root_as_ioheader_unchecked(&raw_buffer) };
        let compression = io_header.compression();
        let file_data_position = io_header.file_data_position();
        let description = common::Description::from_xml_string(match io_header.description() {
            Some(content) => content,
            None => return Err(Error::EmptyDescription),
        })?;
        let id_to_track = description.id_to_track()?;
        let mut file_data_definitions = Vec::new();
        let mut buffer = Vec::new();
        if file_data_position > -1 {
            file.seek(std::io::SeekFrom::Start(file_data_position as u64))?;
            raw_buffer.clear();
            file.read_to_end(&mut raw_buffer)?;
            if !raw_buffer.is_empty() {
                match compression {
                    common::io_header_generated::Compression::None => {
                        std::mem::swap(&mut raw_buffer, &mut buffer);
                    }
                    common::io_header_generated::Compression::Lz4
                    | common::io_header_generated::Compression::Lz4High => {
                        let mut decoder = lz4::Decoder::new(&raw_buffer[..])?;
                        decoder.read_to_end(&mut buffer)?;
                    }
                    common::io_header_generated::Compression::Zstd
                    | common::io_header_generated::Compression::ZstdHigh => {
                        let mut decoder = zstd::Decoder::new(&raw_buffer[..])?;
                        decoder.read_to_end(&mut buffer)?;
                    }
                    _ => return Err(Error::CompressionAlgorithm),
                }
                let file_data_table =
                    common::file_data_table_generated::size_prefixed_root_as_file_data_table(
                        &buffer,
                    )?;
                if let Some(raw_file_data_definitions) = file_data_table.file_data_definitions() {
                    file_data_definitions.reserve_exact(raw_file_data_definitions.len());
                    for raw_file_data_definition in raw_file_data_definitions {
                        if let Some(packet_header) = raw_file_data_definition.packet_header() {
                            let track_id = packet_header.track_id();
                            if track_id >= 0 {
                                file_data_definitions.push(common::FileDataDefinition {
                                    byte_offset: raw_file_data_definition.byte_offset(),
                                    track_id,
                                    size: packet_header.size(),
                                    elements_count: raw_file_data_definition.elements_count(),
                                    start_t: raw_file_data_definition.start_t(),
                                    end_t: raw_file_data_definition.end_t(),
                                });
                            }
                        }
                    }
                }
            }
            file.seek(std::io::SeekFrom::Start(
                (common::MAGIC_NUMBER.len() + 4 + length) as u64,
            ))?;
        }
        Ok(Decoder {
            description,
            id_to_track,
            file_data_definitions,
            file,
            position: (common::MAGIC_NUMBER.len() + 4 + length) as i64,
            compression,
            file_data_position,
            raw_buffer,
            buffer,
        })
    }
}

pub struct Packet<'a> {
    pub buffer: &'a std::vec::Vec<u8>,
    pub track_id: i32,
    pub track: &'a mut common::Track,
}

#[derive(thiserror::Error, Debug)]
pub enum ReadError {
    #[error(transparent)]
    Io(#[from] std::io::Error),

    #[error(transparent)]
    Flatbuffers(#[from] flatbuffers::InvalidFlatbuffer),

    #[error("unknown compression algorithm")]
    CompressionAlgorithm,

    #[error("unknown packet track ID {0}")]
    UnknownPacketTrackId(i32),

    #[error("bad packet prefix for track ID {track_id} (expected \"{expected}\", got \"{got}\")")]
    BadPacketPrefix {
        track_id: i32,
        expected: String,
        got: String,
    },

    #[error("empty events packet")]
    EmptyEventsPacket,

    #[error("missing packet size prefix")]
    MissingPacketSizePrefix,

    #[error("unknown frame format")]
    UnknownFrameFormat,

    #[error("unknown trigger source")]
    UnknownTriggerSource,

    #[error("x overflow (x={x} must be larger than 0 and strictly smaller than width={width})")]
    XOverflow { x: i16, width: u16 },

    #[error("y overflow (y={y} must be larger than 0 and strictly smaller than height={height})")]
    YOverflow { y: i16, height: u16 },
}

impl Decoder {
    pub fn next(&mut self) -> Result<Option<Packet>, ReadError> {
        if self.file_data_position > -1 && self.position == self.file_data_position {
            return Ok(None);
        }
        let (track_id, length) = {
            let mut bytes = [0; 8];
            if let Err(error) = self.file.read_exact(&mut bytes) {
                return if self.file_data_position == -1 {
                    Ok(None)
                } else {
                    Err(error.into())
                };
            }
            let track_id = i32::from_le_bytes(bytes[0..4].try_into().expect("four bytes"));
            let length = u32::from_le_bytes(bytes[4..8].try_into().expect("four bytes"));
            (track_id, length)
        };
        self.position += 8i64 + length as i64;
        self.raw_buffer.resize(length as usize, 0u8);
        self.file.read_exact(&mut self.raw_buffer)?;
        match self.compression {
            common::io_header_generated::Compression::None => {
                std::mem::swap(&mut self.raw_buffer, &mut self.buffer);
            }
            common::io_header_generated::Compression::Lz4
            | common::io_header_generated::Compression::Lz4High => {
                let mut decoder = lz4::Decoder::new(&self.raw_buffer[..])?;
                self.buffer.clear();
                decoder.read_to_end(&mut self.buffer)?;
            }
            common::io_header_generated::Compression::Zstd
            | common::io_header_generated::Compression::ZstdHigh => {
                let mut decoder = zstd::Decoder::new(&self.raw_buffer[..])?;
                self.buffer.clear();
                decoder.read_to_end(&mut self.buffer)?;
            }
            _ => return Err(ReadError::CompressionAlgorithm),
        }
        let track = self
            .id_to_track
            .get_mut(&track_id)
            .ok_or(ReadError::UnknownPacketTrackId(track_id))?;
        let expected = track.to_identifier().to_owned();
        if !flatbuffers::buffer_has_identifier(&self.buffer, &expected, true) {
            let expected_length = expected.len();
            let offset = flatbuffers::SIZE_SIZEPREFIX + flatbuffers::SIZE_UOFFSET;
            return Err(ReadError::BadPacketPrefix {
                track_id,
                expected,
                got: if self.buffer.len() >= offset {
                    String::from_utf8_lossy(
                        &self.buffer
                            [offset..offset + expected_length.min(self.buffer.len() - offset)],
                    )
                    .into_owned()
                } else {
                    "".to_owned()
                }
                .to_string(),
            });
        }
        Ok(Some(Packet {
            buffer: &self.buffer,
            track_id,
            track,
        }))
    }
}
