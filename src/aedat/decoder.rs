use std::io::Read;

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
}

pub struct Decoder {
    pub id_to_track: std::collections::HashMap<u32, common::Track>,
    file: std::io::BufReader<std::fs::File>,
    description: String,
    position: i64,
    compression: common::ioheader_generated::Compression,
    file_data_position: i64,
    raw_buffer: Vec<u8>,
    buffer: Vec<u8>,
}

impl Decoder {
    pub fn new<P: AsRef<std::path::Path>>(path: P) -> Result<Self, Error> {
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
        };
        let mut buffer = std::vec![0; length as usize];
        file.read_exact(&mut buffer)?;
        let ioheader = unsafe { common::ioheader_generated::root_as_ioheader_unchecked(&buffer) };
        let compression = ioheader.compression();
        let file_data_position = ioheader.file_data_position();
        let description = match ioheader.description() {
            Some(content) => content.to_owned(),
            None => return Err(Error::EmptyDescription),
        };
        let id_to_track = common::description_to_id_to_tracks(&description)?;
        Ok(Decoder {
            id_to_track,
            file,
            description,
            position: (common::MAGIC_NUMBER.len() + 4 + length as usize) as i64,
            compression,
            file_data_position,
            raw_buffer: Vec::new(),
            buffer,
        })
    }

    pub fn description(&self) -> &str {
        self.description.as_str()
    }
}

pub struct Packet<'a> {
    pub buffer: &'a std::vec::Vec<u8>,
    pub track_id: u32,
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
    UnknownPacketTrackId(u32),

    #[error("bad packet prefix for track ID {id} (expected \"{expected}\", got \"{got}\")")]
    BadPacketPrefix {
        id: u32,
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
            let track_id = u32::from_le_bytes(bytes[0..4].try_into().expect("four bytes"));
            let length = u32::from_le_bytes(bytes[4..8].try_into().expect("four bytes"));
            (track_id, length)
        };
        self.position += 8i64 + length as i64;
        self.raw_buffer.resize(length as usize, 0u8);
        self.file.read_exact(&mut self.raw_buffer)?;
        match self.compression {
            common::ioheader_generated::Compression::None => {
                std::mem::swap(&mut self.raw_buffer, &mut self.buffer);
            }
            common::ioheader_generated::Compression::Lz4
            | common::ioheader_generated::Compression::Lz4High => {
                let mut decoder = lz4::Decoder::new(&self.raw_buffer[..])?;
                self.buffer.clear();
                decoder.read_to_end(&mut self.buffer)?;
            }
            common::ioheader_generated::Compression::Zstd
            | common::ioheader_generated::Compression::ZstdHigh => {
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
                id: track_id,
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
