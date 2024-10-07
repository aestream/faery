use std::io::Read;
use std::io::Seek;

use crate::dat::common;
use crate::utilities;

pub struct Decoder {
    pub event_type: common::Type,
    version: common::Version,
    file: std::fs::File,
    raw_buffer: Vec<u8>,
    event_buffer: Vec<common::Event>,
    t: u64,
    offset: u64,
    t0: u64,
}

#[derive(thiserror::Error, Debug)]
pub enum Error {
    #[error(transparent)]
    Io(#[from] std::io::Error),

    #[error("the header has no size information and no size fallback was provided")]
    MissingDimensions,

    #[error("the event type ({0}) not supported")]
    UnsupportedType(u8),

    #[error("the event size ({0}) not supported")]
    UnsupportedEventSize(u8),

    #[error("the header has no version information and no versions fallback was provided")]
    MissingVersion,

    #[error("unknown version \"{0}\" (supports \"dat1\" and \"dat2\")")]
    UnknownVersion(String),
}

impl Decoder {
    pub fn new<P: AsRef<std::path::Path>>(
        path: P,
        dimensions_fallback: Option<(u16, u16)>,
        version_fallback: Option<common::Version>,
    ) -> Result<Self, Error> {
        let header = utilities::read_prophesee_header(
            &mut std::io::BufReader::new(std::fs::File::open(&path)?),
            '%',
        )?;
        let version = match header.version {
            Some(version) => match version.as_str() {
                "1" => common::Version::Dat1,
                "2" => common::Version::Dat2,
                _ => return Err(Error::UnknownVersion(version)),
            },
            None => match version_fallback {
                Some(version) => version,
                None => return Err(Error::MissingVersion),
            },
        };
        let mut file = std::fs::File::open(path)?;
        file.seek(std::io::SeekFrom::Start(header.length))?;
        let event_type = {
            let mut type_and_size = [0u8; 2];
            file.read_exact(&mut type_and_size)?;
            if type_and_size[1] != 8 {
                return Err(Error::UnsupportedEventSize(type_and_size[1]));
            }
            match type_and_size[0] {
                0x00 => {
                    let dimensions = match header.dimensions {
                        Some(dimensions) => dimensions,
                        None => match dimensions_fallback {
                            Some(dimensions) => dimensions,
                            None => return Err(Error::MissingDimensions),
                        },
                    };
                    common::Type::Event2d(dimensions.0, dimensions.1)
                }
                0x0C => {
                    let dimensions = match header.dimensions {
                        Some(dimensions) => dimensions,
                        None => match dimensions_fallback {
                            Some(dimensions) => dimensions,
                            None => return Err(Error::MissingDimensions),
                        },
                    };
                    common::Type::EventCd(dimensions.0, dimensions.1)
                }
                0x0E => common::Type::EventExtTrigger,
                event_type => return Err(Error::UnsupportedType(event_type)),
            }
        };
        Ok(Decoder {
            event_type,
            version,
            file,
            raw_buffer: vec![0u8; utilities::BUFFER_SIZE],
            event_buffer: Vec::new(),
            t: 0,
            offset: 0,
            t0: header.t0,
        })
    }

    pub fn version(&self) -> common::Version {
        self.version
    }

    pub fn dimensions(&self) -> Option<(u16, u16)> {
        match self.event_type {
            common::Type::Event2d(width, height) => Some((width, height)),
            common::Type::EventCd(width, height) => Some((width, height)),
            common::Type::EventExtTrigger => None,
        }
    }

    pub fn next(&mut self) -> Result<Option<&Vec<common::Event>>, utilities::ReadError> {
        let read = self.file.read(&mut self.raw_buffer)?;
        if read == 0 {
            return Ok(None);
        }
        self.event_buffer.clear();
        self.event_buffer.reserve(read / 8);
        match self.version {
            common::Version::Dat1 => {
                for index in 0..read / 8 {
                    let word = u64::from_le_bytes(
                        self.raw_buffer[index * 8..(index + 1) * 8]
                            .try_into()
                            .expect("8 bytes"),
                    );
                    let mut candidate_t = (word & 0xFFFFFFFF_u64) + self.offset;
                    if candidate_t < self.t {
                        if self.t - candidate_t > (1_u64 << 31) {
                            candidate_t += 1_u64 << 32;
                            self.offset += 1_u64 << 32;
                            self.t = candidate_t;
                        }
                    } else {
                        self.t = candidate_t;
                    }
                    let x = ((word >> 32) & 0b111111111_u64) as u16;
                    let y = ((word >> 41) & 0b11111111_u64) as u16;
                    match self.event_type {
                        common::Type::Event2d(width, height)
                        | common::Type::EventCd(width, height) => {
                            if x >= width {
                                return Err(utilities::ReadError::XOverflow { x, width });
                            }
                            if y >= height {
                                return Err(utilities::ReadError::YOverflow { y, height });
                            }
                        }
                        common::Type::EventExtTrigger => {}
                    }
                    self.event_buffer.push(common::Event {
                        t: self.t + self.t0,
                        x,
                        y,
                        payload: ((word >> 49) & 0b1111) as u8,
                    });
                }
            }
            common::Version::Dat2 => {
                for index in 0..read / 8 {
                    let word = u64::from_le_bytes(
                        self.raw_buffer[index * 8..(index + 1) * 8]
                            .try_into()
                            .expect("8 bytes"),
                    );
                    let mut candidate_t = (word & 0xFFFFFFFF_u64) + self.offset;
                    if candidate_t < self.t {
                        if self.t - candidate_t > (1_u64 << 31) {
                            candidate_t += 1_u64 << 32;
                            self.offset += 1_u64 << 32;
                            self.t = candidate_t;
                        }
                    } else {
                        self.t = candidate_t;
                    }
                    let x = ((word >> 32) & 0b11111111111111_u64) as u16;
                    let y = ((word >> 46) & 0b11111111111111_u64) as u16;
                    match self.event_type {
                        common::Type::Event2d(width, height)
                        | common::Type::EventCd(width, height) => {
                            if x >= width {
                                return Err(utilities::ReadError::XOverflow { x, width });
                            }
                            if y >= height {
                                return Err(utilities::ReadError::YOverflow { y, height });
                            }
                        }
                        common::Type::EventExtTrigger => {}
                    }
                    self.event_buffer.push(common::Event {
                        t: self.t + self.t0,
                        x,
                        y,
                        payload: (word >> 60) as u8,
                    });
                }
            }
        }
        Ok(Some(&self.event_buffer))
    }
}
