use std::io::Read;

use crate::event_stream::common;
use crate::utilities;

enum GenericState {
    Idle,
    Byte0,
    DataByte,
}

enum DvsState {
    Idle,
    Byte0,
    Byte1,
    Byte2,
    Byte3,
}

enum AtisState {
    Idle,
    Byte0,
    Byte1,
    Byte2,
    Byte3,
}

enum ColorState {
    Idle,
    Byte0,
    Byte1,
    Byte2,
    Byte3,
    Byte4,
    Byte5,
    Byte6,
}

enum State {
    Generic {
        inner: GenericState,
        t: u64,
        index: usize,
        bytes_length: usize,
        bytes: Vec<u8>,
        buffer: Vec<common::OwnedGenericEvent>,
    },
    Dvs {
        inner: DvsState,
        event: neuromorphic_types::DvsEvent<u64, u16, u16>,
        buffer: Vec<neuromorphic_types::DvsEvent<u64, u16, u16>>,
        dimensions: (u16, u16),
    },
    Atis {
        inner: AtisState,
        event: neuromorphic_types::AtisEvent<u64, u16, u16>,
        buffer: Vec<neuromorphic_types::AtisEvent<u64, u16, u16>>,
        dimensions: (u16, u16),
    },
    Color {
        inner: ColorState,
        event: common::ColorEvent,
        buffer: Vec<common::ColorEvent>,
        dimensions: (u16, u16),
    },
}

pub struct Decoder {
    pub version: [u8; 3],
    pub event_type: common::Type,
    file: std::fs::File,
    raw_buffer: Vec<u8>,
    state: State,
}

#[derive(thiserror::Error, Debug)]
pub enum Error {
    #[error(transparent)]
    Io(#[from] std::io::Error),

    #[error("bad magic number (expected \"Event Stream\", got \"{0}\")")]
    MagicNumber(String),

    #[error("bad magic number (expected 2.x.y, got {major}.{minor}.{patch})")]
    UnsupportedVersion { major: u8, minor: u8, patch: u8 },

    #[error("unsupported type {0}")]
    UnsupportedType(u8),
}

impl Decoder {
    pub fn new<P: AsRef<std::path::Path>>(path: P, t0: u64) -> Result<Self, Error> {
        let mut file = std::fs::File::open(path)?;
        {
            let mut magic_number_bytes = [0u8; common::MAGIC_NUMBER.len()];
            file.read_exact(&mut magic_number_bytes)?;
            let magic_number = String::from_utf8_lossy(&magic_number_bytes);
            if magic_number != common::MAGIC_NUMBER {
                return Err(Error::MagicNumber(magic_number.to_string()));
            }
        }
        let mut version = [0u8; 3];
        file.read_exact(&mut version)?;
        if version[0] != common::VERSION[0] {
            return Err(Error::UnsupportedVersion {
                major: version[0],
                minor: version[1],
                patch: version[2],
            });
        }
        let event_type = {
            let mut version_byte = [0u8; 1];
            file.read_exact(&mut version_byte)?;
            match version_byte[0] {
                0 => common::Type::Generic,
                1 => common::Type::Dvs,
                2 => common::Type::Atis,
                4 => common::Type::Color,
                _ => return Err(Error::UnsupportedType(version_byte[0])),
            }
        };
        let dimensions = match event_type {
            common::Type::Generic => None,
            _ => {
                let mut size_bytes = [0u8; 4];
                file.read_exact(&mut size_bytes)?;
                Some((
                    u16::from_le_bytes([size_bytes[0], size_bytes[1]]),
                    u16::from_le_bytes([size_bytes[2], size_bytes[3]]),
                ))
            }
        };
        Ok(Decoder {
            version,
            event_type,
            file,
            raw_buffer: vec![0u8; utilities::BUFFER_SIZE],
            state: match event_type {
                common::Type::Generic => State::Generic {
                    inner: GenericState::Idle,
                    t: t0,
                    index: 0,
                    bytes_length: 0,
                    bytes: Vec::new(),
                    buffer: Vec::new(),
                },
                common::Type::Dvs => State::Dvs {
                    inner: DvsState::Idle,
                    event: neuromorphic_types::DvsEvent::<u64, u16, u16> {
                        t: t0,
                        x: 0,
                        y: 0,
                        polarity: neuromorphic_types::DvsPolarity::Off,
                    },
                    buffer: Vec::new(),
                    dimensions: dimensions.expect("a dvs stream has dimensions"),
                },
                common::Type::Atis => State::Atis {
                    inner: AtisState::Idle,
                    event: neuromorphic_types::AtisEvent::<u64, u16, u16> {
                        t: t0,
                        x: 0,
                        y: 0,
                        polarity: neuromorphic_types::AtisPolarity::Off,
                    },
                    buffer: Vec::new(),
                    dimensions: dimensions.expect("a atis stream has dimensions"),
                },
                common::Type::Color => State::Color {
                    inner: ColorState::Idle,
                    event: common::ColorEvent {
                        t: t0,
                        x: 0,
                        y: 0,
                        r: 0,
                        g: 0,
                        b: 0,
                    },
                    buffer: Vec::new(),
                    dimensions: dimensions.expect("a color stream has dimensions"),
                },
            },
        })
    }

    pub fn version(&self) -> [u8; 3] {
        self.version.clone()
    }

    pub fn dimensions(&self) -> Option<(u16, u16)> {
        match self.state {
            State::Generic { .. } => None,
            State::Dvs { dimensions, .. } => Some(dimensions),
            State::Atis { dimensions, .. } => Some(dimensions),
            State::Color { dimensions, .. } => Some(dimensions),
        }
    }
}

pub enum Packet<'a> {
    Generic(&'a Vec<common::OwnedGenericEvent>),
    Dvs(&'a Vec<neuromorphic_types::DvsEvent<u64, u16, u16>>),
    Atis(&'a Vec<neuromorphic_types::AtisEvent<u64, u16, u16>>),
    Color(&'a Vec<common::ColorEvent>),
}

impl Decoder {
    pub fn next(&mut self) -> Result<Option<Packet>, utilities::ReadError> {
        let read = self.file.read(&mut self.raw_buffer)?;
        if read == 0 {
            return Ok(None);
        }
        match self.state {
            State::Generic {
                ref mut inner,
                ref mut t,
                ref mut index,
                ref mut bytes_length,
                ref mut bytes,
                ref mut buffer,
            } => {
                buffer.clear();
                for byte in self.raw_buffer[0..read].iter() {
                    *inner = match inner {
                        GenericState::Idle => {
                            if *byte == 0b11111111 {
                                *t += 0b11111110;
                                GenericState::Idle
                            } else if *byte != 0b11111110 {
                                *t += *byte as u64;
                                *bytes_length = 0;
                                *index = 0;
                                GenericState::Byte0
                            } else {
                                GenericState::Idle
                            }
                        }
                        GenericState::Byte0 => {
                            *bytes_length |= ((byte >> 1) as usize) << (7 * *index);
                            if ((*byte) & 1) == 0 {
                                bytes.clear();
                                *index = 0;
                                if *bytes_length == 0 {
                                    buffer.push(common::OwnedGenericEvent {
                                        t: *t,
                                        bytes: bytes.clone(),
                                    });
                                    GenericState::Idle
                                } else {
                                    bytes.resize(*bytes_length, 0u8);
                                    GenericState::DataByte
                                }
                            } else {
                                *index += 1;
                                GenericState::Byte0
                            }
                        }
                        GenericState::DataByte => {
                            bytes[*index] = *byte;
                            *index += 1;
                            if index == bytes_length {
                                buffer.push(common::OwnedGenericEvent {
                                    t: *t,
                                    bytes: bytes.clone(),
                                });
                                GenericState::Idle
                            } else {
                                GenericState::DataByte
                            }
                        }
                    }
                }
                Ok(Some(Packet::Generic(buffer)))
            }
            State::Dvs {
                ref mut inner,
                ref mut event,
                ref mut buffer,
                dimensions,
            } => {
                buffer.clear();
                for byte in self.raw_buffer[0..read].iter() {
                    *inner = match inner {
                        DvsState::Idle => {
                            if *byte == 0b11111111 {
                                event.t += 0b1111111;
                                DvsState::Idle
                            } else if *byte != 0b11111110 {
                                event.t += (byte >> 1) as u64;
                                event.polarity = if (byte & 1) == 1 {
                                    neuromorphic_types::DvsPolarity::On
                                } else {
                                    neuromorphic_types::DvsPolarity::Off
                                };
                                DvsState::Byte0
                            } else {
                                DvsState::Idle
                            }
                        }
                        DvsState::Byte0 => {
                            event.x = *byte as u16;
                            DvsState::Byte1
                        }
                        DvsState::Byte1 => {
                            event.x |= (*byte as u16) << 8;
                            if event.x >= dimensions.0 {
                                return Err(utilities::ReadError::XOverflow {
                                    x: event.x,
                                    width: dimensions.0,
                                });
                            }
                            DvsState::Byte2
                        }
                        DvsState::Byte2 => {
                            event.y = *byte as u16;
                            DvsState::Byte3
                        }
                        DvsState::Byte3 => {
                            event.y |= (*byte as u16) << 8;
                            if event.y >= dimensions.1 {
                                return Err(utilities::ReadError::YOverflow {
                                    y: event.y,
                                    height: dimensions.1,
                                });
                            }
                            buffer.push(*event);
                            DvsState::Idle
                        }
                    }
                }
                Ok(Some(Packet::Dvs(buffer)))
            }
            State::Atis {
                ref mut inner,
                ref mut event,
                ref mut buffer,
                dimensions,
            } => {
                buffer.clear();
                for byte in self.raw_buffer[0..read].iter() {
                    *inner = match inner {
                        AtisState::Idle => {
                            if (byte & 0b11111100) == 0b11111100 {
                                event.t += (0b111111_u64) * (byte & 0b11) as u64;
                                AtisState::Idle
                            } else {
                                event.t += (byte >> 2) as u64;
                                event.polarity = match byte & 0b11 {
                                    0b00 => neuromorphic_types::AtisPolarity::Off,
                                    0b01 => neuromorphic_types::AtisPolarity::ExposureStart,
                                    0b10 => neuromorphic_types::AtisPolarity::On,
                                    0b11 => neuromorphic_types::AtisPolarity::ExposureEnd,
                                    _ => unreachable!(),
                                };
                                AtisState::Byte0
                            }
                        }
                        AtisState::Byte0 => {
                            event.x = *byte as u16;
                            AtisState::Byte1
                        }
                        AtisState::Byte1 => {
                            event.x |= (*byte as u16) << 8;
                            if event.x >= dimensions.0 {
                                return Err(utilities::ReadError::XOverflow {
                                    x: event.x,
                                    width: dimensions.0,
                                });
                            }
                            AtisState::Byte2
                        }
                        AtisState::Byte2 => {
                            event.y = *byte as u16;
                            AtisState::Byte3
                        }
                        AtisState::Byte3 => {
                            event.y |= (*byte as u16) << 8;
                            if event.y >= dimensions.1 {
                                return Err(utilities::ReadError::YOverflow {
                                    y: event.y,
                                    height: dimensions.1,
                                });
                            }
                            buffer.push(*event);
                            AtisState::Idle
                        }
                    }
                }
                Ok(Some(Packet::Atis(buffer)))
            }
            State::Color {
                ref mut inner,
                ref mut event,
                ref mut buffer,
                dimensions,
            } => {
                buffer.clear();
                for byte in self.raw_buffer[0..read].iter() {
                    *inner = match inner {
                        ColorState::Idle => {
                            if *byte == 0b11111111 {
                                event.t += 0b11111110;
                                ColorState::Idle
                            } else if *byte != 0b11111110 {
                                event.t += *byte as u64;
                                ColorState::Byte0
                            } else {
                                ColorState::Idle
                            }
                        }
                        ColorState::Byte0 => {
                            event.x = *byte as u16;
                            ColorState::Byte1
                        }
                        ColorState::Byte1 => {
                            event.x |= (*byte as u16) << 8;
                            if event.x >= dimensions.0 {
                                return Err(utilities::ReadError::XOverflow {
                                    x: event.x,
                                    width: dimensions.0,
                                });
                            }
                            ColorState::Byte2
                        }
                        ColorState::Byte2 => {
                            event.y = *byte as u16;
                            ColorState::Byte3
                        }
                        ColorState::Byte3 => {
                            event.y |= (*byte as u16) << 8;
                            if event.y >= dimensions.1 {
                                return Err(utilities::ReadError::YOverflow {
                                    y: event.y,
                                    height: dimensions.1,
                                });
                            }
                            ColorState::Byte4
                        }
                        ColorState::Byte4 => {
                            event.r = *byte;
                            ColorState::Byte5
                        }
                        ColorState::Byte5 => {
                            event.g = *byte;
                            ColorState::Byte6
                        }
                        ColorState::Byte6 => {
                            event.b = *byte;
                            buffer.push(*event);
                            ColorState::Idle
                        }
                    }
                }
                Ok(Some(Packet::Color(buffer)))
            }
        }
    }
}
