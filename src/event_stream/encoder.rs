use std::io::Write;

use crate::event_stream::common;
use crate::utilities;

pub enum EncoderType {
    Generic,
    Dvs(u16, u16),
    Atis(u16, u16),
    Color(u16, u16),
}

#[derive(thiserror::Error, Debug)]
pub enum Error {
    #[error(transparent)]
    Io(#[from] std::io::Error),

    #[error("unknown event type \"{0}\" (must be \"generic\", \"dvs\", \"atis\", or \"color\"")]
    UnknownType(String),

    #[error("dimensions must be None for a stream of generic events (got {0}x{1}")]
    GenericSize(u16, u16),

    #[error("dimensions must not be None")]
    Size,
}

impl EncoderType {
    pub fn new(string: &str, dimensions: Option<(u16, u16)>) -> Result<Self, Error> {
        match string {
            "generic" => match dimensions {
                Some((width, height)) => Err(Error::GenericSize(width, height)),
                None => Ok(EncoderType::Generic),
            },
            "dvs" => match dimensions {
                Some((width, height)) => Ok(EncoderType::Dvs(width, height)),
                None => Err(Error::Size),
            },
            "atis" => match dimensions {
                Some((width, height)) => Ok(EncoderType::Atis(width, height)),
                None => Err(Error::Size),
            },
            "color" => match dimensions {
                Some((width, height)) => Ok(EncoderType::Color(width, height)),
                None => Err(Error::Size),
            },
            string => Err(Error::UnknownType(string.to_owned())),
        }
    }
}

pub struct GenericEncoder {
    file: std::io::BufWriter<std::fs::File>,
    previous_t: u64,
    t0: Option<u64>,
}

pub struct DvsEncoder {
    file: std::io::BufWriter<std::fs::File>,
    dimensions: (u16, u16),
    previous_t: u64,
    t0: Option<u64>,
}

pub struct AtisEncoder {
    file: std::io::BufWriter<std::fs::File>,
    dimensions: (u16, u16),
    previous_t: u64,
    t0: Option<u64>,
}

pub struct ColorEncoder {
    file: std::io::BufWriter<std::fs::File>,
    dimensions: (u16, u16),
    previous_t: u64,
    t0: Option<u64>,
}

pub enum Encoder {
    Generic(GenericEncoder),
    Dvs(DvsEncoder),
    Atis(AtisEncoder),
    Color(ColorEncoder),
}

impl Encoder {
    pub fn new<P: AsRef<std::path::Path>>(
        path: P,
        zero_t0: bool,
        encoder_type: EncoderType,
    ) -> Result<Self, Error> {
        Ok(match encoder_type {
            EncoderType::Generic => Encoder::Generic(GenericEncoder::new(path, zero_t0)?),
            EncoderType::Dvs(width, height) => {
                Encoder::Dvs(DvsEncoder::new(path, zero_t0, (width, height))?)
            }
            EncoderType::Atis(width, height) => {
                Encoder::Atis(AtisEncoder::new(path, zero_t0, (width, height))?)
            }
            EncoderType::Color(width, height) => {
                Encoder::Color(ColorEncoder::new(path, zero_t0, (width, height))?)
            }
        })
    }

    pub fn t0(&self) -> Option<u64> {
        match self {
            Encoder::Generic(encoder) => encoder.t0,
            Encoder::Dvs(encoder) => encoder.t0,
            Encoder::Atis(encoder) => encoder.t0,
            Encoder::Color(encoder) => encoder.t0,
        }
    }
}

fn open<P: AsRef<std::path::Path>>(
    path: P,
    event_type: common::Type,
    dimensions: (u16, u16),
) -> Result<std::io::BufWriter<std::fs::File>, Error> {
    let mut file = std::io::BufWriter::new(std::fs::File::create(path)?);
    file.write_all(common::MAGIC_NUMBER.as_bytes())?;
    file.write_all(&common::VERSION)?;
    file.write_all(&[event_type as u8])?;
    file.write_all(&dimensions.0.to_le_bytes())?;
    file.write_all(&dimensions.1.to_le_bytes())?;
    Ok(file)
}

impl GenericEncoder {
    pub fn new<P: AsRef<std::path::Path>>(path: P, zero_t0: bool) -> Result<Self, Error> {
        Ok(GenericEncoder {
            file: {
                let mut file = std::io::BufWriter::new(std::fs::File::create(path)?);
                file.write_all(common::MAGIC_NUMBER.as_bytes())?;
                file.write_all(&common::VERSION)?;
                file.write_all(&[common::Type::Generic as u8])?;
                file
            },
            previous_t: 0,
            t0: if zero_t0 { None } else { Some(0) },
        })
    }

    pub fn write(&mut self, event: common::GenericEvent) -> Result<(), utilities::WriteError> {
        let t0 = match self.t0 {
            Some(t0) => t0,
            None => {
                let t0 = event.t;
                self.t0 = Some(t0);
                t0
            }
        };
        let t = event.t - t0;
        if t < self.previous_t {
            return Err(utilities::WriteError::NonMonotonic {
                previous_t: self.previous_t + t0,
                t: t + t0,
            });
        }
        let mut relative_t = t - self.previous_t;
        if relative_t >= 0b11111110 {
            let overflows = relative_t / 0b11111110;
            for _ in 0..overflows {
                self.file.write_all(&[0b11111111])?;
            }
            relative_t -= overflows * 0b11111110;
        }
        self.file.write_all(&[relative_t as u8])?;
        let mut length = event.bytes.len();
        loop {
            self.file.write_all(&[(((length & 0b1111111) << 1)
                | (if (length >> 7) > 0 { 1 } else { 0 }))
                as u8])?;
            length >>= 7;
            if length == 0 {
                break;
            }
        }
        self.file.write_all(&event.bytes)?;
        self.previous_t = t;
        Ok(())
    }
}

impl DvsEncoder {
    pub fn new<P: AsRef<std::path::Path>>(
        path: P,
        zero_t0: bool,
        dimensions: (u16, u16),
    ) -> Result<Self, Error> {
        Ok(DvsEncoder {
            file: open(path, common::Type::Dvs, dimensions)?,
            dimensions,
            previous_t: 0,
            t0: if zero_t0 { None } else { Some(0) },
        })
    }

    pub fn write(
        &mut self,
        event: neuromorphic_types::DvsEvent<u64, u16, u16>,
    ) -> Result<(), utilities::WriteError> {
        let t0 = match self.t0 {
            Some(t0) => t0,
            None => {
                let t0 = event.t;
                self.t0 = Some(t0);
                t0
            }
        };
        let t = event.t - t0;
        if t < self.previous_t {
            return Err(utilities::WriteError::NonMonotonic {
                previous_t: self.previous_t + t0,
                t: t + t0,
            });
        }
        if event.x >= self.dimensions.0 {
            return Err(utilities::WriteError::XOverflow {
                x: event.x,
                width: self.dimensions.0,
            });
        }
        if event.y >= self.dimensions.1 {
            return Err(utilities::WriteError::YOverflow {
                y: event.y,
                height: self.dimensions.1,
            });
        }
        let mut relative_t = t - self.previous_t;
        if relative_t >= 0b1111111 {
            let overflows = relative_t / 0b1111111;
            for _ in 0..overflows {
                self.file.write_all(&[0b11111111])?;
            }
            relative_t -= overflows * 0b1111111;
        }
        self.file
            .write_all(&[(relative_t << 1) as u8 | event.polarity as u8])?;
        self.file.write_all(&event.x.to_le_bytes())?;
        self.file.write_all(&event.y.to_le_bytes())?;
        self.previous_t = t;
        Ok(())
    }
}

impl AtisEncoder {
    pub fn new<P: AsRef<std::path::Path>>(
        path: P,
        zero_t0: bool,
        dimensions: (u16, u16),
    ) -> Result<Self, Error> {
        Ok(AtisEncoder {
            file: open(path, common::Type::Atis, dimensions)?,
            dimensions,
            previous_t: 0,
            t0: if zero_t0 { None } else { Some(0) },
        })
    }

    pub fn write(
        &mut self,
        event: neuromorphic_types::AtisEvent<u64, u16, u16>,
    ) -> Result<(), utilities::WriteError> {
        let t0 = match self.t0 {
            Some(t0) => t0,
            None => {
                let t0 = event.t;
                self.t0 = Some(t0);
                t0
            }
        };
        let t = event.t - t0;
        if t < self.previous_t {
            return Err(utilities::WriteError::NonMonotonic {
                previous_t: self.previous_t + t0,
                t: t + t0,
            });
        }
        if event.x >= self.dimensions.0 {
            return Err(utilities::WriteError::XOverflow {
                x: event.x,
                width: self.dimensions.0,
            });
        }
        if event.y >= self.dimensions.1 {
            return Err(utilities::WriteError::YOverflow {
                y: event.y,
                height: self.dimensions.1,
            });
        }
        let mut relative_t = t - self.previous_t;
        if relative_t >= 0b111111 {
            let overflows = relative_t / 0b111111;
            for _ in 0..overflows / 0b11 {
                self.file.write_all(&[0b11111111])?;
            }
            let overflows_left = overflows % 0b11;
            if overflows_left > 0 {
                self.file.write_all(&[0b11111100 | overflows_left as u8])?;
            }
            relative_t -= overflows * 0b111111;
        }
        self.file.write_all(&[(relative_t << 2) as u8
            | match event.polarity {
                neuromorphic_types::AtisPolarity::Off => 0b00,
                neuromorphic_types::AtisPolarity::ExposureStart => 0b01,
                neuromorphic_types::AtisPolarity::On => 0b10,
                neuromorphic_types::AtisPolarity::ExposureEnd => 0b11,
            }])?;
        self.file.write_all(&event.x.to_le_bytes())?;
        self.file.write_all(&event.y.to_le_bytes())?;
        self.previous_t = t;
        Ok(())
    }
}

impl ColorEncoder {
    pub fn new<P: AsRef<std::path::Path>>(
        path: P,
        zero_t0: bool,
        dimensions: (u16, u16),
    ) -> Result<Self, Error> {
        Ok(ColorEncoder {
            file: open(path, common::Type::Color, dimensions)?,
            dimensions,
            previous_t: 0,
            t0: if zero_t0 { None } else { Some(0) },
        })
    }

    pub fn write(&mut self, event: common::ColorEvent) -> Result<(), utilities::WriteError> {
        let t0 = match self.t0 {
            Some(t0) => t0,
            None => {
                let t0 = event.t;
                self.t0 = Some(t0);
                t0
            }
        };
        let t = event.t - t0;
        if t < self.previous_t {
            return Err(utilities::WriteError::NonMonotonic {
                previous_t: self.previous_t + t0,
                t: t + t0,
            });
        }
        if event.x >= self.dimensions.0 {
            return Err(utilities::WriteError::XOverflow {
                x: event.x,
                width: self.dimensions.0,
            });
        }
        if event.y >= self.dimensions.1 {
            return Err(utilities::WriteError::YOverflow {
                y: event.y,
                height: self.dimensions.1,
            });
        }
        let mut relative_t = t - self.previous_t;
        if relative_t >= 0b11111110 {
            let overflows = relative_t / 0b11111110;
            for _ in 0..overflows {
                self.file.write_all(&[0b11111111])?;
            }
            relative_t -= overflows * 0b11111110;
        }
        self.file.write_all(&[relative_t as u8])?;
        self.file.write_all(&event.x.to_le_bytes())?;
        self.file.write_all(&event.y.to_le_bytes())?;
        self.file.write_all(&[event.r, event.g, event.b])?;
        self.previous_t = t;
        Ok(())
    }
}
