use std::io::Write;

use crate::dat::common;

pub struct Encoder {
    file: std::io::BufWriter<std::fs::File>,
    version: common::Version,
    event_type: common::Type,
    previous_t: u64,
    t0: Option<u64>,
}

#[derive(thiserror::Error, Debug)]
pub enum Error {
    #[error(transparent)]
    Io(#[from] std::io::Error),

    #[error("the width must be smaller than {maximum} (got {value}")]
    Width { maximum: u16, value: u16 },

    #[error("the height must be smaller than {maximum} (got {value}")]
    Height { maximum: u16, value: u16 },
}

impl Encoder {
    pub fn new<P: AsRef<std::path::Path>>(
        path: P,
        version: common::Version,
        zero_t0: bool,
        event_type: common::Type,
    ) -> Result<Self, Error> {
        match event_type {
            common::Type::Event2d(width, height) | common::Type::EventCd(width, height) => {
                match version {
                    common::Version::Dat1 => {
                        if width > 512 {
                            return Err(Error::Width {
                                maximum: 512,
                                value: width,
                            });
                        }
                        if height > 256 {
                            return Err(Error::Height {
                                maximum: 256,
                                value: height,
                            });
                        }
                    }
                    common::Version::Dat2 => {
                        if width > 16384 {
                            return Err(Error::Width {
                                maximum: 16384,
                                value: width,
                            });
                        }
                        if height > 16384 {
                            return Err(Error::Height {
                                maximum: 16384,
                                value: height,
                            });
                        }
                    }
                }
            }
            common::Type::EventExtTrigger => {}
        }
        Ok(Self {
            file: {
                let mut file = std::io::BufWriter::new(std::fs::File::create(path)?);
                file.write_all(
                    format!(
                        "% Version {}\n",
                        match version {
                            common::Version::Dat1 => 1,
                            common::Version::Dat2 => 2,
                        }
                    )
                    .as_bytes(),
                )?;
                match event_type {
                    common::Type::Event2d(width, height) | common::Type::EventCd(width, height) => {
                        file.write_all(format!("% Width {width}\n").as_bytes())?;
                        file.write_all(format!("% Height {height}\n").as_bytes())?;
                    }
                    common::Type::EventExtTrigger => {}
                }
                if !zero_t0 {
                    file.write_all(&[
                        match event_type {
                            common::Type::Event2d { .. } => 0x00,
                            common::Type::EventCd { .. } => 0x0C,
                            common::Type::EventExtTrigger => 0x0E,
                        },
                        8,
                    ])?;
                }
                file
            },
            version,
            event_type,
            previous_t: 0,
            t0: if zero_t0 { None } else { Some(0) },
        })
    }

    pub fn t0(&self) -> Option<u64> {
        self.t0
    }
}

#[derive(thiserror::Error, Debug)]
pub enum PacketError {
    #[error(transparent)]
    Io(#[from] std::io::Error),

    #[error(
        "the event's timestamp ({t}) is smaller than the previous event's timestamp ({previous_t})"
    )]
    NonMonotonic { previous_t: u64, t: u64 },

    #[error("x overflow (x={x} must be strictly smaller than width={width})")]
    XOverflow { x: u16, width: u16 },

    #[error("y overflow (y={y} must be strictly smaller than height={height})")]
    YOverflow { y: u16, height: u16 },

    #[error("payload overflow (payload={0} must be strictly smaller than 8)")]
    PayloadOverflow(u8),
}

impl Encoder {
    pub fn write(&mut self, event: common::Event) -> Result<(), PacketError> {
        let t0 = match self.t0 {
            Some(t0) => t0,
            None => {
                let t0 = event.t;
                self.t0 = Some(t0);
                self.file.write_all(format!("% T0 {}\n", t0).as_bytes())?;
                self.file.write_all(&[
                    match self.event_type {
                        common::Type::Event2d { .. } => 0x00,
                        common::Type::EventCd { .. } => 0x0C,
                        common::Type::EventExtTrigger => 0x0E,
                    },
                    8,
                ])?;
                t0
            }
        };
        if event.t < self.previous_t {
            return Err(PacketError::NonMonotonic {
                previous_t: self.previous_t,
                t: event.t,
            });
        }
        match self.event_type {
            common::Type::Event2d(width, height) | common::Type::EventCd(width, height) => {
                if event.x >= width {
                    return Err(PacketError::XOverflow { x: event.x, width });
                }
                if event.y >= height {
                    return Err(PacketError::YOverflow { y: event.y, height });
                }
            }
            common::Type::EventExtTrigger => {}
        }
        if event.payload >= 8 {
            return Err(PacketError::PayloadOverflow(event.payload));
        }
        match self.version {
            common::Version::Dat1 => {
                self.file.write_all(&u64::to_le_bytes(
                    ((event.payload as u64) << 49)
                        | ((event.y as u64) << 41)
                        | ((event.x as u64) << 32)
                        | ((event.t - t0) & 0xFFFFFFFF_u64),
                ))?;
            }
            common::Version::Dat2 => {
                self.file.write_all(&u64::to_le_bytes(
                    ((event.payload as u64) << 60)
                        | ((event.y as u64) << 46)
                        | ((event.x as u64) << 32)
                        | ((event.t - t0) & 0xFFFFFFFF_u64),
                ))?;
            }
        }
        self.previous_t = event.t;
        Ok(())
    }
}
