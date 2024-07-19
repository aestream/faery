use std::io::Write;

use crate::evt::common;
use crate::utilities;

pub struct Evt2Encoder {
    file: std::io::BufWriter<std::fs::File>,
    dimensions: (u16, u16),
    previous_t: u64,
    t_high: u64,
    t0: Option<u64>,
}

pub struct Evt21Encoder {
    file: std::io::BufWriter<std::fs::File>,
    dimensions: (u16, u16),
    previous_t: u64,
    t0: Option<u64>,
}

pub struct Evt3Encoder {
    file: std::io::BufWriter<std::fs::File>,
    dimensions: (u16, u16),
    previous_t: u64,
    msb: u64,
    vector: Vector,
    t0: Option<u64>,
}

pub enum Encoder {
    Evt2Encoder(Evt2Encoder),
    Evt21Encoder(Evt21Encoder),
    Evt3Encoder(Evt3Encoder),
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

const EVT2_MAXIMUM_T_HIGH_DELTA: u64 = 1 << 26;
const EVT3_MAXIMUM_MSB_DELTA: u64 = 1 << 10;

impl Encoder {
    pub fn new<P: AsRef<std::path::Path>>(
        path: P,
        version: common::Version,
        zero_t0: bool,
        dimensions: (u16, u16),
    ) -> Result<Self, Error> {
        Ok(match version {
            common::Version::Evt2 => {
                if dimensions.0 > 2048 {
                    return Err(Error::Width {
                        maximum: 2048,
                        value: dimensions.0,
                    });
                }
                if dimensions.1 > 2048 {
                    return Err(Error::Height {
                        maximum: 2048,
                        value: dimensions.1,
                    });
                }
                Self::Evt2Encoder(Evt2Encoder {
                    file: {
                        let mut file = std::io::BufWriter::new(std::fs::File::create(path)?);
                        file.write_all(b"% evt 2.0\n")?;
                        file.write_all(
                            format!(
                                "% format EVT2;width={};height={}\n",
                                dimensions.0, dimensions.1
                            )
                            .as_bytes(),
                        )?;
                        file.write_all(
                            format!("% geometry {}x{}\n", dimensions.0, dimensions.1).as_bytes(),
                        )?;
                        file
                    },
                    dimensions,
                    previous_t: 0,
                    t_high: u64::MAX,
                    t0: if zero_t0 { None } else { Some(0) },
                })
            }
            common::Version::Evt21 => {
                todo!("implement EVT2.1 (example / test data needed), find width/height limits");
                /*
                Self::Evt21Encoder(Evt21Encoder {
                    file: {
                        let mut file = std::io::BufWriter::new(std::fs::File::create(path)?);
                        file.write_all(b"% evt 2.1\n")?;
                        file.write_all(
                            format!("% format EVT2.1;width={width};height={height}\n").as_bytes(),
                        )?;
                        file.write_all(format!("% geometry {width}x{height}\n").as_bytes())?;
                        file
                    },
                    width,
                    height,
                    previous_t: 0,
                })
                */
            }
            common::Version::Evt3 => {
                if dimensions.0 > 4096 {
                    return Err(Error::Width {
                        maximum: 4096,
                        value: dimensions.0,
                    });
                }
                if dimensions.1 > 4096 {
                    return Err(Error::Height {
                        maximum: 4096,
                        value: dimensions.1,
                    });
                }
                Self::Evt3Encoder(Evt3Encoder {
                    file: {
                        let mut file = std::io::BufWriter::new(std::fs::File::create(path)?);
                        file.write_all(b"% evt 3.0\n")?;
                        file.write_all(
                            format!(
                                "% format EVT3;width={};height={}\n",
                                dimensions.0, dimensions.1
                            )
                            .as_bytes(),
                        )?;
                        file.write_all(
                            format!("% geometry {}x{}\n", dimensions.0, dimensions.1).as_bytes(),
                        )?;
                        file
                    },
                    dimensions,
                    previous_t: 0,
                    msb: u64::MAX,
                    vector: Vector {
                        previous_y: None,
                        previous_x_and_polarity: None,
                        x: 0,
                        y: 0,
                        polarity: neuromorphic_types::DvsPolarity::Off,
                        bits: 0,
                        index: 0,
                    },
                    t0: if zero_t0 { None } else { Some(0) },
                })
            }
        })
    }

    pub fn write_dvs_event(
        &mut self,
        event: neuromorphic_types::DvsEvent<u64, u16, u16>,
    ) -> Result<(), utilities::WriteError> {
        match self {
            Encoder::Evt2Encoder(encoder) => encoder.write_dvs_event(event),
            Encoder::Evt21Encoder(_encoder) => {
                todo!("implement EVT2.1 (example / test data needed)");
            }
            Encoder::Evt3Encoder(encoder) => encoder.write_dvs_event(event),
        }
    }

    pub fn write_trigger_event(
        &mut self,
        event: neuromorphic_types::TriggerEvent<u64, u8>,
    ) -> Result<(), utilities::WriteError> {
        match self {
            Encoder::Evt2Encoder(encoder) => encoder.write_trigger_event(event),
            Encoder::Evt21Encoder(_encoder) => {
                todo!("implement EVT2.1 (example / test data needed)");
            }
            Encoder::Evt3Encoder(encoder) => encoder.write_trigger_event(event),
        }
    }

    pub fn t0(&self) -> Option<u64> {
        match self {
            Encoder::Evt2Encoder(encoder) => encoder.t0,
            Encoder::Evt21Encoder(encoder) => encoder.t0,
            Encoder::Evt3Encoder(encoder) => encoder.t0,
        }
    }
}

impl Evt2Encoder {
    #[inline(always)]
    fn update_t(&mut self, t: u64) -> Result<(), std::io::Error> {
        if self.t_high == u64::MAX {
            let target_t_high = (t >> 6) << 6;
            self.t_high = 0;
            if target_t_high == 0 {
                self.file.write_all(
                    &(((0b1000_u32) << 28) | (((self.t_high >> 6) & 0xFFFFFFF) as u32))
                        .to_le_bytes(),
                )?;
            } else {
                while self.t_high < target_t_high {
                    self.t_high += (target_t_high - self.t_high).min(EVT2_MAXIMUM_T_HIGH_DELTA);
                    self.file.write_all(
                        &(((0b1000_u32) << 28) | (((self.t_high >> 6) & 0xFFFFFFF) as u32))
                            .to_le_bytes(),
                    )?;
                }
            }
        } else if t != self.previous_t {
            let target_t_high = (t >> 6) << 6;
            while self.t_high < target_t_high {
                self.t_high += (target_t_high - self.t_high).min(EVT2_MAXIMUM_T_HIGH_DELTA);
                self.file.write_all(
                    &(((0b1000_u32) << 28) | (((self.t_high >> 6) & 0xFFFFFFF) as u32))
                        .to_le_bytes(),
                )?;
            }
        }
        Ok(())
    }

    pub fn write_dvs_event(
        &mut self,
        event: neuromorphic_types::DvsEvent<u64, u16, u16>,
    ) -> Result<(), utilities::WriteError> {
        let t0 = match self.t0 {
            Some(t0) => t0,
            None => {
                let t0 = event.t;
                self.t0 = Some(t0);
                self.file.write_all(format!("% t0 {}\n", t0).as_bytes())?;
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
        self.update_t(t)?;
        self.file.write_all(
            &((match event.polarity {
                neuromorphic_types::DvsPolarity::Off => 0b0000_u32,
                neuromorphic_types::DvsPolarity::On => 0b0001_u32,
            } << 28)
                | ((((t - self.t_high) & 0b111111) as u32) << 22)
                | ((event.x as u32) << 11)
                | (event.y as u32))
                .to_le_bytes(),
        )?;
        self.previous_t = t;
        Ok(())
    }

    pub fn write_trigger_event(
        &mut self,
        event: neuromorphic_types::TriggerEvent<u64, u8>,
    ) -> Result<(), utilities::WriteError> {
        let t0 = match self.t0 {
            Some(t0) => t0,
            None => {
                let t0 = event.t;
                self.t0 = Some(t0);
                self.file.write_all(format!("% t0 {}\n", t0).as_bytes())?;
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
        if event.id > 0b11111 {
            return Err(utilities::WriteError::TriggerOverflow {
                id: event.id,
                maximum: 32,
            });
        }
        self.update_t(t)?;
        self.file.write_all(
            &((0b1010_u32 << 28)
                | ((((t - self.t_high) & 0b111111) as u32) << 22)
                | ((event.id as u32) << 8)
                | match event.polarity {
                    neuromorphic_types::TriggerPolarity::Falling => 0b0,
                    neuromorphic_types::TriggerPolarity::Rising => 0b1,
                })
            .to_le_bytes(),
        )?;
        self.previous_t = t;
        Ok(())
    }
}

impl Evt3Encoder {
    #[inline(always)]
    fn update_t(&mut self, t: u64) -> Result<(), std::io::Error> {
        let mut update_lsb = false;
        if self.msb == u64::MAX {
            let target_msb = t >> 12;
            self.msb = 0;
            if target_msb == 0 {
                self.file
                    .write_all(&((0b1000 << 12) | (self.msb & 0xFFF) as u16).to_le_bytes())?;
            } else {
                while self.msb < target_msb {
                    self.msb += (target_msb - self.msb).min(EVT3_MAXIMUM_MSB_DELTA);
                    self.file
                        .write_all(&((0b1000 << 12) | (self.msb & 0xFFF) as u16).to_le_bytes())?;
                }
            }
            update_lsb = true;
        } else if t != self.previous_t {
            if self.vector.index > 0 {
                self.vector.flush(&mut self.file)?;
            }
            let target_msb = t >> 12;
            while self.msb < target_msb {
                self.msb += (target_msb - self.msb).min(EVT3_MAXIMUM_MSB_DELTA);
                self.file
                    .write_all(&((0b1000 << 12) | (self.msb & 0xFFF) as u16).to_le_bytes())?;
                update_lsb = true;
            }
        }
        if self.msb == u64::MAX || t != self.previous_t {
            let previous_lsb = (self.previous_t & 0xFFF) as u16;
            let lsb = (t & 0xFFF) as u16;
            if update_lsb || previous_lsb != lsb {
                self.file
                    .write_all(&(((0b0110 << 12) | lsb) as u16).to_le_bytes())?;
            }
        }
        Ok(())
    }

    pub fn write_dvs_event(
        &mut self,
        event: neuromorphic_types::DvsEvent<u64, u16, u16>,
    ) -> Result<(), utilities::WriteError> {
        let t0 = match self.t0 {
            Some(t0) => t0,
            None => {
                let t0 = event.t;
                self.t0 = Some(t0);
                self.file.write_all(format!("% t0 {}\n", t0).as_bytes())?;
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
        self.update_t(t)?;
        self.vector
            .push_dvs_event(event.x, event.y, event.polarity, &mut self.file)?;
        self.previous_t = t;
        Ok(())
    }

    pub fn write_trigger_event(
        &mut self,
        event: neuromorphic_types::TriggerEvent<u64, u8>,
    ) -> Result<(), utilities::WriteError> {
        let t0 = match self.t0 {
            Some(t0) => t0,
            None => {
                let t0 = event.t;
                self.t0 = Some(t0);
                self.file.write_all(format!("% t0 {}\n", t0).as_bytes())?;
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
        if event.id > 0b1111 {
            return Err(utilities::WriteError::TriggerOverflow {
                id: event.id,
                maximum: 16,
            });
        }
        self.update_t(t)?;
        if self.vector.index > 0 {
            self.vector.flush(&mut self.file)?;
        }
        self.file.write_all(
            &(((0b1010 << 12)
                | (((event.id & 0b1111) as u16) << 8)
                | match event.polarity {
                    neuromorphic_types::TriggerPolarity::Falling => 0,
                    neuromorphic_types::TriggerPolarity::Rising => 1,
                }) as u16)
                .to_le_bytes(),
        )?;
        self.previous_t = t;
        Ok(())
    }
}

impl Drop for Evt3Encoder {
    fn drop(&mut self) {
        if self.vector.index > 0 {
            let _ = self.vector.flush(&mut self.file);
        }
    }
}

struct Vector {
    previous_y: Option<u16>,
    previous_x_and_polarity: Option<(u16, neuromorphic_types::DvsPolarity)>,
    x: u16,
    y: u16,
    polarity: neuromorphic_types::DvsPolarity,
    bits: u16,
    index: u8,
}

impl Vector {
    fn flush<W: std::io::Write>(&mut self, mut output: W) -> Result<(), std::io::Error> {
        assert_ne!(self.index, 0);
        if !self
            .previous_y
            .is_some_and(|previous_y| previous_y == self.y)
        {
            output.write_all(&((0b0000 << 12) | self.y).to_le_bytes())?;
        }
        self.previous_y = Some(self.y);
        if self.bits == 1 {
            output.write_all(
                &((0b0010 << 12)
                    | match self.polarity {
                        neuromorphic_types::DvsPolarity::Off => 0b0000_00000000,
                        neuromorphic_types::DvsPolarity::On => 0b1000_00000000,
                    }
                    | self.x)
                    .to_le_bytes(),
            )?;
            self.previous_x_and_polarity = None;
        } else {
            if !self
                .previous_x_and_polarity
                .is_some_and(|previous_x_and_polarity| {
                    previous_x_and_polarity.0 == self.x
                        && previous_x_and_polarity.1 as u8 == self.polarity as u8
                })
            {
                output.write_all(
                    &((0b0011 << 12)
                        | match self.polarity {
                            neuromorphic_types::DvsPolarity::Off => 0b0000_00000000,
                            neuromorphic_types::DvsPolarity::On => 0b1000_00000000,
                        }
                        | self.x)
                        .to_le_bytes(),
                )?;
            }
            if (self.bits >> 8) > 0 {
                output.write_all(&((0b0100 << 12) | self.bits).to_le_bytes())?;
                self.previous_x_and_polarity = Some((self.x + 12, self.polarity));
            } else {
                output.write_all(&((0b0101 << 12) | self.bits).to_le_bytes())?;
                self.previous_x_and_polarity = Some((self.x + 8, self.polarity));
            }
        }
        self.bits = 0;
        self.index = 0;
        Ok(())
    }

    fn push_dvs_event<W: std::io::Write>(
        &mut self,
        x: u16,
        y: u16,
        polarity: neuromorphic_types::DvsPolarity,
        mut output: W,
    ) -> Result<(), std::io::Error> {
        if self.index == 0 {
            self.x = x;
            self.y = y;
            self.polarity = polarity;
            self.bits = 0b1;
            self.index = 1;
        } else {
            if y == self.y
                && polarity as u8 == self.polarity as u8
                && x >= self.x + self.index as u16
                && x < self.x + 12
            {
                let delta = (x - self.x) as u8;
                self.bits |= 0b1 << delta;
                if delta == 11 {
                    self.flush(&mut output)?;
                } else {
                    self.index = delta + 1;
                }
            } else {
                self.flush(&mut output)?;
                self.x = x;
                self.y = y;
                self.polarity = polarity;
                self.bits = 0b1;
                self.index = 1;
            }
        }
        Ok(())
    }
}
