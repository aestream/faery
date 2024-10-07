use std::io::Read;

use crate::utilities;

pub enum Input {
    File(std::io::BufReader<std::fs::File>),
    Stdin(std::io::BufReader<std::io::Stdin>),
}

pub struct Properties {
    dimensions: (u16, u16),
    has_header: bool,
    separator: u8,
    t_index: usize,
    x_index: usize,
    y_index: usize,
    on_index: usize,
    t_scale: f64,
    t0: u64,
    on_value: Vec<u8>,
    off_value: Vec<u8>,
}

impl Properties {
    pub fn new(
        dimensions: (u16, u16),
        has_header: bool,
        separator: u8,
        t_index: usize,
        x_index: usize,
        y_index: usize,
        on_index: usize,
        t_scale: f64,
        t0: u64,
        on_value: Vec<u8>,
        off_value: Vec<u8>,
    ) -> Result<Self, Error> {
        if t_index == x_index {
            return Err(Error::SameIndices {
                first: "t",
                second: "x",
                index: t_index,
            });
        }
        if t_index == y_index {
            return Err(Error::SameIndices {
                first: "t",
                second: "y",
                index: t_index,
            });
        }
        if t_index == on_index {
            return Err(Error::SameIndices {
                first: "t",
                second: "on",
                index: t_index,
            });
        }
        if x_index == y_index {
            return Err(Error::SameIndices {
                first: "x",
                second: "y",
                index: t_index,
            });
        }
        if x_index == on_index {
            return Err(Error::SameIndices {
                first: "x",
                second: "on",
                index: t_index,
            });
        }
        if y_index == on_index {
            return Err(Error::SameIndices {
                first: "y",
                second: "on",
                index: t_index,
            });
        }
        Ok(Self {
            dimensions,
            has_header,
            separator,
            t_index,
            x_index,
            y_index,
            on_index,
            t_scale,
            t0,
            on_value,
            off_value,
        })
    }
}

pub struct Decoder {
    input: Input,
    raw_buffer: Vec<u8>,
    raw_buffer_length: usize,
    eof: bool,
    buffer: Vec<neuromorphic_types::DvsEvent<u64, u16, u16>>,
    header_read: bool,
    t: u64,
    skip_errors: bool,
    properties: Properties,
}

impl Input {
    pub fn stdin() -> Self {
        Self::Stdin(std::io::BufReader::new(std::io::stdin()))
    }

    fn read(&mut self, buf: &mut [u8]) -> std::io::Result<usize> {
        match self {
            Input::File(input) => input.read(buf),
            Input::Stdin(input) => input.read(buf),
        }
    }
}

#[derive(thiserror::Error, Debug)]
pub enum Error {
    #[error(transparent)]
    Io(#[from] std::io::Error),

    #[error("{first} and {second} cannot have the same index (both have the index {index})")]
    SameIndices {
        first: &'static str,
        second: &'static str,
        index: usize,
    },
}

impl Decoder {
    pub fn new(input: Input, properties: Properties, skip_errors: bool) -> Result<Self, Error> {
        Ok(Decoder {
            input,
            raw_buffer: vec![0u8; utilities::BUFFER_SIZE],
            raw_buffer_length: 0,
            eof: false,
            buffer: Vec::with_capacity(utilities::BUFFER_SIZE / 8),
            header_read: !properties.has_header,
            t: 0,
            skip_errors,
            properties,
        })
    }

    pub fn dimensions(&self) -> (u16, u16) {
        self.properties.dimensions
    }
}

#[derive(thiserror::Error, Debug)]
pub enum ReadError {
    #[error(transparent)]
    Io(#[from] std::io::Error),

    #[error("x overflow (x={x} must be strictly smaller than width={width})")]
    XOverflow { x: u16, width: u16 },

    #[error("y overflow (y={y} must be strictly smaller than height={height})")]
    YOverflow { y: u16, height: u16 },

    #[error(transparent)]
    Utf8(#[from] std::str::Utf8Error),

    #[error(
        "parsing field \"{field}\" (\"{value}\") from line \"{line}\" failed (field index {index})"
    )]
    Parse {
        field: &'static str,
        value: String,
        line: String,
        index: usize,
    },

    #[error("the field \"{field}\" is missing from line \"{line}\" (field index {index})")]
    Missing {
        field: &'static str,
        line: String,
        index: usize,
    },

    #[error("parsing field \"on\" (\"{value}\") from line \"{line}\" failed (field index {index}, expected \"{on_value}\" or \"{off_value}\")")]
    UnknownOn {
        value: String,
        line: String,
        index: usize,
        on_value: String,
        off_value: String,
    },
}

impl Decoder {
    #[inline(always)]
    fn parse_word(
        properties: &Properties,
        word_index: usize,
        word: &[u8],
        line: &[u8],
        t: &mut Option<u64>,
        x: &mut Option<u16>,
        y: &mut Option<u16>,
        polarity: &mut Option<neuromorphic_types::DvsPolarity>,
    ) -> Result<(), ReadError> {
        if word_index == properties.t_index {
            let word = std::str::from_utf8(word)?;
            if properties.t_scale == 0.0 {
                t.replace(match word.parse::<u64>() {
                    Ok(value) => value + properties.t0,
                    Err(_) => {
                        return Err(ReadError::Parse {
                            field: "t",
                            value: word.to_owned(),
                            line: std::str::from_utf8(line)?.to_owned(),
                            index: properties.t_index,
                        });
                    }
                });
            } else {
                t.replace(match word.parse::<f64>() {
                    Ok(value) => (value * properties.t_scale).round() as u64 + properties.t0,
                    Err(_) => {
                        return Err(ReadError::Parse {
                            field: "t",
                            value: word.to_owned(),
                            line: std::str::from_utf8(line)?.to_owned(),
                            index: properties.t_index,
                        });
                    }
                });
            }
        } else if word_index == properties.x_index {
            let word = std::str::from_utf8(word)?;
            x.replace(match word.parse() {
                Ok(value) => value,
                Err(_) => {
                    return Err(ReadError::Parse {
                        field: "x",
                        value: word.to_owned(),
                        line: std::str::from_utf8(line)?.to_owned(),
                        index: properties.x_index,
                    });
                }
            });
        } else if word_index == properties.y_index {
            let word = std::str::from_utf8(word)?;
            y.replace(match word.parse() {
                Ok(value) => value,
                Err(_) => {
                    return Err(ReadError::Parse {
                        field: "y",
                        value: word.to_owned(),
                        line: std::str::from_utf8(line)?.to_owned(),
                        index: properties.y_index,
                    });
                }
            });
        } else if word_index == properties.on_index {
            if word == properties.on_value {
                let _ = polarity.replace(neuromorphic_types::DvsPolarity::On);
            } else if word == properties.off_value {
                let _ = polarity.replace(neuromorphic_types::DvsPolarity::Off);
            } else {
                return Err(ReadError::UnknownOn {
                    value: std::str::from_utf8(word)?.to_owned(),
                    line: std::str::from_utf8(line)?.to_owned(),
                    index: properties.on_index,
                    on_value: std::str::from_utf8(&properties.on_value)?.to_owned(),
                    off_value: std::str::from_utf8(&properties.off_value)?.to_owned(),
                });
            }
        }
        Ok(())
    }

    #[inline(always)]
    fn parse_line(
        properties: &Properties,
        t: u64,
        line: &[u8],
        buffer: &mut Vec<neuromorphic_types::DvsEvent<u64, u16, u16>>,
    ) -> Result<u64, ReadError> {
        let mut word_index = 0;
        let mut word_start = 0;
        let mut event_t: Option<u64> = None;
        let mut event_x: Option<u16> = None;
        let mut event_y: Option<u16> = None;
        let mut event_polarity: Option<neuromorphic_types::DvsPolarity> = None;
        for (character_index, character) in line.iter().enumerate() {
            if *character == properties.separator {
                Self::parse_word(
                    properties,
                    word_index,
                    line[word_start..character_index].trim_ascii(),
                    line,
                    &mut event_t,
                    &mut event_x,
                    &mut event_y,
                    &mut event_polarity,
                )?;
                word_index += 1;
                word_start = character_index + 1;
            }
        }
        Self::parse_word(
            properties,
            word_index,
            line[word_start..].trim_ascii(),
            line,
            &mut event_t,
            &mut event_x,
            &mut event_y,
            &mut event_polarity,
        )?;
        match event_t {
            Some(mut event_t) => match event_x {
                Some(event_x) => match event_y {
                    Some(event_y) => match event_polarity {
                        Some(event_polarity) => {
                            if event_t < t {
                                event_t = t;
                            }
                            if event_x >= properties.dimensions.0 {
                                return Err(ReadError::XOverflow {
                                    x: event_x,
                                    width: properties.dimensions.0,
                                });
                            }
                            if event_y >= properties.dimensions.1 {
                                return Err(ReadError::YOverflow {
                                    y: event_y,
                                    height: properties.dimensions.1,
                                });
                            }
                            buffer.push(neuromorphic_types::DvsEvent {
                                t: event_t,
                                x: event_x,
                                y: event_y,
                                polarity: event_polarity,
                            });
                            Ok(event_t)
                        }
                        None => Err(ReadError::Missing {
                            field: "on",
                            line: std::str::from_utf8(line)?.to_owned(),
                            index: properties.on_index,
                        }),
                    },
                    None => Err(ReadError::Missing {
                        field: "y",
                        line: std::str::from_utf8(line)?.to_owned(),
                        index: properties.y_index,
                    }),
                },
                None => Err(ReadError::Missing {
                    field: "x",
                    line: std::str::from_utf8(line)?.to_owned(),
                    index: properties.x_index,
                }),
            },
            None => Err(ReadError::Missing {
                field: "t",
                line: std::str::from_utf8(line)?.to_owned(),
                index: properties.t_index,
            }),
        }
    }

    pub fn next(
        &mut self,
    ) -> Result<Option<&'_ Vec<neuromorphic_types::DvsEvent<u64, u16, u16>>>, ReadError> {
        if self.eof {
            return Ok(None);
        }
        let read = self
            .input
            .read(&mut self.raw_buffer[self.raw_buffer_length..])?;
        if read == 0 {
            self.eof = true;
            if self.raw_buffer_length > 0 && self.header_read {
                self.buffer.clear();
                match Self::parse_line(
                    &self.properties,
                    self.t,
                    self.raw_buffer[0..self.raw_buffer_length].trim_ascii(),
                    &mut self.buffer,
                ) {
                    Ok(t) => {
                        self.t = t;
                    }
                    Err(error) => {
                        if !self.skip_errors {
                            return Err(error);
                        }
                    }
                }
                self.raw_buffer_length = 0;
                self.header_read = true;
                if !self.buffer.is_empty() {
                    return Ok(Some(&self.buffer));
                }
            }
            return Ok(None);
        }
        self.buffer.clear();
        let mut line_start = 0;
        for (character_index, character) in self.raw_buffer[0..self.raw_buffer_length + read]
            .iter()
            .enumerate()
        {
            if *character == b'\n' {
                if self.header_read {
                    match Self::parse_line(
                        &self.properties,
                        self.t,
                        self.raw_buffer[line_start..character_index].trim_ascii(),
                        &mut self.buffer,
                    ) {
                        Ok(t) => {
                            self.t = t;
                        }
                        Err(error) => {
                            if !self.skip_errors {
                                return Err(error);
                            }
                        }
                    }
                } else {
                    self.header_read = true;
                }
                line_start = character_index + 1;
            }
        }
        if line_start < self.raw_buffer_length + read {
            self.raw_buffer
                .copy_within(line_start..self.raw_buffer_length + read, 0);
            self.raw_buffer_length = self.raw_buffer_length + read - line_start;
        } else {
            self.raw_buffer_length = 0;
        }
        return Ok(Some(&self.buffer));
    }
}
