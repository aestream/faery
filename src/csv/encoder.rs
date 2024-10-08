use std::io::Write;

use crate::utilities;

pub enum Output {
    File(std::io::BufWriter<std::fs::File>),
    Stdout(std::io::BufWriter<std::io::Stdout>),
}

impl Output {
    pub fn stdout() -> Self {
        Self::Stdout(std::io::BufWriter::new(std::io::stdout()))
    }

    fn write_all(&mut self, buf: &[u8]) -> Result<(), std::io::Error> {
        match self {
            Output::File(output) => output.write_all(buf),
            Output::Stdout(output) => output.write_all(buf),
        }
    }
}

pub struct Encoder {
    output: Output,
    dimensions: (u16, u16),
    separator: char,
    previous_t: u64,
}

#[derive(thiserror::Error, Debug)]
pub enum Error {
    #[error(transparent)]
    Io(#[from] std::io::Error),

    #[error(transparent)]
    Utf8(#[from] std::str::Utf8Error),

    #[error("the width must be smaller than {maximum} (got {value}")]
    Width { maximum: u16, value: u16 },

    #[error("the height must be smaller than {maximum} (got {value}")]
    Height { maximum: u16, value: u16 },
}

impl Encoder {
    pub fn new(
        mut output: Output,
        separator: u8,
        header: bool,
        dimensions: (u16, u16),
    ) -> Result<Self, Error> {
        let separator = std::str::from_utf8(&[separator])?
            .chars()
            .next()
            .expect("a single byte that is valid UTF-8 must correspond to a character");
        if header {
            output.write_all(
                format!(
                    "t{}x@{}{}y@{}{}on\r\n",
                    separator, dimensions.0, separator, dimensions.1, separator,
                )
                .as_bytes(),
            )?;
        }
        Ok(Self {
            output,
            dimensions,
            separator,
            previous_t: 0,
        })
    }

    pub fn write(
        &mut self,
        event: neuromorphic_types::DvsEvent<u64, u16, u16>,
    ) -> Result<(), utilities::WriteError> {
        if event.t < self.previous_t {
            return Err(utilities::WriteError::NonMonotonic {
                previous_t: self.previous_t,
                t: event.t,
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
        self.output.write_all(
            format!(
                "{}{}{}{}{}{}{}\r\n",
                { event.t },
                self.separator,
                { event.x },
                self.separator,
                { event.y },
                self.separator,
                event.polarity as u8
            )
            .as_bytes(),
        )?;
        self.previous_t = event.t;
        Ok(())
    }
}
