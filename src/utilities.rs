use std::io::BufRead;

pub const BUFFER_SIZE: usize = 65536;
pub const LZ4_MINIMUM_LEVEL: u8 = 1;
pub const LZ4_DEFAULT_LEVEL: u8 = 1;
pub const LZ4_MAXIMUM_LEVEL: u8 = 12;
pub const ZSTD_MINIMUM_LEVEL: u8 = 1;
pub const ZSTD_DEFAULT_LEVEL: u8 = zstd::DEFAULT_COMPRESSION_LEVEL as u8;
pub const ZSTD_MAXIMUM_LEVEL: u8 = 22;

#[derive(thiserror::Error, Debug)]
pub enum ReadError {
    #[error(transparent)]
    Io(#[from] std::io::Error),

    #[error("x overflow (x={x} must be strictly smaller than width={width})")]
    XOverflow { x: u16, width: u16 },

    #[error("y overflow (y={y} must be strictly smaller than height={height})")]
    YOverflow { y: u16, height: u16 },
}

impl From<ReadError> for pyo3::PyErr {
    fn from(error: ReadError) -> Self {
        pyo3::PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(error.to_string())
    }
}

#[derive(thiserror::Error, Debug)]
pub enum WriteError {
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

    #[error("trigger id overflow (id={id} must be strictly smaller than maximum={maximum})")]
    TriggerOverflow { id: u8, maximum: u8 },
}

impl From<WriteError> for pyo3::PyErr {
    fn from(error: WriteError) -> Self {
        pyo3::PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(error.to_string())
    }
}

pub struct Header {
    pub dimensions: Option<(u16, u16)>,
    pub version: Option<String>,
    pub length: u64,
    pub t0: u64,
}

pub fn read_prophesee_header(
    file: &mut std::io::BufReader<std::fs::File>,
    marker: char,
) -> Result<Header, std::io::Error> {
    let mut buffer = String::new();
    let mut width: Option<u16> = None;
    let mut height: Option<u16> = None;
    let mut version: Option<String> = None;
    let mut t0: Option<u64> = None;
    let mut length = 0;
    loop {
        buffer.clear();
        let bytes_read = match file.read_line(&mut buffer) {
            Ok(bytes_read) => bytes_read,
            Err(error) => match error.kind() {
                std::io::ErrorKind::InvalidData => 0,
                _ => return Err(error),
            },
        };
        if bytes_read == 0 || !buffer.starts_with(marker) {
            break;
        }
        length += bytes_read as u64;
        let words: Vec<&str> = buffer[1..]
            .trim()
            .split(&[' ', ';'])
            .map(|word| word.trim())
            .filter(|word| !word.is_empty())
            .collect();
        if words.len() > 1 {
            match words[0] {
                "Version" => {
                    version = Some(match words[1] {
                        "2" | "2.0" => "2".to_owned(),
                        "2.1" => "2.1".to_owned(),
                        "3" | "3.0" => "3".to_owned(),
                        word => word.to_owned(),
                    });
                }
                "Width" => {
                    if let Ok(width_candidate) = words[1].parse() {
                        width = Some(width_candidate);
                    }
                }
                "Height" => {
                    if let Ok(height_candidate) = words[1].parse() {
                        height = Some(height_candidate);
                    }
                }
                "geometry" => {
                    let subwords: Vec<&str> = words[1]
                        .split('x')
                        .map(|subword| subword.trim())
                        .filter(|subword| !subword.is_empty())
                        .collect();
                    if subwords.len() == 2 {
                        if let Ok(width_candidate) = subwords[0].parse() {
                            if let Ok(height_candidate) = subwords[1].parse() {
                                width = Some(width_candidate);
                                height = Some(height_candidate);
                            }
                        }
                    }
                }
                "T0" | "t0" => {
                    if let Ok(t0_candidate) = words[1].parse() {
                        t0 = Some(t0_candidate);
                    }
                }
                "format" => {
                    version = Some(match words[1] {
                        "EVT2" | "evt2" | "EVT2.0" | "evt2.0" => "2".to_owned(),
                        "EVT2.1" | "evt2.1" => "2.1".to_owned(),
                        "EVT3" | "evt3" | "EVT3.0" | "evt3.0" => "3".to_owned(),
                        word => word.to_owned(),
                    });
                    if words.len() == 4 {
                        let mut format_width = None;
                        let mut format_height = None;
                        for word in words[2..4].iter() {
                            let subwords: Vec<&str> = word
                                .split('=')
                                .map(|subword| subword.trim())
                                .filter(|subword| !subword.is_empty())
                                .collect();
                            if subwords.len() == 2 {
                                if subwords[0] == "width" {
                                    if let Ok(width_candidate) = subwords[1].parse() {
                                        format_width = Some(width_candidate);
                                    }
                                } else if subwords[0] == "height" {
                                    if let Ok(height_candidate) = subwords[1].parse() {
                                        format_height = Some(height_candidate);
                                    }
                                }
                            }
                        }
                        if format_width.is_some() && format_height.is_some() {
                            width = format_width;
                            height = format_height;
                        }
                    }
                }
                "evt" => {
                    version = Some(match words[1] {
                        "2" | "2.0" => "2".to_owned(),
                        "2.1" => "2.1".to_owned(),
                        "3" | "3.0" => "3".to_owned(),
                        word => word.to_owned(),
                    });
                }
                _ => (),
            }
        }
    }
    if let Some(width) = width {
        if let Some(height) = height {
            return Ok(Header {
                dimensions: Some((width, height)),
                version,
                length,
                t0: t0.unwrap_or(0),
            });
        }
    }
    Ok(Header {
        dimensions: None,
        version,
        length,
        t0: t0.unwrap_or(0),
    })
}
