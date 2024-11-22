#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Version {
    Dat1,
    Dat2,
}

#[derive(Debug, Copy, Clone)]
#[repr(C, packed)]
pub struct Event {
    pub t: u64,
    pub x: u16,
    pub y: u16,
    pub payload: u8,
}

#[derive(Debug, Copy, Clone)]
pub enum Type {
    Event2d(u16, u16),
    EventCd(u16, u16),
    EventExtTrigger,
}

#[derive(thiserror::Error, Debug)]
pub enum TypeError {
    #[error("unknown type \"{0}\" (must be \"2d\", \"cd\", or \"trigger\")")]
    Unknown(String),

    #[error("the type \"{0}\" must have dimensions")]
    MissingDimensions(String),

    #[error("the type \"trigger\" must not have dimensions (got {0}x{1})")]
    Dimensions(u16, u16),
}

impl Type {
    pub fn new(event_type: &str, dimensions: Option<(u16, u16)>) -> Result<Self, TypeError> {
        match event_type {
            "2d" => match dimensions {
                Some(dimensions) => Ok(Self::Event2d(dimensions.0, dimensions.1)),
                None => Err(TypeError::MissingDimensions(event_type.to_owned())),
            },
            "cd" => match dimensions {
                Some(dimensions) => Ok(Self::EventCd(dimensions.0, dimensions.1)),
                None => Err(TypeError::MissingDimensions(event_type.to_owned())),
            },
            "trigger" => match dimensions {
                Some(dimensions) => Err(TypeError::Dimensions(dimensions.0, dimensions.1)),
                None => Ok(Self::EventExtTrigger),
            },
            event_type => Err(TypeError::Unknown(event_type.to_owned())),
        }
    }

    pub fn to_string(&self) -> &'static str {
        match self {
            Self::Event2d { .. } => "2d",
            Self::EventCd { .. } => "cd",
            Self::EventExtTrigger => "trigger",
        }
    }
}

#[derive(thiserror::Error, Debug)]
pub enum Error {
    #[error("unknown version \"{0}\" (must be \"dat1\" or \"dat2\")")]
    UnknownVersion(String),
}

impl Version {
    pub fn from_string(string: &str) -> Result<Version, Error> {
        match string {
            "dat1" => Ok(Version::Dat1),
            "dat2" => Ok(Version::Dat2),
            string => Err(Error::UnknownVersion(string.to_owned())),
        }
    }

    pub fn to_string(self) -> &'static str {
        match self {
            Version::Dat1 => "dat1",
            Version::Dat2 => "dat2",
        }
    }
}
