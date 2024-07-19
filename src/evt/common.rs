#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Version {
    Evt2,
    Evt21,
    Evt3,
}

#[derive(thiserror::Error, Debug)]
pub enum Error {
    #[error("unknown version \"{0}\" (must be \"evt2\", \"evt2.1\", or \"evt3\")")]
    UnknownVersion(String),
}

impl Version {
    pub fn from_string(string: &str) -> Result<Version, Error> {
        match string {
            "evt2" => Ok(Version::Evt2),
            "evt2.1" => Ok(Version::Evt21),
            "evt3" => Ok(Version::Evt3),
            string => Err(Error::UnknownVersion(string.to_owned())),
        }
    }

    pub fn to_string(self) -> &'static str {
        match self {
            Version::Evt2 => "evt2",
            Version::Evt21 => "evt2.1",
            Version::Evt3 => "evt3",
        }
    }
}
