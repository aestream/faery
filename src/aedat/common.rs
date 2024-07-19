#[allow(
    dead_code,
    unused_imports,
    clippy::derivable_impls,
    clippy::derive_partial_eq_without_eq,
    clippy::extra_unused_lifetimes,
    clippy::size_of_in_element_count,
    clippy::needless_lifetimes,
    clippy::unnecessary_cast
)]
#[path = "./ioheader_generated.rs"]
pub mod ioheader_generated;

#[allow(
    dead_code,
    unused_imports,
    clippy::derivable_impls,
    clippy::derive_partial_eq_without_eq,
    clippy::extra_unused_lifetimes,
    clippy::size_of_in_element_count,
    clippy::needless_lifetimes,
    clippy::unnecessary_cast
)]
#[path = "./events_generated.rs"]
pub mod events_generated;

#[allow(
    dead_code,
    unused_imports,
    clippy::derivable_impls,
    clippy::derive_partial_eq_without_eq,
    clippy::extra_unused_lifetimes,
    clippy::size_of_in_element_count,
    clippy::needless_lifetimes,
    clippy::unnecessary_cast
)]
#[path = "./frame_generated.rs"]
pub mod frame_generated;

#[allow(
    dead_code,
    unused_imports,
    clippy::derivable_impls,
    clippy::derive_partial_eq_without_eq,
    clippy::extra_unused_lifetimes,
    clippy::size_of_in_element_count,
    clippy::needless_lifetimes,
    clippy::unnecessary_cast
)]
#[path = "./imus_generated.rs"]
pub mod imus_generated;

#[allow(
    dead_code,
    unused_imports,
    clippy::derivable_impls,
    clippy::derive_partial_eq_without_eq,
    clippy::extra_unused_lifetimes,
    clippy::size_of_in_element_count,
    clippy::needless_lifetimes,
    clippy::unnecessary_cast
)]
#[path = "./triggers_generated.rs"]
pub mod triggers_generated;

pub const MAGIC_NUMBER: &str = "#!AER-DAT4.0\r\n";

#[derive(Clone, Copy, PartialEq, Eq)]
pub enum Track {
    Events {
        dimensions: (u16, u16),
        previous_t: u64,
    },
    Frame {
        dimensions: (u16, u16),
        previous_t: u64,
    },
    Imus {
        previous_t: u64,
    },
    Triggers {
        previous_t: u64,
    },
}

#[derive(thiserror::Error, Debug)]
pub enum Error {
    #[error(
        "unknown track identifier \"{0}\" (must be \"EVTS\", \"FRME\", \"IMUS\", or \"TRIG\")"
    )]
    UnknownIdentifier(String),

    #[error(
        "unknown track data type \"{0}\" (must be \"events\", \"frame\", \"imus\", or \"triggers\")"
    )]
    UnknownDataType(String),

    #[error("the track \"{0}\" must have dimensions")]
    MissingDimensions(String),

    #[error("the track \"{name}\" must not have dimensions (got {width}x{height})")]
    Dimensions {
        name: String,
        width: u16,
        height: u16,
    },

    #[error("the width must be strictly smaller than {maximum} (got {value})")]
    Width { value: u16, maximum: u16 },

    #[error("the height must be strictly smaller than {maximum} (got {value})")]
    Height { value: u16, maximum: u16 },
}

impl Track {
    pub fn from_identifier(
        identifier: &str,
        dimensions: Option<(u16, u16)>,
    ) -> Result<Self, Error> {
        match identifier {
            "EVTS" | "FRME" => match dimensions {
                Some(dimensions) => {
                    if dimensions.0 >= 32768 {
                        return Err(Error::Width {
                            value: dimensions.0,
                            maximum: 32768,
                        });
                    }
                    if dimensions.1 >= 32768 {
                        return Err(Error::Height {
                            value: dimensions.1,
                            maximum: 32768,
                        });
                    }
                    Ok(if identifier == "EVTS" {
                        Track::Events {
                            dimensions,
                            previous_t: 0,
                        }
                    } else {
                        Track::Frame {
                            dimensions,
                            previous_t: 0,
                        }
                    })
                }
                None => Err(Error::MissingDimensions(identifier.to_owned())),
            },
            "IMUS" | "TRIG" => match dimensions {
                Some(dimensions) => Err(Error::Dimensions {
                    name: identifier.to_owned(),
                    width: dimensions.0,
                    height: dimensions.1,
                }),
                None => Ok(if identifier == "IMUS" {
                    Track::Imus { previous_t: 0 }
                } else {
                    Track::Triggers { previous_t: 0 }
                }),
            },
            identifier => Err(Error::UnknownIdentifier(identifier.to_owned())),
        }
    }

    pub fn to_identifier(&self) -> &'static str {
        match self {
            Self::Events { .. } => "EVTS",
            Self::Frame { .. } => "FRME",
            Self::Imus { .. } => "IMUS",
            Self::Triggers { .. } => "TRIG",
        }
    }

    pub fn from_data_type(data_type: &str, dimensions: Option<(u16, u16)>) -> Result<Self, Error> {
        Self::from_identifier(
            match data_type {
                "events" => "EVTS",
                "frame" => "FRME",
                "imus" => "IMUS",
                "triggers" => "TRIG",
                data_type => return Err(Error::UnknownDataType(data_type.to_owned())),
            },
            dimensions,
        )
    }

    pub fn to_data_type(&self) -> &'static str {
        match self {
            Self::Events { .. } => "events",
            Self::Frame { .. } => "frame",
            Self::Imus { .. } => "imus",
            Self::Triggers { .. } => "triggers",
        }
    }

    pub fn dimensions(&self) -> Option<(u16, u16)> {
        match self {
            Self::Events { dimensions, .. } => Some(*dimensions),
            Self::Frame { dimensions, .. } => Some(*dimensions),
            Self::Imus { .. } => None,
            Self::Triggers { .. } => None,
        }
    }
}

#[derive(thiserror::Error, Debug)]
pub enum DescriptionError {
    #[error(transparent)]
    Common(#[from] Error),

    #[error(transparent)]
    Roxmltree(#[from] roxmltree::Error),

    #[error(transparent)]
    ParseInt(#[from] std::num::ParseIntError),

    #[error("the description has no root node")]
    RootNode,

    #[error("bad root node tag (expected \"dv\", got \"{0}\")")]
    RootNodeTag(String),

    #[error("the description has no \"outInfo\" node")]
    OutInfoNode,

    #[error("unexpected child in \"outInfo\" (expected \"node\", got {0})")]
    TrackNodeTag(String),

    #[error("missing track node ID")]
    MissingTrackId,

    #[error("missing type for track ID {0}")]
    MissingType(u32),

    #[error("empty type for track ID {0}")]
    EmptyType(u32),

    #[error("missing sizeX attribute for track ID {0}")]
    MissingSizeX(u32),

    #[error("empty sizeX attribute for track ID {0}")]
    EmptySizeX(u32),

    #[error("missing sizeX attribute for track ID {0}")]
    MissingSizeY(u32),

    #[error("empty sizeX attribute for track ID {0}")]
    EmptySizeY(u32),

    #[error("missing info node for track ID {0}")]
    MissingInfoNode(u32),

    #[error("duplicated track ID {0}")]
    DuplicatedTrackId(u32),

    #[error("no tracks found in the description")]
    NoTracks,
}

pub fn description_to_id_to_tracks(
    description: &str,
) -> Result<std::collections::HashMap<u32, Track>, DescriptionError> {
    let document = roxmltree::Document::parse(&description)?;
    let dv_node = match document.root().first_child() {
        Some(content) => content,
        None => return Err(DescriptionError::RootNode),
    };
    if !dv_node.has_tag_name("dv") {
        return Err(DescriptionError::RootNodeTag(
            dv_node.tag_name().name().to_owned(),
        ));
    }
    let output_node = match dv_node.children().find(|node| {
        node.is_element() && node.has_tag_name("node") && node.attribute("name") == Some("outInfo")
    }) {
        Some(content) => content,
        None => return Err(DescriptionError::OutInfoNode),
    };
    let mut id_to_track = std::collections::HashMap::new();
    for track_node in output_node.children() {
        if track_node.is_element() && track_node.has_tag_name("node") {
            if !track_node.has_tag_name("node") {
                return Err(DescriptionError::TrackNodeTag(
                    track_node.tag_name().name().to_owned(),
                ));
            }
            let track_id = match track_node.attribute("name") {
                Some(content) => content,
                None => return Err(DescriptionError::MissingTrackId),
            }
            .parse::<u32>()?;
            let identifier = match track_node.children().find(|node| {
                node.is_element()
                    && node.has_tag_name("attr")
                    && node.attribute("key") == Some("typeIdentifier")
            }) {
                Some(content) => match content.text() {
                    Some(content) => content,
                    None => return Err(DescriptionError::EmptyType(track_id)),
                },
                None => return Err(DescriptionError::MissingType(track_id)),
            }
            .to_string();
            let dimensions;
            if identifier == "EVTS" || identifier == "FRME" {
                let info_node = match track_node.children().find(|node| {
                    node.is_element()
                        && node.has_tag_name("node")
                        && node.attribute("name") == Some("info")
                }) {
                    Some(content) => content,
                    None => return Err(DescriptionError::MissingInfoNode(track_id)),
                };
                let width = match info_node.children().find(|node| {
                    node.is_element()
                        && node.has_tag_name("attr")
                        && node.attribute("key") == Some("sizeX")
                }) {
                    Some(content) => match content.text() {
                        Some(content) => content,
                        None => return Err(DescriptionError::EmptySizeX(track_id)),
                    },
                    None => return Err(DescriptionError::MissingSizeX(track_id)),
                }
                .parse::<u16>()?;
                let height = match info_node.children().find(|node| {
                    node.is_element()
                        && node.has_tag_name("attr")
                        && node.attribute("key") == Some("sizeY")
                }) {
                    Some(content) => match content.text() {
                        Some(content) => content,
                        None => return Err(DescriptionError::EmptySizeY(track_id)),
                    },
                    None => return Err(DescriptionError::MissingSizeY(track_id)),
                }
                .parse::<u16>()?;
                dimensions = Some((width, height));
            } else {
                dimensions = None;
            }
            if id_to_track
                .insert(track_id, Track::from_identifier(&identifier, dimensions)?)
                .is_some()
            {
                return Err(DescriptionError::DuplicatedTrackId(track_id));
            }
        }
    }
    if id_to_track.is_empty() {
        Err(DescriptionError::NoTracks)
    } else {
        Ok(id_to_track)
    }
}
