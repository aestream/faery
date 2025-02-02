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
#[path = "./io_header_generated.rs"]
pub mod io_header_generated;

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
#[path = "./file_data_table_generated.rs"]
pub mod file_data_table_generated;

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

#[derive(Debug, Clone)]
pub struct FileDataDefinition {
    pub byte_offset: i64,
    pub track_id: i32,
    pub size: i32,
    pub elements_count: i64,
    pub start_t: i64,
    pub end_t: i64,
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

    #[error("bad root node child tag (expected \"node\", got \"{0}\")")]
    RootNodeChildTag(String),

    #[error("bad node tag (expected \"node\" or \"attr\", got \"{0}\")")]
    NodeTag(String),

    #[error("missing attribute \"name\" on node")]
    MissingName,

    #[error("missing attribute \"path\" on node wit name \"{0}\"")]
    MissingPath(String),

    #[error("missing attribute \"key\" on attr")]
    MissingKey,

    #[error("missing attribute \"type\" on attr with key \"{0}\"")]
    MissingType(String),

    #[error("unsupported type \"{attr_type}\" on attr with key \"{key}\"")]
    UnsupportedType { key: String, attr_type: String },

    #[error("duplicate attr key \"{0}\"")]
    DuplicateKey(String),

    #[error("empty attr with key \"{0}\"")]
    EmptyAttr(String),

    #[error("unexpected children in attr with key \"{0}\"")]
    UnexpectedAttrChildren(String),

    #[error("the description has no \"outInfo\" node")]
    MissingOutInfoNode,

    #[error("duplicate track id {0}")]
    DuplicateTrackId(i32),

    #[error("missing type identifier for track ID {0}")]
    MissingTypeIdentifier(i32),

    #[error("unknown type identifier \"{type_identifier}\" for track ID {track_id}")]
    UnknownTypeIdentifier {
        track_id: i32,
        type_identifier: String,
    },

    #[error("missing info node for track ID {0}")]
    MissingInfoNode(i32),

    #[error("missing sizeX attribute for track ID {0}")]
    MissingSizeX(i32),

    #[error("missing sizeX attribute for track ID {0}")]
    MissingSizeY(i32),

    #[error("unexpected string value \"{0}\" (expected a number in the range [0, 32768))")]
    UnexpectedStringValue(String),

    #[error("unexpected long value {0} (expected a number in the range [0, 32768))")]
    UnexpectedLongValue(i64),

    #[error("unexpected int value {0} (expected a number in the range [0, 32768))")]
    UnexpectedIntValue(i32),

    #[error("no tracks found in the description")]
    NoTracks,
}

pub enum DescriptionAttribute {
    String(String),
    Int(i32),
    Long(i64),
}

impl DescriptionAttribute {
    fn to_string(&self) -> String {
        match self {
            DescriptionAttribute::String(value) => value.clone(),
            DescriptionAttribute::Int(value) => value.to_string(),
            DescriptionAttribute::Long(value) => value.to_string(),
        }
    }

    fn to_u16(&self) -> Result<u16, DescriptionError> {
        match self {
            DescriptionAttribute::String(value) => {
                Err(DescriptionError::UnexpectedStringValue(value.clone()))
            }
            DescriptionAttribute::Int(value) => {
                if *value < 0 || *value > 32768 as i32 {
                    Err(DescriptionError::UnexpectedIntValue(*value))
                } else {
                    Ok(*value as u16)
                }
            }
            DescriptionAttribute::Long(value) => Err(DescriptionError::UnexpectedLongValue(*value)),
        }
    }

    fn to_xml_string(&self, key: &str, indent: usize) -> String {
        match self {
            DescriptionAttribute::String(value) => format!(
                "{}<attr key=\"{}\" type=\"string\">{}</attr>",
                " ".repeat(indent),
                key,
                value
            ),
            DescriptionAttribute::Int(value) => format!(
                "{}<attr key=\"{}\" type=\"int\">{}</attr>",
                " ".repeat(indent),
                key,
                value
            ),
            DescriptionAttribute::Long(value) => format!(
                "{}<attr key=\"{}\" type=\"long\">{}</attr>",
                " ".repeat(indent),
                key,
                value
            ),
        }
    }
}

pub struct DescriptionNode {
    pub name: String,
    pub path: String,
    pub attributes: std::collections::HashMap<String, DescriptionAttribute>,
    pub nodes: Vec<DescriptionNode>,
}

impl DescriptionNode {
    fn to_xml_string(&self, indent: usize) -> String {
        let mut key_and_attribute: Vec<_> = self.attributes.iter().collect();
        key_and_attribute.sort_by(|a, b| a.0.cmp(b.0));
        format!(
            "{}<node name=\"{}\" path=\"{}\">\n{}{}{}{}{}</node>",
            " ".repeat(indent),
            self.name,
            self.path,
            key_and_attribute
                .iter()
                .map(|(key, attribute)| attribute.to_xml_string(key, indent + 4))
                .collect::<Vec<_>>()
                .join("\n"),
            if self.attributes.is_empty() { "" } else { "\n" },
            self.nodes
                .iter()
                .map(|node| node.to_xml_string(indent + 4))
                .collect::<Vec<_>>()
                .join("\n"),
            if self.nodes.is_empty() { "" } else { "\n" },
            " ".repeat(indent),
        )
    }
}

pub struct Description(pub Vec<DescriptionNode>);

fn parse_node(node: roxmltree::Node<'_, '_>) -> Result<DescriptionNode, DescriptionError> {
    match node.attribute("name") {
        Some(name) => match node.attribute("path") {
            Some(path) => {
                let mut description_node = DescriptionNode {
                    name: name.to_owned(),
                    path: path.to_owned(),
                    attributes: std::collections::HashMap::new(),
                    nodes: Vec::new(),
                };
                for child in node.children() {
                    if child.is_comment() || child.is_text() {
                        continue;
                    }
                    if !child.is_element() {
                        return Err(DescriptionError::NodeTag(
                            child.tag_name().name().to_owned(),
                        ));
                    }
                    match child.tag_name().name() {
                        "node" => {
                            description_node.nodes.push(parse_node(child)?);
                        }
                        "attr" => match child.attribute("key") {
                            Some(key) => {
                                let mut contents = None;
                                for attr_child in child.children() {
                                    if attr_child.is_comment() {
                                        continue;
                                    }
                                    if !attr_child.is_text() || contents.is_some() {
                                        return Err(DescriptionError::UnexpectedAttrChildren(
                                            key.to_owned(),
                                        ));
                                    }
                                    contents = Some(
                                        attr_child.text().expect("the attr child contains text"),
                                    );
                                }
                                if (match contents {
                                    Some(contents) => match child.attribute("type") {
                                        Some(attr_type) => match attr_type {
                                            "int" => description_node.attributes.insert(
                                                key.to_owned(),
                                                DescriptionAttribute::Int(contents.parse()?),
                                            ),
                                            "long" => description_node.attributes.insert(
                                                key.to_owned(),
                                                DescriptionAttribute::Long(contents.parse()?),
                                            ),
                                            "string" => description_node.attributes.insert(
                                                key.to_owned(),
                                                DescriptionAttribute::String(contents.to_owned()),
                                            ),
                                            attr_type => {
                                                return Err(DescriptionError::UnsupportedType {
                                                    key: key.to_owned(),
                                                    attr_type: attr_type.to_owned(),
                                                });
                                            }
                                        },
                                        None => {
                                            return Err(DescriptionError::MissingType(
                                                key.to_owned(),
                                            ))
                                        }
                                    },
                                    None => {
                                        return Err(DescriptionError::EmptyAttr(key.to_owned()))
                                    }
                                })
                                .is_some()
                                {
                                    return Err(DescriptionError::DuplicateKey(key.to_owned()));
                                }
                            }
                            None => return Err(DescriptionError::MissingKey),
                        },
                        name => return Err(DescriptionError::NodeTag(name.to_owned())),
                    }
                }
                Ok(description_node)
            }
            None => Err(DescriptionError::MissingPath(name.to_owned())),
        },
        None => Err(DescriptionError::MissingName),
    }
}

impl Description {
    pub fn from_xml_string(text: &str) -> Result<Self, DescriptionError> {
        let document = roxmltree::Document::parse(text)?;
        let dv_node = match document.root().first_child() {
            Some(content) => content,
            None => return Err(DescriptionError::RootNode),
        };
        if !dv_node.has_tag_name("dv") {
            return Err(DescriptionError::RootNodeTag(
                dv_node.tag_name().name().to_owned(),
            ));
        }
        let mut description_nodes = Vec::new();
        for child in dv_node.children() {
            if child.is_comment() || child.is_text() {
                continue;
            }
            if !child.is_element() || !child.has_tag_name("node") {
                return Err(DescriptionError::RootNodeChildTag(
                    child.tag_name().name().to_owned(),
                ));
            }
            description_nodes.push(parse_node(child)?);
        }
        Ok(Description(description_nodes))
    }

    pub fn id_to_track(&self) -> Result<std::collections::HashMap<i32, Track>, DescriptionError> {
        match self.0.iter().find(|node| node.name == "outInfo") {
            Some(node) => {
                let mut result = std::collections::HashMap::new();
                for track_node in node.nodes.iter() {
                    let track_id = track_node.name.parse::<i32>()?;
                    match track_node.attributes.get("typeIdentifier") {
                        Some(type_identifier) => {
                            match type_identifier {
                                DescriptionAttribute::String(type_identifier) => {
                                    let dimensions = match type_identifier.as_str() {
                                        "EVTS" | "FRME" => Some(
                                            match track_node
                                                .nodes
                                                .iter()
                                                .find(|node| node.name == "info")
                                            {
                                                Some(info_node) => {
                                                    match info_node.attributes.get("sizeX") {
                                                        Some(size_x_attr) => {
                                                            match info_node.attributes.get("sizeY")
                                                            {
                                                                Some(size_y_attr) => (
                                                                    size_x_attr.to_u16()?,
                                                                    size_y_attr.to_u16()?,
                                                                ),
                                                                None => {
                                                                    return Err(DescriptionError::MissingSizeX(track_id));
                                                                }
                                                            }
                                                        }
                                                        None => {
                                                            return Err(
                                                                DescriptionError::MissingSizeX(
                                                                    track_id,
                                                                ),
                                                            );
                                                        }
                                                    }
                                                }
                                                None => {
                                                    return Err(DescriptionError::MissingInfoNode(
                                                        track_id,
                                                    ));
                                                }
                                            },
                                        ),
                                        "IMUS" | "TRIG" => None,
                                        type_identifier => {
                                            return Err(DescriptionError::UnknownTypeIdentifier {
                                                track_id,
                                                type_identifier: type_identifier.to_owned(),
                                            });
                                        }
                                    };
                                    if result
                                        .insert(
                                            track_id,
                                            Track::from_identifier(
                                                type_identifier.as_str(),
                                                dimensions,
                                            )?,
                                        )
                                        .is_some()
                                    {
                                        return Err(DescriptionError::DuplicateTrackId(track_id));
                                    }
                                }
                                type_identifier => {
                                    return Err(DescriptionError::UnknownTypeIdentifier {
                                        track_id,
                                        type_identifier: type_identifier.to_string(),
                                    });
                                }
                            };
                        }
                        None => return Err(DescriptionError::MissingTypeIdentifier(track_id)),
                    }
                }
                if result.is_empty() {
                    Err(DescriptionError::NoTracks)
                } else {
                    Ok(result)
                }
            }
            None => Err(DescriptionError::MissingOutInfoNode),
        }
    }

    pub fn to_xml_string(&self) -> String {
        format!(
            "<dv version=\"2.0\">\n{}\n</dv>",
            self.0
                .iter()
                .map(|node| node.to_xml_string(4))
                .collect::<Vec<_>>()
                .join("\n")
        )
    }
}
