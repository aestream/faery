#[derive(thiserror::Error, Debug)]
pub enum Error {
    #[error("io error: {0}")]
    Io(#[from] std::io::Error),

    #[error("rosbag error: {0}")]
    Rosbag(String),
}

pub struct Decoder {
    inner: rosbag::RosBag,
    event_buffer: Vec<neuromorphic_types::PolarityEvent<u64, u16, u16>>,
    dimensions: (u16, u16),
    t: u64,
}

impl Decoder {
    pub fn new<P: AsRef<std::path::Path>>(path: P) -> Result<Self, Error> {
        let inner = rosbag::RosBag::new(path)?;
        for record in inner.chunk_records() {
            match record.map_err(|error| Error::Rosbag(format!("{error:?}")))? {
                rosbag::ChunkRecord::Chunk(chunk) => {
                    for message in chunk.messages() {
                        if message.is_ok() {
                            todo!("decode the message");
                        }
                    }
                }
                rosbag::ChunkRecord::IndexData(_) => {}
            }
        }

        Ok(Decoder {
            inner,
            event_buffer: Vec::new(),
            dimensions: (0, 0),
            t: 0,
        })
    }

    pub fn dimensions(&self) -> (u16, u16) {
        self.dimensions
    }
}
