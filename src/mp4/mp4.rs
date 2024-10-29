use crate::mp4::x264;

// chosen to be divisible by "common" framerates (24, 25, 29.97, 30, 60, 120, 240, 480)
const TIMESCALE: u32 = 2u32.pow(5) * 3u32.pow(4) * 5u32.pow(2) * 37;

const COLORSPACE: u8 = x264::Colorspace::I420 as u8;

pub struct Encoder<Writer: std::io::Write + std::io::Seek> {
    picture: x264::Picture<COLORSPACE>,
    x264_encoder: x264::Encoder<COLORSPACE>,
    writer: mp4::Mp4Writer<Writer>,
    initial_dts: Option<i64>,
    frame_duration_in_timescale: u32,
}

#[derive(thiserror::Error, Debug)]
pub enum Error {
    #[error(transparent)]
    X264Error(#[from] x264::Error),

    #[error(transparent)]
    HeadersError(#[from] x264::HeadersError),

    #[error(transparent)]
    HandleFrameError(#[from] x264::EncodeError<mp4::Error>),

    #[error(transparent)]
    Mp4Error(#[from] mp4::Error),

    #[error(transparent)]
    Io(#[from] std::io::Error),
}

impl<W: std::io::Write + std::io::Seek> Encoder<W> {
    pub fn from_parameters_and_writer(
        parameters: x264::Parameters<COLORSPACE>,
        writer: W,
    ) -> Result<Self, Error> {
        let picture = parameters.picture();
        let x264_encoder = parameters.encoder()?;
        let mut encoder = Self {
            picture,
            x264_encoder,
            writer: mp4::Mp4Writer::write_start(
                writer,
                &mp4::Mp4Config {
                    major_brand: [b'i', b's', b'o', b'm'].into(),
                    minor_version: 512,
                    compatible_brands: vec![
                        [b'i', b's', b'o', b'm'].into(),
                        [b'i', b's', b'o', b'2'].into(),
                        [b'a', b'v', b'c', b'1'].into(),
                        [b'm', b'p', b'4', b'1'].into(),
                    ],
                    timescale: TIMESCALE,
                },
            )?,
            initial_dts: None,
            frame_duration_in_timescale: (TIMESCALE as f64 / parameters.frame_rate).round()
                as u32,
        };
        {
            let headers = encoder.x264_encoder.headers()?;
            encoder.writer.add_track(&mp4::TrackConfig {
                track_type: mp4::TrackType::Video,
                timescale: TIMESCALE,
                language: "und".to_owned(),
                media_conf: mp4::MediaConfig::AvcConfig(mp4::AvcConfig {
                    width: parameters.width,
                    height: parameters.height,
                    seq_param_set: headers.sps[4..].to_vec(),
                    pic_param_set: headers.pps[4..].to_vec(),
                }),
            })?;
        }
        Ok(encoder)
    }

    fn encode_picture(&mut self) -> Result<(), Error> {
        self.x264_encoder.encode(
            &mut self.picture,
            |payload: &[u8], picture: &x264::x264_picture_t| -> Result<(), mp4::Error> {
                let dts_offset = match self.initial_dts {
                    Some(initial_dts) => initial_dts,
                    None => {
                        self.initial_dts = Some(picture.i_dts);
                        picture.i_dts
                    }
                };
                self.writer.write_sample(
                    1,
                    &mp4::Mp4Sample {
                        start_time: ((picture.i_dts - dts_offset) as u64)
                            * self.frame_duration_in_timescale as u64,
                        duration: self.frame_duration_in_timescale,
                        rendering_offset: ((picture.i_pts - picture.i_dts) as i32) * self.frame_duration_in_timescale as i32,
                        is_sync: picture.b_keyframe > 0,
                        bytes: mp4::Bytes::copy_from_slice(payload),
                    },
                )
            },
        )?;
        Ok(())
    }

    pub fn push_rgba(&mut self, rgba_frame: x264::RGBAFrame) -> Result<(), Error> {
        self.picture.copy_from_rgba(rgba_frame)?;
        self.encode_picture()
    }

    pub fn push_rgb(&mut self, rgb_frame: x264::RGBFrame) -> Result<(), Error> {
        self.picture.copy_from_rgb(rgb_frame)?;
        self.encode_picture()
    }

    pub fn finalize(&mut self) -> Result<(), Error> {
        self.x264_encoder.finalize(
            |payload: &[u8], picture: &x264::x264_picture_t| -> Result<(), mp4::Error> {
                let dts_offset = match self.initial_dts {
                    Some(initial_dts) => initial_dts,
                    None => {
                        self.initial_dts = Some(picture.i_dts);
                        picture.i_dts
                    }
                };
                self.writer.write_sample(
                    1,
                    &mp4::Mp4Sample {
                        start_time: ((picture.i_dts - dts_offset) as u64)
                            * self.frame_duration_in_timescale as u64,
                        duration: self.frame_duration_in_timescale,
                        rendering_offset: (picture.i_pts - picture.i_dts) as i32,
                        is_sync: picture.b_keyframe > 0,
                        bytes: mp4::Bytes::copy_from_slice(payload),
                    },
                )
            },
        )?;
        self.writer.write_end()?;
        Ok(())
    }
}

impl Encoder<std::io::BufWriter<std::fs::File>> {
    pub fn from_parameters_and_path<P: AsRef<std::path::Path>>(
        parameters: x264::Parameters<COLORSPACE>,
        path: P,
    ) -> Result<Self, Error> {
        Self::from_parameters_and_writer(
            parameters,
            std::io::BufWriter::new(std::fs::File::create(path)?),
        )
    }
}
