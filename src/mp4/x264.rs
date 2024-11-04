#![allow(non_upper_case_globals)]
#![allow(non_camel_case_types)]
#![allow(non_snake_case)]
#![allow(dead_code)]
include!(concat!(env!("OUT_DIR"), "/x264_bindings.rs"));

pub struct RGBA(u32);

const TIMEBASE_NUMERATOR: u32 = 1_000_000;

#[derive(thiserror::Error, Debug)]
pub enum RGBAParseError {
    #[error("RGBA length parse error (expected 9, got {0})")]
    Length(usize),

    #[error("RGBA header parse error (expected '#', got '{0}')")]
    Header(char),

    #[error("RGBA parse error for channel {channel} (\"{got}\")")]
    Value { channel: char, got: String },
}

impl RGBA {
    pub const fn from_values(r: u8, g: u8, b: u8, a: u8) -> Self {
        Self(u32::from_le_bytes([r, g, b, a]))
    }

    pub fn from_hex_string(hex_string: &str) -> Result<Self, RGBAParseError> {
        if hex_string.len() != 7 {
            return Err(RGBAParseError::Length(hex_string.len()));
        }
        Ok(RGBA(u32::from_le_bytes([
            u8::from_str_radix(&hex_string[1..3], 16).map_err(|_| RGBAParseError::Value {
                channel: 'R',
                got: hex_string[1..3].to_owned(),
            })?,
            u8::from_str_radix(&hex_string[3..5], 16).map_err(|_| RGBAParseError::Value {
                channel: 'G',
                got: hex_string[3..5].to_owned(),
            })?,
            u8::from_str_radix(&hex_string[5..7], 16).map_err(|_| RGBAParseError::Value {
                channel: 'B',
                got: hex_string[5..7].to_owned(),
            })?,
            u8::from_str_radix(&hex_string[7..9], 16).map_err(|_| RGBAParseError::Value {
                channel: 'A',
                got: hex_string[7..9].to_owned(),
            })?,
        ])))
    }
}

#[derive(Clone, Copy)]
pub struct RGBAFrame<'a> {
    width: u16,
    height: u16,
    pixels: &'a [u32],
}

impl<'a> RGBAFrame<'a> {
    pub fn pixels(&self) -> &[u32] {
        self.pixels
    }

    pub fn width(&self) -> u16 {
        self.width
    }

    pub fn height(&self) -> u16 {
        self.height
    }

    pub fn new(width: u16, height: u16, pixels: &'a [u32]) -> Result<Self, Error> {
        if width as usize * height as usize != pixels.len() {
            Err(Error::RgbaLengthMismatch {
                width,
                height,
                slice_length: pixels.len(),
            })
        } else {
            Ok(Self {
                width,
                height,
                pixels,
            })
        }
    }
}

#[derive(Clone, Copy)]
pub struct RGBFrame<'a> {
    width: u16,
    height: u16,
    pixels: &'a [u8],
}

impl<'a> RGBFrame<'a> {
    pub fn pixels(&self) -> &[u8] {
        self.pixels
    }

    pub fn width(&self) -> u16 {
        self.width
    }

    pub fn height(&self) -> u16 {
        self.height
    }

    pub fn new(width: u16, height: u16, pixels: &'a [u8]) -> Result<Self, Error> {
        if width as usize * height as usize * 3 != pixels.len() {
            Err(Error::RgbLengthMismatch {
                width,
                height,
                slice_length: pixels.len(),
            })
        } else {
            Ok(Self {
                width,
                height,
                pixels,
            })
        }
    }
}

#[derive(Debug, Clone, Copy)]
pub enum Preset {
    Ultrafast,
    Superfast,
    Veryfast,
    Faster,
    Fast,
    Medium,
    Slow,
    Slower,
    Veryslow,
    Placebo,
    None,
}

impl Preset {
    pub fn from_string(string: &str) -> Result<Self, Error> {
        match string {
            "ultrafast" => Ok(Self::Ultrafast),
            "superfast" => Ok(Self::Superfast),
            "veryfast" => Ok(Self::Veryfast),
            "faster" => Ok(Self::Faster),
            "fast" => Ok(Self::Fast),
            "medium" => Ok(Self::Medium),
            "slow" => Ok(Self::Slow),
            "slower" => Ok(Self::Slower),
            "veryslow" => Ok(Self::Veryslow),
            "placebo" => Ok(Self::Placebo),
            "none" => Ok(Self::None),
            string => Err(Error::UnknownPreset(string.to_owned())),
        }
    }

    fn char_ptr(&self) -> *const std::os::raw::c_char {
        match self {
            Preset::Ultrafast => "ultrafast\0".as_ptr() as *const std::os::raw::c_char,
            Preset::Superfast => "superfast\0".as_ptr() as *const std::os::raw::c_char,
            Preset::Veryfast => "veryfast\0".as_ptr() as *const std::os::raw::c_char,
            Preset::Faster => "faster\0".as_ptr() as *const std::os::raw::c_char,
            Preset::Fast => "fast\0".as_ptr() as *const std::os::raw::c_char,
            Preset::Medium => "medium\0".as_ptr() as *const std::os::raw::c_char,
            Preset::Slow => "slow\0".as_ptr() as *const std::os::raw::c_char,
            Preset::Slower => "slower\0".as_ptr() as *const std::os::raw::c_char,
            Preset::Veryslow => "veryslow\0".as_ptr() as *const std::os::raw::c_char,
            Preset::Placebo => "placebo\0".as_ptr() as *const std::os::raw::c_char,
            Preset::None => std::ptr::null(),
        }
    }
}

#[derive(Debug, Clone, Copy)]
pub enum Tune {
    Film,
    Animation,
    Grain,
    Stillimage,
    Psnr,
    Ssim,
    Fastdecode,
    Zerolatency,
    None,
}

impl Tune {
    pub fn from_string(string: &str) -> Result<Self, Error> {
        match string {
            "film" => Ok(Self::Film),
            "animation" => Ok(Self::Animation),
            "grain" => Ok(Self::Grain),
            "stillimage" => Ok(Self::Stillimage),
            "psnr" => Ok(Self::Psnr),
            "ssim" => Ok(Self::Ssim),
            "fastdecode" => Ok(Self::Fastdecode),
            "zerolatency" => Ok(Self::Zerolatency),
            "none" => Ok(Self::None),
            string => Err(Error::UnknownTune(string.to_owned())),
        }
    }

    fn char_ptr(&self) -> *const std::os::raw::c_char {
        match self {
            Tune::Film => "film\0".as_ptr() as *const std::os::raw::c_char,
            Tune::Animation => "animaation\0".as_ptr() as *const std::os::raw::c_char,
            Tune::Grain => "grain\0".as_ptr() as *const std::os::raw::c_char,
            Tune::Stillimage => "stillimage\0".as_ptr() as *const std::os::raw::c_char,
            Tune::Psnr => "psnr\0".as_ptr() as *const std::os::raw::c_char,
            Tune::Ssim => "ssim\0".as_ptr() as *const std::os::raw::c_char,
            Tune::Fastdecode => "fastdecode\0".as_ptr() as *const std::os::raw::c_char,
            Tune::Zerolatency => "zerolatency\0".as_ptr() as *const std::os::raw::c_char,
            Tune::None => std::ptr::null(),
        }
    }
}

#[derive(Debug, Clone, Copy)]
pub enum Profile {
    Baseline,
    Main,
    High,
    High10,
    High422,
    High444,
}

impl Profile {
    pub fn from_string(string: &str) -> Result<Self, Error> {
        match string {
            "baseline" => Ok(Self::Baseline),
            "main" => Ok(Self::Main),
            "high" => Ok(Self::High),
            "high10" => Ok(Self::High10),
            "high422" => Ok(Self::High422),
            "high444" => Ok(Self::High444),
            string => Err(Error::UnknownProfile(string.to_owned())),
        }
    }

    fn char_ptr(&self) -> *const std::os::raw::c_char {
        (match self {
            Profile::Baseline => "baseline\0",
            Profile::Main => "main\0",
            Profile::High => "high\0",
            Profile::High10 => "high10\0",
            Profile::High422 => "high422\0",
            Profile::High444 => "high444\0",
        })
        .as_ptr() as *const std::os::raw::c_char
    }
}

#[derive(Debug, Clone, Copy)]
#[repr(u8)]
pub enum Colorspace {
    I400 = 0x01, // monochrome 4:0:0
    I420 = 0x02, // yuv 4:2:0 planar
    YV12 = 0x03, // yvu 4:2:0 planar
    NV12 = 0x04, // yuv 4:2:0, with one y plane and one packed u+v
    NV21 = 0x05, // yuv 4:2:0, with one y plane and one packed v+u
    I422 = 0x06, // yuv 4:2:2 planar
    YV16 = 0x07, // yvu 4:2:2 planar
    NV16 = 0x08, // yuv 4:2:2, with one y plane and one packed u+v
    YUYV = 0x09, // yuyv 4:2:2 packed
    UYVY = 0x0a, // uyvy 4:2:2 packed
    V210 = 0x0b, // 10-bit yuv 4:2:2 packed in 32
    I444 = 0x0c, // yuv 4:4:4 planar
    YV24 = 0x0d, // yvu 4:4:4 planar
    BGR = 0x0e,  // packed bgr 24bits
    BGRA = 0x0f, // packed bgr 32bits
    RGB = 0x10,  // packed rgb 24bits
}

impl Colorspace {
    const fn value(&self) -> u8 {
        *self as u8
    }
}

pub struct Parameters<const Colorspace: u8> {
    pub preset: Preset,
    pub tune: Tune,
    pub profile: Profile,
    pub crf: f32,
    pub width: u16,
    pub height: u16,
    pub frame_rate: f64,
    pub use_opencl: bool,
}

#[derive(thiserror::Error, Debug)]
pub enum Error {
    #[error("initialization failed with {preset:?} and {tune:?}")]
    PresetAndTune { preset: Preset, tune: Tune },

    #[error("initialization failed with {0:?}")]
    Profile(Profile),

    #[error("creating the encoder failed")]
    Encoder,

    #[error("the crf must be in the range [0, 51] (got {0})")]
    Crf(f32),

    #[error("unknown preset \"{0}\"")]
    UnknownPreset(String),

    #[error("unknown tune \"{0}\"")]
    UnknownTune(String),

    #[error("unknown profile \"{0}\"")]
    UnknownProfile(String),

    #[error("size mismatch between the encoder ({encoder_width} x {encoder_height}) and the RGBA frame ({frame_width} x {frame_height})")]
    DimensionsMismatch {
        encoder_width: u16,
        encoder_height: u16,
        frame_width: u16,
        frame_height: u16,
    },

    #[error(
        "incompatible RGBA dimensions ({width} x {height}) and u32 slice length ({slice_length})"
    )]
    RgbaLengthMismatch {
        width: u16,
        height: u16,
        slice_length: usize,
    },

    #[error(
        "incompatible RGB dimensions ({width} x {height}) and u8 slice length ({slice_length})"
    )]
    RgbLengthMismatch {
        width: u16,
        height: u16,
        slice_length: usize,
    },
}

pub struct Picture<const Colorspace: u8> {
    inner: x264_picture_t,
    width: u16,
    height: u16,
}

unsafe impl<const Colorspace: u8> Send for Picture<{ Colorspace }> {}

impl Picture<{ Colorspace::I420.value() }> {
    fn copy_from_y(&mut self, y_as_i32: &[i32]) {
        for (index, rgba) in y_as_i32.iter().enumerate() {
            unsafe {
                *self.inner.img.plane[1].add(index) = (((-43 * (rgba & 0xff)
                    - 84 * ((rgba & 0xff00) >> 8)
                    + 127 * ((rgba & 0xff0000) >> 16)
                    + 128)
                    >> 8)
                    + 128)
                    .clamp(0, 255) as u8;
            }
        }
        for (index, rgba) in y_as_i32.iter().enumerate() {
            unsafe {
                *self.inner.img.plane[2].add(index) = (((127 * (rgba & 0xff)
                    - 106 * ((rgba & 0xff00) >> 8)
                    - 21 * ((rgba & 0xff0000) >> 16)
                    + 128)
                    >> 8)
                    + 128)
                    .clamp(0, 255) as u8;
            }
        }
    }

    pub fn copy_from_rgb(&mut self, rgb_frame: RGBFrame) -> Result<(), Error> {
        // This function has *not* been tested on non-little-endian platforms.
        if self.inner.img.i_csp != X264_CSP_I420 as i32 {
            panic!("colorspace mismatch");
        }
        if self.width != rgb_frame.width || self.height != rgb_frame.height {
            return Err(Error::DimensionsMismatch {
                encoder_width: self.width,
                encoder_height: self.height,
                frame_width: rgb_frame.width,
                frame_height: rgb_frame.height,
            });
        }
        {
            let y_as_i32 = unsafe {
                std::slice::from_raw_parts_mut(
                    self.inner.img.plane[0] as *mut i32,
                    (self.width / 2) as usize * (self.height / 2) as usize,
                )
            };
            for (downsampled_y, y) in (0..self.height).step_by(2).enumerate() {
                for (downsampled_x, x) in (0..self.width).step_by(2).enumerate() {
                    y_as_i32[downsampled_x + downsampled_y * (self.width as usize / 2)] =
                        i32::from_le_bytes([
                            rgb_frame.pixels[(x as usize + y as usize * self.width as usize) * 3],
                            rgb_frame.pixels
                                [(x as usize + y as usize * self.width as usize) * 3 + 1],
                            rgb_frame.pixels
                                [(x as usize + y as usize * self.width as usize) * 3 + 2],
                            0x00,
                        ]);
                }
            }
            self.copy_from_y(y_as_i32);
        }
        for (index, rgb) in rgb_frame.pixels.chunks(3).enumerate() {
            unsafe {
                *self.inner.img.plane[0].add(index) =
                    ((77 * (rgb[0] as u32) + 150 * (rgb[1] as u32) + 29 * (rgb[2] as u32) + 128)
                        >> 8)
                        .clamp(0, 255) as u8;
            }
        }
        Ok(())
    }

    pub fn copy_from_rgba(&mut self, rgba_frame: RGBAFrame) -> Result<(), Error> {
        // This function has *not* been tested on non-little-endian platforms.
        if self.inner.img.i_csp != X264_CSP_I420 as i32 {
            panic!("colorspace mismatch");
        }
        if self.width != rgba_frame.width || self.height != rgba_frame.height {
            return Err(Error::DimensionsMismatch {
                encoder_width: self.width,
                encoder_height: self.height,
                frame_width: rgba_frame.width,
                frame_height: rgba_frame.height,
            });
        }
        // Use the Y (of the YCrCb image) as a temporary buffer to store the chroma downsampled image
        // this *should* make SIMD optimizations more likely since the following conversions
        // (RGB to Cr and RGB to Cb) operate on contiguous memory.
        // The type is i32 instead of u32 to avoid a conversion in the signed conversion operation.
        {
            let y_as_i32 = unsafe {
                std::slice::from_raw_parts_mut(
                    self.inner.img.plane[0] as *mut i32,
                    (self.width / 2) as usize * (self.height / 2) as usize,
                )
            };
            for (downsampled_y, y) in (0..self.height).step_by(2).enumerate() {
                for (downsampled_x, x) in (0..self.width).step_by(2).enumerate() {
                    y_as_i32[downsampled_x + downsampled_y * (self.width as usize / 2)] =
                        ((rgba_frame.pixels[x as usize + y as usize * self.width as usize])
                            & 0xffffff) as i32;
                }
            }
            self.copy_from_y(y_as_i32);
        }
        for (index, rgba) in rgba_frame.pixels.iter().enumerate() {
            unsafe {
                *self.inner.img.plane[0].add(index) = ((77 * (rgba & 0xff)
                    + 150 * ((rgba & 0xff00) >> 8)
                    + 29 * ((rgba & 0xff0000) >> 16)
                    + 128)
                    >> 8)
                    .clamp(0, 255) as u8;
            }
        }
        Ok(())
    }
}

impl<const Colorspace: u8> Drop for Picture<{ Colorspace }> {
    fn drop(&mut self) {
        unsafe { x264_picture_clean(&mut self.inner as *mut x264_picture_t) };
    }
}

#[derive(Debug)]
pub struct Encoder<const Colorspace: u8> {
    inner: *mut x264_t,
    picture_buffer: std::mem::MaybeUninit<x264_picture_t>,
    poisoned: bool,
    picture_index: i64,
}

unsafe impl<const Colorspace: u8> Send for Encoder<{ Colorspace }> {}

impl<const Colorspace: u8> Drop for Encoder<{ Colorspace }> {
    fn drop(&mut self) {
        unsafe { x264_encoder_close(self.inner) };
    }
}

impl<const Colorspace: u8> Parameters<{ Colorspace }> {
    pub fn picture(&self) -> Picture<{ Colorspace }> {
        let inner = {
            let mut uninit_picture: std::mem::MaybeUninit<x264_picture_t> =
                std::mem::MaybeUninit::uninit();
            unsafe {
                if x264_picture_alloc(
                    uninit_picture.as_mut_ptr(),
                    Colorspace as i32,
                    self.width as ::std::os::raw::c_int,
                    self.height as ::std::os::raw::c_int,
                ) < 0
                {
                    panic!("allocating picture failed");
                }
                uninit_picture.assume_init()
            }
        };
        Picture {
            inner,
            width: self.width,
            height: self.height,
        }
    }

    pub fn encoder(&self) -> Result<Encoder<{ Colorspace }>, Error> {
        if self.crf < 0.0 || self.crf > 51.0 {
            return Err(Error::Crf(self.crf));
        }
        let mut uninit_parameters: std::mem::MaybeUninit<x264_param_t> =
            std::mem::MaybeUninit::uninit();
        let mut parameters = unsafe {
            if x264_param_default_preset(
                uninit_parameters.as_mut_ptr(),
                self.preset.char_ptr(),
                self.tune.char_ptr(),
            ) < 0
            {
                return Err(Error::PresetAndTune {
                    preset: self.preset,
                    tune: self.tune,
                });
            }
            if x264_param_apply_profile(uninit_parameters.as_mut_ptr(), self.profile.char_ptr()) < 0
            {
                return Err(Error::Profile(self.profile));
            }
            uninit_parameters.assume_init()
        };
        parameters.i_width = self.width as ::std::os::raw::c_int;
        parameters.i_height = self.height as ::std::os::raw::c_int;
        parameters.i_csp = Colorspace as i32;
        parameters.i_bitdepth = 8;
        parameters.rc.i_rc_method = X264_RC_CRF as i32;
        parameters.rc.f_rf_constant = self.crf;
        parameters.b_opencl = self.use_opencl as ::std::os::raw::c_int;
        parameters.i_log_level = X264_LOG_NONE;
        parameters.b_vfr_input = 0;
        if self.frame_rate.round() == self.frame_rate {
            parameters.i_timebase_num = 1;
            parameters.i_timebase_den = self.frame_rate.round() as u32;
        } else {
            parameters.i_timebase_num = TIMEBASE_NUMERATOR;
            parameters.i_timebase_den =
                (self.frame_rate * TIMEBASE_NUMERATOR as f64).round() as u32;
        }
        parameters.vui.b_fullrange = 1;
        parameters.b_repeat_headers = 0;
        parameters.b_annexb = 0;
        let inner = unsafe { x264_encoder_open_164(&mut parameters as *mut x264_param_t) };
        if inner.is_null() {
            Err(Error::Encoder)
        } else {
            Ok(Encoder {
                inner,
                picture_buffer: std::mem::MaybeUninit::uninit(),
                poisoned: false,
                picture_index: 0,
            })
        }
    }
}

#[derive(Debug)]
pub struct Headers<'a, const Colorspace: u8> {
    pub sps: &'a [u8],
    pub pps: &'a [u8],
    pub sei: &'a [u8],
    encoder: &'a mut Encoder<{ Colorspace }>,
}

#[derive(thiserror::Error, Debug)]
pub enum HeadersError {
    #[error("wrong number of NALs (expected 3, got {0})")]
    NalUnitsCount(i32),

    #[error("x264_encoder_headers returned {0}")]
    X264(::std::os::raw::c_int),
}

#[derive(thiserror::Error, Debug)]
pub enum EncodeError<HandleFrameError> {
    #[error(transparent)]
    HandleFrameError(HandleFrameError),

    #[error("the encoder is poisoned (a previous call to encode returned an error)")]
    Poisoned,

    #[error("x264_encoder_encode returned {0}")]
    X264(::std::os::raw::c_int),
}

impl<const Colorspace: u8> Encoder<{ Colorspace }> {
    pub fn headers(&mut self) -> Result<Headers<{ Colorspace }>, HeadersError> {
        let mut nal_units: std::mem::MaybeUninit<*mut x264_nal_t> = std::mem::MaybeUninit::uninit();
        let mut nal_units_count: ::std::os::raw::c_int = 0;
        let payload_size = unsafe {
            x264_encoder_headers(
                self.inner,
                nal_units.as_mut_ptr(),
                &mut nal_units_count as *mut ::std::os::raw::c_int,
            )
        };
        if payload_size < 0 {
            return Err(HeadersError::X264(payload_size));
        }
        if nal_units_count != 3 {
            return Err(HeadersError::NalUnitsCount(nal_units_count));
        }
        let nal_units = unsafe { nal_units.assume_init() };
        Ok(Headers {
            sps: unsafe {
                std::slice::from_raw_parts(
                    (*nal_units.add(0)).p_payload,
                    (*nal_units.add(0)).i_payload as usize,
                )
            },
            pps: unsafe {
                std::slice::from_raw_parts(
                    (*nal_units.add(1)).p_payload,
                    (*nal_units.add(1)).i_payload as usize,
                )
            },
            sei: unsafe {
                std::slice::from_raw_parts(
                    (*nal_units.add(2)).p_payload,
                    (*nal_units.add(2)).i_payload as usize,
                )
            },
            encoder: self,
        })
    }

    fn encode_or_finalize<HandleFrame, HandleFrameError>(
        &mut self,
        picture: *mut x264_picture_t,
        mut handle_frame: HandleFrame,
    ) -> Result<(), EncodeError<HandleFrameError>>
    where
        HandleFrame: FnMut(&[u8], &x264_picture_t) -> Result<(), HandleFrameError>,
    {
        let mut nal_units: std::mem::MaybeUninit<*mut x264_nal_t> = std::mem::MaybeUninit::uninit();
        let mut nal_units_count: ::std::os::raw::c_int = 0;
        let payload_size = unsafe {
            x264_encoder_encode(
                self.inner,
                nal_units.as_mut_ptr(),
                &mut nal_units_count as *mut ::std::os::raw::c_int,
                picture,
                self.picture_buffer.as_mut_ptr(),
            )
        };
        if payload_size < 0 {
            self.poisoned = true;
            return Err(EncodeError::X264(payload_size));
        }
        if payload_size > 0 {
            if let Err(error) = handle_frame(
                unsafe {
                    std::slice::from_raw_parts(
                        (*nal_units.assume_init()).p_payload,
                        payload_size as usize,
                    )
                },
                unsafe { &self.picture_buffer.assume_init() },
            ) {
                self.poisoned = true;
                return Err(EncodeError::HandleFrameError(error));
            }
        }
        Ok(())
    }

    pub fn encode<HandleFrame, HandleFrameError>(
        &mut self,
        picture: &mut Picture<{ Colorspace }>,
        handle_frame: HandleFrame,
    ) -> Result<(), EncodeError<HandleFrameError>>
    where
        HandleFrame: FnMut(&[u8], &x264_picture_t) -> Result<(), HandleFrameError>,
    {
        if self.poisoned {
            return Err(EncodeError::Poisoned);
        }
        picture.inner.i_pts = self.picture_index;
        self.picture_index += 1;
        self.encode_or_finalize(&mut picture.inner as *mut x264_picture_t, handle_frame)
    }

    pub fn finalize<HandleFrame, HandleFrameError>(
        &mut self,
        mut handle_frame: HandleFrame,
    ) -> Result<(), EncodeError<HandleFrameError>>
    where
        HandleFrame: FnMut(&[u8], &x264_picture_t) -> Result<(), HandleFrameError>,
    {
        if self.poisoned {
            return Err(EncodeError::Poisoned);
        }
        while unsafe { x264_encoder_delayed_frames(self.inner) } > 0 {
            self.encode_or_finalize(std::ptr::null_mut(), &mut handle_frame)?;
        }
        Ok(())
    }
}
