pub const MAGIC_NUMBER: &str = "Event Stream";
pub const VERSION: [u8; 3] = [2, 0, 0];

#[repr(u8)]
#[derive(Debug, Clone, Copy)]
pub enum Type {
    Generic = 0,
    Dvs = 1,
    Atis = 2,
    Color = 4,
}

#[repr(C)]
pub struct OwnedGenericEvent {
    pub t: u64,
    pub bytes: Vec<u8>,
}

#[repr(C)]
pub struct GenericEvent<'a> {
    pub t: u64,
    pub bytes: &'a [u8],
}

#[repr(C, packed)]
#[derive(Debug, Copy, Clone)]
pub struct ColorEvent {
    pub t: u64,
    pub x: u16,
    pub y: u16,
    pub r: u8,
    pub g: u8,
    pub b: u8,
}
