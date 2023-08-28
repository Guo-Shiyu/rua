use std::{
    cell::Cell,
    collections::{hash_map::DefaultHasher, HashMap},
    fmt::Display,
    hash::{Hash, Hasher},
    ptr::NonNull,
};

use crate::{codegen::Proto, LuaErr};

use super::{state::State, RuntimeErr};

/// # NanBoxing Technique
///
/// for a IEE754 NAN 64 bit floating number, it has a structure like this:
/// ``` text
///
///                     64 bit
///     +-------------------------------------+
///     0111 1111 1111 xxxx xxxx .... xxxx xxxx
///     +------------+ +----------------------+
///         12 bit             52 bit
///
/// in x86 64bit OS, there are only 48 address lines used for addressing
/// so, a pointer can be stored in a nan float, and extra 4 bits can be
/// used to mark the ptr's type:
///
///          head      tag          payload
///     +------------+ +--+ +---------------------+
///     0111 1111 1111 xxxx yyyy yyyy ... yyyy yyyy
///     +------------+ +--+ +---------------------+
///         12 bit     4 bit         48 bit
///
/// ```
///

/// https://github.com/Tencent/rapidjson/pull/546
///

#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub struct TaggedBox {
    repr: u64,
}

pub const PAYLOAD_MASK: u64 = 0x0000_FFFF_FFFF_FFFF;
pub const TAG_MASK: u64 = 0xFFFF_0000_0000_0000;

/// Heap address head mask for 64 bit OS
pub const HEAP_ADDR_HEAD_MASK: u64 = 0xFFFF_0000_0000_0000;

#[cfg(target_os = "linux")]
pub const HEAP_ADDR_HEAD: u64 = 0x7ff0_0000_0000_0000;

#[cfg(target_os = "windows")]
pub const HEAP_ADDR_HEAD: u64 = 0x0000_0000_0000_0000;

pub type Tag = u8;

impl TaggedBox {
    pub fn new(tag: Tag, payload: usize) -> Self {
        let mut repr = u64::MIN;

        // set tag bits
        repr |= (tag as u64) << 48;

        // set payload bits
        let fixed_pl = PAYLOAD_MASK & payload as u64;
        repr |= fixed_pl as u64;

        Self { repr }
    }

    pub fn in_raw(&self) -> u64 {
        u64::from_ne_bytes(self.repr.to_ne_bytes())
    }

    pub fn tag(&self) -> Tag {
        ((self.in_raw() & TAG_MASK) >> 48) as u8
    }

    pub fn payload(&self) -> u64 {
        self.in_raw() & PAYLOAD_MASK
    }

    pub fn set_tag(&mut self, tag: Tag) -> &Self {
        *self = Self::new(tag, self.payload() as usize);
        self
    }

    pub fn set_payload(&mut self, payload: usize) -> &Self {
        *self = Self::new(self.tag(), payload);
        self
    }

    pub fn replace_payload_with(&mut self, payload: usize) -> usize {
        let old = self.payload();
        self.set_payload(payload);
        old as usize
    }

    pub fn replace_tag_with(&mut self, tag: Tag) -> Tag {
        let old = self.tag();
        self.set_tag(tag);
        old
    }

    pub fn as_ptr<T>(&self) -> *const T {
        (self.payload() | HEAP_ADDR_HEAD) as *const T
    }

    pub fn as_mut<T>(&self) -> *mut T {
        (self.payload() | HEAP_ADDR_HEAD) as *mut T
    }

    pub fn from_heap_ptr<T>(tag: Tag, payload: *const T) -> Self {
        Self::new(tag, payload as usize)
    }
}

#[repr(u8)]
#[derive(Clone, Copy)]
pub enum GcColor {
    Wild,         // not mamaged by any vm
    Black,        // reachable
    Gray,         // reachable, but not scanned
    White,        // unreachable  flag 1
    AnotherWhite, // unreachable  flag 2
}

impl Default for GcColor {
    fn default() -> Self {
        Self::Wild
    }
}

pub trait EstimatedSize {
    fn estimate_size(&self) -> usize;
}

pub trait GcObject: EstimatedSize + WithTag {
    fn gch_addr(&self) -> *const GcMark;
    fn mark_reachable(&self);
    fn mark_unreachable(&self);
    fn mark_newborned(&self, white: GcColor) -> &Self;
}

// pub type GcMark = Cell<GcColor>;
pub type GcMark = Cell<GcColor>;

macro_rules! impl_gc_cycle {
    ($type: ty) => {
        impl GcObject for $type {
            fn gch_addr(&self) -> *const GcMark {
                &self.gch as *const GcMark
            }

            fn mark_reachable(&self) {
                self.gch.set(GcColor::White);
            }

            fn mark_unreachable(&self) {
                self.gch.set(GcColor::Black);
            }

            fn mark_newborned(&self, white: GcColor) -> &Self {
                self.gch.set(white);
                self
            }
        }
    };
}

pub enum StrInner {
    Short { ptr: *mut u8, len: u8, capacity: u8 },
    Long { inner: String },
}

pub type StrHashVal = u32;

/// Depends on the fixed layout, so repr as C.
#[repr(C)]
pub struct StrWrapper {
    gch: GcMark,
    extra: bool, // reserved (short)  | hashed (long string)
    hash: StrHashVal,
    inner: StrInner,
}

impl EstimatedSize for StrWrapper {
    fn estimate_size(&self) -> usize {
        let inner_buf_size = match &self.inner {
            StrInner::Short {
                ptr: _,
                len,
                capacity: _,
            } => *len as usize,
            StrInner::Long { inner } => inner.capacity(),
        };
        inner_buf_size + std::mem::size_of::<StrWrapper>()
    }
}

impl_gc_cycle!(StrWrapper);

// impl Drop for StrWrapper {
//     fn drop(&mut self) {
//         match self.inner {
//             StrInner::Short {
//                 ptr: content,
//                 len,
//                 capacity,
//             } => unsafe {
//                 let _ = String::from_raw_parts(content, len as usize, capacity as usize);
//             },
//             StrInner::Long { inner: _ } => {}
//         }
//     }
// }

impl StrWrapper {
    pub const MAX_SHORT_LEN: usize = 40;

    pub fn hashid(&mut self) -> StrHashVal {
        match &self.inner {
            StrInner::Short {
                ptr: _,
                len: _,
                capacity: _,
            } => self.hash,
            StrInner::Long { inner } => {
                if !self.extra {
                    self.hash = Self::hash_long(inner);
                    self.extra = true;
                }

                self.hash
            }
        }
    }

    pub fn hash_long(s: &str) -> StrHashVal {
        let mut hasher = DefaultHasher::new();
        s.chars()
            .step_by(s.len() >> 5)
            .for_each(|c| c.hash(&mut hasher));
        hasher.finish() as StrHashVal
    }

    pub fn hash_short(ss: &str) -> StrHashVal {
        let mut hasher = DefaultHasher::new();
        ss.chars().for_each(|c| c.hash(&mut hasher));
        hasher.finish() as StrHashVal
    }

    pub fn len(&self) -> usize {
        match &self.inner {
            StrInner::Short {
                ptr: _,
                len,
                capacity: _,
            } => *len as usize,
            StrInner::Long { inner } => inner.len(),
        }
    }

    pub fn as_str(&self) -> &str {
        match &self.inner {
            StrInner::Short {
                ptr: content,
                len,
                capacity: _,
            } => unsafe {
                std::str::from_utf8_unchecked(std::slice::from_raw_parts(*content, *len as usize))
            },
            StrInner::Long { inner } => inner.as_str(),
        }
    }

    pub fn from_short(mut str: String, hash: Option<StrHashVal>, reserve: bool) -> Self {
        debug_assert!(str.len() <= Self::MAX_SHORT_LEN);
        let hash = hash.unwrap_or_else(|| Self::hash_short(&str));
        let (ptr, len, cap) = (str.as_mut_ptr(), str.len(), str.capacity());
        std::mem::forget(str);

        Self {
            gch: GcMark::default(),
            extra: reserve,
            hash,
            inner: StrInner::Short {
                ptr,
                len: len as u8,      // len < Self::MaxShortLen
                capacity: cap as u8, // capaciry has shrinked
            },
        }
    }

    pub fn from_long(mut s: String) -> Self {
        if s.capacity() > 2 * s.len() {
            s.shrink_to_fit();
        }
        Self {
            gch: GcMark::default(),
            extra: false,
            hash: 0,
            inner: StrInner::Long { inner: s },
        }
    }

    pub fn free_short(&mut self) {
        match self.inner {
            StrInner::Short {
                ptr: mut content,
                mut len,
                mut capacity,
            } => unsafe {
                let _ = String::from_raw_parts(content, len as usize, capacity as usize);
                content = std::ptr::null_mut();
                len = 0;
                capacity = 0;
            },
            StrInner::Long { inner: _ } => {}
        }
    }

    pub fn is_long(&self) -> bool {
        match self.inner {
            StrInner::Short { .. } => false,
            StrInner::Long { .. } => true,
        }
    }

    pub fn is_short(&self) -> bool {
        !self.is_long()
    }

    pub fn inner(&self) -> &StrInner {
        &self.inner
    }

    pub fn is_reserved(&self) -> bool {
        if let StrInner::Short {
            ptr: _,
            len: _,
            capacity: _,
        } = self.inner
        {
            self.extra
        } else {
            false
        }
    }
}

#[repr(C)]
pub struct Table {
    gch: GcMark,
    hashpart: HashMap<usize, LValue>,

    // TODO:
    // replace arrary part to binary heap
    array: Vec<LValue>,
}

impl Default for Table {
    fn default() -> Self {
        Self {
            gch: GcMark::default(),
            hashpart: HashMap::new(),
            array: Vec::new(),
        }
    }
}

impl EstimatedSize for Table {
    fn estimate_size(&self) -> usize {
        // TODO:
        // implement this

        std::mem::size_of::<Table>()
    }
}

impl_gc_cycle!(Table);

impl Table {
    pub fn set(&mut self, k: &LValue, v: LValue) -> Result<(), RuntimeErr> {
        let unique = Self::as_key(k);
        if unique == 0 {
            return Err(RuntimeErr::TableIndexIsNil);
        }
        self.hashpart.insert(unique, v);
        Ok(())
    }

    pub fn find_by_key(&self, k: &LValue) -> Result<LValue, RuntimeErr> {
        let unique = Self::as_key(k);
        if unique == 0 {
            return Err(RuntimeErr::TableIndexIsNil);
        }
        Ok(self.hashpart.get(&unique).cloned().unwrap_or_default())
    }

    fn as_key(k: &LValue) -> usize {
        match k.clone() {
            LValue::Nil => 0,
            LValue::Bool(b) => b as usize,
            LValue::Int(i) => i as usize,
            LValue::Float(f) => {
                let u = u64::from_ne_bytes(f.to_ne_bytes());
                u as usize
            }
            LValue::String(ref mut s) => unsafe { s.as_mut().hashid() as usize },
            LValue::Table(t) => unsafe { t.as_ref().gch_addr() as usize },
            LValue::Function(f) => unsafe { f.as_ref().gch_addr() as usize },
            LValue::RsFn(rf) => rf as usize,
            LValue::UserData(ud) => unsafe { ud.as_ref().gch_addr() as usize },
        }
    }
}

/// a flexible array
#[repr(C)]
pub struct UserData {
    gch: GcMark,
    len: usize,
}
impl EstimatedSize for UserData {
    fn estimate_size(&self) -> usize {
        self.len
    }
}

impl_gc_cycle!(UserData);

// usize, stack size after call
pub type RsFunc = fn(&mut State) -> Result<usize, RuntimeErr>;

// mark gc objects id
pub trait WithTag {
    fn tagid() -> Tag;
}

macro_rules! tagid_declare {
    ($s:ty, $id:expr) => {
        impl WithTag for $s {
            fn tagid() -> Tag {
                $id
            }
        }
    };
}

#[derive(Clone, Copy, PartialEq)]
pub enum LValue {
    Nil,

    Bool(bool),
    Int(i64),
    Float(f64),

    String(NonNull<StrWrapper>), // lua immutable string
    Table(NonNull<Table>),       // lua table
    Function(NonNull<Proto>),    // lua function
    RsFn(RsFunc),                // light rs function
    UserData(NonNull<UserData>), // light rs user data, managed by lua
}

tagid_declare!(StrWrapper, 4);
tagid_declare!(Table, 5);
tagid_declare!(Proto, 6);
tagid_declare!(UserData, 7);

// impl WithTag for LValue {
//     fn tagid(&self) -> Tag {
//         use LValue::*;
//         match self {
//             Nil => 0,
//             Bool(_) => 1,
//             Int(_) => 2,
//             Float(_) => 3,
//             String(s) => unsafe { s.as_ref().tagid() },
//             Table(t) => unsafe { t.as_ref().tagid() },
//             Function(f) => unsafe { f.as_ref().tagid() },
//             RsFn(rf) => unsafe { rf.as_ref().tagid() },
//             UserData { .. } => 8,
//         }
//     }
// }

impl Default for LValue {
    fn default() -> Self {
        LValue::Nil
    }
}

impl From<TaggedBox> for LValue {
    fn from(value: TaggedBox) -> Self {
        match value.tag() {
            4 => LValue::String(unsafe { NonNull::new_unchecked(value.as_mut::<StrWrapper>()) }),
            5 => LValue::Table(unsafe { NonNull::new_unchecked(value.as_mut::<Table>()) }),
            6 => LValue::Function(unsafe { NonNull::new_unchecked(value.as_mut::<Proto>()) }),
            7 => LValue::RsFn(unsafe { std::mem::transmute(value.payload()) }),
            _ => unreachable!(),
        }
    }
}

impl From<Proto> for LValue {
    fn from(value: Proto) -> Self {
        LValue::Function(unsafe { NonNull::new_unchecked(Box::into_raw(Box::new(value))) })
    }
}

impl From<String> for LValue {
    fn from(value: String) -> Self {
        let len = value.len();
        if len <= StrWrapper::MAX_SHORT_LEN {
            LValue::String(unsafe {
                NonNull::new_unchecked(Box::into_raw(Box::new(StrWrapper::from_short(
                    value, None, false,
                ))))
            })
        } else {
            LValue::String(unsafe {
                NonNull::new_unchecked(Box::into_raw(Box::new(StrWrapper::from_long(value))))
            })
        }
    }
}

impl EstimatedSize for LValue {
    fn estimate_size(&self) -> usize {
        match self {
            // thry are not allocated on heap
            LValue::Nil | LValue::Bool(_) | LValue::Int(_) | LValue::Float(_) | LValue::RsFn(_) => {
                0
            }

            LValue::String(sw) => unsafe { sw.as_ref().estimate_size() },
            LValue::Table(tb) => unsafe { tb.as_ref().estimate_size() },
            LValue::Function(p) => unsafe { p.as_ref().estimate_size() },
            LValue::UserData(ud) => unsafe { ud.as_ref().estimate_size() },
        }
    }
}

// impl GcObject for LValue {
//     fn gch_addr(&self) -> *const GcMark {
//         match self {
//             LValue::String(s) => unsafe { s.as_ref().gch_addr() },
//             LValue::Table(t) => unsafe { t.as_ref().gch_addr() },
//             LValue::Function(f) => unsafe { f.as_ref().gch_addr() },
//             LValue::UserData { .. } => todo!(),
//             _ => std::ptr::null(),
//         }
//     }

//     fn mark_reachable(&self) {
//         match self {
//             LValue::String(s) => unsafe {
//                 s.as_ref().mark_reachable();
//             },
//             LValue::Table(t) => unsafe {
//                 t.as_ref().mark_reachable();
//             },
//             LValue::Function(f) => unsafe {
//                 f.as_ref().mark_reachable();
//             },
//             _ => {}
//         };
//     }

//     fn mark_unreachable(&self) {
//         match self {
//             LValue::String(s) => unsafe {
//                 s.as_ref().mark_unreachable();
//             },
//             LValue::Table(t) => unsafe {
//                 t.as_ref().mark_unreachable();
//             },
//             LValue::Function(f) => unsafe {
//                 f.as_ref().mark_unreachable();
//             },
//             _ => {}
//         };
//     }

//     fn mark_newborned(&self, white: GcColor) -> &Self {
//         let _ = match self {
//             LValue::String(s) => unsafe {
//                 s.as_ref().mark_newborned(white);
//             },
//             LValue::Table(t) => unsafe {
//                 t.as_ref().mark_newborned(white);
//             },
//             LValue::Function(f) => unsafe {
//                 f.as_ref().mark_newborned(white);
//             },
//             _ => {}
//         };
//         self
//     }
// }

impl Display for LValue {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        use LValue::*;
        match self {
            Nil => write!(f, "nil"),
            Bool(b) => write!(f, "{}", b),
            Int(i) => write!(f, "{}", i),
            Float(fl) => write!(f, "{}", fl),
            String(s) => write!(f, "{}", unsafe { s.as_ref().as_str() }),
            Table(t) => write!(f, "table: 0x{:x}", unsafe {
                t.as_ref().gch_addr() as usize
            }),
            Function(func) => write!(f, "function: 0x{:x}", unsafe {
                func.as_ref().gch_addr() as usize
            }),
            RsFn(rsf) => write!(f, "rsfn: 0x{:x}", rsf as *const _ as usize),
            UserData(ud) => write!(f, "userdata: 0x{:x}", unsafe {
                ud.as_ref().gch_addr() as usize
            }),
        }
    }
}

impl LValue {
    /// Check if the value is managed by lua gc
    pub fn is_managed(&self) -> bool {
        use LValue::*;
        match self {
            Nil | Bool(_) | Int(_) | Float(_) => false,
            String(_) | Table(_) | Function(_) | RsFn(_) | UserData { .. } => true,
        }
    }

    pub fn is_nil(&self) -> bool {
        matches!(self, LValue::Nil)
    }

    pub fn is_bool(&self) -> bool {
        matches!(self, LValue::Bool(_))
    }

    pub fn is_int(&self) -> bool {
        matches!(self, LValue::Int(_))
    }

    pub fn is_float(&self) -> bool {
        matches!(self, LValue::Float(_))
    }

    pub fn is_string(&self) -> bool {
        matches!(self, LValue::String(_))
    }

    pub fn is_table(&self) -> bool {
        matches!(self, LValue::Table(_))
    }

    pub fn is_luafn(&self) -> bool {
        matches!(self, LValue::Function(_))
    }

    pub fn is_rsfn(&self) -> bool {
        matches!(self, LValue::RsFn(_))
    }

    pub fn is_userdata(&self) -> bool {
        matches!(self, LValue::UserData { .. })
    }

    pub fn as_bool(&self) -> Option<bool> {
        match self {
            LValue::Bool(b) => Some(*b),
            _ => None,
        }
    }

    pub fn as_int(&self) -> Option<i64> {
        match self {
            LValue::Int(i) => Some(*i),
            _ => None,
        }
    }

    pub fn as_float(&self) -> Option<f64> {
        match self {
            LValue::Float(f) => Some(*f),
            _ => None,
        }
    }

    pub fn as_string(&self) -> Option<&StrWrapper> {
        match self {
            LValue::String(s) => Some(unsafe { s.as_ref() }),
            _ => None,
        }
    }

    pub fn as_table(&self) -> Option<&Table> {
        match self {
            LValue::Table(t) => Some(unsafe { t.as_ref() }),
            _ => None,
        }
    }

    pub fn as_luafn(&self) -> Option<&Proto> {
        match self {
            LValue::Function(f) => Some(unsafe { f.as_ref() }),
            _ => None,
        }
    }

    pub fn as_rsfn(&self) -> Option<RsFunc> {
        match self {
            LValue::RsFn(rf) => Some(*rf),
            _ => None,
        }
    }
}

mod size_check {
    #[test]
    fn strwrapper_size_check() {
        use super::StrWrapper;
        use std::mem::size_of;

        assert_eq!(size_of::<StrWrapper>(), 32);
    }

    #[test]
    fn value_size_check() {
        use super::LValue;
        use std::mem::size_of;

        assert_eq!(size_of::<LValue>(), 16);
    }
}

mod platform_check {

    #[test]
    fn x64_os_check() {
        assert!(cfg!(target_pointer_width = "64"))
    }

    #[test]
    fn heap_addr_lookup() {
        use super::{TaggedBox, HEAP_ADDR_HEAD, HEAP_ADDR_HEAD_MASK, PAYLOAD_MASK, TAG_MASK};

        // notice: this will cause a memory leak
        for _ in 0..128 {
            let heap_addr = Box::into_raw(Box::new(1u64));
            let nb = TaggedBox::from_heap_ptr(0x0, heap_addr);

            // print!("{:0>16x}  : ", heap_addr as u64);
            // print!("{:0>16x}  : ", nb.in_raw());
            // print!("{:0>16x}  : ", nb.as_ptr::<u64>() as usize);
            // println!();

            assert_eq!(nb.in_raw() & HEAP_ADDR_HEAD_MASK, HEAP_ADDR_HEAD);
        }
    }

    #[test]
    fn f64nan_head_lookup() {
        use super::{TaggedBox, HEAP_ADDR_HEAD_MASK, PAYLOAD_MASK, TAG_MASK};
        let nan = u64::from_ne_bytes(f64::NAN.to_ne_bytes());
        assert_ne!(nan & HEAP_ADDR_HEAD_MASK, 0);
    }

    #[test]
    fn constants_check() {
        use super::{TaggedBox, HEAP_ADDR_HEAD_MASK, PAYLOAD_MASK, TAG_MASK};
        assert!(PAYLOAD_MASK & TAG_MASK == 0);
        assert!(PAYLOAD_MASK | TAG_MASK == u64::MAX);
    }
}

mod test {

    #[test]
    fn basic() {
        use super::TaggedBox;

        let mut nb = TaggedBox::new(0, 0);
        assert_eq!(nb.tag(), 0);
        assert_eq!(nb.payload(), 0);

        nb.set_payload(137);
        assert_eq!(nb.payload(), 137);

        nb.set_tag(2);
        assert_eq!(nb.tag(), 2);

        let some_u48 = 0x0000_0000_1234_1111;
        nb.set_payload(some_u48);
        assert_eq!(nb.payload(), some_u48 as u64);

        assert_eq!(nb.replace_tag_with(4), 2);
        assert_eq!(nb.replace_tag_with(255), 4);
        assert_eq!(nb.replace_tag_with(9), 255);

        let some_u64 = 0xFFFF_0000_1234_1111_u64;
        assert_eq!(nb.replace_payload_with(some_u64 as usize), some_u48);
        assert_eq!(nb.replace_payload_with(1), some_u48);
        assert_eq!(nb.payload(), 1);
    }

    #[test]
    fn load_and_restore_heap_ptr() {
        use super::TaggedBox;

        let init = 123456789_u64;
        for _ in 0..1024 {
            let heap_addr = Box::into_raw(Box::new(init));
            let nb = TaggedBox::from_heap_ptr(1, heap_addr);

            assert_eq!(nb.tag(), 1);
            assert_eq!(unsafe { *nb.as_ptr::<u64>() }, init);
        }
    }

    #[test]
    fn hash_test() {
        use std::collections::hash_map::DefaultHasher;
        use std::hash::{Hash, Hasher};

        let i = 0;
        let k = 0;

        // k.hash(&mut hasher);
        // let kh = hasher.finish();
        // assert_ne!(ih, kh);

        // let s = "123".to_string();

        // assert_ne!(s.as_ptr(), e.as_ptr());

        // s.chars().for_each(|c| c.hash(&mut hasher));
        // let sh = hasher.finish();

        let e = "123".to_string();

        let mut hasher = DefaultHasher::new();

        // i.hash(&mut hasher);
        // let ih = hasher.finish();

        e.chars().for_each(|c| c.hash(&mut hasher));
        let eh = hasher.finish();

        let mut ohasher = DefaultHasher::new();
        e.chars().for_each(|c| c.hash(&mut ohasher));
        let oeh = ohasher.finish();

        assert_eq!(oeh, eh);

        // assert_eq!(sh, eh);
    }
}
