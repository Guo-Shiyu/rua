use std::{
    cell::Cell,
    collections::{BTreeMap, LinkedList},
    fmt::Debug,
    hash::{DefaultHasher, Hash, Hasher},
    ops::{Deref, DerefMut},
    ptr::NonNull,
};

use crate::{
    codegen::Proto,
    value::{RsFunc, Value},
};

/// # Tagged Ptr Layout of NanBox
///
/// ``` text
/// on 64-bit OS, there are only 48 address lines used for addressing
/// so, a pointer can be stored in low 48 bit, and extra 16 bits can be
/// used to mark the ptr's type (in fact there are only 4 bits used)):
///
///          head      tag          payload
///     +------------+ +--+ +---------------------+
///     0111 1111 1111 xxxx yyyy yyyy ... yyyy yyyy
///     +------------+ +--+ +---------------------+
///         12 bit     4 bit         48 bit
///
/// ```

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub struct TagBox {
    repr: u64, // inner representation
}

#[derive(PartialEq, Debug)]
pub enum Tag {
    String = 1,
    Table = 2,
    Proto = 3,
    LuaClosure = 4,
    RsClosure = 5,
    UserData = 6,
}

impl TagBox {
    const PAYLOAD_MASK: u64 = 0x0000_FFFF_FFFF_FFFF;
    const TAG_MASK: u64 = 0xFFFF_0000_0000_0000;

    /// Heap address head mask for 64 bit OS
    const HEAP_ADDR_HEAD: u64 = 0x0000_0000_0000_0000;

    pub fn new(tag: Tag, payload: usize) -> Self {
        let mut repr = u64::MIN;

        // set tag bits
        repr |= (tag as u64) << 48;

        // set payload bits
        let fixed_pl = Self::PAYLOAD_MASK & payload as u64;
        repr |= fixed_pl;

        Self { repr }
    }

    pub fn null() -> Self {
        Self { repr: 0 }
    }

    pub fn is_null(&self) -> bool {
        *self == Self::null()
    }

    pub fn raw_repr(&self) -> u64 {
        u64::from_ne_bytes(self.repr.to_ne_bytes())
    }

    pub fn tag(&self) -> Tag {
        unsafe { std::mem::transmute(((self.raw_repr() & Self::TAG_MASK) >> 48) as u8) }
    }

    pub fn payload(&self) -> u64 {
        self.raw_repr() & Self::PAYLOAD_MASK
    }

    pub fn as_gcheader(&self) -> &GcHeader {
        unsafe {
            &(self.payload() as *mut WithGcHeader<Table>)
                .as_ref()
                .unwrap() // TODO: unwrap unchecked
                .gcheader
        }
    }

    pub fn get<T: GcObject>(&self) -> Option<Gc<T>> {
        (self.tag() == T::TAGID).then(|| unsafe {
            Gc {
                ptr: NonNull::<WithGcHeader<T>>::new_unchecked(self.as_ptr_mut()),
            }
        })
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
        (self.payload() | Self::HEAP_ADDR_HEAD) as *const T
    }

    pub fn as_ptr_mut<T>(&self) -> *mut T {
        (self.payload() | Self::HEAP_ADDR_HEAD) as *mut T
    }

    pub fn from_heap_ptr<T>(tag: Tag, payload: *const T) -> Self {
        Self::new(tag, payload as usize)
    }
}

/// Method for type level
pub trait TypeTag {
    const TAGID: Tag;
}

/// GC color, used for thr-colo mark and sweep gc
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GcColor {
    Wild, // not mamaged by any vm
    Black,
    Gray,
    White,
    AnotherWhite,
}

impl Default for GcColor {
    fn default() -> Self {
        Self::Wild
    }
}

pub type GcAge = u8;

pub struct GcHeader {
    pub color: Cell<GcColor>,
    pub age: Cell<GcAge>,
}

impl Default for GcHeader {
    fn default() -> Self {
        Self {
            color: Cell::new(GcColor::Wild),
            age: Cell::new(0),
        }
    }
}

pub trait MemStat {
    /// How many extra memory resource hold by this object.
    fn mem_ref(&self) -> usize;
}

pub trait GcOp {
    fn mark_newborned(&self, _white: GcColor) {}
    fn mark_reachable(&self) {}
    fn mark_unreachable(&self) {}
}

pub struct WithGcHeader<T: GcOp + MemStat> {
    pub gcheader: GcHeader,
    pub inner: T,
}

impl<T: GcOp + MemStat> GcOp for WithGcHeader<T> {
    /// Mark Gc Color to White
    fn mark_newborned(&self, white: GcColor) {
        self.gcheader.color.set(white);
        self.inner.mark_newborned(white);
    }

    /// Mark Gc Color to Black
    fn mark_reachable(&self) {
        self.gcheader.color.set(GcColor::Black);
        self.inner.mark_reachable()
    }

    /// Mark Gc Color to Gray
    fn mark_unreachable(&self) {
        self.gcheader.color.set(GcColor::Gray);
        self.inner.mark_unreachable()
    }
}

impl<T: GcOp + MemStat> MemStat for WithGcHeader<T> {
    fn mem_ref(&self) -> usize {
        std::mem::size_of::<Self>() + self.inner.mem_ref()
    }
}

pub trait GcObject: GcOp + MemStat + TypeTag {}

impl<T: GcOp + MemStat + TypeTag> GcObject for T {}

/// Pointer type for gc managed objects
#[derive(Debug, Hash)]
pub struct Gc<T: GcObject> {
    ptr: NonNull<WithGcHeader<T>>,
}

impl<T: GcObject> Clone for Gc<T> {
    /// Shallow copy.
    fn clone(&self) -> Self {
        *self
    }
}

impl<T: GcObject> Copy for Gc<T> {}

impl<T: GcObject> PartialEq for Gc<T> {
    fn eq(&self, other: &Self) -> bool {
        self.address() == other.address()
    }
}

impl<T: GcObject> Eq for Gc<T> {}

impl<T: GcObject> Deref for Gc<T> {
    type Target = T;
    fn deref(&self) -> &T {
        unsafe { &self.ptr.as_ref().inner }
    }
}

impl<T: GcObject> DerefMut for Gc<T> {
    fn deref_mut(&mut self) -> &mut T {
        unsafe { &mut self.ptr.as_mut().inner }
    }
}

impl<T: GcObject> From<*mut WithGcHeader<T>> for Gc<T> {
    fn from(value: *mut WithGcHeader<T>) -> Self {
        debug_assert!(!value.is_null());
        Self {
            ptr: unsafe { NonNull::new_unchecked(value) },
        }
    }
}

impl<T: GcObject> From<*const WithGcHeader<T>> for Gc<T> {
    fn from(value: *const WithGcHeader<T>) -> Self {
        Gc {
            ptr: unsafe { NonNull::new_unchecked(value as *mut _) },
        }
    }
}

impl<T: GcObject> From<T> for Gc<T> {
    fn from(value: T) -> Self {
        Self::new(value)
    }
}

impl<T: GcObject> From<Gc<T>> for TagBox {
    fn from(val: Gc<T>) -> Self {
        TagBox::from_heap_ptr(T::TAGID, val.ptr.as_ptr())
    }
}

impl<T: GcObject> GcOp for Gc<T> {
    fn mark_newborned(&self, white: GcColor) {
        self.header().color.set(white);
        unsafe { self.ptr.as_ref().mark_newborned(white) };
    }

    fn mark_reachable(&self) {
        self.header().color.set(GcColor::Black);
        unsafe { self.ptr.as_ref().mark_reachable() };
    }

    fn mark_unreachable(&self) {
        self.header().color.set(GcColor::Gray);
        unsafe { self.ptr.as_ref().mark_unreachable() };
    }
}

impl<T: GcObject> MemStat for Gc<T> {
    fn mem_ref(&self) -> usize {
        unsafe { self.ptr.as_ref().mem_ref() }
    }
}

// impl PartialEq for Gc<StrImpl> {
//     fn eq(&self, r: &Self) -> bool {
//         // NOTE & TODO:
//         // Gen hashid with a random seed to prevent hash collision attck
//         if self.heap_address() == r.heap_address() {
//             true
//         } else {
//             unsafe { self.ptr.as_ref().inner == r.ptr.as_ref().inner }
//         }
//     }
// }

impl<T: GcObject> Gc<T> {
    pub fn dangling() -> Gc<T> {
        Gc {
            ptr: NonNull::dangling(),
        }
    }

    pub fn new(val: T) -> Gc<T> {
        let boxed = Box::new(WithGcHeader {
            gcheader: GcHeader::default(),
            inner: val,
        });

        Gc {
            ptr: unsafe { NonNull::new_unchecked(Box::into_raw(boxed)) },
        }
    }

    pub fn drop(val: Gc<T>) {
        let _ = val.into_inner();
    }

    // Return heap address of GcHeader (not the object)
    pub fn address(&self) -> usize {
        self.ptr.as_ptr() as usize
    }

    fn header(&self) -> &GcHeader {
        unsafe { &self.ptr.as_ref().gcheader }
    }

    pub fn into_inner(mut self) -> Box<WithGcHeader<T>> {
        unsafe { Box::from_raw(self.ptr.as_mut()) }
    }
}

pub type StrHashVal = u32;

/// Indecates that the length of inplace buffer in struct`Short`.
/// if given compile flag "long_inplace_str", the buffer length is 47 to keep the size of `WithGcHeader<StrImpl>` is 64.  
/// if not (in default) the buffer length is 23, and the size of `WithGcHeader<StrImpl>` is 40, same with Lua 5.4.4.
const MAX_INPLACE_STR_LEN: usize = if cfg!(long_inplace_str) { 47 } else { 23 };
#[derive(Debug)]
pub struct Short {
    hash: Cell<StrHashVal>,
    len: u8,
    data: [u8; MAX_INPLACE_STR_LEN], // not ensured to end with '\0'
}

impl Short {
    pub fn is_reserved(&self) -> bool {
        0x80_u8 & self.len == 0x80_u8
    }

    pub fn len(&self) -> usize {
        (0x0F_u8 & self.len) as usize
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }
}
#[derive(Debug)]
pub struct Long {
    hash: Cell<StrHashVal>,
    data: Box<str>,
}

#[derive(Debug)]
pub enum StrImpl {
    Short(Short),
    Long(Long),
}

impl TypeTag for StrImpl {
    const TAGID: Tag = Tag::String;
}

impl GcOp for StrImpl {}

impl MemStat for StrImpl {
    fn mem_ref(&self) -> usize {
        match self {
            StrImpl::Short(_) => 0, //  no extra heap memory used for short string
            StrImpl::Long(l) => l.data.len(),
        }
    }
}

impl PartialEq for StrImpl {
    fn eq(&self, r: &Self) -> bool {
        if self.len() == r.len() && self.hashval() == r.hashval() {
            if self.is_internalized() {
                true
            } else {
                self.as_str() == r.as_str()
            }
        } else {
            false
        }
    }
}

impl Deref for StrImpl {
    type Target = str;
    fn deref(&self) -> &Self::Target {
        self.as_str()
    }
}

impl From<String> for StrImpl {
    fn from(value: String) -> Self {
        if StrImpl::able_to_internalize(&value) {
            StrImpl::from_short(&value, None)
        } else {
            StrImpl::from_long(value)
        }
    }
}

impl From<&str> for StrImpl {
    fn from(value: &str) -> Self {
        if StrImpl::able_to_internalize(value) {
            StrImpl::from_short(value, None)
        } else {
            StrImpl::from_long(value.to_string())
        }
    }
}

impl StrImpl {
    const NOT_HASHED: StrHashVal = 0;

    pub fn from_long(val: String) -> StrImpl {
        StrImpl::Long(Long {
            hash: Cell::new(Self::NOT_HASHED),
            data: val.into(),
        })
    }

    /// Construct a short string from given string.
    pub fn from_short(short: &str, hashval: Option<StrHashVal>) -> StrImpl {
        debug_assert!(Self::able_to_internalize(short));
        let hash = Cell::new(hashval.unwrap_or_else(|| Self::hash_str(short)));
        let len = short.len() as u8;
        let mut data = [0_u8; MAX_INPLACE_STR_LEN];
        data[..short.len()].copy_from_slice(short.as_bytes());
        StrImpl::Short(Short { hash, len, data })
    }

    /// Construct a reserved short string from given string.
    pub fn new_reserved(short: &str, hashval: Option<StrHashVal>) -> StrImpl {
        debug_assert!(Self::able_to_internalize(short));
        let hash = Cell::new(hashval.unwrap_or_else(|| Self::hash_str(short)));
        let len = short.len() as u8 | 0x80;
        let mut data = [0_u8; MAX_INPLACE_STR_LEN];
        data[..short.len()].copy_from_slice(short.as_bytes());
        StrImpl::Short(Short { hash, len, data })
    }

    pub fn hashval(&self) -> StrHashVal {
        match self {
            StrImpl::Short(s) => s.hash.get(),
            StrImpl::Long(l) => {
                if self.has_hashed() {
                    l.hash.get()
                } else {
                    let hashid = Self::hash_str(&l.data);
                    l.hash.set(hashid);
                    hashid
                }
            }
        }
    }

    pub fn len(&self) -> usize {
        match self {
            StrImpl::Short(s) => s.len(),
            StrImpl::Long(l) => l.data.len(),
        }
    }

    pub fn is_reserved(&self) -> bool {
        matches!(&self, Self::Short(s) if s.is_reserved())
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    pub fn as_str(&self) -> &str {
        match self {
            StrImpl::Short(s) => unsafe { std::str::from_utf8_unchecked(&s.data[..self.len()]) },
            StrImpl::Long(l) => &l.data,
        }
    }

    pub fn is_internalized(&self) -> bool {
        match self {
            StrImpl::Short(_) => true,
            StrImpl::Long(_) => false,
        }
    }

    pub fn is_long(&self) -> bool {
        !self.is_internalized()
    }

    pub fn has_hashed(&self) -> bool {
        if let StrImpl::Long(l) = self {
            l.hash.get() != Self::NOT_HASHED
        } else {
            true
        }
    }

    /// Return true if s is able to be internalized.
    pub fn able_to_internalize(s: &str) -> bool {
        s.len() <= MAX_INPLACE_STR_LEN
    }

    /// Incomplete hash for long string and complete for short string.
    pub fn hash_str(ss: &str) -> StrHashVal {
        // incomplete hash for long string
        let step = if StrImpl::able_to_internalize(ss) {
            1
        } else {
            ss.len() >> 5
        };

        let mut hasher = DefaultHasher::new();
        ss.chars().step_by(step).for_each(|c| c.hash(&mut hasher));
        let full = hasher.finish();
        (full as u32 & u32::MAX) ^ (((full >> 32) as u32) & u32::MAX)
    }
}

#[derive(Clone, Copy)]
pub enum MetaOperator {
    MetaTable,
    Index,
    NewIndex,
    Gc,
    Mode,
    Len,
    Equal,
    Add,
    Sub,
    Mul,
    Mod,
    Pow,
    Div,
    IDiv,
    BitAnd,
    BitOr,
    BitXor,
    Shl,
    Shr,
    Unm,
    BitNot,
    Less,
    LessEqual,
    Concat,
    Call,
    Close,
    ToString,
    Pairs,
    IPairs,
}

impl MetaOperator {
    // Do *not* change the order of literals
    #[rustfmt::skip]
    pub const METATOPS_STRS: [&'static str; 29]  = [
        "__metatable",
        "__index", "__newindex",
        "__gc", "__mode", "__len", "__eq",
        "__add", "__sub", "__mul", "__mod", "__pow",
        "__div", "__idiv",
        "__band", "__bor", "__bxor", "__shl", "__shr",
        "__unm", "__bnot", "__lt", "__le",
        "__concat", "__call", "__close",
        "__tostring", "__pairs", "__ipairs",
    ];

    fn as_str(&self) -> &'static str {
        Self::METATOPS_STRS[*self as u8 as usize]
    }
}

impl From<MetaOperator> for &str {
    fn from(val: MetaOperator) -> Self {
        val.as_str()
    }
}

#[derive(Default, Debug)]
pub struct Table {
    hmap: BTreeMap<Value, Value>,
    meta: Option<Gc<Table>>,
}

impl TypeTag for Table {
    const TAGID: Tag = Tag::Table;
}

impl MemStat for Table {
    fn mem_ref(&self) -> usize {
        std::mem::size_of::<Self>()
    }
}

impl GcOp for Table {
    fn mark_newborned(&self, _: GcColor) {
        debug_assert!(self.hmap.is_empty());
    }

    fn mark_reachable(&self) {
        self.hmap.iter().for_each(|(key, val)| {
            key.mark_reachable();
            val.mark_reachable();
        });
    }

    fn mark_unreachable(&self) {
        self.hmap.iter().for_each(|(key, val)| {
            key.mark_unreachable();
            val.mark_unreachable();
        });
    }
}

impl Deref for Table {
    type Target = BTreeMap<Value, Value>;
    fn deref(&self) -> &Self::Target {
        &self.hmap
    }
}

impl DerefMut for Table {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.hmap
    }
}

impl Table {
    pub fn meta_get(&self, op: MetaOperator) -> Option<Value> {
        let target = op.as_str();
        self.meta.and_then(|meta| {
            for (k, v) in meta.iter() {
                match k {
                    Value::Str(s) => {
                        if s.as_str() == target {
                            return Some(v.clone());
                        }
                    }
                    _ => continue,
                }
            }
            None
        })
    }

    pub fn index<T>(&mut self, key: T) -> Value
    where
        Value: From<T>,
    {
        self.get(&Value::from(key)).cloned().unwrap_or_default()
    }
}

#[derive(Debug, Clone, Copy)]
pub struct UserData {
    ptr: NonNull<()>,
    size: usize,
}

impl MemStat for UserData {
    fn mem_ref(&self) -> usize {
        self.size
    }
}

impl GcOp for UserData {}

impl UserData {
    pub fn new<T>(data: T) -> UserData {
        UserData {
            ptr: NonNull::from(Box::leak(Box::new(data))).cast(),
            size: std::mem::size_of::<T>(),
        }
    }

    pub fn as_ptr<T>(&self) -> *const T {
        self.ptr.cast().as_ptr()
    }

    pub unsafe fn as_mut<T>(&mut self) -> *mut T {
        self.ptr.cast().as_mut()
    }

    pub unsafe fn as_ref<T>(&self) -> &T {
        self.ptr.cast().as_ref()
    }
}

#[derive(Debug)]
pub enum UpVal {
    Open(u32),    // stack index
    Close(Value), // value itself
}

#[derive(Debug)]
pub struct LuaClosure {
    pub p: Gc<Proto>,
    pub upvals: Vec<UpVal>,
}

impl LuaClosure {
    fn new(proto: Gc<Proto>) -> Self {
        Self {
            p: proto,
            upvals: Vec::new(),
        }
    }
}

impl TypeTag for LuaClosure {
    const TAGID: Tag = Tag::LuaClosure;
}

impl MemStat for LuaClosure {
    fn mem_ref(&self) -> usize {
        self.p.mem_ref() + self.upvals.capacity()
    }
}

impl GcOp for LuaClosure {
    fn mark_newborned(&self, white: GcColor) {
        self.p.mark_newborned(white);
        self.upvals.iter().for_each(|up| {
            if let UpVal::Close(val) = up {
                val.mark_newborned(white)
            }
        });
    }

    fn mark_reachable(&self) {
        self.p.mark_reachable();
        self.upvals.iter().for_each(|up| {
            if let UpVal::Close(val) = up {
                val.mark_reachable()
            }
        });
    }

    fn mark_unreachable(&self) {
        self.p.mark_unreachable();
        self.upvals.iter().for_each(|up| {
            if let UpVal::Close(val) = up {
                val.mark_unreachable()
            }
        });
    }
}

impl Deref for LuaClosure {
    type Target = Proto;
    fn deref(&self) -> &Self::Target {
        &self.p
    }
}

impl DerefMut for LuaClosure {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.p
    }
}

#[derive(Debug)]
pub struct RsClosure {
    fnptr: RsFunc,
    upvals: Box<[Value]>,
}

impl MemStat for RsClosure {
    fn mem_ref(&self) -> usize {
        self.upvals.len() * std::mem::size_of::<Value>()
    }
}

impl GcOp for RsClosure {
    fn mark_newborned(&self, white: GcColor) {
        self.upvals.iter().for_each(|val| val.mark_newborned(white));
    }

    fn mark_reachable(&self) {
        self.upvals.iter().for_each(|val| val.mark_reachable());
    }

    fn mark_unreachable(&self) {
        self.upvals.iter().for_each(|val| val.mark_unreachable());
    }
}

impl TypeTag for RsClosure {
    const TAGID: Tag = Tag::RsClosure;
}

impl TypeTag for UserData {
    const TAGID: Tag = Tag::UserData;
}

impl MemStat for Value {
    fn mem_ref(&self) -> usize {
        match self {
            // thry are not allocated on heap
            Value::Nil | Value::Bool(_) | Value::Int(_) | Value::Float(_) | Value::RsFn(_) => 0,

            Value::Str(sw) => sw.mem_ref(),
            Value::Table(tb) => tb.mem_ref(),
            Value::Fn(p) => p.mem_ref(),
            Value::RsCl(c) => c.mem_ref(),
            Value::UserData(ud) => ud.mem_ref(),
        }
    }
}

impl GcOp for Value {
    fn mark_newborned(&self, white: GcColor) {
        match self {
            Value::Str(s) => s.mark_newborned(white),
            Value::Table(t) => t.mark_newborned(white),
            Value::Fn(f) => f.mark_newborned(white),
            Value::RsCl(c) => c.mark_newborned(white),
            Value::UserData(ud) => ud.mark_newborned(white),
            _ => {}
        }
    }

    fn mark_reachable(&self) {
        match self {
            Value::Str(s) => s.mark_reachable(),
            Value::Table(t) => t.mark_reachable(),
            Value::Fn(f) => f.mark_reachable(),
            Value::RsCl(c) => c.mark_reachable(),
            Value::UserData(ud) => ud.mark_reachable(),
            _ => {}
        }
    }

    fn mark_unreachable(&self) {
        match self {
            Value::Str(s) => s.mark_unreachable(),
            Value::Table(t) => t.mark_unreachable(),
            Value::Fn(f) => f.mark_unreachable(),
            Value::UserData(ud) => ud.mark_unreachable(),
            _ => {}
        }
    }
}

impl From<TagBox> for Value {
    /// See 'Value::tagid()' method for each case
    fn from(tp: TagBox) -> Self {
        type RsCl = self::RsClosure;
        type LC = self::LuaClosure;
        match tp.tag() {
            StrImpl::TAGID => Value::Str(Gc::from(tp.as_ptr_mut::<WithGcHeader<StrImpl>>())),
            Table::TAGID => Value::Table(Gc::from(tp.as_ptr_mut::<WithGcHeader<Table>>())),
            LC::TAGID => Value::Fn(Gc::from(tp.as_ptr_mut::<WithGcHeader<LC>>())),
            RsCl::TAGID => Value::RsCl(Gc::from(tp.as_ptr_mut::<WithGcHeader<RsCl>>())),
            UserData::TAGID => Value::UserData(Gc::from(tp.as_ptr_mut::<WithGcHeader<UserData>>())),
            _ => unreachable!(),
        }
    }
}

#[derive(Clone, Copy, PartialEq, Default)]
pub enum GcStage {
    #[default]
    Start,
    Propgate,
    Sweep,
    Finalize,
}

pub struct Heap {
    total: usize,                                // total bytes of total allocated object
    debt: isize,                                 //
    estimate: usize,                             //
    allocs: LinkedList<TagBox>,                  // all object allocated by gc
    fixed: Vec<Gc<StrImpl>>,                     // fixed gc objects
    sstrpool: BTreeMap<StrHashVal, Gc<StrImpl>>, // short string cache

    stage: GcStage,
    curwhite: GcColor,
}

impl Default for Heap {
    fn default() -> Self {
        Heap {
            total: 0,
            debt: Self::FIRST_GC_LIMIT,
            estimate: 0,

            allocs: LinkedList::<_>::new(),

            fixed: Vec::<_>::new(),
            sstrpool: BTreeMap::<_, _>::new(),

            curwhite: GcColor::White,
            stage: GcStage::Start,
        }
    }
}

impl Drop for Heap {
    fn drop(&mut self) {
        // println!(
        //     "#-- Heap: total allocs: {}, fixed object: {}",
        //     self.total,
        //     self.fixed.len()
        // );
        std::mem::take(&mut self.allocs)
            .into_iter()
            .for_each(|obj| {
                self.free_obj(obj);
            });
        std::mem::take(&mut self.fixed)
            .into_iter()
            .for_each(|tbox| {
                self.free_obj(tbox.into());
            });
        self.sstrpool.clear();
    }
}

impl Heap {
    /// How many allocated bytes to trig first GC
    const FIRST_GC_LIMIT: isize = 1024;

    /// Factor
    // const GC_GROW_FACTOR: f64 = 2.0;

    pub fn new() -> Heap {
        Self::default()
    }

    pub fn check_gc(&self) -> bool {
        self.debt > 0 && self.estimate < self.total / 2
    }

    pub fn total_alloc_bytes(&self) -> usize {
        self.total
    }

    pub fn alloc_fixed(&mut self, reserved: &str) -> Value {
        debug_assert!(StrImpl::able_to_internalize(reserved));
        let hash = StrImpl::hash_str(reserved);
        let val = if self.sstrpool.contains_key(&hash) {
            // SAFETY: key has been checked to be exisited in `sstrpool`
            Value::from(unsafe { *self.sstrpool.get(&hash).unwrap_unchecked() })
        } else {
            let ptr = Gc::new(StrImpl::new_reserved(reserved, Some(hash)));
            self.sstrpool.entry(hash).or_insert(ptr);
            self.fixed.push(ptr);
            ptr.into()
        };
        val.mark_newborned(self.curwhite);
        val
    }

    pub fn alloc_str(&mut self, view: &str) -> Gc<StrImpl> {
        if StrImpl::able_to_internalize(view) {
            self.alloc_internal_str(view)
        } else {
            self.take_str(view.to_string())
        }
    }

    fn alloc_internal_str(&mut self, short: &str) -> Gc<StrImpl> {
        let hash = StrImpl::hash_str(short);
        let ptr = if self.sstrpool.contains_key(&hash) {
            // SAFETY: key has been checked to be exisited in `sstrpool`
            unsafe { *self.sstrpool.get(&hash).unwrap_unchecked() }
        } else {
            let ptr = Gc::new(StrImpl::from_short(short, Some(hash)));
            self.allocs.push_front(ptr.into());
            self.sstrpool.entry(hash).or_insert(ptr);
            self.record_mem_incr(ptr.mem_ref());
            ptr
        };
        ptr.mark_newborned(self.curwhite);
        ptr
    }

    pub fn take_str(&mut self, val: String) -> Gc<StrImpl> {
        if StrImpl::able_to_internalize(&val) {
            self.alloc_internal_str(&val)
        } else {
            let ptr = Gc::new(StrImpl::from_long(val));
            self.allocs.push_front(ptr.into());
            self.record_mem_incr(ptr.mem_ref());
            ptr.mark_newborned(self.curwhite);
            ptr
        }
    }

    pub fn alloc_table(&mut self) -> Gc<Table> {
        let ptr = Gc::new(Table::default());
        self.allocs.push_front(ptr.into());
        self.record_mem_incr(ptr.mem_ref());
        ptr.mark_newborned(self.curwhite);
        ptr
    }

    pub fn alloc_closure(&mut self, pfn: Gc<Proto>) -> Gc<LuaClosure> {
        let ptr = Gc::new(LuaClosure::new(pfn));
        self.allocs.push_front(ptr.into());
        self.record_mem_incr(ptr.mem_ref());
        ptr.mark_newborned(self.curwhite);
        ptr
    }

    pub(crate) fn mark_all_obj_unreachable(&self) {
        // mark all allocs object to gray.
        for ptr in self.allocs.iter() {
            ptr.as_gcheader().color.set(GcColor::Gray);
        }
    }

    pub(crate) fn sweep_unreachable(&mut self) -> Vec<Gc<Table>> {
        // sweep unreachable strings, table without __gc method, and other data
        let mut stack = std::mem::take(&mut self.allocs);
        let mut to_finalize = Vec::new();

        while let Some(tbox) = stack.back() {
            // alive objects
            if tbox.as_gcheader().color.get() != GcColor::Gray
                || tbox.get::<StrImpl>().is_some_and(|s| s.is_reserved())
            {
                self.allocs.append(&mut stack.split_off(stack.len() - 1));
                continue;
            }

            // table with __gc method needs to drop within VM levle
            if tbox
                .get::<Table>()
                .and_then(|table| table.meta_get(MetaOperator::Gc))
                .is_none()
            {
                self.sweep_garbage(*tbox);
                stack.pop_back();
            } else {
                // # SAFETY: we has checked the tail of stack is a table object
                to_finalize.push(unsafe {
                    stack
                        .pop_back()
                        .unwrap_unchecked()
                        .get::<Table>()
                        .unwrap_unchecked()
                });
            }
        }
        to_finalize
    }

    pub(crate) fn sweep_garbage(&mut self, tbox: TagBox) {
        debug_assert!(tbox.as_gcheader().color.get() == GcColor::Gray);
        // common object
        self.total -= self.free_obj(tbox);
    }

    fn free_obj(&mut self, tbox: TagBox) -> usize {
        match tbox.tag() {
            Tag::String => Self::do_free_with::<StrImpl, _>(tbox, |s| {
                if s.is_internalized() && !s.is_reserved() {
                    self.sstrpool.remove(&s.hashval());
                }
            }),
            Tag::Table => Self::do_free_with::<Table, _>(tbox, |_| {}),
            Tag::Proto => Self::do_free_with::<Proto, _>(tbox, |_| {}),
            Tag::LuaClosure => Self::do_free_with::<LuaClosure, _>(tbox, |_| {}),
            Tag::UserData => Self::do_free_with::<UserData, _>(tbox, |_| {
                // TODO: light userdata ?
            }),
            Tag::RsClosure => Self::do_free_with::<RsClosure, _>(tbox, |_| {}),
        }
    }

    fn do_free_with<T: GcObject + Debug, F: FnOnce(Gc<T>)>(tbox: TagBox, f: F) -> usize {
        let val = Gc::from(tbox.as_ptr_mut::<WithGcHeader<T>>());
        f(val);

        // {
        //     print!(
        //         "#-- drop {:?} at: 0x{:X} , free mem size: {}, val:",
        //         tbox.tag(),
        //         val.address(),
        //         val.mem_ref(),
        //     );
        //     if tbox.tag() == Tag::String {
        //         println!(" {}", tbox.get::<StrImpl>().unwrap().as_str());
        //     } else {
        //         println!(" {:?}", val);
        //     }
        // }

        let used = val.mem_ref();
        Gc::drop(val);
        used
    }

    pub fn is_generational(&self) -> bool {
        false
    }

    pub fn is_incremental(&self) -> bool {
        false
    }

    fn switch_white(&mut self) {
        self.curwhite = match self.curwhite {
            GcColor::White => GcColor::AnotherWhite,
            GcColor::AnotherWhite => GcColor::White,
            _ => unreachable!(),
        };
    }

    // fn cur_white(&self) -> GcColor {
    //     self.curwhite
    // }

    // fn is_cur_white(&self, header: &GcHeader) -> bool {
    //     header.color.get() == self.curwhite
    // }

    fn record_mem_incr(&mut self, incr: usize) {
        self.total += incr;
        self.debt += incr as isize;
    }
}

#[cfg(test)]
mod test {
    use super::*;

    #[test]
    fn x64_os_check() {
        assert!(cfg!(target_pointer_width = "64"))
    }

    #[test]
    fn constants_check() {
        use super::TagBox;

        assert!(TagBox::PAYLOAD_MASK & TagBox::TAG_MASK == 0);
        assert!(TagBox::PAYLOAD_MASK | TagBox::TAG_MASK == u64::MAX);
    }

    #[test]
    fn reserved_string() {
        let mut heap = Heap::default();
        let raw = "function";
        let rst = heap.alloc_fixed(raw);
        let ptr = rst.as_str().unwrap();
        assert_eq!(ptr.len(), raw.len());
        assert!(ptr.is_reserved());
    }

    #[test]
    fn tagbox_and_gcptr() {
        let ptr = Gc::new(StrImpl::from("value"));
        let tagbox = TagBox::from(ptr);

        assert_eq!(tagbox.tag(), StrImpl::TAGID);
        assert_eq!(tagbox.payload(), ptr.address() as u64);

        assert_eq!(
            tagbox.as_ptr::<StrImpl>(), //
            ptr.address() as *const StrImpl
        );
        assert_eq!(
            tagbox.as_ptr_mut::<StrImpl>(),
            ptr.address() as *mut StrImpl
        );

        let gcp = tagbox.get::<StrImpl>().unwrap();
        assert_eq!(gcp, ptr);

        Gc::drop(ptr);
    }

    // #[test]
    // #[should_panic(expected = "free(): double free detected")]
    // fn double_free_gcptr() {
    //     let ptr = Gc::new(StrImpl::from("value"));
    //     Gc::drop(ptr); // first free
    //     Gc::drop(ptr); // double free, this should panic
    // }

    #[test]
    fn test_table() {
        use crate::state::VM;
        use crate::InterpretError;

        let mut heap = Heap::default();
        let mut tb = heap.alloc_table();
        assert_ne!(tb.mem_ref(), 0);

        {
            fn f1(_: &mut VM) -> Result<usize, InterpretError> {
                Ok(1)
            }

            fn f2(_: &mut VM) -> Result<usize, InterpretError> {
                Ok(2)
            }

            let v1 = Value::from(f1 as RsFunc);
            let v2 = Value::from(f2 as RsFunc);
            assert_eq!(v2, v2);
            assert_eq!(v1, Value::from(f1 as RsFunc));
            assert_ne!(v1, v2);

            let v1c = Value::from(f1 as RsFunc);
            assert_eq!(v1, v1c);
            assert_eq!(v1.to_ne_bytes(), v1c.to_ne_bytes());

            tb.insert(Value::from(1), v1);
            tb.insert(Value::from(2), v2);
            tb.insert(v1, v2);
            tb.insert(v2, v1);

            assert_eq!(tb.index(Value::from(1)), v1);
            assert_eq!(tb.index(Value::from(2)), v2);
            assert_eq!(tb.index(v1), v2);
            assert_eq!(tb.index(1), v1);
            assert_eq!(tb.index(2), v2);

            assert_eq!(Value::from(f1 as RsFunc), Value::from(f1 as RsFunc),);
            let l = tb.index(f1 as RsFunc);
            let r = tb.index(f1 as RsFunc);
            assert_eq!(l, r);

            assert_eq!(tb.index(v2), v1);
        }
    }

    #[test]
    fn destroy_heap_with_fixed_object() {
        use crate::heap::{Heap, MetaOperator};

        let mut heap = Heap::default();
        let origin = heap.total;

        for builtin in MetaOperator::METATOPS_STRS {
            heap.alloc_fixed(builtin);
        }
        assert_eq!(origin, heap.total);
    }
}
