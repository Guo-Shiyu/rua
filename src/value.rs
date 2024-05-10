use core::fmt;
use std::{
    cell::Cell,
    collections::{hash_map::DefaultHasher, BTreeMap},
    fmt::{Debug, Display},
    hash::{Hash, Hasher},
    ops::{Deref, DerefMut},
    ptr::NonNull,
};

use crate::{
    codegen::Proto,
    heap::{
        Gc, GcColor, Heap, HeapMemUsed, MarkAndSweepGcOps, Tag, TaggedBox, TypeTag, WithGcHeader,
    },
    RuaErr,
};

use super::state::State;

pub type StrHashVal = u32;

/// Indecates that the length of inplace buffer in struct`Short`.
/// if given compile flag "long_inplace_str", the buffer length is 47 to keep the size of `WithGcHeader<StrImpl>` is 64.  
/// if not (in default) the buffer length is 23, and the size of `WithGcHeader<StrImpl>` is 40.
const MAX_SHORT_STR_LEN: usize = if cfg!(long_inplace_str) { 47 } else { 23 };

pub struct Short {
    hash: Cell<StrHashVal>,
    len: u8,
    data: [u8; MAX_SHORT_STR_LEN], // not ensured to end with '\0'
}

pub struct Long {
    hash: Cell<StrHashVal>,
    hashed: Cell<bool>,
    data: Box<str>,
}

pub enum StrImpl {
    Short(Short),
    Long(Long),
}

impl TypeTag for StrImpl {
    const TAGID: Tag = Tag::String;
}

impl MarkAndSweepGcOps for StrImpl {}

impl HeapMemUsed for StrImpl {
    fn heap_mem_used(&self) -> usize {
        match self {
            StrImpl::Short(_) => 0, //  no extra heap memory used for short string
            StrImpl::Long(l) => l.data.len(),
        }
    }
}

impl PartialEq for StrImpl {
    fn eq(&self, r: &Self) -> bool {
        if self.len() == r.len() && self.hashid() == r.hashid() {
            if self.is_short() {
                true
            } else {
                self.as_str() == r.as_str()
            }
        } else {
            false
        }
    }
}

impl From<String> for Gc<StrImpl> {
    fn from(s: String) -> Self {
        let sobj = if StrImpl::able_to_store_inplace(&s) {
            StrImpl::inplace(&s)
        } else {
            StrImpl::Long(Long {
                hash: Cell::new(0),
                hashed: Cell::new(false),
                data: s.into_boxed_str(),
            })
        };
        Gc::new(sobj)
    }
}

impl From<&str> for Gc<StrImpl> {
    fn from(s: &str) -> Self {
        let sobj = if StrImpl::able_to_store_inplace(s) {
            StrImpl::inplace(s)
        } else {
            StrImpl::Long(Long {
                hash: Cell::new(0),
                hashed: Cell::new(false),
                data: s.into(),
            })
        };
        Gc::new(sobj)
    }
}

impl StrImpl {
    pub fn hash_str(ss: &str) -> StrHashVal {
        let step = if Self::able_to_internal(ss) {
            1
        } else {
            ss.len() >> 5
        };

        let mut hasher = DefaultHasher::new();
        ss.chars().step_by(step).for_each(|c| c.hash(&mut hasher));
        hasher.finish() as StrHashVal
    }

    pub fn hashid(&self) -> StrHashVal {
        match self {
            StrImpl::Short(s) => s.hash.get(),
            StrImpl::Long(l) => {
                if l.hashed.get() {
                    l.hash.get()
                } else {
                    let hashid = Self::hash_str(&l.data);
                    l.hash.set(hashid);
                    l.hashed.set(true);
                    hashid
                }
            }
        }
    }

    pub fn len(&self) -> usize {
        match self {
            StrImpl::Short(s) => s.len as usize,
            StrImpl::Long(l) => l.data.len(),
        }
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

    pub fn is_short(&self) -> bool {
        match self {
            StrImpl::Short(_) => true,
            StrImpl::Long(_) => false,
        }
    }

    pub fn is_long(&self) -> bool {
        !self.is_short()
    }

    pub fn is_hashed(&self) -> bool {
        if let StrImpl::Long(l) = self {
            l.hashed.get()
        } else {
            true
        }
    }

    /// Return true if s is able to be internalized.
    pub fn able_to_internal(s: &str) -> bool {
        s.len() <= MAX_SHORT_STR_LEN
    }

    pub fn able_to_store_inplace(s: &str) -> bool {
        s.len() <= MAX_SHORT_STR_LEN
    }

    /// Construct a short string from given string.
    fn inplace(short: &str) -> StrImpl {
        debug_assert!(Self::able_to_store_inplace(short));
        let hash = Cell::new(StrImpl::hash_str(short));
        let len = short.len() as u8;
        let mut data = [0_u8; MAX_SHORT_STR_LEN];
        data[..short.len()].copy_from_slice(short.as_bytes());
        StrImpl::Short(Short { hash, len, data })
    }
}

pub trait AsTableKey {
    fn as_key(&self) -> usize;
}

#[repr(C)]
#[derive(Default)]
pub struct TableImpl {
    hmap: BTreeMap<LValue, LValue>,
    array: Vec<LValue>,
    meta: Option<Gc<TableImpl>>,
}

impl TypeTag for TableImpl {
    const TAGID: Tag = Tag::Table;
}

impl HeapMemUsed for TableImpl {
    fn heap_mem_used(&self) -> usize {
        self.hmap
            .values()
            .chain(self.array.iter())
            .fold(0, |acc, val| acc + val.heap_mem_used())
    }
}

impl MarkAndSweepGcOps for TableImpl {
    fn mark_newborned(&self, _: GcColor) {
        unimplemented!()
    }

    fn mark_reachable(&self) {
        self.hmap.iter().for_each(|(key, val)| {
            key.mark_reachable();
            val.mark_reachable();
        });
        self.array.iter().for_each(|val| val.mark_reachable());
    }

    fn mark_untouched(&self) {
        todo!()
    }
}

impl TableImpl {
    pub fn delegate_to(_p: &mut TableImpl, _heap: &mut Heap) {
        todo!()
    }

    pub fn with_capacity(hash_cap: usize, array_cap: usize) -> Self {
        Self {
            // hmap: HashMap::with_capacity(hash_cap),
            hmap: Default::default(),
            array: Vec::with_capacity(array_cap),
            meta: None,
        }
    }

    pub fn set(&mut self, k: LValue, v: LValue) {
        self.hmap.insert(k, v);
    }

    pub fn get(&self, k: LValue) -> LValue {
        self.hmap.get(&k).cloned().unwrap_or_default()
    }
}

#[derive(Debug, Clone, Copy)]
pub struct UserDataImpl {
    ptr: NonNull<()>,
    size: usize,
}

impl HeapMemUsed for UserDataImpl {
    fn heap_mem_used(&self) -> usize {
        self.size
    }
}

impl MarkAndSweepGcOps for UserDataImpl {}

impl UserDataImpl {
    pub fn new<T>(data: Box<T>) -> UserDataImpl {
        UserDataImpl {
            ptr: NonNull::from(Box::leak(data)).cast(),
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

// usize: stack size after call
pub type RsFunc = fn(&mut State) -> Result<usize, RuaErr>;

pub enum UpVal {
    Open(u32),     // stack index
    Close(LValue), // itself
}

pub struct LClosure {
    pub p: Gc<Proto>,
    pub upvals: Vec<UpVal>,
}

impl Deref for LClosure {
    type Target = Proto;
    fn deref(&self) -> &Self::Target {
        &self.p
    }
}

impl DerefMut for LClosure {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.p
    }
}

impl HeapMemUsed for LClosure {
    fn heap_mem_used(&self) -> usize {
        self.source.heap_mem_used()
            + self.code.capacity()
            + self.pcline.capacity()
            + self
                .kst
                .iter()
                .fold(self.kst.capacity(), |acc, k| acc + k.heap_mem_used())
            + self
                .locvars
                .iter()
                .fold(self.locvars.capacity(), |acc, l| acc + l.name.capacity())
            + self
                .updecl
                .iter()
                .fold(self.updecl.capacity(), |acc, u| acc + u.name().len())
            + self.subfn.iter().fold(self.subfn.capacity(), |acc, f| {
                // acc + f.heap_mem_used()
                acc
            })
            + self.upvals.capacity()
    }
}

impl MarkAndSweepGcOps for LClosure {
    fn delegate_to(&mut self, heap: &mut Heap) {
        heap.delegate(&mut self.source);
        self.kst.iter_mut().for_each(|k| heap.delegate(k));
        // self.subfn.iter_mut().for_each(|s| s.delegate_to(heap));
    }

    fn mark_newborned(&self, white: crate::heap::GcColor) {
        self.source.mark_newborned(white);
        self.kst.iter().for_each(|k| k.mark_newborned(white));
        // self.subfn.iter().for_each(|s| s.mark_newborned(white));
    }

    fn mark_reachable(&self) {
        self.source.mark_reachable();
        self.kst.iter().for_each(|k| k.mark_reachable());
        // self.subfn.iter().for_each(|s| s.mark_reachable());
    }

    fn mark_untouched(&self) {
        todo!()
    }
}

pub struct RsClosure {
    f: RsFunc,
    upvals: Vec<UpVal>,
}

impl HeapMemUsed for RsClosure {
    fn heap_mem_used(&self) -> usize {
        self.upvals.capacity()
    }
}

impl MarkAndSweepGcOps for RsClosure {}

#[derive(Clone, Copy, Default)]
pub enum LValue {
    #[default]
    Nil,
    Bool(bool),
    Int(i64),
    Float(f64),
    RsFn(RsFunc),               // light rs function
    String(Gc<StrImpl>),        // lua immutable string
    Table(Gc<TableImpl>),       // lua table
    Function(Gc<LClosure>),     // lua function
    RsCll(Gc<RsClosure>),       // rust closure
    UserData(Gc<UserDataImpl>), // user data, managed by lua
}

impl TypeTag for LClosure {
    const TAGID: Tag = Tag::LuaClosure;
}

impl TypeTag for RsClosure {
    const TAGID: Tag = Tag::RsClosure;
}

impl TypeTag for UserDataImpl {
    const TAGID: Tag = Tag::UserData;
}

impl HeapMemUsed for LValue {
    fn heap_mem_used(&self) -> usize {
        match self {
            // thry are not allocated on heap
            LValue::Nil | LValue::Bool(_) | LValue::Int(_) | LValue::Float(_) | LValue::RsFn(_) => {
                0
            }

            LValue::String(sw) => sw.heap_mem_used(),
            LValue::Table(tb) => tb.heap_mem_used(),
            LValue::Function(p) => p.heap_mem_used(),
            LValue::RsCll(c) => c.heap_mem_used(),
            LValue::UserData(ud) => ud.heap_mem_used(),
        }
    }
}

impl MarkAndSweepGcOps for LValue {
    fn mark_newborned(&self, white: GcColor) {
        match self {
            LValue::String(s) => s.mark_newborned(white),
            LValue::Table(t) => t.mark_newborned(white),
            LValue::Function(f) => f.mark_newborned(white),
            LValue::RsCll(c) => c.mark_newborned(white),
            LValue::UserData(ud) => ud.mark_newborned(white),
            _ => {}
        }
    }

    fn mark_reachable(&self) {
        match self {
            LValue::String(s) => s.mark_reachable(),
            LValue::Table(t) => t.mark_reachable(),
            LValue::Function(f) => f.mark_reachable(),
            LValue::RsCll(c) => c.mark_reachable(),
            LValue::UserData(ud) => ud.mark_reachable(),
            _ => {}
        }
    }

    fn mark_untouched(&self) {
        match self {
            LValue::String(s) => s.mark_untouched(),
            LValue::Table(t) => t.mark_untouched(),
            LValue::Function(f) => f.mark_untouched(),
            LValue::UserData(ud) => ud.mark_untouched(),
            _ => {}
        }
    }
}

impl PartialEq for LValue {
    fn eq(&self, other: &Self) -> bool {
        match (*self, *other) {
            (LValue::Nil, LValue::Nil) => true,
            (LValue::Bool(l), LValue::Bool(r)) => l == r,
            (LValue::Int(l), LValue::Int(r)) => l == r,
            (LValue::Float(l), LValue::Float(r)) => l == r,
            (LValue::RsFn(l), LValue::RsFn(r)) => l == r,
            (LValue::String(l), LValue::String(r)) => l == r,
            (LValue::Table(l), LValue::Table(r)) => l.heap_address() == r.heap_address(),
            (LValue::Function(l), LValue::Function(r)) => l.heap_address() == r.heap_address(),
            (LValue::UserData(l), LValue::UserData(r)) => l.heap_address() == r.heap_address(),
            _ => false,
        }
    }
}

impl Eq for LValue {}

impl PartialOrd for LValue {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        todo!()
    }
}

impl Ord for LValue {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        std::cmp::Ordering::Less
    }
}

impl From<TaggedBox> for LValue {
    /// See 'LValue::tagid()' method for each case
    fn from(tp: TaggedBox) -> Self {
        type RsCl = self::RsClosure;
        type Udi = UserDataImpl;
        type LC = self::LClosure;
        use LValue::*;
        match tp.tag() {
            StrImpl::TAGID => String(Gc::from(tp.as_mut::<WithGcHeader<StrImpl>>())),
            TableImpl::TAGID => Table(Gc::from(tp.as_mut::<WithGcHeader<TableImpl>>())),
            LC::TAGID => Function(Gc::from(tp.as_mut::<WithGcHeader<LC>>())),
            RsCl::TAGID => RsCll(Gc::from(tp.as_mut::<WithGcHeader<RsCl>>())),
            Udi::TAGID => UserData(Gc::from(tp.as_mut::<WithGcHeader<UserDataImpl>>())),
            _ => unreachable!(),
        }
    }
}

// impl AsTableKey for LValue {
//     fn as_key(&self) -> usize {
//
//     }
// }

// impl Hash for LValue {
//     fn hash<H: Hasher>(&self, state: &mut H) {
//         match *self {
//             LValue::Nil => unreachable!("index table by nil key"),
//             LValue::String(s) => state.write_usize(((s.hashid() as usize) << 32) | s.len()),
//             LValue::Table(t) => state.write_usize(t.heap_address()),
//             LValue::Function(f) => state.write_usize(f.heap_address()),
//             LValue::RsCll(c) => state.write_usize(c.heap_address()),
//             LValue::UserData(ud) => state.write_usize(ud.heap_address()),
//             LValue::RsFn(rf) => state.write_usize(rf as usize),
//             LValue::Bool(i) => {
//                 state.write_usize(&self as *const _ as usize);
//                 state.write_u8(i as u8);
//                 state.write_u8(!i as u8);
//             }
//             LValue::Int(i) => {
//                 state.write_usize(&self as *const _ as usize);
//                 state.write_i64(i);
//             }
//             LValue::Float(f) => {
//                 state.write_usize(&self as *const _ as usize);
//                 state.write(&f.to_ne_bytes());
//             }
//         }
//     }
// }

impl From<bool> for LValue {
    fn from(value: bool) -> Self {
        LValue::Bool(value)
    }
}

impl From<i64> for LValue {
    fn from(value: i64) -> Self {
        LValue::Int(value)
    }
}

impl From<f64> for LValue {
    fn from(value: f64) -> Self {
        LValue::Float(value)
    }
}

impl From<u64> for LValue {
    fn from(value: u64) -> Self {
        if value > i64::MAX as u64 {
            LValue::Float(value as f64)
        } else {
            LValue::Int(value as i64)
        }
    }
}

impl From<usize> for LValue {
    fn from(value: usize) -> Self {
        if value > i64::MAX as usize {
            LValue::Float(value as f64)
        } else {
            LValue::Int(value as i64)
        }
    }
}

impl From<&str> for LValue {
    fn from(value: &str) -> Self {
        LValue::String(Gc::from(value.to_string()))
    }
}

impl From<String> for LValue {
    fn from(value: String) -> Self {
        LValue::String(Gc::from(value))
    }
}

impl From<RsFunc> for LValue {
    fn from(value: RsFunc) -> Self {
        LValue::RsFn(value)
    }
}

impl From<Gc<StrImpl>> for LValue {
    fn from(value: Gc<StrImpl>) -> Self {
        LValue::String(value)
    }
}

impl From<Gc<TableImpl>> for LValue {
    fn from(value: Gc<TableImpl>) -> Self {
        LValue::Table(value)
    }
}

impl From<Gc<LClosure>> for LValue {
    fn from(value: Gc<LClosure>) -> Self {
        LValue::Function(value)
    }
}

impl From<Gc<RsClosure>> for LValue {
    fn from(value: Gc<RsClosure>) -> Self {
        LValue::RsCll(value)
    }
}

impl From<Gc<UserDataImpl>> for LValue {
    fn from(value: Gc<UserDataImpl>) -> Self {
        LValue::UserData(value)
    }
}

impl TryInto<TaggedBox> for LValue {
    type Error = ();
    fn try_into(self) -> Result<TaggedBox, Self::Error> {
        match self {
            LValue::String(s) => Ok(TaggedBox::new(StrImpl::TAGID, s.heap_address())),
            LValue::Table(t) => Ok(TaggedBox::new(TableImpl::TAGID, t.heap_address())),
            LValue::Function(f) => Ok(TaggedBox::new(LClosure::TAGID, f.heap_address())),
            LValue::UserData(u) => Ok(TaggedBox::new(UserDataImpl::TAGID, u.heap_address())),
            _ => Err(()),
        }
    }
}

impl Debug for LValue {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self)
    }
}

impl Display for LValue {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        use LValue::*;
        match self {
            Nil => write!(f, "nil"),
            Bool(b) => write!(f, "{}", b),
            Int(i) => write!(f, "{}", i),
            Float(fl) => write!(f, "{}", fl),
            String(s) => f.write_str(s.as_str()),
            Table(t) => write!(f, "table: 0x{:X}", t.heap_address()),
            Function(func) => write!(f, "function: 0x{:X}", func.heap_address()),
            RsCll(c) => write!(f, "function: 0x{:X}", c.heap_address()),
            RsFn(rsf) => write!(f, "rsfn: 0x{:X}", rsf as *const _ as usize),
            UserData(ud) => write!(f, "userdata: 0x{:X}", ud.heap_address()),
        }
    }
}

impl LValue {
    /// Check if the value is managed by lua gc
    pub fn is_managed(&self) -> bool {
        use LValue::*;
        match self {
            Nil | Bool(_) | Int(_) | Float(_) => false,
            String(_) | Table(_) | Function(_) | RsFn(_) | RsCll(_) | UserData { .. } => true,
        }
    }

    pub fn tagid(&self) -> Option<Tag> {
        use LValue::*;
        match self {
            Nil | Bool(_) | Int(_) | Float(_) | RsFn(_) => None,
            String(_) => Some(StrImpl::TAGID),
            Table(_) => Some(TableImpl::TAGID),
            Function(_) => Some(LClosure::TAGID),
            RsCll(_) => Some(RsClosure::TAGID),
            UserData(_) => Some(UserDataImpl::TAGID),
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

    pub fn is_str(&self) -> bool {
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

    pub fn is_callable(&self) -> bool {
        // TODO:
        // metatable __call metamethod
        self.is_luafn() || self.is_rsfn()
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

    pub fn as_str(&self) -> Option<&str> {
        match self {
            LValue::String(s) => Some(s.as_str()),
            _ => None,
        }
    }

    pub unsafe fn as_str_unchecked(&self) -> &str {
        match self {
            LValue::String(s) => s.as_str(),
            _ => unreachable!(),
        }
    }

    pub fn as_table(&self) -> Option<&TableImpl> {
        match self {
            LValue::Table(t) => Some(t),
            _ => None,
        }
    }

    pub fn as_luafn(&self) -> Option<&Proto> {
        match self {
            LValue::Function(f) => Some(f),
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

mod test {
    use crate::heap::Tag;

    #[test]
    fn value_size_check() {
        use super::LValue;
        use std::mem::size_of;

        assert_eq!(size_of::<LValue>(), 16);
    }

    #[test]
    fn basic() {
        use super::TaggedBox;

        let mut nb = TaggedBox::new(Tag::String, 0);
        assert_eq!(nb.tag(), Tag::String);
        assert_eq!(nb.payload(), 0);

        nb.set_payload(137);
        assert_eq!(nb.payload(), 137);

        nb.set_tag(Tag::Table);
        assert_eq!(nb.tag(), Tag::Table);

        let some_u48 = 0x0000_0000_1234_1111;
        nb.set_payload(some_u48);
        assert_eq!(nb.payload(), some_u48 as u64);

        assert_eq!(nb.replace_tag_with(Tag::String), Tag::Table);
        assert_eq!(nb.replace_tag_with(Tag::LuaClosure), Tag::String);
        assert_eq!(nb.replace_tag_with(Tag::RsClosure), Tag::LuaClosure);

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
            let nb = TaggedBox::from_heap_ptr(Tag::UserData, heap_addr);

            assert_eq!(nb.tag(), Tag::UserData);
            assert_eq!(unsafe { *nb.as_ptr::<u64>() }, init);
        }
    }

    /// Test for string implementation, this function will casuse memory leak
    #[test]
    fn strimpl() {
        use super::Gc;
        use super::StrImpl;
        use crate::heap::WithGcHeader;

        {
            // Expected to be 64 to fill the size of cache line in 64-bit platform.
            const STR_IMPL_SIZE: usize = std::mem::size_of::<WithGcHeader<StrImpl>>();
            assert!(STR_IMPL_SIZE <= 64);
        }

        {
            let s = "hello world";
            let gs = Gc::from(s);

            assert!(gs.is_short());
            assert_eq!(gs.len(), s.len());
            assert_eq!(gs.as_str(), s);

            let ogs = Gc::from(s);
            assert_eq!(ogs.as_str(), gs.as_str());
            assert_eq!(ogs.hashid(), gs.hashid());

            assert_eq!(ogs.as_str(), gs.as_str());
        }

        {
            let s = "hello world".repeat(10);
            assert!(!StrImpl::able_to_store_inplace(s.as_str()));

            let gs = Gc::<StrImpl>::from(s.as_str());
            assert!(gs.is_long());
            assert_eq!(gs.len(), s.len());
            assert_eq!(gs.as_str(), s);
        }
    }
}
