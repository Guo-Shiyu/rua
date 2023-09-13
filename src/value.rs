use std::{
    cell::Cell,
    collections::{hash_map::DefaultHasher, HashMap},
    fmt::{Debug, Display},
    hash::{Hash, Hasher},
};

use crate::{
    codegen::Proto,
    heap::{
        Gc, GcColor, GcHeader, HeapMemUsed, MarkAndSweepGcOps, Tag, TaggedBox, TypeTag,
        WithGcHeader,
    },
};

use super::{state::State, RuntimeErr};

#[derive(Debug)]
struct FlexibleArrayMarker();

pub type StrHashVal = u32;

/// String are implemented as a flexible array
/// Depends on the fixed layout, so repr as C.
#[repr(C)]
#[derive(Debug)]
pub struct StrImpl {
    len: usize,             // len of string excluding header
    capacity: usize,        // len of buffer including header
    hash: Cell<StrHashVal>, // hash unique
    extra: Cell<bool>,      // reserved (for short)  | hashed (for long)
    _inplace_ptr: FlexibleArrayMarker,
}

impl HeapMemUsed for StrImpl {
    fn heap_mem_used(&self) -> usize {
        self.len + std::mem::size_of::<StrImpl>()
    }
}

impl Debug for Gc<StrImpl> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Gc<StrImpl>")
            .field("len", &self.len)
            .field("capacity", &self.capacity)
            .field("hash", &self.hash.get())
            .field("extra", &self.extra.get())
            .finish()
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
    fn from(mut s: String) -> Self {
        let strlen = s.len();
        let total_need = s.len() + StrImpl::EXTRA_HEADERS_SIZE;
        if total_need > s.capacity() {
            return Self::from(s.as_str());
        }

        if s.capacity() > 2 * total_need {
            s.shrink_to(total_need);
        }

        unsafe {
            // shuffle string to the end of buffer
            let mut rsrc = s.as_ptr().add(s.len());
            let mut rdest = rsrc.add(StrImpl::EXTRA_HEADERS_SIZE) as *mut u8;
            for _ in 0..=s.len() {
                *rdest = *rsrc;
                rdest = rdest.sub(1);
                rsrc = rsrc.sub(1);
            }

            let (with_gch, cap) = (
                std::mem::transmute::<*mut u8, *mut WithGcHeader<StrImpl>>(s.as_mut_ptr()),
                s.capacity(),
            );
            std::mem::forget(s);

            let obj = with_gch.as_mut().unwrap_unchecked();
            obj.gcheader.color.set(GcColor::Wild);
            obj.gcheader.age.set(0);
            obj.inner.len = strlen;
            obj.inner.capacity = cap;
            obj.inner.extra.set(false);
            if obj.inner.is_short() {
                obj.inner.hash = Cell::new(StrImpl::hash_str(obj.inner.as_str()));
            } else {
                obj.inner.hash = Cell::new(StrHashVal::MIN);
            }

            Gc::from(with_gch)
        }
    }
}

impl From<&str> for Gc<StrImpl> {
    fn from(s: &str) -> Self {
        let (with_gch, cap) = unsafe {
            let mut buf = String::with_capacity(s.len() + StrImpl::EXTRA_HEADERS_SIZE);
            let dest = std::slice::from_raw_parts_mut(
                buf.as_ptr().add(StrImpl::EXTRA_HEADERS_SIZE) as *mut u8,
                s.len(),
            );
            dest.copy_from_slice(std::slice::from_raw_parts(s.as_ptr(), s.len()));

            let res = (
                buf.as_mut_ptr() as *mut WithGcHeader<StrImpl>,
                buf.capacity(),
            );
            std::mem::forget(buf);
            res // avoid drop buf
        };

        unsafe {
            let obj = with_gch.as_mut().unwrap_unchecked();
            obj.gcheader.color.set(GcColor::Wild);
            obj.gcheader.age.set(0);
            obj.inner.len = s.len();
            obj.inner.capacity = cap;
            obj.inner.extra.set(false);
            if obj.inner.is_short() {
                obj.inner.hash = Cell::new(StrImpl::hash_str(s));
            } else {
                obj.inner.hash = Cell::new(StrHashVal::MIN);
            }
        }
        Gc::from(with_gch)
    }
}

impl StrImpl {
    pub const STR_HEADER_SIZE: usize = std::mem::size_of::<StrImpl>();
    pub const CACHELINE_SIZE: usize = 64;
    pub const EXTRA_HEADERS_SIZE: usize = Self::STR_HEADER_SIZE + std::mem::size_of::<GcHeader>();
    pub const MAX_SHORT_LEN: usize = Self::CACHELINE_SIZE - Self::EXTRA_HEADERS_SIZE;

    pub fn hash_str(ss: &str) -> StrHashVal {
        let step = if Self::check_short(ss) {
            1
        } else {
            ss.len() >> 5
        };

        let mut hasher = DefaultHasher::new();
        ss.chars().step_by(step).for_each(|c| c.hash(&mut hasher));
        hasher.finish() as StrHashVal
    }

    pub fn hashid(&self) -> StrHashVal {
        if self.is_long() && !self.extra.get() {
            self.hash.set(Self::hash_str(self.as_str()));
            self.extra.set(true);
        };
        self.hash.get()
    }

    pub fn len(&self) -> usize {
        self.len
    }

    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    pub fn as_str(&self) -> &str {
        unsafe {
            let ptr = (self as *const StrImpl as *const u8).add(Self::STR_HEADER_SIZE);
            let slc = std::slice::from_raw_parts(ptr, self.len);
            std::str::from_utf8(slc).unwrap()
        }
    }

    pub fn is_short(&self) -> bool {
        self.len <= Self::MAX_SHORT_LEN
    }

    pub fn is_long(&self) -> bool {
        !self.is_short()
    }

    pub fn is_reserved(&self) -> bool {
        self.is_short() && self.extra.get()
    }

    /// Return true if s is short
    pub fn check_short(s: &str) -> bool {
        s.len() <= StrImpl::MAX_SHORT_LEN
    }

    pub fn drop(val: Gc<StrImpl>) {
        unsafe {
            let ptr = val.heap_address() as *mut u8;
            let _len = val.len;
            let cap = val.capacity;
            let _ = Box::from_raw(std::slice::from_raw_parts_mut(ptr, cap));
        }
    }

    pub fn new_reserved(s: &str) -> Gc<StrImpl> {
        let gc = Gc::from(s);
        debug_assert!(gc.is_short());
        gc.extra.set(true);
        gc
    }
}

pub trait AsTableKey {
    fn as_key(&self) -> usize;
}

#[repr(C)]
#[derive(Default)]
pub struct TableImpl {
    hmap: HashMap<usize, LValue>,
    array: Vec<LValue>,
    meta: Option<Gc<TableImpl>>,
}

impl HeapMemUsed for TableImpl {
    fn heap_mem_used(&self) -> usize {
        let self_and_map = self
            .hmap
            .iter()
            .fold(std::mem::size_of::<TableImpl>(), |acc, (_, val)| {
                acc + val.heap_mem_used()
            });

        let ary = self
            .array
            .iter()
            .fold(0, |acc, val| acc + val.heap_mem_used());

        self_and_map + ary
    }
}

impl MarkAndSweepGcOps for TableImpl {
    fn mark_newborned(&self, _: GcColor) {
        unimplemented!()
    }

    fn mark_reachable(&self) {
        self.hmap.iter().for_each(|(_, val)| val.mark_reachable());
        self.array.iter().for_each(|val| val.mark_reachable());
    }

    fn mark_untouched(&self) {
        todo!()
    }
}

impl TableImpl {
    pub fn with_capacity(cap: (usize, usize)) -> Self {
        Self {
            hmap: HashMap::with_capacity(cap.0),
            array: Vec::with_capacity(cap.1),
            meta: None,
        }
    }

    pub fn set<K>(&mut self, k: K, v: LValue)
    where
        K: AsTableKey,
    {
        self.hmap.insert(k.as_key(), v);
    }

    pub fn get<K>(&self, k: K) -> LValue
    where
        K: AsTableKey,
    {
        self.hmap.get(&k.as_key()).cloned().unwrap_or_default()
    }
}

/// a flexible array
#[repr(C)]
pub struct UserDataImpl {
    pub size: usize,
    _inplace_ptr: FlexibleArrayMarker,
}

impl HeapMemUsed for UserDataImpl {
    fn heap_mem_used(&self) -> usize {
        self.size
    }
}

impl UserDataImpl {
    pub fn get_raw_data(&self) -> &[u8] {
        unsafe {
            let ptr = (self as *const UserDataImpl as *const u8).add(std::mem::size_of::<Self>());
            std::slice::from_raw_parts(ptr, self.size)
        }
    }

    pub fn as_ptr<T>(&self) -> *const T {
        self.get_raw_data().as_ptr() as *const T
    }

    pub fn as_mut<T>(&mut self) -> *mut T {
        unsafe {
            (self as *const UserDataImpl as *const u8).add(std::mem::size_of::<Self>()) as *mut T
        }
    }

    pub fn drop(val: Gc<UserDataImpl>) {
        unsafe {
            let ptr = val.heap_address() as *mut u8;
            let len = val.size;
            let _ = Box::from_raw(std::slice::from_raw_parts_mut(ptr, len));
        }
    }
}

// usize: stack size after call
pub type RsFunc = fn(&mut State) -> Result<usize, RuntimeErr>;

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
    Function(Gc<Proto>),        // lua function
    UserData(Gc<UserDataImpl>), // light rs user data, managed by lua
}

impl TypeTag for StrImpl {
    fn tagid() -> Tag {
        0
    }
}

impl TypeTag for TableImpl {
    fn tagid() -> Tag {
        1
    }
}

impl TypeTag for Proto {
    fn tagid() -> Tag {
        2
    }
}

impl TypeTag for UserDataImpl {
    fn tagid() -> Tag {
        3
    }
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
            LValue::UserData(ud) => ud.mark_newborned(white),
            _ => {}
        }
    }

    fn mark_reachable(&self) {
        match self {
            LValue::String(s) => s.mark_reachable(),
            LValue::Table(t) => t.mark_reachable(),
            LValue::Function(f) => f.mark_reachable(),
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

impl From<TaggedBox> for LValue {
    /// See 'LValue::tagid()' method for each case
    fn from(tp: TaggedBox) -> Self {
        use LValue::*;
        type Udi = UserDataImpl;
        match tp.tag() {
            s if s == StrImpl::tagid() => String(Gc::from(tp.as_mut::<WithGcHeader<StrImpl>>())),
            t if t == TableImpl::tagid() => Table(Gc::from(tp.as_mut::<WithGcHeader<TableImpl>>())),
            f if f == Proto::tagid() => Function(Gc::from(tp.as_mut::<WithGcHeader<Proto>>())),
            u if u == Udi::tagid() => UserData(Gc::from(tp.as_mut::<WithGcHeader<UserDataImpl>>())),
            _ => unreachable!(),
        }
    }
}

impl AsTableKey for LValue {
    fn as_key(&self) -> usize {
        match *self {
            LValue::Nil => unreachable!("index table by nil key"),
            LValue::Bool(b) => b as usize,
            LValue::Int(i) => i as usize,
            LValue::Float(f) => usize::from_ne_bytes(f.to_ne_bytes()),
            LValue::String(s) => ((s.hashid() as usize) << 32) | s.len(),
            LValue::Table(t) => t.gch_ptr() as usize,
            LValue::Function(f) => f.gch_ptr() as usize,
            LValue::RsFn(rf) => rf as usize,
            LValue::UserData(ud) => ud.gch_ptr() as usize,
        }
    }
}

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

impl From<Gc<Proto>> for LValue {
    fn from(value: Gc<Proto>) -> Self {
        LValue::Function(value)
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
            LValue::String(s) => Ok(TaggedBox::from_heap_ptr(StrImpl::tagid(), s.gch_ptr())),
            LValue::Table(t) => Ok(TaggedBox::from_heap_ptr(TableImpl::tagid(), t.gch_ptr())),
            LValue::Function(f) => Ok(TaggedBox::from_heap_ptr(Proto::tagid(), f.gch_ptr())),
            LValue::UserData(u) => Ok(TaggedBox::from_heap_ptr(UserDataImpl::tagid(), u.gch_ptr())),
            _ => Err(()),
        }
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
            String(s) => write!(f, "\"{}\"", s.as_str()),
            Table(t) => write!(f, "table: 0x{:X}", t.heap_address()),
            Function(func) => write!(f, "function: 0x{:X}", { func.heap_address() }),
            RsFn(rsf) => write!(f, "rsfn: 0x{:X}", rsf as *const _ as usize),
            UserData(ud) => write!(f, "userdata: 0x{:X}", ud.heap_address()),
        }
    }
}

impl LValue {
    pub fn from_wild<T>(val: Gc<T>) -> LValue
    where
        T: TypeTag + HeapMemUsed,
        LValue: From<Gc<T>>,
    {
        LValue::from(val)
    }

    /// Check if the value is managed by lua gc
    pub fn is_managed(&self) -> bool {
        use LValue::*;
        match self {
            Nil | Bool(_) | Int(_) | Float(_) => false,
            String(_) | Table(_) | Function(_) | RsFn(_) | UserData { .. } => true,
        }
    }

    pub fn tagid(&self) -> Option<Tag> {
        use LValue::*;
        match self {
            Nil | Bool(_) | Int(_) | Float(_) | RsFn(_) => None,
            String(_) => Some(StrImpl::tagid()),
            Table(_) => Some(TableImpl::tagid()),
            Function(_) => Some(Proto::tagid()),
            UserData(_) => Some(UserDataImpl::tagid()),
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

    pub fn as_string(&self) -> Option<&StrImpl> {
        match self {
            LValue::String(s) => Some(s),
            _ => None,
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

mod size_check {
    #[test]
    fn value_size_check() {
        use super::LValue;
        use std::mem::size_of;

        assert_eq!(size_of::<LValue>(), 16);
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

    /// Test for string implementation, this function will casuse memory leak
    #[test]
    fn strimpl() {
        use super::Gc;
        use super::StrImpl;

        {
            let s = "hello world";
            let gs = Gc::from(s);

            assert!(gs.is_short());
            assert!(!gs.is_reserved());
            assert_eq!(gs.len(), s.len());
            assert_eq!(gs.as_str(), s);

            let ogs = Gc::from(s);
            assert_eq!(ogs.as_str(), gs.as_str());
            assert_eq!(ogs.hashid(), gs.hashid());

            assert_eq!(*ogs, *gs);
        }

        {
            let s = "hello world".repeat(10);
            assert!(!StrImpl::check_short(s.as_str()));

            let gs = Gc::<StrImpl>::from(s.as_str());
            assert!(gs.is_long());
            assert!(!gs.is_reserved());
            assert_eq!(gs.len(), s.len());
            assert_eq!(gs.as_str(), s);
        }

        {
            let mut s = String::with_capacity(StrImpl::MAX_SHORT_LEN * 4);
            let original_buf_size = s.capacity();
            s.push_str("short string");
            let gs = Gc::<StrImpl>::from(s);
            assert!(gs.is_short());

            // buffer will shirnk to fit the length if it is too large
            assert!(gs.capacity < original_buf_size);
        }
    }
}
