use std::{
    fmt::{Debug, Display},
    hash::{Hash, Hasher},
};

use crate::{
    heap::{Gc, LuaClosure, RsClosure, StrImpl, Table, Tag, TagBox, TypeTag, UserData},
    state::VM,
    InterpretError,
};

#[repr(C)]
#[derive(Clone, Copy, Default)]
pub enum Value {
    #[default]
    Nil,
    Bool(bool),
    Int(i64),
    Float(f64),
    RsFn(RsFunc),           // light rs function
    Str(Gc<StrImpl>),       // lua immutable string
    Table(Gc<Table>),       // lua table
    Fn(Gc<LuaClosure>),     // lua function
    RsCl(Gc<RsClosure>),    // rust closure
    UserData(Gc<UserData>), // user data, managed by lua
}

impl From<bool> for Value {
    fn from(value: bool) -> Self {
        Value::Bool(value)
    }
}

impl From<i32> for Value {
    fn from(val: i32) -> Self {
        Value::Int(val as i64)
    }
}

impl From<i64> for Value {
    fn from(value: i64) -> Self {
        Value::Int(value)
    }
}

impl From<f64> for Value {
    fn from(value: f64) -> Self {
        Value::Float(value)
    }
}

impl From<u64> for Value {
    fn from(value: u64) -> Self {
        if value > i64::MAX as u64 {
            Value::Float(value as f64)
        } else {
            Value::Int(value as i64)
        }
    }
}

impl From<usize> for Value {
    fn from(value: usize) -> Self {
        if value > i64::MAX as usize {
            Value::Float(value as f64)
        } else {
            Value::Int(value as i64)
        }
    }
}

impl From<RsFunc> for Value {
    fn from(value: RsFunc) -> Self {
        Value::RsFn(value)
    }
}

impl From<Gc<StrImpl>> for Value {
    fn from(value: Gc<StrImpl>) -> Self {
        Value::Str(value)
    }
}

impl From<Gc<Table>> for Value {
    fn from(value: Gc<Table>) -> Self {
        Value::Table(value)
    }
}

impl From<Gc<LuaClosure>> for Value {
    fn from(value: Gc<LuaClosure>) -> Self {
        Value::Fn(value)
    }
}

impl From<Gc<RsClosure>> for Value {
    fn from(value: Gc<RsClosure>) -> Self {
        Value::RsCl(value)
    }
}

impl From<Gc<UserData>> for Value {
    fn from(value: Gc<UserData>) -> Self {
        Value::UserData(value)
    }
}

impl<T> From<Option<T>> for Value
where
    Value: From<T>,
{
    fn from(value: Option<T>) -> Self {
        value.map(Into::into).unwrap_or_default()
    }
}

pub struct NotGcObject();

impl TryInto<TagBox> for Value {
    type Error = NotGcObject;
    fn try_into(self) -> Result<TagBox, NotGcObject> {
        match self {
            Value::Str(s) => Ok(TagBox::new(StrImpl::TAGID, s.address())),
            Value::Table(t) => Ok(TagBox::new(Table::TAGID, t.address())),
            Value::Fn(f) => Ok(TagBox::new(LuaClosure::TAGID, f.address())),
            Value::UserData(u) => Ok(TagBox::new(UserData::TAGID, u.address())),
            _ => Err(NotGcObject()),
        }
    }
}

impl Debug for Value {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        if self.is_str() {
            write!(f, "\"{}\"", self)
        } else {
            write!(f, "{}", self)
        }
    }
}

impl Display for Value {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        use Value::*;
        const LAST_8_DIGIT: usize = 0xFFFFFFFF;
        match self {
            Nil => write!(f, "nil"),
            Bool(b) => write!(f, "{}", b),
            Int(i) => write!(f, "{}", i),
            Float(fl) => write!(f, "{}", fl),
            Str(s) => write!(f, "{}", s.as_str()),
            Table(t) => write!(f, "table: 0x{:X}", t.address() & LAST_8_DIGIT),
            Fn(cls) => write!(f, "function: 0x{:X}", cls.address() & LAST_8_DIGIT),
            RsCl(rsc) => write!(f, "function: 0x{:X}", rsc.address() & LAST_8_DIGIT),
            RsFn(rsf) => write!(f, "rsfn: 0x{:X}", *rsf as usize & LAST_8_DIGIT),
            UserData(ud) => write!(f, "userdata: 0x{:X}", ud.address() & LAST_8_DIGIT),
        }
    }
}

impl Hash for Value {
    fn hash<H: Hasher>(&self, state: &mut H) {
        state.write(&self.to_ne_bytes());
    }
}

impl PartialEq for Value {
    fn eq(&self, other: &Self) -> bool {
        match (*self, *other) {
            (Value::Nil, Value::Nil) => true,
            (Value::Bool(l), Value::Bool(r)) => l.eq(&r),
            (Value::Int(l), Value::Int(r)) => l.eq(&r),
            (Value::Float(l), Value::Float(r)) => l.eq(&r),
            (Value::RsFn(l), Value::RsFn(r)) => l.eq(&r),
            (Value::Str(l), Value::Str(r)) => l.eq(&r),
            (Value::Table(l), Value::Table(r)) => l.eq(&r),
            (Value::Fn(l), Value::Fn(r)) => l.eq(&r),
            (Value::UserData(l), Value::UserData(r)) => l.eq(&r),
            (Value::RsCl(l), Value::RsCl(r)) => l.eq(&r),
            _ => false,
        }
    }
}

impl Eq for Value {}

impl Ord for Value {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        self.to_ne_bytes().cmp(&other.to_ne_bytes())
    }
}

impl PartialOrd for Value {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

impl Value {
    /// Check if the value is managed by lua heap.
    pub fn is_gcobj(&self) -> bool {
        self.tag().is_some()
    }

    pub fn tag(&self) -> Option<Tag> {
        use Value::{Bool, Float, Fn, Int, Nil, RsCl, RsFn, Str};
        match self {
            Nil | Bool(_) | Int(_) | Float(_) | RsFn(_) => None,
            Str(_) => Some(StrImpl::TAGID),
            Value::Table(_) => Some(Table::TAGID),
            Fn(_) => Some(LuaClosure::TAGID),
            RsCl(_) => Some(RsClosure::TAGID),
            Value::UserData(_) => Some(UserData::TAGID),
        }
    }

    // pub fn as_obj<T: GcObject>(&self) -> Option<Gc<T>> {
    //     self.tag().filter(|tag| *tag == T::TAGID).and_then(|| {self})
    // }

    pub fn is_falsey(&self) -> bool {
        match &self {
            Value::Nil => true,
            Value::Bool(b) => !b,
            _ => false,
        }
    }

    pub fn is_nil(&self) -> bool {
        matches!(self, Value::Nil)
    }

    pub fn is_bool(&self) -> bool {
        matches!(self, Value::Bool(_))
    }

    pub fn is_int(&self) -> bool {
        matches!(self, Value::Int(_))
    }

    pub fn is_float(&self) -> bool {
        matches!(self, Value::Float(_))
    }

    pub fn is_str(&self) -> bool {
        matches!(self, Value::Str(_))
    }

    pub fn is_table(&self) -> bool {
        matches!(self, Value::Table(_))
    }

    pub fn is_luafn(&self) -> bool {
        matches!(self, Value::Fn(_))
    }

    pub fn is_rsfn(&self) -> bool {
        matches!(self, Value::RsFn(_))
    }

    pub fn is_callable(&self) -> bool {
        use crate::heap::MetaOperator;
        self.is_luafn()
            || self.is_rsfn()
            || self
                .as_table()
                .is_some_and(|table| table.meta_get(MetaOperator::Call).is_some())
    }

    pub fn is_userdata(&self) -> bool {
        matches!(self, Value::UserData { .. })
    }

    pub fn as_bool(&self) -> Option<bool> {
        match self {
            Value::Bool(b) => Some(*b),
            _ => None,
        }
    }

    pub fn as_int(&self) -> Option<i64> {
        match self {
            Value::Int(i) => Some(*i),
            _ => None,
        }
    }

    pub fn as_float(&self) -> Option<f64> {
        match self {
            Value::Float(f) => Some(*f),
            _ => None,
        }
    }

    pub fn as_str(&self) -> Option<Gc<StrImpl>> {
        match self {
            Value::Str(s) => Some(*s),
            _ => None,
        }
    }

    pub unsafe fn as_str_unchecked(&self) -> &str {
        match self {
            Value::Str(s) => s.as_str(),
            _ => unreachable!(),
        }
    }

    pub fn as_table(&self) -> Option<Gc<Table>> {
        match self {
            Value::Table(t) => Some(*t),
            _ => None,
        }
    }

    pub fn as_luafn(&self) -> Option<&LuaClosure> {
        match self {
            Value::Fn(f) => Some(f),
            _ => None,
        }
    }

    pub fn as_rsfn(&self) -> Option<RsFunc> {
        match self {
            Value::RsFn(rf) => Some(*rf),
            _ => None,
        }
    }

    pub fn typestr(&self) -> &'static str {
        match self {
            Value::Nil => "nil",
            Value::Bool(_) => "boolean",
            Value::Int(_) | Value::Float(_) => "number",
            Value::Str(_) => "string",
            Value::Table(_) => "table",
            Value::RsFn(_) | Value::RsCl(_) | Value::Fn(_) => "function",
            Value::UserData(_) => "userdata",
        }
    }

    pub fn to_ne_bytes(&self) -> ValueBits {
        let (l, h): (u64, u64) = match *self {
            Value::Nil => (0, 0),
            Value::Bool(b) => (0x1, b as _),
            Value::Int(i) => (0x2, i as _),
            Value::Float(f) => (0x4, f.to_bits()),
            Value::RsFn(f) => (0x8, f as _),
            Value::Str(p) => (0x10, TagBox::from(p).raw_repr()),
            Value::Table(p) => (0x20, TagBox::from(p).raw_repr()),
            Value::Fn(p) => (0x40, TagBox::from(p).raw_repr()),
            Value::RsCl(p) => (0x80, TagBox::from(p).raw_repr()),
            Value::UserData(p) => (0x100, TagBox::from(p).raw_repr()),
        };
        unsafe { std::mem::transmute([l, h]) }
    }

    pub fn from_ne_bytes(repr: &ValueBits) -> Value {
        unsafe { std::mem::transmute_copy(repr) }
    }
}

pub type ValueBits = [u8; std::mem::size_of::<Value>()];

// usize: stack size after call
pub type RsFunc = fn(&mut VM) -> Result<usize, InterpretError>;

#[cfg(test)]
mod test {
    use super::*;

    #[test]
    fn value_size_check() {
        assert_eq!(std::mem::size_of::<Value>(), 16);

        let nil = Value::Nil;
        let other_nil = Value::Nil;
        assert_eq!(nil, other_nil);

        let l = nil.to_ne_bytes();
        let r = other_nil.to_ne_bytes();
        assert_eq!(l, r);

        assert_eq!(nil.to_ne_bytes(), other_nil.to_ne_bytes());
        assert!(l.into_iter().all(|c| c == 0));
    }

    #[test]
    fn tagbox_basic() {
        let mut nb = TagBox::new(Tag::String, 0);
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
        let data = 123456789_u64;
        for _ in 0..1024 {
            let heap_addr = Box::into_raw(Box::new(data));
            let nb = TagBox::from_heap_ptr(Tag::UserData, heap_addr);

            assert_eq!(nb.tag(), Tag::UserData);
            assert_eq!(unsafe { *nb.as_ptr::<u64>() }, data);
        }
    }

    /// Test for string implementation, this function will casuse memory leak
    #[test]
    fn test_strimpl() {
        use crate::heap::WithGcHeader;

        {
            // Expected to be 64 to fill the size of cache line in 64-bit platform.
            const STR_IMPL_SIZE: usize = std::mem::size_of::<WithGcHeader<StrImpl>>();
            assert!(STR_IMPL_SIZE <= 64);
        }

        {
            let short = "hello world";
            let str_repr = StrImpl::from_short(short, None);

            assert!(str_repr.is_internalized());
            assert_eq!(str_repr.len(), short.len());
            assert_eq!(str_repr.as_str(), short);

            let other_str = StrImpl::from_short(short, None);
            assert_eq!(other_str.as_str(), str_repr.as_str());
            assert_eq!(other_str.hashval(), str_repr.hashval());

            assert_eq!(other_str.as_str(), str_repr.as_str());
        }

        {
            let long_str = "hello world".repeat(10);
            assert!(!StrImpl::able_to_internalize(long_str.as_str()));

            let long = StrImpl::from(long_str.clone());
            assert!(long.is_long());
            assert_eq!(long.len(), long_str.len());
            assert_eq!(long.as_str(), long_str);
            assert!(!long.has_hashed());
            assert_ne!(long.hashval(), 0);
        }
    }

    #[test]
    fn test_num() {
        let (n1, n2) = (Value::from(1), Value::from(2));
        let n1c = Value::from(1);

        assert_eq!(n1, n1c);
        assert_eq!(n1.cmp(&n1c), std::cmp::Ordering::Equal);
        assert_ne!(n1, n2);

        assert_eq!(n1.to_ne_bytes(), n1c.to_ne_bytes());
        assert!(n1 < n2);
        assert!(n2 > n1);
        assert!(n2 >= n1);

        let (f1, f2) = (Value::from(2.0), Value::from(9.99));
        let f3 = Value::from(2.0000000000000000001);
        assert_eq!(f1, f3);
        assert_ne!(f2, f1);
        assert_ne!(f1, n2);
    }
}
