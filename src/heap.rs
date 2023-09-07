use std::{
    alloc::Layout,
    cell::Cell,
    collections::{BTreeMap, BTreeSet},
    ops::{Deref, DerefMut},
    ptr::NonNull,
};

use crate::{
    codegen::Proto,
    state::Rvm,
    value::{LValue, StrHashVal, StrImpl, TableImpl, UserDataImpl},
};

/// # Tagged Ptr Layout
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

#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub struct TaggedBox {
    repr: u64, // inner representation
}

pub type Tag = u8;

impl TaggedBox {
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

    pub fn in_raw(&self) -> u64 {
        u64::from_ne_bytes(self.repr.to_ne_bytes())
    }

    pub fn tag(&self) -> Tag {
        ((self.in_raw() & Self::TAG_MASK) >> 48) as u8
    }

    pub fn payload(&self) -> u64 {
        self.in_raw() & Self::PAYLOAD_MASK
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

    pub fn as_mut<T>(&self) -> *mut T {
        (self.payload() | Self::HEAP_ADDR_HEAD) as *mut T
    }

    pub fn from_heap_ptr<T>(tag: Tag, payload: *const T) -> Self {
        Self::new(tag, payload as usize)
    }
}

/// Method for type level
pub trait TypeTag {
    fn tagid() -> Tag;
}

/// GC color, used for thr-colo mark and sweep gc
#[repr(u32)]
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

pub type GcAge = u32;

#[repr(C)]
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

pub trait MarkAndSweepGcOps {
    /// Mark Gc Color to White
    fn mark_newborned(&self, white: GcColor);

    /// Mark Gc Color to Black
    fn mark_reachable(&self);

    /// Mark Gc Color to Gray
    fn mark_untouched(&self);
}

#[repr(C)]
pub struct WithGcHeader<T> {
    pub gcheader: GcHeader,
    pub inner: T,
}

impl<T> WithGcHeader<T> {
    pub fn raw_gch(&self) -> *const GcHeader {
        &self.gcheader as *const GcHeader
    }
}

pub trait HeapMemUsed {
    fn heap_mem_used(&self) -> usize;
}

/// Pointer type for gc managed objects
pub struct Gc<T: HeapMemUsed + TypeTag> {
    ptr: NonNull<WithGcHeader<T>>,
}

impl<T: HeapMemUsed + TypeTag> Clone for Gc<T> {
    fn clone(&self) -> Self {
        Self { ptr: self.ptr }
    }
}

impl<T: HeapMemUsed + TypeTag> Copy for Gc<T> {}

impl<T: HeapMemUsed + TypeTag> Deref for Gc<T> {
    type Target = T;
    fn deref(&self) -> &T {
        unsafe { &self.ptr.as_ref().inner }
    }
}

impl<T: HeapMemUsed + TypeTag> DerefMut for Gc<T> {
    fn deref_mut(&mut self) -> &mut T {
        unsafe { &mut self.ptr.as_mut().inner }
    }
}

impl<T: HeapMemUsed + TypeTag> From<*mut WithGcHeader<T>> for Gc<T> {
    fn from(value: *mut WithGcHeader<T>) -> Self {
        Self {
            ptr: unsafe { NonNull::new_unchecked(value) },
        }
    }
}

impl<T: HeapMemUsed + TypeTag> From<T> for Gc<T> {
    fn from(value: T) -> Self {
        Self::new(value)
    }
}

impl<T: HeapMemUsed + TypeTag> From<Gc<T>> for TaggedBox {
    fn from(val: Gc<T>) -> Self {
        TaggedBox::from_heap_ptr(T::tagid(), val.ptr.as_ptr())
    }
}

impl<T: HeapMemUsed + TypeTag> MarkAndSweepGcOps for Gc<T> {
    fn mark_newborned(&self, alive_white: GcColor) {
        self.mark_color(alive_white)
    }

    fn mark_reachable(&self) {
        self.mark_color(GcColor::Black)
    }

    fn mark_untouched(&self) {
        self.mark_color(GcColor::Gray)
    }
}

impl<T: HeapMemUsed + TypeTag> Gc<T> {
    pub fn new(val: T) -> Gc<T> {
        let ptr = Box::into_raw(Box::new(WithGcHeader {
            gcheader: GcHeader::default(),
            inner: val,
        }));

        Gc {
            ptr: unsafe { NonNull::new_unchecked(ptr) },
        }
    }

    pub unsafe fn from_raw(ptr: *mut WithGcHeader<T>) -> Gc<T> {
        Gc {
            ptr: NonNull::new_unchecked(ptr),
        }
    }

    pub fn gch_ptr(&self) -> *const GcHeader {
        unsafe { self.ptr.as_ref().raw_gch() }
    }

    pub fn gch(&self) -> &GcHeader {
        unsafe { &*self.gch_ptr() }
    }

    // pub fn data_ptr(&self) -> *const T {
    //     unsafe { &self.ptr.as_ref().inner as *const T }
    // }

    /// Return heap address of GcHeader (not the object)
    pub fn heap_address(&self) -> usize {
        self.ptr.as_ptr() as usize
    }

    fn mark_color(&self, color: GcColor) {
        let wh = unsafe { self.ptr.as_ref() };
        wh.gcheader.color.set(color)
    }
}

impl PartialEq for Gc<StrImpl> {
    fn eq(&self, r: &Self) -> bool {
        // NOTE & TODO:
        // Gen hashid with a random seed to prevent hash collision attck
        if self.heap_address() == r.heap_address() {
            true
        } else {
            unsafe { self.ptr.as_ref().inner == r.ptr.as_ref().inner }
        }
    }
}

pub enum GcKind {
    Incremental,
    Generational,
}

#[derive(Clone, Copy, PartialEq)]
pub enum GcStage {
    Pause,
    Propgate,
    Sweep,
    Finalize,
}

impl From<usize> for GcStage {
    fn from(n: usize) -> Self {
        use GcStage::*;
        match n {
            0 => Pause,
            1 => Propgate,
            2 => Sweep,
            3 => Finalize,
            _ => unreachable!(),
        }
    }
}

impl From<GcStage> for usize {
    fn from(val: GcStage) -> Self {
        use GcStage::*;
        match val {
            Pause => 0,
            Propgate => 1,
            Sweep => 2,
            Finalize => 3,
        }
    }
}

impl GcStage {
    fn to_next_stage(&mut self) {
        use GcStage::*;
        *self = match self {
            Pause => Propgate,
            Propgate => Sweep,
            Sweep => Finalize,
            Finalize => Pause,
        }
    }
}

#[repr(u8)]
#[derive(Debug, Clone, Copy)]
enum ManagedObjKind {
    Str,
    Table,
    Function,
    UserData,
}

impl From<ManagedObjKind> for Tag {
    fn from(val: ManagedObjKind) -> Self {
        val as u8
    }
}

impl From<Tag> for ManagedObjKind {
    fn from(tag: Tag) -> Self {
        let map: [ManagedObjKind; 4] = [
            ManagedObjKind::Str,
            ManagedObjKind::Table,
            ManagedObjKind::Function,
            ManagedObjKind::UserData,
        ];
        map[tag as usize]
    }
}

#[derive(Default)]
struct IncrGcImpl {}

struct GenGcImpl {}

enum GcImpl {
    Incremental(IncrGcImpl),
    Generational(GenGcImpl),
}

pub struct ManagedHeap {
    enablegc: bool, // allow gc or not

    impls: GcImpl, // gc implementation
    curwhite: GcColor,
    stage: GcStage,
    allocs: BTreeSet<TaggedBox>,
    sstrpool: BTreeMap<StrHashVal, Gc<StrImpl>>,
    fixed: Vec<TaggedBox>,

    total: usize,
    debt: isize,
    estimate: usize,
}

impl Default for ManagedHeap {
    fn default() -> Self {
        let mut heap = ManagedHeap {
            enablegc: true,
            impls: GcImpl::Incremental(IncrGcImpl::default()),
            curwhite: GcColor::White,
            stage: GcStage::Pause,
            allocs: BTreeSet::new(),
            sstrpool: BTreeMap::new(),
            fixed: Vec::with_capacity(64),
            total: 0,
            debt: -1024,
            estimate: 0,
        };

        // add builtin string to string pool
        let builtin = ManagedHeap::TYPE_STRS
            .iter()
            .chain(Self::METATOPS_STRS.iter());

        builtin.for_each(|s| {
            let val = StrImpl::new_reserved(s);
            heap.fixed.push(val.into());
            heap.sstrpool.insert(val.hashid(), val);
            heap.total += val.heap_mem_used();
        });
        heap.estimate = heap.total;

        heap
    }
}

impl ManagedHeap {
    #[rustfmt::skip]
    const TYPE_STRS: [&str; 8] = [
        "number", "integer", "name", "string", 
        "userdata", "table", "function", "thread",
    ];

    #[rustfmt::skip]
    const METATOPS_STRS: [&str; 29]  = [
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

    fn switch_white(&mut self) {
        self.curwhite = match self.curwhite {
            GcColor::White => GcColor::AnotherWhite,
            GcColor::AnotherWhite => GcColor::White,
            _ => unreachable!(),
        };
    }

    fn is_cur_white(&self, obj: &GcHeader) -> bool {
        obj.color.get() == self.curwhite
    }

    fn record_mem_incr(&mut self, incr: usize) {
        self.total += incr;
        self.debt += incr as isize;
    }

    fn check_gc(&self) -> bool {
        self.debt > 0 && self.estimate < self.total / 2
    }

    fn full_gc(&mut self, set: &mut Rvm) {
        self.stage = GcStage::Pause;

        // mark all allocs object to gray
        for ptr in self.allocs.iter() {
            let gcobj = unsafe { (ptr.payload() as *mut WithGcHeader<()>).as_ref().unwrap() };
            let gch = &gcobj.gcheader;
            gch.color.set(GcColor::Gray);
        }

        // mark all reachable object to black
        set.stack.iter().for_each(|val| val.mark_reachable());
        set.globaltab.mark_reachable();

        // sweep and collect remained object
        let (remain, release) = std::mem::take(&mut self.allocs)
            .into_iter()
            .map(|tp| (tp.tag(), tp.payload()))
            .fold(
                (BTreeSet::<TaggedBox>::new(), 0),
                |mut acc: (BTreeSet<TaggedBox>, usize), (tag, payload): (Tag, u64)| {
                    let gcobj = unsafe { (payload as *mut WithGcHeader<()>).as_ref().unwrap() };
                    let gch = &gcobj.gcheader;
                    if gch.color.get() == GcColor::Gray {
                        // release
                        let freed = match tag.into() {
                            ManagedObjKind::Str => {
                                let val = Gc::from(payload as *mut WithGcHeader<StrImpl>);
                                self.sstrpool.remove(&val.hashid());
                                let used = val.heap_mem_used();
                                StrImpl::drop(val);
                                used
                            }
                            ManagedObjKind::Table => unsafe {
                                let tmp = payload as *mut WithGcHeader<TableImpl>;
                                let val = Gc::from(tmp);

                                // TODO:
                                // finalize
                                // self.finalize();

                                let used = val.heap_mem_used();
                                let _ = Box::from_raw(tmp);
                                used
                            },
                            ManagedObjKind::Function => unsafe {
                                let tmp = payload as *mut WithGcHeader<Proto>;
                                let val = Gc::from(tmp);
                                let used = val.heap_mem_used();
                                let _ = Box::from_raw(tmp);
                                used
                            },
                            ManagedObjKind::UserData => unsafe {
                                let tmp = payload as *mut WithGcHeader<UserDataImpl>;
                                let val = Gc::from(tmp);
                                let used = val.heap_mem_used();
                                let _ = Box::from_raw(tmp);
                                used
                            },
                        };
                        (acc.0, acc.1 + freed)
                    } else {
                        acc.0.insert(TaggedBox::new(tag, payload as usize));
                        (acc.0, acc.1)
                    }
                },
            );

        let _ = std::mem::replace(&mut self.allocs, remain);

        // update gc infomation
        self.total -= release;
        self.estimate = self.total;
    }

    fn single_step_gc(&mut self, _set: &mut Rvm, _mannual: bool) {
        todo!("single step gc")
        // self.stage = GcStage::Pause;
        // self.switch_white();

        // // mark rootset
        // self.mark_rootset(set);

        // // mark fixed
        // self.mark_fixed();

        // // mark allocs
        // self.mark_allocs();

        // // sweep
        // self.sweep();

        // // finalize
        // self.finalize();

        // // estimate
        // self.estimate = self.total / 2;
    }
}

pub struct Heap<'h> {
    rootset: &'h mut Rvm,
    heap: &'h mut ManagedHeap,
}

impl Deref for Heap<'_> {
    type Target = ManagedHeap;

    fn deref(&self) -> &Self::Target {
        self.heap
    }
}

impl DerefMut for Heap<'_> {
    fn deref_mut(&mut self) -> &mut Self::Target {
        self.heap
    }
}

impl Heap<'_> {
    pub fn new<'h>(rootset: &'h mut Rvm, heap: &'h mut ManagedHeap) -> Heap<'h> {
        Heap { rootset, heap }
    }

    pub fn delegate(&mut self, gcval: &mut LValue) {
        match gcval {
            LValue::String(new) => {
                if new.is_short() {
                    let hash = new.hashid();
                    if self.sstrpool.contains_key(&hash) {
                        // SAFETY: key has been checked above
                        let old = unsafe { self.sstrpool.get(&hash).unwrap_unchecked() };
                        if old.heap_address() != new.heap_address() {
                            // release new str and use old one
                            old.mark_newborned(self.curwhite);
                            StrImpl::drop(std::mem::replace(new, old.clone()));
                        }
                    } else {
                        self.delegate_from(*new);
                        self.sstrpool.insert(hash, *new);
                    }
                } else {
                    self.delegate_from(*new)
                }
            }

            LValue::Table(tb) => self.delegate_from(*tb),
            LValue::Function(f) => {
                self.delegate_from(*f);
                Proto::delegate_to(f, self);
            }
            LValue::UserData(ud) => self.delegate_from(*ud),

            _ => {}
        }
    }

    pub fn delegate_from<T: TypeTag + HeapMemUsed>(&mut self, gcptr: Gc<T>) {
        debug_assert_eq!(gcptr.gch().color.get(), GcColor::Wild);
        debug_assert_eq!(gcptr.gch().age.get(), 0);
        self.allocs.insert(gcptr.into());
        gcptr.mark_newborned(self.curwhite);
        self.record_mem_incr(gcptr.heap_mem_used());
    }

    pub fn alloc<S, D>(&mut self, data: S) -> LValue
    where
        D: TypeTag + HeapMemUsed,
        Gc<D>: From<S>,
        LValue: From<Gc<D>>,
    {
        let mut val = LValue::from(data.into());
        self.delegate(&mut val);
        self.gc_checkpoint();
        val
    }

    // /// Alloc UserData in single api because it's size should be added to gc state.
    // fn alloc_userdata<T>(&mut self, zeroed: bool) -> LValue {
    //     // unsafe {
    //     //     let lo = Layout::new::<WithGcHeader<T>>();
    //     //     let data = if zeroed {
    //     //         std::alloc::alloc_zeroed(lo)
    //     //     } else {
    //     //         std::alloc::alloc(lo)
    //     //     } as *mut WithGcHeader<UserDataImpl>;

    //     //     let whith_gch = data.as_mut().unwrap_unchecked();
    //     //     whith_gch.gcheader.color.set(self.curwhite);
    //     //     whith_gch.gcheader.age.set(0);
    //     //     whith_gch.inner.size = std::mem::size_of::<T>();
    //     //     let gc = Gc::from_raw(whith_gch);

    //     //     self.delegate_from(gc);
    //     //     self.record_mem_incr(lo.size());
    //     //     self.check_gc();

    //     //     LValue::UserData(gc)
    //     // }
    //     todo!()
    // }

    fn gc_checkpoint(&mut self) {
        if self.heap.enablegc && self.heap.check_gc() {
            self.collect_garbage(true);
        }
    }

    pub fn collect_garbage(&mut self, full: bool) {
        if full {
            self.heap.full_gc(self.rootset);
        } else {
            self.heap.single_step_gc(self.rootset, true);
        }
    }
}

mod platform_check {
    #[test]
    fn x64_os_check() {
        assert!(cfg!(target_pointer_width = "64"))
    }

    #[test]
    fn constants_check() {
        use super::TaggedBox;

        assert!(TaggedBox::PAYLOAD_MASK & TaggedBox::TAG_MASK == 0);
        assert!(TaggedBox::PAYLOAD_MASK | TaggedBox::TAG_MASK == u64::MAX);
    }
}

mod gc_check {

    #[test]
    fn trig_gull_gc_simple() {
        use crate::heap::HeapMemUsed;
        use crate::state::State;

        let mut state = State::new();

        for _ in 0..=1 {
            let old = state.heap_view().total;
            let long_str = ".".repeat(1024);
            let cloned = long_str.clone();

            // create a long string on managed heap to trig full GC
            let mut incr = 0;
            state
                .context(|vm, heap| {
                    let val = heap.alloc(long_str);
                    incr = val.heap_mem_used();
                    vm.stk_push(val)
                })
                .unwrap();

            let new = state.heap_view().total;
            assert_eq!(old + incr, new);

            let val = state.vm_view().stk_pop().unwrap();
            assert_eq!(val.as_string().unwrap().as_str(), cloned.as_str());

            // trig full gc
            state.heap_view().collect_garbage(true);

            assert_eq!(old, state.heap_view().total);

            std::mem::forget(cloned);
        }
    }
}
