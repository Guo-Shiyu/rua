use std::{
    cell::Cell,
    collections::{BTreeMap, BTreeSet},
    fmt::Debug,
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
        ((self.raw_repr() & Self::TAG_MASK) >> 48) as u8
    }

    pub fn payload(&self) -> u64 {
        self.raw_repr() & Self::PAYLOAD_MASK
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
#[repr(u8)]
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

pub trait HeapMemUsed {
    fn heap_mem_used(&self) -> usize;
}

pub trait MarkAndSweepGcOps {
    fn mark_newborned(&self, _white: GcColor) {}
    fn mark_reachable(&self) {}
    fn mark_untouched(&self) {}
    fn delegate_to(&mut self, _heap: &mut Heap) {}
}

impl MarkAndSweepGcOps for () {}
impl HeapMemUsed for () {
    fn heap_mem_used(&self) -> usize {
        unreachable!()
    }
}

#[repr(C)]
pub struct WithGcHeader<T: MarkAndSweepGcOps + HeapMemUsed> {
    pub gcheader: GcHeader,
    pub inner: T,
}

impl<T: MarkAndSweepGcOps + HeapMemUsed> MarkAndSweepGcOps for WithGcHeader<T> {
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
    fn mark_untouched(&self) {
        self.gcheader.color.set(GcColor::Gray);
        self.inner.mark_untouched()
    }

    fn delegate_to(&mut self, heap: &mut Heap) {
        self.inner.delegate_to(heap)
    }
}

impl<T: MarkAndSweepGcOps + HeapMemUsed> HeapMemUsed for WithGcHeader<T> {
    fn heap_mem_used(&self) -> usize {
        std::mem::size_of::<Self>() + self.inner.heap_mem_used()
    }
}

pub trait GcObject: MarkAndSweepGcOps + HeapMemUsed + TypeTag {}

impl<T: MarkAndSweepGcOps + HeapMemUsed + TypeTag> GcObject for T {}

/// Pointer type for gc managed objects
#[derive(Debug)]
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

impl<T: GcObject> From<Gc<T>> for TaggedBox {
    fn from(val: Gc<T>) -> Self {
        TaggedBox::from_heap_ptr(T::tagid(), val.ptr.as_ptr())
    }
}

impl<T: GcObject> MarkAndSweepGcOps for Gc<T> {
    fn delegate_to(&mut self, heap: &mut Heap) {
        debug_assert_eq!(self.header().color.get(), GcColor::Wild);
        debug_assert_eq!(self.header().age.get(), 0);
        heap.allocs.insert(Into::<TaggedBox>::into(*self));
        self.mark_newborned(heap.curwhite);
        heap.record_mem_incr(self.heap_mem_used());
    }
}

impl<T: GcObject> HeapMemUsed for Gc<T> {
    fn heap_mem_used(&self) -> usize {
        unsafe { self.ptr.as_ref().heap_mem_used() }
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

impl<T: GcObject> Gc<T> {
    pub fn take(mut val: Gc<T>) -> Box<WithGcHeader<T>> {
        unsafe { Box::from_raw(val.ptr.as_mut()) }
    }

    pub fn drop(val: Gc<T>) {
        let _ = Self::take(val);
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

    // Return heap address of GcHeader (not the object)
    pub fn heap_address(&self) -> usize {
        self.ptr.as_ptr() as usize
    }

    fn header(&self) -> &GcHeader {
        unsafe { &self.ptr.as_ref().gcheader }
    }

    // fn mark_color(&self, color: GcColor) {
    //     let wh = unsafe { self.ptr.as_ref() };
    //     wh.gcheader.color.set(color)
    // }
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
        ManagedHeap {
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
        }
    }
}

impl ManagedHeap {
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

        // mark all allocs object to gray.
        // skip new borned obj just now.
        for ptr in self.allocs.iter() {
            let gcobj = unsafe { (ptr.payload() as *mut WithGcHeader<()>).as_ref().unwrap() };
            let gch = &gcobj.gcheader;
            gch.color.set(GcColor::Gray);
        }

        // mark all reachable object to black
        set.stack.iter().for_each(|val| val.mark_reachable());
        set.globaltab.mark_reachable();

        // sweep and collect remained object
        let (remain, release) = std::mem::take(&mut self.allocs).into_iter().fold(
            (BTreeSet::new(), 0),
            |mut acc: (_, usize), tp| {
                let _tag = tp.tag();
                let gch = &unsafe { tp.as_ptr::<WithGcHeader<()>>().as_ref() }
                    .unwrap()
                    .gcheader;

                if gch.color.get() == GcColor::Gray {
                    (acc.0, acc.1 + self.free_garbage(tp))
                } else {
                    acc.0.insert(tp);
                    (acc.0, acc.1)
                }
            },
        );

        self.allocs = remain;

        // update gc infomation
        self.total -= release;
        self.estimate = self.total;
    }

    fn free_garbage(&mut self, tbox: TaggedBox) -> usize {
        // release

        match tbox.tag().into() {
            ManagedObjKind::Str => Self::drop_garbage_with::<StrImpl, _>(tbox, |s| {
                self.sstrpool.remove(&s.hashid());
            }),
            ManagedObjKind::Table => {
                Self::drop_garbage_with::<TableImpl, _>(tbox, |_tab| {
                    // TODO:
                    // finalize
                    // self.finalize();
                })
            }
            ManagedObjKind::Function => Self::drop_garbage_with::<Proto, _>(tbox, |_| {}),
            ManagedObjKind::UserData => Self::drop_garbage_with::<UserDataImpl, _>(tbox, |_| {}),
        }
    }

    fn drop_garbage_with<T: GcObject, F: FnOnce(Gc<T>)>(taggedptr: TaggedBox, f: F) -> usize
    where
        LValue: From<Gc<T>>,
    {
        let val = Gc::from(taggedptr.as_mut::<WithGcHeader<T>>());
        f(val);

        // {
        //     println!(
        //         "Drop Object at: 0x{:X} , free mem size: {},  val: {}",
        //         val.heap_address(),
        //         val.heap_mem_used(),
        //         LValue::from(val)
        //     );
        // }

        let used = val.heap_mem_used();
        Gc::drop(val);
        used
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
                    let contains = self.sstrpool.contains_key(&hash);
                    if contains {
                        // SAFETY: key has been checked above
                        let old = unsafe { self.sstrpool.get(&hash).unwrap_unchecked() };

                        if old.heap_address() != new.heap_address() {
                            // release new str and use old one
                            old.mark_newborned(self.curwhite);
                            Gc::drop(*new);
                            *gcval = (*old).into();
                        }
                    } else {
                        new.delegate_to(self);
                        self.sstrpool.insert(hash, *new);
                    }
                } else {
                    new.delegate_to(self);
                }
            }

            LValue::Table(mut tb) => {
                tb.delegate_to(self);
            }
            LValue::Function(mut f) => f.delegate_to(self),
            LValue::UserData(ud) => ud.delegate_to(self),

            _ => {}
        };
    }

    pub fn alloc<S, D>(&mut self, data: S) -> LValue
    where
        D: GcObject,
        Gc<D>: From<S>,
        LValue: From<Gc<D>>,
    {
        let mut val = LValue::from(data.into());
        self.delegate(&mut val);
        self.gc_checkpoint();
        val
    }

    pub fn alloc_fixed(&mut self, s: &str) -> LValue {
        let val: Gc<StrImpl> = s.into();

        if val.is_short() {
            self.sstrpool.insert(val.hashid(), val);
        }
        self.fixed.push(val.into());

        LValue::from(val)
    }

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

    // This method will be called when State was dropped.
    // Release all managed object in heap.
    pub fn destroy(&mut self) {
        self.collect_garbage(true);

        let iters = std::mem::take(&mut self.allocs)
            .into_iter()
            .chain(std::mem::take(&mut self.fixed));

        let _ = iters.fold(0, |acc, obj| acc + self.free_garbage(obj));

        // println!("destroy the heap ... total realesed memory: {}", freed);
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

        for n in 1..=5 {
            let long_str = ".".repeat(64 * n);
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

            let val = state.vm_view().stk_pop().unwrap();
            assert_eq!(val.as_str().unwrap(), cloned.as_str());

            // trig full gc
            state.heap_view().collect_garbage(true);
        }
    }
}
