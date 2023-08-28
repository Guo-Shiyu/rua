use std::collections::{BTreeMap, LinkedList};

use std::convert::From;
use std::ops::{Deref, DerefMut, Index, IndexMut};
use std::ptr::NonNull;

use crate::codegen::{CodeGen, Instruction, Proto};
use crate::ffi::StdLib;
use crate::parser::{Parser, SyntaxError};
use crate::value::{EstimatedSize, UserData};
use crate::StaticErr;

use super::value::{
    GcColor, GcMark, GcObject, LValue, RsFunc, StrHashVal, StrWrapper, Table, Tag, TaggedBox,
    WithTag,
};
use super::LuaErr;

pub struct VmStack {
    stk: Vec<LValue>,
    top: u32,
}

impl VmStack {
    pub const MIN_STACK_SIZE: usize = 32;
    pub const MAX_STACK_SIZE: usize = 1024;

    pub fn init() -> Self {
        let mut stk = Vec::new();
        stk.resize_with(Self::MIN_STACK_SIZE, || LValue::Nil);
        VmStack { stk, top: 0 }
    }

    pub fn top(&mut self) -> i64 {
        // TODO:
        self.stk.last().unwrap().as_int().unwrap()
    }

    pub fn ati(&self, i: i32) -> LValue {
        self.stk[i as usize].clone()
    }

    pub fn seti(&mut self, i: i32, val: LValue) {
        self.stk[i as usize] = val;
    }
}

impl Deref for VmStack {
    type Target = Vec<LValue>;

    fn deref(&self) -> &Self::Target {
        &self.stk
    }
}

impl DerefMut for VmStack {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.stk
    }
}

#[derive(Clone, Copy)]
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

impl Into<usize> for GcStage {
    fn into(self) -> usize {
        use GcStage::*;
        match self {
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
enum ManagedObjKind {
    Str,
    Table,
    Function,
    UserData,
}

impl Into<Tag> for ManagedObjKind {
    fn into(self) -> Tag {
        self as u8
    }
}

impl From<Tag> for ManagedObjKind {
    fn from(tag: Tag) -> Self {
        let map: [u8; 4] = [
            StrWrapper::tagid(),
            Table::tagid(),
            Proto::tagid(),
            UserData::tagid(),
        ];
        map[tag as usize].into()
    }
}

pub struct Frame<'f> {
    oldpc: u32, // previous frame pc state
    pc: u32,    // current frame pc

    reg_base: i32, // first avaialbe register
    reg_top: i32,  // last available register + 1

    upvalues: &'f [LValue],       // refer to proto upvalues
    constants: &'f [LValue],      // refer to proto constants
    bytescode: &'f [Instruction], // ref to proto bytescode
}

impl<'f> Frame<'f> {
    pub const LVALUE_PLACE_HOLDER: Vec<LValue> = Vec::new();
    pub const BYTECODE_PLACE_HOLDER: Vec<Instruction> = Vec::new();

    // First frame of a vm (rc call lua)
    pub fn init() -> Self {
        Frame {
            oldpc: u32::MAX,
            pc: 0,
            reg_base: 0,
            reg_top: 0,
            upvalues: &[],
            constants: &[],
            bytescode: &[],
        }
    }

    pub fn is_init_frame(&self) -> bool {
        self.oldpc == u32::MAX
    }

    pub fn nregs(&self) -> usize {
        (self.reg_top - self.reg_base) as usize
    }
}

pub struct DebugPoint {}

type HookFn = fn(&mut State, &DebugPoint);

pub enum HookMask {
    One,
    Two,
    Three,
}

pub enum GcKind {
    Incremental,
    Generational,
}

#[derive(Debug)]
pub enum RuntimeErr {
    TableIndexIsNil,
    WrongCallFuncValue { typ: String, name: String },
}

pub struct Stack<'s> {
    stkref: &'s mut VmStack,
}

impl<'s> Stack<'s> {
    pub fn top(&self) -> usize {
        self.stkref.top as usize
    }

    pub fn push(&mut self, val: LValue) {
        self.stkref.top += 1;
        let idx = self.stkref.top as usize;
        self.stkref[idx] = val;
    }

    // pub fn pop(&mut self) -> LValue {
    //     self.stkref.pop().unwrap_or_default()
    // }

    pub fn last(&self) -> LValue {
        self.stkref[self.stkref.top as usize].clone()
    }

    pub fn get(&self, idx: i32) -> LValue {
        self.stkref[idx as usize].clone()
    }
}

pub struct Hook<'k> {
    hook: &'k HookFn,
    mask: &'k HookMask,
}

pub struct Heap<'h> {
    sstrpool: &'h mut BTreeMap<StrHashVal, NonNull<StrWrapper>>,
    curwhite: &'h mut GcColor,
    curstage: &'h mut GcStage,
    incr: &'h mut usize,                  // incremental mem size
    threshold: &'h mut usize,             // threshold for next gc
    total: &'h mut usize,                 // total mem size
    gckind: &'h mut GcKind,               // which kind in using
    allocs: &'h mut Vec<TaggedBox>,       // other managed object
    stralloc: &'h mut Vec<*const GcMark>, // strings are pooled in a separate set
}

impl<'h> Heap<'h> {
    pub fn delegate(&mut self, gcval: LValue) {
        match gcval {
            // strings are pooled in a separate set
            LValue::String(mut sw) => {
                let new = unsafe { sw.as_mut() };

                let need_pool = new.is_long() || {
                    let hash = new.hashid();
                    if self.sstrpool.contains_key(&hash) {
                        let old = self.sstrpool.get(&hash).unwrap();
                        unsafe { old.as_ref().mark_newborned(*self.curwhite) };
                        if unsafe { old.as_ref().as_str().as_ptr() } != new.as_str().as_ptr() {
                            new.free_short();
                        }
                        false
                    } else {
                        true
                    }
                };

                if need_pool {
                    self.stralloc.push(sw.as_ptr().cast::<GcMark>());
                    unsafe { sw.as_ref().mark_newborned(*self.curwhite) };
                }
            }

            LValue::Table(tb) => self.delegate_from(tb),
            LValue::Function(f) => {
                self.delegate_from(f);

                // SAFETY ?
                unsafe { f.as_ref().delegat_by(self) };
            }
            LValue::UserData(ud) => self.delegate_from(ud),

            _ => {}
        }
    }

    fn delegate_from<T: GcObject>(&mut self, v: NonNull<T>) {
        let tp = TaggedBox::from_heap_ptr(T::tagid(), v.as_ptr());
        self.allocs.push(tp);

        // SAFETY ?
        unsafe { v.as_ref().mark_newborned(*self.curwhite) };
    }

    fn delegate_str(&mut self, s: String) {
        if s.len() > StrWrapper::MAX_SHORT_LEN {
            let sw = StrWrapper::from_long(s);
            let nnsw = unsafe { NonNull::new_unchecked(Box::leak(Box::new(sw))) };
            self.delegate(LValue::String(nnsw));
        } else {
            let hash = StrWrapper::hash_short(&s);
            if !self.sstrpool.contains_key(&hash) {
                let sw = StrWrapper::from_short(s, Some(hash), false);
                let nnsw = unsafe { NonNull::new_unchecked(Box::leak(Box::new(sw))) };
                self.delegate(LValue::String(nnsw));
            } else {
                // TODO:
                // frees.push (s.leak())
                // let k = Box::leak(s.into_boxed_str());
            }
        }
    }

    fn alloc<O>(&mut self, content: O) -> LValue
    where
        LValue: From<O>,
    {
        let val = LValue::from(content);
        self.delegate(val.clone());
        *self.total += val.estimate_size();
        self.check_gc();
        val
    }

    /// Alloc UserData in single api because it's size should be added to gc state.
    fn alloc_userdata(&mut self, bytes: usize) -> LValue {
        let userdata = Vec::<u8>::with_capacity(bytes).as_ptr().cast::<UserData>();

        let nonnull = unsafe {
            userdata.clone().read().mark_newborned(*self.curwhite);
            NonNull::new(userdata.cast_mut()).unwrap_unchecked()
        };

        let val = LValue::UserData(nonnull);
        self.delegate(val.clone());
        *self.total += bytes;
        self.check_gc();

        val
    }

    // If gc is needed, run single step gc
    fn check_gc(&mut self) -> bool {
        if self.total >= self.threshold {
            self.gc_single_step();
            true
        } else {
            false
        }
    }

    fn gc_single_step(&mut self) {
        // match self.stage {
        //     GcStage::Pause => self.gc_restart(),
        //     GcStage::Propgate => {
        //         self.stage.to_next_stage();
        //     }
        //     GcStage::Sweep => {
        //         self.stage.to_next_stage();
        //     }
        //     GcStage::Finalize => {
        //         self.stage.to_next_stage();
        //     }
        // }
    }

    /// Mark all reachable objects from root set and add them to gray list
    fn gc_restart(&mut self) {
        // global table

        // stack

        // mainthread
    }
}

// enum LoadMode {
//     Auto,   // auto detect mode by input
//     Text,
//     Binary,
// }

pub struct State<'a> {
    allowhook: bool, // weither enable hooks

    stack: VmStack,                   // vm stack
    callchain: LinkedList<Frame<'a>>, // call frames

    glotab: Table,

    sstrpool: BTreeMap<StrHashVal, NonNull<StrWrapper>>, // short string pool

    // gc state
    using_white: GcColor, // GcMark::White or GcMark::AnotherWhite
    stage: GcStage,

    // gc param
    incr: usize,      // incremental mem size
    threshold: usize, // threshold for next gc
    total: usize,     // total mem size
    gckind: GcKind,   // which kind in using

    // gc object marks
    allocs: Vec<TaggedBox>,       // other managed object
    stralloc: Vec<*const GcMark>, // strings are pooled in a separate set

    mematbs: Vec<Table>, // meta table used for basic types, indexed by LValue.tagid

    warnfn: Option<RsFunc>,  // warn handler
    panicfn: Option<RsFunc>, // panic handler

    hook: HookFn,       // hook function
    hookmask: HookMask, // hook mask

    _unused: bool,
    // TODO:
    // executor for coroutine

    // TODO:
    // BinaryHeap
    // frees: BTreeMap<usize, (usize, *const char)>, // an ordered list of free memory
}

impl<'a> State<'a> {
    pub fn stack_view(&mut self) -> Stack {
        Stack {
            stkref: &mut self.stack,
        }
    }

    pub fn heap_view(&mut self) -> Heap {
        Heap {
            sstrpool: &mut self.sstrpool,
            curwhite: &mut self.using_white,
            curstage: &mut self.stage,
            incr: &mut self.incr,
            threshold: &mut self.threshold,
            total: &mut self.total,
            gckind: &mut self.gckind,
            allocs: &mut self.allocs,
            stralloc: &mut self.stralloc,
        }
    }

    pub fn hook_view(&mut self) -> Hook {
        Hook {
            hook: &self.hook,
            mask: &self.hookmask,
        }
    }

    pub fn stack<O>(&mut self, op: O) -> &Self
    where
        O: FnOnce(&mut Stack, &mut Heap),
    {
        op(
            &mut Stack {
                stkref: &mut self.stack,
            },
            &mut Heap {
                sstrpool: &mut self.sstrpool,
                curwhite: &mut self.using_white,
                curstage: &mut self.stage,
                incr: &mut self.incr,
                threshold: &mut self.threshold,
                total: &mut self.total,
                gckind: &mut self.gckind,
                allocs: &mut self.allocs,
                stralloc: &mut self.stralloc,
            },
        );
        self
    }

    pub fn heap<O>(&mut self, op: O) -> &Self
    where
        O: FnOnce(&mut Heap) -> (),
    {
        op(&mut self.heap_view());
        self
    }
}

// method about GC
impl State<'_> {
    pub fn new() -> Self {
        let mut state = State {
            allowhook: false,
            stack: VmStack::init(),
            callchain: LinkedList::new(),
            glotab: Table::default(),
            sstrpool: BTreeMap::new(),
            using_white: GcColor::White,
            stage: GcStage::Pause,
            incr: 0,
            threshold: 0,
            total: 0,
            gckind: GcKind::Incremental,
            allocs: Vec::new(),
            stralloc: Vec::new(),
            mematbs: Vec::new(),
            warnfn: None,
            panicfn: None,
            hook: |_, _| {},
            hookmask: HookMask::One,
            _unused: false,
        };

        // init first frame, rust call lua entry
        state.callchain.push_back(Frame::init());

        // init meta table for basic types
        // state.mematbs.resize(ManagedObjKind::UserData.into() + 1, Table::new());
        state
    }

    pub fn open(&mut self, lib: StdLib) {
        let entrys = crate::ffi::get_std_libs(lib);
        for (name, func) in entrys.iter() {
            // SAFETY: function names in std lib is not nil
            unsafe { self.set_func(name, *func).unwrap_unchecked() };
        }
    }

    pub fn open_libs(&mut self, libs: &[StdLib]) {
        for lib in libs.iter() {
            self.open(*lib);
        }
    }

    // pub fn safe_script()

    // Return the retval count of the script, and the val has pushed on the stack in return order.
    pub fn script(&mut self, src: &str) -> Result<usize, LuaErr> {
        let preidx = self.stack_view().top();

        self.load_str(src, Some("main".to_string()))
            .map_err(LuaErr::CompileErr)?;

        self.do_call(None)?;

        let postidx = self.stack_view().top();
        Ok(postidx - preidx)
    }

    pub fn script_file(&mut self, _filepath: &str) -> Result<(), LuaErr> {
        Ok(())
    }

    /// Loads a Lua chunk without running it. If there are no errors, `load` pushes the compiled chunk as a Lua function on top of the stack.
    pub fn load_str(&mut self, src: &str, chunkname: Option<String>) -> Result<(), StaticErr> {
        let proto = Self::compile(src, chunkname)?;
        let fval = self.heap_view().alloc(proto);
        self.stack_view().push(fval.clone());
        Ok(())
    }

    pub fn load_file(&mut self, _filepath: &str) -> Result<(), LuaErr> {
        Ok(())
    }

    // pub fn call() -> Result<(), LuaErr> {
    //     todo!()
    // }

    pub fn do_call(&mut self, fnidx: Option<i32>) -> Result<(), RuntimeErr> {
        let f = if let Some(idx) = fnidx {
            self.stack_view().get(idx)
        } else {
            self.stack_view().last()
        };
        match f {
            LValue::RsFn(f) => {
                f(self)?;
            }
            LValue::Function(pptr) => {
                let proto = unsafe { pptr.as_ref() };
                debug_assert!(self.stack.len() - self.stack_view().top() >= proto.nparams());

                let base_reg = self.stack_view().top() - proto.nparams() - 1; // fn is a slot also
                let prev_frame = self.callchain.back().unwrap(); // unchecked
                self.callchain.push_back(Frame {
                    oldpc: prev_frame.pc,
                    pc: 0,
                    reg_base: base_reg as i32,
                    reg_top: (base_reg + proto.reg_count()) as i32,
                    upvalues: &[],
                    constants: proto.constants(),
                    bytescode: proto.bytecode(),
                });

                while self.callchain.back().unwrap().pc < proto.bytecode().len() as u32 {
                    let code = proto.bytecode()[self.callchain.back().unwrap().pc as usize];
                    self.interpret(code)?;
                }
            }
            LValue::Table(_) => todo!("get metatable __call"),
            LValue::UserData(_) => todo!(),

            LValue::Nil => {
                println!("try to call a nil value");
                todo!("error report")
            }
            LValue::Bool(_) => {
                println!("try to call a bool value");
                todo!("error report")
            }
            LValue::Int(_) => {
                println!("try to call a int value");
                todo!("error report")
            }
            LValue::Float(_) => {
                println!("try to call a float value");
                todo!("error report")
            }
            LValue::String(sw) => unsafe {
                println!("try to call a string value: {}", sw.as_ref().as_str())
            },
        }
        Ok(())
    }

    pub fn compile(src: &str, chunkname: Option<String>) -> Result<Proto, StaticErr> {
        let block = Parser::parse(src, chunkname)?;

        // TODO:
        // optimizer scheduler
        // let mut cfp = ConstantFoldPass::new();
        // assert!(cfp.walk(&mut block).is_ok())

        let proto = CodeGen::generate(block, false)?;
        Ok(proto)
    }

    pub fn set_func(&mut self, name: &str, func: RsFunc) -> Result<(), RuntimeErr> {
        let delegated = self.heap_view().alloc(name.to_string());
        self.glotab.set(&delegated, LValue::RsFn(func))?;
        Ok(())
    }

    pub fn set_warn_handler(&mut self, warnfn: RsFunc) {
        self.warnfn = Some(warnfn);
    }

    pub fn set_panic_handler(&mut self, panicfn: RsFunc) {
        self.panicfn = Some(panicfn);
    }

    pub fn set_hook_handler(&mut self, hook: HookFn, mask: HookMask) {
        self.hook = hook;
        self.hookmask = mask;
    }

    pub fn set_gc_threshold(&mut self, threshold: usize) {
        self.threshold = threshold;
    }

    pub fn set_gc_kind(&mut self, kind: GcKind) {
        self.gckind = kind;
    }

    pub fn set_gc_incr(&mut self, incr: usize) {
        self.incr = incr;
    }

    // pub fn set_hook(&mut self, allow: bool) {
    //     self.allowhook = allow;
    // }

    fn pre_call(&mut self) {}

    fn interpret(&mut self, code: Instruction) -> Result<(), RuntimeErr> {
        println!("{:?}", code);
        use super::codegen::OpCode::*;
        use super::codegen::OpMode;
        match code.mode() {
            OpMode::IABC => {
                let (op, a, b, c, k) = code.repr_abck();
                match op {
                    VARARGPREP => {
                        // TODO:
                        // do nothing for now
                    }
                    LOADK => {
                        let mut fa = self.callchain.back_mut().unwrap();
                        self.stack
                            .seti(fa.reg_base + a, fa.constants[b as usize].clone());
                    }

                    GETTABUP => {
                        // let upval = fa.upvalues[c as usize];
                        // let key = fa.constants[b as usize].clone();
                        // let val = upval.get(&key)?;
                        // let val = ;
                        let mut fa = self.callchain.back_mut().unwrap();
                        self.stack.seti(
                            fa.reg_base + a,
                            self.glotab.find_by_key(&fa.constants[c as usize])?,
                        );
                    }

                    MOVE => {
                        let mut fa = self.callchain.back_mut().unwrap();
                        self.stack
                            .seti(fa.reg_base + a, self.stack.ati(fa.reg_base + b));
                    }
                    CALL => {
                        let mut fa = self.callchain.back_mut().unwrap();
                        let dest_idx = fa.reg_base + a;
                        self.do_call(Some(dest_idx));
                    }
                    _ => unimplemented!("unimplemented opcode: {:?}", op),
                }
            }
            OpMode::IABx => todo!(),
            OpMode::IAsBx => todo!(),
            OpMode::IAx => todo!(),
            OpMode::IsJ => todo!(),
        }

        // TODO:
        let mut fa = self.callchain.back_mut().unwrap();
        fa.pc += 1;

        Ok(())
    }
}
