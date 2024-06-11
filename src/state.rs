use std::{
    cmp::Ordering,
    collections::BTreeMap,
    fs::File,
    io::BufReader,
    ops::{Deref, DerefMut},
    path::Path,
    ptr,
};

use crate::{
    ast::Block,
    codegen::{ChunkDumper, CodeGen, Instruction, OpCode, OpMode, Proto, UpvalDecl},
    ffi::{self, Stdlib},
    heap::{Gc, GcOp, Heap, LuaClosure, MetaOperator, Table, Tag, UpVal},
    parser::Parser,
    passes,
    value::Value,
    InterpretError,
};

pub type RegIndex = i32;
type Slot = Value;

/// CallInfo
pub struct Frame {
    pc: i32,   // current frame pc
    nret: i32, // how many returns

    func: *mut Slot, // function register index, (first avaliable register index - 1)
    slotend: *mut Slot, // the last available slots

    up_ptr: *mut UpVal, // refer to closure upvalues
    uplen: i32,

    kst_ptr: *const Value, // refer to proto constants
    kstlen: i32,

    code_ptr: *const Instruction, // ref to proto bytescode
    codelen: i32,
}

impl Frame {
    const INIT_FRAME_PC: i32 = i32::MAX;
    const LUA_MULTI_RET: i32 = i32::MAX;

    /// Get next instruction and increase pc counter with 1.
    fn fetch(&mut self) -> Instruction {
        debug_assert!(self.pc < self.codelen);
        let isc = unsafe { *self.code_ptr.offset(self.pc as isize) };
        self.pc += 1;
        isc
    }

    fn kget(&self, idx: RegIndex) -> Value {
        debug_assert!(idx <= self.kstlen);
        unsafe { self.kst_ptr.offset(idx as isize).read() }
    }

    fn upget(&self, idx: RegIndex) -> UpVal {
        debug_assert!(idx <= self.uplen);
        unsafe { self.up_ptr.offset(idx as isize).read() }
    }

    /// slot pointer in current frame
    fn slot_ptr_of(&self, idx: RegIndex) -> *mut Slot {
        debug_assert!(idx >= 0);
        unsafe {
            // FIXME: remove the condition
            if idx < 0 {
                self.slotend.sub(idx.unsigned_abs() as usize)
            } else {
                self.func.add(idx as usize + 1) // +1 to convert to register index
            }
        }
    }

    fn access<T>(
        &self,
        idx: RegIndex,
        op: impl FnOnce(*mut Slot) -> T,
    ) -> Result<T, InterpretError> {
        let ptr = self.slot_ptr_of(idx);
        if cfg!(debug_assertions) && (ptr > self.slotend || ptr <= self.func) {
            return Err(InterpretError::InvalidRegisterAccess {
                target: idx,
                max: self.max_reg_idx(),
            });
        }
        Ok(op(ptr))
    }

    fn rget(&self, idx: RegIndex) -> Result<Value, InterpretError> {
        self.access(idx, |v| unsafe { *v })
    }

    fn rset<V>(&self, idx: RegIndex, val: V) -> Result<(), InterpretError>
    where
        Value: From<V>,
    {
        self.access(idx, |v| unsafe { *v = Value::from(val) })
    }

    fn rkget(&mut self, k: bool, reg: i32) -> Result<Value, InterpretError> {
        Ok(if k { self.kget(reg) } else { self.rget(reg)? })
    }

    fn slot_iter(&self) -> StackIter {
        StackIter {
            begin: self.func,
            end: self.slotend,
        }
    }

    fn max_reg_idx(&self) -> RegIndex {
        unsafe { self.slotend.offset_from(self.func) as i32 }
    }

    fn inicall(callee_ptr: *mut Value) -> Frame {
        Frame {
            pc: Self::INIT_FRAME_PC,
            nret: 0,
            func: callee_ptr,
            slotend: callee_ptr,
            up_ptr: ptr::null_mut(),
            uplen: 0,
            kst_ptr: ptr::null_mut(),
            kstlen: 0,
            code_ptr: ptr::null_mut(),
            codelen: 0,
        }
    }

    fn rscall(callee_ptr: *mut Value) -> Frame {
        Frame {
            pc: 0,
            nret: 0,
            func: callee_ptr,
            slotend: callee_ptr,
            up_ptr: ptr::null_mut(),
            uplen: 0,
            kst_ptr: ptr::null_mut(),
            kstlen: 0,
            code_ptr: ptr::null_mut(),
            codelen: 0,
        }
    }

    fn luacall(callee_ptr: *mut Value, mut luacl: Gc<LuaClosure>) -> Frame {
        Frame {
            pc: 0,
            nret: Frame::LUA_MULTI_RET,
            func: callee_ptr,
            slotend: unsafe { callee_ptr.add(luacl.nreg() as usize) },
            up_ptr: luacl.upvals.as_mut_ptr(),
            uplen: luacl.upvals.len() as i32,
            kst_ptr: luacl.constant().as_ptr(),
            kstlen: luacl.constant().len() as i32,
            code_ptr: luacl.code.as_ptr(),
            codelen: luacl.code.len() as i32,
        }
    }
}

pub struct StackIter {
    begin: *const Slot,
    end: *const Slot,
}

impl Iterator for StackIter {
    type Item = Value;
    fn next(&mut self) -> Option<Self::Item> {
        if self.begin >= self.end {
            None
        } else {
            let val = unsafe { *self.begin };
            unsafe {
                self.begin = self.begin.add(1);
            }
            Some(val)
        }
    }
}

// pub struct DebugPoint {}

// type HookFn = fn(&mut VM, &DebugPoint);

#[derive(Debug, Clone, Copy)]
pub struct WarnFn {
    handler: fn(&mut VM),
}

impl Default for WarnFn {
    fn default() -> Self {
        WarnFn {
            handler: |_vm: &mut VM| todo!(),
        }
    }
}

#[derive(Debug, Clone, Copy)]
pub struct PanicFn {
    handler: fn(&mut VM, InterpretError),
}

impl PanicFn {
    fn pass_on_error(vm: &mut VM, e: InterpretError) {
        // print backtrace and show local variables
        let mut last_srcinfo = Value::Nil;
        let mut line = 0;
        for frame in vm.callchain.iter().rev() {
            match unsafe { *frame.func } {
                Value::Fn(cl) => {
                    last_srcinfo = cl.source;
                    line = cl.locate(frame.pc - 1); // pc - 1: pc is the next isc
                    break;
                }
                _ => continue,
            }
        }
        eprintln!(
            "{}: {}:{}: {}",
            std::env::args().next().unwrap(),
            last_srcinfo,
            line,
            e
        );

        // stack unwind and log traceback
        eprintln!("stack traceback:");
        while vm.callchain.len() != 1 {
            let frame = vm.callchain.pop().unwrap();
            match unsafe { *frame.func } {
                Value::Fn(cl) => {
                    eprintln!(
                        "\t {}:{}: in function 0x{:X}",
                        cl.source,
                        cl.locate(frame.pc - 1), // pc - 1: pc is the next isc
                        cl.address() & 0xFFFFFFFF  // FIXME: use function name instead of address
                    )
                }
                Value::RsFn(_) => {
                    eprintln!("\t [Rust]: in ?",)
                }
                _ => continue,
            }
        }
    }
}

impl Default for PanicFn {
    fn default() -> Self {
        PanicFn {
            handler: Self::pass_on_error,
        }
    }
}

#[derive(Debug, Clone, Copy, Default)]
pub enum Hook {
    #[default]
    NoHook,
    Call(fn()),    // on function call
    Return(fn()),  // on function return
    NewLine(fn()), // a new line
    Count(fn()),   // instruction count
}

pub enum LoadMode {
    AutoDetect,
    Text,
    Binary,
}

pub struct VM {
    heap: Heap,

    stack: Box<[Slot]>,    // vm register stack
    top: *mut Slot,        // top of stack
    callchain: Vec<Frame>, // call frames

    global: Gc<Table>,                 // global table
    metatab: BTreeMap<Tag, Gc<Table>>, // meta table for basic types

    warn: WarnFn,   // warn handler
    panic: PanicFn, // panic handler
    hook: Hook,     // hook state

    calldepth: u32, // call depth of rust code
}

impl Deref for VM {
    type Target = Frame;

    fn deref(&self) -> &Self::Target {
        // SAFETY: there are at least one init frame in call chain
        unsafe { self.callchain.last().unwrap_unchecked() }
    }
}

impl DerefMut for VM {
    fn deref_mut(&mut self) -> &mut Self::Target {
        // SAFETY: there are at least one init frame in call chain
        unsafe { self.callchain.last_mut().unwrap_unchecked() }
    }
}

impl Default for VM {
    fn default() -> Self {
        Self::new()
    }
}

impl VM {
    pub const MAX_RS_CALL_DEPTH: u32 = 200; // recursion limit
    pub const MIN_STACK_SPACE: usize = 32; // min stack size
    pub const MAX_STACK_SPACE: usize = 256; // max stack limit
    pub const RESERVE_STACK_SPACE: usize = 64 / std::mem::size_of::<Value>(); // extra slot for error handling

    #[rustfmt::skip]
    pub const TYPE_STRS: [&'static str; 8] = [
        "number", "integer", "name", "string",
        "userdata", "table", "function", "thread",
    ];

    pub fn new() -> Self {
        let mut init_stack = Vec::new();
        init_stack.reserve_exact(Self::MIN_STACK_SPACE);
        init_stack.resize(init_stack.capacity(), Value::Nil);

        let mut vm = VM {
            global: Gc::dangling(),
            heap: Heap::default(),
            top: ptr::null_mut(),
            stack: init_stack.into_boxed_slice(),
            callchain: Vec::with_capacity(4),
            metatab: BTreeMap::default(),
            warn: WarnFn::default(),
            panic: PanicFn::default(),
            hook: Hook::default(),
            calldepth: 0,
        };

        let initframe = Frame::inicall(vm.stack.as_mut_ptr());
        vm.callchain.push(initframe);
        vm.top = unsafe { vm.slotend.offset(1) };
        vm.global = vm.heap.alloc_table();
        vm
    }

    pub fn open(&mut self, lib: Stdlib) -> Result<usize, InterpretError> {
        let mut nfunc = 0;
        for entry in crate::ffi::get_std_libs(lib) {
            nfunc += ffi::open_lib(self, entry)?;
        }

        // create string about meta operators lazily
        if lib == Stdlib::Base {
            for builtin in Self::TYPE_STRS
                .iter()
                .chain(MetaOperator::METATOPS_STRS.iter())
            {
                self.new_fixed(builtin);
            }
        }

        Ok(nfunc as usize)
    }

    pub fn open_libs(&mut self, libs: &[Stdlib]) -> Result<(), InterpretError> {
        for lib in libs.iter() {
            self.open(*lib)?;
        }
        Ok(())
    }

    pub fn genv(&self) -> Value {
        self.global.into()
    }

    pub fn set_global<V>(&mut self, key: Value, val: V) -> Option<Value>
    where
        V: Into<Value>,
    {
        self.global.insert(key, val.into())
    }

    pub fn get_global(&mut self, key: Value) -> Value {
        self.global.get(&key).cloned().unwrap_or_default()
    }

    /// Get stack height of current frame.
    pub fn top(&self) -> i32 {
        let offset = unsafe { self.top.offset_from(self.slotend) } - 1;
        debug_assert!(offset >= 0);
        offset as i32
    }

    /// Return how many slots is available to use in current stack.
    fn stack_remain(&self) -> usize {
        let slots_in_using = unsafe { self.top.offset_from(self.stack.as_ptr()) } - 1;
        debug_assert_ne!(slots_in_using, -1);
        self.stack.len() - VM::RESERVE_STACK_SPACE - slots_in_using as usize
    }

    pub fn push<V>(&mut self, val: V) -> Result<(), InterpretError>
    where
        Value: From<V>,
    {
        self.try_extend_stack()?;
        unsafe {
            *self.top = Value::from(val);
            self.top = self.top.add(1)
        };
        Ok(())
    }

    pub fn pop(&mut self) -> Option<Value> {
        debug_assert!(self.top >= self.slotend);
        self.try_shrink_stack();
        (self.top > self.slotend).then(|| unsafe {
            self.top = self.top.sub(1);
            *self.top
        })
    }

    pub unsafe fn pop_unchecked(&mut self) -> Value {
        debug_assert!(self.top > self.slotend);
        self.top = self.top.sub(1);
        *self.top
    }

    fn try_shrink_stack(&mut self) {
        // shrink stack to half if stack size is less than 1/2 of capacity
        if self.stack.len() == Self::MIN_STACK_SPACE || self.stack_remain() * 2 < self.stack.len() {
            return;
        }

        let new_size = Self::MIN_STACK_SPACE.max(self.stack.len() / 2);
        let mut shrink = Vec::new();
        shrink.reserve_exact(new_size);
        shrink.resize(new_size, Value::Nil);
        shrink.copy_from_slice(&self.stack[..new_size]);

        self.recorrect_stack_ptr(shrink);
    }

    fn try_extend_stack(&mut self) -> Result<(), InterpretError> {
        debug_assert!(self.stack_remain() <= self.stack.len());
        if self.stack_remain() >= 1 {
            return Ok(());
        };

        if self.stack.len() == Self::MAX_STACK_SPACE {
            return Err(InterpretError::StackOverflow);
        }

        let mut expand = Vec::new();
        let new_size = Self::MAX_STACK_SPACE.min(self.stack.len() * 2);
        expand.reserve_exact(new_size);
        expand.resize(expand.capacity(), Value::Nil);
        expand[..self.stack.len()].copy_from_slice(&self.stack);

        self.recorrect_stack_ptr(expand);
        Ok(())
    }

    fn recorrect_stack_ptr(&mut self, new_stack: Vec<Slot>) {
        let height = self.top();
        let old = self.stack.as_mut_ptr();
        self.stack = new_stack.into_boxed_slice();
        let new = self.stack.as_mut_ptr();

        self.frame_iter_mut().for_each(|frame| unsafe {
            frame.func = new.offset(frame.func.offset_from(old));
            frame.slotend = new.offset(frame.slotend.offset_from(old));
        });
        self.top = unsafe { self.slotend.add(1 + height as usize) };
    }

    pub fn unsafe_script(
        &mut self,
        src: &str,
        chunkname: Option<String>,
    ) -> Result<(), InterpretError> {
        // load and push chunk
        self.load(src, chunkname)?;

        // virtual register id of loaded function in current frame
        // [func, 0, 1, 2, ... slotend, ...  top-1(loaded register), top]
        let vregid = unsafe { self.top.offset_from(self.func) } as i32 - 2;
        self.do_call(vregid, 0, 0)
    }

    pub fn script(&mut self, src: &str, chunkname: Option<String>) {
        self.safe_script(src, chunkname, Some(PanicFn::default()))
    }

    pub fn script_file<P: AsRef<Path>>(&mut self, path: P) -> Result<(), std::io::Error> {
        let chunkname = path
            .as_ref()
            .to_str()
            .map_or(String::from("main chunk"), |s| s.to_string());
        let source = std::fs::read_to_string(path)?;
        self.script(&source, Some(chunkname));
        Ok(())
    }

    pub fn safe_script(&mut self, src: &str, chunkname: Option<String>, on_err: Option<PanicFn>) {
        let old = self.set_panic_handler(on_err.unwrap_or_default());
        if let Err(e) = self.unsafe_script(src, chunkname) {
            (self.panic.handler)(self, e);
        }
        self.set_panic_handler(old);
    }

    pub fn safe_script_file<P: AsRef<Path>>(
        &mut self,
        path: P,
        chunkname: Option<String>,
        on_err: Option<PanicFn>,
    ) -> Result<(), std::io::Error> {
        self.safe_script(&std::fs::read_to_string(path)?, chunkname, on_err);
        Ok(())
    }

    /// Loads a Lua chunk without running it. If there are no errors, `load` pushes the compiled chunk as a Lua function on top of the stack.
    pub fn load(&mut self, src: &str, chunkname: Option<String>) -> Result<(), InterpretError> {
        let proto = self.compile(src, chunkname, true)?;
        dbg!(&proto);
        let cls = self.new_closure(Gc::new(proto), Some(self.genv()));
        self.push(cls)
    }

    pub fn load_file<P>(&mut self, path: P, mode: LoadMode) -> Result<(), InterpretError>
    where
        P: AsRef<Path>,
    {
        match mode {
            LoadMode::AutoDetect => todo!(),
            LoadMode::Text => {
                let chunkname = path
                    .as_ref()
                    .file_name()
                    .and_then(|osstr| osstr.to_str())
                    .unwrap_or(Block::ANONYMOUS_CHUNK)
                    .to_string();

                self.load(&std::fs::read_to_string(path)?, Some(chunkname))
            }

            LoadMode::Binary => {
                let mut reader = BufReader::new(File::open(path)?);
                let proto = ChunkDumper::undump(&mut reader, &mut self.heap)?;
                let fnval = self.heap.alloc_closure(Gc::new(proto));
                self.push(fnval)
            }
        }
    }

    pub fn compile(
        &mut self,
        src: &str,
        chunkname: Option<String>,
        enable_optimize: bool,
    ) -> Result<Proto, InterpretError> {
        let mut block = Parser::parse(src, chunkname)?;
        if enable_optimize {
            passes::constant_fold(&mut block);
        }
        Ok(CodeGen::codegen(block, false, &mut self.heap)?)
    }

    pub fn cur_frame(&self) -> &Frame {
        unsafe { self.callchain.last().unwrap_unchecked() }
    }

    pub fn cur_frame_mut(&mut self) -> &mut Frame {
        unsafe { self.callchain.last_mut().unwrap_unchecked() }
    }

    pub fn stack_iter(&mut self) -> StackIter {
        StackIter {
            begin: self.stack.as_mut_ptr(),
            end: self.top,
        }
    }

    pub fn frame_iter(&self) -> std::slice::Iter<Frame> {
        self.callchain.iter()
    }

    pub fn frame_iter_mut(&mut self) -> std::slice::IterMut<Frame> {
        self.callchain.iter_mut()
    }

    pub fn new_fixed(&mut self, reserved: &str) -> Value {
        self.heap.alloc_fixed(reserved)
    }

    pub fn new_str(&mut self, view: &str) -> Value {
        self.heap.alloc_str(view).into()
    }

    pub fn take_str(&mut self, val: String) -> Value {
        self.heap.take_str(val).into()
    }

    pub fn new_table(&mut self) -> Value {
        self.heap.alloc_table().into()
    }

    pub fn new_closure(&mut self, proto: Gc<Proto>, env: Option<Value>) -> Value {
        let mut cls = self.heap.alloc_closure(proto);
        let mut ups = Vec::with_capacity(cls.updecl().len());
        for updecl in cls.updecl().iter() {
            match updecl {
                UpvalDecl::Env => ups.push(UpVal::Close(
                    env.unwrap_or_else(|| Value::from(self.global)),
                )),
                UpvalDecl::OnStack { .. } => todo!(),
                UpvalDecl::InUpList { .. } => todo!(),
            };
        }
        let _ = std::mem::replace(&mut cls.upvals, ups);
        cls.into()
    }

    pub fn check_gc(&self) -> bool {
        self.heap.check_gc()
    }

    unsafe fn stack_peek(&self, idx: i32) -> *mut Value {
        match idx.cmp(&0) {
            Ordering::Equal | Ordering::Greater => self.slotend.offset(idx as isize),
            Ordering::Less => self.top.offset(idx as isize),
        }
    }

    /// Peek a value on stack with it's index. Same with PUC-Rio Lua.
    ///
    /// The `slotend` of current frame has index 0.
    /// If a positive number was given, then index stack from bottom to top. And in a reversed order with a negetive number.
    ///
    /// If the index out of the bound of stack, `None` will be returned.
    pub fn peek(&self, idx: i32) -> Option<Value> {
        let slot_ptr = unsafe { self.stack_peek(idx) };
        if slot_ptr >= self.func && slot_ptr < self.top {
            Some(unsafe { *slot_ptr })
        } else {
            None
        }
    }

    pub unsafe fn peek_unchecked(&self, idx: i32) -> Value {
        *self.stack_peek(idx)
    }

    pub fn peek_and_then<T>(&self, idx: i32, op: impl FnOnce(Value) -> T) -> Option<T> {
        self.peek(idx).map(op)
    }

    pub fn check(&self, idx: i32, pred: impl FnOnce(Value) -> bool) -> bool {
        self.peek(idx).map(pred).unwrap_or(false)
    }

    pub fn check_and_then<T>(
        &self,
        idx: i32,
        pred: impl FnOnce(&Value) -> bool,
        op: impl FnOnce(Value) -> T,
    ) -> Option<T> {
        self.peek(idx).filter(pred).map(op)
    }

    pub fn call(&mut self, callee: Value, args: &[Value]) -> Result<i32, InterpretError> {
        self.push(callee)?;
        for arg in args {
            self.push(*arg)?;
        }
        self.prep_call_frame(0, args.len() as i32, 0)?;
        self.execute()?;
        Ok(self.top())
    }

    fn do_call(
        &mut self,
        fnreg: RegIndex,
        n_arg: i32,
        n_exp_ret: i32,
    ) -> Result<(), InterpretError> {
        let nret = self.prep_call_frame(fnreg, n_arg, n_exp_ret)?;
        if nret == Frame::LUA_MULTI_RET as usize {
            self.clear_call_frame(n_exp_ret, self.nret)?;
        }
        Ok(())
    }

    /// fnindex: function index based on self.func,  return how many returns.
    /// light C function will be called strightly, luafn will return `Frame::LUA_MULTI_RET`.
    fn prep_call_frame(
        &mut self,
        fnreg: RegIndex,
        narg: i32,
        n_exp_ret: i32,
    ) -> Result<usize, InterpretError> {
        let slot = self.slot_ptr_of(fnreg);
        let callee = unsafe { *slot };
        if !callee.is_callable() {
            return Err(InterpretError::InvalidInvocation { callee });
        };

        match callee {
            Value::RsFn(rsfn) => {
                self.callchain.push(Frame::rscall(slot));
                self.top = unsafe { self.func.offset(1 + narg as isize) };
                let nret = rsfn(self)?;
                self.clear_call_frame(n_exp_ret, nret as i32)?;
                Ok(nret)
            }
            Value::Fn(luacl) => {
                // dbg!()
                if luacl.is_vararg() {
                    dbg!("skip load vararg for luacl");
                } else if narg < luacl.nparams() {
                    for idx in (narg + 1)..=luacl.nparams() {
                        self.rset(idx, Value::Nil)?;
                    }
                }

                self.callchain.push(Frame::luacall(slot, luacl));
                self.top = unsafe { self.slotend.add(1) };
                self.execute()?;
                Ok(Frame::LUA_MULTI_RET as usize)
            }
            Value::RsCl(_) => todo!(),
            Value::Table(_) => todo!(),
            _ => unreachable!(),
        }
    }

    fn clear_call_frame(&mut self, nexpect: i32, nret: i32) -> Result<(), InterpretError> {
        debug_assert!(nexpect >= 0);
        debug_assert!(nret >= 0);

        // fix top pointer to adapt n returned value
        self.top = unsafe { self.slotend.offset(1 + nret as isize) };

        // write return value to  [caller, caller + 1, ... caller + nexpect - 1]
        for idx in 0..nexpect {
            let reti = self.peek(idx + 1).unwrap_or_default();
            dbg!(reti);
            unsafe { self.func.offset(idx as isize).write(reti) };
        }
        self.callchain.pop();

        // fix top pointer to slotend + 1
        self.top = unsafe { self.slotend.offset(1) };
        Ok(())
    }

    fn execute(&mut self) -> Result<(), InterpretError> {
        if self.calldepth >= VM::MAX_RS_CALL_DEPTH {
            return Err(InterpretError::RsCallDepthLimit {
                max: VM::MAX_RS_CALL_DEPTH,
            });
        }
        self.calldepth += 1;
        let origin_frame = self.callchain.len();

        use OpCode::*;
        loop {
            let code = self.fetch();
            dbg!(code.to_string());

            match code.mode() {
                OpMode::IABC => {
                    let (op, a, b, c, k) = code.repr_abck();
                    match op {
                        MOVE => {
                            self.rset(a, self.rget(b)?)?;
                        }

                        LOADFALSE => {
                            self.rset(a, false)?;
                        }

                        LOADTRUE => {
                            self.rset(a, true)?;
                        }

                        LOADNIL => {
                            for reg in a..=a + b {
                                self.rset(reg, Value::Nil)?;
                            }
                        }

                        GETTABUP => {
                            let mut table = match self.upget(b) {
                                UpVal::Close(val) => {
                                    debug_assert_eq!(val.tag(), Some(Tag::Table));
                                    unsafe { val.as_table().unwrap_unchecked() }
                                }
                                UpVal::Open(_) => todo!(),
                            };
                            self.rset(a, table.index(self.kget(c)))?;
                        }

                        SETTABUP => match self.upget(a) {
                            UpVal::Close(val) => {
                                debug_assert_eq!(val.tag(), Some(Tag::Table));
                                let key = self.kget(b);
                                debug_assert_eq!(key.tag(), Some(Tag::String));
                                let mut table = unsafe { val.as_table().unwrap_unchecked() };
                                table.insert(key, self.rkget(k, c)?);
                            }
                            UpVal::Open(_) => todo!(),
                        },

                        SETFIELD => {
                            let key = self.kget(b);
                            debug_assert_eq!(key.tag(), Some(Tag::String));
                            let toset = self.rget(a)?;
                            debug_assert_eq!(toset.tag(), Some(Tag::Table));
                            let table = unsafe { &mut toset.as_table().unwrap_unchecked() };
                            if let Some(_idxop) = table.meta_get(MetaOperator::Index) {
                                todo!("metaop: index")
                            } else {
                                table.insert(key, self.kget(c));
                            }
                        }

                        NEWTABLE => {
                            debug_assert!(b >= 0 && c >= 0);
                            let table = self.new_table();
                            self.rset(a, table)?;
                        }

                        MUL => {
                            self.rset(a, 6)?;
                        }

                        CALL => {
                            self.do_call(a, b - 1, c - 1)?;
                        }

                        RETURN1 => {
                            self.push(self.rget(a)?)?;
                            self.nret = 1;
                            self.calldepth -= 1;
                            return Ok(());
                        }

                        RETURN0 => {
                            debug_assert!(self.callchain.len() >= origin_frame);
                            if self.callchain.len() == origin_frame {
                                self.nret = 0;
                                self.calldepth -= 1;
                                return Ok(());
                            }
                        }

                        VARARGPREP => {
                            // TODO:
                            // do nothing for now
                        }

                        _ => unimplemented!("opcode: {:?}", op),
                    }
                }
                OpMode::IABx => {
                    let (op, a, bx) = code.repr_abx();
                    match op {
                        LOADK => {
                            self.rset(a, self.kget(bx))?;
                        }

                        CLOSURE => {
                            let ocf = unsafe { self.func.read() };
                            debug_assert!(ocf
                                .as_luafn()
                                .is_some_and(|cl| { cl.subproto().len() > bx as usize }));

                            let cf = unsafe { ocf.as_luafn().unwrap_unchecked() };
                            let r = *unsafe { cf.subproto().get_unchecked(bx as usize) };
                            let newfn = self.new_closure(r, Some(self.genv()));
                            self.rset(a, newfn)?;
                        }

                        _ => unimplemented!("unimplemented opcode: {:?}", op),
                    }
                }
                OpMode::IAsBx => {
                    let (op, a, sbx) = code.repr_asbx();
                    match op {
                        LOADI => {
                            self.rset(a, sbx)?;
                        }

                        LOADF => todo!(),

                        LOADK => {
                            self.rset(a, self.kget(sbx))?;
                        }

                        _ => unimplemented!("IASBX"),
                    }
                }
                OpMode::IAx => {
                    let (op, _) = code.repr_ax();
                    match op {
                        OpCode::EXTRAARG => {}
                        _ => unreachable!(),
                    };
                }
                OpMode::IsJ => todo!(),
            }
        }
    }

    /// Set warn handler, return old handler
    pub fn set_warn_handler(&mut self, warn: WarnFn) -> WarnFn {
        std::mem::replace(&mut self.warn, warn)
    }

    /// Set panic handler, return old handler
    pub fn set_panic_handler(&mut self, panic: PanicFn) -> PanicFn {
        std::mem::replace(&mut self.panic, panic)
    }

    pub fn set_hook_handler(&mut self, hook: Hook) -> Hook {
        std::mem::replace(&mut self.hook, hook)
    }

    fn mark_rootset_reachable(&mut self) {
        // mark all reachable object to black
        self.stack_iter().for_each(|val| val.mark_reachable());
        self.global.mark_reachable();
    }

    pub fn full_gc(&mut self) {
        self.heap.mark_all_obj_unreachable();
        self.mark_rootset_reachable();

        // table with finalizer
        let twfs = self.heap.sweep_unreachable();
        for tofinal in twfs.into_iter() {
            // TODO:
            // process twfs and drop internal strings
            Gc::drop(tofinal);
        }
        // dbg!(self.heap.total_alloc_bytes());
    }
}

#[cfg(test)]
mod test {
    use super::*;

    #[test]
    fn trig_gc_with_simple_objects() {
        use crate::heap::{MemStat, StrImpl};

        let mut vm = VM::new();
        let long = "l.o.n.g.".repeat(4);
        assert!(!StrImpl::able_to_internalize(&long));

        for _ in 0..3 {
            let origin = vm.heap.total_alloc_bytes();
            vm.take_str(long.clone());
            vm.full_gc();
            assert_eq!(origin, vm.heap.total_alloc_bytes());
        }

        let origin = vm.heap.total_alloc_bytes();
        for n in 1..=3 {
            let mut to_push = long.clone();
            to_push.push_str(&n.to_string());
            let val = vm.take_str(to_push);
            assert_eq!(vm.push(val).unwrap(), ());
            vm.full_gc();
            assert_eq!(origin + val.mem_ref() * n, vm.heap.total_alloc_bytes());
        }

        // pop 2 item, remain 1 long string (n=1)
        let _ = vm.pop();
        let val = vm.pop().unwrap();
        vm.full_gc();
        assert_eq!(origin + val.mem_ref(), vm.heap.total_alloc_bytes());
    }

    #[test]
    fn stack_operation() {
        let mut vm = VM::new();
        assert_eq!(vm.top(), 0);
        debug_assert_eq!(vm.slotend, vm.func);

        let init = vm.stack.as_mut_ptr();
        debug_assert_eq!(vm.func, init);
        debug_assert_eq!(vm.top, unsafe { vm.func.add(1) });

        // frame check
        assert_eq!(vm.cur_frame().pc, Frame::INIT_FRAME_PC);

        // check init state
        let old_stack_base = vm.stack.as_mut_ptr();
        assert_eq!(vm.cur_frame().func, old_stack_base);
        assert_eq!(vm.cur_frame().func, vm.cur_frame().slotend);
        assert_eq!(vm.top, unsafe { vm.cur_frame().slotend.offset(1) });
        assert_eq!(vm.stack_remain(), vm.stack.len() - VM::RESERVE_STACK_SPACE);

        // push until stack overflow
        let mut cnt = 0;
        while vm.push(Value::Int(cnt as i64)).is_ok() {
            cnt += 1;
            assert_eq!(vm.top(), cnt);
        }

        let new_stack_base = vm.stack.as_mut_ptr();
        assert_eq!(vm.cur_frame().func, new_stack_base);
        assert_eq!(vm.cur_frame().func, vm.cur_frame().slotend);
        assert_eq!(vm.cur_frame().slotend, new_stack_base);
        assert_eq!(vm.stack_remain(), 0);
        assert_eq!(
            vm.top() as usize,
            VM::MAX_STACK_SPACE - VM::RESERVE_STACK_SPACE
        );

        assert!(vm
            .push(Value::Int(999))
            .is_err_and(|e| matches!(e, InterpretError::StackOverflow)));

        // pop until stak has been clear
        while vm.top() != 0 {
            cnt -= 1;
            assert_eq!(vm.pop().unwrap().as_int().unwrap(), cnt as i64);
        }
        assert_eq!(vm.cur_frame().func, vm.cur_frame().slotend);
        assert_eq!(vm.stack_remain(), vm.stack.len() - VM::RESERVE_STACK_SPACE);
        assert_eq!(vm.stack.len(), VM::MIN_STACK_SPACE);
    }

    #[test]
    fn before_and_after_call() -> Result<(), InterpretError> {
        fn check_init_state(vm: &mut VM) {
            assert_eq!(vm.top(), 0);

            assert_eq!(vm.callchain.len(), 1);
            // frame check
            assert_eq!(vm.cur_frame().pc, Frame::INIT_FRAME_PC);

            // check init state
            let old_stack_base = vm.stack.as_mut_ptr();
            assert_eq!(vm.cur_frame().func, old_stack_base);
            assert_eq!(vm.cur_frame().func, vm.cur_frame().slotend);
            assert_eq!(vm.top, unsafe { vm.cur_frame().slotend.offset(1) });
            assert_eq!(vm.stack_remain(), vm.stack.len() - VM::RESERVE_STACK_SPACE);
        }

        let mut vm = VM::new();
        vm.open(Stdlib::Base)?;

        check_init_state(&mut vm);
        let call = r#"
            print ("")
        "#;
        assert_eq!(vm.unsafe_script(call, None).unwrap(), ());
        check_init_state(&mut vm);
        Ok(())
    }
}
