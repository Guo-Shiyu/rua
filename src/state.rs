use std::{
    cell::UnsafeCell,
    cmp::Ordering,
    collections::{BTreeMap, LinkedList},
    convert::From,
    ops::{Deref, DerefMut},
    ptr,
};

use crate::{
    codegen::{CodeGen, Instruction, OpCode, Proto},
    ffi::StdLib,
    heap::{Heap, ManagedHeap, MarkAndSweepGcOps, Tag},
    parser::Parser,
    passes,
    value::{AsTableKey, LValue, RsFunc, TableImpl},
};

use super::RuaErr;

pub type RegIndex = i32;

/// CallInfo
#[derive(Clone)]
pub struct Frame {
    oldpc: u32, // previous frame pc state
    pc: u32,    // current frame pc

    stk_base: *mut LValue, // stack base pointer
    stk_last: *mut LValue, // stack last pointer (extra stack space is not included )
    reg_base: RegIndex, // function register index, (first avaliable register index - 1), based on stk_base
    reg_top: RegIndex, // last available register index + 1, first unavailable register index, based on stk_base

    upvalues: (*const LValue, u32),       // refer to proto upvalues
    constants: (*const LValue, u32),      // refer to proto constants
    bytescode: (*const Instruction, u32), // ref to proto bytescode
}

impl Frame {
    pub fn is_init_frame(&self) -> bool {
        self.oldpc == u32::MAX
    }

    // Get stack height of current frame.
    pub fn top(&self) -> i32 {
        self.reg_top - self.reg_base - 1
    }

    pub fn iget(&self) -> Instruction {
        unsafe { self.bytescode.0.offset(self.pc as isize).read() }
    }

    pub fn kget(&self, idx: RegIndex) -> LValue {
        unsafe { self.constants.0.offset(idx as isize).read() }
    }

    pub fn upget(&self, idx: RegIndex) -> LValue {
        unsafe { self.upvalues.0.offset(idx as isize).read() }
    }

    pub fn rget(&self, idx: RegIndex) -> Result<LValue, RuaErr> {
        match idx.cmp(&0) {
            Ordering::Less => {
                let absidx = self.reg_top + idx;
                if absidx > self.reg_base {
                    let val = unsafe { self.stk_base.offset(absidx as isize).read() };
                    Ok(val)
                } else {
                    Err(RuaErr::InvalidRegisterAccess {
                        target: absidx,
                        max: self.reg_top,
                    })
                }
            }
            Ordering::Greater | Ordering::Equal => {
                let absidx = self.reg_base + idx;
                if absidx < self.reg_top {
                    let val = unsafe { self.stk_base.offset(absidx as isize).read() };
                    Ok(val)
                } else {
                    Err(RuaErr::InvalidRegisterAccess {
                        target: absidx,
                        max: self.reg_top,
                    })
                }
            }
        }
    }

    pub fn rset(&self, idx: RegIndex, val: LValue) {
        unsafe { *self.stk_base.offset((self.reg_base + idx) as isize) = val }
    }
}

pub struct DebugPoint {}

type HookFn = fn(&mut State, &DebugPoint);

#[derive(Debug, Clone, Copy, Default)]
pub enum HookMask {
    #[default]
    One,
    Two,
    Three,
}

pub struct Rvm {
    stack: Vec<LValue>,                // vm registers stack
    callchain: LinkedList<Frame>,      // call frames
    ci: Frame,                         // current frame
    globaltab: TableImpl,              // global table
    metatab: BTreeMap<Tag, TableImpl>, // meta table for basic types
    warnfn: RsFunc,                    // warn handler
    panicfn: RsFunc,                   // panic handler
    allowhook: bool,                   // enable hook
    hook: HookFn,                      // hook handler
    hkmask: HookMask,                  // hook mask
}

impl Deref for Rvm {
    type Target = Frame;

    fn deref(&self) -> &Self::Target {
        &self.ci
    }
}

impl DerefMut for Rvm {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.ci
    }
}

impl Rvm {
    pub const MAX_CALL_DEPTH: usize = 200; // recursion limit
    pub const MIN_STACK_SPACE: usize = 32; // min stack size
    pub const MAX_STACK_SPACE: usize = 256; // max stack limit
    pub const EXTRA_STACK_SPACE: usize = 8; // extra slot for error handling

    pub fn height(&mut self) -> u32 {
        self.stack.len() as u32
    }

    pub fn new() -> Self {
        let mut stack = Vec::with_capacity(Self::MIN_STACK_SPACE + Self::EXTRA_STACK_SPACE);

        // First frame of a vm (rust call lua frame)
        let init_frame = Frame {
            oldpc: u32::MAX,
            pc: 0,
            stk_base: stack.as_mut_ptr(),
            stk_last: unsafe {
                stack
                    .as_mut_ptr()
                    .add(stack.capacity() - Self::EXTRA_STACK_SPACE)
            },
            reg_base: -1,
            reg_top: 0,
            upvalues: (ptr::null(), 0),
            constants: (ptr::null(), 0),
            bytescode: (ptr::null(), 0),
        };

        Rvm {
            stack,
            callchain: LinkedList::new(),
            ci: init_frame,
            globaltab: TableImpl::with_capacity((24, 0)),
            metatab: init_metatable(),
            warnfn: default_warnfn,
            panicfn: default_panicfn,
            allowhook: false,
            hook: |_, _| {},
            hkmask: HookMask::default(),
        }
    }

    pub fn set_global<K>(&mut self, k: K, val: LValue)
    where
        K: AsTableKey,
    {
        self.globaltab.set(k, val);
    }

    pub fn get_global<K>(&mut self, k: K) -> LValue
    where
        K: AsTableKey,
    {
        self.globaltab.get(k)
    }

    pub fn stk_push(&mut self, val: LValue) -> Result<(), RuaErr> {
        self.try_extend_stack().map_err(|e| e as RuaErr)?;
        self.stack.push(val);
        self.reg_top += 1;
        Ok(())
    }

    pub fn stk_pop(&mut self) -> Option<LValue> {
        self.try_shrink_stack();
        debug_assert!(self.reg_top >= self.reg_base);
        if self.reg_top > self.reg_base {
            self.reg_top -= 1;
            self.stack.pop()
        } else {
            None
        }
    }

    // pub fn stk_get(&self, idx: RegIndex) -> Result<LValue, RuntimeErr> {
    // match idx.cmp(&0) {
    //     std::cmp::Ordering::Greater => {
    //   RuaErr::InvalidRegisterAccess { target: absidx, max: self.reg_top }{
    //             Err(RuntimeErr::InvalidStackIndex)
    //         }
    //     }
    //     std::cmp::Ordering::Equal => Err(RuntimeErr::InvalidStackIndex),
    //     std::cmp::Ordering::Less => {
    //         let idx = self.reg_top + idx;
    //         if idx >= self.reg_base {
    //             Ok(self.stack[idx as usize].clone())
    //         } else {
    //             Err(RuntimeErr::InvalidStackIndex)
    //         }
    //     }
    // }
    // }

    pub fn stk_remain(&self) -> usize {
        let total = unsafe { self.stk_last.offset_from(self.stk_base) };
        (total - self.reg_top as isize) as usize
    }

    // pub fn stk_check_with(&self, idx: RegIndex, op: impl FnOnce(&LValue) -> bool) -> bool {
    //     if idx > self.callinfo().reg_top {
    //         false
    //     } else {
    //         let val = self.stk_get(idx).unwrap();
    //         op(&val)
    //     }
    // }

    fn try_shrink_stack(&mut self) {
        // shrink stack to half if stack size is less than 1/3 of capacity
        let cur_len = self.stack.capacity() - Self::EXTRA_STACK_SPACE;
        if cur_len > Self::MIN_STACK_SPACE && cur_len < self.stack.capacity() / 3 {
            let fixed = Self::MIN_STACK_SPACE.max(self.stack.capacity() / 2);
            self.stack.shrink_to(fixed);
            debug_assert!(fixed >= Self::MIN_STACK_SPACE);
            self.correct_callchain();
        }
    }

    fn try_extend_stack(&mut self) -> Result<(), RuaErr> {
        if self.stack.len() + Self::EXTRA_STACK_SPACE >= self.stack.capacity() {
            let need = self.stack.capacity() * 2;
            if need > Self::MAX_STACK_SPACE {
                return Err(RuaErr::StackOverflow);
            }
            self.stack.reserve(need + Self::EXTRA_STACK_SPACE);
            self.correct_callchain();
        }
        Ok(())
    }

    fn correct_callchain(&mut self) {
        self.ci.stk_base = self.stack.as_mut_ptr();
        self.ci.stk_last = unsafe { self.ci.stk_base.add(self.stack.capacity()) };

        self.callchain.iter_mut().for_each(|ci| {
            ci.stk_base = self.ci.stk_base;
            ci.stk_last = self.ci.stk_last;
        })
    }
}
// enum LoadMode {
//     Auto,   // auto detect mode by input
//     Text,
//     Binary,
// }

pub struct State {
    vm: UnsafeCell<Rvm>, // runtime
    heap: ManagedHeap,   // GC heap
}

impl Deref for State {
    type Target = Rvm;

    fn deref(&self) -> &Self::Target {
        unsafe { &*self.vm.get() }
    }
}

impl DerefMut for State {
    fn deref_mut(&mut self) -> &mut Self::Target {
        unsafe { &mut *self.vm.get() }
    }
}

impl Default for State {
    fn default() -> Self {
        Self::new()
    }
}

impl Drop for State {
    fn drop(&mut self) {
        self.context(|_vm, heap| {
            heap.collect_garbage(true);
            heap.destroy();
            Ok(())
        })
        .unwrap();
    }
}

impl State {
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

    pub fn new() -> Self {
        let mut state = State {
            vm: UnsafeCell::new(Rvm::new()),
            heap: ManagedHeap::default(),
        };

        state.heap_with(|h| {
            // add builtin string to string pool
            for builtin in Self::TYPE_STRS {
                h.alloc_fixed(builtin);
            }
        });

        state
    }

    pub fn with_libs(libs: &[StdLib]) -> Self {
        let mut vm = Self::new();
        vm.open_libs(libs);
        vm
    }

    pub fn vm_view(&mut self) -> &mut Rvm {
        self.vm.get_mut()
    }

    pub fn heap_view(&mut self) -> Heap {
        Heap::new(self.vm.get_mut(), &mut self.heap)
    }

    pub fn context<O>(&mut self, op: O) -> Result<(), RuaErr>
    where
        O: FnOnce(&mut Rvm, &mut Heap) -> Result<(), RuaErr>,
    {
        unsafe {
            op(
                &mut *self.vm.get(),
                &mut Heap::new(self.vm.get_mut(), &mut self.heap),
            )
        }
    }

    pub fn vm_with<O>(&mut self, op: O) -> Result<(), RuaErr>
    where
        O: FnOnce(&mut Rvm) -> Result<(), RuaErr>,
    {
        op(self.vm_view())
    }

    pub fn heap_with<O>(&mut self, op: O) -> &Self
    where
        O: FnOnce(&mut Heap),
    {
        op(&mut self.heap_view());
        self
    }

    pub fn open(&mut self, lib: StdLib) {
        let entrys = crate::ffi::get_std_libs(lib);
        for (name, func) in entrys.iter() {
            let res = self.context(|vm, heap| {
                vm.globaltab.set(heap.alloc(*name), LValue::from(*func));
                Ok(())
            });
            debug_assert!(res.is_ok())
        }

        // create string about meta operators lazyly
        if lib == StdLib::Base {
            self.heap_with(|h| {
                for builtin in Self::METATOPS_STRS {
                    h.alloc_fixed(builtin);
                }
            });
        }
    }

    pub fn open_libs(&mut self, libs: &[StdLib]) {
        for lib in libs.iter() {
            self.open(*lib);
        }
    }

    pub fn safe_script(&mut self, _src: &str) -> Result<(), RuaErr> {
        todo!()
    }

    // Return the retval count of the script, and the val has pushed on the stack in return order.
    pub fn script(&mut self, src: &str) -> Result<usize, RuaErr> {
        let preidx = self.top();

        self.load_str(src, Some("main".to_string()))?;
        self.do_call(self.top(), 0, 0)?;

        let postidx = self.top();
        Ok((postidx - preidx) as usize)
    }

    pub fn script_file(&mut self, file: &str) -> Result<usize, RuaErr> {
        let src = std::fs::read_to_string(file)?;
        self.script(&src)
    }

    /// Loads a Lua chunk without running it. If there are no errors, `load` pushes the compiled chunk as a Lua function on top of the stack.
    pub fn load_str(&mut self, src: &str, chunkname: Option<String>) -> Result<(), RuaErr> {
        let proto = Self::compile(src, chunkname, true)?;
        dbg!(&proto);
        let luaf = self.heap_view().alloc(proto);
        self.vm_view().stk_push(luaf)?;
        Ok(())
    }

    pub fn load_file(&mut self, _filepath: &str) -> Result<(), RuaErr> {
        Ok(())
    }

    // pub fn call() -> Result<(), LuaErr> {
    //     todo!()
    // }

    pub fn do_call(&mut self, fnidx: RegIndex, nin: i32, _nout: i32) -> Result<(), RuaErr> {
        let f = self.rget(fnidx)?;
        assert!(f.is_callable());

        match f {
            LValue::RsFn(f) => {
                self.ci.reg_top += nin;
                f(self)?;
                self.ci.reg_top -= nin;
            }
            LValue::Function(p) => {
                self.vm_with(|vm| {
                    if vm.callchain.len() > Rvm::MAX_CALL_DEPTH {
                        Err(RuaErr::StackOverflow)
                    } else if vm.stk_remain() < nin as usize {
                        vm.try_extend_stack().map_err(|so| so as RuaErr)
                    } else {
                        Ok(())
                    }
                })?;

                let ncodes = p.bytecode();
                let nkst = p.constant();
                let nups = p.updecl();

                let mut nextci = Frame {
                    oldpc: self.ci.pc,
                    pc: 0,
                    stk_base: self.ci.stk_base,
                    stk_last: self.ci.stk_last,
                    reg_base: self.ci.reg_top - 1,
                    reg_top: self.ci.reg_top,
                    upvalues: (ptr::null(), nups.len() as u32),
                    constants: (nkst.as_ptr(), nkst.len() as u32),
                    bytescode: (ncodes.as_ptr(), ncodes.len() as u32),
                };

                // update ci
                std::mem::swap(&mut nextci, &mut self.ci);
                self.callchain.push_back(nextci);

                // balance parameters and argument
                let narg = self.top();
                let nparam = p.nparams();
                let diff = narg.abs_diff(nparam);
                match narg.cmp(&nparam) {
                    Ordering::Less => {
                        for _ in 0..diff {
                            self.stk_push(LValue::Nil)?;
                        }
                    }
                    Ordering::Greater => {
                        for _ in 0..diff {
                            self.stk_pop();
                        }
                    }
                    Ordering::Equal => {}
                }

                self.interpret()?;

                // TODO:
                self.pos_call();
            }
            LValue::Table(_) => todo!("get metatable __call"),
            LValue::UserData(_) => todo!(),

            LValue::Nil => {
                println!("try to call a nil value");
                todo!("error report")
            }
            LValue::Bool(b) => {
                println!("try to call a bool value: {}", b);
                todo!("error report")
            }
            LValue::Int(i) => {
                println!("try to call a int value: {}", i);
                todo!("error report")
            }
            LValue::Float(f) => {
                println!("try to call a float value: {}", f);
                todo!("error report")
            }
            LValue::String(sw) => println!("try to call a string value: {}", sw.as_str()),
        }
        Ok(())
    }

    pub fn compile(src: &str, chunkname: Option<String>, optimize: bool) -> Result<Proto, RuaErr> {
        let mut block = Parser::parse(src, chunkname)?;

        // TODO:
        // optimizer scheduler
        if optimize {
            passes::constant_fold(&mut block);
        }

        let proto = CodeGen::generate(*block, false)?;
        Ok(proto)
    }

    /// Set warn handler, return old handler
    pub fn set_warn_handler(&mut self, new: RsFunc) -> RsFunc {
        let vm = self.vm.get_mut();
        let old = vm.warnfn;
        vm.warnfn = new;
        old
    }

    /// Set panic handler, return old handler
    pub fn set_panic_handler(&mut self, new: RsFunc) -> RsFunc {
        let vm = self.vm.get_mut();
        let old = vm.panicfn;
        vm.panicfn = new;
        old
    }

    pub fn set_hook_handler(&mut self, hook: HookFn, mask: HookMask) {
        let vm = self.vm.get_mut();
        vm.hook = hook;
        vm.hkmask = mask;
    }

    fn pre_call(&mut self) {}

    fn pos_call(&mut self) {}

    fn interpret(&mut self) -> Result<(), RuaErr> {
        use super::codegen::OpCode::*;
        use super::codegen::OpMode;

        loop {
            let code = self.ci.iget();
            // println!("{}", code);

            match code.mode() {
                OpMode::IABC => {
                    let (op, a, b, c, _k) = code.repr_abck();
                    match op {
                        VARARGPREP => {
                            // TODO:
                            // do nothing for now
                        }
                        GETTABUP => {
                            let gval = self.globaltab.get(self.ci.kget(b));
                            self.ci.rset(a, gval);
                        }

                        MOVE => self.ci.rset(a, self.ci.rget(b)?),

                        CALL => self.do_call(a, b - 1, c)?,

                        NEWTABLE => {
                            debug_assert!(b >= 0 && c >= 0);
                            let _table = self
                                .heap_view()
                                .alloc(TableImpl::with_capacity((b as usize, c as usize)));
                        }

                        SETFIELD => {}

                        RETURN0 => {}
                        _ => unimplemented!("unimplemented opcode: {:?}", op),
                    }
                }
                OpMode::IABx => {
                    let (op, a, bx) = code.repr_abx();
                    match op {
                        LOADK => self.ci.rset(a, self.ci.kget(bx)),
                        _ => unimplemented!("unimplemented opcode: {:?}", op),
                    }
                }
                OpMode::IAsBx => {
                    let (op, a, sbx) = code.repr_asbx();
                    match op {
                        LOADI => self.ci.rset(a, LValue::Int(sbx as i64)),
                        LOADF => todo!(),
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
            self.ci.pc += 1;
            if self.ci.pc >= self.ci.bytescode.1 {
                break;
            }
        }
        Ok(())
    }
}

fn init_metatable() -> BTreeMap<Tag, TableImpl> {
    // TODO:
    BTreeMap::new()
}

fn default_panicfn(_vm: &mut State) -> Result<usize, RuaErr> {
    todo!()
}

pub fn default_warnfn(_vm: &mut State) -> Result<usize, RuaErr> {
    todo!()
}

pub(super) fn mark_rootset(vm: &mut Rvm) {
    // mark all reachable object to black
    vm.stack.iter().for_each(|val| val.mark_reachable());
    vm.globaltab.mark_reachable();
}
