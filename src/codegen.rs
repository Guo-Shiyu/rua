use std::{
    collections::BTreeMap,
    fmt::Debug,
    io::{BufWriter, Write},
    rc::Rc,
};

use crate::{
    ast::{BinOp, Block, Expr, ExprNode, FuncCall, ParaList, Stmt, StmtNode, UnOp},
    heap::{Gc, Heap, HeapMemUsed},
    state::RegIndex,
    value::LValue,
};

/// We assume that instructions are unsigned 32-bit integers.
/// All instructions have an opcode in the first 7 bits.
/// Instructions can have the following formats:
///
/// ``` text
///       3 3 2 2 2 2 2 2 2 2 2 2 1 1 1 1 1 1 1 1 1 1 0 0 0 0 0 0 0 0 0 0
///       1 0 9 8 7 6 5 4 3 2 1 0 9 8 7 6 5 4 3 2 1 0 9 8 7 6 5 4 3 2 1 0
/// iABC        C(8)     |      B(8)     |k|     A(8)      |   Op(7)     |
/// iABx              Bx(17)               |     A(8)      |   Op(7)     |
/// iAsBx            sBx (signed)(17)      |     A(8)      |   Op(7)     |
/// iAx                         Ax(25)                     |   Op(7)     |
/// isJ                         sJ(25)                     |   Op(7)     |
/// ```               
///                 
/// A signed argument is represented in excess K: the represented value is
/// the written unsigned value minus K, where K is half the maximum for the
/// corresponding unsigned argument.
#[derive(Debug, PartialEq, Eq, Clone, Copy)]
pub enum OpMode {
    IABC,
    IABx,
    IAsBx,
    IAx,
    IsJ,
}

#[derive(Clone, Copy)]
pub struct Instruction {
    code: u32,
}

///  Notes:
///
///  (*) Opcode OP_LFALSESKIP is used to convert a condition to a boolean
///  value, in a code equivalent to (not cond ? false : true).  (It
///  produces false and skips the next instruction producing true.)
///
///  (*) Opcodes OP_MMBIN and variants follow each arithmetic and
///  bitwise opcode. If the operation succeeds, it skips this next
///  opcode. Otherwise, this opcode calls the corresponding metamethod.
///
///  (*) Opcode OP_TESTSET is used in short-circuit expressions that need
///  both to jump and to produce a value, such as (a = b or c).
///
///  (*) In OP_CALL, if (B == 0) then B = top - A. If (C == 0), then
///  'top' is set to last_result+1, so next open instruction (OP_CALL,
///  OP_RETURN*, OP_SETLIST) may use 'top'.
///
///  (*) In OP_VARARG, if (C == 0) then use actual number of varargs and
///  set top (like in OP_CALL with C == 0).
///
///  (*) In OP_RETURN, if (B == 0) then return up to 'top'.
///
///  (*) In OP_LOADKX and OP_NEWTABLE, the next instruction is always
///  OP_EXTRAARG.
///
///  (*) In OP_SETLIST, if (B == 0) then real B = 'top'; if k, then
///  real C = EXTRAARG _ C (the bits of EXTRAARG concatenated with the
///  bits of C).
///
///  (*) In OP_NEWTABLE, B is log2 of the hash size (which is always a
///  power of 2) plus 1, or zero for size zero. If not k, the array size
///  is C. Otherwise, the array size is EXTRAARG _ C.
///
///  (*) For comparisons, k specifies what condition the test should accept
///  (true or false).
///
///  (*) In OP_MMBINI/OP_MMBINK, k means the arguments were flipped
///   (the constant is the first operand).
///
///  (*) All 'skips' (pc++) assume that next instruction is a jump.
///
///  (*) In instructions OP_RETURN/OP_TAILCALL, 'k' specifies that the
///  function builds upvalues, which may need to be closed. C > 0 means
///  the function is vararg, so that its 'func' must be corrected before
///  returning; in this case, (C - 1) is its number of fixed parameters.
///
///  (*) In comparisons with an immediate operand, C signals whether the
///  original operand was a float. (It must be corrected in case of
///  metamethods.)

#[repr(u8)]
#[derive(Debug, Clone, Copy)]
pub enum OpCode {
    MOVE,       //  A B       R[A] := R[B]
    LOADI,      //  A sBx     R[A] := sBx
    LOADF,      //  A sBx     R[A] := (lua_Number)sBx
    LOADK,      //  A Bx      R[A] := K[Bx]
    LOADKX,     //  A         R[A] := K[extra arg]
    LOADFALSE,  //  A         R[A] := false
    LFALSESKIP, //  A         R[A] := false; pc++  (*)
    LOADTRUE,   //  A         R[A] := true
    LOADNIL,    //  A B       R[A], R[A+1], ..., R[A+B] := nil
    GETUPVAL,   //  A B       R[A] := UpValue[B]
    SETUPVAL,   //  A B       UpValue[B] := R[A]
    GETTABUP,   //  A B C     R[A] := UpValue[B][K[C]:string]
    GETTABLE,   //  A B C     R[A] := R[B][R[C]]
    GETI,       //  A B C     R[A] := R[B][C]
    GETFIELD,   //  A B C     R[A] := R[B][K[C]:string]
    SETTABUP,   //  A B C     UpValue[A][K[B]:string] := RK(C)
    SETTABLE,   //  A B C     R[A][R[B]] := RK(C)
    SETI,       //  A B C     R[A][B] := RK(C)
    SETFIELD,   //  A B C     R[A][K[B]:string] := RK(C)
    NEWTABLE,   //  A B C k   R[A] := {}
    SELF,       //  A B C     R[A+1] := R[B]; R[A] := R[B][RK(C):string]
    ADDI,       //  A B sC    R[A] := R[B] + sC
    ADDK,       //  A B C     R[A] := R[B] + K[C]:number
    SUBK,       //  A B C     R[A] := R[B] - K[C]:number
    MULK,       //  A B C     R[A] := R[B] * K[C]:number
    MODK,       //  A B C     R[A] := R[B] % K[C]:number
    POWK,       //  A B C     R[A] := R[B] ^ K[C]:number
    DIVK,       //  A B C     R[A] := R[B] / K[C]:number
    IDIVK,      //  A B C     R[A] := R[B] // K[C]:number
    BANDK,      //  A B C     R[A] := R[B] & K[C]:integer
    BORK,       //  A B C     R[A] := R[B] | K[C]:integer
    BXORK,      //  A B C     R[A] := R[B] ~ K[C]:integer
    SHRI,       //  A B sC    R[A] := R[B] >> sC
    SHLI,       //  A B sC    R[A] := sC << R[B]
    ADD,        //  A B C     R[A] := R[B] + R[C]
    SUB,        //  A B C     R[A] := R[B] - R[C]
    MUL,        //  A B C     R[A] := R[B] * R[C]
    MOD,        //  A B C     R[A] := R[B] % R[C]
    POW,        //  A B C     R[A] := R[B] ^ R[C]
    DIV,        //  A B C     R[A] := R[B] / R[C]
    IDIV,       //  A B C     R[A] := R[B] // R[C]
    BAND,       //  A B C     R[A] := R[B] & R[C]
    BOR,        //  A B C     R[A] := R[B] | R[C]
    BXOR,       //  A B C     R[A] := R[B] ~ R[C]
    SHL,        //  A B C     R[A] := R[B] << R[C]
    SHR,        //  A B C     R[A] := R[B] >> R[C]
    MMBIN,      //  A B C     call C metamethod over R[A] and R[B]  (*)
    MMBINI,     //  A sB C k  call C metamethod over R[A] and sB
    MMBINK,     //  A B C k   call C metamethod over R[A] and K[B]
    UNM,        //  A B       R[A] := -R[B]
    BNOT,       //  A B       R[A] := ~R[B]
    NOT,        //  A B       R[A] := not R[B]
    LEN,        //  A B       R[A] := #R[B] (length operator)
    CONCAT,     //  A B       R[A] := R[A].. ... ..R[A + B - 1]
    CLOSE,      //  A         close all upvalues >= R[A]
    TBC,        //  A         mark variable A "to be closed"
    JMP,        //  sJ        pc += sJ
    EQ,         //  A B k     if ((R[A] == R[B]) ~= k) then pc++
    LT,         //  A B k     if ((R[A] <  R[B]) ~= k) then pc++
    LE,         //  A B k     if ((R[A] <= R[B]) ~= k) then pc++
    EQK,        //  A B k     if ((R[A] == K[B]) ~= k) then pc++
    EQI,        //  A sB k    if ((R[A] == sB) ~= k) then pc++
    LTI,        //  A sB k    if ((R[A] < sB) ~= k) then pc++
    LEI,        //  A sB k    if ((R[A] <= sB) ~= k) then pc++
    GTI,        //  A sB k    if ((R[A] > sB) ~= k) then pc++
    GEI,        //  A sB k    if ((R[A] >= sB) ~= k) then pc++
    TEST,       //  A k       if (not R[A] == k) then pc++
    TESTSET,    //  A B k     if (not R[B] == k) then pc++ else R[A] := R[B] (*)
    CALL,       //  A B C     R[A], ... ,R[A+C-2] := R[A](R[A+1], ... ,R[A+B-1])
    TAILCALL,   //  A B C k   return R[A](R[A+1], ... ,R[A+B-1])
    RETURN,     //  A B C k   return R[A], ... ,R[A+B-2]  (see note)
    RETURN0,    //            return
    RETURN1,    //  A         return R[A]
    FORLOOP,    //  A Bx      update counters; if loop continues then pc-=Bx;
    FORPREP,    //  A Bx      <check values and prepare counters>; if not to run then pc+=Bx+1;
    TFORPREP,   //  A Bx      create upvalue for R[A + 3]; pc+=Bx
    TFORCALL,   //  A C       R[A+4], ... ,R[A+3+C] := R[A](R[A+1], R[A+2]);
    TFORLOOP,   //  A Bx      if R[A+2] ~= nil then { R[A]=R[A+2]; pc -= Bx }
    SETLIST,    //  A B C k   R[A][C+i] := R[A+i], 1 <= i <= B
    CLOSURE,    //  A Bx      R[A] := closure(KPROTO[Bx])
    VARARG,     //  A C       R[A], R[A+1], ..., R[A+C-2] = vararg
    VARARGPREP, //  A         (adjust vararg parameters)
    EXTRAARG,   //  Ax        extra (larger) argument for previous opcode
}

impl Instruction {
    // Offset of the lowest bit of each field
    const OFFSET_OP: u32 = 0;
    const OFFSET_A: u32 = 7;
    const OFFSET_B: u32 = 16;
    const OFFSET_C: u32 = 24;
    const OFFSET_K: u32 = 15;
    const OFFSET_BX: u32 = 15;
    const OFFSET_SBX: u32 = 15;
    const OFFSET_AX: u32 = 7;
    const OFFSET_SJ: u32 = 7;

    const MAX_A: i32 = u8::MAX as i32; // 8 bit
    const MAX_B: i32 = Self::MAX_A; // 8 bit
    const MAX_C: i32 = Self::MAX_A; // 8 bit
    const MAX_BX: i32 = 0x000F_FFF1; // 17 bit
    const MAX_SBX: i32 = Self::MAX_BX - 1; // 17 bit but signed
    const MAX_AX: i32 = 0x0FFF_FFF1; // 25 bit
    const MAX_SJ: i32 = Self::MAX_AX; // 25 bit

    const MASK_OP: u32 = 0x7F << Self::OFFSET_OP;
    const MASK_A: u32 = 0xFF << Self::OFFSET_A;
    const MASK_B: u32 = 0xFF << Self::OFFSET_B;
    const MASK_C: u32 = 0xFF << Self::OFFSET_C;
    const MASK_K: u32 = 0x1 << Self::OFFSET_K;
    const MASK_BX: u32 = (Self::MAX_BX as u32) << Self::OFFSET_BX;
    const MASK_SBX: u32 = Self::MASK_BX;
    const MASK_AX: u32 = (Self::MAX_AX as u32) << Self::OFFSET_AX;
    const MASK_SJ: u32 = Self::MASK_AX;

    fn set_op(code: &mut u32, op: OpCode) {
        debug_assert!(op as u8 <= OpCode::EXTRAARG as u8);
        *code |= (op as u32) << Self::OFFSET_OP;
    }

    fn set_a(code: &mut u32, a: i32) {
        debug_assert!(a <= Self::MAX_A);
        *code |= (a as u32) << Self::OFFSET_A;
    }

    fn set_b(code: &mut u32, b: i32) {
        debug_assert!(b <= Self::MAX_B);
        *code |= (b as u32) << Self::OFFSET_B;
    }

    fn set_c(code: &mut u32, c: i32) {
        debug_assert!(c <= Self::MAX_C);
        *code |= (c as u32) << Self::OFFSET_C;
    }

    fn set_k(code: &mut u32, k: bool) {
        *code |= (k as u32) << Self::OFFSET_K;
    }

    fn set_bx(code: &mut u32, bx: i32) {
        debug_assert!(bx <= Self::MAX_BX);
        *code |= (bx as u32) << Self::OFFSET_BX;
    }

    fn set_sbx(code: &mut u32, sbx: i32) {
        debug_assert!(sbx <= Self::MAX_SBX);
        *code |= (sbx as u32) << Self::OFFSET_SBX;
    }

    fn set_ax(code: &mut u32, ax: i32) {
        debug_assert!(ax <= Self::MAX_AX);
        *code |= (ax as u32) << Self::OFFSET_AX;
    }

    fn set_sj(code: &mut u32, sj: i32) {
        debug_assert!(sj <= Self::MAX_SJ);
        *code |= (sj as u32) << Self::OFFSET_SJ;
    }

    pub fn iabc(op: OpCode, a: i32, b: i32, c: i32, isk: bool) -> Self {
        debug_assert_eq!(op.mode(), OpMode::IABC);
        let mut code = 0;
        Self::set_op(&mut code, op);
        Self::set_a(&mut code, a);
        Self::set_b(&mut code, b);
        Self::set_c(&mut code, c);
        Self::set_k(&mut code, isk);
        Self { code }
    }

    pub fn iabx(op: OpCode, a: i32, bx: i32) -> Self {
        debug_assert_eq!(op.mode(), OpMode::IABx);
        let mut code = 0;
        Self::set_op(&mut code, op);
        Self::set_a(&mut code, a);
        Self::set_bx(&mut code, bx);
        Self { code }
    }

    pub fn iasbx(op: OpCode, a: i32, sbx: i32) -> Self {
        debug_assert_eq!(op.mode(), OpMode::IAsBx);
        let mut code = 0;
        Self::set_op(&mut code, op);
        Self::set_a(&mut code, a);
        Self::set_sbx(&mut code, sbx);
        Self { code }
    }

    pub fn iax(op: OpCode, ax: i32) -> Self {
        debug_assert_eq!(op.mode(), OpMode::IAx);
        let mut code = 0;
        Self::set_op(&mut code, op);
        Self::set_ax(&mut code, ax);
        Self { code }
    }

    pub fn isj(op: OpCode, sj: i32) -> Self {
        debug_assert_eq!(op.mode(), OpMode::IsJ);
        let mut code = 0;
        Self::set_op(&mut code, op);
        Self::set_sj(&mut code, sj);
        Self { code }
    }

    pub fn get_op(&self) -> OpCode {
        let op = (self.code & Self::MASK_OP) as u8;

        // SAFETY: OpCode is marked with repr(u8)
        unsafe { std::mem::transmute(op) }
    }

    pub fn mode(&self) -> OpMode {
        self.get_op().mode()
    }

    fn get_a(&self) -> i32 {
        ((self.code & Self::MASK_A) >> Self::OFFSET_A) as i32
    }

    fn get_b(&self) -> i32 {
        ((self.code & Self::MASK_B) >> Self::OFFSET_B) as i32
    }

    fn get_c(&self) -> i32 {
        ((self.code & Self::MASK_C) >> Self::OFFSET_C) as i32
    }

    fn get_k(&self) -> bool {
        (self.code & Self::MASK_K) != 0
    }

    fn get_bx(&self) -> i32 {
        ((self.code & Self::MASK_BX) >> Self::OFFSET_BX) as i32
    }

    fn get_sbx(&self) -> i32 {
        ((self.code & Self::MASK_SBX) >> Self::OFFSET_SBX) as i32
    }

    fn get_ax(&self) -> i32 {
        ((self.code & Self::MASK_AX) >> Self::OFFSET_AX) as i32
    }

    fn get_sj(&self) -> i32 {
        ((self.code & Self::MASK_SJ) >> Self::OFFSET_SJ) as i32
    }

    pub fn repr_abck(&self) -> (OpCode, i32, i32, i32, bool) {
        (
            self.get_op(),
            self.get_a(),
            self.get_b(),
            self.get_c(),
            self.get_k(),
        )
    }

    pub fn repr_abx(&self) -> (OpCode, i32, i32) {
        (self.get_op(), self.get_a(), self.get_bx())
    }

    pub fn repr_asbx(&self) -> (OpCode, i32, i32) {
        (self.get_op(), self.get_a(), self.get_sbx())
    }

    pub fn repr_ax(&self) -> (OpCode, i32) {
        (self.get_op(), self.get_ax())
    }

    pub fn repr_sj(&self) -> (OpCode, i32) {
        (self.get_op(), self.get_sj())
    }
}

impl OpCode {
    pub fn mode(&self) -> OpMode {
        match &self {
            OpCode::LOADK
            | OpCode::FORLOOP
            | OpCode::FORPREP
            | OpCode::TFORPREP
            | OpCode::TFORLOOP
            | OpCode::CLOSURE => OpMode::IABx,
            OpCode::LOADI | OpCode::LOADF => OpMode::IAsBx,
            OpCode::JMP => OpMode::IsJ,
            OpCode::EXTRAARG => OpMode::IAx,
            _ => OpMode::IABC,
        }
    }
}

impl Debug for Instruction {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self.mode() {
            OpMode::IABC => write!(f, "IABC: {:?}", self.repr_abck()),
            OpMode::IABx => write!(f, "IABx: {:?}", self.repr_abx()),
            OpMode::IAsBx => write!(f, "IAsBx: {:?}", self.repr_asbx()),
            OpMode::IAx => write!(f, "IAx: {:?}", self.repr_ax()),
            OpMode::IsJ => write!(f, "IsJ: {:?}", self.repr_sj()),
        }
    }
}

pub struct UpValue {
    name: Option<String>, // for debug infomation
    on_stack: bool,       // weither this upvalue is on stack
    index: RegIndex,      // stack index
}

/// Lua Function Prototype
pub struct Proto {
    is_vararg: bool,
    param_num: u8, // positional parameter count
    reg_count: u8, // number of registers used by this function

    linedef: u32,     // start define line number
    lastlinedef: u32, // end define line number

    kst: Vec<LValue>,       // constants  (assert(!kst[i].is_manged()))
    code: Vec<Instruction>, // bytecodes
    subfn: Vec<Gc<Proto>>,  // sub functions

    abslines: Vec<(u32, u32)>, // sorted vector record pc -> line, always recorded for error info
    source: Option<Rc<String>>, // source file name, used for debug info
    abslineinfo: Option<Vec<u32>>, // line number of each bytecode, used for debug info
    locvars: Option<Vec<LocVar>>, // local variable name, used for debug info
    upvals: Option<Vec<UpValue>>, // upvalue name, used for debug info
}

impl HeapMemUsed for Proto {
    fn heap_mem_used(&self) -> usize {
        std::mem::size_of::<Self>()
            + self.kst.capacity()
            + self.code.capacity()
            + self.subfn.capacity()
            + self.abslines.capacity()
            + self.abslineinfo.as_ref().map_or(0, |v| v.capacity())
            + self.locvars.as_ref().map_or(0, |l| l.capacity())
            + self.upvals.as_ref().map_or(0, |u| u.capacity())
            + self.source.as_ref().map_or(0, |s| s.capacity())
    }
}

impl Proto {
    pub fn delegate_to(p: &Proto, heap: &mut Heap) {
        for k in p.kst.iter() {
            heap.delegate(*k);
        }

        for sub in p.subfn.iter() {
            heap.delegate_from(*sub);
            Self::delegate_to(sub, heap);
        }
    }

    pub fn nparams(&self) -> i32 {
        self.param_num as i32
    }

    pub fn reg_count(&self) -> usize {
        self.reg_count as usize
    }

    pub fn bytecode(&self) -> &[Instruction] {
        &self.code
    }

    pub fn constants(&self) -> &[LValue] {
        &self.kst
    }

    pub fn subprotos(&self) -> &[Gc<Proto>] {
        &self.subfn
    }

    pub fn upvalues(&self) -> &[LValue] {
        // self.upvals.as_ref().map_or(&[], |v| v.as_slice())
        &[]
    }
}

pub struct LocVar {
    name: String,
    regid: RegIndex, // register index on stack

                     // start_pc: u32, // first point where variable is active
                     // end_pc: u32,   // first point where variable is dead
}

#[derive(Clone)]
enum ExprStatus {
    LitNil,
    LitTrue,
    LitFalse,
    // FnDef(RegIndex), // function definition has stored in local register
    Const(RegIndex), // index in ksts (int / float / string)
    Reg(RegIndex),   // result has stored in local register
                     // UpVal(RegIndex), // expr refers to a upvalue
}

/// Code generation intermidiate state for each Proto
struct GenState {
    pub exprstate: BTreeMap<usize, ExprStatus>, // map expr_address -> expr reg status
    pub regs: usize,                            // current register count (vm stack length)
    pub ksts: Vec<LValue>,                      // constants
    pub upvals: Vec<UpValue>,                   // upvalue name
    pub code: Vec<Instruction>,                 // byte code
    pub subproto: Vec<Proto>,                   // sub functions
    pub locals: Vec<LocVar>,                    // local variable name
    pub srcfile: Rc<String>,                    // source file name
    pub absline: Vec<u32>,                      // line number of each bytecode
}

impl Default for GenState {
    fn default() -> Self {
        GenState::new(1, Rc::new("?".to_string()))
    }
}

/// a short name for Instruction
type Isc = Instruction;

impl GenState {
    pub fn new(regs: usize, srcfile: Rc<String>) -> Self {
        Self {
            exprstate: BTreeMap::new(),
            regs,
            ksts: Vec::with_capacity(8),
            upvals: Vec::with_capacity(8),
            subproto: Vec::with_capacity(2),
            code: Vec::with_capacity(32),
            locals: Vec::with_capacity(8),
            srcfile,
            absline: Vec::with_capacity(8),
        }
    }

    fn gen(&mut self, inst: Instruction, line: u32) {
        self.code.push(inst);
        self.absline.push(line);
    }

    /// Allocate a free register on vm stack and return its index
    fn alloc_free_reg(&mut self) -> RegIndex {
        let idx = self.regs;
        self.regs += 1;
        idx as i32
    }

    fn reserve_reg(&mut self) {
        self.regs -= 1;
    }

    ///
    fn with_free_reg(&mut self, generation: impl Fn(RegIndex) -> ExprStatus) -> ExprStatus {
        let idx = self.alloc_free_reg();
        let s = generation(idx);
        self.reserve_reg();
        s
    }

    // Allocate a constant register and return its index
    fn alloc_const_reg(&mut self, k: LValue) -> i32 {
        let idx = self.ksts.len();
        self.ksts.push(k);
        idx as i32
    }

    fn find_const_reg(&self, k: &LValue) -> Option<i32> {
        for (idx, v) in self.ksts.iter().enumerate() {
            if v == k {
                return Some(idx as i32);
            }
        }
        None
    }

    fn find_const_str(&self, s: &str) -> Option<i32> {
        for (idx, v) in self.ksts.iter().enumerate() {
            if let LValue::String(sw) = v {
                if sw.as_str() == s {
                    return Some(idx as i32);
                }
            }
        }
        None
    }

    fn find_local(&self, name: &str) -> Option<i32> {
        for (idx, v) in self.locals.iter().rev().enumerate() {
            if v.name == name {
                return Some(idx as i32);
            }
        }
        None
    }

    /// load global to register
    fn load_global(&mut self, name: &str, line: u32) -> ExprStatus {
        let sidx = self.alloc_free_reg();
        let cidx = self.find_const_str(name).unwrap(); // must be find
        self.gen(Isc::iabc(OpCode::GETTABUP, sidx, 0, cidx, false), line);
        ExprStatus::Reg(sidx)
    }

    ///
    // fn load_upval_reg(&mut self, name_reg: RegIndex) -> i32 {
    //     let idx = self.upvals.len();
    //     self.gen(Isc::iabc(OpCode::GETUPVAL, a, b, c, isk), line);
    //     idx as i32
    // }

    fn gen_unary_const(&mut self, _op: OpCode, kidx: RegIndex, line: u32) -> ExprStatus {
        let sidx = self.alloc_free_reg();
        self.gen(Isc::iabx(OpCode::LOADK, sidx, kidx), line);
        self.gen(Isc::iabc(OpCode::UNM, sidx, sidx, 0, false), line);
        ExprStatus::Reg(sidx)
    }

    fn gen_unary_reg(&mut self, _op: OpCode, reg: RegIndex, line: u32) -> ExprStatus {
        let sidx = self.alloc_free_reg();
        self.gen(Isc::iabc(OpCode::UNM, sidx, reg, 0, false), line);
        ExprStatus::Reg(sidx)
    }
}

#[derive(Debug)]
pub enum CodeGenErr {
    RegisterOverflow,
}

/// Code generation pass
pub struct CodeGen {
    strip: bool,           // strip debug information
    genstk: Vec<GenState>, // generation state stack
}

// impl PassRunStatus for CodeGen {
//     fn is_err(&self) -> bool {
//         self.chunk.is_none()
//     }

//     fn is_ok(&self) -> bool {
//         self.chunk.is_some()
//     }
// }

// impl PassRunRes<Proto, PassHasNotRun> for CodeGen {
//     fn output(&self) -> Option<&Proto> {
//         self.chunk.as_ref()
//     }

//     fn take_output(&mut self) -> Proto {
//         self.chunk.take().unwrap()
//     }

//     fn error(&self) -> Option<&PassHasNotRun> {
//         todo!()
//     }
// }

impl CodeGen {
    fn new(strip: bool) -> Self {
        Self {
            strip,
            genstk: Vec::new(),
        }
    }

    fn consume(&mut self, mut ast_root: Block) -> Proto {
        debug_assert_eq!(self.genstk.len(), 0);
        // main chunk as a function
        let mut main_param = ParaList::new(true, Vec::new());
        let main_name = Rc::new(std::mem::take(&mut ast_root.chunk));
        let res = self.walk_fn_block(&mut main_param, &mut ast_root, Some(main_name));
        debug_assert_eq!(self.genstk.len(), 0);
        res
    }

    fn walk_fn_block(
        &mut self,
        params: &mut ParaList,
        body: &mut Block,
        name: Option<Rc<String>>,
    ) -> Proto {
        let is_vararg = params.vargs;
        let param_num = params.namelist.len();

        self.enter_func(params, name);
        for stmt in body.stats.iter_mut() {
            self.walk_stmt(stmt);
        }

        if let Some(_retstmt) = body.ret.as_mut() {
            // self.walk_ret_stmt(retstmt);
            todo!()
        }

        let state = self.leave_func();
        let subfns = state.subproto.into_iter().map(Gc::from).collect();

        Proto {
            is_vararg,
            param_num: param_num as u8,
            reg_count: state.regs as u8,
            linedef: *state.absline.first().unwrap(),
            lastlinedef: *state.absline.last().unwrap(),
            kst: state.ksts,
            code: state.code,
            subfn: subfns,

            // optional debug infomation
            abslines: if self.strip { Vec::new() } else { Vec::new() },

            source: if self.strip {
                None
            } else {
                Some(state.srcfile)
            },

            abslineinfo: if self.strip {
                None
            } else {
                Some(state.absline)
            },

            locvars: if self.strip { None } else { Some(state.locals) },

            upvals: if self.strip { None } else { Some(state.upvals) },
        }
    }

    /// vaarag and params, set _ENV
    fn enter_func(&mut self, params: &mut ParaList, name: Option<Rc<String>>) {
        let mut this_frame = if let Some(last_state) = self.genstk.last() {
            GenState::new(
                last_state.regs,
                name.unwrap_or_else(|| last_state.srcfile.clone()),
            )
        } else {
            GenState::default()
        };

        if params.vargs {
            let regrep = params.namelist.len();
            this_frame.gen(Isc::iabc(OpCode::VARARGPREP, regrep as i32, 0, 0, false), 0);
        }

        self.genstk.push(this_frame);
    }

    // TODO:
    /// set debug info for last pc
    /// add close flag for closed variable
    fn leave_func(&mut self) -> GenState {
        self.genstk.pop().unwrap()
    }

    fn take_current_state(&mut self) -> GenState {
        self.genstk.pop().unwrap_or_default()
    }

    fn recover_current_state(&mut self, state: GenState) {
        self.genstk.push(state);
    }

    fn walk_common_expr(&mut self, node: &mut ExprNode) -> ExprStatus {
        let ln = node.lineinfo().0;
        let unique = Self::take_expr_unique(node);
        let expr = node.inner_mut();
        let mut cs = self.take_current_state();

        let s = if let Some(status) = cs.exprstate.get(&unique) {
            status.clone()
        } else {
            let status = match expr {
                Expr::Nil => ExprStatus::LitNil,
                Expr::True => ExprStatus::LitTrue,
                Expr::False => ExprStatus::LitFalse,

                Expr::Int(i) => {
                    let sidx = cs.alloc_free_reg();
                    let cidx = cs
                        .find_const_reg(&LValue::Int(*i))
                        .unwrap_or_else(|| cs.alloc_const_reg(LValue::Int(*i))); // lazy
                    cs.gen(Isc::iabx(OpCode::LOADK, sidx, cidx), ln);
                    ExprStatus::Const(cidx)
                }
                Expr::Float(f) => {
                    let sidx = cs.alloc_free_reg();
                    let cidx = cs
                        .find_const_reg(&LValue::Float(*f))
                        .unwrap_or_else(|| cs.alloc_const_reg(LValue::Float(*f))); // lazy
                    cs.gen(Isc::iabx(OpCode::LOADK, sidx, cidx), ln);
                    ExprStatus::Const(cidx)
                }
                Expr::Literal(s) => {
                    let sidx = cs.alloc_free_reg();
                    let cidx = cs.find_const_str(s).unwrap_or_else(|| {
                        let strval = LValue::from(Gc::from(std::mem::take(s)));
                        cs.alloc_const_reg(strval)
                    }); // lazyly

                    cs.gen(Isc::iabx(OpCode::LOADK, sidx, cidx), ln);
                    ExprStatus::Const(cidx)
                }
                Expr::Ident(id) => {
                    // local variable ?
                    if let Some(regidx) = cs.find_local(id) {
                        return ExprStatus::Reg(regidx);
                    }

                    // upvalue ?
                    for frame in self.genstk.iter().rev() {
                        if let Some(regidx) = frame.find_local(id) {
                            // cs.gen(loadupval)?
                            return ExprStatus::Reg(regidx);
                        }
                    }

                    // FIX:
                    // remove copy
                    cs.alloc_const_reg(LValue::from_wild(Gc::from(id.as_str())));

                    cs.load_global(id, ln)
                }

                Expr::UnaryOp { op, expr } => {
                    let es = self.walk_common_expr(expr);
                    let unop_code = match op {
                        UnOp::Minus => OpCode::UNM,
                        UnOp::Not => OpCode::NOT,
                        UnOp::Length => OpCode::LEN,
                        UnOp::BitNot => OpCode::BNOT,
                    };

                    match es {
                        ExprStatus::Const(kidx) => cs.gen_unary_const(unop_code, kidx, ln),
                        ExprStatus::Reg(reg) => cs.gen_unary_reg(unop_code, reg, ln),
                        _ => unreachable!(),
                    }
                }

                Expr::BinaryOp { lhs, op, rhs } => {
                    let (lst, rst) = (self.walk_common_expr(lhs), self.walk_common_expr(rhs));
                    let destreg = cs.alloc_free_reg();

                    let select_arithmetic_kop = |bop: BinOp| -> OpCode {
                        match bop {
                            BinOp::Add => OpCode::ADDK,
                            BinOp::Minus => OpCode::SUBK,
                            BinOp::Mul => OpCode::MULK,
                            BinOp::Mod => OpCode::MODK,
                            BinOp::Pow => OpCode::POWK,
                            BinOp::Div => OpCode::DIVK,
                            BinOp::IDiv => OpCode::IDIVK,
                            BinOp::BitAnd => OpCode::BANDK,
                            BinOp::BitOr => OpCode::BORK,
                            BinOp::BitXor => OpCode::BXORK,
                            _ => unreachable!(),
                        }
                    };

                    let select_arithemic_op = |bop: BinOp| -> OpCode {
                        match bop {
                            BinOp::Add => OpCode::ADD,
                            BinOp::Minus => OpCode::SUB,
                            BinOp::Mul => OpCode::MUL,
                            BinOp::Mod => OpCode::MOD,
                            BinOp::Pow => OpCode::POW,
                            BinOp::Div => OpCode::DIV,
                            BinOp::IDiv => OpCode::IDIV,
                            BinOp::BitAnd => OpCode::BAND,
                            BinOp::BitOr => OpCode::BOR,
                            BinOp::BitXor => OpCode::BXOR,
                            _ => unreachable!(),
                        }
                    };

                    match (lst, rst) {
                        (ExprStatus::Const(lk), ExprStatus::Const(rk)) => {
                            // load left to dest reg and cover dest reg
                            cs.gen(Isc::iabx(OpCode::LOADK, destreg, lk), ln);
                            cs.gen(
                                Isc::iabc(select_arithmetic_kop(*op), destreg, destreg, rk, false),
                                ln,
                            );
                        }
                        (ExprStatus::Const(lk), ExprStatus::Reg(rk)) => {
                            cs.gen(
                                Isc::iabc(select_arithmetic_kop(*op), destreg, rk, lk, false),
                                ln,
                            );
                        }
                        (ExprStatus::Reg(lk), ExprStatus::Const(rk)) => {
                            cs.gen(
                                Isc::iabc(select_arithmetic_kop(*op), destreg, lk, rk, false),
                                ln,
                            );
                        }
                        (ExprStatus::Reg(lk), ExprStatus::Reg(rk)) => {
                            cs.gen(
                                Isc::iabc(select_arithemic_op(*op), destreg, lk, rk, false),
                                ln,
                            );
                        }
                        _ => unreachable!(),
                    };
                    ExprStatus::Reg(destreg)
                }

                Expr::FuncDefine(def) => {
                    let pto = self.walk_fn_block(&mut def.params, &mut def.body, None);
                    let pidx = cs.subproto.len();
                    cs.subproto.push(pto);
                    let reg = cs.alloc_free_reg();
                    cs.gen(Isc::iabx(OpCode::CLOSURE, reg, pidx as i32), ln);
                    ExprStatus::Reg(reg)
                }

                Expr::Index { prefix: _, key: _ } => todo!("Table Index"),

                Expr::FuncCall(_call) => {
                    // todo
                    todo!("func call")
                }

                Expr::TableCtor(flist) => {
                    // idx of this table
                    let tbidx = cs.alloc_free_reg();
                    cs.gen(Isc::iabc(OpCode::NEWTABLE, tbidx, 0, 0, false), ln);

                    let aryidx = 1; // array in lua start at index 1
                    for field in flist.iter_mut() {
                        let valstatus = self.walk_common_expr(&mut field.val);

                        let valreg = match valstatus {
                            ExprStatus::Const(k) => {
                                let free = cs.alloc_free_reg();
                                cs.gen(Isc::iabx(OpCode::LOADK, free, k), ln);
                                free
                            }

                            ExprStatus::Reg(r) => r,
                            _ => unreachable!(),
                        };

                        if let Some(key) = &mut field.key {
                            let keystatus = self.walk_common_expr(key);
                            match keystatus {
                                ExprStatus::Const(kidx) => {
                                    let free = cs.alloc_free_reg();
                                    cs.gen(Isc::iabx(OpCode::LOADK, free, kidx), ln);
                                    cs.gen(
                                        Isc::iabc(OpCode::SETTABLE, tbidx, free, valreg, false),
                                        ln,
                                    )
                                }
                                ExprStatus::Reg(reg) => cs.gen(
                                    Isc::iabc(OpCode::SETTABLE, tbidx, reg, valreg, false),
                                    ln,
                                ),
                                _ => unreachable!(), // constant fold
                            }
                        } else {
                            let kidx = cs.alloc_const_reg(LValue::Int(aryidx));
                            let free = cs.alloc_free_reg();
                            cs.gen(Isc::iabx(OpCode::LOADK, free, kidx), ln);
                            cs.gen(Isc::iabc(OpCode::SETTABLE, tbidx, free, valreg, false), ln)
                        }
                    }
                    ExprStatus::Reg(tbidx)
                }
                Expr::Dots => unreachable!(),
            };
            cs.exprstate.insert(unique, status.clone());
            status
        };
        self.recover_current_state(cs);
        s
    }

    /// use (line, col) as unique id
    fn take_expr_unique(node: &ExprNode) -> usize {
        let line = node.lineinfo().0 as usize;
        line << (16 + node.lineinfo().1 as usize)
    }

    fn walk_stmt(&mut self, stmt: &mut StmtNode) {
        match stmt.inner_mut() {
            Stmt::Assign { vars: _, exprs: _ } => todo!(),
            Stmt::FuncCall(call) => self.walk_fncall(call),
            Stmt::Lable(_) => todo!(),
            Stmt::Goto(_) => todo!(),
            Stmt::Break => todo!(),
            Stmt::DoEnd(_) => todo!(),
            Stmt::While { exp: _, block: _ } => todo!(),
            Stmt::Repeat { block: _, exp: _ } => todo!(),
            Stmt::IfElse {
                exp: _,
                then: _,
                els: _,
            } => todo!(),
            Stmt::NumericFor {
                iter: _,
                init: _,
                limit: _,
                step: _,
                body: _,
            } => todo!(),
            Stmt::GenericFor {
                iters: _,
                exprs: _,
                body: _,
            } => todo!(),
            Stmt::FnDef {
                pres: _,
                method: _,
                body: _,
            } => todo!(),
            Stmt::LocalVarDecl { names: _, exprs: _ } => todo!(),
            Stmt::Expr(exp) => {
                self.walk_common_expr(&mut ExprNode::new(std::mem::take(exp), stmt.lineinfo()))
            }
        };
    }

    fn walk_fncall(&mut self, call: &mut FuncCall) -> ExprStatus {
        match call {
            FuncCall::MethodCall {
                prefix: _,
                method: _,
                args: _,
            } => {
                todo!()
            }
            FuncCall::FreeFnCall { prefix, args } => {
                let nparam = args.namelist.len();
                let ln = prefix.lineinfo().0;

                let ret_begin;
                let status = self.walk_common_expr(prefix);
                let cs = self.take_current_state();
                let call_inst = match status {
                    ExprStatus::Reg(reg) => {
                        // a : fn index
                        // b : b = 0 => vararg, b > 0 => param count + 1
                        // c : c = 0 => varret, c > 0 => return count + 1
                        ret_begin = reg;
                        Isc::iabc(OpCode::CALL, reg, (nparam + 1) as i32, 0, false)
                    }
                    _ => unreachable!(),
                };
                self.recover_current_state(cs);

                for param in args.namelist.iter_mut() {
                    let status = self.walk_common_expr(&mut ExprNode::new(
                        std::mem::take(param),
                        prefix.lineinfo(),
                    ));
                    let mut cs = self.take_current_state();
                    match status {
                        ExprStatus::Const(k) => {
                            let reg = cs.alloc_free_reg();
                            cs.gen(Isc::iabx(OpCode::LOADK, reg, k), 0);
                        }
                        ExprStatus::Reg(r) => {
                            let reg = cs.alloc_free_reg();
                            cs.gen(Isc::iabc(OpCode::MOVE, reg, r, 0, false), ln)
                        }
                        _ => unreachable!(),
                    }
                    self.recover_current_state(cs);
                }
                let mut cs = self.take_current_state();
                cs.gen(call_inst, ln);
                self.recover_current_state(cs);
                ExprStatus::Reg(ret_begin)
            }
        }
    }
}

impl CodeGen {
    pub fn generate(ast_root: Block, strip: bool) -> Result<Proto, CodeGenErr> {
        let mut gen = Self::new(strip);
        Ok(gen.consume(ast_root))
    }
}

pub struct ChunkDumper {}

impl Default for ChunkDumper {
    fn default() -> Self {
        Self::new()
    }
}

impl ChunkDumper {
    pub fn new() -> Self {
        Self {}
    }

    pub fn dump(_chunk: &Proto, _buf: &mut BufWriter<impl Write>) -> std::io::Result<()> {
        todo!()
    }
}

mod test {

    #[test]
    fn instruction_size_check() {
        use super::Instruction;
        assert_eq!(std::mem::size_of::<Instruction>(), 4);
    }
}
