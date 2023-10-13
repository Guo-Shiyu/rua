use std::{
    collections::{BTreeMap, LinkedList},
    fmt::{Debug, Display},
    io::{BufReader, BufWriter, Read, Write},
    num::NonZeroU32,
    ops::{Deref, DerefMut},
};

use crate::{
    ast::{
        Attribute, BasicBlock, BinOp, Block, Expr, ExprNode, Field, FuncBody, FuncCall, GenericFor,
        NumericFor, ParaList, SrcLoc, Stmt, UnOp,
    },
    heap::{Gc, Heap, HeapMemUsed, MarkAndSweepGcOps},
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

impl Display for OpMode {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            OpMode::IABC => f.write_str("IABC "),
            OpMode::IABx => f.write_str("IABx "),
            OpMode::IAsBx => f.write_str("IAsBx"),
            OpMode::IAx => f.write_str("IAx  "),
            OpMode::IsJ => f.write_str("IsJ  "),
        }
    }
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
#[derive(Debug, Clone, Copy, PartialEq)]
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

impl Display for OpCode {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let buf = format!("{:?}", self);
        f.write_str(buf.as_str())?;
        for _ in buf.len()..12 {
            f.write_str(" ")?;
        }
        Ok(())
    }
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
    const MAX_BX: i32 = 0x0001_FFFF; // 17 bit
    const MAX_SBX: i32 = Self::MAX_BX - 1; // 17 bit but signed
    const MAX_AX: i32 = 0x0FFF_FFF1; // 25 bit
    const MAX_SJ: i32 = Self::MAX_AX; // 25 bit

    const MASK_OP: u32 = 0x7F << Self::OFFSET_OP;
    const MASK_A: u32 = 0xFF << Self::OFFSET_A;
    const MASK_B: u32 = 0xFF << Self::OFFSET_B;
    const MASK_C: u32 = 0xFF << Self::OFFSET_C;
    const MASK_K: u32 = 0x1 << Self::OFFSET_K;
    const MASK_BX: u32 = (u32::MAX) << Self::OFFSET_BX;
    const MASK_SBX: u32 = Self::MASK_BX;
    const MASK_AX: u32 = (u32::MAX) << Self::OFFSET_AX;
    const _MASK_SJ: u32 = Self::MASK_AX;

    fn set_op(code: &mut u32, op: OpCode) {
        debug_assert!(op as u8 <= EXTRAARG as u8);
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
        *code |= (bx as u32) << Self::OFFSET_BX;
    }

    fn set_sbx(code: &mut u32, sbx: i32) {
        Self::set_bx(code, sbx & 0x1_FFFF)
    }

    fn set_ax(code: &mut u32, ax: i32) {
        debug_assert!(ax <= Self::MAX_AX);
        *code |= (ax as u32) << Self::OFFSET_AX;
    }

    fn set_sj(code: &mut u32, sj: i32) {
        debug_assert!(sj <= Self::MAX_SJ);
        *code |= (sj as u32) << Self::OFFSET_SJ;
    }

    pub fn iabc(op: OpCode, a: i32, b: i32, c: i32) -> Self {
        debug_assert_eq!(op.mode(), OpMode::IABC);
        let mut code = 0;
        Self::set_op(&mut code, op);
        Self::set_a(&mut code, a);
        Self::set_b(&mut code, b);
        Self::set_c(&mut code, c);
        Self::set_k(&mut code, false);
        Self { code }
    }

    pub fn iabck(op: OpCode, a: i32, b: i32, c: i32) -> Self {
        debug_assert_eq!(op.mode(), OpMode::IABC);
        let mut code = 0;
        Self::set_op(&mut code, op);
        Self::set_a(&mut code, a);
        Self::set_b(&mut code, b);
        Self::set_c(&mut code, c);
        Self::set_k(&mut code, true);
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
        (self.code as i32 & Self::MASK_SBX as i32) >> Self::OFFSET_SBX
    }

    fn get_ax(&self) -> i32 {
        ((self.code & Self::MASK_AX) >> Self::OFFSET_AX) as i32
    }

    fn get_sj(&self) -> i32 {
        (self.code as i32) >> Self::OFFSET_SJ
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

    pub fn placeholder() -> Self {
        Isc::isj(JMP, 9999)
    }
}

impl OpCode {
    pub fn mode(&self) -> OpMode {
        match &self {
            LOADK | FORLOOP | FORPREP | TFORPREP | TFORLOOP | CLOSURE => OpMode::IABx,
            LOADI | LOADF => OpMode::IAsBx,
            JMP => OpMode::IsJ,
            EXTRAARG => OpMode::IAx,
            _ => OpMode::IABC,
        }
    }
}

impl Display for Instruction {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let mode = self.mode();
        write!(f, "{} ", mode)?;

        match mode {
            OpMode::IABC => {
                let (code, a, b, c, k) = self.repr_abck();
                write!(f, "{:<16}\t{:<3} {:<3} {:<3}", code, a, b, c)?;
                if k {
                    f.write_str("k")
                } else {
                    f.write_str(" ")
                }
            }
            OpMode::IABx => {
                let (code, a, bx) = self.repr_abx();
                write!(f, "{code:<16}\t{:<3} {:<3}    ", a, bx)
            }
            OpMode::IAsBx => {
                let (code, a, sbx) = self.repr_asbx();
                write!(f, "{code:<16}\t{a:<3} {sbx:<3}     ")
            }
            OpMode::IAx => {
                let (code, ax) = self.repr_ax();
                write!(f, "{code:<16}\t{ax:<3}      ")
            }
            OpMode::IsJ => {
                let (code, sj) = self.repr_sj();
                write!(f, "{code:<16}\t{sj:<3}         ")
            }
        }
    }
}

/// Upval information (in runtime)
#[derive(Clone)]
pub enum UpvalDecl {
    // upvalue is outter function's local variable
    OnStack { name: String, register: RegIndex },

    // upvalue in outter function's upvalue list
    InUpList { name: String, offset: RegIndex },

    // _ENV itself
    Env,
}

impl UpvalDecl {
    fn name(&self) -> &str {
        match self {
            UpvalDecl::OnStack { name, register: _ } => name.as_str(),
            UpvalDecl::InUpList { name, offset: _ } => name.as_str(),
            UpvalDecl::Env => "_ENV",
        }
    }
}

/// Lua Function Prototype
pub struct Proto {
    vararg: bool,
    nparam: u8, // positional parameter count
    nreg: u8,   // number of registers used by this function

    begline: u32, // begin define line number
    endline: u32, // end define line number

    kst: Vec<LValue>,       // constants
    code: Vec<Instruction>, // bytecodes
    subfn: Vec<Gc<Proto>>,  // sub functions
    pcline: Vec<u32>,       // line number of each instruction
    updecl: Vec<UpvalDecl>, // upvalue information

    source: LValue,           // source file name, used for debug info
    locvars: Vec<LocVarDecl>, // local variable name, used for debug info
}

impl HeapMemUsed for Proto {
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
            + self
                .subfn
                .iter()
                .fold(self.subfn.capacity(), |acc, f| acc + f.heap_mem_used())
    }
}

impl MarkAndSweepGcOps for Proto {
    fn delegate_to(&mut self, heap: &mut Heap) {
        heap.delegate(&mut self.source);
        self.kst.iter_mut().for_each(|k| heap.delegate(k));
        self.subfn.iter_mut().for_each(|s| s.delegate_to(heap));
    }

    fn mark_newborned(&self, white: crate::heap::GcColor) {
        self.source.mark_newborned(white);
        self.kst.iter().for_each(|k| k.mark_newborned(white));
        self.subfn.iter().for_each(|s| s.mark_newborned(white));
    }

    fn mark_reachable(&self) {
        self.source.mark_reachable();
        self.kst.iter().for_each(|k| k.mark_reachable());
        self.subfn.iter().for_each(|s| s.mark_reachable());
    }

    fn mark_untouched(&self) {
        todo!()
    }
}

impl Display for Proto {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        self.display_fmt(f)?;
        for each in self.subfn.iter() {
            writeln!(f)?;
            each.display_fmt(f)?;
        }
        Ok(())
    }
}

impl Debug for Proto {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        self.debug_fmt(f)?;
        for each in self.subfn.iter() {
            writeln!(f)?;
            each.debug_fmt(f)?;
        }
        Ok(())
    }
}

impl Proto {
    pub fn nparams(&self) -> i32 {
        self.nparam as i32
    }

    pub fn nreg(&self) -> usize {
        self.nreg as usize
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

    fn basic_fmt(&self, f: &mut std::fmt::Formatter<'_>) -> Result<(), std::fmt::Error> {
        debug_assert!(self.source.is_str());
        writeln!(
            f,
            "function <{}:{},{}> ({} instructions at 0x{:X})",
            self.source,
            self.begline,
            self.endline,
            self.code.len(),
            self as *const Proto as usize
        )?;
        if self.vararg {
            f.write_str("vararg params, ")?;
        } else {
            write!(f, "{} params, ", self.nparam)?;
        }
        writeln!(
            f,
            "{} slots, {} upvalue, {} locals, {}, constants, {} functions",
            self.nreg,
            self.updecl.len(),
            self.locvars.len(),
            self.kst.len(),
            self.subfn.len()
        )?;

        for (idx, code) in self.code.iter().enumerate() {
            let line = self.pcline.get(idx).unwrap_or(&0);
            writeln!(f, "\t{idx}\t[{}]\t{:?>8} ;", line, code)?;
        }

        Ok(())
    }

    fn debug_fmt(&self, f: &mut std::fmt::Formatter<'_>) -> Result<(), std::fmt::Error> {
        self.basic_fmt(f)?;

        let self_addr = self as *const Proto as usize;
        writeln!(f, "{} constants for 0x{:X}", self.kst.len(), self_addr)?;
        for (idx, k) in self.kst.iter().enumerate() {
            writeln!(f, "\t{idx}\t{k}")?;
        }

        writeln!(f, "{} locals for 0x{:X}", self.locvars.len(), self_addr)?;
        for (idx, loc) in self.locvars.iter().enumerate() {
            writeln!(f, "\t{idx}\t\"{}\"", loc.name.as_str())?;
        }

        writeln!(f, "{} upvalues for 0x{:X}", self.updecl.len(), self_addr)?;
        for (idx, up) in self.updecl.iter().enumerate() {
            writeln!(f, "\t{idx}\t\"{}\"", up.name())?;
        }

        Ok(())
    }

    fn display_fmt(&self, f: &mut std::fmt::Formatter<'_>) -> Result<(), std::fmt::Error> {
        self.basic_fmt(f)
    }
}

#[derive(Default)]
pub struct LocVarDecl {
    name: String,
    reg: RegIndex, // register index on stack
    start_pc: u32, // first pc where variable is active
    end_pc: u32,   // first pc where variable is dead
}

#[derive(Clone)]
pub enum ExprStatus {
    LitNil,
    LitTrue,
    LitFalse,
    LitInt(i64),
    LitFlt(f64),
    Call(RegIndex), // register index of first return value
    Kst(RegIndex),  // index of constants
    Reg(RegIndex),  // index of local register
}

use OpCode::*;

/// Code generation intermidiate state for each Proto
pub struct GenState {
    pub lables: BTreeMap<String, u32>,       // map lable -> pc
    pub jumpbp: Vec<(u32, String)>,          // jump backpatch (iscidx, lable), used for `goto`
    pub loopbp: LinkedList<Vec<(u32, u32)>>, // loop backpatch (iscidx, pc), used for `break`
    pub regs: RegIndex,                      // current register count (vm stack length)
    pub ksts: Vec<LValue>,                   // constants
    pub upvals: Vec<UpvalDecl>,              // upvalue declration
    pub code: Vec<Instruction>,              // byte code
    pub subproto: Vec<Proto>,                // sub functions
    pub locstate: Vec<Vec<LocVarDecl>>,      // local variable infomation for each basic block
    pub local: Vec<LocVarDecl>,              // all local variable infomation
    pub srcfile: LValue,                     // source file name
    pub absline: Vec<u32>,                   // line number of each bytecode
}

impl Default for GenState {
    fn default() -> Self {
        GenState::new(CodeGen::ANONYMOUS.into())
    }
}

/// a short name for Instruction
type Isc = Instruction;

impl GenState {
    pub fn new(srcfile: LValue) -> Self {
        Self {
            lables: BTreeMap::new(),
            jumpbp: Vec::default(),
            loopbp: LinkedList::default(),
            regs: 0,
            ksts: Vec::with_capacity(4),
            upvals: Vec::with_capacity(8),
            subproto: Vec::with_capacity(2),
            code: Vec::with_capacity(32),
            locstate: Vec::with_capacity(4),
            local: Vec::with_capacity(4),
            srcfile,
            absline: Vec::with_capacity(8),
        }
    }

    pub fn with_env(srcfile: LValue) -> Self {
        let mut res = Self::new(srcfile);
        res.upvals.push(UpvalDecl::Env);
        res
    }

    fn cur_pc(&self) -> u32 {
        self.code.len() as u32
    }

    fn enter_loop(&mut self) {
        self.loopbp.push_back(Vec::default());
    }

    fn leave_loop(&mut self) {
        let loop_end = self.code.len();
        if let Some(bps) = self.loopbp.pop_back() {
            for (iscidx, pc) in bps.into_iter() {
                debug_assert!(self.code.get_mut(iscidx as usize).is_some());
                let step = loop_end as i32 - pc as i32;
                self.emit_backpatch(iscidx as usize, Isc::isj(JMP, step));
            }
        }
    }

    fn emit(&mut self, inst: Instruction, line: u32) {
        self.code.push(inst);
        self.absline.push(line);
    }

    fn emit_placeholder(&mut self, line: u32) {
        self.emit(Isc::placeholder(), line);
    }

    fn next_free_reg(&self) -> RegIndex {
        self.regs
    }

    /// Allocate a free register on vm stack and return its index
    fn alloc_free_reg(&mut self) -> RegIndex {
        let idx = self.regs;
        self.regs += 1;
        idx
    }

    /// Free last allocated register on vm stack, return next available register
    fn free_reg(&mut self) -> RegIndex {
        self.regs -= 1;
        self.regs
    }

    // Allocate a constant register and return its index
    fn alloc_const_reg(&mut self, k: LValue) -> i32 {
        let idx = self.ksts.len();
        self.ksts.push(k);
        idx as i32
    }

    /// Find string in constants and return its index, this will alloc a const string if not found.
    fn find_or_create_kstr(&mut self, s: &str) -> RegIndex {
        for (idx, v) in self.ksts.iter().enumerate() {
            if let LValue::String(sw) = v {
                if sw.as_str() == s {
                    return idx as RegIndex;
                }
            }
        }
        self.alloc_const_reg(LValue::from(Gc::from(s)))
    }

    /// Find local variable and return its index
    fn find_local_decl(&self, name: &str) -> Option<RegIndex> {
        for block in self.locstate.iter().rev() {
            for var in block.iter().rev() {
                if var.name == name {
                    return Some(var.reg);
                }
            }
        }
        None
    }

    /// Load constant to register
    fn load_const(&mut self, kidx: i32, ln: u32) -> i32 {
        let reg = self.alloc_free_reg();
        self.emit(Isc::iabx(LOADK, reg, kidx), ln);
        reg
    }

    fn emit_local_decl(&mut self, name: String, status: ExprStatus, lineinfo: (u32, u32)) {
        let mut vreg = self.alloc_free_reg();

        match status {
            ExprStatus::LitNil => self.emit(Isc::iabc(LOADNIL, vreg, 0, 0), lineinfo.0),
            ExprStatus::LitTrue => self.emit(Isc::iabc(LOADTRUE, vreg, 0, 0), lineinfo.0),
            ExprStatus::LitFalse => self.emit(Isc::iabc(LOADFALSE, vreg, 0, 0), lineinfo.0),
            ExprStatus::LitInt(i) => {
                if i.abs() > Isc::MAX_SBX.abs() as i64 {
                    let kreg = self.alloc_const_reg(LValue::Int(i));
                    self.emit(Isc::iabx(LOADK, vreg, kreg), lineinfo.0)
                } else {
                    self.emit(Isc::iasbx(LOADI, vreg, i as i32), lineinfo.0)
                }
            }
            ExprStatus::LitFlt(f) => {
                // FIX ME:
                // use LOADF for small float.

                let kreg = self.alloc_const_reg(LValue::Float(f));
                self.emit(Isc::iabx(LOADK, vreg, kreg), lineinfo.0)
            }
            ExprStatus::Kst(reg) => {
                self.emit(Isc::iabx(LOADK, vreg, reg), lineinfo.0);
            }

            ExprStatus::Call(reg) | ExprStatus::Reg(reg) => {
                vreg = reg;
                self.free_reg();
            }
        }

        let locdecl = LocVarDecl {
            name,
            reg: vreg,
            start_pc: self.cur_pc(),
            end_pc: self.cur_pc(),
        };
        self.locstate.last_mut().unwrap().push(locdecl);
    }

    fn set_recover_point(&mut self, line: u32) -> (usize, u32) {
        let (forprep_idx, pc) = (self.code.len(), self.cur_pc());
        self.emit(Isc::placeholder(), line);
        (forprep_idx, pc)
    }

    fn emit_backpatch(&mut self, iscidx: usize, isc: Isc) {
        if let Some(dest) = self.code.get_mut(iscidx) {
            *dest = isc
        } else {
            unreachable!("there must be a instruction in backpatch point.")
        }
    }

    fn emit_index_local(
        &mut self,
        keys: ExprStatus,
        ln: u32,
        pre_reg: RegIndex,
        dest: RegIndex,
    ) -> ExprStatus {
        match keys {
            ExprStatus::Kst(kreg) => {
                if let Some(imidiate) = self.try_emit_inmidiate_index(kreg) {
                    self.emit(Isc::iabc(GETI, dest, imidiate, 0), ln);
                } else {
                    self.emit(Isc::iabck(GETFIELD, dest, pre_reg, kreg), ln)
                };
            }
            ExprStatus::Reg(reg) => {
                self.emit(Isc::iabc(GETFIELD, dest, pre_reg, reg), ln);
            }
            _ => unreachable!(),
        };
        ExprStatus::Reg(dest)
    }

    fn try_emit_inmidiate_index(&mut self, kreg: RegIndex) -> Option<i32> {
        debug_assert!((kreg as usize) < self.ksts.len());
        // SAFETY: constant must exist
        let val = unsafe { self.ksts.get_unchecked(kreg as usize) };
        debug_assert!(!val.is_managed());
        if let LValue::Int(i) = val {
            if *i < Isc::MAX_B as i64 {
                Some(*i as i32)
            } else {
                None
            }
        } else {
            None
        }
    }

    /// Load an Expr to local if it is not in local reg.
    fn try_load_expr_to_local(&mut self, es: ExprStatus, ln: u32) -> RegIndex {
        match es {
            ExprStatus::LitNil => {
                let free = self.alloc_free_reg();
                self.emit(Isc::iabc(LOADNIL, free, 0, 0), ln);
                free
            }
            ExprStatus::LitTrue => {
                let free = self.alloc_free_reg();
                self.emit(Isc::iabc(LOADTRUE, free, 0, 0), ln);
                free
            }
            ExprStatus::LitFalse => {
                let free = self.alloc_free_reg();
                self.emit(Isc::iabc(LOADFALSE, free, 0, 0), ln);
                free
            }
            ExprStatus::LitInt(i) => {
                let free = self.alloc_free_reg();
                self.emit(Isc::iasbx(LOADI, free, i as i32), ln);
                free
            }
            ExprStatus::LitFlt(_f) => {
                // let free = self.alloc_free_reg();
                // self.emit(Isc::, line)
                todo!()
            }
            ExprStatus::Call(c) => c,
            ExprStatus::Kst(kreg) => self.load_const(kreg, ln),
            ExprStatus::Reg(r) => r,
        }
    }

    fn peek_const_expr(exp: &Expr) -> Option<ExprStatus> {
        match exp {
            Expr::Nil => Some(ExprStatus::LitNil),
            Expr::True => Some(ExprStatus::LitTrue),
            Expr::False => Some(ExprStatus::LitFalse),
            Expr::Int(i) => {
                if *i as i32 > Isc::MAX_SBX {
                    None
                } else {
                    Some(ExprStatus::LitInt(*i))
                }
            }
            Expr::Float(_f) => None, // TODO
            _ => None,
        }
    }
}

/// Expression generation context to optimize
enum ExprGenCtx {
    // extra expr in multi assignment
    Ignore,

    // expr needs to alloc 1 register to store result
    Allocate,

    // there is already a free register for expr to use (for function argument)
    NonRealloc { dest: RegIndex },

    // expr is the single value to be returned
    PotentialTailCall,

    // intermidiate table may be cached
    MultiLevelTableIndex { depth: u32 },
}

impl ExprGenCtx {
    fn must_use(reg: RegIndex) -> Self {
        Self::NonRealloc { dest: reg }
    }
}

type Ctx = ExprGenCtx;

struct BranchBackPatchPoint {
    cond_jmp_idx: u32,
    then_end_jmp_idx: Option<NonZeroU32>,
    else_entry_pc: Option<NonZeroU32>,
    def_end_pc: u32,
}

#[derive(Debug)]
pub enum CodeGenErr {
    TooManyLocalVariable,
    RegisterOverflow,
    RepeatedLable { lable: String },
    NonexistedLable { lable: String },
    BreakNotInLoopBlock,
}

/// Code generation pass
pub struct CodeGen {
    strip: bool,                  // strip debug information
    genstk: LinkedList<GenState>, // generation state stack
    cgs: GenState,                // current generation state
}

impl Deref for CodeGen {
    type Target = GenState;

    fn deref(&self) -> &Self::Target {
        &self.cgs
    }
}

impl DerefMut for CodeGen {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.cgs
    }
}

impl CodeGen {
    const ANONYMOUS: &'static str = "?";

    fn new(strip: bool) -> Self {
        Self {
            strip,
            genstk: LinkedList::new(),
            cgs: GenState::default(),
        }
    }

    fn consume(&mut self, ast_root: SrcLoc<Block>) -> Result<Proto, CodeGenErr> {
        debug_assert_eq!(self.genstk.len(), 0);

        // take chunk name, construct an empty para list
        let name = LValue::from(ast_root.chunk.as_str());
        let plist = ParaList {
            vargs: true,
            namelist: Vec::new(),
        };

        let mut res = self.walk_fn_body(name, plist, ast_root, false, true)?;

        if self.strip {
            Self::strip_src_info(&mut res, Self::ANONYMOUS.into())
        }

        debug_assert!(!res.updecl.is_empty());
        debug_assert_eq!(self.cgs.locstate.len(), 0);
        debug_assert_eq!(self.genstk.len(), 0);
        Ok(res)
    }

    fn walk_fn_body(
        &mut self,
        name: LValue,
        params: ParaList,
        body: SrcLoc<Block>,
        selfcall: bool,
        is_main_chunk: bool,
    ) -> Result<Proto, CodeGenErr> {
        // prepare another GenState
        let mut new_state = GenState::new(name);
        std::mem::swap(&mut self.cgs, &mut new_state);
        self.genstk.push_back(new_state);

        // vararg
        if params.vargs {
            let nparam = params.namelist.len();
            self.emit(Isc::iabc(VARARGPREP, nparam as i32, 0, 0), body.def_begin());
        }

        // reserve space for self call
        if selfcall {
            todo!("self call ")
            // let free = self.alloc_free_reg();
            // self.emit_local_decl("self".to_string(), ExprStatus::Reg(free), (0, 0));
        }

        // init _ENV upvalue
        if is_main_chunk {
            self.upvals.push(UpvalDecl::Env);
        }

        self.walk_basic_block(body, true)?;

        // backpatch goto
        while let Some((pc, lable)) = self.jumpbp.pop() {
            let index = pc as usize;
            if let Some(dest) = self.lables.get(&lable) {
                let step = (*dest as i64) - pc as i64;
                self.code[index].code = Isc::isj(JMP, step as i32).code;
            } else {
                return Err(CodeGenErr::NonexistedLable { lable });
            }
        }

        // TODO:
        // Check <close> variable and generate CLOSE instruction
        // Check <const> variable

        let subfn = std::mem::take(&mut self.subproto)
            .into_iter()
            .map(Gc::from)
            .collect();

        let res = Proto {
            vararg: params.vargs,
            nparam: params.namelist.len() as u8,
            nreg: self.regs as u8,
            begline: self.absline.first().copied().unwrap_or(0),
            endline: self.absline.last().copied().unwrap_or(0),
            kst: std::mem::take(&mut self.ksts),
            code: std::mem::take(&mut self.code),
            subfn,
            source: self.srcfile,
            pcline: std::mem::take(&mut self.absline),
            locvars: std::mem::take(&mut self.local),
            updecl: std::mem::take(&mut self.upvals),
        };

        // reset to previous GenState
        if let Some(prev) = self.genstk.pop_back() {
            let _ = std::mem::replace(&mut self.cgs, prev);
        } else {
            unreachable!()
        }

        Ok(res)
    }

    /// Strip debug infomation in proto.
    fn strip_src_info(p: &mut Proto, anonymous: LValue) {
        p.begline = 0;
        p.endline = 0;

        for up in p.updecl.iter_mut() {
            match up {
                UpvalDecl::OnStack { name, register: _ } => {
                    let _ = std::mem::take(name);
                }
                UpvalDecl::InUpList { name, offset: _ } => {
                    let _ = std::mem::take(name);
                }
                UpvalDecl::Env => {}
            }
        }
        let _ = std::mem::take(&mut p.locvars);
        let _ = std::mem::take(&mut p.pcline);

        p.source = anonymous;
        p.subfn
            .iter_mut()
            .for_each(|p| Self::strip_src_info(p, anonymous))
    }

    fn walk_basic_block(
        &mut self,
        body: SrcLoc<Block>,
        must_return: bool,
    ) -> Result<(), CodeGenErr> {
        self.locstate.push(Vec::with_capacity(4));

        let (body_def_end, body) = (body.def_end(), body.into_inner());
        for stmt in body.stats.into_iter() {
            self.walk_stmt(*stmt)?;
        }
        self.walk_return(body.ret, body_def_end, must_return)?;

        if let Some(ls) = self.locstate.pop() {
            self.local.extend(ls);
        } else {
            unreachable!()
        }
        Ok(())
    }

    fn walk_stmt(&mut self, stmt: SrcLoc<Stmt>) -> Result<(), CodeGenErr> {
        let lineinfo = stmt.lineinfo;
        match stmt.into_inner() {
            Stmt::Assign { vars, exprs } => {
                self.walk_assign_stmt(vars, exprs)?;
            }
            Stmt::FuncCall(call) => {
                let next = self.alloc_free_reg();
                let _ = self.walk_fn_call(call, next, 0, false);
                self.free_reg();
            }
            Stmt::Lable(lable) => {
                let dest = self.cur_pc();
                if self.lables.contains_key(lable.as_str()) {
                    return Err(CodeGenErr::RepeatedLable { lable });
                } else {
                    self.lables.insert(lable, dest);
                }
            }
            Stmt::Goto(lable) => {
                if let Some(dest) = self.lables.get(&lable) {
                    let step = (*dest as i64) - self.cur_pc() as i64;
                    self.emit(Isc::isj(JMP, step as i32), lineinfo.0);
                } else {
                    // set a placeholder in code series
                    self.emit(Isc::isj(JMP, 0), lineinfo.0);
                    let pc = self.cur_pc();
                    self.jumpbp.push((pc, lable));
                }
            }
            Stmt::Break => {
                let (idx, pc) = (self.code.len(), self.cur_pc());
                if let Some(lop) = self.loopbp.back_mut() {
                    lop.push((idx as u32, pc));
                    self.emit(Isc::placeholder(), 0);
                } else {
                    return Err(CodeGenErr::BreakNotInLoopBlock);
                }
            }
            Stmt::DoEnd(block) => {
                self.walk_basic_block(*block, false)?;
            }
            Stmt::While { exp, block } => {
                self.walk_while_loop(*exp, *block)?;
            }
            Stmt::Repeat { block, exp } => {
                self.walk_repeat_loop(*exp, *block)?;
            }
            Stmt::IfElse {
                cond: exp,
                then,
                els,
            } => {
                self.walk_branch_stmt(*exp, *then, els)?;
            }
            Stmt::NumericFor(num) => {
                self.walk_numberic_loop(*num)?;
            }
            Stmt::GenericFor(gen) => {
                self.walk_generic_for(*gen)?;
            }
            Stmt::FnDef { pres, method, body } => {
                self.walk_fn_def(pres, method, *body)?;
            }
            Stmt::LocalVarDecl { names, exprs } => {
                self.walk_local_decl(names, exprs, lineinfo)?;
            }
            Stmt::Expr(exp) => {
                let _ = self.walk_common_expr(*exp, Ctx::Ignore)?;
            }
        };
        Ok(())
    }

    fn walk_repeat_loop(
        &mut self,
        cond: SrcLoc<Expr>,
        block: SrcLoc<Block>,
    ) -> Result<(), CodeGenErr> {
        let def = cond.lineinfo.0;
        self.enter_loop();
        self.walk_basic_block(block, false)?;
        self.leave_loop();
        let cond_reg = {
            let s = self.walk_common_expr(cond, Ctx::Allocate)?;
            self.try_load_expr_to_local(s, def)
        };
        self.emit(Isc::iabc(TEST, cond_reg, true as i32, 0), def);
        Ok(())
    }

    fn walk_while_loop(
        &mut self,
        cond: SrcLoc<Expr>,
        block: SrcLoc<Block>,
    ) -> Result<(), CodeGenErr> {
        self.enter_loop();
        let ln = cond.lineinfo.0;
        let cond_reg = {
            let s = self.walk_common_expr(cond, Ctx::Allocate)?;
            self.try_load_expr_to_local(s, ln)
        };
        self.emit(Isc::iabc(TEST, cond_reg, false as i32, 0), ln);
        let loop_begin = self.set_recover_point(ln);
        self.walk_basic_block(block, false)?;
        self.leave_loop();

        let step = (self.cur_pc() - loop_begin.1) as i32;
        self.emit_backpatch(loop_begin.0, Isc::isj(JMP, step));
        Ok(())
    }

    fn walk_numberic_loop(&mut self, n: NumericFor) -> Result<(), CodeGenErr> {
        let mut init_reg = {
            let line = n.init.def_begin();
            let s = self.walk_common_expr(*n.init, Ctx::Allocate)?;
            self.try_load_expr_to_local(s, line)
        };
        let mut limit_reg = {
            let line = n.limit.def_begin();
            let s = self.walk_common_expr(*n.limit, Ctx::Allocate)?;
            self.try_load_expr_to_local(s, line)
        };
        let mut step_reg = {
            let line = n.step.def_begin();
            let s = self.walk_common_expr(*n.step, Ctx::Allocate)?;
            self.try_load_expr_to_local(s, line)
        };

        let loopdef = n.body.lineinfo;
        let offset = (limit_reg - init_reg, step_reg - limit_reg);
        if offset.0 != offset.1 || offset.0 != 1 {
            let new_init = self.alloc_free_reg();
            let new_limit = self.alloc_free_reg();
            let new_step = self.alloc_free_reg();

            let mut reset_loop_reg = |dest, src| {
                self.emit(Isc::iabc(MOVE, dest, src, 0), loopdef.0);
            };
            reset_loop_reg(new_init, init_reg);
            reset_loop_reg(new_limit, limit_reg);
            reset_loop_reg(new_step, step_reg);
            init_reg = new_init;
            limit_reg = new_limit;
            step_reg = new_step;
        }
        debug_assert!(step_reg - init_reg == 2);

        let (forprep_idx, recover_pc) = self.set_recover_point(loopdef.0);

        // let iter_reg = self.alloc_free_reg();
        // let iter_begin_pc = self.cur_pc();

        // loop block
        self.enter_loop();
        self.walk_basic_block(*n.body, false)?;
        self.leave_loop();

        // loop end
        self.emit(Isc::iabc(FORLOOP, init_reg, limit_reg, 0), loopdef.1);

        let step = (self.cur_pc() - recover_pc) as i32;
        self.emit_backpatch(forprep_idx, Isc::iabx(FORPREP, init_reg, step));

        // TODO:
        // iter is upvalue?
        // let iter_end_pc = self.cur_pc();

        Ok(())
    }

    fn walk_generic_for(&mut self, g: GenericFor) -> Result<(), CodeGenErr> {
        let def = g.body.lineinfo;

        let niter = g.iters.len();
        // alloc 4 register to store loop state
        let state_reg = self.alloc_free_reg();
        for _ in 0..3 {
            self.alloc_free_reg();
        }
        let (iscidx, recover_pc) = self.set_recover_point(def.0);

        // treat iters as local var decl
        let vars = g.iters.into_iter().map(|name| (name, None)).collect();
        self.walk_local_decl(vars, g.exprs, g.body.lineinfo)?;

        // loop body
        self.enter_loop();
        self.walk_basic_block(*g.body, false)?;

        self.emit(Isc::iabc(TFORCALL, state_reg, 0, niter as i32), def.1);
        let step = (self.cur_pc() - recover_pc) as i32;
        self.emit(Isc::iabx(TFORLOOP, state_reg, step), def.1);
        self.emit_backpatch(iscidx, Isc::iabx(TFORPREP, state_reg, step - 2));
        self.leave_loop();

        // TODO:
        // miss a CLOSE isc here.
        Ok(())
    }

    fn walk_branch_stmt(
        &mut self,
        exp: SrcLoc<Expr>,
        then: SrcLoc<Block>,
        els: Option<BasicBlock>,
    ) -> Result<(), CodeGenErr> {
        let cond_def = exp.def_begin();
        let cond = self.walk_common_expr(exp, Ctx::Allocate)?;
        let reg = self.try_load_expr_to_local(cond, cond_def);

        self.emit(Isc::iabck(TEST, reg, 0, 0), cond_def);

        let mut branch = BranchBackPatchPoint {
            cond_jmp_idx: self.cur_pc(),
            then_end_jmp_idx: None,
            else_entry_pc: None,
            def_end_pc: 0,
        };
        // place holder for JMP to else block entry
        self.emit_placeholder(cond_def);

        let then_defend = then.def_end();
        self.walk_basic_block(then, false)?;

        if let Some(bk) = els {
            // place holder fpr JMP to else block end
            branch.then_end_jmp_idx = Some(unsafe { NonZeroU32::new_unchecked(self.cur_pc()) });
            self.emit_placeholder(then_defend);

            branch.else_entry_pc = Some(unsafe { NonZeroU32::new_unchecked(self.cur_pc()) });
            self.walk_basic_block(*bk, false)?;
        }
        branch.def_end_pc = self.cur_pc();

        // back patch JMP
        if let Some(pc) = branch.else_entry_pc {
            let pc: u32 = pc.into();

            self.emit_backpatch(
                unsafe { Into::<u32>::into(branch.then_end_jmp_idx.unwrap_unchecked()) } as usize,
                Isc::isj(JMP, (branch.def_end_pc - pc) as i32),
            );

            self.emit_backpatch(
                branch.cond_jmp_idx as usize,
                Isc::isj(JMP, (pc - branch.cond_jmp_idx) as i32),
            );
        } else {
            self.emit_backpatch(
                branch.cond_jmp_idx as usize,
                Isc::isj(JMP, (branch.def_end_pc - branch.cond_jmp_idx) as i32),
            );
        };

        Ok(())
    }

    fn walk_fn_def(
        &mut self,
        pres: Vec<SrcLoc<String>>,
        method: Option<Box<SrcLoc<String>>>,
        body: SrcLoc<FuncBody>,
    ) -> Result<(), CodeGenErr> {
        let (defln, body) = (body.def_begin(), body.into_inner());

        let mut fnname = pres
            .iter()
            .fold(String::with_capacity(32), |mut acc, next| {
                acc.extend(next.chars());
                acc
            });

        if let Some(ref metd) = method {
            fnname.extend(metd.chars())
        }

        let pre_name_kregs = pres
            .iter()
            .map(|s| self.find_or_create_kstr(s))
            .collect::<Vec<_>>();

        let mut prefix_iter = pres.into_iter();
        let rootreg = if let Some(root) = prefix_iter.next() {
            let rootdef = root.def_begin();
            let r = self.try_load_variable(root.into_inner(), None, rootdef);
            for idx in pre_name_kregs.into_iter().skip(1) {
                self.emit(Isc::iabc(GETFIELD, r, r, idx), defln);
            }
            Some(r)
        } else {
            None
        };

        let fnreg = self.alloc_free_reg();
        let pirdx = self.subproto.len();
        let pro = self.walk_fn_body(fnname.into(), body.params, *body.body, false, false)?;
        self.subproto.push(pro);
        self.emit(Isc::iabx(CLOSURE, fnreg, pirdx as i32), defln);

        // method call, set field
        if let Some(metd_name) = method {
            debug_assert!(rootreg.is_some());
            let metdkidx = self.find_or_create_kstr(&metd_name);
            self.emit(
                Isc::iabck(SETFIELD, rootreg.unwrap(), metdkidx, metdkidx),
                defln,
            );
        }
        Ok(())
    }

    fn walk_local_decl(
        &mut self,
        mut names: Vec<(SrcLoc<String>, Option<Attribute>)>,
        mut exprs: Vec<ExprNode>,
        lineinfo: (u32, u32),
    ) -> Result<(), CodeGenErr> {
        debug_assert!(!names.is_empty());
        debug_assert!(!exprs.is_empty());

        let (nvar, nexp) = (names.len(), exprs.len());
        if nvar <= exprs.len() {
            // TODO:
            // add attribute support
            for ((def, _attr), exp) in names.into_iter().zip(exprs.iter_mut()) {
                let status = self.walk_common_expr(std::mem::take(exp), Ctx::Allocate)?;
                self.emit_local_decl(def.into_inner(), status, lineinfo);
            }
            for extra in exprs.into_iter().skip(nvar) {
                self.walk_common_expr(*extra, Ctx::Ignore)?;
            }
        } else {
            // SAFETY:
            // there are at least 1 expr
            let last = unsafe { exprs.pop().unwrap_unchecked() };

            for (idx, e) in exprs.into_iter().enumerate() {
                let status = self.walk_common_expr(*e, Ctx::Allocate)?;
                // TODO:
                // add attribute support
                let desc = unsafe { &mut names.get_mut(idx).unwrap_unchecked() };
                let name = std::mem::take(&mut desc.0);
                self.emit_local_decl(name.into_inner(), status, lineinfo);
            }

            let remain = names.iter().skip(nexp).count();
            debug_assert!(remain > 0);
            let last_sta = self.walk_common_expr(*last, Ctx::Allocate)?;
            if let ExprStatus::Call(reg) = last_sta {
                for (idx, (name, _attr)) in names.into_iter().skip(nexp).enumerate() {
                    self.emit_local_decl(
                        name.into_inner(),
                        ExprStatus::Reg(reg + idx as RegIndex),
                        lineinfo,
                    );
                }
            } else {
                let mut iter = names.into_iter().skip(nexp);
                // SAFETY: there are must at least 1 remain variable
                // let next = unsafe { iter.next().unwrap_unchecked() };
                let next = iter.next().unwrap();
                self.emit_local_decl(next.0.into_inner(), last_sta, lineinfo);

                for (name, _attr) in iter {
                    self.emit_local_decl(name.into_inner(), ExprStatus::LitNil, lineinfo);
                }
            }
        }
        Ok(())
    }

    fn walk_return(
        &mut self,
        ret: Option<Vec<ExprNode>>,
        line_of_on_empty_ret: u32,
        must_return: bool,
    ) -> Result<(), CodeGenErr> {
        if let Some(mut rets) = ret {
            match rets.len() {
                1 => {
                    // SAFETY: rets.len() == 1
                    let ret_node = unsafe { rets.pop().unwrap_unchecked() };
                    let ln = ret_node.lineinfo.0;

                    let status = self.walk_common_expr(*ret_node, Ctx::PotentialTailCall)?;

                    let mut gen_lit_template = |op| {
                        let reg = self.alloc_free_reg();
                        self.emit(Isc::iabc(op, reg, 0, 0), ln);
                        self.emit(Isc::iabc(RETURN1, reg, 0, 0), 0);
                    };

                    match status {
                        ExprStatus::LitNil => gen_lit_template(LOADNIL),
                        ExprStatus::LitTrue => gen_lit_template(LOADTRUE),
                        ExprStatus::LitFalse => gen_lit_template(LOADFALSE),
                        ExprStatus::LitInt(i) => {
                            let reg = self.alloc_free_reg();
                            self.emit(Isc::iasbx(LOADI, reg, i as i32), ln);
                            self.emit(Isc::iabc(RETURN1, reg, 0, 0), ln);
                        }
                        ExprStatus::LitFlt(f) => {
                            let reg = self.alloc_free_reg();
                            self.emit(Isc::iabx(LOADF, reg, f as i32), ln);
                            self.emit(Isc::iabc(RETURN1, reg, 0, 0), ln);
                        }
                        ExprStatus::Call(ret) => self.emit(Isc::iabc(RETURN, ret, 0, 0), ln),
                        ExprStatus::Kst(k) => {
                            let reg = self.load_const(k, ln);
                            self.emit(Isc::iabc(RETURN1, reg, 0, 0), ln);
                        }
                        ExprStatus::Reg(r) => {
                            self.emit(Isc::iabc(RETURN1, r, 0, 0), ln);
                        }
                    }
                }
                n if n > 1 => {
                    // todo: multi return
                    todo!("multi return")
                }
                _ => unreachable!(),
            }
        } else if must_return {
            self.emit(Isc::iabc(RETURN0, 0, 0, 0), line_of_on_empty_ret);
        }
        Ok(())
    }

    fn walk_fn_call(
        &mut self,
        call: FuncCall,
        fnreg: RegIndex,
        exp_ret: u32,
        tail_call: bool,
    ) -> Result<ExprStatus, CodeGenErr> {
        match call {
            FuncCall::MethodCall {
                prefix: _,
                method: _,
                args: _,
            } => {
                todo!("method call")
            }
            FuncCall::FreeFnCall { prefix, args } => {
                let nparam = args.namelist.len();
                let callln = prefix.def_begin();

                let fnreg_real = match self.walk_common_expr(*prefix, Ctx::must_use(fnreg))? {
                    ExprStatus::Reg(reg) => reg,
                    _ => unreachable!(),
                };

                // if fnreg_real != fnreg {
                //     self.emit(Isc::iabc(MOVE, fnreg, fnreg_real, 0), callln);
                // }
                debug_assert_eq!(fnreg, fnreg_real);

                for (n, param) in args.into_inner().namelist.into_iter().enumerate() {
                    let _ = self.walk_common_expr(*param, Ctx::must_use(1 + fnreg + n as i32));
                }

                let callisc = if tail_call { TAILCALL } else { CALL };
                self.emit(
                    Isc::iabc(callisc, fnreg, (nparam + 1) as i32, exp_ret as i32 + 1),
                    callln,
                );

                // reserve enuogh space for return value
                if exp_ret as usize > nparam {
                    for _ in nparam..exp_ret as usize {
                        self.alloc_free_reg();
                    }
                }

                Ok(ExprStatus::Reg(fnreg))
            }
        }
    }

    fn walk_assign_stmt(
        &mut self,
        vars: Vec<ExprNode>,
        mut exprs: Vec<ExprNode>,
    ) -> Result<(), CodeGenErr> {
        debug_assert!(!vars.is_empty());
        debug_assert!(!exprs.is_empty());

        let (nvar, nexp) = (vars.len(), exprs.len());
        if nvar <= nexp {
            for (var, exp) in vars.into_iter().zip(exprs.iter_mut()) {
                let vardef = var.def_begin();

                let var_s = self.walk_common_expr(*var, Ctx::Allocate)?;

                if let ExprStatus::Reg(vreg) = var_s {
                    let value_s =
                        self.walk_common_expr(*std::mem::take(exp), Ctx::must_use(vreg))?;
                    match value_s {
                        ExprStatus::LitNil => self.emit(Isc::iabc(LOADNIL, vreg, 0, 0), vardef),
                        ExprStatus::LitTrue => self.emit(Isc::iabc(LOADTRUE, vreg, 0, 0), vardef),
                        ExprStatus::LitFalse => self.emit(Isc::iabc(LOADFALSE, vreg, 0, 0), vardef),
                        ExprStatus::LitInt(i) => {
                            self.emit(Isc::iasbx(LOADI, vreg, i as i32), vardef)
                        }
                        ExprStatus::LitFlt(f) => {
                            // FIX ME:
                            // use LOADF for small float.

                            let kreg = self.alloc_const_reg(LValue::Float(f));
                            self.emit(Isc::iabx(LOADK, vreg, kreg), vardef);
                        }
                        ExprStatus::Kst(kreg) => self.emit(Isc::iabx(LOADK, vreg, kreg), vardef),
                        ExprStatus::Call(creg) | ExprStatus::Reg(creg) => {
                            debug_assert_eq!(creg, vreg);
                            // self.emit(Isc::iabc(MOVE, vreg, creg, 0), vardef)
                        }
                    }
                } else {
                    unreachable!()
                }
            }

            for extra in exprs.into_iter().skip(nvar) {
                let _ = self.walk_common_expr(*extra, Ctx::Ignore)?;
            }
        } else {
            todo!()
        }
        Ok(())
    }

    fn walk_common_expr(
        &mut self,
        node: SrcLoc<Expr>,
        ctx: ExprGenCtx,
    ) -> Result<ExprStatus, CodeGenErr> {
        let (def, node) = (node.lineinfo, node.into_inner());

        let status = match ctx {
            // case that the value of expression will be ignored,
            // but sub expr may make an side effect
            ExprGenCtx::Ignore => {
                let pre_state = self.next_free_reg();
                let status = match node {
                    Expr::Index { prefix, key } => {
                        let _ = self.walk_common_expr(*prefix, Ctx::Ignore);
                        let _ = self.walk_common_expr(*key, Ctx::Ignore);
                        ExprStatus::Reg(RegIndex::MAX)
                    }
                    Expr::FuncCall(call) => {
                        let next = self.alloc_free_reg();
                        let _ = self.walk_fn_call(call, next, 0, false);
                        self.free_reg();
                        ExprStatus::Reg(RegIndex::MAX)
                    }
                    Expr::TableCtor(ctor) => {
                        for field in ctor {
                            let _ = self.walk_common_expr(**field.val, Ctx::Ignore);
                        }
                        ExprStatus::Reg(RegIndex::MAX)
                    }
                    Expr::BinaryOp {
                        lhs: l,
                        op: _,
                        rhs: r,
                    } => {
                        let _ = self.walk_common_expr(*l, Ctx::Ignore);
                        let _ = self.walk_common_expr(*r, Ctx::Ignore);
                        ExprStatus::Reg(RegIndex::MAX)
                    }
                    Expr::UnaryOp { op: _, expr } => {
                        let _ = self.walk_common_expr(*expr, Ctx::Ignore);
                        ExprStatus::Reg(RegIndex::MAX)
                    }
                    _ => ExprStatus::Reg(RegIndex::MAX),
                };

                // recover register state
                let mut try_recover = self.next_free_reg();
                debug_assert!(try_recover >= pre_state);
                while try_recover != pre_state {
                    try_recover = self.free_reg();
                }
                status
            }

            ExprGenCtx::Allocate => {
                let free = self.alloc_free_reg();
                let status = self.emit_expr(node, free, def)?;
                if let ExprStatus::Reg(ref r) = status {
                    if *r < free {
                        self.free_reg();
                        debug_assert_eq!(self.next_free_reg(), free);
                    }
                }
                status
            }

            ExprGenCtx::NonRealloc { dest } => {
                let mut status = self.emit_expr(node, dest, def)?;
                if let ExprStatus::Reg(real) = status {
                    if real != dest {
                        self.emit(Isc::iabc(MOVE, dest, real, 0), def.0);
                        status = ExprStatus::Reg(dest)
                    }
                }
                status
            }

            ExprGenCtx::PotentialTailCall => {
                let reg = self.alloc_free_reg();
                match node {
                    Expr::FuncCall(call) => self.walk_fn_call(call, reg, 1, true)?,
                    _ => self.emit_expr(node, reg, def)?,
                }
            }

            ExprGenCtx::MultiLevelTableIndex { depth: _ } => {
                todo!("expr codegen: multi level Table Index optimize")
            }
        };

        Ok(status)
    }

    fn emit_expr(
        &mut self,
        exp: Expr,
        dest: RegIndex,
        def: (u32, u32),
    ) -> Result<ExprStatus, CodeGenErr> {
        let status = match exp {
            Expr::Nil => {
                self.emit(Isc::iabc(LOADNIL, dest, 0, 0), def.0);
                ExprStatus::Reg(dest)
            }
            Expr::False => {
                self.emit(Isc::iabc(LOADFALSE, dest, 0, 0), def.0);
                ExprStatus::Reg(dest)
            }
            Expr::True => {
                self.emit(Isc::iabc(LOADTRUE, dest, 0, 0), def.0);
                ExprStatus::Reg(dest)
            }
            Expr::Int(i) => {
                if i as i32 > Isc::MAX_SBX {
                    let kreg = self.alloc_const_reg(i.into());
                    self.emit(Isc::iabx(LOADK, dest, kreg), def.0);
                } else {
                    self.emit(Isc::iasbx(LOADI, dest, i as i32), def.0);
                }
                ExprStatus::Reg(dest)
            }

            Expr::Float(f) => {
                if f as i32 > Isc::MAX_SBX {
                    let kreg = self.alloc_const_reg(f.into());
                    self.emit(Isc::iabx(LOADK, dest, kreg), def.0);
                } else {
                    todo!("load small float to register")
                }
                ExprStatus::Reg(dest)
            }

            Expr::Literal(s) => {
                let kreg = self.find_or_create_kstr(&s);
                self.emit(Isc::iabx(LOADK, dest, kreg), def.0);
                ExprStatus::Reg(dest)
            }

            Expr::Ident(id) => {
                let reg = self.try_load_variable(id, Some(dest), def.0);
                ExprStatus::Reg(reg)
            }

            Expr::UnaryOp { op, expr } => self.emit_unary_expr(*expr, op, dest)?,

            Expr::BinaryOp { lhs, op, rhs } => self.emit_binary_expr(*lhs, *rhs, def, op, dest)?,

            Expr::FuncDefine(fnbody) => {
                let pto = self.walk_fn_body(
                    CodeGen::ANONYMOUS.into(),
                    fnbody.params,
                    *fnbody.body,
                    false,
                    false,
                )?;
                self.subproto.push(pto);
                let pidx = self.subproto.len();

                self.emit(Isc::iabx(CLOSURE, dest, pidx as i32), def.0);
                ExprStatus::Reg(dest)
            }

            Expr::Index { prefix, key } => {
                let key_status = self.walk_common_expr(*key, Ctx::Allocate)?;
                match self.walk_common_expr(*prefix, Ctx::MultiLevelTableIndex { depth: 1 })? {
                    ExprStatus::Reg(pre) | ExprStatus::Call(pre) => {
                        self.emit_index_local(key_status, def.0, pre, dest)
                    }
                    _ => unreachable!(),
                }
            }

            Expr::TableCtor(fields) => self.walk_table_ctor(fields, dest, def)?,
            Expr::FuncCall(call) => {
                let free = self.alloc_free_reg();
                self.walk_fn_call(call, free, 1, false)?
            }
            Expr::Dots => unreachable!(),
        };
        Ok(status)
    }

    fn emit_binary_expr(
        &mut self,
        lhs: SrcLoc<Expr>,
        rhs: SrcLoc<Expr>,
        def: (u32, u32),
        op: BinOp,
        destreg: RegIndex,
    ) -> Result<ExprStatus, CodeGenErr> {
        let (lst, rst) = {
            let l = if let Some(ls) = GenState::peek_const_expr(&lhs) {
                ls
            } else {
                self.walk_common_expr(lhs, Ctx::must_use(destreg))?
            };

            let right_reg = self.alloc_free_reg();
            let r = if let Some(rs) = GenState::peek_const_expr(&rhs) {
                rs
            } else {
                self.walk_common_expr(rhs, Ctx::must_use(right_reg))?
            };
            self.free_reg(); // free right reg
            (l, r)
        };

        let select_arithmetic_kop = |bop: BinOp| -> OpCode {
            match bop {
                BinOp::Add => ADDK,
                BinOp::Minus => SUBK,
                BinOp::Mul => MULK,
                BinOp::Mod => MODK,
                BinOp::Pow => POWK,
                BinOp::Div => DIVK,
                BinOp::IDiv => IDIVK,
                BinOp::BitAnd => BANDK,
                BinOp::BitOr => BORK,
                BinOp::BitXor => BXORK,
                _ => unreachable!(),
            }
        };
        let select_arithemic_op = |bop: BinOp| -> OpCode {
            match bop {
                BinOp::Add => ADD,
                BinOp::Minus => SUB,
                BinOp::Mul => MUL,
                BinOp::Mod => MOD,
                BinOp::Pow => POW,
                BinOp::Div => DIV,
                BinOp::IDiv => IDIV,
                BinOp::BitAnd => BAND,
                BinOp::BitOr => BOR,
                BinOp::BitXor => BXOR,
                _ => unreachable!(),
            }
        };

        let select_immediate_op = |bop: BinOp| -> OpCode {
            match bop {
                BinOp::Add => ADDI,
                BinOp::Shl => SHLI,
                BinOp::Shr => SHRI,
                _ => unreachable!(),
            }
        };

        const IMMEDIATE_OP: [BinOp; 3] = [BinOp::Add, BinOp::Shl, BinOp::Shr];

        match (lst, rst) {
            // both of [l, r] is kst, fallthrough
            (ExprStatus::Kst(lk), ExprStatus::Kst(rk)) => {
                // load left to dest reg and cover dest reg
                self.emit(Isc::iabx(LOADK, destreg, lk), def.0);
                self.emit(
                    Isc::iabc(select_arithmetic_kop(op), destreg, destreg, rk),
                    def.0,
                );
            }

            // one of [l, r] is kst
            (ExprStatus::Kst(lk), ExprStatus::Reg(rk))
            | (ExprStatus::Reg(lk), ExprStatus::Kst(rk)) => {
                self.emit(Isc::iabc(select_arithmetic_kop(op), destreg, lk, rk), def.0);
            }

            // one of [l, r] is imidiate oprand
            (ExprStatus::LitInt(i), other) | (other, ExprStatus::LitInt(i))
                if IMMEDIATE_OP.contains(&op) =>
            {
                let reg = self.try_load_expr_to_local(other, def.0);
                self.emit(
                    Isc::iabc(select_immediate_op(op), destreg, reg, i as i32),
                    def.0,
                );
            }

            // both of [l, r] is active variable
            (ExprStatus::Reg(lk), ExprStatus::Reg(rk)) => {
                self.emit(Isc::iabc(select_arithemic_op(op), destreg, lk, rk), def.0);
            }

            _ => unreachable!(),
        };
        Ok(ExprStatus::Reg(destreg))
    }

    fn emit_unary_expr(
        &mut self,
        expr: SrcLoc<Expr>,
        op: UnOp,
        freg: RegIndex,
    ) -> Result<ExprStatus, CodeGenErr> {
        let ln = expr.def_begin();
        let oprand = self.walk_common_expr(expr, ExprGenCtx::Allocate)?;

        let unop_code = match op {
            UnOp::Minus => UNM,
            UnOp::Not => NOT,
            UnOp::Length => LEN,
            UnOp::BitNot => BNOT,
        };

        match oprand {
            ExprStatus::Kst(kreg) => {
                self.emit(Isc::iabx(LOADK, freg, kreg), ln);
                self.emit(Isc::iabc(unop_code, freg, freg, 0), ln);
                Ok(ExprStatus::Reg(freg))
            }

            ExprStatus::Reg(reg) => {
                self.emit(Isc::iabc(unop_code, freg, reg, 0), ln);
                Ok(ExprStatus::Reg(freg))
            }

            ExprStatus::Call(reg) => {
                self.emit(Isc::iabc(unop_code, reg, reg, 0), ln);
                Ok(ExprStatus::Reg(reg))
            }

            _ => unreachable!(),
        }
    }

    fn walk_table_ctor(
        &mut self,
        flist: Vec<Field>,
        dest: RegIndex,
        def: (u32, u32),
    ) -> Result<ExprStatus, CodeGenErr> {
        self.emit(Isc::iabc(NEWTABLE, dest, 0, 0), def.0);
        self.emit(Isc::iax(EXTRAARG, 0), def.0);

        let mut aryidx = 1;
        for field in flist.into_iter() {
            let fdefloc = field.val.def_begin();
            let valstatus = self.walk_common_expr(**field.val, Ctx::Allocate)?;

            if let Some(key) = field.key {
                let keystatus = self.walk_common_expr(**key, Ctx::Allocate)?;
                match keystatus {
                    ExprStatus::Kst(kidx) => {
                        if let ExprStatus::Kst(valreg) = valstatus {
                            self.emit(Isc::iabck(SETFIELD, dest, kidx, valreg), fdefloc)
                        } else {
                            let valreg = self.try_load_expr_to_local(valstatus, fdefloc);
                            self.emit(Isc::iabck(SETFIELD, dest, kidx, valreg), fdefloc)
                        }
                    }

                    ExprStatus::Reg(reg) => {
                        if let ExprStatus::Kst(valreg) = valstatus {
                            self.emit(Isc::iabck(SETTABLE, dest, reg, valreg), fdefloc)
                        } else {
                            let valreg = self.try_load_expr_to_local(valstatus, fdefloc);
                            self.emit(Isc::iabc(SETTABLE, dest, reg, valreg), fdefloc)
                        }
                    }

                    ExprStatus::LitInt(i) => {
                        if let ExprStatus::Kst(valreg) = valstatus {
                            self.emit(Isc::iabck(SETI, dest, i as i32, valreg), fdefloc);
                        } else {
                            let valreg = self.try_load_expr_to_local(valstatus, fdefloc);
                            self.emit(Isc::iabc(SETI, dest, i as i32, valreg), fdefloc);
                        }
                    }

                    _ => todo!(),
                }
                continue;
            }

            // array field
            if let ExprStatus::Kst(valreg) = valstatus {
                self.emit(Isc::iabck(SETI, dest, aryidx, valreg), fdefloc);
            } else {
                let valreg = self.try_load_expr_to_local(valstatus, fdefloc);
                self.emit(Isc::iabc(SETI, dest, aryidx, valreg), fdefloc);
            }
            aryidx += 1;
        }
        Ok(ExprStatus::Reg(dest))
    }

    /// Load an variable to local register by name.
    fn try_load_variable(&mut self, id: String, dest: Option<RegIndex>, ln: u32) -> RegIndex {
        enum SearchState {
            Local { reg: RegIndex },
            UpList { idx: RegIndex },
        }
        use SearchState::*;

        let find_in_frame = |f: &GenState| -> Option<SearchState> {
            if let Some(reg) = f.find_local_decl(&id) {
                return Some(SearchState::Local { reg });
            }
            for (idx, up) in f.upvals.iter().enumerate() {
                if up.name() == id {
                    return Some(SearchState::UpList {
                        idx: idx as RegIndex,
                    });
                }
            }
            None
        };

        let reg = dest.unwrap_or_else(|| self.alloc_free_reg());
        // try find in current frame
        if let Some(s) = find_in_frame(&self.cgs) {
            match s {
                Local { reg } => reg,
                UpList { idx } => {
                    self.emit(Isc::iabc(GETUPVAL, reg, idx, 0), ln);
                    reg
                }
            }
        } else {
            // find in outter function upvalue list
            if let Some((nrev, peek)) = self.genstk.iter().rev().enumerate().fold(
                None,
                |state, (idx, frame)| -> Option<(usize, SearchState)> {
                    if state.is_none() {
                        find_in_frame(frame).map(|s| (idx, s))
                    } else {
                        state
                    }
                },
            ) {
                // add upvalue decl for all intermidiate function during search
                let fin = self.genstk.iter_mut().rev().take(nrev).rev().fold(
                    peek,
                    |prev_frame_state, gs| {
                        // set _ENV
                        let pos = if gs.upvals.is_empty() {
                            gs.upvals.push(UpvalDecl::Env);
                            1
                        } else {
                            gs.upvals.len()
                        };
                        let updecl = match prev_frame_state {
                            Local { reg } => UpvalDecl::OnStack {
                                name: id.clone(),
                                register: reg,
                            },
                            UpList { idx } => UpvalDecl::InUpList {
                                name: id.clone(),
                                offset: idx,
                            },
                        };
                        gs.upvals.push(updecl);

                        SearchState::UpList {
                            idx: pos as RegIndex,
                        }
                    },
                );

                let upidx = match fin {
                    UpList { idx } => idx,
                    Local { reg: _ } => unreachable!(),
                };

                self.emit(Isc::iabc(GETUPVAL, reg, upidx as RegIndex, 0), ln);
            } else {
                // not found in all protos, get  from _ENV
                let name_kreg = self.find_or_create_kstr(&id);
                self.emit(Isc::iabc(GETTABUP, reg, 0, name_kreg), ln);
            }
            reg
        }
    }
}

impl CodeGen {
    pub fn generate(ast_root: SrcLoc<Block>, strip: bool) -> Result<Proto, CodeGenErr> {
        let mut gen = Self::new(strip);
        gen.consume(ast_root)
    }
}

#[derive(Debug)]
pub enum BinLoadErr {
    IOErr(std::io::Error),
    NotBinaryChunk,
    VersionMismatch,
    UnsupportedFormat,
    IncompatiablePlatform,
}

impl From<std::io::Error> for BinLoadErr {
    fn from(value: std::io::Error) -> Self {
        BinLoadErr::IOErr(value)
    }
}

pub struct ChunkDumper();

impl ChunkDumper {
    const LUA_SIGNATURE: &'static str = "\x1bLua";
    const LUA_VERSION: u8 = 0x54;

    const LUAC_MAGIC: [u8; 6] = [0x19, 0x93, 0x0d, 0x0a, 0x1a, 0x0a];

    // standard luac format
    const LUAC_FORMAT: u8 = 0;

    // from luac 5.4
    const LUAC_INT: u64 = 0x5678;

    // from luac 5.4
    const LUAC_FLOAT: f64 = 370.5;

    pub fn dump(chunk: &Proto, bw: &mut BufWriter<impl Write>) -> std::io::Result<()> {
        bw.write_all(Self::LUA_SIGNATURE.as_bytes())?;
        bw.write_all(&[Self::LUA_VERSION, Self::LUAC_FORMAT])?;
        bw.write_all(&Self::LUAC_MAGIC)?;
        bw.write_all(&[
            std::mem::size_of::<Instruction>() as u8,
            std::mem::size_of::<isize>() as u8,
            std::mem::size_of::<f64>() as u8,
        ])?;
        bw.write_all(&Self::LUAC_INT.to_ne_bytes())?;
        bw.write_all(&Self::LUAC_FLOAT.to_ne_bytes())?;

        Self::dump_proto(chunk, bw)
    }

    pub fn undump(r: &mut BufReader<impl Read>) -> Result<Proto, BinLoadErr> {
        let mut signature = [0_u8; 4];
        r.read_exact(&mut signature)?;
        for (i, s) in Self::LUA_SIGNATURE.bytes().enumerate() {
            if signature[i] != s {
                return Err(BinLoadErr::NotBinaryChunk);
            }
        }

        let mut version = [0_u8; 1];
        r.read_exact(&mut version)?;
        if u8::from_ne_bytes(version) != Self::LUA_VERSION {
            return Err(BinLoadErr::VersionMismatch);
        }

        let mut format = [0_u8; 1];
        r.read_exact(&mut format)?;
        if u8::from_ne_bytes(format) != Self::LUAC_FORMAT {
            return Err(BinLoadErr::UnsupportedFormat);
        }

        let mut magic = [0_u8; 6];
        r.read_exact(&mut magic)?;
        for (i, m) in Self::LUAC_MAGIC.iter().enumerate() {
            if magic[i] != *m {
                return Err(BinLoadErr::NotBinaryChunk);
            }
        }

        let mut code_size = [0_u8; 1];
        let mut int_size = [0_u8; 1];
        let mut flt_size = [0_u8; 1];
        let mut luac_int = [0_u8; 8];
        let mut luac_flt = [0_u8; 8];

        r.read_exact(&mut code_size)?;
        r.read_exact(&mut int_size)?;
        r.read_exact(&mut flt_size)?;
        r.read_exact(&mut luac_int)?;
        r.read_exact(&mut luac_flt)?;

        let csize_check = u8::from_ne_bytes(code_size) != std::mem::size_of::<Instruction>() as u8;
        let size_check = u8::from_ne_bytes(int_size) != std::mem::size_of::<isize>() as u8;
        let flt_check = u8::from_ne_bytes(flt_size) != std::mem::size_of::<f64>() as u8;
        let luacint_check = u64::from_ne_bytes(luac_int) != Self::LUAC_INT;
        let luacflt_check = f64::from_ne_bytes(luac_flt) != Self::LUAC_FLOAT;
        if csize_check || size_check || flt_check || luacint_check || luacflt_check {
            return Err(BinLoadErr::IncompatiablePlatform);
        }

        Self::undump_proto(r)
    }

    fn dump_proto(chunk: &Proto, bw: &mut BufWriter<impl Write>) -> std::io::Result<()> {
        Self::dump_varint(chunk.updecl.len(), bw)?;
        Self::dump_string(unsafe { chunk.source.as_str_unchecked() }, bw)?;
        Self::dump_varint(chunk.begline as usize, bw)?;
        Self::dump_varint(chunk.endline as usize, bw)?;
        bw.write_all(&[chunk.nparam, chunk.vararg as u8, chunk.nreg])?;

        Self::dump_varint(chunk.code.len(), bw)?;
        for isc in chunk.code.iter() {
            bw.write_all(&isc.code.to_ne_bytes())?;
        }

        Self::dump_varint(chunk.kst.len(), bw)?;
        for k in chunk.kst.iter() {
            Self::dump_const(k, bw)?;
        }

        Self::dump_varint(chunk.updecl.len(), bw)?;
        for up in chunk.updecl.iter() {
            let (onstk, stkid) = match up {
                UpvalDecl::OnStack { name: _, register } => (true, *register),

                UpvalDecl::InUpList { name: _, offset } => (false, *offset),
                UpvalDecl::Env => (false, 0),
            };
            // let attr: u8 = if let Some(a) = up.kind { a as u8 } else { 0 };
            Self::dump_varint(onstk as usize, bw)?;
            Self::dump_varint(stkid as usize, bw)?;
        }

        Self::dump_varint(chunk.subfn.len(), bw)?;
        for p in chunk.subfn.iter() {
            Self::dump_proto(p, bw)?;
        }

        // TODO
        // support dump line infomation
        Self::dump_varint(0, bw)?; // size_line_info
        Self::dump_varint(0, bw)?; // size_abs_line_info

        Self::dump_varint(chunk.locvars.len(), bw)?;
        for loc in chunk.locvars.iter() {
            Self::dump_string(&loc.name, bw)?;
            Self::dump_varint(loc.start_pc as usize, bw)?;
            Self::dump_varint(loc.end_pc as usize, bw)?;
        }

        Self::dump_varint(chunk.updecl.len(), bw)?;
        for up in chunk.updecl.iter() {
            Self::dump_string(up.name(), bw)?;
        }

        Ok(())
    }

    pub fn undump_proto(r: &mut BufReader<impl Read>) -> Result<Proto, BinLoadErr> {
        let nupval = Self::undump_varint(r)?;
        let src = LValue::from(Self::undump_string(r)?);
        let begline = Self::undump_varint(r)?;
        let endline = Self::undump_varint(r)?;

        let mut byte = [0; 1];
        r.read_exact(&mut byte)?;
        let nparam = u8::from_ne_bytes(byte);
        r.read_exact(&mut byte)?;
        let is_vararg = u8::from_ne_bytes(byte) != 0;
        r.read_exact(&mut byte)?;
        let nreg = u8::from_ne_bytes(byte);

        let code_size = Self::undump_varint(r)?;
        let mut code = Vec::with_capacity(code_size);
        let mut isc_buf = 0_u32.to_ne_bytes();
        for _ in 0..code_size {
            r.read_exact(&mut isc_buf)?;
            code.push(Instruction {
                code: u32::from_ne_bytes(isc_buf),
            })
        }

        let kst_size = Self::undump_varint(r)?;
        let mut kst = Vec::with_capacity(kst_size);
        for _ in 0..kst_size {
            kst.push(Self::undump_const(r)?)
        }

        let up_size = Self::undump_varint(r)?;
        let mut ups = Vec::with_capacity(up_size);
        for _ in 0..up_size {
            r.read_exact(&mut byte)?;
            let _onstk = u8::from_ne_bytes(byte) != 0;

            r.read_exact(&mut byte)?;
            let _stkid = u8::from_ne_bytes(byte);

            r.read_exact(&mut byte)?;
            // TODO:
            // support attribute dump
            // let attr = ...
            // let updecl = match (onstk, stkid) {
            //     (false, 0) => UpvalDecl::Env,
            //     (false, _) => UpvalDecl::InEnv {
            //         name: LValue::Nil,
            //         self_upidx: stkid as usize,
            //     },
            //     (true, _) => UpvalDecl::OnStack {
            //         name: LValue::Nil,
            //         parent_upidx: stkid as usize,
            //     },
            // };
            // ups.push(updecl);
        }

        let proto_size = Self::undump_varint(r)?;
        let mut subfn = Vec::with_capacity(proto_size);
        for _ in 0..proto_size {
            let mut p = Self::undump_proto(r)?;
            p.source = src;
            subfn.push(Gc::new(p));
        }

        // TODO:
        // support line info undump
        let line_info_size = Self::undump_varint(r)?;
        for _ in 0..line_info_size {
            r.read_exact(&mut byte)?;
        }

        let abs_line_info_size = Self::undump_varint(r)?;
        for _ in 0..abs_line_info_size {
            let _pc = Self::undump_varint(r)?;
            let _line = Self::undump_varint(r)?;
        }

        let loc_num = Self::undump_varint(r)?;
        for _ in 0..loc_num {
            let _name = Self::undump_string(r)?;
            let _start_pc = Self::undump_varint(r)?;
            let _end_pc = Self::undump_varint(r)?;
        }

        let upval_num = Self::undump_varint(r)?;
        debug_assert_eq!(upval_num, nupval);
        for up in ups.iter_mut().take(upval_num) {
            let upname = Self::undump_string(r)?;
            match up {
                UpvalDecl::OnStack { name, register: _ } => *name = upname,
                UpvalDecl::InUpList { name, offset: _ } => *name = upname,
                UpvalDecl::Env => {}
            }
        }

        Ok(Proto {
            vararg: is_vararg,
            nparam,
            nreg,
            begline: begline as u32,
            endline: endline as u32,
            kst,
            code,
            subfn,
            pcline: Vec::new(),
            source: src,
            locvars: Vec::new(),
            updecl: ups,
        })
    }

    const DUMP_INT_BUFFER_SIZE: usize = std::mem::size_of::<usize>() * 8 / 7 + 1;
    fn dump_varint(mut x: usize, bw: &mut BufWriter<impl Write>) -> std::io::Result<()> {
        let mut buff = [0_u8; Self::DUMP_INT_BUFFER_SIZE];
        let mut n = 0;
        loop {
            buff[n] = x as u8 & 0x7f;
            x >>= 7;
            if x == 0 {
                buff[n] |= 0x80;
                break;
            }
            n += 1;
        }
        bw.write_all(&buff[..n + 1])?;
        Ok(())
    }

    fn dump_const(val: &LValue, w: &mut BufWriter<impl Write>) -> std::io::Result<()> {
        match val {
            LValue::Nil => {
                w.write_all(&[0x00])?;
            }
            LValue::Bool(b) => {
                if *b {
                    w.write_all(&[0x11])?;
                } else {
                    w.write_all(&[0x1])?;
                }
            }
            LValue::Int(i) => {
                w.write_all(&[0x03])?;
                unsafe { Self::dump_varint(std::mem::transmute(i), w)? }
            }
            LValue::Float(f) => {
                w.write_all(&[0x13])?;
                Self::dump_float(*f, w)?;
            }
            LValue::String(s) => {
                if s.is_short() {
                    w.write_all(&[0x04])?;
                } else {
                    w.write_all(&[0x14])?;
                }
                Self::dump_string(s.as_str(), w)?;
            }
            _ => unreachable!(),
        };
        Ok(())
    }

    fn undump_const(r: &mut BufReader<impl Read>) -> std::io::Result<LValue> {
        let mut byte = [0; 1];
        r.read_exact(&mut byte)?;
        let val = match u8::from_ne_bytes(byte) {
            0x00 => LValue::Nil,
            0x01 => LValue::Bool(false),
            0x11 => LValue::Bool(true),
            0x03 => LValue::Int(unsafe { std::mem::transmute(Self::undump_varint(r)?) }),
            0x13 => LValue::Float(Self::undump_float(r)?),
            0x04 | 0x14 => LValue::from(Self::undump_string(r)?),
            _ => unreachable!(),
        };
        Ok(val)
    }

    fn dump_string(s: &str, w: &mut BufWriter<impl Write>) -> std::io::Result<()> {
        let len = s.len();
        if len == 0 {
            Self::dump_varint(0, w)?;
        } else {
            Self::dump_varint(len + 1, w)?;
            w.write_all(s.as_bytes())?;
        }
        Ok(())
    }

    fn undump_string(r: &mut BufReader<impl Read>) -> std::io::Result<String> {
        let len = Self::undump_varint(r)?;
        if len == 0 {
            Ok(String::new())
        } else {
            let mut buf = String::with_capacity(len);
            let reallen = len - 1;
            for _ in 0..reallen {
                buf.push('\0');
            }
            r.read_exact(&mut unsafe { buf.as_bytes_mut() }[0..reallen])?;
            Ok(buf)
        }
    }

    fn undump_varint(buf: &mut BufReader<impl Read>) -> std::io::Result<usize> {
        let mut x: usize = 0;
        let mut n = 0;
        loop {
            let mut byte = [0_u8; 1];
            buf.read_exact(&mut byte)?;
            let byte = u8::from_ne_bytes(byte);
            let pad = (byte & 0x7f) as usize;
            x |= pad << (7 * n);
            if (byte & 0x80) != 0 {
                break;
            }
            n += 1;
        }
        Ok(x)
    }

    fn dump_float(f: f64, bw: &mut BufWriter<impl Write>) -> std::io::Result<()> {
        Self::dump_varint(f.to_bits() as usize, bw)
    }

    fn undump_float(buf: &mut BufReader<impl Read>) -> std::io::Result<f64> {
        Ok(f64::from_bits(Self::undump_varint(buf)? as u64))
    }
}

mod test {
    use std::{
        fmt::Debug,
        fs::File,
        io::{BufReader, BufWriter, Write},
    };

    #[test]
    fn instruction_build() {
        use crate::codegen::{Isc, OpCode, OpMode};
        for signed in [0, 1, 123, 999, -1, -999] {
            let i = Isc::iasbx(OpCode::LOADI, 0, signed);
            assert_eq!(i.mode(), OpMode::IAsBx);
            let (_, a, sbx) = i.repr_asbx();
            assert_eq!(a, 0);
            assert_eq!(sbx, signed);
        }

        for step in [1, -1, 100, -100] {
            let i = Isc::isj(OpCode::JMP, step);
            let (_, sj) = i.repr_sj();
            debug_assert_eq!(sj, step);
        }
    }

    #[test]
    fn instruction_size_check() {
        use super::Instruction;
        assert_eq!(std::mem::size_of::<Instruction>(), 4);
    }

    #[allow(dead_code)]
    fn dump_and_undump<W, R, T>(filename: &str, to_test: &[T], wop: W, rop: R)
    where
        W: Fn(&T, &mut BufWriter<File>),
        R: Fn(&mut BufReader<File>) -> T,
        T: Eq + Debug,
    {
        let tmpfile = {
            let mut temp_dir = std::env::temp_dir();
            temp_dir.push(filename);
            temp_dir
        };

        for i in to_test.iter() {
            let tmp_file = std::fs::File::create(tmpfile.clone()).unwrap();
            let mut writer = BufWriter::new(tmp_file);
            wop(i, &mut writer);
            writer.flush().unwrap();

            let same_file = std::fs::File::open(tmpfile.clone()).unwrap();
            let mut reader = BufReader::new(same_file);
            let k = rop(&mut reader);

            debug_assert_eq!(*i, k);
        }
    }

    #[test]
    fn proto() {
        use super::ChunkDumper;
        use crate::parser::Parser;
        use crate::state::State;
        use std::io::{BufReader, BufWriter, Write};

        let tmpfile = {
            let mut temp_dir = std::env::temp_dir();
            temp_dir.push("ruac.test.binary.proto");
            temp_dir
        };

        // let src = r#"
        //     local function f(a, b)
        //         return a + b
        //     end
        // "#;

        let src = r#"
            print "Hello World"
        "#;

        let ast = Parser::parse(src, None).unwrap();
        let origin = super::CodeGen::generate(*ast, false).unwrap();
        // println!("{:?}", origin);

        let tmp_file = std::fs::File::create(tmpfile.clone()).unwrap();
        let mut writer = BufWriter::new(tmp_file);
        ChunkDumper::dump(&origin, &mut writer).unwrap();
        writer.flush().unwrap();

        let same_file = std::fs::File::open(tmpfile.clone()).unwrap();
        let mut reader = BufReader::new(same_file);
        let _recover = ChunkDumper::undump(&mut reader).unwrap();
        // println!("{:?}", recover);

        let _vm = State::new();
        // TODO:
        // load and execute origin and recover to check result
        // vm.load(proto);
    }

    #[test]
    fn string() {
        use super::ChunkDumper;
        let str_to_write = [
            "123",
            "",
            "中文",
            "ksta#$%^&*\x01\x46\r\n",
            "longggggggggggggggggggggggggggggggggggggggggggggggggggggggggggggg string",
        ]
        .iter()
        .map(|s| s.to_string())
        .collect::<Vec<_>>();

        dump_and_undump(
            "ruac.test.binary.string",
            &str_to_write,
            |k, w| ChunkDumper::dump_string(k, w).unwrap(),
            |r| ChunkDumper::undump_string(r).unwrap(),
        );
    }

    #[test]
    fn varint() {
        use super::ChunkDumper;
        let int_to_write = [999, 0, 1, 2, 3, 123, 999999, -1, -2, -999];
        dump_and_undump(
            "ruac.test.binary.varint",
            &int_to_write.map(|x| x as usize),
            |k, w| ChunkDumper::dump_varint(*k, w).unwrap(),
            |r| ChunkDumper::undump_varint(r).unwrap(),
        );
    }

    #[test]
    fn float() {
        use super::ChunkDumper;
        use std::io::{BufReader, BufWriter, Write};
        let tmpfile = {
            let mut temp_dir = std::env::temp_dir();
            temp_dir.push("ruac.test.binary.folat");
            temp_dir
        };

        let flt_to_write = [0.01, 789.0, 449.7, -1000000.555];
        for f in flt_to_write.into_iter() {
            let tmp_file = std::fs::File::create(tmpfile.clone()).unwrap();
            let mut writer = BufWriter::new(tmp_file);
            ChunkDumper::dump_float(f, &mut writer).unwrap();
            writer.flush().unwrap();

            let same_file = std::fs::File::open(tmpfile.clone()).unwrap();
            let mut reader = BufReader::new(same_file);
            let r = ChunkDumper::undump_float(&mut reader).unwrap();
            debug_assert_eq!(f, r);
        }
    }
}
