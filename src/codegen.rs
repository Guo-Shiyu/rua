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
        NumericFor, ParameterList, SrcLoc, Stmt, StmtNode, UnOp,
    },
    heap::{Gc, GcOp, Heap, MemStat, Tag, TypeTag},
    state::RegIndex,
    value::Value,
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

#[derive(Debug, Clone, Copy)]
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
                    f.write_str("k")?;
                } else {
                    f.write_str(" ")?;
                }
                Ok(())
            }
            OpMode::IABx => {
                let (code, a, bx) = self.repr_abx();
                write!(f, "{code:<16}\t{:<3} {:<3}     ", a, bx)
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
    pub fn name(&self) -> &str {
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

    kst: Box<[Value]>,            // constants
    pub code: Box<[Instruction]>, // bytecodes
    subfn: Box<[Gc<Proto>]>,      // sub function
    pcline: Box<[u32]>,           // line number of each instruction
    updecl: Box<[UpvalDecl]>,     // upvalue information

    pub source: Value,          // source file name, used for debug info
    locvars: Box<[LocVarDecl]>, // local variable name, used for debug info
}

impl MemStat for Proto {
    fn mem_ref(&self) -> usize {
        fn acc_mem<T>(ary: &[T]) -> usize {
            std::mem::size_of_val(ary)
        }

        std::mem::size_of::<Self>()
            + acc_mem(&self.code)
            + acc_mem(&self.pcline)
            + acc_mem(&self.kst)
            + acc_mem(&self.locvars)
            + acc_mem(&self.subfn)
            + acc_mem(&self.updecl)
    }
}

impl GcOp for Proto {
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

    fn mark_unreachable(&self) {
        todo!()
    }
}

impl TypeTag for Proto {
    const TAGID: Tag = Tag::Proto;
}

impl Display for Proto {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        self.basic_fmt(self, f)?;
        for each in self.subfn.iter() {
            writeln!(f)?;
            self.basic_fmt(each, f)?;
        }
        Ok(())
    }
}

impl Debug for Proto {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        self.debug_fmt(self, f)?;
        for each in self.subfn.iter() {
            writeln!(f)?;
            self.debug_fmt(each, f)?;
        }
        Ok(())
    }
}

const LAST_8_DIGIT: usize = 0xFFFFFFFF;

impl Proto {
    pub fn nparams(&self) -> i32 {
        self.nparam as i32
    }

    pub fn nreg(&self) -> i32 {
        self.nreg as i32
    }

    pub fn nupdecl(&self) -> i32 {
        self.updecl.len() as i32
    }

    pub fn nconst(&self) -> i32 {
        self.kst.len() as i32
    }

    pub fn is_vararg(&self) -> bool {
        self.vararg
    }

    pub fn def_info(&self) -> (u32, u32) {
        (self.begline, self.endline)
    }

    pub fn bytecode(&self) -> &[Instruction] {
        &self.code
    }

    pub fn constant(&self) -> &[Value] {
        &self.kst
    }

    pub fn subproto(&self) -> &[Gc<Proto>] {
        &self.subfn
    }

    pub fn updecl(&self) -> &[UpvalDecl] {
        &self.updecl
    }

    pub fn is_pure(&self) -> bool {
        !self.vararg && self.updecl.is_empty()
    }

    /// Locate source code line infomation by a pc counter.
    pub fn locate(&self, pc: i32) -> u32 {
        self.pcline[pc as usize]
    }

    fn basic_fmt(&self, p: &Proto, f: &mut std::fmt::Formatter<'_>) -> Result<(), std::fmt::Error> {
        debug_assert!(p.source.is_str());
        writeln!(
            f,
            "function < {}:{},{} > ({} instructions at 0x{:X})",
            unsafe { p.source.as_str_unchecked() },
            p.begline,
            p.endline,
            p.code.len(),
            p as *const Proto as usize & LAST_8_DIGIT
        )?;
        if p.vararg {
            f.write_str("vararg params, ")?;
        } else {
            write!(f, "{} params, ", p.nparam)?;
        }
        writeln!(
            f,
            "{} slots, {} upvalue, {} locals, {}, constants, {} functions",
            p.nreg,
            p.updecl.len(),
            p.locvars.len(),
            p.kst.len(),
            p.subfn.len()
        )?;

        for (idx, code) in p.code.iter().enumerate() {
            let line = p.pcline.get(idx).unwrap_or(&0);
            write!(f, "\t{idx}\t[{}]\t{:?>8} ; ", line, code)?;
            self.isc_extra_info(code, f)?;
            writeln!(f)?;
        }

        Ok(())
    }

    fn debug_fmt(&self, p: &Proto, f: &mut std::fmt::Formatter<'_>) -> Result<(), std::fmt::Error> {
        self.basic_fmt(p, f)?;

        let self_addr = p as *const Proto as usize;
        writeln!(
            f,
            "constants ({}) for 0x{:X}",
            p.kst.len(),
            self_addr & LAST_8_DIGIT
        )?;
        for (idx, k) in p.kst.iter().enumerate() {
            writeln!(f, "\t{}\t{:?}", idx, k)?;
        }

        writeln!(
            f,
            "locals ({}) for 0x{:X}",
            p.locvars.len(),
            self_addr & LAST_8_DIGIT
        )?;
        for (idx, loc) in p.locvars.iter().enumerate() {
            writeln!(f, "\t{idx}\t{}", loc.name.as_str())?;
        }

        writeln!(
            f,
            "upvalues ({}) for 0x{:X}",
            p.updecl.len(),
            self_addr & LAST_8_DIGIT
        )?;
        for (idx, up) in p.updecl.iter().enumerate() {
            writeln!(f, "\t{idx}\t{}", up.name())?;
        }

        Ok(())
    }

    fn isc_extra_info(
        &self,
        code: &Instruction,
        f: &mut std::fmt::Formatter,
    ) -> Result<(), std::fmt::Error> {
        match code.mode() {
            OpMode::IABC => {
                let (isc, a, b, c, k) = code.repr_abck();
                match isc {
                    MOVE => write!(f, "r({}) = r({})", a, b)?,
                    LOADFALSE => write!(f, "r({}) = false", a)?,
                    LOADTRUE => write!(f, "r({}) = true", a)?,
                    LOADNIL => write!(f, "r({}) ... r({}) = nil", a, a + b)?,
                    GETTABUP => write!(
                        f,
                        "r({}) = {}[{:?}]",
                        a,
                        self.updecl[b as usize].name(),
                        self.kst[c as usize]
                    )?,
                    SETTABUP => write!(
                        f,
                        "{}[{}] = {}",
                        self.updecl[a as usize].name(),
                        self.kst[b as usize],
                        self.kst[c as usize]
                    )?,
                    SETFIELD => {
                        if k {
                            write!(
                                f,
                                "r({})[{}] = {}",
                                a, self.kst[b as usize], self.kst[c as usize]
                            )?
                        } else {
                            write!(f, "r({})[{}] = r({})", a, self.kst[b as usize], c)?
                        }
                    }
                    NEWTABLE => write!(f, "r({}) = {{}}", a)?,
                    CALL => write!(f, "Call r({}) with {} in, {} out  <--", a, b - 1, c - 1)?,
                    RETURN => write!(
                        f,
                        "Return {} values of r({}) ... r({})  ==>",
                        b - 1,
                        a,
                        b - 2
                    )?,
                    RETURN0 => write!(f, "Return 0 value  -->")?,
                    RETURN1 => write!(f, "Return r({}) -=>", a)?,
                    _ => {}
                };
            }
            OpMode::IABx => {
                let (op, a, bx) = code.repr_abx();
                match op {
                    LOADK => write!(f, "r({}) = {:?}", a, self.kst[bx as usize])?,
                    CLOSURE => write!(
                        f,
                        "r({}) = Closure[{}] at 0x{:X}",
                        a,
                        bx,
                        self.subfn[bx as usize].address()
                    )?,
                    NEWTABLE => write!(f, "r({}) = {{}}", a)?,
                    _ => {}
                }
            }
            OpMode::IAsBx => {
                let (op, a, sbx) = code.repr_asbx();
                match op {
                    LOADI => write!(f, "r({}) = {}", a, sbx)?,
                    LOADF => todo!(),
                    _ => {}
                }
            }
            OpMode::IAx => {
                let (op, _) = code.repr_ax();
                match op {
                    OpCode::EXTRAARG => {}
                    _ => unreachable!(),
                };
            }
            OpMode::IsJ => {
                // let (isc, jmp) = code.repr_sj();
                todo!()
            }
        };
        Ok(())
    }
}

#[derive(Debug, Default)]
pub struct LocVarDecl {
    pub name: String,
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
    Up(RegIndex),   // index of updecl
}

use OpCode::*;

/// Code generation intermidiate state for each Proto
pub struct GenState {
    pub lables: BTreeMap<String, u32>,       // map lable -> pc
    pub jumpbp: Vec<(u32, String)>,          // jump backpatch (iscidx, lable), used for `goto`
    pub loopbp: LinkedList<Vec<(u32, u32)>>, // loop backpatch (iscidx, pc), used for `break`
    pub nextreg: RegIndex,                   // next free reg index
    pub maxreg: u8,                          // max reg index
    pub ksts: Vec<Value>,                    // constants
    pub upvals: Vec<UpvalDecl>,              // upvalue declration
    pub code: Vec<Instruction>,              // byte code
    pub subproto: Vec<Proto>,                // sub functions
    pub locstate: Vec<Vec<LocVarDecl>>,      // local variable infomation for each basic block
    pub local: Vec<LocVarDecl>,              // all local variable infomation
    pub srcfile: Value,                      // source file name
    pub absline: Vec<u32>,                   // line number of each bytecode
}

/// a short name for Instruction
type Isc = Instruction;

enum LookupState {
    Local { reg: RegIndex },
    UpList { idx: RegIndex },
}

impl GenState {
    pub fn new(srcfile: Value) -> Self {
        Self {
            lables: BTreeMap::new(),
            jumpbp: Vec::default(),
            loopbp: LinkedList::default(),
            nextreg: 0,
            maxreg: 0,
            ksts: Vec::new(),
            upvals: Vec::new(),
            subproto: Vec::new(),
            code: Vec::with_capacity(16),
            locstate: Vec::with_capacity(4),
            local: Vec::with_capacity(4),
            srcfile,
            absline: Vec::with_capacity(16),
        }
    }

    pub fn with_env(srcfile: Value) -> Self {
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

    /// Allocate a free register on vm stack and return its index
    fn alloc_free_reg(&mut self) -> RegIndex {
        let idx = self.nextreg;
        self.nextreg += 1;
        self.maxreg = self.maxreg.max(self.nextreg as u8);
        idx
    }

    /// Free last allocated register on vm stack, return next available register
    fn free_reg(&mut self) -> RegIndex {
        self.nextreg -= 1;
        self.nextreg
    }

    // Allocate a constant register and return its index
    fn alloc_const_reg(&mut self, k: Value) -> i32 {
        // reuse const register
        for (idx, kval) in self.ksts.iter().enumerate() {
            if kval == &k {
                return idx as i32;
            }
        }
        let idx = self.ksts.len();
        self.ksts.push(k);
        idx as i32
    }

    /// Find string in constants and return its index, this will alloc a const string if not found.
    fn find_or_create_kstr(&mut self, ks: &str, mem: &mut Heap) -> RegIndex {
        for (idx, v) in self.ksts.iter().enumerate() {
            if let Value::Str(sw) = v {
                if sw.as_str() == ks {
                    return idx as RegIndex;
                }
            }
        }
        self.alloc_const_reg(mem.alloc_str(ks).into())
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

        for params in self.local.iter() {
            if params.name == name {
                return Some(params.reg);
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

    fn emit_local_decl(&mut self, name: String, status: ExprStatus, ln: u32) {
        let locdecl = LocVarDecl {
            name,
            reg: self.try_load_expr_to_local(status, ln),
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
        debug_assert!(!val.is_gcobj());
        if let Value::Int(i) = val {
            if *i < Isc::MAX_B as i64 {
                Some(*i as i32)
            } else {
                None
            }
        } else {
            None
        }
    }

    fn try_load_expr_to_const(&mut self, es: ExprStatus) -> RegIndex {
        match es {
            ExprStatus::LitNil => self.alloc_const_reg(Value::Nil),
            ExprStatus::LitTrue => self.alloc_const_reg(Value::Bool(true)),
            ExprStatus::LitFalse => self.alloc_const_reg(Value::Bool(false)),
            ExprStatus::LitInt(i) => self.alloc_const_reg(Value::Int(i)),
            ExprStatus::LitFlt(f) => self.alloc_const_reg(Value::Float(f)),
            ExprStatus::Kst(kidx) => kidx,
            _ => unreachable!(),
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
            ExprStatus::LitFlt(f) => {
                // TODO:
                // use LOADF for small float.
                let free = self.alloc_free_reg();
                let kreg = self.alloc_const_reg(Value::Float(f));
                self.emit(Isc::iabx(LOADK, free, kreg), ln);
                free
            }
            ExprStatus::Kst(k) => self.load_const(k, ln),
            ExprStatus::Up(u) => {
                let free = self.alloc_free_reg();
                self.emit(Isc::iabc(GETUPVAL, free, u, 0), ln);
                free
            }
            ExprStatus::Call(c) => c,
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
    // keep origin status of a expression
    Keep,

    // extra expr in multi assignment, all expression will be evaluated and ignored
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
pub enum CodeGenError {
    TooManyLocalVariable,
    RegisterOverflow,
    BadVarargUse,
    BreakNotInLoopBlock,
    RepeatedLable { lable: String },
    NonexistedLable { lable: String },
}

/// Code generation pass
pub struct CodeGen {
    strip: bool,                  // strip debug information
    genstk: LinkedList<GenState>, // generation state stack
}

impl Deref for CodeGen {
    type Target = GenState;

    fn deref(&self) -> &Self::Target {
        debug_assert!(!self.genstk.is_empty());
        // SAFETY: a new state will be pushed to generation state stack when into a function
        unsafe { self.genstk.back().unwrap_unchecked() }
    }
}

impl DerefMut for CodeGen {
    fn deref_mut(&mut self) -> &mut Self::Target {
        debug_assert!(!self.genstk.is_empty());
        // SAFETY: a new state will be pushed to generation state stack when into a function
        unsafe { self.genstk.back_mut().unwrap_unchecked() }
    }
}

impl CodeGen {
    const ANONYMOUS: &'static str = "?";

    fn new(strip: bool) -> Self {
        Self {
            strip,
            genstk: LinkedList::new(),
        }
    }

    fn conume(
        &mut self,
        ast_root: Box<SrcLoc<Block>>,
        mem: &mut Heap,
    ) -> Result<Proto, CodeGenError> {
        debug_assert_eq!(self.genstk.len(), 0);

        // take chunk name, construct an empty para list
        let name: Value = mem.alloc_str(ast_root.name()).into();
        let plist = ParameterList {
            vargs: true,
            namelist: Vec::new(),
        };

        let mut res = self.walk_fn_body(name, plist, ast_root, false, mem)?;

        // _ENV must existed in main chunk
        if res.updecl.is_empty() {
            let _ = std::mem::replace(&mut res.updecl, {
                let mut vec = Vec::with_capacity(1);
                vec.push(UpvalDecl::Env);
                vec.into_boxed_slice()
            });
        }

        if self.strip {
            Self::strip_src_info(&mut res, mem.alloc_fixed(Self::ANONYMOUS))
        }

        debug_assert!(!res.updecl.is_empty());
        debug_assert_eq!(self.genstk.len(), 0);
        Ok(res)
    }

    fn walk_fn_body(
        &mut self,
        src: Value,
        params: ParameterList,
        body: Box<SrcLoc<Block>>,
        selfcall: bool,
        mem: &mut Heap,
    ) -> Result<Proto, CodeGenError> {
        // prepare another GenState for new function
        self.genstk.push_back(GenState::new(src));

        // parameters defination begin line
        let parline = body.def_begin();
        let nparam = params.namelist.len();

        // init parameters
        for name in params.namelist.into_iter() {
            let preg = self.alloc_free_reg();
            self.local.push(LocVarDecl {
                name,
                reg: preg,
                start_pc: 0,
                end_pc: 0,
            });
        }

        // vararg
        if params.vargs {
            self.emit(Isc::iabc(VARARGPREP, nparam as i32, 0, 0), parline);
        }

        // reserve space for self call
        if selfcall {
            todo!("self call ")
            // let free = self.alloc_free_reg();
            // self.emit_local_decl("self".to_string(), ExprStatus::Reg(free), (0, 0));
        }

        let defend = body.def_end();
        let returned = self.walk_basic_block(body, mem)?;

        if !returned {
            self.emit(Isc::iabc(RETURN0, 0, 0, 0), defend);
        }

        // backpatch goto
        while let Some((pc, lable)) = self.jumpbp.pop() {
            let index = pc as usize;
            if let Some(dest) = self.lables.get(&lable) {
                let step = (*dest as i64) - pc as i64;
                self.code[index].code = Isc::isj(JMP, step as i32).code;
            } else {
                return Err(CodeGenError::NonexistedLable { lable });
            }
        }

        // TODO:
        // Check <close> variable and generate CLOSE instruction
        // Check <const> variable

        fn steal<T>(elem: &mut Vec<T>) -> Box<[T]> {
            std::mem::take(elem).into_boxed_slice()
        }

        let res = Proto {
            vararg: params.vargs,
            nparam: nparam as u8,
            nreg: self.maxreg,
            begline: self.absline.first().copied().unwrap_or(0),
            endline: self.absline.last().copied().unwrap_or(0),
            source: self.srcfile,
            kst: steal(&mut self.ksts),
            code: steal(&mut self.code),
            pcline: steal(&mut self.absline),
            locvars: steal(&mut self.local),
            updecl: steal(&mut self.upvals),
            subfn: std::mem::take(&mut self.subproto)
                .into_iter()
                .map(Gc::from)
                .collect(),
        };

        // reset gen state stack
        self.genstk.pop_back();
        Ok(res)
    }

    /// Strip debug infomation in proto.
    fn strip_src_info(p: &mut Proto, anonymous: Value) {
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

    // Generate bytecode for basic block, return weither this block contains return statement.
    fn walk_basic_block(
        &mut self,
        mut body: Box<SrcLoc<Block>>,
        mem: &mut Heap,
    ) -> Result<bool, CodeGenError> {
        self.locstate.push(Vec::with_capacity(4));

        let end = body.def_end();
        for stmt in std::mem::take(&mut body.stats).into_iter() {
            self.walk_stmt(stmt, mem)?;
        }

        if let Some(ls) = self.locstate.pop() {
            self.local.extend(ls);
        } else {
            unreachable!()
        }

        self.walk_return(std::mem::take(&mut body.ret), end, mem)
    }

    fn walk_stmt(&mut self, stmt: StmtNode, mem: &mut Heap) -> Result<(), CodeGenError> {
        let lineinfo = stmt.def_info();
        match stmt.inner() {
            Stmt::Assign { vars, exprs } => self.walk_assign_stmt(vars, exprs, mem),
            Stmt::Lable(lable) => {
                let dest = self.cur_pc();
                if self.lables.contains_key(lable.as_str()) {
                    Err(CodeGenError::RepeatedLable { lable })
                } else {
                    self.lables.insert(lable, dest);
                    Ok(())
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
                Ok(())
            }
            Stmt::Break => {
                let (idx, pc) = (self.code.len(), self.cur_pc());
                if let Some(lop) = self.loopbp.back_mut() {
                    lop.push((idx as u32, pc));
                    self.emit(Isc::placeholder(), 0);
                    Ok(())
                } else {
                    Err(CodeGenError::BreakNotInLoopBlock)
                }
            }
            Stmt::DoEnd(block) => {
                let has_ret = self.walk_basic_block(block, mem)?;
                debug_assert!(!has_ret);
                Ok(())
            }
            Stmt::While { exp, block } => self.walk_while_loop(exp, block, mem),
            Stmt::Repeat { block, exp } => self.walk_repeat_loop(exp, block, mem),
            Stmt::IfElse {
                cond: exp,
                then,
                els,
            } => self.walk_branch_stmt(exp, then, els, mem),
            Stmt::NumericFor(num) => self.walk_numberic_loop(num, mem),
            Stmt::GenericFor(gen) => self.walk_generic_for(gen, mem),
            Stmt::LocalVarDecl { names, exprs } => self.walk_local_decl(names, exprs, mem),
            Stmt::Expr(exp) => {
                let _ = self.walk_common_expr(exp, Ctx::Ignore, mem)?;
                Ok(())
            }
        }
    }

    fn walk_repeat_loop(
        &mut self,
        cond: ExprNode,
        block: Box<SrcLoc<Block>>,
        mem: &mut Heap,
    ) -> Result<(), CodeGenError> {
        let def = cond.def_begin();
        self.enter_loop();
        self.walk_basic_block(block, mem)?;
        self.leave_loop();
        let cond_reg = {
            let s = self.walk_common_expr(cond, Ctx::Allocate, mem)?;
            self.try_load_expr_to_local(s, def)
        };
        self.emit(Isc::iabc(TEST, cond_reg, true as i32, 0), def);
        Ok(())
    }

    fn walk_while_loop(
        &mut self,
        cond: ExprNode,
        block: Box<SrcLoc<Block>>,
        mem: &mut Heap,
    ) -> Result<(), CodeGenError> {
        self.enter_loop();
        let ln = cond.def_begin();
        let cond_reg = {
            let sts = self.walk_common_expr(cond, Ctx::Allocate, mem)?;
            self.try_load_expr_to_local(sts, ln)
        };
        self.emit(Isc::iabc(TEST, cond_reg, false as i32, 0), ln);
        let loop_begin = self.set_recover_point(ln);
        self.walk_basic_block(block, mem)?;
        self.leave_loop();

        let step = (self.cur_pc() - loop_begin.1) as i32;
        self.emit_backpatch(loop_begin.0, Isc::isj(JMP, step));
        Ok(())
    }

    fn walk_numberic_loop(
        &mut self,
        n: Box<NumericFor>,
        mem: &mut Heap,
    ) -> Result<(), CodeGenError> {
        let mut init_reg = {
            let line = n.init.def_begin();
            let s = self.walk_common_expr(n.init, Ctx::Allocate, mem)?;
            self.try_load_expr_to_local(s, line)
        };
        let mut limit_reg = {
            let line = n.limit.def_begin();
            let s = self.walk_common_expr(n.limit, Ctx::Allocate, mem)?;
            self.try_load_expr_to_local(s, line)
        };
        let mut step_reg = {
            let line = n.step.def_begin();
            let s = self.walk_common_expr(n.step, Ctx::Allocate, mem)?;
            self.try_load_expr_to_local(s, line)
        };

        let loopdef = n.body.def_info();
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
        self.walk_basic_block(n.body, mem)?;
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

    fn walk_generic_for(&mut self, g: Box<GenericFor>, mem: &mut Heap) -> Result<(), CodeGenError> {
        let def = g.body.def_info();

        let niter = g.iters.len();
        // alloc 4 register to store loop state
        let state_reg = self.alloc_free_reg();
        for _ in 0..3 {
            self.alloc_free_reg();
        }
        let (iscidx, recover_pc) = self.set_recover_point(def.0);

        // treat iters as local var decl
        let vars = g.iters.into_iter().map(|name| (name, None)).collect();
        self.walk_local_decl(vars, g.exprs, mem)?;

        // loop body
        self.enter_loop();
        self.walk_basic_block(g.body, mem)?;

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
        exp: ExprNode,
        then: Box<SrcLoc<Block>>,
        els: Option<BasicBlock>,
        mem: &mut Heap,
    ) -> Result<(), CodeGenError> {
        let cond_def = exp.def_begin();
        let cond = self.walk_common_expr(exp, Ctx::Allocate, mem)?;
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
        self.walk_basic_block(then, mem)?;

        if let Some(bk) = els {
            // place holder fpr JMP to else block end
            branch.then_end_jmp_idx = Some(unsafe { NonZeroU32::new_unchecked(self.cur_pc()) });
            self.emit_placeholder(then_defend);

            branch.else_entry_pc = Some(unsafe { NonZeroU32::new_unchecked(self.cur_pc()) });
            self.walk_basic_block(bk, mem)?;
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

    fn walk_local_decl(
        &mut self,
        names: Vec<(SrcLoc<String>, Option<Attribute>)>,
        mut exprs: Vec<ExprNode>,
        mem: &mut Heap,
    ) -> Result<(), CodeGenError> {
        debug_assert!(!names.is_empty());
        debug_assert!(!exprs.is_empty());

        let ndecl = names.len();

        // fill expr with nil if there are too many variables than the number of expressions.
        if exprs.len() < ndecl {
            let line = unsafe { names.last().unwrap_unchecked().0.def_info() };
            exprs.resize_with(ndecl, || ExprNode::new(SrcLoc::new(Expr::Nil, line)));
        }

        // TODO:
        // add attribute support
        for ((name, _attr), expr) in names.into_iter().zip(exprs.iter_mut()) {
            let ln = name.def_begin();
            let status = self.walk_common_expr(std::mem::take(expr), Ctx::Allocate, mem)?;
            self.emit_local_decl(name.inner(), status, ln);
        }

        // evaluate extra expression and drop result
        for extra in exprs.into_iter().skip(ndecl) {
            let _ = self.walk_common_expr(extra, Ctx::Ignore, mem)?;
        }

        Ok(())
    }

    fn walk_fn_def(
        &mut self,
        fnbody: FuncBody,
        dest: i32,
        def: (u32, u32),
        mem: &mut Heap,
    ) -> Result<ExprStatus, CodeGenError> {
        let pto = self.walk_fn_body(self.srcfile, fnbody.params, fnbody.body, false, mem)?;
        self.subproto.push(pto);
        let pidx = self.subproto.len() - 1;
        self.emit(Isc::iabx(CLOSURE, dest, pidx as i32), def.0);
        debug_assert_eq!(self.nextreg, dest + 1);
        self.free_reg();

        Ok(ExprStatus::Reg(dest))
    }

    fn walk_return(
        &mut self,
        ret: Option<Vec<ExprNode>>,
        ln_if_no_ret: u32,
        mem: &mut Heap,
    ) -> Result<bool, CodeGenError> {
        if let Some(mut rets) = ret {
            match rets.len() {
                0 => {
                    self.emit(Isc::iabc(RETURN0, 0, 0, 0), ln_if_no_ret);
                }
                1 => {
                    // SAFETY: rets.len() == 1
                    let ret = unsafe { rets.pop().unwrap_unchecked() };
                    let ln = ret.def_begin();
                    let status = self.walk_common_expr(ret, Ctx::PotentialTailCall, mem)?;
                    match status {
                        ExprStatus::Call(_ret) => {
                            todo!("tail call")
                            // self.emit(Isc::iabc(RETURN, ret, 0, 0), ln)
                        }
                        otherwise => {
                            let reg = self.try_load_expr_to_local(otherwise, ln);
                            self.emit(Isc::iabc(RETURN1, reg, 0, 0), ln);
                        }
                    }
                }
                n if n > 1 => {
                    // todo: multi return
                    todo!("multi return")
                }
                _ => unreachable!(),
            };
            Ok(true)
        } else {
            Ok(false)
        }
    }

    fn walk_fn_call(
        &mut self,
        call: FuncCall,
        fnreg: RegIndex,
        exp_ret: u32,
        tail_call: bool,
        mem: &mut Heap,
    ) -> Result<ExprStatus, CodeGenError> {
        match call {
            FuncCall::MethodCall {
                prefix: _,
                method: _,
                args: _,
            } => {
                todo!("method call")
            }
            FuncCall::FreeFnCall { prefix, mut args } => {
                let nparam = args.namelist.len();
                let callln = prefix.def_begin();

                let fnreg_real = match self.walk_common_expr(prefix, Ctx::must_use(fnreg), mem)? {
                    ExprStatus::Reg(reg) => {
                        // function
                        if reg != self.nextreg - 1 {
                            let tail = self.alloc_free_reg();
                            self.emit(Isc::iabc(MOVE, tail, fnreg, 0), callln);
                            tail
                        } else {
                            reg
                        }
                    }
                    _ => unreachable!(),
                };

                debug_assert_eq!(fnreg, fnreg_real);

                let mut n: u32 = 0;
                for param in std::mem::take(&mut args.namelist).into_iter() {
                    n += 1;
                    let preg = self.alloc_free_reg();
                    let _ = self.walk_common_expr(param, Ctx::must_use(preg), mem);
                }
                while n != 0 {
                    self.free_reg();
                    n -= 1;
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
        mem: &mut Heap,
    ) -> Result<(), CodeGenError> {
        debug_assert!(!vars.is_empty());
        debug_assert!(!exprs.is_empty());

        let nvar = vars.len();

        // fill expr with nil if there are too many variables than the number of expressions.
        if nvar > exprs.len() {
            let line = unsafe { vars.last().unwrap_unchecked().def_info() };
            exprs.resize_with(exprs.len(), || ExprNode::new(SrcLoc::new(Expr::Nil, line)));
        }

        for (mut decl, exp) in vars.into_iter().zip(exprs.iter_mut()) {
            let ln = decl.def_begin();
            let valnode = std::mem::take(exp);
            match decl.inner_mut() {
                Expr::Ident(id) => self.emit_assign_to_ident(id, valnode, mem, ln)?,
                _subscript @ Expr::Subscript { .. } => {
                    todo!("suscript assignment")
                    // TODO:
                    // a.b.c = 1
                    //  VARARGPREP      0
                    //  GETTABUP        0 0 0   ; _ENV "a"
                    //  GETFIELD        0 0 1   ; "b"
                    //  SETFIELD        0 2 3k  ; "c" 1
                    //  RETURN          0 1 1   ; 0 out
                }
                _ => unreachable!("there must be a Expr::Ident or Expr::Subscript node to assign."),
            };
        }

        // evaluate extra expression and drop result
        for extra in exprs.into_iter().skip(nvar) {
            let _ = self.walk_common_expr(extra, Ctx::Ignore, mem)?;
        }

        Ok(())
    }

    fn emit_assign_to_ident(
        &mut self,
        id: &mut String,
        valnode: Box<SrcLoc<Expr>>,
        mem: &mut Heap,
        vardef: u32,
    ) -> Result<(), CodeGenError> {
        let valdef = valnode.def_begin();
        match self.lookup_name(id) {
            LookupState::Local { reg } => {
                self.walk_common_expr(valnode, Ctx::must_use(reg), mem)?;
            }
            LookupState::UpList { idx: upidx } => {
                let valsts = self.walk_common_expr(valnode, Ctx::Keep, mem)?;
                let decl_kreg = self.alloc_const_reg(Value::from(mem.take_str(std::mem::take(id))));
                match &self.upvals[upidx as usize] {
                    UpvalDecl::Env => match valsts {
                        ExprStatus::Call(reg) | ExprStatus::Reg(reg) => {
                            self.emit(Isc::iabc(SETTABUP, 0, decl_kreg, reg), valdef);
                        }

                        upstatus @ ExprStatus::Up(_) => {
                            let valreg = self.try_load_expr_to_local(upstatus, valdef);
                            self.emit(Isc::iabc(SETTABUP, 0, decl_kreg, valreg), valdef);
                        }

                        ExprStatus::Kst(val_kreg) => {
                            self.emit(Isc::iabck(SETTABUP, 0, decl_kreg, val_kreg), valdef);
                        }

                        literals => {
                            let val_kreg = self.try_load_expr_to_const(literals);
                            self.emit(Isc::iabck(SETTABUP, 0, decl_kreg, val_kreg), valdef);
                        }
                    },

                    _ => {
                        let valreg = self.try_load_expr_to_local(valsts, valdef);
                        self.emit(Isc::iabc(SETUPVAL, valreg, upidx, 0), vardef);
                    }
                };
            }
        };
        Ok(())
    }

    fn walk_common_expr(
        &mut self,
        node: ExprNode,
        ctx: ExprGenCtx,
        mem: &mut Heap,
    ) -> Result<ExprStatus, CodeGenError> {
        let (def, node) = (node.def_info(), node.inner());

        match ctx {
            Ctx::Keep => {
                let status = match node {
                    Expr::Nil => ExprStatus::LitNil,
                    Expr::False => ExprStatus::LitFalse,
                    Expr::True => ExprStatus::LitTrue,
                    Expr::Int(i) => ExprStatus::LitInt(i),
                    Expr::Float(f) => ExprStatus::LitFlt(f),
                    Expr::Literal(l) => {
                        let kreg = self.alloc_const_reg(mem.take_str(l).into());
                        ExprStatus::Kst(kreg)
                    }
                    // Expr::Ident()
                    otherwhise => {
                        let free = self.alloc_free_reg();
                        self.emit_expr(otherwhise, free, def, mem)?
                    }
                };
                Ok(status)
            }

            // case that the value of expression will be ignored,
            // but sub expr may make an side effect
            Ctx::Ignore => self.emit_ignored_expr(node, mem),

            Ctx::Allocate => {
                let free = self.alloc_free_reg();
                let status = self.emit_expr(node, free, def, mem)?;
                // if let ExprStatus::Reg(ref r) = status {
                //     if *r < free {
                //         self.free_reg();
                //         debug_assert_eq!(self.nextreg, free);
                //     }
                // }
                Ok(status)
            }

            Ctx::NonRealloc { dest } => {
                let mut status = self.emit_expr(node, dest, def, mem)?;
                if let ExprStatus::Reg(real) = status {
                    if real != dest {
                        self.emit(Isc::iabc(MOVE, dest, real, 0), def.0);
                        status = ExprStatus::Reg(dest)
                    }
                }
                Ok(status)
            }

            Ctx::PotentialTailCall => match node {
                Expr::FuncCall(call) => {
                    let reg = self.alloc_free_reg();
                    self.walk_fn_call(call, reg, 1, true, mem)
                }
                Expr::Dots => {
                    // check weither in vararg function
                    if let Some(isc) = self.code.first() {
                        if isc.get_op() != VARARGPREP {
                            Err(CodeGenError::BadVarargUse)
                        } else {
                            let vararg_reg = self.local.len() as RegIndex;
                            self.emit(Isc::iabc(VARARG, vararg_reg, 0, 0), def.0);
                            Ok(ExprStatus::Reg(vararg_reg))
                        }
                    } else {
                        Err(CodeGenError::BadVarargUse)
                    }
                }

                Expr::Ident(id) => {
                    let reg = self.lookup_and_load(id, None, def.0, mem);
                    Ok(ExprStatus::Reg(reg))
                }

                Expr::True
                | Expr::False
                | Expr::Nil
                | Expr::Int(_)
                | Expr::Float(_)
                | Expr::Literal(_)
                | Expr::Lambda(_)
                | Expr::TableCtor(_)
                | Expr::Subscript { .. } => {
                    let reg = self.alloc_free_reg();
                    let status = self.emit_expr(node, reg, def, mem)?;
                    // debug_assert!()
                    if let ExprStatus::Reg(ref r) = status {
                        debug_assert_eq!(*r, reg);
                    } else {
                        unreachable!()
                    }
                    Ok(status)
                }

                Expr::UnaryOp { op, expr } => {
                    let next = self.nextreg;
                    self.emit_unary_expr(expr, op, next, mem)
                }

                Expr::BinaryOp { lhs, op, rhs } => {
                    //  left
                    let ln = lhs.def_begin();
                    let left = self.walk_common_expr(lhs, Ctx::PotentialTailCall, mem)?;
                    let reg = self.try_load_expr_to_local(left, ln);

                    // right
                    let right = self.walk_common_expr(rhs, Ctx::PotentialTailCall, mem)?;
                    self.emit_binop_optimized(ExprStatus::Reg(reg), right, reg, def, op)
                }
            },

            Ctx::MultiLevelTableIndex { depth: _ } => {
                todo!("expr codegen: multi level Table Index optimize")
            }
        }
    }

    fn emit_ignored_expr(
        &mut self,
        node: Expr,
        mem: &mut Heap,
    ) -> Result<ExprStatus, CodeGenError> {
        let pre_state = self.nextreg;
        let status = match node {
            Expr::Subscript { prefix, key } => {
                let _ = self.walk_common_expr(prefix, Ctx::Ignore, mem);
                let _ = self.walk_common_expr(key, Ctx::Ignore, mem);
                ExprStatus::Reg(RegIndex::MAX)
            }
            Expr::FuncCall(call) => {
                let next = self.alloc_free_reg();
                let _ = self.walk_fn_call(call, next, 0, false, mem);
                self.free_reg();
                ExprStatus::Reg(RegIndex::MAX)
            }
            Expr::TableCtor(ctor) => {
                for field in ctor {
                    let _ = self.walk_common_expr(field.val, Ctx::Ignore, mem);
                }
                ExprStatus::Reg(RegIndex::MAX)
            }
            Expr::BinaryOp {
                lhs: l,
                op: _,
                rhs: r,
            } => {
                let _ = self.walk_common_expr(l, Ctx::Ignore, mem);
                let _ = self.walk_common_expr(r, Ctx::Ignore, mem);
                ExprStatus::Reg(RegIndex::MAX)
            }
            Expr::UnaryOp { op: _, expr } => {
                let _ = self.walk_common_expr(expr, Ctx::Ignore, mem);
                ExprStatus::Reg(RegIndex::MAX)
            }
            _ => ExprStatus::Reg(RegIndex::MAX),
        };
        // recover register state
        let mut try_recover = self.nextreg;
        debug_assert!(try_recover >= pre_state);
        while try_recover != pre_state {
            try_recover = self.free_reg();
        }
        Ok(status)
    }

    fn emit_expr(
        &mut self,
        exp: Expr,
        dest: RegIndex,
        def: (u32, u32),
        mem: &mut Heap,
    ) -> Result<ExprStatus, CodeGenError> {
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

            Expr::Literal(lit) => {
                let kreg = self.find_or_create_kstr(&lit, mem);
                self.emit(Isc::iabx(LOADK, dest, kreg), def.0);
                ExprStatus::Reg(dest)
            }

            Expr::Ident(id) => {
                let reg = self.lookup_and_load(id, Some(dest), def.0, mem);
                ExprStatus::Reg(reg)
            }

            Expr::UnaryOp { op, expr } => self.emit_unary_expr(expr, op, dest, mem)?,

            Expr::BinaryOp { lhs, op, rhs } => {
                self.emit_binary_expr(lhs, rhs, def, op, dest, mem)?
            }

            Expr::Lambda(fnbody) => self.walk_fn_def(fnbody, dest, def, mem)?,

            Expr::Subscript { prefix, key } => {
                let key_status = self.walk_common_expr(key, Ctx::Allocate, mem)?;
                match self.walk_common_expr(prefix, Ctx::MultiLevelTableIndex { depth: 1 }, mem)? {
                    ExprStatus::Reg(pre) | ExprStatus::Call(pre) => {
                        self.emit_index_local(key_status, def.0, pre, dest)
                    }
                    _ => unreachable!(),
                }
            }

            Expr::TableCtor(fields) => self.walk_table_ctor(fields, dest, def, mem)?,
            Expr::FuncCall(call) => self.walk_fn_call(call, dest, 1, false, mem)?,
            Expr::Dots => unreachable!(),
        };
        Ok(status)
    }

    fn emit_binary_expr(
        &mut self,
        lhs: ExprNode,
        rhs: ExprNode,
        def: (u32, u32),
        op: BinOp,
        destreg: RegIndex,
        mem: &mut Heap,
    ) -> Result<ExprStatus, CodeGenError> {
        let (lst, rst) = {
            let l = if let Some(ls) = GenState::peek_const_expr(&lhs) {
                ls
            } else {
                self.walk_common_expr(lhs, Ctx::must_use(destreg), mem)?
            };

            let right_reg = self.alloc_free_reg();
            let r = if let Some(rs) = GenState::peek_const_expr(&rhs) {
                rs
            } else {
                self.walk_common_expr(rhs, Ctx::must_use(right_reg), mem)?
            };
            self.free_reg(); // free right reg
            (l, r)
        };

        self.emit_binop_optimized(lst, rst, destreg, def, op)
    }

    fn emit_binop_optimized(
        &mut self,
        lst: ExprStatus,
        rst: ExprStatus,
        destreg: i32,
        def: (u32, u32),
        op: BinOp,
    ) -> Result<ExprStatus, CodeGenError> {
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
        expr: ExprNode,
        op: UnOp,
        freg: RegIndex,
        mem: &mut Heap,
    ) -> Result<ExprStatus, CodeGenError> {
        let ln = expr.def_begin();
        let oprand = self.walk_common_expr(expr, ExprGenCtx::Allocate, mem)?;

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
        mem: &mut Heap,
    ) -> Result<ExprStatus, CodeGenError> {
        self.emit(Isc::iabc(NEWTABLE, dest, 0, 0), def.0);
        self.emit(Isc::iax(EXTRAARG, 0), def.0);

        let mut aryidx = 1;
        for field in flist.into_iter() {
            let fdefloc = field.val.def_begin();
            let valstatus = self.walk_common_expr(field.val, Ctx::Allocate, mem)?;

            if let Some(key) = field.key {
                let keystatus = self.walk_common_expr(key, Ctx::Allocate, mem)?;
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

    /// Lookup a identifier and return LookupState. If ident was found as a upval in outter function,
    /// then add upvalue decl for all intermidiate function during search.
    fn lookup_name(&mut self, id: &str) -> LookupState {
        let try_lookup_in_frame = |f: &GenState| -> Option<LookupState> {
            if let Some(reg) = f.find_local_decl(id) {
                return Some(LookupState::Local { reg });
            }
            for (idx, up) in f.upvals.iter().enumerate() {
                if up.name() == id {
                    return Some(LookupState::UpList {
                        idx: idx as RegIndex,
                    });
                }
            }
            None
        };

        // find in self and outter function upvalue list
        if let Some((nrev, peek)) = self.genstk.iter().rev().enumerate().fold(
            None,
            |state, (idx, frame)| -> Option<(usize, LookupState)> {
                if state.is_none() {
                    try_lookup_in_frame(frame).map(|s| (idx, s))
                } else {
                    state
                }
            },
        ) {
            // add upvalue decl for all intermidiate function during search
            self.genstk
                .iter_mut()
                .rev()
                .take(nrev)
                .rev()
                .fold(peek, |prev_frame_state, gs| {
                    // set _ENV
                    let pos = if gs.upvals.is_empty() {
                        gs.upvals.push(UpvalDecl::Env);
                        0
                    } else {
                        gs.upvals.len()
                    };
                    let updecl = match prev_frame_state {
                        LookupState::Local { reg } => UpvalDecl::OnStack {
                            name: id.to_string(),
                            register: reg,
                        },
                        LookupState::UpList { idx } => UpvalDecl::InUpList {
                            name: id.to_string(),
                            offset: idx,
                        },
                    };
                    gs.upvals.push(updecl);
                    LookupState::UpList {
                        idx: pos as RegIndex,
                    }
                })
        } else {
            if self.upvals.is_empty() {
                self.upvals.push(UpvalDecl::Env);
            }
            LookupState::UpList { idx: 0 }
        }
    }
    fn lookup_and_load(
        &mut self,
        id: String,
        dest: Option<RegIndex>,
        ln: u32,
        mem: &mut Heap,
    ) -> RegIndex {
        match self.lookup_name(&id) {
            LookupState::Local { reg } => reg,
            LookupState::UpList { idx } => {
                let reg = dest.unwrap_or_else(|| self.alloc_free_reg());
                // get from _ENV
                if idx == 0 {
                    let name_kreg = self.find_or_create_kstr(&id, mem);
                    self.emit(Isc::iabc(GETTABUP, reg, 0, name_kreg), ln);
                    reg
                } else {
                    self.emit(Isc::iabc(GETUPVAL, reg, idx as RegIndex, 0), ln);
                    reg
                }
            }
        }
    }
}

impl CodeGen {
    pub fn codegen(
        ast_root: Box<SrcLoc<Block>>,
        strip: bool,
        mem: &mut Heap,
    ) -> Result<Proto, CodeGenError> {
        CodeGen::new(strip).conume(ast_root, mem)
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

pub fn dump_chunk(chunk: &Proto, bw: &mut BufWriter<impl Write>) -> std::io::Result<()> {
    ChunkDumper::dump(chunk, bw)
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

    pub fn undump(r: &mut BufReader<impl Read>, mem: &mut Heap) -> Result<Proto, BinLoadErr> {
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

        Self::undump_proto(r, mem)
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

    pub fn undump_proto(r: &mut BufReader<impl Read>, mem: &mut Heap) -> Result<Proto, BinLoadErr> {
        let nupval = Self::undump_varint(r)?;
        let src = mem.take_str(Self::undump_string(r)?).into();
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
            kst.push(Self::undump_const(r, mem)?)
        }

        let up_size = Self::undump_varint(r)?;
        let mut ups = Vec::with_capacity(up_size);
        for _ in 0..up_size {
            r.read_exact(&mut byte)?;
            let _onstk = u8::from_ne_bytes(byte) != 0;

            r.read_exact(&mut byte)?;
            let _stkid = u8::from_ne_bytes(byte);

            // r.read_exact(&mut byte)?;
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
            let mut p = Self::undump_proto(r, mem)?;
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
            kst: kst.into(),
            code: code.into(),
            subfn: subfn.into(),
            pcline: Box::new([]), // TODO
            source: src,
            locvars: Box::new([]), // TODO
            updecl: ups.into(),
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

    fn dump_const(val: &Value, w: &mut BufWriter<impl Write>) -> std::io::Result<()> {
        match val {
            Value::Nil => {
                w.write_all(&[0x00])?;
            }
            Value::Bool(b) => {
                if *b {
                    w.write_all(&[0x11])?;
                } else {
                    w.write_all(&[0x1])?;
                }
            }
            Value::Int(i) => {
                w.write_all(&[0x03])?;
                unsafe { Self::dump_varint(std::mem::transmute(i), w)? }
            }
            Value::Float(f) => {
                w.write_all(&[0x13])?;
                Self::dump_float(*f, w)?;
            }
            Value::Str(s) => {
                if s.is_internalized() {
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

    fn undump_const(r: &mut BufReader<impl Read>, mem: &mut Heap) -> std::io::Result<Value> {
        let mut byte = [0; 1];
        r.read_exact(&mut byte)?;
        let val = match u8::from_ne_bytes(byte) {
            0x00 => Value::Nil,
            0x01 => Value::Bool(false),
            0x11 => Value::Bool(true),
            0x03 => Value::Int(unsafe { std::mem::transmute(Self::undump_varint(r)?) }),
            0x13 => Value::Float(Self::undump_float(r)?),
            0x04 | 0x14 => mem.take_str(Self::undump_string(r)?).into(),
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

#[cfg(test)]
mod test {
    use super::*;

    use std::{
        fmt::Debug,
        fs::File,
        io::{BufReader, BufWriter, Write},
    };

    #[test]
    fn instruction_build() {
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
        use crate::parser::Parser;
        let mut heap = Heap::default();
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
        let origin = super::CodeGen::codegen(ast, false, &mut heap).unwrap();
        // println!("{:?}", origin);

        let tmp_file = std::fs::File::create(tmpfile.clone()).unwrap();
        let mut writer = BufWriter::new(tmp_file);
        ChunkDumper::dump(&origin, &mut writer).unwrap();
        writer.flush().unwrap();

        let same_file = std::fs::File::open(tmpfile.clone()).unwrap();
        let mut reader = BufReader::new(same_file);
        let _recover = ChunkDumper::undump(&mut reader, &mut heap).unwrap();
        // println!("{:?}", recover);

        // let _vm = State::new();
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
