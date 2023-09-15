use std::{
    collections::{BTreeMap, LinkedList},
    fmt::{Debug, Display},
    io::{BufReader, BufWriter, Read, Write},
    ops::{Deref, DerefMut},
    rc::Rc,
};

use crate::{
    ast::{Attribute, BinOp, Block, Expr, ExprNode, FuncCall, ParaList, Stmt, StmtNode, UnOp},
    heap::{Gc, Heap, HeapMemUsed},
    state::RegIndex,
    value::{LValue, StrImpl},
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
            OpMode::IAx => f.write_str("IAx "),
            OpMode::IsJ => f.write_str("IsJ "),
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

    pub fn iabck(op: OpCode, a: i32, b: i32, c: i32, isk: bool) -> Self {
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
        (self.code as i32 & Self::MASK_SBX as i32) >> Self::OFFSET_SBX
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

impl Display for Instruction {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let mode = self.mode();
        write!(f, "{:<4} ", mode)?;

        match mode {
            OpMode::IABC => {
                let (code, a, b, c, k) = self.repr_abck();
                write!(f, "{:<16}\t{:<3} {:<3} {:<3}", code, a, b, c)?;
                if k {
                    f.write_str("k")
                } else {
                    Ok(())
                }
            }
            OpMode::IABx => {
                let (code, a, bx) = self.repr_abx();
                write!(f, "{code:<16}\t{:<3} {:<3}    ", a, bx)
            }
            OpMode::IAsBx => {
                let (code, a, sbx) = self.repr_asbx();
                write!(f, "{code:<16}\t{a:<3} {sbx:<3}")
            }
            OpMode::IAx => {
                let (code, ax) = self.repr_ax();
                write!(f, "{code:<16}\t{ax:<3}      ")
            }
            OpMode::IsJ => {
                let (code, sj) = self.repr_sj();
                write!(f, "{code:<16}\t{sj:<3}      ")
            }
        }
    }
}

/// Upval in _ENV table or stack.
pub struct UpValue {
    name: String,             // for debug infomation
    stkidx: Option<RegIndex>, // stack index
    kind: Option<Attribute>,  // close / const ...
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

    source: Rc<String>,   // source file name, used for debug info
    locvars: Vec<LocVar>, // local variable name, used for debug info
    upvals: Vec<UpValue>, // upvalue name, used for debug info
}

impl HeapMemUsed for Proto {
    fn heap_mem_used(&self) -> usize {
        std::mem::size_of::<Self>()
            + self.kst.capacity()
            + self.code.capacity()
            + self.subfn.capacity()
            + self.pcline.capacity()
            + self.locvars.capacity()
            + self.upvals.capacity()
            + self.source.capacity()
    }
}

impl Display for Proto {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        writeln!(
            f,
            "function <{}:{},{}> ({} instructions at 0x{:X})",
            self.source.as_str(),
            self.begline,
            self.endline,
            self.code.len(),
            self as *const Proto as usize
        )?;

        if self.vararg {
            f.write_str("0+ params, ")?;
        } else {
            write!(f, "{} params, ", self.nparam)?;
        }

        writeln!(
            f,
            "{} slots, {} upvalue, {} locals, {}, constants, {} functions",
            self.nreg,
            self.upvals.len(),
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
}

impl Debug for Proto {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        writeln!(f, "{}", self)?;
        let self_addr = self as *const Proto as usize;
        writeln!(f, "{} constants for 0x{:X}", self.kst.len(), self_addr)?;
        for (idx, k) in self.kst.iter().enumerate() {
            writeln!(f, "\t{idx}\t{k}")?;
        }

        writeln!(f, "{} locals for 0x{:X}", self.locvars.len(), self_addr)?;
        for (idx, loc) in self.locvars.iter().enumerate() {
            writeln!(f, "\t{idx}\t\"{}\"", loc.name.as_str())?;
        }

        writeln!(f, "{} upvalues for 0x{:X}", self.upvals.len(), self_addr)?;
        for (idx, up) in self.upvals.iter().enumerate() {
            writeln!(f, "\t{idx}\t\"{}\"", up.name.as_str())?;
        }

        Ok(())
    }
}

impl Proto {
    pub fn delegate_to(p: &mut Proto, heap: &mut Heap) {
        for k in p.kst.iter_mut() {
            heap.delegate(k);
        }

        for sub in p.subfn.iter_mut() {
            heap.delegate_from(*sub);
            Self::delegate_to(sub, heap);
        }
    }

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
}

pub struct LocVar {
    name: String,
    reg: RegIndex, // register index on stack
    start_pc: u32, // first point where variable is active
    end_pc: u32,   // first point where variable is dead
}

#[derive(Clone)]
pub enum ExprStatus {
    LitNil,
    LitTrue,
    LitFalse,
    LitInt(i64),
    LitFlt(f64),
    Call(RegIndex), // function index in local register
    Kst(RegIndex),  // index in constants (int / float / string)
    Reg(RegIndex),  // result has stored in local register
    Up(RegIndex),   // expr refers to a upvalue (index in upvalue list)
}

/// Code generation intermidiate state for each Proto
pub struct GenState {
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
        GenState::new(Rc::new("".to_string()))
    }
}

/// a short name for Instruction
type Isc = Instruction;

impl GenState {
    pub fn new(srcfile: Rc<String>) -> Self {
        Self {
            exprstate: BTreeMap::new(),
            regs: 0,
            ksts: Vec::with_capacity(4),
            upvals: Vec::with_capacity(8),
            subproto: Vec::with_capacity(2),
            code: Vec::with_capacity(32),
            locals: Vec::with_capacity(8),
            srcfile,
            absline: Vec::with_capacity(8),
        }
    }

    fn cur_pc(&self) -> u32 {
        self.code.len() as u32
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

    /// Free last allocated register on vm stack
    fn free_reg(&mut self) {
        self.regs -= 1;
    }

    fn with_free_reg(&mut self, generation: impl FnOnce(RegIndex) -> ExprStatus) -> ExprStatus {
        let idx = self.alloc_free_reg();
        let s = generation(idx);
        self.free_reg();
        s
    }

    // Allocate a constant register and return its index
    fn alloc_const_reg(&mut self, k: LValue) -> i32 {
        let idx = self.ksts.len();
        self.ksts.push(k);
        idx as i32
    }

    /// Find value in constants and return its index
    fn find_const_reg(&self, k: &LValue) -> Option<i32> {
        for (idx, v) in self.ksts.iter().enumerate() {
            if v == k {
                return Some(idx as i32);
            }
        }
        None
    }

    /// Find string in constants and return its index
    fn find_const_str(&self, s: &str) -> Option<RegIndex> {
        for (idx, v) in self.ksts.iter().enumerate() {
            if let LValue::String(sw) = v {
                if sw.as_str() == s {
                    return Some(idx as RegIndex);
                }
            }
        }
        None
    }

    /// Find local variable and return its index
    fn find_local(&self, name: &str) -> Option<RegIndex> {
        for v in self.locals.iter().rev() {
            if v.name == name {
                return Some(v.reg);
            }
        }
        None
    }

    /// Find upvalue and return its index in upvalue list
    fn find_up(&self, name: &str) -> Option<RegIndex> {
        for (idx, v) in self.upvals.iter().enumerate() {
            if v.name == name {
                return Some(idx as RegIndex);
            }
        }
        None
    }

    // /// load global to register
    // fn load_global(&mut self, name: &str, line: u32) -> RegIndex {
    //     let sidx = self.alloc_free_reg();
    //     let cidx = self.find_const_str(name).unwrap(); // must be find
    //     self.gen(Isc::iabc(OpCode::GETTABUP, sidx, 0, cidx), line);
    //     sidx
    // }

    /// Load upval to register.
    fn load_upval(&mut self, upidx: RegIndex) -> RegIndex {
        // let reg = self.alloc_free_reg();
        // let up = self.upvals.get(upidx as usize).unwrap();
        // let b = up.index;
        // let c = self.find_const_str(up.name.as_str()).unwrap(); // must be find
        // self.gen(Isc::iabc(OpCode::GETTABUP, reg, b, c), );
        // self.gen(Isc::iabc(OpCode::GETUPVAL, a, b, c, isk), line);
        // reg
        todo!()
    }

    fn gen_local_desc(&mut self, name: String, status: ExprStatus, lineinfo: (u32, u32)) -> LocVar {
        let mut vreg = self.alloc_free_reg();

        match status {
            ExprStatus::LitNil => self.gen(Isc::iabc(OpCode::LOADNIL, vreg, 0, 0), lineinfo.0),
            ExprStatus::LitTrue => self.gen(Isc::iabc(OpCode::LOADTRUE, vreg, 0, 0), lineinfo.0),
            ExprStatus::LitFalse => self.gen(Isc::iabc(OpCode::LOADFALSE, vreg, 0, 0), lineinfo.0),
            ExprStatus::LitInt(i) => {
                if i.abs() > Isc::MAX_SBX.abs() as i64 {
                    let kreg = self.alloc_const_reg(LValue::Int(i));
                    self.gen(Isc::iabx(OpCode::LOADK, vreg, kreg), lineinfo.0)
                } else {
                    self.gen(Isc::iasbx(OpCode::LOADI, vreg, i as i32), lineinfo.0)
                }
            }
            ExprStatus::LitFlt(f) => {
                // FIX ME:
                // use LOADF for small float.

                let kreg = self.alloc_const_reg(LValue::Float(f));
                self.gen(Isc::iabx(OpCode::LOADK, vreg, kreg), lineinfo.0)
            }
            ExprStatus::Call(reg) => {
                vreg = reg;
                self.free_reg();
            }
            ExprStatus::Kst(reg) => {
                self.gen(Isc::iabc(OpCode::MOVE, vreg, reg, 0), lineinfo.0);
            }
            ExprStatus::Reg(reg) => {
                vreg = reg;
                self.free_reg();
            }
            ExprStatus::Up(ureg) => {
                // self.load_upval(ureg);
                todo!()
            }
        }

        LocVar {
            name,
            reg: vreg,
            start_pc: self.cur_pc(),
            end_pc: self.cur_pc(),
        }
    }

    fn gen_unary_const(&mut self, _op: OpCode, kidx: RegIndex, line: u32) -> ExprStatus {
        let sidx = self.alloc_free_reg();
        self.gen(Isc::iabx(OpCode::LOADK, sidx, kidx), line);
        self.gen(Isc::iabc(OpCode::UNM, sidx, sidx, 0), line);
        ExprStatus::Reg(sidx)
    }

    fn gen_unary_reg(&mut self, _op: OpCode, reg: RegIndex, line: u32) -> ExprStatus {
        let sidx = self.alloc_free_reg();
        self.gen(Isc::iabc(OpCode::UNM, sidx, reg, 0), line);
        ExprStatus::Reg(sidx)
    }
}

#[derive(Debug)]
pub enum CodeGenErr {
    RegisterOverflow,
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
    fn new(strip: bool) -> Self {
        Self {
            strip,
            genstk: LinkedList::new(),
            cgs: GenState::default(),
        }
    }

    fn consume(&mut self, mut ast_root: Block) -> Proto {
        debug_assert_eq!(self.genstk.len(), 0);

        // take chunk name, construct an empty para list
        let name = Rc::new(std::mem::take(&mut ast_root.chunk));
        let plist = ParaList::new(true, Vec::new());

        let mut main = GenState::new(name);
        std::mem::swap(&mut main, &mut self.cgs);
        let res = self.walk_fn_block(plist, ast_root);

        debug_assert_eq!(self.genstk.len(), 0);
        res
    }

    fn walk_fn_block(&mut self, params: ParaList, body: Block) -> Proto {
        // vararg
        if params.vargs {
            let nparam = params.namelist.len();
            self.gen(Isc::iabc(OpCode::VARARGPREP, nparam as i32, 0, 0), 0);
        }

        // set _ENV
        self.upvals.push(UpValue {
            name: "_ENV".to_string(),
            stkidx: None,
            kind: None,
        });

        // statements
        for stmt in body.stats.into_iter() {
            self.walk_stmt(stmt);
        }

        // return statement
        self.walk_return(body.ret);

        // strip debug infomation
        if self.strip {
            let _ = std::mem::replace(&mut self.srcfile, Rc::new("?".to_string()));
            for loc in self.locals.iter_mut() {
                let _ = std::mem::take(&mut loc.name);
            }
            for up in self.upvals.iter_mut() {
                let _ = std::mem::take(&mut up.name);
            }
            self.absline.clear();
        }

        // TODO:
        // Check <close> variable and generate CLOSE instruction
        // Check <const> variable

        let subfns = std::mem::take(&mut self.subproto)
            .into_iter()
            .map(Gc::from)
            .collect();
        Proto {
            vararg: params.vargs,
            nparam: params.namelist.len() as u8,
            nreg: self.regs as u8,
            begline: self.absline.first().copied().unwrap_or(0),
            endline: self.absline.last().copied().unwrap_or(0),
            kst: std::mem::take(&mut self.ksts),
            code: std::mem::take(&mut self.code),
            subfn: subfns,
            source: std::mem::take(&mut self.srcfile),
            pcline: std::mem::take(&mut self.absline),
            locvars: std::mem::take(&mut self.locals),
            upvals: std::mem::take(&mut self.upvals),
        }
    }

    fn walk_stmt(&mut self, stmt: StmtNode) {
        let lineinfo = stmt.lineinfo();
        match stmt.into_inner() {
            Stmt::Assign { vars: _, exprs: _ } => todo!(),
            Stmt::FuncCall(call) => {
                let _ = self.walk_fncall(call, 0);
            }
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
            Stmt::LocalVarDecl { names, exprs } => {
                self.walk_local_decl(names, exprs, lineinfo);
            }
            Stmt::Expr(exp) => {
                let _ = self.walk_common_expr(ExprNode::new(*exp, lineinfo), 0);
            }
        };
    }

    fn walk_local_decl(
        &mut self,
        mut names: Vec<(String, Option<Attribute>)>,
        mut exprs: Vec<crate::ast::WithSrcLoc<Expr>>,
        lineinfo: (u32, u32),
    ) {
        debug_assert!(names.len() >= 1);
        debug_assert!(exprs.len() >= 1);

        let (nvar, nexp) = (names.len(), exprs.len());
        if nvar <= exprs.len() {
            // TODO:
            // add attribute support
            for (idx, (def, _attr)) in names.into_iter().enumerate() {
                let status = exprs
                    .get_mut(idx)
                    .map(|e| self.walk_common_expr(std::mem::take(e), 1))
                    .unwrap_or(ExprStatus::LitNil);
                let local_desc = self.gen_local_desc(def, status, lineinfo);
                self.locals.push(local_desc);
            }

            for extra in exprs.into_iter().skip(nvar) {
                let _ = self.walk_common_expr(extra, 0);
            }
        } else {
            // SAFETY:
            // there are at least 1 expr
            let last = unsafe { exprs.pop().unwrap_unchecked() };

            for (idx, e) in exprs.into_iter().enumerate() {
                let status = self.walk_common_expr(e, 1);
                // TODO:
                // add attribute support
                let desc = unsafe { &mut names.get_mut(idx).unwrap_unchecked() };
                let name = std::mem::take(&mut desc.0);
                let local_desc = self.gen_local_desc(name, status, lineinfo);
                self.locals.push(local_desc);
            }

            let remain = names.iter().skip(nexp).count();
            debug_assert!(remain > 0);
            let last_sta = self.walk_common_expr(last, remain);
            if let ExprStatus::Call(reg) = last_sta {
                for (idx, (name, _attr)) in names.into_iter().skip(nexp).enumerate() {
                    self.gen_local_desc(name, ExprStatus::Reg(reg + idx as RegIndex), lineinfo);
                }
            } else {
                let mut iter = names.into_iter().skip(nexp);
                // SAFETY: there are must at least 1 remain variable
                let next = unsafe { iter.next().unwrap_unchecked() };
                self.gen_local_desc(next.0, last_sta, lineinfo);

                while let Some((name, _attr)) = iter.next() {
                    self.gen_local_desc(name, ExprStatus::LitNil, lineinfo);
                }
            }
        }
    }

    fn walk_return(&mut self, ret: Option<Vec<ExprNode>>) {
        if let Some(mut rets) = ret {
            match rets.len() {
                1 => {
                    // SAFETY: rets.len() == 1
                    let ret_node = unsafe { rets.pop().unwrap_unchecked() };
                    let line = ret_node.lineinfo().0;

                    let status = self.walk_common_expr(ret_node, usize::MAX);
                    let mut gen_lit_template = |op| {
                        let reg = self.alloc_free_reg();
                        self.gen(Isc::iabc(op, reg, 0, 0), line);
                        self.gen(Isc::iabc(OpCode::RETURN1, reg, 0, 0), 0);
                    };

                    match status {
                        ExprStatus::LitNil => gen_lit_template(OpCode::LOADNIL),
                        ExprStatus::LitTrue => gen_lit_template(OpCode::LOADTRUE),
                        ExprStatus::LitFalse => gen_lit_template(OpCode::LOADFALSE),
                        ExprStatus::LitInt(i) => {
                            let reg = self.alloc_free_reg();
                            self.gen(Isc::iasbx(OpCode::LOADI, reg, i as i32), line);
                            self.gen(Isc::iabc(OpCode::RETURN1, reg, 0, 0), line);
                        }
                        ExprStatus::LitFlt(f) => {
                            let reg = self.alloc_free_reg();
                            self.gen(Isc::iabx(OpCode::LOADF, reg, f as i32), line);
                            self.gen(Isc::iabc(OpCode::RETURN1, reg, 0, 0), line);
                        }
                        ExprStatus::Call(_) => todo!("tail call"),
                        ExprStatus::Kst(k) => {
                            let reg = self.alloc_free_reg();
                            self.gen(Isc::iabx(OpCode::LOADK, reg, k), line);
                            self.gen(Isc::iabc(OpCode::RETURN1, reg, 0, 0), line);
                        }
                        ExprStatus::Reg(r) => {
                            self.gen(Isc::iabc(OpCode::RETURN1, r, 0, 0), line);
                        }
                        ExprStatus::Up(upidx) => {
                            // let up = self.upvals.get(upidx as usize).unwrap();
                            // let reg = self.alloc_free_reg();
                            // let c = self.find_const_str(up.name.as_str()).unwrap(); // must be find
                            // if let Some(stkidx) = up.stkidx {
                            //     let b = stkidx;
                            //     todo!("move fn to stk top");
                            //     Isc::iabc(OpCode::CALL, reg, (nparam + 1) as i32, 0)
                            // } else {
                            //     Isc::iabc(OpCode::GETTABUP, reg, 0, c)
                            // }
                            todo!()
                        }
                    }
                }
                n if n > 1 => {
                    // todo: multi return
                    todo!("multi return")
                }
                _ => unreachable!(),
            }
        } else {
            self.gen(Isc::iabc(OpCode::RETURN0, 0, 0, 0), 0);
        }
    }

    fn walk_fncall(&mut self, call: FuncCall, exp_ret: usize) -> ExprStatus {
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
                let lineinfo = prefix.lineinfo();

                let fnreg = match self.walk_common_expr(*prefix, 1) {
                    ExprStatus::Reg(reg) => reg,
                    ExprStatus::Up(upidx) => {
                        let reg = self.alloc_free_reg();
                        let up = self.upvals.get(upidx as usize).unwrap();
                        let c = self.find_const_str(up.name.as_str()).unwrap(); // must be find
                        if let Some(stkidx) = up.stkidx {
                            let b = stkidx;
                            Isc::iabc(OpCode::CALL, reg, (nparam + 1) as i32, 0);
                            todo!("move fn to stk top");
                        } else {
                            self.gen(Isc::iabc(OpCode::GETTABUP, reg, 0, c), lineinfo.0);
                            reg
                        }
                    }
                    _ => unreachable!(),
                };

                for (posi, param) in args.namelist.into_iter().enumerate() {
                    let penode = ExprNode::new(param, (lineinfo.0, lineinfo.1 + (posi + 1) as u32));
                    let expect = if posi == nparam - 1 { usize::MAX } else { 1 };
                    match self.walk_common_expr(penode, expect) {
                        ExprStatus::Kst(k) => {
                            let reg = self.alloc_free_reg();
                            self.gen(Isc::iabx(OpCode::LOADK, reg, k), 0);
                        }
                        ExprStatus::Reg(r) => {
                            let reg = self.alloc_free_reg();
                            self.gen(Isc::iabc(OpCode::MOVE, reg, r, 0), lineinfo.0)
                        }
                        _ => unreachable!(),
                    }
                }

                self.gen(
                    Isc::iabc(OpCode::CALL, fnreg, (nparam + 1) as i32, exp_ret as i32 + 1),
                    lineinfo.0,
                );

                // reserve enuogh space for return value
                if exp_ret > nparam {
                    for _ in nparam..exp_ret {
                        self.alloc_free_reg();
                    }
                }

                ExprStatus::Reg(fnreg)
            }
        }
    }

    fn walk_common_expr(&mut self, node: ExprNode, expect_return: usize) -> ExprStatus {
        let ln = node.lineinfo().0;
        let unique = Self::take_expr_unique(&node);

        if let Some(status) = self.exprstate.get(&unique) {
            status.clone()
        } else {
            let status = match node.into_inner() {
                Expr::Nil => ExprStatus::LitNil,
                Expr::True => ExprStatus::LitTrue,
                Expr::False => ExprStatus::LitFalse,

                Expr::Int(i) => {
                    if i as i32 > Isc::MAX_SBX {
                        let kreg = self.alloc_const_reg(i.into());
                        ExprStatus::Kst(kreg)
                    } else {
                        ExprStatus::LitInt(i)
                    }
                }
                Expr::Float(f) => {
                    // TODO:
                    // Fix float check loggic
                    if f as i32 > Isc::MAX_SBX {
                        let kreg = self.alloc_const_reg(f.into());
                        ExprStatus::Kst(kreg)
                    } else {
                        ExprStatus::LitFlt(f)
                    }
                }
                Expr::Literal(s) => {
                    let kreg = self.find_const_str(&s).unwrap_or_else(|| {
                        let strval = LValue::from(Gc::from(s));
                        self.alloc_const_reg(strval)
                    }); // lazyly
                    ExprStatus::Kst(kreg)
                }
                Expr::Ident(id) => {
                    // local variable ?
                    if let Some(regidx) = self.find_local(&id) {
                        return ExprStatus::Reg(regidx);
                    }

                    // upvalue occurred ?
                    if let Some(upidx) = self.find_up(&id) {
                        return ExprStatus::Up(upidx);
                    }

                    // find in hestory stack
                    for frame in self.genstk.iter().rev() {
                        if let Some(regidx) = frame.find_local(&id) {
                            self.upvals.push(UpValue {
                                name: id,
                                stkidx: Some(regidx),
                                kind: None,
                            });
                            return ExprStatus::Up(self.upvals.len() as i32 - 1);
                        }
                    }

                    // if not found, acquire _ENV table, record upvalue's name
                    // TODO:
                    // support for variable with attribute
                    self.upvals.push(UpValue {
                        name: id.clone(),
                        stkidx: None,
                        kind: None,
                    });
                    self.alloc_const_reg(LValue::from_wild(Gc::from(id)));
                    ExprStatus::Up(self.upvals.len() as i32 - 1)
                }

                Expr::UnaryOp { op, expr } => {
                    let es = self.walk_common_expr(*expr, 1);
                    let unop_code = match op {
                        UnOp::Minus => OpCode::UNM,
                        UnOp::Not => OpCode::NOT,
                        UnOp::Length => OpCode::LEN,
                        UnOp::BitNot => OpCode::BNOT,
                    };

                    match es {
                        ExprStatus::Kst(kidx) => self.gen_unary_const(unop_code, kidx, ln),
                        ExprStatus::Reg(reg) => self.gen_unary_reg(unop_code, reg, ln),
                        _ => unreachable!(),
                    }
                }

                Expr::BinaryOp { lhs, op, rhs } => {
                    let (lst, rst) = (
                        self.walk_common_expr(*lhs, 1),
                        self.walk_common_expr(*rhs, 1),
                    );
                    let destreg = self.alloc_free_reg();

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
                        (ExprStatus::Kst(lk), ExprStatus::Kst(rk)) => {
                            // load left to dest reg and cover dest reg
                            self.gen(Isc::iabx(OpCode::LOADK, destreg, lk), ln);
                            self.gen(
                                Isc::iabc(select_arithmetic_kop(op), destreg, destreg, rk),
                                ln,
                            );
                        }
                        (ExprStatus::Kst(lk), ExprStatus::Reg(rk)) => {
                            self.gen(Isc::iabc(select_arithmetic_kop(op), destreg, rk, lk), ln);
                        }
                        (ExprStatus::Reg(lk), ExprStatus::Kst(rk)) => {
                            self.gen(Isc::iabc(select_arithmetic_kop(op), destreg, lk, rk), ln);
                        }
                        (ExprStatus::Reg(lk), ExprStatus::Reg(rk)) => {
                            self.gen(Isc::iabc(select_arithemic_op(op), destreg, lk, rk), ln);
                        }
                        _ => unreachable!(),
                    };
                    ExprStatus::Reg(destreg)
                }

                Expr::FuncDefine(def) => {
                    let pto = self.walk_fn_block(def.params, *def.body);
                    let pidx = self.subproto.len();
                    self.subproto.push(pto);
                    let reg = self.alloc_free_reg();
                    self.gen(Isc::iabx(OpCode::CLOSURE, reg, pidx as i32), ln);
                    ExprStatus::Reg(reg)
                }

                Expr::Index { prefix: _, key: _ } => todo!("expr codegen : Table Index"),

                Expr::FuncCall(call) => self.walk_fncall(call, expect_return),

                Expr::TableCtor(flist) => {
                    // idx of this table
                    let tbidx = self.alloc_free_reg();
                    self.gen(Isc::iabc(OpCode::NEWTABLE, tbidx, 0, 0), ln);

                    let aryidx = 1; // array in lua start at index 1
                    for field in flist.into_iter() {
                        let valstatus = self.walk_common_expr(*field.val, 1);

                        let valreg = match valstatus {
                            ExprStatus::Kst(k) => {
                                let free = self.alloc_free_reg();
                                self.gen(Isc::iabx(OpCode::LOADK, free, k), ln);
                                free
                            }

                            ExprStatus::Reg(r) => r,
                            _ => unreachable!(),
                        };

                        if let Some(key) = field.key {
                            let keystatus = self.walk_common_expr(*key, 1);
                            match keystatus {
                                ExprStatus::Kst(kidx) => {
                                    let free = self.alloc_free_reg();
                                    self.gen(Isc::iabx(OpCode::LOADK, free, kidx), ln);
                                    self.gen(Isc::iabc(OpCode::SETTABLE, tbidx, free, valreg), ln)
                                }
                                ExprStatus::Reg(reg) => {
                                    self.gen(Isc::iabc(OpCode::SETTABLE, tbidx, reg, valreg), ln)
                                }
                                _ => unreachable!(), // constant fold
                            }
                        } else {
                            let kidx = self.alloc_const_reg(LValue::Int(aryidx));
                            let free = self.alloc_free_reg();
                            self.gen(Isc::iabx(OpCode::LOADK, free, kidx), ln);
                            self.gen(Isc::iabc(OpCode::SETTABLE, tbidx, free, valreg), ln)
                        }
                    }
                    ExprStatus::Reg(tbidx)
                }
                Expr::Dots => unreachable!(),
            };
            self.exprstate.insert(unique, status.clone());
            status
        }
    }

    /// use (line, col) as unique id
    fn take_expr_unique(node: &ExprNode) -> usize {
        let line = node.lineinfo().0 as usize;
        line << (16 + node.lineinfo().1 as usize)
    }
}

impl CodeGen {
    pub fn generate(ast_root: Block, strip: bool) -> Result<Proto, CodeGenErr> {
        let mut gen = Self::new(strip);
        Ok(gen.consume(ast_root))
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
    const LUAC_FORMAT: u8 = 0; // standard luac format
    const LUAC_INT: u64 = 0x5678; // from luac 5.4
    const LUAC_FLOAT: f64 = 370.5; // from luac 5.4

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
        Self::dump_varint(chunk.upvals.len(), bw)?;
        Self::dump_string(&chunk.source, bw)?;
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

        Self::dump_varint(chunk.upvals.len(), bw)?;
        for up in chunk.upvals.iter() {
            let (onstk, stkid) = if let Some(idx) = up.stkidx {
                (true, idx)
            } else {
                (false, 0)
            };
            let attr: u8 = if let Some(a) = up.kind { a as u8 } else { 0 };
            Self::dump_varint(onstk as usize, bw)?;
            Self::dump_varint(stkid as usize, bw)?;
            Self::dump_varint(attr as usize, bw)?;
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

        Self::dump_varint(chunk.upvals.len(), bw)?;
        for up in chunk.upvals.iter() {
            Self::dump_string(&up.name, bw)?;
        }

        Ok(())
    }

    pub fn undump_proto(r: &mut BufReader<impl Read>) -> Result<Proto, BinLoadErr> {
        let nupval = Self::undump_varint(r)?;
        let src = Rc::new(Self::undump_string(r)?);
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
            let onstk = u8::from_ne_bytes(byte) != 0;

            r.read_exact(&mut byte)?;
            let stkid = u8::from_ne_bytes(byte);

            r.read_exact(&mut byte)?;
            // TODO:
            // support attribute dump
            // let attr = ...

            ups.push(UpValue {
                name: String::new(),
                stkidx: if onstk { Some(stkid as RegIndex) } else { None },
                kind: None,
            })
        }

        let proto_size = Self::undump_varint(r)?;
        let mut subfn = Vec::with_capacity(proto_size);
        for _ in 0..proto_size {
            let mut p = Self::undump_proto(r)?;
            p.source = src.clone();
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
            up.name = Self::undump_string(r)?;
        }

        Ok(Proto {
            vararg: is_vararg,
            nparam: nparam,
            nreg: nreg,
            begline: begline as u32,
            endline: endline as u32,
            kst: kst,
            code: code,
            subfn: subfn,
            pcline: Vec::new(),
            source: src,
            locvars: Vec::new(),
            upvals: ups,
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
            0x04 | 0x14 => LValue::from_wild(Self::undump_string(r)?.into()),
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
            Ok(String::with_capacity(StrImpl::EXTRA_HEADERS_SIZE))
        } else {
            let mut buf = String::with_capacity(len + StrImpl::EXTRA_HEADERS_SIZE);
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
        use super::Isc;
        use crate::codegen::OpMode;

        for signed in [0, 1, 123, 999, -1, -999] {
            let i = Isc::iasbx(super::OpCode::LOADI, 0, signed);
            assert_eq!(i.mode(), OpMode::IAsBx);
            let (_, a, sbx) = i.repr_asbx();
            assert_eq!(a, 0);
            assert_eq!(sbx, signed);
        }
    }

    #[test]
    fn instruction_size_check() {
        use super::Instruction;
        assert_eq!(std::mem::size_of::<Instruction>(), 4);
    }

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
        let origin = super::CodeGen::generate(ast, false).unwrap();
        // println!("{:?}", origin);

        let tmp_file = std::fs::File::create(tmpfile.clone()).unwrap();
        let mut writer = BufWriter::new(tmp_file);
        ChunkDumper::dump(&origin, &mut writer).unwrap();
        writer.flush().unwrap();

        let same_file = std::fs::File::open(tmpfile.clone()).unwrap();
        let mut reader = BufReader::new(same_file);
        let recover = ChunkDumper::undump(&mut reader).unwrap();
        // println!("{:?}", recover);

        let mut vm = State::new();
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
