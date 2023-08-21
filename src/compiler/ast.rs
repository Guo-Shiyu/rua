use std::{
    io::{BufWriter, Error, Write},
    ops::{Deref, DerefMut},
};

/// Warpper for an ast-node to attach source location info
pub struct WithSrcLoc<T> {
    node: T,
    // lineinfo: (begin, end)
    lineinfo: (u32, u32),
}

impl<T> WithSrcLoc<T> {
    pub fn new(n: T, lines: (u32, u32)) -> Self {
        WithSrcLoc {
            lineinfo: lines,
            node: n,
        }
    }

    pub fn into_inner(self) -> T {
        self.node
    }

    pub fn inner(&self) -> &T {
        &self.node
    }

    pub fn lineinfo(&self) -> (u32, u32) {
        self.lineinfo
    }

    pub fn mem_address(&self) -> usize {
        self as *const Self as usize
    }
}

impl<T> Deref for WithSrcLoc<T> {
    type Target = T;
    fn deref(&self) -> &Self::Target {
        &self.node
    }
}

impl<T> DerefMut for WithSrcLoc<T> {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.node
    }
}

/// block ::= {stat} [retstat]
pub struct Block {
    pub chunk: String,
    pub stats: Vec<StmtNode>,
    pub ret: Option<Vec<ExprNode>>,
}

impl Block {
    pub const NAMELESS_CHUNK: &'static str = "<anonymous>";

    /// Create an anonymous block.
    pub fn new(stats: Vec<StmtNode>, ret: Option<Vec<ExprNode>>) -> Self {
        Block {
            chunk: Self::NAMELESS_CHUNK.to_string(),
            stats,
            ret,
        }
    }

    /// Create block with a chunk name.
    pub fn chunk(chunk: String, stats: Vec<StmtNode>, ret: Option<Vec<ExprNode>>) -> Self {
        Block { chunk, stats, ret }
    }
}

/// stat ::=  `;` |
///        varlist `=` explist |
///        functioncall |
///        label |
///        break |
///        goto Name |
///        do block end |
///        while exp do block end |
///        repeat block until exp |
///        if exp then block {elseif exp then block} [else block] end |
///        for Name `=` exp `,` exp [`,` exp] do block end |
///        for namelist in explist do block end |
///        function funcname funcbody |
///        local function Name funcbody |
///        local attnamelist [`=` explist]

pub enum Stmt {
    // global assignment
    Assign {
        vars: Vec<ExprNode>,
        exprs: Vec<ExprNode>,
    },

    FuncCall(FuncCall),

    Lable(String),

    Goto(String),

    Break,

    DoEnd(Box<Block>),

    While {
        exp: Box<ExprNode>,
        block: Box<Block>,
    },

    Repeat {
        block: Box<Block>,
        exp: Box<ExprNode>,
    },

    IfElse {
        exp: Box<ExprNode>,
        then: Box<Block>,
        els: Box<Block>,
    },

    NumericFor {
        iter: String,
        init: Box<Expr>,
        limit: Box<Expr>,
        step: Box<Expr>,
        body: Box<Block>,
    },

    GenericFor {
        iters: Vec<String>,
        exprs: Vec<Expr>,
        body: Box<Block>,
    },

    FnDef {
        pres: Vec<String>,
        method: Option<String>,
        body: Box<FuncBody>,
    },

    LocalVarDecl {
        names: Vec<(String, Option<Attribute>)>,
        exprs: Vec<ExprNode>,
    },

    Expr(Box<Expr>),
}

/// functioncall ::=  prefixexp args | prefixexp `:` Name args
///
/// funcname ::= Name {`.` Name} [`:` Name]
///
/// prefixexp ::= var | functioncall | `(` exp `)`
///
/// args ::=  `(` [explist] `)` | tablector | LiteralString
pub enum FuncCall {
    // i.e: func(1, 2, 3, ...)
    FreeFnCall {
        prefix: Box<ExprNode>,
        args: ParaList,
    },

    // i.e: class:func(1, 2, 3, ...)
    MethodCall {
        prefix: Box<ExprNode>,
        method: String,
        args: ParaList,
    },
}

/// functiondef ::= function funcbody
/// funcbody ::= `(` [parlist] `)` block end
/// paralist ::= namelist [`,` `...`] | `...`
pub struct FuncBody {
    params: ParaList,
    body: Box<Block>,
}

impl FuncBody {
    pub fn new(params: ParaList, body: Box<Block>) -> Self {
        FuncBody { params, body }
    }
}

/// exp ::=  nil | false | true | Numeral | LiteralString | `...` |
///     functiondef | prefixexp | tablector |
///     exp binop exp | unop exp
///
/// prefixexp ::= var | functioncall | `(` exp `)`
pub enum Expr {
    Nil,
    False,
    True,
    Int(i64),
    Float(f64),
    Literal(String),
    Dots,

    Ident(String),

    // global function defination
    FuncDefine(FuncBody),

    // table.key | table[key]
    Index {
        prefix: Box<ExprNode>,
        key: Box<Expr>,
    },

    // functioncall ::=  prefixexp args | prefixexp `:` Name args
    FuncCall(FuncCall),

    // fieldlist ::= field {fieldsep field} [fieldsep]
    TableCtor(Vec<Field>),

    BinaryOp {
        lhs: Box<ExprNode>,
        op: BinOp,
        rhs: Box<ExprNode>,
    },

    UnaryOp {
        op: UnOp,
        expr: Box<ExprNode>,
    },
}

#[derive(Debug)]
pub enum Attribute {
    Const,
    Close,
}

pub struct ParaList {
    vargs: bool,
    namelist: Vec<Expr>,
}

impl ParaList {
    pub fn new(vargs: bool, namelist: Vec<Expr>) -> Self {
        ParaList { vargs, namelist }
    }
}

/// field ::= `[` exp `]` `=` exp | Name `=` exp | exp
pub struct Field {
    key: Option<Box<ExprNode>>,
    val: Box<ExprNode>,
}

impl Field {
    pub fn new(key: Option<Box<ExprNode>>, val: Box<ExprNode>) -> Self {
        Field { key, val }
    }
}

#[rustfmt::skip]
#[derive(Clone, Copy, PartialEq)]
pub enum BinOp {
    /* arithmetic operators */
    Add, Minus, Mul, Mod, Pow, Div, IDiv,

    /* bitwise operators */   
    BitAnd, BitOr, BitXor, Shl, Shr,   
    
    /* string operator */
    Concat,

    /* comparison operators */
    Eq, Less, LE, Neq, Great, GE, 

    /* logical operators */
    And, Or,
}

/// unop ::= `-` | not | `#` | `~`
pub enum UnOp {
    Minus,
    Not,
    Length,
    BitNot,
}

pub type ExprNode = WithSrcLoc<Expr>;
pub type StmtNode = WithSrcLoc<Stmt>;

pub trait PassRunStatus {
    fn is_ok(&self) -> bool;
    fn is_err(&self) -> bool;
}

/// Take run result from a pass impl
pub trait PassRunRes<Output = (), Error = ()>: PassRunStatus {
    fn output(&self) -> Option<&Output>;
    fn take_output(&mut self) -> Output;
    fn error(&self) -> Option<&Error>;
}

pub trait AnalysisPass {
    fn walk(&mut self, root: &Block) -> &Self;
}

pub trait TransformPass {
    fn walk(&mut self, root: &mut Block) -> &Self;
}

pub enum DumpPrecison {
    Statement,
    Expression,
}

/// Dump state struct, record ident depth and dump level
pub struct AstDumper {
    depth: usize,        // ident state
    colored: bool,       // whether dump with color
    level: DumpPrecison, // dump level
    dump_buf: Vec<u8>,   // dumped string buffer
    errinfo: Option<AstDumpErr>,
}

pub struct PassHasNotRun();

enum AstDumpErr {
    IOErr(std::io::Error),
    PassHasNotRun,
}

impl PassRunStatus for AstDumper {
    fn is_ok(&self) -> bool {
        self.errinfo.is_none() && self.dump_buf.len() > 0
    }

    fn is_err(&self) -> bool {
        !self.is_ok()
    }
}

impl PassRunRes<Vec<u8>, AstDumpErr> for AstDumper {
    fn output(&self) -> Option<&Vec<u8>> {
        if self.is_ok() {
            Some(&self.dump_buf)
        } else {
            None
        }
    }

    fn take_output(&mut self) -> Vec<u8> {
        std::mem::take(&mut self.dump_buf)
    }

    fn error(&self) -> Option<&AstDumpErr> {
        if self.errinfo.is_some() {
            self.errinfo.as_ref()
        } else if self.dump_buf.len() == 0 {
            Some(&AstDumpErr::PassHasNotRun)
        } else {
            None
        }
    }
}

impl AnalysisPass for AstDumper {
    fn walk(&mut self, block: &Block) -> &Self {
        const BUF_SIZE: usize = 8192; // equal with std::io::DEFAULT_BUF_SIZE
        if self.dump_buf.len() < BUF_SIZE {
            self.dump_buf.reserve(BUF_SIZE);
        }

        let inner_buf = std::mem::take(&mut self.dump_buf);
        let mut bw = BufWriter::new(inner_buf);

        let _ = self
            .dump_block(block, &mut bw)
            .map(|_| std::mem::swap(&mut bw.into_inner().unwrap(), &mut self.dump_buf))
            .map_err(|e| self.errinfo = Some(AstDumpErr::IOErr(e)));

        self
    }
}

impl AstDumper {
    pub fn new(level: DumpPrecison, colored: bool, buf: Option<Vec<u8>>) -> Self {
        AstDumper {
            depth: 0,
            colored,
            level,
            dump_buf: buf.unwrap_or_default(),
            errinfo: None,
        }
    }

    pub fn dump(
        block: &Block,
        precision: DumpPrecison,
        buf: &mut BufWriter<impl Write>,
        colored: bool,
    ) -> Result<(), Error> {
        let mut dumper = AstDumper::new(precision, colored, None);
        dumper.dump_block(block, buf)
    }
}

const EXPAND: char = '+';
const COLLAPSE: char = '-';

const NORMAL: &str = "\\033[32m";
const GREEN: &str = "\\033[32m";
const YELLOW: &str = "\\033[33m";
const RED: &str = "\\033[33m";
const BLUE: &str = "\\033[34m";
const GREY: &str = "\\033[30m";

impl AstDumper {
    fn dump_block(&mut self, bk: &Block, buf: &mut BufWriter<impl Write>) -> Result<(), Error> {
        self.write_lable(buf, EXPAND)?;
        self.inc_indent();
        self.write_name(buf, "Block", Self::mem_address(bk))?;

        self.color(buf, BLUE)?;
        write!(
            buf,
            "statement: [{}], with-return: {:?}, chunk-name: \"{}\"",
            bk.stats.len(),
            bk.ret.is_some(),
            bk.chunk.as_str(),
        )?;

        let line = bk.stats.first().map_or(0, |node| node.lineinfo.0);
        self.write_lineinfo(buf, line)?;

        for stmt in &bk.stats {
            self.dump_stmt(stmt, buf)?;
        }

        self.dec_indent();
        Ok(())
    }

    fn dump_stmt(&mut self, stmt: &StmtNode, buf: &mut BufWriter<impl Write>) -> Result<(), Error> {
        let this = stmt.mem_address();
        match stmt.inner() {
            Stmt::Assign { vars, exprs } => self.write_lable(buf, COLLAPSE).and_then(|_| {
                self.write_name(buf, "Assignment", this)?;
                if vars.len() == 1 {
                    write!(
                        buf,
                        "decl: {}, expr-count: [{}]",
                        Self::inspect(vars[0].inner()),
                        exprs.len(),
                    )?;
                } else {
                    let names = vars
                        .iter()
                        .map(|e| Self::inspect(e.inner()))
                        .collect::<Vec<_>>();

                    write!(buf, "decls: {:?}, expr-count: [{}]", names, exprs.len(),)?;
                }
                self.write_lineinfo(buf, stmt.lineinfo.0)
            }),

            Stmt::Lable(name) => self.write_lable(buf, COLLAPSE).and_then(|_| {
                self.write_name(buf, "Lable", this)?;
                write!(buf, "name: {}", name,)?;
                self.write_lineinfo(buf, stmt.lineinfo.0)
            }),

            Stmt::Goto(lable) => self.write_lable(buf, COLLAPSE).and_then(|_| {
                self.write_name(buf, "Goto", this)?;
                write!(buf, "lable: {}", lable,)?;
                self.write_lineinfo(buf, stmt.lineinfo.0)
            }),

            Stmt::Break => self.write_lable(buf, COLLAPSE).and_then(|_| {
                self.write_name(buf, "Break", this)?;
                self.write_lineinfo(buf, stmt.lineinfo.0)
            }),

            Stmt::LocalVarDecl { names, exprs } => self.write_lable(buf, COLLAPSE).and_then(|_| {
                let var_names = names
                    .iter()
                    .map(|(name, _)| name.as_str())
                    .collect::<Vec<_>>();
                write!(
                    buf,
                    "LocalVarDecl: 0x{:x}, var: [{}], expr: [{}], decls: {:?}",
                    this,
                    names.len(),
                    exprs.len(),
                    var_names
                )?;
                self.write_lineinfo(buf, stmt.lineinfo.0)
            }),

            Stmt::Expr(exp) => self.write_lable(buf, COLLAPSE).and_then(|_| {
                self.write_name(buf, "Expression", this)?;

                self.color(buf, RED)?;
                if let DumpPrecison::Expression = self.level {
                    // TODO:
                    // dump expr with detail precission
                    write!(buf, "expr: {}", Self::inspect(exp))?;
                } else {
                    write!(buf, "exp: {}", Self::inspect(exp))?;
                }

                self.write_lineinfo(buf, stmt.lineinfo.0)
            }),

            Stmt::FuncCall(call) => match call {
                FuncCall::FreeFnCall { prefix, args } => {
                    self.write_lable(buf, COLLAPSE).and_then(|_| {
                        self.write_name(buf, "FreeFuncCall", this)?;
                        write!(
                            buf,
                            "prefix: {}, vararg: {}, positional-args-num: {}",
                            Self::inspect(prefix.inner()),
                            args.vargs,
                            args.namelist.len(),
                        )?;
                        self.write_lineinfo(buf, stmt.lineinfo.0)
                    })
                }

                FuncCall::MethodCall {
                    prefix,
                    method,
                    args,
                } => self.write_lable(buf, COLLAPSE).and_then(|_| {
                    self.write_name(buf, "MethodCall", this)?;
                    write!(
                        buf,
                        "prefix: {}, method: {}, vararg:{}, positional-args-num: {:x}",
                        Self::inspect(prefix.inner()),
                        method,
                        args.vargs,
                        args.namelist.len(),
                    )?;
                    self.write_lineinfo(buf, stmt.lineinfo.0)
                }),
            },

            Stmt::DoEnd(block) => self.write_lable(buf, '+').and_then(|_| {
                self.write_name(buf, "DoEnd", this)?;
                write!(buf, "statements: [{}]", block.stats.len(),)?;
                self.write_lineinfo(buf, stmt.lineinfo.0)?;

                self.inc_indent();
                self.dump_block(block, buf)?;
                self.dec_indent();
                Ok(())
            }),

            Stmt::While { exp, block } => self.write_lable(buf, '+').and_then(|_| {
                self.write_name(buf, "WhileDoEnd", this)?;
                write!(
                    buf,
                    "exp: 0x{:x}, statements: [{}] ",
                    exp.mem_address(),
                    block.stats.len(),
                )?;
                self.write_lineinfo(buf, stmt.lineinfo.0)?;

                self.inc_indent();
                self.dump_block(block, buf)?;
                self.dec_indent();
                Ok(())
            }),

            Stmt::Repeat { block, exp } => {
                self.write_lable(buf, '+').and_then(|_| {
                    self.write_name(buf, "RepeatUntil", this)?;
                    write!(
                        buf,
                        " exp: 0x{:x}, statements: [{}]",
                        exp.mem_address(),
                        block.stats.len(),
                    )
                })?;
                self.write_lineinfo(buf, stmt.lineinfo.0)?;

                self.inc_indent();
                self.dump_block(block, buf)?;
                self.dec_indent();
                Ok(())
            }

            Stmt::IfElse { exp, then, els } => {
                self.write_lable(buf, '+').and_then(|_| {
                    self.write_name(buf, "IfElseEnd", this)?;
                    write!(
                        buf,
                        "exp: {}, then: 0x{:x}, else: 0x{:x}",
                        Self::inspect(exp.inner()),
                        Self::mem_address(then),
                        Self::mem_address(els),
                    )
                })?;
                self.write_lineinfo(buf, stmt.lineinfo.0)?;

                // TODO: if-else-then block
                if !then.stats.is_empty() {
                    self.inc_indent();
                    self.dump_block(then, buf)?;
                    self.dec_indent();
                }

                if !els.stats.is_empty() {
                    self.inc_indent();
                    self.dump_block(els, buf)?;
                    self.dec_indent();
                }

                Ok(())
            }

            Stmt::NumericFor {
                iter: name,
                init,
                limit,
                step,
                body,
            } => {
                self.write_lable(buf, '+')?;
                self.write_name(buf, "NumericFor", this)?;
                write!(
                    buf,
                    "name: {}, init: {}, limit: {}, step: {}",
                    name,
                    Self::inspect(init),
                    Self::inspect(limit),
                    Self::inspect(step),
                )?;

                self.write_lineinfo(buf, stmt.lineinfo.0)?;

                self.inc_indent();
                self.dump_block(body, buf)?;
                self.dec_indent();
                Ok(())
            }
            Stmt::GenericFor {
                iters: names,
                exprs,
                body,
            } => {
                self.write_lable(buf, '+')?;
                self.write_name(buf, "GenericFor", this)?;
                write!(buf, "names: {:?}, exprs-num: {}", names, exprs.len(),)?;
                self.write_lineinfo(buf, stmt.lineinfo.0)?;

                self.inc_indent();
                self.dump_block(body, buf)?;
                self.dec_indent();
                Ok(())
            }
            Stmt::FnDef {
                pres,
                method,
                body: def,
            } => {
                self.write_lable(buf, '+')?;
                self.write_name(buf, "FuncDef", this)?;

                let prefixs = pres.iter().map(|name| name.as_str()).collect::<Vec<_>>();
                write!(
                    buf,
                    "prefix:{:?}, method: {:?}, def: 0x{:x}",
                    prefixs,
                    method,
                    Self::mem_address(def),
                )?;
                self.write_lineinfo(buf, stmt.lineinfo.0)?;

                self.inc_indent();
                self.dump_block(&def.body, buf)?;
                self.dec_indent();
                Ok(())
            }
        }
    }

    fn write_indent(&mut self, buf: &mut BufWriter<impl Write>) -> Result<&mut Self, Error> {
        for _ in 0..self.depth {
            write!(buf, "  ")?;
        }
        Ok(self)
    }

    fn write_lable(&mut self, buf: &mut BufWriter<impl Write>, lable: char) -> Result<(), Error> {
        self.color(buf, NORMAL)?;
        self.write_indent(buf)
            .and_then(|_| write!(buf, "{} ", lable))
    }

    fn write_name(
        &mut self,
        buf: &mut BufWriter<impl Write>,
        name: &str,
        addr: usize,
    ) -> Result<(), Error> {
        self.color(buf, GREEN)?;
        write!(buf, "{}:", name)?;
        self.color(buf, YELLOW)?;
        write!(buf, " 0x{:x} ", addr)?;
        self.color(buf, NORMAL)
    }

    fn write_lineinfo(&mut self, buf: &mut BufWriter<impl Write>, line: u32) -> Result<(), Error> {
        self.color(buf, GREY)?;
        writeln!(buf, "  <line: {}>", line)
    }

    fn inc_indent(&mut self) {
        self.depth += 1;
    }

    fn dec_indent(&mut self) {
        self.depth -= 1;
    }

    fn color(&self, buf: &mut BufWriter<impl Write>, color_ctrl: &str) -> Result<(), Error> {
        if self.colored {
            write!(buf, "{}", color_ctrl)
        } else {
            Ok(())
        }
    }
}

impl AstDumper {
    /// Get memory address of an AST node
    fn mem_address<T>(t: &T) -> usize {
        t as *const T as usize
    }

    /// Inspect an expression, return string representation of it.
    fn inspect(exp: &Expr) -> String {
        match exp {
            Expr::Nil => "nil".to_string(),
            Expr::False => "false".to_string(),
            Expr::True => "true".to_string(),
            Expr::Int(i) => i.to_string(),
            Expr::Float(f) => f.to_string(),
            Expr::Literal(l) => l.clone(),
            Expr::Dots => "<VARG>".to_string(),
            Expr::Ident(id) => id.clone(),
            Expr::FuncDefine(_) => "<FuncDef>".to_string(),
            Expr::Index { prefix, key } => {
                let mut buf = String::with_capacity(32);
                buf.push_str(&Self::inspect(prefix.inner()));
                buf.push_str(&Self::inspect(key));
                buf
            }
            Expr::FuncCall(_) => "<FuncCall>".to_string(),
            Expr::TableCtor(_) => "<TableConstruct>".to_string(),
            Expr::BinaryOp {
                lhs: _,
                op: _,
                rhs: _,
            } => "<BinaryOp>".to_string(),
            Expr::UnaryOp { op: _, expr: _ } => "<UnaryOp>".to_string(),
        }
    }
}

enum AfterFoldStatus {
    StillConst,
    NonConst,
}

#[cfg(flag = "trace_optimize")]
enum FoldOperation {
    BinaryOp { op: BinOp },
    UnaryOp { op: UnOp },
}

pub struct FoldInfo {
    #[cfg(flag = "trace_optimize")]
    srcloc: (u32, u32), // source location

    #[cfg(flag = "trace_optimize")]
    derive_n: usize, //

    #[cfg(flag = "trace_optimize")]
    status: AfterFoldStatus, //

    #[cfg(flag = "trace_optimize")]
    // op: FoldOperation,       //
    new: Expr, // updated node (must be a constant)
}

pub struct ConstantFoldPass {
    #[cfg(flag = "trace_optimize")]
    record: Vec<FoldInfo>,
}

impl PassRunStatus for ConstantFoldPass {
    fn is_ok(&self) -> bool {
        true
    }

    fn is_err(&self) -> bool {
        false
    }
}

impl PassRunRes<Vec<FoldInfo>, PassHasNotRun> for ConstantFoldPass {
    #[cfg(flag = "trace_optimize")]
    fn output(&self) -> Option<&Vec<FoldInfo>> {
        self.record.as_ref()
    }

    fn output(&self) -> Option<&Vec<FoldInfo>> {
        None
    }

    #[cfg(flag = "trace_optimize")]
    fn take_output(&mut self) -> Vec<FoldInfo> {
        std::mem::take(&mut self.record)
    }

    fn take_output(&mut self) -> Vec<FoldInfo> {
        Vec::new()
    }

    fn error(&self) -> Option<&PassHasNotRun> {
        None
    }
}

impl TransformPass for ConstantFoldPass {
    fn walk(&mut self, root: &mut Block) -> &Self {
        self.run_on_bolck(root);
        self
    }
}

impl ConstantFoldPass {
    pub fn new() -> Self {
        ConstantFoldPass {
            #[cfg(flag = "trace_optimize")]
            record: Vec::new(),
        }
    }

    fn run_on_bolck(&mut self, root: &mut Block) {
        for stmt in root.stats.iter_mut() {
            match &mut stmt.node {
                Stmt::Assign { vars: _, exprs } => {
                    exprs.iter_mut().for_each(|e| {
                        self.try_fold(e);
                    });
                }
                Stmt::FuncCall(call) => match call {
                    FuncCall::FreeFnCall { prefix: _, args } => {
                        args.namelist.iter_mut().for_each(|e| {
                            self.try_fold(e);
                        });
                    }
                    FuncCall::MethodCall {
                        prefix: _,
                        method: _,
                        args,
                    } => {
                        args.namelist.iter_mut().for_each(|e| {
                            self.try_fold(e);
                        });
                    }
                },
                Stmt::DoEnd(block) => self.run_on_bolck(block),
                Stmt::While { exp: _, block } => self.run_on_bolck(block),
                Stmt::Repeat { block, exp: _ } => self.run_on_bolck(block),
                Stmt::IfElse { exp: _, then, els } => {
                    self.run_on_bolck(then);
                    self.run_on_bolck(els);
                }
                Stmt::NumericFor {
                    iter: _,
                    init: _,
                    limit: _,
                    step: _,
                    body,
                } => self.run_on_bolck(body),
                Stmt::GenericFor {
                    iters: _,
                    exprs: _,
                    body,
                } => self.run_on_bolck(body),
                Stmt::FnDef {
                    pres: _,
                    method: _,
                    body: func,
                } => {
                    func.params.namelist.iter_mut().for_each(|e| {
                        self.try_fold(e);
                    });
                    self.run_on_bolck(&mut func.body);
                }
                Stmt::LocalVarDecl { names: _, exprs } => {
                    exprs.iter_mut().for_each(|e| {
                        self.try_fold(e);
                    });
                }
                Stmt::Expr(e) => {
                    self.try_fold(e);
                }
                _ => {}
            }
        }
    }

    fn try_fold(&mut self, exp: &mut Expr) -> AfterFoldStatus {
        use AfterFoldStatus::*;
        match exp {
            Expr::Nil
            | Expr::False
            | Expr::True
            | Expr::Int(_)
            | Expr::Float(_)
            | Expr::Literal(_) => StillConst,

            Expr::BinaryOp { lhs, op, rhs } => {
                let ls = self.try_fold(lhs);
                let rs = self.try_fold(rhs);
                match (ls, rs) {
                    (StillConst, StillConst) => {
                        let (mut i1, mut i2) = (0, 0);
                        // intergral promotion
                        if let Some(promoted) = match (lhs.inner(), rhs.inner()) {
                            (Expr::Int(l), Expr::Int(r)) => {
                                i1 = *l;
                                i2 = *r;
                                None
                            }
                            (Expr::Int(to), Expr::Float(f)) => Some((*to as f64, *f)),
                            (Expr::Float(f), Expr::Int(to)) => Some((*f, *to as f64)),
                            (Expr::Float(f1), Expr::Float(f2)) => Some((*f1, *f2)),
                            _ => None,
                        } {
                            if let Some(fop) = Self::gen_arithmetic_op_float(*op) {
                                *exp = Self::apply_arithmetic_op_float(promoted.0, promoted.1, fop);

                                // TODO:
                                // record fold op
                                // #[cfg(flag = "trace_optimize")]
                                // self.record.push(FoldInfo {
                                //     srcloc: expr.lineinfo(),
                                //     derive_n: *derive,
                                //     status: StillConst,
                                //     new: new_exp,
                                // });

                                StillConst
                            } else {
                                NonConst
                            }
                        } else if let Some(iop) = Self::gen_arithmetic_op_int(*op) {
                            if i2 == 0 && (*op == BinOp::Div || *op == BinOp::IDiv) {
                                *exp = Expr::Float(match i1.cmp(&0) {
                                    std::cmp::Ordering::Less => f64::NEG_INFINITY,
                                    std::cmp::Ordering::Equal => f64::NAN,
                                    std::cmp::Ordering::Greater => f64::INFINITY,
                                });
                            } else {
                                *exp = Self::apply_arithmetic_op_int(i1, i2, iop);

                                // TODO:
                                // record fold op
                                // #[cfg(flag = "trace_optimize")]
                                // self.record.push(FoldInfo {
                                //     srcloc: expr.lineinfo(),
                                //     derive_n: *derive,
                                //     status: StillConst,
                                //     new: new_exp,
                                // });
                            }
                            StillConst
                        } else {
                            match (op, lhs.inner(), rhs.inner()) {
                                (BinOp::Concat, Expr::Literal(l1), Expr::Literal(l2)) => {
                                    *exp = Expr::Int((l1.len() + l2.len()) as i64);
                                    StillConst
                                }
                                _ => NonConst,
                            }
                        }
                    }
                    _ => NonConst,
                }
            }

            Expr::UnaryOp { op, expr } => {
                if let StillConst = self.try_fold(expr) {
                    // execute fold operation
                    if let Some(new_exp) = match (op, expr.inner()) {
                        // not nil => true
                        (UnOp::Not, Expr::Nil) => Some(Expr::True),

                        // not literial => false
                        (UnOp::Not, _) => Some(Expr::False),

                        // # str => len(str)
                        (UnOp::Length, Expr::Literal(lit)) => Some(Expr::Int(lit.len() as i64)),

                        // - number => 0 - number
                        (UnOp::Minus, Expr::Int(i)) => Some(Expr::Int(0 - i)),
                        (UnOp::Minus, Expr::Float(f)) => Some(Expr::Float(0.0 - f)),

                        // ~int
                        (UnOp::BitNot, Expr::Int(i)) => Some(Expr::Int(!i)),

                        _ => None,
                    } {
                        // update expr node
                        *exp = new_exp;

                        #[cfg(flag = "trace_optimize")]
                        self.record.push(FoldInfo {
                            srcloc: expr.lineinfo(),
                            derive_n: *derive,
                            status: StillConst,
                            new: new_exp,
                        });

                        StillConst
                    } else {
                        NonConst
                    }
                } else {
                    NonConst
                }
            }

            _ => NonConst,
        }
    }

    fn gen_arithmetic_op_int(op: BinOp) -> Option<fn(i64, i64) -> i64> {
        match op {
            BinOp::Add => Some(|l, r| l + r),
            BinOp::Minus => Some(|l: i64, r: i64| l - r),
            BinOp::Mul => Some(|l, r| l * r),
            // BinOp::Mod => Some(|l, r| l % r),
            BinOp::Pow => Some(|l, r| l ^ r),
            BinOp::IDiv => Some(|l, r| l / r),

            // 1 / 1 => 1.0
            // BinOp::Div => Some(|l, r| l / r),
            _ => None,
        }
    }

    fn gen_arithmetic_op_float(op: BinOp) -> Option<fn(f64, f64) -> f64> {
        match op {
            BinOp::Add => Some(|l, r| l + r),
            BinOp::Minus => Some(|l, r| l - r),
            BinOp::Mul => Some(|l, r| l * r),
            BinOp::Mod => Some(|l, r| l % r),
            BinOp::Pow => Some(|l, r| l.powf(r)),
            BinOp::IDiv => Some(|l, r| l / r),
            BinOp::Div => Some(|l, r| l / r),
            _ => None,
        }
    }

    fn apply_arithmetic_op_int(lhs: i64, rhs: i64, arth: impl Fn(i64, i64) -> i64) -> Expr {
        Expr::Int(arth(lhs, rhs))
    }

    fn apply_arithmetic_op_float(lhs: f64, rhs: f64, arth: impl Fn(f64, f64) -> f64) -> Expr {
        Expr::Float(arth(lhs, rhs))
    }

    // fn try_fold_binary_op(&mut self, op: &BinOp, lhs: &mut Expr, rhs: &mut Expr) -> Option<Expr> {
    //     match op {

    //         // bitwise op
    //         BinOp::BitAnd => todo!(),
    //         BinOp::BitOr => todo!(),
    //         BinOp::BitXor => todo!(),
    //         BinOp::Shl => todo!(),
    //         BinOp::Shr => todo!(),

    //         // str1..str2
    //         BinOp::Concat => todo!(),

    //         // logic op
    //         BinOp::And => todo!(),
    //         BinOp::Or => todo!(),
    //         // BinOp::Eq => todo!(),
    //         // BinOp::Less => todo!(),
    //         // BinOp::LE => todo!(),
    //         // BinOp::Neq => todo!(),
    //         // BinOp::Great => todo!(),
    //         // BinOp::GE => todo!(),
    //         _ => NonConst
    //     }
    // }
}

mod test {
    #[test]
    fn ast_node_size_check() {
        use crate::compiler::ast::*;
        use std::mem::size_of;

        const _B_SIZE: usize = size_of::<Block>();
        const _P_SIZE: usize = size_of::<ParaList>();
        const _D_SIZE: usize = size_of::<FuncBody>();
        const _C_SIZE: usize = size_of::<FuncCall>();

        assert_eq!(size_of::<Expr>(), 64);
        assert_eq!(size_of::<Stmt>(), 64);
    }

    #[test]
    fn ast_dump_test() {
        use crate::compiler::ast::*;
        use crate::Parser;
        use std::io::BufWriter;

        let lua_src_path = "testes/all.lua";
        let src = std::fs::read_to_string(lua_src_path).unwrap();
        let block = Parser::parse(&src, lua_src_path.to_string()).unwrap();

        let tmp_file = {
            let mut temp_dir = std::env::temp_dir();
            temp_dir.push("luac-rs.test.ast.dump");
            std::fs::File::create(temp_dir).unwrap()
        };

        let mut buf = BufWriter::new(tmp_file);
        let result = AstDumper::dump(&block, DumpPrecison::Statement, &mut buf, false);

        assert!(result.is_ok())
    }

    #[test]
    fn constant_fold_exec_test() {
        use crate::compiler::ast::{ConstantFoldPass, PassRunStatus, TransformPass};
        use crate::compiler::parser::Parser;

        let emsg = format!(
            "unable to find directory: \"testes\" with base dir:{}",
            std::env::current_dir().unwrap().display()
        );

        let dir = std::fs::read_dir("./testes/").expect(&emsg);

        let mut src_paths = dir
            .map(|e| e.map(|e| e.path()))
            .collect::<Result<Vec<_>, _>>()
            .unwrap();

        src_paths.sort();

        src_paths
            .into_iter()
            .filter(|p| {
                // filter filename ends with '.lua'
                matches! { p.extension().map(|ex| ex.to_str().unwrap_or_default()), Some("lua")}
            })
            .map(|p| {
                // take file name
                let file_name = p
                    .file_name()
                    .and_then(|s| s.to_str())
                    .map(|s| s.to_string())
                    .unwrap_or_default();
                // read content to string
                let content = std::fs::read_to_string(p).unwrap_or_default();
                (file_name, content)
            })
            .flat_map(|(file, content)| {
                // execute parse
                Parser::parse(&content, file)
            })
            .for_each(|mut block| {
                let mut cfp = ConstantFoldPass::new();
                assert!(cfp.walk(&mut block).is_ok())
            });
    }
}
