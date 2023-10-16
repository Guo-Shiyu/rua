use std::{
    io::{BufWriter, Error, Write},
    ops::{Deref, DerefMut},
};

#[derive(Default)]
/// Warpper for an ast-node to attach source location info.
///
/// `lineinfo` represent that (begin, end) of defination line number.
///
/// Because AST node are stored on heap. `mem_address()` method used to get it's
/// address as hex number. This method used in `AstDumper` only.
pub struct SrcLoc<T> {
    node: T,
    // lineinfo: (begin, end)
    pub lineinfo: (u32, u32),
}

impl<T> SrcLoc<T> {
    pub fn new(n: T, lines: (u32, u32)) -> Self {
        SrcLoc {
            lineinfo: lines,
            node: n,
        }
    }

    pub fn inner(self) -> T {
        self.node
    }

    pub fn inner_ref(&self) -> &T {
        &self.node
    }

    pub fn inner_mut(&mut self) -> &mut T {
        &mut self.node
    }

    pub fn def_begin(&self) -> u32 {
        self.lineinfo.0
    }

    pub fn def_end(&self) -> u32 {
        self.lineinfo.1
    }

    pub fn mem_address(&self) -> usize {
        self as *const _ as usize
    }
}

impl<T> Deref for SrcLoc<T> {
    type Target = T;
    fn deref(&self) -> &Self::Target {
        &self.node
    }
}

impl<T> DerefMut for SrcLoc<T> {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.node
    }
}

/// ``` text
/// block ::= {stat} [retstat]
/// ```
pub struct Block {
    pub chunk: String,
    pub stats: Vec<StmtNode>,
    pub ret: Option<Vec<ExprNode>>,
}

impl Block {
    pub const NAMELESS_CHUNK: &'static str = "<anonymous>";

    pub fn new(name: Option<String>, stats: Vec<StmtNode>, ret: Option<Vec<ExprNode>>) -> Self {
        Block {
            chunk: name.unwrap_or_else(|| Self::NAMELESS_CHUNK.to_string()),
            stats,
            ret,
        }
    }

    /// Check self weither a empty block. (Empty block was created by ConstantFold.)
    pub fn is_empty(&self) -> bool {
        self.stats.is_empty() && self.ret.is_none()
    }
}

/// ``` text
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
/// ```

pub enum Stmt {
    // assignment,  x = A | x.y.z = A
    Assign {
        vars: Vec<ExprNode>,
        exprs: Vec<ExprNode>,
    },

    // fn(a, b, c) | x.y.fn(a, b, c)
    FnCall(FuncCall),

    Lable(String),

    Goto(String),

    Break,

    DoEnd(BasicBlock),

    While {
        exp: ExprNode,
        block: BasicBlock,
    },

    Repeat {
        block: BasicBlock,
        exp: ExprNode,
    },

    IfElse {
        cond: ExprNode,
        then: BasicBlock,
        els: Option<BasicBlock>,
    },

    NumericFor(Box<NumericFor>),

    GenericFor(Box<GenericFor>),

    FnDef {
        pres: Vec<SrcLoc<String>>,
        method: Option<Box<SrcLoc<String>>>,
        body: Box<SrcLoc<FuncBody>>,
    },

    LocalVarDecl {
        names: Vec<(SrcLoc<String>, Option<Attribute>)>,
        exprs: Vec<ExprNode>,
    },

    // single expression as a statement
    Expr(ExprNode),
}

/// ``` text
/// functioncall ::=  prefixexp args | prefixexp `:` Name args
///
/// funcname ::= Name {`.` Name} [`:` Name]
///
/// prefixexp ::= var | functioncall | `(` exp `)`
///
/// args ::=  `(` [explist] `)` | tablector | LiteralString
/// ```
pub enum FuncCall {
    // i.e: func(1, 2, 3, ...)
    FreeFnCall {
        prefix: ExprNode,
        args: SrcLoc<ArgumentList>,
    },

    // i.e: class:func(1, 2, 3, ...)
    MethodCall {
        prefix: ExprNode,
        method: Box<SrcLoc<String>>,
        args: SrcLoc<ArgumentList>,
    },
}

/// ``` text
/// funcbody ::= `(` [parlist] `)` block end
/// paralist ::= namelist [`,` `...`] | `...`
/// ```
pub struct FuncBody {
    pub params: ParameterList,
    pub body: BasicBlock,
}

/// ``` text
/// for Name `=` exp `,` exp [`,` exp] do block end
/// ```
pub struct NumericFor {
    pub iter: SrcLoc<String>,
    pub init: ExprNode,
    pub limit: ExprNode,
    pub step: ExprNode,
    pub body: BasicBlock,
}

/// ``` text
/// for namelist in explist do block end
/// ```
pub struct GenericFor {
    pub iters: Vec<SrcLoc<String>>,
    pub exprs: Vec<ExprNode>,
    pub body: BasicBlock,
}

/// ``` text
/// exp ::=  nil | false | true | Numeral | LiteralString | `...` |
///     functiondef | prefixexp | tablector |
///     exp binop exp | unop exp
///
/// prefixexp ::= var | functioncall | `(` exp `)`
/// ```
#[derive(Default)]
pub enum Expr {
    #[default]
    Nil,
    False,
    True,
    Int(i64),
    Float(f64),
    Literal(String),
    Dots,

    Ident(String),

    // anonymous function defination, e.g, : function () ... end
    Lambda(FuncBody),

    // table.key | table[key]
    Index {
        prefix: ExprNode,
        key: ExprNode,
    },

    // fn(a, b, c) | fn(...) | fn { ... } | fn "..."
    FuncCall(FuncCall),

    // x = expr | ["xx"] = expr
    TableCtor(Vec<Field>),

    BinaryOp {
        lhs: ExprNode,
        op: BinOp,
        rhs: ExprNode,
    },

    UnaryOp {
        op: UnOp,
        expr: ExprNode,
    },
}

/// There are two possible attributes: const, which declares a constant variable, that is, a variable that cannot be assigned to after its initialization;
/// and close, which declares a to-be-closed variable. A list of variables can contain at most one to-be-closed variable.
///
/// A to-be-closed variable behaves like a constant local variable, except that its value is closed whenever the variable goes out of scope, including normal
/// block termination, exiting its block by `break` / `goto` / `return` , or exiting by an error.
///
/// Here, to close a value means to call its __close metamethod. When calling the metamethod, the value itself is passed as the first argument and the error
/// object that caused the exit (if any) is passed as a second argument; if there was no error, the second argument is nil.
///
/// Attribute syntax e.g. :
/// ``` lua
/// local x <const> = 1
///
/// local y <close> = {}
///
/// setmetatable(y, {__close = function(y_self, err)  ... end}
///
/// ```
#[repr(u8)]
#[derive(Debug, Clone, Copy)]
pub enum Attribute {
    Const,
    Close,
}

impl From<Attribute> for u8 {
    fn from(value: Attribute) -> Self {
        match value {
            Attribute::Const => 0,
            Attribute::Close => 1,
        }
    }
}

pub struct FnHeader<H> {
    pub namelist: Vec<H>,
    pub vargs: bool,
}

/// Parameter list in function defination.
pub type ParameterList = FnHeader<String>;

/// Argument list in function call.
pub type ArgumentList = FnHeader<ExprNode>;

/// ``` text
/// field ::= `[` exp `]` `=` exp | Name `=` exp | exp
/// ```
pub struct Field {
    pub key: Option<Box<ExprNode>>,
    pub val: Box<ExprNode>,
}

impl Field {
    pub fn new(key: Option<Box<ExprNode>>, val: Box<ExprNode>) -> Self {
        Field { key, val }
    }
}

/// Binary operators in Lua 5.4
/// 
#[rustfmt::skip]
#[derive(Clone, Copy, PartialEq)]
pub enum BinOp {
    // Notice that do not change the order of operators, there is a 
    // `binary_operator_priority` table in parser depends on that.
    
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

/// ``` text
/// unop ::= `-` | not | `#` | `~`
/// ```
pub enum UnOp {
    Minus,
    Not,
    Length,
    BitNot,
}

pub type ExprNode = Box<SrcLoc<Expr>>;
pub type StmtNode = Box<SrcLoc<Stmt>>;
pub type BasicBlock = Box<SrcLoc<Block>>;

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

impl AstDumper {
    fn new(level: DumpPrecison, colored: bool, buf: Option<Vec<u8>>) -> Self {
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

    fn walk(&mut self, block: &Block) {
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
        match stmt.inner_ref() {
            Stmt::Assign { vars, exprs } => self.write_lable(buf, COLLAPSE).and_then(|_| {
                self.write_name(buf, "Assignment", this)?;
                if vars.len() == 1 {
                    write!(
                        buf,
                        "decl: {}, expr-count: [{}]",
                        Self::inspect(vars[0].inner_ref()),
                        exprs.len(),
                    )?;
                } else {
                    let names = vars
                        .iter()
                        .map(|e| Self::inspect(e.inner_ref()))
                        .collect::<Vec<_>>();

                    write!(buf, "decls: {:?}, expr-count: [{}]", names, exprs.len(),)?;
                }
                self.write_lineinfo(buf, stmt.lineinfo.0)
            }),

            Stmt::Lable(name) => self.write_lable(buf, COLLAPSE).and_then(|_| {
                self.write_name(buf, "Lable", this)?;
                write!(buf, "name: {}", name.as_str(),)?;
                self.write_lineinfo(buf, stmt.lineinfo.0)
            }),

            Stmt::Goto(lable) => self.write_lable(buf, COLLAPSE).and_then(|_| {
                self.write_name(buf, "Goto", this)?;
                write!(buf, "lable: {}", lable.as_str(),)?;
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

            Stmt::FnCall(call) => match call {
                FuncCall::FreeFnCall { prefix, args } => {
                    self.write_lable(buf, COLLAPSE).and_then(|_| {
                        self.write_name(buf, "FreeFuncCall", this)?;
                        write!(
                            buf,
                            "prefix: {}, vararg: {}, positional-args-num: {}",
                            Self::inspect(prefix.inner_ref()),
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
                        Self::inspect(prefix.inner_ref()),
                        method.as_str(),
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

            Stmt::IfElse {
                cond: exp,
                then,
                els,
            } => {
                self.write_lable(buf, '+').and_then(|_| {
                    self.write_name(buf, "IfElseEnd", this)?;
                    write!(
                        buf,
                        "exp: {}, then: 0x{:x}, else: 0x{:x}",
                        Self::inspect(exp.inner_ref()),
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

                if let Some(e) = els {
                    self.inc_indent();
                    self.dump_block(e, buf)?;
                    self.dec_indent();
                }

                Ok(())
            }

            Stmt::NumericFor(numfor) => {
                self.write_lable(buf, '+')?;
                self.write_name(buf, "NumericFor", this)?;
                write!(
                    buf,
                    "name: {}, init: {}, limit: {}, step: {}",
                    numfor.iter.as_str(),
                    Self::inspect(&numfor.init),
                    Self::inspect(&numfor.limit),
                    Self::inspect(&numfor.step),
                )?;

                self.write_lineinfo(buf, stmt.lineinfo.0)?;

                self.inc_indent();
                self.dump_block(&numfor.body, buf)?;
                self.dec_indent();
                Ok(())
            }
            Stmt::GenericFor(genfor) => {
                self.write_lable(buf, '+')?;
                self.write_name(buf, "GenericFor", this)?;
                write!(
                    buf,
                    "names: {:?}, exprs-num: {}",
                    genfor.iters.iter().map(|s| s.as_str()).collect::<Vec<_>>(),
                    genfor.exprs.len(),
                )?;
                self.write_lineinfo(buf, stmt.lineinfo.0)?;

                self.inc_indent();
                self.dump_block(&genfor.body, buf)?;
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
                    method.as_ref().map(|s| s.as_str()),
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
            Expr::Lambda(_) => "<FuncDef>".to_string(),
            Expr::Index { prefix, key } => {
                let mut buf = String::with_capacity(32);
                buf.push_str(&Self::inspect(prefix.inner_ref()));
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

mod test {
    #[test]
    fn ast_node_size_check() {
        use super::*;
        use std::mem::size_of;

        const _B_SIZE: usize = size_of::<Block>();
        const _P_SIZE: usize = size_of::<ArgumentList>();
        const _D_SIZE: usize = size_of::<FuncBody>();
        const _C_SIZE: usize = size_of::<FuncCall>();

        assert!(size_of::<Expr>() <= 64);
        assert!(size_of::<Stmt>() <= 64);
    }

    #[test]
    fn ast_dump_test() {
        use super::*;
        use crate::parser::Parser;
        use std::io::BufWriter;
        let lua_src_path = "test/all.lua";
        let src = std::fs::read_to_string(lua_src_path).unwrap();
        let block = Parser::parse(&src, Some(lua_src_path.to_string())).unwrap();

        let tmp_file = {
            let mut temp_dir = std::env::temp_dir();
            temp_dir.push("luac-rs.test.ast.dump");
            std::fs::File::create(temp_dir).unwrap()
        };

        let mut buf = BufWriter::new(tmp_file);
        let result = AstDumper::dump(&block, DumpPrecison::Statement, &mut buf, false);

        assert!(result.is_ok())
    }
}
