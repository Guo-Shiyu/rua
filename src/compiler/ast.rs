use std::{
    io::{BufWriter, Error, Write},
    process::Output,
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
#[derive(Clone, Copy)]
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
    NoUnary,
}

pub type ExprNode = WithSrcLoc<Expr>;
pub type StmtNode = WithSrcLoc<Stmt>;

pub trait AstWalker {
    type Output;
    type Error;
    fn walk(&mut self, root: &Block) -> Result<Self::Output, Self::Error>;
}

pub trait AstRewriter {
    type Output;
    type Error;
    fn rewrite_stmt(&mut self, node: &mut StmtNode) -> Result<Output, Error>;
    fn rewrite_expr(&mut self, node: &mut ExprNode) -> Result<Output, Error>;
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
}

impl AstDumper {
    pub fn new(level: DumpPrecison, colored: bool) -> Self {
        AstDumper {
            depth: 0,
            colored,
            level,
        }
    }

    pub fn dump(
        block: &Block,
        precision: DumpPrecison,
        buf: &mut BufWriter<impl Write>,
        colored: bool,
    ) -> Result<(), Error> {
        let mut dumper = AstDumper::new(precision, colored);
        dumper.dump_block(block, buf)
    }
}

impl AstWalker for AstDumper {
    type Output = Vec<u8>;
    type Error = std::io::Error;
    fn walk(&mut self, block: &Block) -> Result<Vec<u8>, Error> {
        const BUF_SIZE: usize = 8192; // equal with std::io::DEFAULT_BUF_SIZE
        let dest: Self::Output = Vec::with_capacity(BUF_SIZE);
        let mut buf = BufWriter::new(dest);
        self.dump_block(block, &mut buf)?;
        Ok(buf.into_inner()?)
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
                let mut buf = String::with_capacity(64);
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
}
