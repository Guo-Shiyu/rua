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

    pub fn inner(self) -> T {
        self.node
    }

    pub fn lineinfo(&self) -> (u32, u32) {
        self.lineinfo
    }
}

/// block ::= {stat} [retstat]
pub struct Block {
    stats: Vec<StmtNode>,
    ret: Option<Vec<ExprNode>>,
}

impl Block {
    pub fn new(stats: Vec<StmtNode>, ret: Option<Vec<ExprNode>>) -> Self {
        Block { stats, ret }
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

    // Lable{ tag: String }
    Lable(String),

    // Goto { lable: String }
    Goto(String),

    Break,

    // DoEnd { block: Block }
    DoEnd(Block),

    While {
        exp: Box<ExprNode>,
        block: Box<Block>,
    },

    Repeat {
        block: Block,
        exp: Box<ExprNode>,
    },

    IfElse {
        exp: Box<ExprNode>,
        then: Box<Block>,
        els: Box<Block>,
    },

    NumericFor {
        name: String,
        init: Box<Expr>,
        limit: Box<Expr>,
        step: Box<Expr>,
        body: Box<Block>,
    },

    GenericFor {
        names: Vec<String>,
        exprs: Vec<Expr>,
        body: Box<Block>,
    },

    FnDef {
        namelist: FuncName,
        def: Box<FuncBody>,
    },

    LocalVarDecl {
        names: Vec<(String, Option<Attribute>)>,
        exprs: Vec<ExprNode>,
    },

    Expr(Box<Expr>),
}

/// funcname ::= Name {`.` Name} [`:` Name]
pub struct FuncName {
    pres: Vec<String>,
    method: Option<String>,
}

impl FuncName {
    pub fn new(pres: Vec<String>, method: Option<String>) -> Self {
        FuncName { pres, method }
    }
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

mod test {
    #[test]
    fn struct_size_check() {
        use crate::compiler::ast::*;
        use std::mem::size_of;

        const _B_SIZE: usize = size_of::<Block>();
        const _N_SIZE: usize = size_of::<FuncName>();
        const _P_SIZE: usize = size_of::<ParaList>();
        const _D_SIZE: usize = size_of::<FuncBody>();
        const _C_SIZE: usize = size_of::<FuncCall>();

        assert!(size_of::<Expr>() <= 64);
        assert!(size_of::<Stmt>() <= 64);
    }
}
