/// Warpper for an ast-node to attach source location info
pub struct WithSrcLoc<T> {
    // lineinfo: (begin, end)
    lineinfo: (u32, u32),
    node: T,
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
///
/// var ::=  Name | prefixexp `[` exp `]` | prefixexp `.` Name
///
/// varlist ::= var {`,` var}
///
/// prefixexp ::= var | functioncall | `(` exp `)`
///
/// label ::= `::` Name `::`
///
/// namelist ::= Name {`,` Name}
///
/// attr ::= [`<` Name `>`]
pub enum Stmt {
    // global assignment
    Assign {
        vars: Vec<ExprNode>,
        exprs: Vec<ExprNode>,
    },

    FuncCall(Box<FuncCall>),

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
        names: Vec<String>,
        exprs: Vec<ExprNode>,
    },
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
        args: Vec<ExprNode>,
    },

    // i.e: class:func(1, 2, 3, ...)
    MethodCall {
        prefix: Box<ExprNode>,
        method_name: String,
        args: Vec<ExprNode>,
    },
}

/// functiondef ::= function funcbody
///
/// funcbody ::= `(` [parlist] `)` block end
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

    // function funcbody
    FuncDefine(FuncBody),

    // i.e:  table[key]
    OffsetIndex {
        prefix: Box<ExprNode>,
        exp: Box<ExprNode>,
    },

    // i.e:  table.key | key
    NameIndex {
        prefix: Option<Box<ExprNode>>,
        exp: Box<ExprNode>,
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

pub struct ParaList {
    vargs: bool,
    namelist: Vec<String>,
}

impl ParaList {
    pub fn new(vargs: bool, namelist: Vec<String>) -> Self {
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

/// binop ::=  `+` | `-` | `*` | `/` | `//` | `^` | `%` | 
///     `&` | `~` | `|` | `>>` | `<<` | `..` | 
///     `<` | `<=` | `>` | `>=` | `==` | `~=` | 
#[rustfmt::skip]
pub enum BinOp {
    Add,     Minus,   Mul,    Div,   IDiv,  Pow,    Mod,   
    BitAnd,  BitXor,  BitOr,  Shr,   Shl,   Concat,
    Less,    LE,      Great,  GE,    Eq,    Neq 
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
