use crate::{SyntaxErrKind, SyntaxError};

use super::{
    ast::*,
    lexer::{Lexer, Token},
};

/// An LL(2) parser for Lua 5.4       
///
/// Full syntax of Lua 5.4 can be find [here](https://www.lua.org/manual/5.4/manual.html#9).
pub struct Parser<'a> {
    lex: Lexer<'a>,
    current: Token,
    ahead: Token,
}

type Error = Box<SyntaxError>;

impl Parser<'_> {
    /// priority table for binary operators.
    #[rustfmt::skip]
    const BINARY_PRIORITY: [(usize, usize); 21] = [
        (10, 10), (10, 10),      // +  -  
        (11, 11), (11, 11),      // *  %  
        (14, 13),                // ^       (right associatie) 
        (11, 11), (11, 11),      // /  //  
        (6, 6), (4, 4), (5, 5),  // &  |  ~ 
        (7, 7), (7, 7),          // << >> 
        (9, 8),                  // ..      (right associatie) 
        (3, 3), (3, 3), (3, 3),  // ==, <, <= 
        (3, 3), (3, 3), (3, 3),  // ~=, >, >= 
        (2, 2), (1, 1)           // and, or   
    ];

    /// All unary operator has same priority.
    const UNARY_PRIORITY: usize = 12;

    /// Parse given source string into `BasicBlock` or `Error`.  Return the root node
    /// of source code. if no chunk name given, chunk will be named to `Block::NAMELESS_CHUNK`.
    pub fn parse(src: &str, chunkname: Option<String>) -> Result<BasicBlock, Box<SyntaxError>> {
        let mut parser = Parser {
            lex: Lexer::new(src),
            current: Token::Eof,
            ahead: Token::Eof,
        };

        // init parser state to first token
        parser.next()?;

        // treat whole file as a function body
        let mut block = parser.block()?;

        debug_assert!(block.chunk.as_str() == Block::NAMELESS_CHUNK);
        debug_assert!(parser.current.is_eof());
        debug_assert!(parser.ahead.is_eof());

        if let Some(cn) = chunkname {
            block.chunk = cn;
        }

        Ok(block)
    }

    /// Motivate lexer to next token and update self.current.
    /// This will clear self.ahead if self.ahead is't EOF.
    fn next(&mut self) -> Result<(), Error> {
        if let Token::Eof = self.ahead {
            self.current = self.lex.tokenize().map_err(|e| self.error(e))?;
        } else {
            self.current = std::mem::replace(&mut self.ahead, Token::Eof);
        }
        Ok(())
    }

    /// Motivate lexer to next token and update self.ahead
    fn look_ahead(&mut self) -> Result<(), Error> {
        debug_assert!(self.ahead == Token::Eof);
        self.ahead = self.lex.tokenize().map_err(|e| self.error(e))?;
        Ok(())
    }

    /// ``` text
    /// block ::= {stat} [retstat]
    /// ```
    fn block(&mut self) -> Result<BasicBlock, Error> {
        const DEFAULT_BLOCK_SIZE: usize = 32;
        let mut stats = Vec::with_capacity(DEFAULT_BLOCK_SIZE);

        let beg = self.lex.line();
        loop {
            // ';' after each statement is optional
            while self.test_and_next(Token::Semi)? {}

            if self.is_block_end() {
                stats.shrink_to_fit();
                break;
            }

            stats.push(self.stmt()?);
        }

        // optional return statement
        let ret = if self.test_and_next(Token::Return)? {
            self.return_stmt()?
        } else {
            None
        };

        let end = self.lex.line();
        Ok(Box::new(SrcLoc::new(
            Block::new(None, stats, ret),
            (beg, end),
        )))
    }

    fn is_block_end(&self) -> bool {
        matches!(
            self.current,
            Token::Eof | Token::Else | Token::Elseif | Token::Until | Token::End | Token::Return
        )
    }

    /// ``` text
    /// stat ::=  `;` |
    ///     varlist `=` explist |
    ///     functioncall |
    ///     label |
    ///     break |
    ///     goto Name |
    ///     do block end |
    ///     while exp do block end |
    ///     repeat block until exp |
    ///     if exp then block {elseif exp then block} [else block] end |
    ///     for Name `=` exp `,` exp [`,` exp] do block end |
    ///     for namelist in explist do block end |
    ///     function funcname funcbody |
    ///     local function Name funcbody |
    ///     local attnamelist [`=` explist]
    /// ```
    fn stmt(&mut self) -> Result<StmtNode, Error> {
        // skip ';'
        while self.test_and_next(Token::Semi)? {}

        let begin = self.lex.line();

        let stmt = {
            match &self.current {
                Token::Break => {
                    self.next()?;
                    Stmt::Break
                }
                Token::Follow => self.lable()?,
                Token::Goto => self.goto()?,
                Token::Do => self.do_end()?,
                Token::While => self.while_do()?,
                Token::Repeat => self.repeat_until()?,
                Token::If => self.if_else()?,
                Token::For => self.for_stmt()?,
                Token::Function => self.global_fndef()?,
                Token::Local => {
                    self.next()?; // skip `local`
                    if self.test_and_next(Token::Function)? {
                        self.local_fndef()?
                    } else {
                        self.local_assign()?
                    }
                }

                // expr as statment
                _ => return self.expr_stat(),
            }
        };

        let end = self.lex.line();
        Ok(Box::new(SrcLoc::new(stmt, (begin, end))))
    }

    /// Parse single expr as statememt.
    fn expr_stat(&mut self) -> Result<StmtNode, Error> {
        let expr = self.expr()?;
        let (lineinfo, inner) = (expr.lineinfo, expr.inner());

        let stmt = if let Expr::FuncCall(call) = inner {
            SrcLoc::new(Stmt::FnCall(call), lineinfo)
        } else {
            let first = Box::new(SrcLoc::new(inner, lineinfo));
            if self.test(Token::Comma) {
                // multi assignment
                let varlist = self.exprlist(Some(first))?;
                self.check_and_next(Token::Assign)?;
                let exprlist = self.exprlist(None)?;
                SrcLoc::new(
                    Stmt::Assign {
                        vars: varlist,
                        exprs: exprlist,
                    },
                    lineinfo,
                )
            } else if self.test_and_next(Token::Assign)? {
                // single assignment
                let exprlist = self.exprlist(None)?;
                SrcLoc::new(
                    Stmt::Assign {
                        vars: vec![first],
                        exprs: exprlist,
                    },
                    lineinfo,
                )
            } else {
                // single expr as statememt
                SrcLoc::new(Stmt::Expr(first), lineinfo)
            }
        };
        Ok(Box::new(stmt))
    }

    /// ``` text
    /// label ::= `::` Name `::`
    /// ```
    fn lable(&mut self) -> Result<Stmt, Error> {
        self.next()?;

        let ret = match &mut self.current {
            Token::Ident(id) => Ok(Stmt::Lable(std::mem::take(id))),
            _ => Err(self.unexpected(&[Token::Follow])),
        };
        if ret.is_ok() {
            self.next()?; // skip lable
            self.check_and_next(Token::Follow)?;
        }
        ret
    }

    /// ``` text
    /// goto ::= `goto` Name
    /// ```
    fn goto(&mut self) -> Result<Stmt, Error> {
        self.next()?; // skip `goto`

        if let Token::Ident(lable) = &mut self.current {
            Ok(Stmt::Goto(std::mem::take(lable)))
        } else {
            Err(self.unexpected(&[id()]))
        }
    }

    /// ``` text
    /// do ::= `do` block `end`
    /// ```
    fn do_end(&mut self) -> Result<Stmt, Error> {
        self.next()?; // skip 'do'
        let block = self.block()?;
        self.test_and_next(Token::End)?;
        Ok(Stmt::DoEnd(block))
    }

    /// ``` text
    /// while ::= `while` exp `do` block `end`
    /// ```
    fn while_do(&mut self) -> Result<Stmt, Error> {
        self.next()?; // skip 'while'
        let exp = self.expr()?;
        self.check_and_next(Token::Do)?;
        let block = self.block()?;
        self.check_and_next(Token::End)?;

        Ok(Stmt::While { exp, block })
    }

    /// ``` text
    /// repeat ::= `repeat` block `until` exp
    /// ```
    fn repeat_until(&mut self) -> Result<Stmt, Error> {
        // debug_assert!(self.ahead == Token::Repeat);
        self.next()?;
        let block = self.block()?;

        self.test_and_next(Token::Until)?;
        let exp = self.expr()?;

        Ok(Stmt::Repeat { block, exp })
    }

    /// ``` text
    /// if ::= `if` exp `then` block {`elseif` exp `then` block} [`else` block] `end`
    /// ```
    fn if_else(&mut self) -> Result<Stmt, Error> {
        self.next()?; // skip 'if'
        let cond = self.expr()?;
        self.check_and_next(Token::Then)?;

        let then = self.block()?;

        if self.test_and_next(Token::Else)? {
            // if-else-end
            let els = self.block()?;
            self.check_and_next(Token::End)?;
            Ok(Stmt::IfElse {
                cond,
                then,
                els: Some(els),
            })
        } else if self.test(Token::Elseif) {
            // if-elseif-...-end
            let begin = self.lex.line();
            let else_stmt = {
                let stmt = self.if_else()?;
                let end = self.lex.line();
                Box::new(SrcLoc::new(stmt, (begin, end)))
            };

            let els = Box::new(SrcLoc::new(
                Block::new(None, vec![else_stmt], None),
                (begin, self.lex.line()),
            ));

            Ok(Stmt::IfElse {
                cond,
                then,
                els: Some(els),
            })
        } else {
            // if-then-end
            self.check_and_next(Token::End)?;
            Ok(Stmt::IfElse {
                cond,
                then,
                els: None,
            })
        }
    }

    /// ``` text
    /// numerical_for ::= `for` Name `=` exp `,` exp [`,` exp] `do` block `end`
    /// generic_for ::= `for` namelist `in` explist `do` block `end`
    /// ```
    fn for_stmt(&mut self) -> Result<Stmt, Error> {
        self.next()?; // skip `for`

        let iter = if let Token::Ident(first) = &mut self.current {
            let line = self.lex.line();
            SrcLoc::new(std::mem::take(first), (line, line))
        } else {
            return Err(self.unexpected(&[id()]));
        };

        self.next()?;
        let stmt = match &self.current {
            Token::Comma | Token::In => self.generic_for(iter)?,
            Token::Assign => self.numeric_for(iter)?,
            _ => return Err(self.unexpected(&[Token::Eq, Token::Comma])),
        };

        Ok(stmt)
    }

    /// ``` text
    /// generic_for ::= `for` namelist `in` explist `do` block `end`    
    /// ```
    fn generic_for(&mut self, iter: SrcLoc<String>) -> Result<Stmt, Error> {
        // "for k, v in ..." is most common senario
        let mut iters = Vec::with_capacity(2);
        iters.push(iter);

        loop {
            match &mut self.current {
                Token::Ident(id) => {
                    let line = self.lex.line();
                    iters.push(SrcLoc::new(std::mem::take(id), (line, line)));
                    self.next()?;
                }
                Token::Comma => self.next()?,
                Token::In => break,
                _ => return Err(self.unexpected(&[id(), Token::In])),
            }
        }
        self.check_and_next(Token::In)?;
        let mut exprs = Vec::with_capacity(4);
        loop {
            match &self.current {
                Token::Comma => self.next()?,
                Token::Do => break,
                _ => exprs.push(self.expr()?),
            }
        }
        self.check_and_next(Token::Do)?;
        let body = self.block()?;
        self.check_and_next(Token::End)?;

        Ok(Stmt::GenericFor(Box::new(GenericFor {
            iters,
            exprs,
            body,
        })))
    }

    /// ``` text
    /// numerical_for ::= `for` Name `=` exp `,` exp [`,` exp] `do` block `end`
    /// ```
    fn numeric_for(&mut self, iter: SrcLoc<String>) -> Result<Stmt, Error> {
        self.next()?;
        let init = self.expr()?;
        self.check_and_next(Token::Comma)?;
        let limit = self.expr()?;
        let step = match &self.current {
            Token::Comma => {
                self.next()?;
                self.expr()?
            }
            _ => {
                let ln = self.lex.line();
                Box::new(SrcLoc::new(Expr::Int(1), (ln, ln)))
            }
        };
        self.check_and_next(Token::Do)?;
        let body = self.block()?;
        self.check_and_next(Token::End)?;
        Ok(Stmt::NumericFor(Box::new(NumericFor {
            iter,
            init,
            limit,
            step,
            body,
        })))
    }

    /// function ::= `function` funcname funcbody
    /// funcname ::= Name {`.` Name} [`:` Name]
    fn global_fndef(&mut self) -> Result<Stmt, Error> {
        self.next()?; // skip `function`
        let (pres, method) = self.func_name()?;
        let beg = self.lex.line();
        let body = self.func_body()?;
        let end = self.lex.line();
        Ok(Stmt::FnDef {
            pres,
            method,
            body: Box::new(SrcLoc::new(body, (beg, end))),
        })
    }

    /// ``` text
    /// local_function ::= `local` `function` Name funcbody
    /// ```
    fn local_fndef(&mut self) -> Result<Stmt, Error> {
        let func_name = if let Token::Ident(name) = &mut self.current {
            let line = self.lex.line();
            SrcLoc::new(std::mem::take(name), (line, line))
        } else {
            return Err(self.unexpected(&[id()]));
        };

        self.next()?; // eat func name

        let beg = self.lex.line();
        let fndef = self.func_body()?;
        let end = self.lex.line();
        let fndef = SrcLoc::new(Expr::Lambda(fndef), (beg, end));
        Ok(Stmt::LocalVarDecl {
            names: vec![(func_name, None)],
            exprs: vec![Box::new(fndef)],
        })
    }

    /// ``` text
    /// local_assign ::= `local` attnamelist [`=` explist]
    /// attnamelist ::=  Name attr {`,` Name attr }
    /// ```
    fn local_assign(&mut self) -> Result<Stmt, Error> {
        let mut names = Vec::with_capacity(4);
        loop {
            let line = self.lex.line();
            if let Token::Ident(ident) = &mut self.current {
                let name = SrcLoc::new(std::mem::take(ident), (line, line));
                self.next()?;

                // optional attribute
                let attribute = if self.test_and_next(Token::Less)? {
                    let mut attr_holder = String::new();
                    if let Token::Ident(attrid) = &mut self.current {
                        std::mem::swap(attrid, &mut attr_holder);
                    } else {
                        return Err(self.unexpected(&[id()]));
                    };

                    let attr = match attr_holder.as_str() {
                        "const" => Some(Attribute::Const),
                        "close" => Some(Attribute::Close),
                        other => {
                            return Err(self.error(SyntaxErrKind::InvalidAttribute {
                                attr: other.to_string(),
                            }))
                        }
                    };

                    self.next()?; // skip attribute
                    self.check_and_next(Token::Great)?;

                    attr
                } else {
                    None
                };

                names.push((name, attribute));
            }

            if let Token::Comma = self.current {
                self.next()?;
            } else {
                break;
            }
        }

        let exprs = if let Token::Assign = self.current {
            self.next()?;
            self.exprlist(None)?
        } else {
            Vec::with_capacity(names.len())
        };

        Ok(Stmt::LocalVarDecl { names, exprs })
    }

    /// ``` text
    /// funcname ::= Name {`.` Name} [`:` Name]
    /// ```
    fn func_name(&mut self) -> Result<(Vec<SrcLoc<String>>, Option<Box<SrcLoc<String>>>), Error> {
        const DOT: bool = true;
        const COLON: bool = false;
        let mut pres = Vec::with_capacity(4);
        let mut method = None;

        let mut dot_or_colon = DOT;
        loop {
            let mut is_id = false;
            match &mut self.current {
                Token::Ident(id) => {
                    let line = self.lex.line();
                    if dot_or_colon == COLON {
                        method = Some(SrcLoc::new(std::mem::take(id), (line, line)));
                        is_id = true;
                    } else {
                        pres.push(SrcLoc::new(std::mem::take(id), (line, line)));
                    }
                }

                Token::Dot => {
                    dot_or_colon = DOT;
                }

                Token::Colon => {
                    dot_or_colon = COLON;
                }
                _ => break,
            };

            self.next()?;

            if is_id {
                break;
            }
        }

        Ok((pres, method.map(Box::new)))
    }

    /// ``` text
    /// funcbody ::= `(` [parlist] `)` block `end`
    /// ```
    fn func_body(&mut self) -> Result<FuncBody, Error> {
        self.check_and_next(Token::LP)?;
        let params = self.param_list()?;
        self.check_and_next(Token::RP)?;
        let body = self.block()?;
        self.check_and_next(Token::End)?;
        Ok(FuncBody { params, body })
    }

    /// ``` text
    /// paralist ::= namelist [`,` `...`] | `...`
    /// namelist ::= Name {`,` Name}
    /// ```
    fn param_list(&mut self) -> Result<ParameterList, Error> {
        let mut vargs = false;

        if let Token::RP = self.current {
            return Ok(ParameterList {
                vargs,
                namelist: vec![],
            });
        }

        let mut namelist = Vec::with_capacity(4);
        loop {
            match &mut self.current {
                Token::Dots => {
                    vargs = true;
                    self.next()?;
                }

                Token::Ident(id) => {
                    namelist.push(std::mem::take(id));
                    self.next()?;
                }

                Token::Comma => {
                    self.next()?;
                }

                Token::RP => {
                    break;
                }

                _ => {
                    return Err(self.unexpected(&[Token::Dots, Token::RP]));
                }
            }
        }

        Ok(ParameterList { vargs, namelist })
    }

    /// paralist ::= namelist [`,` `...`] | `...`
    /// namelist ::= exp {`,` exp}
    fn argument_list(&mut self) -> Result<ArgumentList, Error> {
        let mut vargs = false;

        if let Token::RP = self.current {
            return Ok(ArgumentList {
                vargs,
                namelist: vec![],
            });
        }

        let paras = self.exprlist(None)?;

        match &self.current {
            Token::Dots => {
                vargs = true;
                self.next()?;
            }

            Token::RP => {}

            _ => {
                return Err(self.unexpected(&[Token::Dots, Token::RP]));
            }
        };

        Ok(ArgumentList {
            vargs,
            namelist: paras,
        })
    }

    /// ``` text
    /// retstat ::= return [explist] [`;`]
    /// ```
    fn return_stmt(&mut self) -> Result<Option<Vec<ExprNode>>, Error> {
        if self.is_block_end() || self.test_and_next(Token::Semi)? {
            Ok(None)
        } else {
            let exprs = self.exprlist(None)?;
            self.test_and_next(Token::Semi)?;
            Ok(Some(exprs))
        }
    }

    /// ``` text
    /// exp ::=  `nil` |
    ///  `false` |
    ///  `true` |
    ///  Numeral |
    ///  LiteralString |
    ///  `...` |
    ///  functiondef |
    ///  prefixexp |
    ///  tablector |
    ///  exp binop exp |
    ///  unop exp
    /// ```
    fn expr(&mut self) -> Result<ExprNode, Error> {
        self.subexpr(0)
    }

    fn subexpr(&mut self, limit: usize) -> Result<ExprNode, Error> {
        let begin = self.lex.line();
        let mut lhs = if let Some(uniop) = self.unary_op() {
            self.next()?;
            let sub = self.subexpr(Self::UNARY_PRIORITY)?;
            let end = self.lex.line();
            Box::new(SrcLoc::new(
                Expr::UnaryOp {
                    op: uniop,
                    expr: sub,
                },
                (begin, end),
            ))
        } else {
            self.simple_expr()?
        };

        while let Some(binop) = self.binary_op() {
            // if current binop's priority is higher than limit, then
            // parse it as a binary expression
            if Self::BINARY_PRIORITY[binop as usize].0 > limit {
                self.next()?;

                // parse rhs with right associativity poritity
                let rhs = self.subexpr(Self::BINARY_PRIORITY[binop as usize].1)?;
                let end = self.lex.line();
                lhs = Box::new(SrcLoc::new(
                    Expr::BinaryOp {
                        lhs,
                        op: binop,
                        rhs,
                    },
                    (begin, end),
                ));
            } else {
                break;
            }
        }
        Ok(lhs)
    }

    /// ``` text
    /// simpleexp -> Float | Int | String | Nil | `true` | `false` | `...` |
    ///             Constructor | `function` funcbody | suffixedexp
    /// ```
    fn simple_expr(&mut self) -> Result<ExprNode, Error> {
        if let Some(exp) = match &mut self.current {
            Token::Nil => Some(Expr::Nil),
            Token::False => Some(Expr::False),
            Token::True => Some(Expr::True),
            Token::Dots => Some(Expr::Dots),
            Token::Integer(i) => Some(Expr::Int(*i)),
            Token::Float(f) => Some(Expr::Float(*f)),
            Token::Literal(lit) => Some(Expr::Literal(std::mem::take(lit))),
            _ => None,
        } {
            let line_info = (self.lex.line(), self.lex.line());
            self.next()?;
            return Ok(Box::new(SrcLoc::new(exp, line_info)));
        };

        match &self.current {
            Token::Function => {
                self.next()?;
                let beg = self.lex.line();
                let body = self.func_body()?;
                let end = self.lex.line();
                Ok(Box::new(SrcLoc::new(Expr::Lambda(body), (beg, end))))
            }
            Token::LB => Ok(self.table_constructor()?),

            _ => self.suffixed_expr(),
        }
    }

    fn unary_op(&mut self) -> Option<UnOp> {
        match self.current {
            Token::Minus => Some(UnOp::Minus),
            Token::Not => Some(UnOp::Not),
            Token::Len => Some(UnOp::Length),
            Token::BitXor => Some(UnOp::BitNot),
            _ => None,
        }
    }

    fn binary_op(&mut self) -> Option<BinOp> {
        match self.current {
            Token::Add => Some(BinOp::Add),
            Token::Minus => Some(BinOp::Minus),
            Token::Mul => Some(BinOp::Mul),
            Token::Mod => Some(BinOp::Mod),
            Token::Pow => Some(BinOp::Pow),
            Token::Div => Some(BinOp::Div),
            Token::IDiv => Some(BinOp::IDiv),
            Token::BitAnd => Some(BinOp::BitAnd),
            Token::BitXor => Some(BinOp::BitXor),
            Token::BitOr => Some(BinOp::BitOr),
            Token::Shl => Some(BinOp::Shl),
            Token::Shr => Some(BinOp::Shr),
            Token::Concat => Some(BinOp::Concat),
            Token::Eq => Some(BinOp::Eq),
            Token::Less => Some(BinOp::Less),
            Token::LE => Some(BinOp::LE),
            Token::Neq => Some(BinOp::Neq),
            Token::GE => Some(BinOp::GE),
            Token::Great => Some(BinOp::Great),
            Token::And => Some(BinOp::And),
            Token::Or => Some(BinOp::Or),
            _ => None,
        }
    }

    /// ``` text
    /// exprlist ::= expr {`,` expr}
    /// ```
    fn exprlist(&mut self, first: Option<ExprNode>) -> Result<Vec<ExprNode>, Error> {
        let mut exprs = Vec::with_capacity(4);
        if let Some(first) = first {
            exprs.push(first);
        } else {
            exprs.push(self.expr()?);
        };
        loop {
            if self.test_and_next(Token::Comma)? {
                exprs.push(self.expr()?);
            } else {
                break Ok(exprs);
            }
        }
    }

    /// ``` text
    /// tablector ::= `{` [fieldlist] `}`
    /// fieldlist ::= field {fieldsep field} [fieldsep]
    /// fieldsep ::= `,` | `;`
    /// ```
    fn table_constructor(&mut self) -> Result<ExprNode, Error> {
        self.next()?; // skip '{'
        let beg = self.lex.line();
        let mut fieldlist = Vec::with_capacity(4);
        loop {
            match &self.current {
                Token::Comma | Token::Semi => self.next()?,
                Token::RB => break,
                _ => fieldlist.push(self.field()?),
            }
        }
        self.next()?;
        let end = self.lex.line();

        Ok(Box::new(SrcLoc::new(
            Expr::TableCtor(fieldlist),
            (beg, end),
        )))
    }

    /// ``` text
    /// field ::= `[` exp `]` `=` exp | Name `=` exp | exp
    /// ```
    fn field(&mut self) -> Result<Field, Error> {
        let mut holder = String::new();
        match &mut self.current {
            // `[` expr `]` `=` expr
            Token::LS => {
                self.next()?; // skip '['
                let key = self.expr()?;
                self.check_and_next(Token::RS)?;
                self.check_and_next(Token::Assign)?;
                return Ok(Field::new(Some(Box::new(key)), Box::new(self.expr()?)));
            }

            Token::Ident(key) => {
                std::mem::swap(&mut holder, key);
            }

            // expr
            _ => return Ok(Field::new(None, Box::new(self.expr()?))),
        };

        self.look_ahead()?;
        if let Token::Assign = self.ahead {
            // Name `=` expr
            self.next()?; // skip name
            self.next()?; // skip '='
            let line = self.lex.line();
            let expr = Box::new(SrcLoc::new(Expr::Ident(holder), (line, line)));
            Ok(Field::new(Some(Box::new(expr)), Box::new(self.expr()?)))
        } else {
            // expr
            Ok(Field::new(None, Box::new(self.expr()?)))
        }
    }

    /// ``` text
    /// primaryexp ::= NAME | '(' expr ')'
    /// ```
    fn primary_expr(&mut self) -> Result<ExprNode, Error> {
        let mut holder = String::new();
        match &mut self.current {
            Token::Ident(name) => {
                std::mem::swap(name, &mut holder);
            }

            Token::LP => {
                self.next()?;
                let exp = self.expr()?;
                self.check_and_next(Token::RP)?;
                return Ok(exp);
            }

            _ => return Err(self.unexpected(&[id(), Token::LP])),
        };

        let begin = self.lex.line();
        self.next()?;
        Ok(Box::new(SrcLoc::new(
            Expr::Ident(holder),
            (begin, self.lex.line()),
        )))
    }

    /// ``` text
    /// suffixedexp ->
    ///      primaryexp { '.' NAME | '[' exp ']' | ':' NAME funcargs | funcargs }
    /// ```
    fn suffixed_expr(&mut self) -> Result<ExprNode, Error> {
        let mut prefix = self.primary_expr()?;

        loop {
            let begin = self.lex.line();
            let expr_node = match &self.current {
                Token::Dot => {
                    self.next()?;
                    let mut holder = String::new();
                    if let Token::Ident(id) = &mut self.current {
                        std::mem::swap(id, &mut holder);
                    } else {
                        return Err(self.unexpected(&[id()]));
                    };

                    let attr = Expr::Literal(holder);
                    let curln = self.lex.line();
                    let key = Box::new(SrcLoc::new(attr, (curln, curln)));
                    self.next()?;
                    SrcLoc::new(Expr::Index { prefix, key }, (begin, self.lex.line()))
                }

                Token::LS => {
                    self.next()?; // skip '['
                    let key = self.expr()?;
                    self.check_and_next(Token::RS)?;

                    SrcLoc::new(Expr::Index { prefix, key }, (begin, self.lex.line()))
                }

                // method call
                Token::Colon => {
                    self.next()?;
                    let line = self.lex.line();
                    let mut holder = String::new();
                    if let Token::Ident(method) = &mut self.current {
                        std::mem::swap(method, &mut holder);
                    } else {
                        return Err(self.unexpected(&[id()]));
                    };

                    self.next()?; // skip name

                    let args = self.func_args()?;
                    SrcLoc::new(
                        Expr::FuncCall(FuncCall::MethodCall {
                            prefix,
                            method: Box::new(SrcLoc::new(holder, (line, line))),
                            args,
                        }),
                        (begin, self.lex.line()),
                    )
                }

                Token::Literal(_) | Token::LB | Token::LP => SrcLoc::new(
                    Expr::FuncCall(FuncCall::FreeFnCall {
                        prefix,
                        args: self.func_args()?,
                    }),
                    (begin, self.lex.line()),
                ),

                _ => return Ok(prefix),
            };

            prefix = Box::new(expr_node);
        }
    }

    /// ``` text
    /// args ::=  `(` [explist] `)` | tablector | LiteralString
    /// ```
    fn func_args(&mut self) -> Result<SrcLoc<ArgumentList>, Error> {
        let beg = self.lex.line();
        let mut holder = String::new();
        match &mut self.current {
            Token::LP => {
                self.next()?; // skip '('
                let args = self.argument_list()?;
                self.check_and_next(Token::RP)?;
                Ok(SrcLoc::new(args, (beg, self.lex.line())))
            }

            Token::LB => {
                let table = self.table_constructor()?;
                Ok(SrcLoc::new(
                    ArgumentList {
                        vargs: false,
                        namelist: vec![table],
                    },
                    (beg, self.lex.line()),
                ))
            }

            Token::Literal(s) => {
                std::mem::swap(s, &mut holder);

                let line = self.lex.line();
                let litnode = Box::new(SrcLoc::new(Expr::Literal(holder), (line, line)));
                self.next()?; // skip string literal

                Ok(SrcLoc::new(
                    ArgumentList {
                        vargs: false,
                        namelist: vec![litnode],
                    },
                    (beg, self.lex.line()),
                ))
            }

            _ => Err(self.unexpected(&[Token::LP, Token::LB, Token::Literal(String::new())])),
        }
    }

    fn test(&self, expect: Token) -> bool {
        self.current == expect
    }

    fn check(&self, expect: Token) -> Result<(), Error> {
        // this `clone()` will never deep clone a String object, it's always cheap
        if self.test(expect.clone()) {
            Ok(())
        } else {
            Err(self.unexpected(&[expect]))
        }
    }

    /// Perhaps self.ahead is equal to expect, if true, call `next`
    fn test_and_next(&mut self, expect: Token) -> Result<bool, Error> {
        if self.test(expect) {
            self.next()?;
            Ok(true)
        } else {
            Ok(false)
        }
    }

    /// Must that self.ahead is equal to expect, if true, call next
    fn check_and_next(&mut self, expect: Token) -> Result<(), Error> {
        self.check(expect)?;
        self.next()
    }
}

impl Parser<'_> {
    fn error(&self, e: SyntaxErrKind) -> Error {
        Box::new(SyntaxError {
            kind: e,
            line: self.lex.line(),
            column: self.lex.column(),
        })
    }

    fn unexpected(&self, expect: &[Token]) -> Error {
        self.error(SyntaxErrKind::UnexpectedToken {
            expect: Vec::from(expect),
            found: self.current.clone(),
        })
    }
}

/// Make an empty token for error report.
fn id() -> Token {
    Token::Ident(String::new())
}

mod test {
    #[test]
    /// parse all file in path matched pattern "/test/*.lua" .
    fn parser_all_tests() {
        use super::Parser;

        let emsg = format!(
            "unable to find directory: \"test\" with base dir:{}",
            std::env::current_dir().unwrap().display()
        );

        let dir = std::fs::read_dir("./test/").expect(&emsg);

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
                    .unwrap();
                // read content to string
                let content = std::fs::read_to_string(p).unwrap();
                (file_name, content)
            })
            .for_each(|(file, content)| {
                // execute parse
                assert!(Parser::parse(&content, Some(file)).is_ok())
            });
    }
}
