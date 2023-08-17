use super::{
    ast::*,
    lexer::{Lexer, LexicalError},
    token::Token,
};

#[derive(Debug)]
pub struct SyntaxError {
    pub reason: String,
    pub line: u32,
    pub column: u32,
}

impl SyntaxError {
    fn new(state: &Parser, reason: String) -> Self {
        SyntaxError {
            reason,
            line: state.lex.line(),
            column: state.lex.column(),
        }
    }
}

/// An LL(1) parser for Lua 5.4       
///
/// Full syntax of [Lua 5.4](https://github.com/Guo-Shiyu/luac-rs/blob/a84ea7bbcfcc05028865f2ea04cf294d94acdb40/doc/lua54%20syntax.ebnf)
/// can be find here.
pub struct Parser<'a> {
    lex: Lexer<'a>,
    current: Token,
    ahead: Token,
}

impl Parser<'_> {
    /// Motivate lexer to next token and update self.current.
    /// This will clear self.ahead if self.ahead is't EOF.
    fn next(&mut self) -> Result<(), SyntaxError> {
        if let Token::Eof = self.ahead {
            self.current = self.lex.pump().map_err(|e| self.lexerr(e))?;
        } else {
            self.current = std::mem::replace(&mut self.ahead, Token::Eof);
        }
        Ok(())
    }

    /// Motivate lexer to next token and update self.ahead
    fn look_ahead(&mut self) -> Result<(), SyntaxError> {
        debug_assert!(self.ahead == Token::Eof);
        self.ahead = self.lex.pump().map_err(|e| self.lexerr(e))?;
        Ok(())
    }

    /// block ::= {stat} [retstat]
    fn block(&mut self) -> Result<Block, SyntaxError> {
        const DEFAULT_BLOCK_SIZE: usize = 64;
        let mut stats = Vec::with_capacity(DEFAULT_BLOCK_SIZE);

        loop {
            // ';' after each statement is optional
            while self.test_and_next(Token::Semi)? {}

            if self.is_block_end() {
                stats.shrink_to(stats.len());
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

        Ok(Block::new(stats, ret))
    }

    fn is_block_end(&self) -> bool {
        matches!(
            self.current,
            Token::Eof | Token::Else | Token::Elseif | Token::Until | Token::End | Token::Return
        )
    }

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
    fn stmt(&mut self) -> Result<StmtNode, SyntaxError> {
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
                _ => self.expr_stat()?,
            }
        };

        let end = self.lex.line();
        Ok(StmtNode::new(stmt, (begin, end)))
    }

    /// single expr as statememt|
    fn expr_stat(&mut self) -> Result<Stmt, SyntaxError> {
        let expr = self.expr()?;
        let (lineinfo, inner) = (expr.lineinfo(), expr.into_inner());

        let stmt = if let Expr::FuncCall(call) = inner {
            Stmt::FuncCall(call)
        } else {
            let first = ExprNode::new(inner, lineinfo);
            if self.test(Token::Comma) {
                // multi assignment
                let varlist = self.exprlist(Some(first))?;
                self.check_and_next(Token::Assign)?;
                let exprlist = self.exprlist(None)?;
                Stmt::Assign {
                    vars: varlist,
                    exprs: exprlist,
                }
            } else if self.test_and_next(Token::Assign)? {
                // single assignment
                let exprlist = self.exprlist(None)?;
                Stmt::Assign {
                    vars: vec![first],
                    exprs: exprlist,
                }
            } else {
                // single expr as statememt
                Stmt::Expr(Box::new(first.into_inner()))
            }
        };
        Ok(stmt)
    }

    /// label ::= `::` Name `::`
    fn lable(&mut self) -> Result<Stmt, SyntaxError> {
        self.next()?;

        let ret = match &mut self.current {
            Token::Ident(id) => {
                let inner = std::mem::take(id);
                Ok(Stmt::Lable(inner))
            }
            _ => Err(self.unexpected("::")),
        };
        if ret.is_ok() {
            self.next()?; // skip lable
            self.check_and_next(Token::Follow)?;
        }
        ret
    }

    /// goto ::= `goto` Name
    fn goto(&mut self) -> Result<Stmt, SyntaxError> {
        self.next()?; // skip `goto`

        let mut holder = String::new();
        if let Token::Ident(lable) = &mut self.current {
            std::mem::swap(&mut holder, lable);
        } else {
            return Err(self.unexpected("identifier"));
        };

        Ok(Stmt::Goto(holder))
    }

    /// do ::= `do` block `end`
    fn do_end(&mut self) -> Result<Stmt, SyntaxError> {
        self.next()?; // skip 'do'
        let block = self.block()?;
        self.test_and_next(Token::End)?;
        Ok(Stmt::DoEnd(Box::new(block)))
    }

    /// while ::= `while` exp `do` block `end`
    fn while_do(&mut self) -> Result<Stmt, SyntaxError> {
        self.next()?; // skip 'while'
        let exp = self.expr()?;
        self.check_and_next(Token::Do)?;
        let block = self.block()?;
        self.check_and_next(Token::End)?;

        Ok(Stmt::While {
            exp: Box::new(exp),
            block: Box::new(block),
        })
    }

    /// repeat ::= `repeat` block `until` exp
    fn repeat_until(&mut self) -> Result<Stmt, SyntaxError> {
        // debug_assert!(self.ahead == Token::Repeat);
        self.next()?;
        let block = self.block()?;

        self.test_and_next(Token::Until)?;
        let exp = self.expr()?;

        Ok(Stmt::Repeat {
            block: Box::new(block),
            exp: Box::new(exp),
        })
    }

    /// if ::= `if` exp `then` block {`elseif` exp `then` block} [`else` block] `end`
    fn if_else(&mut self) -> Result<Stmt, SyntaxError> {
        self.next()?; // skip 'if'
        let cond = self.expr()?;
        self.check_and_next(Token::Then)?;

        let then = self.block()?;

        // if-else-end
        if self.test_and_next(Token::Else)? {
            let els = self.block()?;
            self.check_and_next(Token::End)?;
            Ok(Stmt::IfElse {
                exp: Box::new(cond),
                then: Box::new(then),
                els: Box::new(els),
            })
        } else
        // if-elseif-...-end
        if self.test(Token::Elseif) {
            let begin = self.lex.line();
            let else_stmt = self.if_else()?;
            let end = self.lex.line();

            let node = WithSrcLoc::new(else_stmt, (begin, end));
            let els = Block::new(vec![node], None);
            Ok(Stmt::IfElse {
                exp: Box::new(cond),
                then: Box::new(then),
                els: Box::new(els),
            })
        }
        // if-then-end
        else {
            self.check_and_next(Token::End)?;
            let empty = Block::new(vec![], None);
            Ok(Stmt::IfElse {
                exp: Box::new(cond),
                then: Box::new(then),
                els: Box::new(empty),
            })
        }
    }

    /// numerical_for ::= `for` Name `=` exp `,` exp [`,` exp] `do` block `end`
    /// generic_for ::= `for` namelist `in` explist `do` block `end`
    fn for_stmt(&mut self) -> Result<Stmt, SyntaxError> {
        self.next()?; // skip `for`

        let mut holder = String::new();
        if let Token::Ident(first) = &mut self.current {
            std::mem::swap(&mut holder, first);
        } else {
            return Err(self.unexpected("identifier"));
        };

        self.next()?;
        let stmt = match &self.current {
            Token::Comma | Token::In => self.generic_for(holder)?,
            Token::Assign => self.numeric_for(holder)?,
            _ => return Err(self.unexpected("'=' or ','")),
        };

        Ok(stmt)
    }

    /// generic_for ::= `for` namelist `in` explist `do` block `end`    
    fn generic_for(&mut self, first: String) -> Result<Stmt, SyntaxError> {
        // "for k, v in ..." is most common senario
        let mut names = Vec::with_capacity(2);
        names.push(first);

        loop {
            match &mut self.current {
                Token::Ident(id) => {
                    names.push(std::mem::take(id));
                    self.next()?;
                }
                Token::Comma => self.next()?,
                Token::In => break,
                _ => return Err(self.unexpected("identifier or 'in'")),
            }
        }
        self.check_and_next(Token::In)?;
        let mut exprs = Vec::with_capacity(4);
        loop {
            match &self.current {
                Token::Comma => self.next()?,
                Token::Do => break,
                _ => exprs.push(self.expr()?.into_inner()),
            }
        }
        self.check_and_next(Token::Do)?;
        let body = self.block()?;
        self.check_and_next(Token::End)?;
        Ok(Stmt::GenericFor {
            iters: names,
            exprs,
            body: Box::new(body),
        })
    }

    /// numerical_for ::= `for` Name `=` exp `,` exp [`,` exp] `do` block `end`
    fn numeric_for(&mut self, var: String) -> Result<Stmt, SyntaxError> {
        self.next()?;
        let init = self.expr()?.into_inner();
        self.check_and_next(Token::Comma)?;
        let limit = self.expr()?.into_inner();
        let step = match &self.current {
            Token::Comma => {
                self.next()?;
                self.expr()?.into_inner()
            }
            _ => Expr::Int(1),
        };
        self.check_and_next(Token::Do)?;
        let body = self.block()?;
        self.check_and_next(Token::End)?;
        Ok(Stmt::NumericFor {
            iter: var,
            init: Box::new(init),
            limit: Box::new(limit),
            step: Box::new(step),
            body: Box::new(body),
        })
    }

    /// function ::= `function` funcname funcbody
    /// funcname ::= Name {`.` Name} [`:` Name]
    fn global_fndef(&mut self) -> Result<Stmt, SyntaxError> {
        self.next()?; // skip `function`
        let (pres, method) = self.func_name()?;
        let body = self.func_body()?;
        Ok(Stmt::FnDef {
            pres,
            method,
            body: Box::new(body),
        })
    }

    /// local_function ::= `local` `function` Name funcbody
    fn local_fndef(&mut self) -> Result<Stmt, SyntaxError> {
        let mut func_name = String::new();
        if let Token::Ident(name) = &mut self.current {
            std::mem::swap(&mut func_name, name);
        } else {
            return Err(self.unexpected("identifier"));
        };

        self.next()?; // eat func name
        let begin = self.lex.line();
        let fndef = self.func_body()?;
        let end = self.lex.line();
        let node = ExprNode::new(Expr::FuncDefine(fndef), (begin, end));

        Ok(Stmt::LocalVarDecl {
            names: vec![(func_name, None)],
            exprs: vec![node],
        })
    }

    /// local_assign ::= `local` attnamelist [`=` explist]
    /// attnamelist ::=  Name attr {`,` Name attr }
    fn local_assign(&mut self) -> Result<Stmt, SyntaxError> {
        let mut names = Vec::with_capacity(4);
        loop {
            let (mut decl_holder, mut is_id) = (String::new(), false);
            if let Token::Ident(id) = &mut self.current {
                std::mem::swap(id, &mut decl_holder);
                is_id = true;
            };

            if is_id {
                self.next()?;

                // optional attribute
                let attribute = if self.test_and_next(Token::Less)? {
                    let mut attr_holder = String::new();
                    if let Token::Ident(id) = &mut self.current {
                        std::mem::swap(id, &mut attr_holder);
                    } else {
                        return Err(self.unexpected("attribute"));
                    };

                    let attr = match attr_holder.as_str() {
                        "const" => Some(Attribute::Const),
                        "close" => Some(Attribute::Close),
                        other => return Err(self.error(format!("invalid attribute '{}'", other))),
                    };

                    self.next()?; // skip attribute
                    self.check_and_next(Token::Great)?;

                    attr
                } else {
                    None
                };

                names.push((decl_holder, attribute));
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

    /// funcname ::= Name {`.` Name} [`:` Name]
    fn func_name(&mut self) -> Result<(Vec<String>, Option<String>), SyntaxError> {
        const DOT: bool = true;
        const COLON: bool = false;
        let mut pres = Vec::with_capacity(4);
        let mut method = None;

        let mut dot_or_colon = DOT;
        loop {
            let mut is_id = false;
            match &mut self.current {
                Token::Ident(id) => {
                    if dot_or_colon == COLON {
                        method = Some(std::mem::take(id));
                        is_id = true;
                    } else {
                        pres.push(std::mem::take(id));
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

        Ok((pres, method))
    }

    /// funcbody ::= `(` [parlist] `)` block `end`
    fn func_body(&mut self) -> Result<FuncBody, SyntaxError> {
        self.check_and_next(Token::LP)?;
        let params = self.param_list()?;
        self.check_and_next(Token::RP)?;
        let body = self.block()?;
        self.check_and_next(Token::End)?;

        Ok(FuncBody::new(params, Box::new(body)))
    }

    /// paralist ::= namelist [`,` `...`] | `...`
    /// namelist ::= Name {`,` Name}
    fn param_list(&mut self) -> Result<ParaList, SyntaxError> {
        let mut vargs = false;

        if let Token::RP = self.current {
            return Ok(ParaList::new(vargs, Vec::new()));
        }

        let paras = self.exprlist(None)?;

        match &self.current {
            Token::Dots => {
                vargs = true;
                self.next()?;
            }

            Token::RP => {}

            _ => {
                return Err(self.error(format!(
                    "'...' or ')' are expected but {:?} was found",
                    self.current
                )))
            }
        };

        let paras = paras.into_iter().map(|node| node.into_inner()).collect();
        Ok(ParaList::new(vargs, paras))
    }

    /// retstat ::= return [explist] [`;`]
    fn return_stmt(&mut self) -> Result<Option<Vec<ExprNode>>, SyntaxError> {
        if self.is_block_end() || self.test_and_next(Token::Semi)? {
            Ok(None)
        } else {
            let exprs = self.exprlist(None)?;
            self.test_and_next(Token::Semi)?;
            Ok(Some(exprs))
        }
    }

    /// priority table for binary operators
    #[rustfmt::skip]
    const BINARY_PRIORITY: [(usize, usize); 21] = [
        (10, 10), (10, 10),      /* + -  */
        (11, 11), (11, 11),      /* * %  */
        (14, 13),                /* ^ (right associatie) */
        (11, 11), (11, 11),      /* / //  */
        (6, 6), (4, 4), (5, 5),  /* & | ~ */
        (7, 7), (7, 7),          /* << >> */
        (9, 8),                  /* .. (right associatie) */
        (3, 3), (3, 3), (3, 3),  /* ==, <, <= */
        (3, 3), (3, 3), (3, 3),  /* ~=, >, >= */
        (2, 2), (1, 1)           /* and, or   */
    ];

    const UNARY_PRIORITY: usize = 12;

    fn expr(&mut self) -> Result<ExprNode, SyntaxError> {
        self.subexpr(0)
    }

    /// exp ::=  `nil` |
    ///	   `false` |
    ///	   `true` |
    ///	   Numeral |
    ///	   LiteralString |
    ///	   `...` |
    ///	   functiondef |
    ///	   prefixexp |
    ///	   tablector |
    ///	   exp binop exp |
    ///	   unop exp
    fn subexpr(&mut self, limit: usize) -> Result<ExprNode, SyntaxError> {
        let begin = self.lex.line();
        let mut lhs = if let Some(uniop) = self.unary_op() {
            self.next()?;
            let sub = self.subexpr(Self::UNARY_PRIORITY)?;
            let end = self.lex.line();
            ExprNode::new(
                Expr::UnaryOp {
                    op: uniop,
                    expr: Box::new(sub),
                },
                (begin, end),
            )
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
                lhs = ExprNode::new(
                    Expr::BinaryOp {
                        lhs: Box::new(lhs),
                        op: binop,
                        rhs: Box::new(rhs),
                    },
                    (begin, end),
                );
            } else {
                break;
            }
        }
        Ok(lhs)
    }

    /// simpleexp -> Float | Int | String | Nil | `true` | `false` | `...` |
    ///             Constructor | `function` funcbody | suffixedexp
    fn simple_expr(&mut self) -> Result<ExprNode, SyntaxError> {
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
            return Ok(ExprNode::new(exp, line_info));
        };

        let begin = self.lex.line();
        match &self.current {
            Token::Function => {
                self.next()?;
                let body = self.func_body()?;
                Ok(ExprNode::new(
                    Expr::FuncDefine(body),
                    (begin, self.lex.line()),
                ))
            }
            Token::LB => Ok(ExprNode::new(
                Expr::TableCtor(self.table_constructor()?),
                (begin, self.lex.line()),
            )),

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

    /// exprlist ::= expr {`,` expr}
    fn exprlist(&mut self, first: Option<ExprNode>) -> Result<Vec<ExprNode>, SyntaxError> {
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

    /// tablector ::= `{` [fieldlist] `}`
    /// fieldlist ::= field {fieldsep field} [fieldsep]
    /// fieldsep ::= `,` | `;`
    fn table_constructor(&mut self) -> Result<Vec<Field>, SyntaxError> {
        self.next()?; // skip '{'

        let mut fieldlist = Vec::with_capacity(4);
        loop {
            match &self.current {
                Token::Comma | Token::Semi => self.next()?,
                Token::RB => break,
                _ => fieldlist.push(self.field()?),
            }
        }

        self.next()?;
        Ok(fieldlist)
    }

    /// field ::= `[` exp `]` `=` exp | Name `=` exp | exp
    fn field(&mut self) -> Result<Field, SyntaxError> {
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
            let expr = ExprNode::new(Expr::Ident(holder), (line, line));
            Ok(Field::new(Some(Box::new(expr)), Box::new(self.expr()?)))
        } else {
            // expr
            Ok(Field::new(None, Box::new(self.expr()?)))
        }
    }

    /// primaryexp ::= NAME | '(' expr ')'
    fn primary_expr(&mut self) -> Result<ExprNode, SyntaxError> {
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

            _ => return Err(self.unexpected("identifier or '('")),
        };

        let begin = self.lex.line();
        self.next()?;
        Ok(ExprNode::new(Expr::Ident(holder), (begin, self.lex.line())))
    }

    /// suffixedexp ->
    ///      primaryexp { '.' NAME | '[' exp ']' | ':' NAME funcargs | funcargs }
    fn suffixed_expr(&mut self) -> Result<ExprNode, SyntaxError> {
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
                        return Err(self.unexpected("identifier"));
                    };

                    let attr = Expr::Literal(holder);
                    self.next()?;
                    ExprNode::new(
                        Expr::Index {
                            prefix: Box::new(prefix),
                            key: Box::new(attr),
                        },
                        (begin, self.lex.line()),
                    )
                }

                Token::LS => {
                    self.next()?; // skip '['
                    let key = self.expr()?;
                    self.check_and_next(Token::RS)?;

                    ExprNode::new(
                        Expr::Index {
                            prefix: Box::new(prefix),
                            key: Box::new(key.into_inner()),
                        },
                        (begin, self.lex.line()),
                    )
                }

                // method call
                Token::Colon => {
                    self.next()?;
                    let mut holder = String::new();
                    if let Token::Ident(method) = &mut self.current {
                        std::mem::swap(method, &mut holder);
                    } else {
                        return Err(self.unexpected("identifier"));
                    };

                    self.next()?; // skip name

                    let args = self.func_args()?;
                    ExprNode::new(
                        Expr::FuncCall(FuncCall::MethodCall {
                            prefix: Box::new(prefix),
                            method: holder,
                            args,
                        }),
                        (begin, self.lex.line()),
                    )
                }

                Token::Literal(_) | Token::LB | Token::LP => ExprNode::new(
                    Expr::FuncCall(FuncCall::FreeFnCall {
                        prefix: Box::new(prefix),
                        args: self.func_args()?,
                    }),
                    (begin, self.lex.line()),
                ),

                _ => return Ok(prefix),
            };

            prefix = expr_node;
        }
    }

    /// args ::=  `(` [explist] `)` | tablector | LiteralString
    fn func_args(&mut self) -> Result<ParaList, SyntaxError> {
        let mut holder = String::new();
        match &mut self.current {
            Token::LP => {
                self.next()?; // skip '('
                let args = self.param_list()?;
                self.check_and_next(Token::RP)?;
                return Ok(args);
            }

            Token::LB => {
                let table = self.table_constructor()?;
                return Ok(ParaList::new(false, vec![Expr::TableCtor(table)]));
            }

            Token::Literal(s) => {
                std::mem::swap(s, &mut holder);
            }

            _ => return Err(self.unexpected("function arguments")),
        };
        self.next()?; // skip string literal

        Ok(ParaList::new(false, vec![Expr::Literal(holder)]))
    }

    fn test(&self, expect: Token) -> bool {
        self.current == expect
    }

    fn check(&self, expect: Token) -> Result<(), SyntaxError> {
        // this `clone()` will never deep clone a String object, it's always cheap
        if self.test(expect.clone()) {
            Ok(())
        } else {
            Err(self.unexpected(format!("{:?}", expect).as_str()))
        }
    }

    /// Perhaps self.ahead is equal to expect, if true, call `next`
    fn test_and_next(&mut self, expect: Token) -> Result<bool, SyntaxError> {
        if self.test(expect) {
            self.next()?;
            Ok(true)
        } else {
            Ok(false)
        }
    }

    /// Must that self.ahead is equal to expect, if true, call next
    fn check_and_next(&mut self, expect: Token) -> Result<(), SyntaxError> {
        self.check(expect)?;
        self.next()
    }
}

impl Parser<'_> {
    fn error(&self, r: String) -> SyntaxError {
        SyntaxError::new(self, r)
    }

    fn unexpected(&self, expect: &str) -> SyntaxError {
        self.error(format!(
            "Unexpected symbol, {expect} is expected but {:?} was found",
            self.current
        ))
    }

    fn lexerr(&self, le: LexicalError) -> SyntaxError {
        self.error(le.reason)
    }
}

impl Parser<'_> {
    pub fn parse(src: &str, chunkname: String) -> Result<Block, SyntaxError> {
        let mut parser = Parser {
            lex: Lexer::new(src),
            current: Token::Eof,
            ahead: Token::Eof,
        };

        // init parser state to first token
        parser.next()?;

        // parse statements
        let block = parser.block()?;

        debug_assert!(block.chunk.as_str() == Block::NAMELESS_CHUNK);
        debug_assert!(parser.current.is_eof());
        debug_assert!(parser.ahead.is_eof());

        Ok(Block::chunk(chunkname, block.stats, block.ret))
    }
}

mod test {
    #[test]
    fn parser_all_testes() {
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
            .for_each(|(file, content)| {
                // execute parse
                assert!(Parser::parse(&content, file).is_ok())
            });
    }
}
