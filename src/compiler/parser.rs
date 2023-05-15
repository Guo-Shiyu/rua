use super::{
    ast::*,
    lexer::{Lexer, LexicalErr},
    token::Token,
};

#[derive(Debug)]
pub struct SyntaxError {
    reason: String,
    line: u32,
    column: u32,
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


/// Full syntax of Lua 5.4 : /doc/lua54 syntax.ebnf
pub struct Parser<'a> {
    lex: Lexer<'a>,
    ahead: Token,
}

impl Parser<'_> {
    /// Motivate lexer to next token and update self.ahead
    fn next(&mut self) -> Result<(), SyntaxError> {
        self.lex
            .next()
            .map(|tk| self.ahead = tk)
            .map_err(|e| self.lexerr(e))
    }

    /// block ::= {stat} [retstat]
    fn block(&mut self) -> Result<Block, SyntaxError> {
        const DEFAULT_BLOCK_SIZE: usize = 64;
        let mut stats = Vec::with_capacity(DEFAULT_BLOCK_SIZE);

        loop {
            if self.is_block_end() {
                stats.shrink_to(stats.len());
                break;
            }

            stats.push(self.stmt()?);

            // ';' after each statement is optional
            self.test_and_next(Token::Colon)?;
        }

        // optional return statement
        let ret = if self.test(Token::Return) {
            self.next()?;
            Some(self.retstmt()?)
        } else {
            None
        };

        // Top level block end at the end of file
        // self.check(Token::Eof)?;

        Ok(Block::new(stats, ret))
    }

    fn is_block_end(&self) -> bool {
        matches!(
            self.ahead,
            Token::Eof | Token::Else | Token::Elseif | Token::Until | Token::End
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
        loop {
            if self.test_and_next(Token::Colon)? == false {
                break;
            }
        }

        let begin = self.lex.line();

        let stmt = {
            match &self.ahead {
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
                    self.next()?;
                    if self.test_and_next(Token::Function).is_ok() {
                        self.local_fndef()?
                    } else {
                        self.local_assign()?
                    }
                }
                ///// TODO
                _ => {
                    todo!()
                }
            }
        };

        let end = self.lex.line();
        Ok(StmtNode::new(stmt, (begin, end)))
    }

    fn lable(&mut self) -> Result<Stmt, SyntaxError> {
        // debug_assert!(self.ahead == Token::Follow);
        self.next()?;

        let ret = match &self.ahead {
            Token::Ident(id) => Ok(Stmt::Lable(id.to_string())),
            _ => Err(self.unexpected("::")),
        };

        if ret.is_ok() {
            self.check_and_next(Token::Follow)?;
        }
        ret
    }

    fn goto(&mut self) -> Result<Stmt, SyntaxError> {
        self.next()?; // skip `goto`

        let ret = match &self.ahead {
            Token::Ident(lable) => Ok(Stmt::Goto(lable.to_string())),
            _ => Err(self.unexpected("ident")),
        };

        if ret.is_ok() {
            self.next()?;
        };
        ret
    }

    fn do_end(&mut self) -> Result<Stmt, SyntaxError> {
        // debug_assert!(self.ahead == Token::Do);
        self.next()?;
        let block = self.block()?;
        self.test_and_next(Token::End)?;
        Ok(Stmt::DoEnd(block))
    }

    fn while_do(&mut self) -> Result<Stmt, SyntaxError> {
        // debug_assert!(self.ahead == Token::While);
        self.next()?;
        let exp = self.expr()?;

        self.check_and_next(Token::Do)?;
        let block = self.block()?;

        Ok(Stmt::While {
            exp: Box::new(exp),
            block: Box::new(block),
        })
    }

    fn repeat_until(&mut self) -> Result<Stmt, SyntaxError> {
        // debug_assert!(self.ahead == Token::Repeat);
        self.next()?;
        let block = self.block()?;

        self.test_and_next(Token::Until)?;
        let exp = self.expr()?;

        Ok(Stmt::Repeat {
            block: block,
            exp: Box::new(exp),
        })
    }

    fn if_else(&mut self) -> Result<Stmt, SyntaxError> {
        todo!()
    }

    fn for_stmt(&mut self) -> Result<Stmt, SyntaxError> {
        // for
        self.next()?;

        if let Token::Ident(first) = self.ahead.clone() {
            self.next()?;
            let stmt = match &self.ahead {
                // gerneric for
                Token::Comma => {
                    self.next()?;
                    let mut names = Vec::with_capacity(4);
                    names.push(first);

                    loop {
                        match &self.ahead {
                            Token::Ident(id) => names.push(id.clone()),
                            Token::Comma => self.next()?,
                            Token::In => break,
                            _ => return Err(self.unexpected("'<Ident>' or 'in'")),
                        }
                    }

                    // in
                    self.next()?;

                    let mut exprs = Vec::with_capacity(4);
                    loop {
                        match &self.ahead {
                            Token::Comma => self.next()?,
                            Token::Do => break,
                            _ => exprs.push(self.expr()?.inner()),
                        }
                    }

                    // do
                    self.next()?;

                    let body = self.block()?;

                    Stmt::GenericFor {
                        names,
                        exprs,
                        body: Box::new(body),
                    }
                }

                // numeric for
                Token::Eq => {
                    self.next()?;
                    let init = self.expr()?.inner();
                    self.check_and_next(Token::Comma)?;
                    let limit = self.expr()?.inner();

                    let step = match &self.ahead {
                        Token::Comma => {
                            self.next()?;
                            self.expr()?.inner()
                        }
                        _ => Expr::Int(1),
                    };

                    let body = self.block()?;

                    Stmt::NumericFor {
                        name: first,
                        init: Box::new(init),
                        limit: Box::new(limit),
                        step: Box::new(step),
                        body: Box::new(body),
                    }
                }
                _ => return Err(self.unexpected("'=' or ','")),
            };
            Ok(stmt)
        } else {
            return Err(self.unexpected("Ident"));
        }
    }

    fn global_fndef(&mut self) -> Result<Stmt, SyntaxError> {
        self.next()?;
        let namelist = self.funcname()?;
        let def = self.func_body()?;
        Ok(Stmt::FnDef {
            namelist,
            def: Box::new(def),
        })
    }

    fn local_fndef(&mut self) -> Result<Stmt, SyntaxError> {
        if let Token::Ident(fnname) = self.ahead.clone() {
            let begin = self.lex.line();
            let fndef = self.func_body()?;
            let end = self.lex.line();
            let node = ExprNode::new(Expr::FuncDefine(fndef), (begin, end));

            Ok(Stmt::LocalVarDecl {
                names: vec![fnname],
                exprs: vec![node],
            })
        } else {
            Err(self.unexpected("ident"))
        }
    }

    fn local_assign(&mut self) -> Result<Stmt, SyntaxError> {
        let mut names = Vec::with_capacity(4);
        loop {
            match &self.ahead {
                Token::Ident(id) => names.push(id.to_string()),
                Token::Comma => self.next()?,
                Token::Assign => break,
                _ => {
                    return Err(self.error(format!(
                        "Ident, ',' or '=' are expected but {:?} was found",
                        self.ahead
                    )))
                }
            }
        }

        let count = names.len();
        let mut exprs = Vec::with_capacity(count);
        for _ in 0..count {
            exprs.push(self.expr()?);
        }

        Ok(Stmt::LocalVarDecl {
            names: names,
            exprs: exprs,
        })
    }

    fn funcname(&mut self) -> Result<FuncName, SyntaxError> {
        let mut pres = Vec::with_capacity(4);
        let mut method = None;

        let mut is_follow = false;
        loop {
            match &self.ahead {
                Token::Ident(id) => {
                    if is_follow {
                        method = Some(id.to_string());
                        break;
                    } else {
                        pres.push(id.to_string());
                    }
                }
                Token::Dot => {}
                Token::Follow => is_follow = true,
                _ => break,
            }
        }

        Ok(FuncName::new(pres, method))
    }

    fn func_body(&mut self) -> Result<FuncBody, SyntaxError> {
        Ok(FuncBody::new(self.param_list()?, Box::new(self.block()?)))
    }

    fn param_list(&mut self) -> Result<ParaList, SyntaxError> {
        self.check_and_next(Token::LP)?;

        let mut namelist = Vec::with_capacity(4);
        let mut vargs = false;

        loop {
            match &self.ahead {
                Token::Ident(id) => namelist.push(id.to_string()),
                Token::Comma => self.next()?,
                Token::Dots => vargs = true,
                Token::RP => break,
                _ => {
                    return Err(self.error(format!(
                        "Ident, ',' or '...' are expected but {:?} was found",
                        self.ahead
                    )))
                }
            }
        }

        self.check_and_next(Token::RP)?;

        Ok(ParaList::new(vargs, namelist))
    }

    fn retstmt(&mut self) -> Result<Vec<ExprNode>, SyntaxError> {
        let mut exprs = Vec::with_capacity(4);
        exprs.push(self.expr()?);

        loop {
            if self.test_and_next(Token::Comma).is_ok() {
                exprs.push(self.expr()?);
            } else {
                break;
            }
        }

        self.test_and_next(Token::Colon)?;
        Ok(exprs)
    }

    fn expr(&mut self) -> Result<ExprNode, SyntaxError> {
        let begin = self.lex.line();

        if let Some(exp) = self.unary_expr()?.or(self.simple_expr()?) {
            let end = self.lex.line();
            return Ok(ExprNode::new(exp, (begin, end)));
        };

        let exp = match &self.ahead {
            // functiondef
            Token::Function => {
                self.next()?;
                Expr::FuncDefine(self.func_body()?)
            }

            Token::LB => Expr::TableCtor(self.table_constructor()?),

            Token::LP => {
                self.next()?;
                let lhs = self.expr()?;
                self.check_and_next(Token::RP)?;
                let op = self.bin_op()?;
                if op.is_none() {
                    return Err(self.unexpected("binary operator"));
                };

                let rhs = self.expr()?;
                Expr::BinaryOp {
                    lhs: Box::new(lhs),
                    op: op.unwrap(),
                    rhs: Box::new(rhs),
                }
            }

            Token::Ident(first) => {
                // FIXME:
                //
                let mut names = Vec::with_capacity(5);
                names.push(first);

                loop {
                    self.next()?;
                    match self.ahead {
                        // offset index expr
                        Token::LS => {}

                        // name index expr
                        Token::Dot => {} // func
                        _ => todo!(),
                    }
                }
            }

            _ => todo!(),
        };

        let end = self.lex.line();
        Ok(ExprNode::new(exp, (begin, end)))
    }

    fn simple_expr(&mut self) -> Result<Option<Expr>, SyntaxError> {
        let exp = match &self.ahead {
            Token::Nil => Some(Expr::Nil),
            Token::False => Some(Expr::False),
            Token::True => Some(Expr::True),
            Token::Dots => Some(Expr::Dots),
            Token::Integer(i) => Some(Expr::Int(*i)),
            Token::Float(f) => Some(Expr::Float(*f)),
            Token::Literal(s) => Some(Expr::Literal(s.to_string())),
            _ => None,
        };
        if exp.is_some() {
            self.next()?;
        };
        Ok(exp)
    }

    fn unary_expr(&mut self) -> Result<Option<Expr>, SyntaxError> {
        let op = match self.ahead {
            // -
            Token::Minus => Some(UnOp::Minus),

            // not
            Token::Not => Some(UnOp::Not),

            // #
            Token::Len => Some(UnOp::Length),

            // ~
            Token::BitXor => Some(UnOp::NoUnary),

            _ => None,
        };

        if op.is_some() {
            self.next()?;
            let expr = self.expr()?;
            Ok(Some(Expr::UnaryOp {
                op: op.unwrap(),
                expr: Box::new(expr),
            }))
        } else {
            Ok(None)
        }
    }

    fn bin_op(&mut self) -> Result<Option<BinOp>, SyntaxError> {
        let op = match self.ahead {
            Token::Add => Some(BinOp::Add),
            Token::Minus => Some(BinOp::Minus),
            Token::Mul => Some(BinOp::Mul),
            Token::Div => Some(BinOp::Div),
            Token::Mod => Some(BinOp::Mod),
            Token::Pow => Some(BinOp::Pow),
            Token::BitAnd => Some(BinOp::BitAnd),
            Token::BitXor => Some(BinOp::BitXor),
            Token::BitOr => Some(BinOp::BitOr),
            Token::Shl => Some(BinOp::Shl),
            Token::Shr => Some(BinOp::Shr),
            Token::IDiv => Some(BinOp::IDiv),
            Token::Eq => Some(BinOp::Eq),
            Token::Neq => Some(BinOp::Neq),
            Token::LE => Some(BinOp::LE),
            Token::GE => Some(BinOp::GE),
            Token::Less => Some(BinOp::Less),
            Token::Great => Some(BinOp::Great),
            Token::Concat => Some(BinOp::Concat),
            _ => None,
        };

        if op.is_some() {
            self.next()?;
        };
        Ok(op)
    }

    fn table_constructor(&mut self) -> Result<Vec<Field>, SyntaxError> {
        // skip '{'
        self.next()?;

        let mut fieldlist = Vec::with_capacity(8);

        loop {
            match &self.ahead {
                Token::Comma => self.next()?,
                Token::RB => break,
                _ => fieldlist.push(self.field()?),
            }
        }

        self.next()?;

        Ok(fieldlist)
    }

    fn field(&mut self) -> Result<Field, SyntaxError> {
        let field = match self.ahead.clone() {
            // [ <expr> ] = expr
            Token::LB => {
                self.next()?;
                let key = self.expr()?;
                self.next()?;
                self.check_and_next(Token::Eq)?;
                Field::new(Some(Box::new(key)), Box::new(self.expr()?))
            }

            // Name = expr
            Token::Ident(key) => {
                self.check_and_next(Token::Eq)?;
                let line = self.lex.line();
                let key = ExprNode::new(Expr::Literal(key), (line, line));
                Field::new(Some(Box::new(key)), Box::new(self.expr()?))
            }

            // expr
            _ => Field::new(None, Box::new(self.expr()?)),
        };
        Ok(field)
    }

    fn test(&self, expect: Token) -> bool {
        self.ahead == expect
    }

    fn check(&self, expect: Token) -> Result<(), SyntaxError> {
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
            "unexpected symbol: <{}> is expected but {:?} found",
            expect, self.ahead
        ))
    }

    fn lexerr(&self, le: LexicalErr) -> SyntaxError {
        self.error(le.reason)
    }
}

impl Parser<'_> {
    pub fn parse(src: &str) -> Result<Block, SyntaxError> {
        let mut parser = Parser {
            lex: Lexer::new(src),
            ahead: Token::Eof,
        };

        // init parser state to first token
        parser.next()?;

        parser.block()
    }
}
