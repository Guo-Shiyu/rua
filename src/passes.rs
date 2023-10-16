#![allow(clippy::ptr_arg)]

use crate::ast::*;

/// AST walker. Each overridden visit method has full control over what
/// happens with its node, it can do its own traversal of the node's children,
/// call `walk_*` to apply the default traversal algorithm, or prevent
/// deeper traversal by doing nothing.

pub trait MutVisitor: Sized {
    fn visit_chunk(&mut self, chunk: &mut BasicBlock) {
        walk_chunk(self, chunk)
    }

    fn visit_return(&mut self, ret: &mut Option<Vec<ExprNode>>) {
        walk_return(self, ret)
    }

    fn visit_stmt(&mut self, stmt: &mut StmtNode) {
        walk_stmt(self, stmt)
    }

    fn visit_assign(&mut self, vars: &mut Vec<ExprNode>, exprs: &mut Vec<ExprNode>) {
        walk_assign(self, vars, exprs)
    }

    fn visit_fncall(&mut self, call: &mut FuncCall) {
        walk_fncall(self, call)
    }

    fn visit_argument_list(&mut self, arg: &mut SrcLoc<ArgumentList>) {
        walk_argument_list(self, arg)
    }

    fn visit_do_end(&mut self, chunk: &mut BasicBlock) {
        walk_do_end(self, chunk)
    }

    fn visit_while(&mut self, cond: &mut ExprNode, chunk: &mut BasicBlock) {
        walk_while(self, cond, chunk)
    }

    fn visit_repeat(&mut self, cond: &mut ExprNode, chunk: &mut BasicBlock) {
        walk_repeat(self, cond, chunk)
    }

    fn visit_if_else(
        &mut self,
        cond: &mut ExprNode,
        chunk: &mut BasicBlock,
        else_chunk: &mut Option<BasicBlock>,
    ) {
        walk_if_else(self, cond, chunk, else_chunk)
    }

    fn visit_numeric_for(
        &mut self,
        name: &mut SrcLoc<String>,
        init: &mut ExprNode,
        limit: &mut ExprNode,
        step: &mut ExprNode,
        chunk: &mut BasicBlock,
    ) {
        walk_numeric_for(self, name, init, limit, step, chunk)
    }

    fn visit_generic_for(
        &mut self,
        names: &mut Vec<SrcLoc<String>>,
        exprs: &mut Vec<ExprNode>,
        chunk: &mut BasicBlock,
    ) {
        walk_generic_for(self, names, exprs, chunk)
    }

    fn visit_fndef(
        &mut self,
        pres: &mut Vec<SrcLoc<String>>,
        method: &mut Option<Box<SrcLoc<String>>>,
        chunk: &mut Box<SrcLoc<FuncBody>>,
    ) {
        walk_fndef(self, pres, method, chunk)
    }

    fn visit_local_decl(
        &mut self,
        names: &mut Vec<(SrcLoc<String>, Option<Attribute>)>,
        exprs: &mut Vec<ExprNode>,
    ) {
        walk_local_decl(self, names, exprs)
    }

    fn visit_basic_block(&mut self, block: &mut BasicBlock) {
        walk_chunk(self, block)
    }

    fn visit_expr(&mut self, expr: &mut ExprNode) {
        walk_expr(self, expr)
    }
}

pub fn walk_chunk<T: MutVisitor>(vis: &mut T, chunk: &mut BasicBlock) {
    for stmt in chunk.stats.iter_mut() {
        vis.visit_stmt(stmt);
    }
    vis.visit_return(&mut chunk.ret);
}

pub fn walk_return<T: MutVisitor>(vis: &mut T, ret: &mut Option<Vec<ExprNode>>) {
    if let Some(ret) = ret {
        for expr in ret.iter_mut() {
            vis.visit_expr(expr);
        }
    }
}

pub fn walk_stmt<T: MutVisitor>(vis: &mut T, stmt: &mut StmtNode) {
    match stmt.inner_mut() {
        Stmt::Lable(_) => {}
        Stmt::Goto(_) => {}
        Stmt::Break => {}
        Stmt::Assign { vars, exprs } => vis.visit_assign(vars, exprs),
        Stmt::FnCall(call) => vis.visit_fncall(call),
        Stmt::DoEnd(chunk) => vis.visit_do_end(chunk),
        Stmt::While { exp, block } => vis.visit_while(exp, block),
        Stmt::Repeat { exp, block } => vis.visit_repeat(exp, block),
        Stmt::IfElse { cond, then, els } => {
            vis.visit_if_else(cond, then, els);
        }
        Stmt::NumericFor(num) => {
            vis.visit_numeric_for(
                &mut num.iter,
                &mut num.init,
                &mut num.limit,
                &mut num.step,
                &mut num.body,
            );
        }
        Stmt::GenericFor(gen) => {
            vis.visit_generic_for(&mut gen.iters, &mut gen.exprs, &mut gen.body);
        }
        Stmt::FnDef { pres, method, body } => {
            vis.visit_fndef(pres, method, body);
        }
        Stmt::LocalVarDecl { names, exprs } => {
            vis.visit_local_decl(names, exprs);
        }
        Stmt::Expr(exp) => vis.visit_expr(exp),
    }
}

pub fn walk_expr<T: MutVisitor>(vis: &mut T, expr: &mut ExprNode) {
    match expr.inner_mut() {
        Expr::Nil => {}
        Expr::False => {}
        Expr::True => {}
        Expr::Int(_) => {}
        Expr::Float(_) => {}
        Expr::Dots => {}
        Expr::Ident(id) => walk_ident(vis, id),
        Expr::Literal(l) => walk_literal(vis, l),
        Expr::Lambda(la) => walk_lambda(vis, la),
        Expr::Index { prefix, key } => walk_index(vis, prefix, key),
        Expr::FuncCall(call) => walk_fncall(vis, call),
        Expr::TableCtor(ctor) => walk_ctor(vis, ctor),
        Expr::BinaryOp { lhs, op, rhs } => walk_binary_op(vis, lhs, op, rhs),
        Expr::UnaryOp { op, expr } => walk_unary_op(vis, op, expr),
    }
}

pub fn walk_assign<T: MutVisitor>(
    vis: &mut T,
    vars: &mut Vec<ExprNode>,
    exprs: &mut Vec<ExprNode>,
) {
    for var in vars.iter_mut() {
        vis.visit_expr(var);
    }
    for expr in exprs.iter_mut() {
        vis.visit_expr(expr);
    }
}

pub fn walk_fncall<T: MutVisitor>(vis: &mut T, call: &mut FuncCall) {
    match call {
        FuncCall::FreeFnCall { prefix: _, args } => vis.visit_argument_list(args),
        FuncCall::MethodCall {
            prefix: _,
            method: _,
            args,
        } => vis.visit_argument_list(args),
    }
}

pub fn walk_argument_list<T: MutVisitor>(vis: &mut T, arg: &mut SrcLoc<ArgumentList>) {
    for expr in arg.namelist.iter_mut() {
        vis.visit_expr(expr);
    }
}

pub fn walk_do_end<T: MutVisitor>(vis: &mut T, chunk: &mut BasicBlock) {
    vis.visit_chunk(chunk);
}

pub fn walk_while<T: MutVisitor>(vis: &mut T, cond: &mut ExprNode, chunk: &mut BasicBlock) {
    vis.visit_expr(cond);
    vis.visit_chunk(chunk);
}

pub fn walk_repeat<T: MutVisitor>(vis: &mut T, cond: &mut ExprNode, chunk: &mut BasicBlock) {
    vis.visit_expr(cond);
    vis.visit_chunk(chunk);
}

pub fn walk_if_else<T: MutVisitor>(
    vis: &mut T,
    cond: &mut ExprNode,
    chunk: &mut BasicBlock,
    else_chunk: &mut Option<BasicBlock>,
) {
    vis.visit_expr(cond);
    vis.visit_chunk(chunk);
    if let Some(els_chunk) = else_chunk {
        vis.visit_chunk(els_chunk);
    }
}

pub fn walk_numeric_for<T: MutVisitor>(
    vis: &mut T,
    _name: &mut String,
    init: &mut ExprNode,
    limit: &mut ExprNode,
    step: &mut ExprNode,
    chunk: &mut BasicBlock,
) {
    vis.visit_expr(init);
    vis.visit_expr(limit);
    vis.visit_expr(step);
    vis.visit_chunk(chunk);
}

pub fn walk_generic_for<T: MutVisitor>(
    vis: &mut T,
    _names: &mut Vec<SrcLoc<String>>,
    exprs: &mut Vec<ExprNode>,
    chunk: &mut BasicBlock,
) {
    for expr in exprs.iter_mut() {
        vis.visit_expr(expr);
    }

    vis.visit_chunk(chunk);
}

pub fn walk_fndef<T: MutVisitor>(
    vis: &mut T,
    _pres: &mut Vec<SrcLoc<String>>,
    _method: &mut Option<Box<SrcLoc<String>>>,
    body: &mut Box<SrcLoc<FuncBody>>,
) {
    vis.visit_chunk(&mut body.body);
}

pub fn walk_local_decl<T: MutVisitor>(
    vis: &mut T,
    _names: &mut Vec<(SrcLoc<String>, Option<Attribute>)>,
    exprs: &mut Vec<ExprNode>,
) {
    for expr in exprs.iter_mut() {
        vis.visit_expr(expr);
    }
}

pub fn walk_literal<T: MutVisitor>(_vis: &mut T, _lit: &mut String) {}

pub fn walk_ident<T: MutVisitor>(_vis: &mut T, _id: &mut String) {}

pub fn walk_lambda<T: MutVisitor>(vis: &mut T, la: &mut FuncBody) {
    vis.visit_chunk(&mut la.body);
}

pub fn walk_index<T: MutVisitor>(vis: &mut T, prefix: &mut ExprNode, key: &mut ExprNode) {
    vis.visit_expr(prefix);
    vis.visit_expr(key);
}

pub fn walk_ctor<T: MutVisitor>(vis: &mut T, ctor: &mut Vec<Field>) {
    for field in ctor.iter_mut() {
        walk_field(vis, field);
    }
}

pub fn walk_field<T: MutVisitor>(vis: &mut T, field: &mut Field) {
    walk_expr(vis, &mut field.val)
}

pub fn walk_binary_op<T: MutVisitor>(
    vis: &mut T,
    lhs: &mut ExprNode,
    _op: &mut BinOp,
    rhs: &mut ExprNode,
) {
    vis.visit_expr(lhs);
    vis.visit_expr(rhs);
}

pub fn walk_unary_op<T: MutVisitor>(vis: &mut T, _op: &mut UnOp, expr: &mut ExprNode) {
    vis.visit_expr(expr);
}

pub fn constant_fold(root: &mut BasicBlock) {
    let mut cf = ConstantFolder::default();
    cf.visit_chunk(root);
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

#[derive(Default)]
struct ConstantFolder {}

fn try_fold(exp: &mut Expr) -> AfterFoldStatus {
    use AfterFoldStatus::*;
    match exp {
        Expr::Nil | Expr::False | Expr::True | Expr::Int(_) | Expr::Float(_) | Expr::Literal(_) => {
            StillConst
        }

        Expr::BinaryOp { lhs, op, rhs } => {
            let ls = try_fold(lhs);
            let rs = try_fold(rhs);
            match (ls, rs) {
                (StillConst, StillConst) => {
                    let (mut i1, mut i2) = (0, 0);
                    // intergral promotion
                    if let Some(promoted) = match (lhs.inner_ref(), rhs.inner_ref()) {
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
                        if let Some(fop) = gen_arithmetic_op_float(*op) {
                            *exp = apply_arithmetic_op_float(promoted.0, promoted.1, fop);
                            StillConst
                        } else {
                            NonConst
                        }
                    } else if let Some(iop) = gen_arithmetic_op_int(*op) {
                        if i2 == 0 {
                            if *op == BinOp::Div || *op == BinOp::IDiv {
                                *exp = Expr::Float(match i1.cmp(&0) {
                                    std::cmp::Ordering::Less => f64::NEG_INFINITY,
                                    std::cmp::Ordering::Equal => f64::NAN,
                                    std::cmp::Ordering::Greater => f64::INFINITY,
                                });
                            } else if *op == BinOp::Mod {
                                // do nothing. perform mod 0 is a runtime error in lua.
                            }
                        } else {
                            *exp = apply_arithmetic_op_int(i1, i2, iop);
                        }
                        StillConst
                    } else {
                        match (op, lhs.inner_ref(), rhs.inner_ref()) {
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
            if let StillConst = try_fold(expr) {
                // execute fold operation
                if let Some(new_exp) = match (op, expr.inner_ref()) {
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
        BinOp::Mod => Some(|l, r| l % r),
        BinOp::Pow => Some(|l, r| l ^ r),
        BinOp::IDiv => Some(|l, r| l / r),
        BinOp::Div => Some(|l, r| l / r),
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

impl MutVisitor for ConstantFolder {
    fn visit_assign(&mut self, _vars: &mut Vec<ExprNode>, exprs: &mut Vec<ExprNode>) {
        for expr in exprs.iter_mut() {
            try_fold(expr);
        }
    }

    fn visit_argument_list(&mut self, arg: &mut SrcLoc<ArgumentList>) {
        for expr in arg.namelist.iter_mut() {
            try_fold(expr);
        }
    }

    fn visit_while(&mut self, cond: &mut ExprNode, chunk: &mut BasicBlock) {
        try_fold(cond);
        walk_chunk(self, chunk);
    }

    fn visit_repeat(&mut self, cond: &mut ExprNode, chunk: &mut BasicBlock) {
        try_fold(cond);
        walk_chunk(self, chunk);
    }

    fn visit_if_else(
        &mut self,
        cond: &mut ExprNode,
        chunk: &mut BasicBlock,
        else_chunk: &mut Option<BasicBlock>,
    ) {
        try_fold(cond);
        walk_chunk(self, chunk);
        if let Some(els_chunk) = else_chunk {
            walk_chunk(self, els_chunk);
        }
    }

    fn visit_numeric_for(
        &mut self,
        _name: &mut SrcLoc<String>,
        init: &mut ExprNode,
        limit: &mut ExprNode,
        step: &mut ExprNode,
        chunk: &mut BasicBlock,
    ) {
        try_fold(init);
        try_fold(limit);
        try_fold(step);
        walk_chunk(self, chunk);
    }

    fn visit_fndef(
        &mut self,
        _pres: &mut Vec<SrcLoc<String>>,
        _method: &mut Option<Box<SrcLoc<String>>>,
        chunk: &mut Box<SrcLoc<FuncBody>>,
    ) {
        walk_chunk(self, &mut chunk.body);
    }

    fn visit_local_decl(
        &mut self,
        _names: &mut Vec<(SrcLoc<String>, Option<Attribute>)>,
        exprs: &mut Vec<ExprNode>,
    ) {
        for expr in exprs.iter_mut() {
            try_fold(expr);
        }
    }

    fn visit_expr(&mut self, expr: &mut ExprNode) {
        try_fold(expr);
    }
}

mod test {

    #[test]
    fn constant_fold_exec_test() {
        use crate::parser::Parser;
        use crate::passes::ConstantFolder;
        use crate::passes::MutVisitor;

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
                    .unwrap_or_default();
                // read content to string
                let content = std::fs::read_to_string(p).unwrap_or_default();
                (file_name, content)
            })
            .flat_map(|(file, content)| {
                // execute parse
                Parser::parse(&content, Some(file))
            })
            .for_each(|mut block| {
                // execute constant fold
                let mut cf = ConstantFolder::default();
                cf.visit_chunk(&mut block);
            });
    }
}
