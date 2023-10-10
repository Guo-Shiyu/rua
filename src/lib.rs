pub mod ast;
pub mod codegen;
pub mod ffi;
pub mod heap;
pub mod lexer;
pub mod lstd;
pub mod parser;
pub mod state;
pub mod value;

use std::fmt::Debug;

use codegen::CodeGenErr;
use parser::SyntaxError;
use state::RegIndex;
use value::LValue;

use crate::state::Rvm;

#[derive(Debug)]
pub enum StaticErr {
    SyntaxErr(Box<SyntaxError>),
    CodeGenErr(Box<CodeGenErr>),
}

impl From<SyntaxError> for StaticErr {
    fn from(err: SyntaxError) -> Self {
        StaticErr::SyntaxErr(Box::new(err))
    }
}

impl From<CodeGenErr> for StaticErr {
    fn from(err: CodeGenErr) -> Self {
        StaticErr::CodeGenErr(Box::new(err))
    }
}

pub trait RuntimeErr {
    fn reason(&self) -> String;
}

impl Debug for Box<dyn RuntimeErr> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(&self.reason())
    }
}

struct RecursionLimit();

impl RuntimeErr for RecursionLimit {
    fn reason(&self) -> String {
        format!(
            "Recursion limit reached. (MAX_CALL_DEPTH: {})",
            Rvm::MAX_CALL_DEPTH
        )
    }
}

struct StackOverflow();

impl RuntimeErr for StackOverflow {
    fn reason(&self) -> String {
        format!(
            "Stack overflow. (MAX_STACK_SPACE: {})",
            Rvm::MAX_STACK_SPACE
        )
    }
}

struct InvalidRegisterAccess {
    target: RegIndex, // target register to use
    max: RegIndex,    // max local register
}

impl RuntimeErr for InvalidRegisterAccess {
    fn reason(&self) -> String {
        format!(
            "Invalid register access: {} (max: {})",
            self.target, self.max
        )
    }
}

struct BadTableIndex();

impl RuntimeErr for BadTableIndex {
    fn reason(&self) -> String {
        "Table index is nil.".to_string()
    }
}

struct InvalidInvocation {
    callee: LValue,
}

impl RuntimeErr for InvalidInvocation {
    fn reason(&self) -> String {
        format!("Try to call a non-callable object: {}", self.callee)
    }
}

struct WrongNumberOfArgs {
    expected: u32,
    got: u32,
}

impl RuntimeErr for WrongNumberOfArgs {
    fn reason(&self) -> String {
        format!(
            "Wrong number of arguments: expected {}, got {}",
            self.expected, self.got
        )
    }
}

#[derive(Debug)]
pub enum RuaErr {
    IOErr(std::io::Error),
    CompileErr(StaticErr),
    RuntimeErr(Box<dyn RuntimeErr>),
}

impl From<StaticErr> for RuaErr {
    fn from(err: StaticErr) -> Self {
        RuaErr::CompileErr(err)
    }
}

impl From<Box<dyn RuntimeErr>> for RuaErr {
    fn from(err: Box<dyn RuntimeErr>) -> Self {
        RuaErr::RuntimeErr(err)
    }
}

impl From<std::io::Error> for RuaErr {
    fn from(err: std::io::Error) -> Self {
        RuaErr::IOErr(err)
    }
}
