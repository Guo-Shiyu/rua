pub mod ast;
pub mod codegen;
pub mod ffi;
pub mod heap;
pub mod lexer;
pub mod lstd;
pub mod parser;
pub mod passes;
pub mod state;
pub mod value;

use std::fmt::{Debug, Display};

use codegen::CodeGenErr;
use lexer::Token;
use state::RegIndex;
use value::LValue;

use crate::state::Rvm;

#[derive(Debug)]
pub enum SyntaxErr {
    // Tokenizer Error
    InvalidCharacter { ch: char },
    BadFloatRepresentation { repr: String },
    BadIntergerRepresentation { repr: String },
    UnclosedStringLiteral { literal: String },
    InvalidHexEscapeSequence { seq: String },
    InvalidUtf8EscapeSequence { seq: String },
    InvalidDecimalEscapeSequence { seq: String },

    // Parser Error
    UnexpectedToken { expect: Vec<Token>, found: Token },
    InvalidAttribute { attr: String },
}

#[derive(Debug)]
pub struct SyntaxError {
    pub kind: SyntaxErr,
    pub line: u32,
    pub column: u32,
}

impl From<Box<SyntaxError>> for RuaErr {
    fn from(err: Box<SyntaxError>) -> Self {
        RuaErr::SyntaxErr(err)
    }
}

impl From<CodeGenErr> for RuaErr {
    fn from(err: CodeGenErr) -> Self {
        RuaErr::CodeGenErr(Box::new(err))
    }
}

#[derive(Debug)]
pub enum RuaErr {
    IOErr(std::io::Error),

    // static error (compile error )
    SyntaxErr(Box<SyntaxError>),
    CodeGenErr(Box<CodeGenErr>),

    // dynamic error (runtime error)
    FuncDepthLimit,
    StackOverflow,
    InvalidRegisterAccess {
        target: RegIndex, // target register to access
        max: RegIndex,    // max local register number
    },
    BadTanleIndex,
    InvalidInvocation {
        callee: LValue,
    },
    ArgumentMismatch {
        expect: u32,
        got: u32,
    },
}

impl Display for RuaErr {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            RuaErr::IOErr(e) => writeln!(f, "IO error: {}", e),
            RuaErr::FuncDepthLimit => {
                writeln!(
                    f,
                    "Too deep function call. (MAX_CALL_DEPTH: {})",
                    Rvm::MAX_CALL_DEPTH
                )
            }
            RuaErr::StackOverflow => writeln!(
                f,
                "Stack over flow. (MAX_STACK_SPACE: {})",
                Rvm::MAX_STACK_SPACE
            ),
            RuaErr::InvalidRegisterAccess { target, max } => {
                writeln!(
                    f,
                    "Invalid register access: {} (max available: {})",
                    target, max
                )
            }
            RuaErr::BadTanleIndex => {
                writeln!(f, "Table index is nil")
            }
            RuaErr::InvalidInvocation { callee } => {
                writeln!(f, "Try to call a non-callable object: {}", callee)
            }
            RuaErr::ArgumentMismatch { expect, got } => {
                writeln!(
                    f,
                    "Wrong number of arguments.  expected {}, got {}",
                    expect, got
                )
            }
            RuaErr::SyntaxErr(_) => todo!(),
            RuaErr::CodeGenErr(_) => todo!(),
        }
    }
}

impl From<std::io::Error> for RuaErr {
    fn from(err: std::io::Error) -> Self {
        RuaErr::IOErr(err)
    }
}
