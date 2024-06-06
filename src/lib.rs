pub mod ast;
pub mod codegen;
pub mod ffi;
pub mod heap;
pub mod lexer;
pub mod luastd;
pub mod parser;
pub mod passes;
pub mod state;
pub mod value;

use std::fmt::{Debug, Display};

use codegen::{BinLoadErr, CodeGenError};
use lexer::Token;
use state::RegIndex;
use value::Value;

use crate::state::VM;

#[derive(Debug)]
pub enum SyntaxError {
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
    BadAssignment,
}

#[derive(Debug)]
pub struct ParseError {
    pub kind: SyntaxError,
    pub line: u32,
    pub column: u32,
}

impl From<Box<ParseError>> for InterpretError {
    fn from(err: Box<ParseError>) -> Self {
        InterpretError::SyntaxErr(err)
    }
}

impl From<CodeGenError> for InterpretError {
    fn from(err: CodeGenError) -> Self {
        InterpretError::CodeGenErr(Box::new(err))
    }
}

impl From<BinLoadErr> for InterpretError {
    fn from(value: BinLoadErr) -> Self {
        match value {
            BinLoadErr::IOErr(e) => Self::IOErr(e),
            BinLoadErr::NotBinaryChunk => Self::NotBinaryChunk,
            BinLoadErr::VersionMismatch => Self::BinaryChunkVersionMismatch,
            BinLoadErr::UnsupportedFormat => Self::UnsupportedBinaryChunkFormat,
            BinLoadErr::IncompatiablePlatform => Self::IncompatiablePlatform,
        }
    }
}

#[derive(Debug)]
pub enum InterpretError {
    IOErr(std::io::Error),

    // static error (compile error )
    SyntaxErr(Box<ParseError>),

    CodeGenErr(Box<CodeGenError>),

    NotBinaryChunk,

    BinaryChunkVersionMismatch,

    UnsupportedBinaryChunkFormat,

    IncompatiablePlatform,

    // dynamic error (runtime error)
    RsCallDepthLimit {
        max: u32,
    },

    StackOverflow,

    InvalidRegisterAccess {
        target: RegIndex, // target register to access
        max: RegIndex,    // max local register number
    },

    // table index is nil
    BadTableIndex,

    // try to call a value which is not callable
    InvalidInvocation {
        callee: Value,
    },

    // argument check for rs function
    ArgumentMismatch {
        expect: u8,
        got: u8,
    },

    AssertionFail,
}

impl Display for InterpretError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        use InterpretError::*;
        match self {
            IOErr(e) => writeln!(f, "IO error: {}", e),
            SyntaxErr(_) => todo!(),
            CodeGenErr(_) => todo!(),

            RsCallDepthLimit { max } => {
                writeln!(f, "Too deep function call. (MAX_CALL_DEPTH: {})", max)
            }

            StackOverflow => writeln!(
                f,
                "Stack over flow. (MAX_STACK_SPACE: {})",
                VM::MAX_STACK_SPACE
            ),

            InvalidRegisterAccess { target, max } => {
                writeln!(
                    f,
                    "Invalid register access: {} (max available: {})",
                    target, max
                )
            }

            BadTableIndex => {
                writeln!(f, "Table index is nil")
            }

            InvalidInvocation { callee } => {
                writeln!(f, "Try to call a non-callable object: {}", callee)
            }

            ArgumentMismatch { expect, got } => {
                writeln!(
                    f,
                    "Wrong number of arguments, expected {}, got {}",
                    expect, got
                )
            }

            AssertionFail => {
                writeln!(f, "Assertion failed!")
            }
            _ => todo!(),
        }
    }
}

impl From<std::io::Error> for InterpretError {
    fn from(err: std::io::Error) -> Self {
        InterpretError::IOErr(err)
    }
}
