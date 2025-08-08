pub mod ast;
pub mod codegen;
pub mod ffi;
pub mod heap;
pub mod lexer;
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
    BadIntegerRepresentation { repr: String },
    UnclosedStringLiteral { literal: String },
    InvalidHexEscapeSequence { seq: String },
    InvalidUtf8EscapeSequence { seq: String },
    InvalidDecimalEscapeSequence { seq: String },

    // Parser Error
    UnexpectedToken { expect: Vec<Token>, found: Token },
    InvalidAttribute { attr: String },
    BadAssignment,
}

impl Display for SyntaxError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            SyntaxError::InvalidCharacter { ch } => write!(f, "invalid character {ch}"),
            SyntaxError::BadFloatRepresentation { repr } => {
                write!(f, "bad float representation {repr}")
            }
            SyntaxError::BadIntegerRepresentation { repr } => {
                write!(f, "bad integer representation {repr}")
            }
            SyntaxError::UnclosedStringLiteral { literal } => {
                let slice = if literal.len() >= 5 {
                    &literal[0..5]
                } else {
                    &literal[0..literal.len()]
                };
                write!(f, "unclosed string literal {}...", slice)
            }
            SyntaxError::InvalidHexEscapeSequence { seq } => {
                write!(f, "invalid hex escape sequence {seq}")
            }
            SyntaxError::InvalidUtf8EscapeSequence { seq } => {
                write!(f, "invalid utf8 escape sequence {seq}")
            }
            SyntaxError::InvalidDecimalEscapeSequence { seq } => {
                write!(f, "invalid decimal escape sequence {seq}")
            }
            SyntaxError::UnexpectedToken { expect, found } => {
                let pretty: Vec<_> = expect
                    .into_iter()
                    .map(|tk| {
                        if tk.is_ident() {
                            "Identifier".to_string()
                        } else {
                            format!("{:?}", tk)
                        }
                    })
                    .collect();
                write!(
                    f,
                    "token: {:?} was found, but {:?} was expect",
                    found, pretty
                )
            }
            SyntaxError::InvalidAttribute { attr } => write!(f, "invalid attribute: {}", attr),
            SyntaxError::BadAssignment => {
                write!(f, "bad assignment statement")
            }
        }
    }
}

#[derive(Debug)]
pub struct ParseError {
    pub kind: SyntaxError,
    pub line: u32,
    pub column: u32,
}

impl Display for ParseError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "{} at line: {}, column: {}",
            self.kind, self.line, self.column
        )
    }
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

pub enum InterpretError {
    IOErr(std::io::Error),

    /* static error (compile error ) */
    SyntaxErr(Box<ParseError>),

    CodeGenErr(Box<CodeGenError>),

    NotBinaryChunk,

    BinaryChunkVersionMismatch,

    UnsupportedBinaryChunkFormat,

    IncompatiablePlatform,

    /* dynamic error (runtime error) */
    RawError {
        msg: Box<String>,
    },

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

    ForeignModuleNotFound(Box<ModuleNotFound>),

    BadForeignModule(Box<BadModule>),

    AssertionFail {
        msg: Box<String>,
    },
}

impl Display for InterpretError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        use InterpretError::*;
        match self {
            IOErr(e) => writeln!(f, "IO error: {e}"),
            SyntaxErr(se) => writeln!(f, "Syntax error: {}", se),
            CodeGenErr(_) => todo!(),

            RawError { msg } => f.write_str(msg),

            RsCallDepthLimit { max } => {
                writeln!(f, "Too deep function call. (MAX_CALL_DEPTH: {max})")
            }

            StackOverflow => writeln!(
                f,
                "Stack over flow. (MAX_STACK_SPACE: {})",
                VM::MAX_STACK_SPACE
            ),

            InvalidRegisterAccess { target, max } => {
                writeln!(
                    f,
                    "Invalid register access: {target} (max available: {max})"
                )
            }

            BadTableIndex => {
                writeln!(f, "Table index is nil")
            }

            InvalidInvocation { callee } => {
                writeln!(f, "Try to call a non-callable object: {callee}")
            }

            ArgumentMismatch { expect, got } => {
                writeln!(f, "Wrong number of arguments, expected {expect}, got {got}")
            }

            AssertionFail { msg } => f.write_str(msg),

            ForeignModuleNotFound(info) => {
                writeln!(
                    f,
                    "Foreign module: {} not found in dir: {} and it's sub dirs.",
                    info.lib, info.path
                )
            }

            BadForeignModule(info) => {
                writeln!(
                    f,
                    "Entry {} not found in C extension: {}.",
                    info.entry, info.path
                )
            }

            _ => todo!(),
        }
    }
}

impl Debug for InterpretError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self)
    }
}

impl From<std::io::Error> for InterpretError {
    fn from(err: std::io::Error) -> Self {
        InterpretError::IOErr(err)
    }
}

pub struct ModuleNotFound {
    path: String,
    lib: String,
}

impl From<ModuleNotFound> for InterpretError {
    fn from(info: ModuleNotFound) -> Self {
        InterpretError::ForeignModuleNotFound(Box::new(info))
    }
}

pub struct BadModule {
    path: String,
    entry: String,
}

impl From<BadModule> for InterpretError {
    fn from(info: BadModule) -> Self {
        InterpretError::BadForeignModule(Box::new(info))
    }
}
