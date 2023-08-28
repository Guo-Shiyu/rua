pub mod ast;
pub mod codegen;
pub mod ffi;
pub mod lexer;
pub mod parser;
pub mod state;
pub mod value;

pub mod lstd;

use codegen::CodeGenErr;
use parser::SyntaxError;
use state::RuntimeErr;

#[derive(Debug)]
pub enum StaticErr {
    SyntaxErr(SyntaxError),
    CodeGenErr(CodeGenErr),
}

impl From<SyntaxError> for StaticErr {
    fn from(err: SyntaxError) -> Self {
        StaticErr::SyntaxErr(err)
    }
}

impl From<CodeGenErr> for StaticErr {
    fn from(err: CodeGenErr) -> Self {
        StaticErr::CodeGenErr(err)
    }
}

#[derive(Debug)]
pub enum LuaErr {
    IOErr(std::io::Error),
    CompileErr(StaticErr),
    RuntimeErr(RuntimeErr),
}

impl From<RuntimeErr> for LuaErr {
    fn from(err: RuntimeErr) -> Self {
        LuaErr::RuntimeErr(err)
    }
}
