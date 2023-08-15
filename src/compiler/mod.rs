pub mod ast;
pub mod codegen;
pub mod lexer;
pub mod parser;
pub mod scanner;
pub mod token;

#[derive(Debug)]
pub enum CompileErr {
    IOErr(std::io::Error),
    InternalErr,
    SyntaxErr(parser::SyntaxError),
}

impl CompileErr {
    pub fn internal() -> Self {
        CompileErr::InternalErr
    }
}
