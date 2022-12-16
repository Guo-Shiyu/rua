use crate::compiler::token::Token;

pub mod compiler;

fn main() {
    test();
}

fn test() {
    let src = std::fs::read_to_string("testes/strings.lua").expect("can't open test source file");

    let mut lex = compiler::lexer::Lexer::new(&src);
    loop {
        let some = lex
            .next()
            .map(|tk| {
                println!(
                    "{:?} \t\t line: {} column: {}",
                    tk,
                    lex.line(),
                    lex.column()
                );
                tk
            })
            .map_err(|e| {
                println!(
                    "tokenize error: {:?} line: {} column: {}",
                    e,
                    lex.line(),
                    lex.column()
                );
                assert!(false);
                ()
            })
            .unwrap();

        match some {
            Token::Eof => break,
            _ => continue,
        }
    }
}

fn add(x: usize) -> usize {
    x + 1
}
