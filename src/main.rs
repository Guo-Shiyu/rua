use std::fs;

use crate::compiler::token::Token;

pub mod compiler;

fn main() {
    let dir = fs::read_dir("./testes/").unwrap();
    let mut paths = dir
        .map(|e| e.map(|e| e.path()))
        .collect::<Result<Vec<_>, _>>()
        .unwrap();

    paths.sort();

    paths.iter().for_each(|p| {
        if let Some("lua") = p.extension().map(|ex| ex.to_str().unwrap_or("")) {
            test(&p)
        }
    });
}

fn test(path: &std::path::PathBuf) {
    let src = std::fs::read_to_string(path)
        .map_err(|e| println!("error: {} while read src file: {:?}", e, path))
        .unwrap();

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
                    "tokenize error: {:?} line: {} column: {} file: {:?}",
                    e,
                    lex.line(),
                    lex.column(),
                    path
                );
                assert!(false);
                e
            });
        if let Ok(Token::Eof) = some {
            break;
        }
    }
}