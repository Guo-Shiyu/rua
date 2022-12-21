use std::fs;

use compiler::CompileErr;

use crate::compiler::token::Token;

pub mod compiler;

fn main() {
    let dir = fs::read_dir("./testes/").expect("unable to find testes firectory!");
    let mut paths = dir
        .map(|e| e.map(|e| e.path()))
        .collect::<Result<Vec<_>, _>>()
        .unwrap();

    paths.sort();

    paths.iter().for_each(|p| {
        if let Some("lua") = p.extension().map(|ex| ex.to_str().unwrap_or_default()) {
            luac_load(&p)
        }
    });
}

fn luac_load(path: &std::path::PathBuf) {
    let _ = std::fs::read_to_string(path)
        .map_err(|e| {
            println!(
                "Load test file error: {} while reading from : {:?}",
                e, path
            );
            CompileErr::internal()
        })
        .and_then(|src| luac_test(src))
        .map_err(|e| {
            println!("luc compile error: {:?}", e);
            panic!()
        });
}

fn luac_test(src: String) -> Result<(), CompileErr> {
    let mut lex = compiler::lexer::Lexer::new(&src);
    loop {
        let some = lex
            .next()
            .map(|tk| {
                // println!("{:<16?} line: {}, column:{}", tk, lex.line(), lex.column());
                tk
            })
            .map_err(|e| {
                println!(
                    "tokenize error: {:?} line: {} column: {}",
                    e.reason(),
                    lex.line(),
                    lex.column(),
                );
                e
            });
            
        if let Ok(Token::Eof) = some {
            break Ok(());
        };
    }
}
