use std::{collections::HashMap, fs};

use compiler::{parser, CompileErr};

use crate::compiler::token::Token;

pub mod compiler;

fn main() {
    // parser::Parser::parse("123").unwrap();
    perf();
}

fn perf() {
    let dir = fs::read_dir("./testes/").expect("unable to find testes firectory!");
    let mut paths = dir
        .map(|e| e.map(|e| e.path()))
        .collect::<Result<Vec<_>, _>>()
        .unwrap();

    paths.sort();

    let srcs = paths
        .into_iter()
        // file name end with '.lua'
        .filter(
            |p| matches! { p.extension().map(|ex| ex.to_str().unwrap_or_default()), Some("lua")},
        )
        // read to string
        .filter_map(|p| std::fs::read_to_string(p).ok())
        .collect::<Vec<_>>();

    let total_bytes = srcs.iter().map(|s| s.len()).sum::<usize>();

    let repeat = 1;

    let total_ms = cost_ms(|| {
        for _ in 0..repeat {
            for s in srcs.iter() {
                let _ = luac_lexer_test(s);
            }
        }
    });

    let mb = (repeat * total_bytes / 3) as f64 / (1024 * 1024) as f64;
    let sec = total_ms as f64 / 1000.0;

    println!(
        "{} mb, {} ms, \naverage : {} mb / s",
        mb,
        total_ms,
        mb / sec
    );
}

fn cost_ms(f: impl Fn()) -> usize {
    let begin = std::time::Instant::now();
    f();
    let end = std::time::Instant::now();
    (end - begin).as_millis() as usize
}

fn luac_lexer_test(src: &str) -> Result<(), CompileErr> {
    let mut lex = compiler::lexer::Lexer::new(&src);
    let mut vec = Vec::with_capacity(1024 * 1024);
    loop {
        match lex.next() {
            Ok(Token::Eof) => break,
            Ok(tk) => vec.push(tk),
            Err(e) => println!("error: {:?}, line: {}, col:{}", e, lex.line(), lex.column()),
        }
    }
    Ok(())
}


mod test {
    #[test]
    fn t() {
        let s = "128";
        let u = u8::from_str_radix(s, 10);
        assert!(u.is_ok());
    }
}