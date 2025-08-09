use std::{env, io::Write as _};

extern crate rua_core;
use rua_core::{InterpretError, state::VM};

fn eval(vm: &mut VM, line: &str) -> Result<(), InterpretError> {
    if line.trim().is_empty() {
        return Ok(());
    }

    vm.unsafe_script(line, None)?;
    Ok(())
}

fn main() -> Result<(), InterpretError> {
    let mut vm = VM::new();
    // vm.open(Stdlib::Base)?;
    if let Some(file) = env::args().nth(1) {
        vm.script_file(&file)?;
    } else {
        repl(vm)?
    }

    Ok(())
}

fn repl(mut vm: VM) -> Result<(), InterpretError> {
    println!("Rua REPL (Read-Eval-Print Loop)");
    println!("Type your Rua scripts below. Press Ctrl+C to exit.");
    const CHUNK: &str = "stdin";

    let mut line = String::with_capacity(128);
    loop {
        print!("> ");
        std::io::stdout().flush().unwrap();
        std::io::stdin().read_line(&mut line)?;

        let trimed = line.trim();
        if trimed.is_empty() {
            line.clear();
            continue;
        }

        let loaded = vm.load(&line, Some(CHUNK.to_string())).or_else(|_| {
            let as_expr = format!("return {line}");
            vm.load(&as_expr, Some(CHUNK.to_string()))
        });

        match loaded {
            // the expression was valid, so call the function on the top of stack.
            Ok(_) => {}

            // the expression was invalid, so eval the line as a statement.
            Err(err) => {
                println!("{err}");
            }
        };

        line.clear();
    }
}

#[cfg(test)]
mod test {
    use super::*;
    use rua_core::state::PanicFn;
    use std::path::PathBuf;

    #[test]
    fn hello_world() -> Result<(), InterpretError> {
        let mut vm = VM::new();
        // vm.open(Stdlib::Base)?;

        let src = r#"
            print "Hello Rua!"
        "#;

        let res = vm.unsafe_script(src, None);
        assert!(res.is_ok());
        vm.full_gc();

        Ok(())
    }

    #[test]
    fn test_rua_scripts() {
        let mut srcdir = std::env::current_dir().unwrap();
        srcdir.push(["..", "test", "rua"].iter().collect::<PathBuf>());
        let message = format!("Can not find test directory: {:?}. ", srcdir.clone());

        let mut src_paths = std::fs::read_dir(srcdir)
            .expect(&message)
            .map(|e| e.map(|e| e.path()))
            .collect::<Result<Vec<_>, _>>()
            .unwrap();

        src_paths.sort();

        src_paths
            .into_iter()
            .filter(|p| {
                // filter filename ends with '.lua'
                matches! { p.extension().map(|ex| ex.to_str().unwrap_or_default()), Some("lua")}
            })
            .for_each(|filepath| {
                let mut vm = VM::new();
                let res = vm.safe_script_file(filepath, None, Some(PanicFn::PANIC));
                assert!(res.is_ok());
            });
    }
}
