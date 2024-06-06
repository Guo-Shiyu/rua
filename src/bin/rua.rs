use std::env;

use rua::{ffi::Stdlib, state::VM, InterpretError};

fn main() -> Result<(), InterpretError> {
    let mut vm = VM::with_libs(&[Stdlib::Base]);
    if let Some(file) = env::args().nth(1) {
        vm.script_file(&file)?;
        Ok(())
    } else {
        let hello = r#"
            print "Hello Rua!"
        "#;

        vm.script(hello, None);
        Ok(())
    }
}

#[cfg(test)]
mod test {
    use super::*;

    #[test]
    fn hello_world() {
        let mut vm = VM::new();
        vm.open(Stdlib::Base);

        let src = r#"
            print "Hello Rua! 1"
            print("Hello Rua! 2")
            print("Hello Rua! 3", "Hello Rua! 4", "Hello Rua! 5")
        "#;

        let res = vm.unsafe_script(src, None);
        // dbg!(&res);
        assert!(res.is_ok());
        vm.full_gc();
        // println!("# after gc");
    }

    #[test]
    fn test_rua_scripts() {
        let srcdir = "./test/rua/";
        let emsg = format!(
            "unable to find directory: {} with base dir:{}",
            srcdir,
            std::env::current_dir().unwrap().display()
        );

        let dir = std::fs::read_dir(srcdir).expect(&emsg);

        let mut src_paths = dir
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
                let mut vm = VM::with_libs(&[Stdlib::Base]);
                let res = vm.script_file(filepath);
                if res.is_err() {
                    dbg!(res.as_ref().err().unwrap());
                }
                assert!(res.is_ok());
            });
    }
}
