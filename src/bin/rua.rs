use std::env;

use rua::{ffi::StdLib, state::State, LuaErr};

fn main() -> Result<(), LuaErr> {
    let mut vm = State::with_libs(&[StdLib::Base]);
    if let Some(file) = env::args().nth(1) {
        vm.script_file(&file)?;
    } else {
        let hello = r#"
            print "Hello Rua!"
        "#;

        vm.script(hello)?;
    }
    Ok(())
}

mod test {

    #[test]
    fn hello_world() {
        use rua::ffi::StdLib;
        use rua::state::State;

        let mut vm = State::new();
        vm.open(StdLib::Base);

        let src = r#"
            print "Hello Rua!"
        "#;

        let res = vm.script(src);
        assert!(res.is_ok());
        assert_eq!(res.unwrap(), 0);
    }

    #[test]
    fn stack_op() {
        // use rua::state::State;
        // let mut vm = State::new();
        // vm.with_stack(|stk| stk.push(LValue::default()));
    }
}
