use rua::{ffi::StdLib, state::State, LuaErr};

fn main() -> Result<(), LuaErr> {
    let mut vm = State::new();
    vm.open(StdLib::Base);
    let src = r#" print 'Hello Rua!' "#;
    let _res = vm.script(src)?;
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
