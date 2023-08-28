fn main() {
    println!("Hello, world!")
}

mod test {
    use rua::{ffi::StdLib, state::State, value::LValue};

    #[test]
    fn hello_world() {
        let mut vm = State::new();
        vm.open(StdLib::Base);

        let src = r##"
            print "Hello Rua!"
        "##;
        assert!(vm.script(src).is_ok());
    }

    #[test]
    fn stack_op() {
        let mut vm = State::new();
        // vm.with_stack(|stk| stk.push(LValue::default()));

        vm.stack(|s, h| {
            // s.pop();
            // h.
        });
    }
}
