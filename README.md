# Rua   
An (under development) Lua 5.4 VM implemented in pure Rust.   

# Quick Start 
``` rust 
    let mut vm = State::new();
    vm.open(StdLib::Base);

    let src = r#"
        print "Hello Rua!"
    "#;
    
    let res = vm.script(src);
    assert!(res.is_ok());
```

# License    
MIT License
