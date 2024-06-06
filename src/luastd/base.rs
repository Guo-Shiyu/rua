use crate::{state::VM, value::RsFunc, InterpretError};

fn lua_print(vm: &mut VM) -> Result<usize, InterpretError> {
    let n = vm.top();
    for i in 1..=n {
        if i > 1 {
            print!(" ");
        }
        // SAFETY: we has checked the size of stack
        print!("{}", unsafe { vm.peek_unchecked(i) });
    }
    if n >= 1 {
        println!();
    }
    Ok(0)
}

fn lua_assert(vm: &mut VM) -> Result<usize, InterpretError> {
    if vm.top() >= 1 {
        // clear extra argument
        while vm.top() != 1 {
            vm.pop();
        }

        // take first variable to check
        let var = unsafe { vm.pop_unchecked() };
        if !var.is_falsey() {
            vm.push(true)?; // set true as return value
            Ok(1)
        } else {
            Err(InterpretError::AssertionFail) // raise an error
        }
    } else {
        Err(InterpretError::ArgumentMismatch { expect: 1, got: 0 })
    }
}

fn lua_error(_lua: &mut VM) -> Result<usize, InterpretError> {
    todo!();
}

fn lua_pcall(_lua: &mut VM) -> Result<usize, InterpretError> {
    todo!();
}

fn lua_xpcall(_lua: &mut VM) -> Result<usize, InterpretError> {
    todo!();
}

fn lua_getmetatable(_lua: &mut VM) -> Result<usize, InterpretError> {
    todo!();
}

fn lua_setmetatable(_lua: &mut VM) -> Result<usize, InterpretError> {
    todo!();
}

fn lua_next(_lua: &mut VM) -> Result<usize, InterpretError> {
    todo!();
}

fn lua_pairs(_lua: &mut VM) -> Result<usize, InterpretError> {
    todo!();
}

fn lua_ipairs(_lua: &mut VM) -> Result<usize, InterpretError> {
    todo!();
}

fn lua_load(_lua: &mut VM) -> Result<usize, InterpretError> {
    todo!();
}

fn lua_type(vm: &mut VM) -> Result<usize, InterpretError> {
    if vm.top() >= 1 {
        let var = unsafe { vm.pop_unchecked() };
        let ts = vm.new_str(var.typestr()); // typestr
        vm.push(ts)?;
        Ok(1)
    } else {
        Err(InterpretError::ArgumentMismatch { expect: 1, got: 0 })
    }
}

pub const BASE_LIBS: [(&str, RsFunc); 12] = [
    ("print", lua_print),
    ("assert", lua_assert),
    ("error", lua_error),
    ("pcall", lua_pcall),
    ("xpcall", lua_xpcall),
    ("getmetatable", lua_getmetatable),
    ("setmetatable", lua_setmetatable),
    ("next", lua_next),
    ("pairs", lua_pairs),
    ("ipairs", lua_ipairs),
    ("load", lua_load),
    ("type", lua_type),
];
