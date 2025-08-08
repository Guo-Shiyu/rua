extern crate rua_core;
extern crate rua_std_decl;
use rua_std_decl::ruastd;

#[ruastd]
mod base {
    use rua_core::{state::VM, value::Value, InterpretError};

    pub fn print(vm: &mut VM) -> Result<usize, InterpretError> {
        let n = vm.top();
        for i in 1..=n {
            if i > 1 {
                print!(" ");
            }
            // SAFETY: we has checked the size of stack
            print!("{}", vm.peek_unchecked(i));
        }
        if n >= 1 {
            println!();
        }
        Ok(0)
    }

    pub fn assert(vm: &mut VM) -> Result<usize, InterpretError> {
        if vm.top() < 1 {
            return Err(InterpretError::ArgumentMismatch { expect: 1, got: 0 });
        }

        // clear extra argument
        while vm.top() > 2 {
            vm.pop();
        }

        // take first variable to check
        let first = vm.peek_unchecked(1);
        if first.is_falsey() {
            let errmsg = if vm.top() == 2 {
                let errobj = unsafe { vm.pop().unwrap_unchecked() };
                match errobj {
                    Value::Str(s) => s.to_string(),
                    num if num.is_number() => num.to_string(),
                    other => format!("error object is a {} value", other.typestr()),
                }
            } else {
                "assertion failed!".to_string()
            };
            Err(InterpretError::AssertionFail {
                msg: Box::new(errmsg),
            })
        } else {
            Ok(vm.top() as usize)
        }
    }

    pub fn error(_lua: &mut VM) -> Result<usize, InterpretError> {
        todo!()
    }

    pub fn pcall(_lua: &mut VM) -> Result<usize, InterpretError> {
        todo!();
    }

    pub fn xpcall(_lua: &mut VM) -> Result<usize, InterpretError> {
        todo!();
    }

    pub fn getmetatable(_lua: &mut VM) -> Result<usize, InterpretError> {
        todo!();
    }

    pub fn setmetatable(_lua: &mut VM) -> Result<usize, InterpretError> {
        todo!();
    }

    pub fn next(_lua: &mut VM) -> Result<usize, InterpretError> {
        todo!();
    }

    pub fn pairs(_lua: &mut VM) -> Result<usize, InterpretError> {
        todo!();
    }

    pub fn ipairs(_lua: &mut VM) -> Result<usize, InterpretError> {
        todo!();
    }

    pub fn load(_lua: &mut VM) -> Result<usize, InterpretError> {
        todo!();
    }

    pub fn type_(vm: &mut VM) -> Result<usize, InterpretError> {
        if vm.top() >= 1 {
            let var = vm.pop_unchecked();
            let ts = vm.new_str(var.typestr());
            vm.push(ts)?;
            Ok(1)
        } else {
            Err(InterpretError::ArgumentMismatch { expect: 1, got: 0 })
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Test `ruastd` macro here to avoid introduce dependency to `stddecl`.
    #[test]
    fn empty_foreign_module() {
        #[ruastd]
        mod empty {}
    }

    #[test]
    fn show_entry_info() {
        let mut vm: VM = VM::new();

        assert!(luaopen_base(&mut vm) == __rua_base_num);
        let mut genv = vm.genv().as_table().unwrap();
        use std::ops::Deref;
        println!("{:?}", genv.deref());

        // type is keyword in rust, which is a function name in lua
        let key = vm.new_str("type");
        assert!(genv.index(key).is_rsfn());

        for (k, v) in genv.iter() {
            assert!(k.is_str());
            assert!(v.is_rsfn());
        }
    }
}
