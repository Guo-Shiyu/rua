#![allow(clippy::module_inception)]

extern crate rua_core;
extern crate rua_std_macro;
use rua_std_macro::rua_std_decl;

#[rua_std_decl]
mod string {
    use rua_core::{InterpretError, heap::MetaOperator, state::VM, value::Value};

    pub fn find(vm: &mut VM) -> Result<usize, InterpretError> {
        Ok(0)
    }

    pub fn gsub(vm: &mut VM) -> Result<usize, InterpretError> {
        Ok(0)
    }
}
