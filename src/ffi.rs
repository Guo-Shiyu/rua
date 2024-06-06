use crate::state::VM;

#[derive(Clone, Copy, PartialEq)]
pub enum Stdlib {
    Base,
    Package,
    Coroutine,
    String,
    Table,
    Math,
    IO,
    OS,
    Debug,
    Bit32,
    Utf8,
    // Ffi,
    // Jit,
    All,
}

pub const fn get_std_libs(lib: Stdlib) -> &'static [&'static str] {
    use Stdlib::*;
    match lib {
        Base => &["base"],
        Package => todo!(),
        Coroutine => todo!(),
        String => &["string"],
        Table => &["table"],
        Math => todo!(),
        IO => todo!(),
        OS => todo!(),
        Debug => todo!(),
        Bit32 => todo!(),
        Utf8 => todo!(),
        All => todo!(),
    }
}

pub fn open_lib(vm: &mut VM, modname: &str) -> u32 {
    open_lib_posix_impl(vm, modname)
}

use std::ffi::CString;
use std::os::raw::{c_char, c_void};

extern "C" {
    fn dlopen(filename: *const c_char, flags: i32) -> *mut c_void;
    fn dlsym(handle: *mut c_void, symbol: *const c_char) -> *mut c_void;
    // fn dlclose(handle: *mut std::ffi::c_void) -> std::os::raw::c_int;
}

fn open_lib_posix_impl(vm: &mut VM, modname: &str) -> u32 {
    let handle = {
        let dll = format!("lib{}.so", modname);
        let cname = CString::new(dll).expect("CString::new failed");
        unsafe { dlopen(cname.as_ptr(), 2) } // 2: RTLD_NOW,  1: RTLD_LAZY
    };
    if handle.is_null() {
        panic!("Failed to open the dynamic library");
    }

    let entry_sym = {
        let symbol = format!("luaopen_{}", modname);
        let cname = CString::new(symbol).unwrap();
        unsafe { dlsym(handle, cname.as_ptr()) }
    };
    if entry_sym.is_null() {
        panic!("Failed to get the function pointer");
    }

    type CdylibEntry = extern "C" fn(&mut VM) -> u32;
    let dllentry: CdylibEntry = unsafe { std::mem::transmute(entry_sym) };
    dllentry(vm)

    // unsafe { dlclose(handle) };
}
