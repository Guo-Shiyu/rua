use crate::state::VM;
use crate::InterpretError;

use std::ffi::CString;
use std::os::raw::{c_char, c_void};
use std::path::{Path, PathBuf};

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

pub fn open_lib(vm: &mut VM, modname: &str) -> Result<u32, InterpretError> {
    open_lib_posix_impl(vm, modname)
}

extern "C" {
    fn dlopen(filename: *const c_char, flags: i32) -> *mut c_void;
    fn dlsym(handle: *mut c_void, symbol: *const c_char) -> *mut c_void;
    // fn dlclose(handle: *mut std::ffi::c_void) -> std::os::raw::c_int;
}

fn find_dylib_recursive(dir: &Path, target: &str) -> Option<PathBuf> {
    if let Ok(entries) = std::fs::read_dir(dir) {
        for entry in entries.flatten() {
            let path = entry.path();
            if path.is_dir() {
                let rec = find_dylib_recursive(&path, target);
                if rec.is_some() {
                    return rec;
                }
            } else if let Some(name) = path.file_name() {
                if name == target {
                    return Some(path);
                }
            }
        }
    }
    None
}

fn open_lib_posix_impl(vm: &mut VM, modname: &str) -> Result<u32, InterpretError> {
    // search dynamic library in current dir recursively.
    let target = format!("lib{}.so", modname);
    let curdir = std::env::current_dir()?;

    // TODO: detect environment variable LUA_PATH
    match find_dylib_recursive(&curdir, &target) {
        None => Err(InterpretError::ForeignModuleNotFound {
            path: Box::new(
                curdir
                    .into_os_string()
                    .into_string()
                    .expect("CString::into_string failed"),
            ),
        }),
        Some(dllpath) => {
            // execute `dlopen` and get handle of dylib
            let handle = {
                let dll = dllpath
                    .into_os_string()
                    .into_string()
                    .expect("CString::into_string failed");
                let cname = CString::new(dll).expect("CString::new failed");
                let handle = unsafe { dlopen(cname.as_ptr(), 2) }; // 2: RTLD_NOW,  1: RTLD_LAZY
                if handle.is_null() {
                    return Err(InterpretError::ForeignModuleNotFound {
                        path: Box::new(cname.into_string().expect("CString::into_string failed")),
                    });
                }
                handle
            };

            // get `luaopen_*` from dylib
            let entry_sym = {
                let symbol = format!("luaopen_{}", modname);
                let cname = CString::new(symbol).unwrap();
                let sym = unsafe { dlsym(handle, cname.as_ptr()) };

                if sym.is_null() {
                    return Err(InterpretError::BadForeignModule {
                        entry: Box::new(cname.into_string().expect("CString::into_string failed")),
                    });
                }
                sym
            };

            // execute entry symbol of dylib
            type CdylibEntry = extern "C" fn(&mut VM) -> u32;
            let dllentry: CdylibEntry = unsafe { std::mem::transmute(entry_sym) };
            Ok(dllentry(vm))
        }
    }
}
