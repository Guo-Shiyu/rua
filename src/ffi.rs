use crate::state::VM;
use crate::{BadModule, InterpretError, ModuleNotFound};

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
    let (dlopen, dlsym, dlfmt) = select_dll_operation_platform();

    // search dynamic library in current dir recursively.
    let target = dlfmt(modname);
    let curdir = std::env::current_dir()?;

    // TODO: detect environment variable LUA_PATH
    match find_dylib_recursive(&curdir, &target) {
        None => {
            let notfound = ModuleNotFound {
                path: curdir
                    .into_os_string()
                    .into_string()
                    .expect("CString::into_string failed"),
                lib: target,
            };
            Err(notfound.into())
        }

        Some(dllpath) => {
            let dll = dllpath
                .into_os_string()
                .into_string()
                .expect("CString::into_string failed");

            // execute `dlopen` and get handle of dylib
            let handle = {
                let cname = CString::new(dll.clone()).expect("CString::new failed");
                let handle = unsafe { dlopen(cname.as_ptr()) };
                if handle.is_null() {
                    let badmod = BadModule {
                        path: dll,
                        entry: cname.into_string().expect("CString::into_string failed"),
                    };
                    return Err(badmod.into());
                }
                handle
            };

            // get `luaopen_*` from dylib
            let entry_point = {
                let symbol = format!("luaopen_{}", modname);
                let cname = CString::new(symbol).unwrap();
                let sym = unsafe { dlsym(handle, cname.as_ptr()) };
                if sym.is_null() {
                    let badmod = BadModule {
                        path: dll,
                        entry: cname.into_string().expect("CString::into_string failed"),
                    };
                    return Err(badmod.into());
                }
                sym
            };

            // execute entry symbol of dylib
            let dllentry: CdylibEntry = unsafe { std::mem::transmute(entry_point) };
            Ok(dllentry(vm))
        }
    }
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

/// equal to `dlopen` on unix
type DllOpen = unsafe extern "C" fn(dllpath: *const c_char) -> *mut c_void;

/// equal to `dlsym` on unix
type DllGetSym = unsafe extern "C" fn(handle: *mut c_void, symbol: *const c_char) -> *mut c_void;

/// (dll-open, dll-find-symbol, dll-name-format)
type DllOperationGroup = (DllOpen, DllGetSym, fn(&str) -> String);

/// signature of rua library's entry
type CdylibEntry = extern "C" fn(vm: &mut VM) -> u32;

#[cfg(target_family = "windows")]
fn select_dll_operation_platform() -> DllOperationGroup {
    extern "C" {
        fn LoadLibraryA(filename: *const c_char) -> *mut c_void;
        fn GetProcAddress(handle: *mut c_void, symbol: *const c_char) -> *mut c_void;
        // fn FreeLibrary(handle: *mut std::ffi::c_void) -> std::os::raw::c_int;
    }
    return (LoadLibraryA, GetProcAddress, |name| format!("{}.dll", name));
}

#[cfg(target_family = "unix")]
fn select_dll_operation_platform() -> DllOperationGroup {
    extern "C" {
        fn dlopen(filename: *const c_char, flags: i32) -> *mut c_void;
        fn dlsym(handle: *mut c_void, symbol: *const c_char) -> *mut c_void;
        // fn dlclose(handle: *mut std::ffi::c_void) -> std::os::raw::c_int;
    }
    extern "C" fn dlopen_wrapper(filename: *const c_char) -> *mut c_void {
        unsafe { dlopen(filename, 1) } // 2: RTLD_NOW,  1: RTLD_LAZY
    }
    return (dlopen_wrapper, dlsym, |name| format!("lib{}.so", name));
}

#[cfg(not(any(target_family = "windows", target_family = "unix")))]
fn select_dll_operation_platform() -> DllOperationGroup {
    compile_error!("Unsupported platform to select dynamic library operation functions.")
}
