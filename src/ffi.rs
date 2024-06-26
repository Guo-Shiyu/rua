use crate::{luastd::base, value::RsFunc};

#[derive(Clone, Copy, PartialEq)]
pub enum Stdlib {
    Base,
    Package,
    Coroutine,
    String,
    Table,
    Math,
    Io,
    Os,
    Debug,
    Bit32,
    Utf8,
    // Ffi,
    // Jit,
}

pub fn get_std_libs(lib: Stdlib) -> &'static [(&'static str, RsFunc)] {
    use Stdlib::*;
    match lib {
        Base => &base::BASE_LIBS,
        Package => todo!(),
        Coroutine => todo!(),
        String => todo!(),
        Table => todo!(),
        Math => todo!(),
        Io => todo!(),
        Os => todo!(),
        Debug => todo!(),
        Bit32 => todo!(),
        Utf8 => todo!(),
    }
}
