use crate::{
    state::{RuntimeErr, State},
    value::RsFunc,
};

fn lua_print(_lua: &mut State) -> Result<usize, RuntimeErr> {
    // let n = lua.top();
    println!("Hello Rua!");
    // for i in 1..=4 {
    //     if i > 1 {
    //         print!("\t");
    //     }
    //     print!("{i}:{}", lua.rget(i)?);
    // }

    Ok(0)
}

fn lua_assert(_lua: &mut State) -> Result<usize, RuntimeErr> {
    todo!();
}

fn lua_error(_lua: &mut State) -> Result<usize, RuntimeErr> {
    todo!();
}

fn lua_pcall(_lua: &mut State) -> Result<usize, RuntimeErr> {
    todo!();
}

fn lua_xpcall(_lua: &mut State) -> Result<usize, RuntimeErr> {
    todo!();
}

fn lua_getmetatable(_lua: &mut State) -> Result<usize, RuntimeErr> {
    todo!();
}

fn lua_setmetatable(_lua: &mut State) -> Result<usize, RuntimeErr> {
    todo!();
}

fn lua_next(_lua: &mut State) -> Result<usize, RuntimeErr> {
    todo!();
}

fn lua_pairs(_lua: &mut State) -> Result<usize, RuntimeErr> {
    todo!();
}

fn lua_ipairs(_lua: &mut State) -> Result<usize, RuntimeErr> {
    todo!();
}

fn lua_load(_lua: &mut State) -> Result<usize, RuntimeErr> {
    todo!();
}

pub const BASE_LIBS: [(&str, RsFunc); 11] = [
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
];
