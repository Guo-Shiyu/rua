use crate::{state::State, value::RsFunc, RuaErr};

fn lua_print(lua: &mut State) -> Result<usize, RuaErr> {
    let n = lua.top();
    for i in 1..=n {
        if i > 1 {
            print!("\t");
        }
        print!("{}", lua.stk_get(i));
    }
    println!();
    Ok(0)
}

fn lua_assert(_lua: &mut State) -> Result<usize, RuaErr> {
    todo!();
}

fn lua_error(_lua: &mut State) -> Result<usize, RuaErr> {
    todo!();
}

fn lua_pcall(_lua: &mut State) -> Result<usize, RuaErr> {
    todo!();
}

fn lua_xpcall(_lua: &mut State) -> Result<usize, RuaErr> {
    todo!();
}

fn lua_getmetatable(_lua: &mut State) -> Result<usize, RuaErr> {
    todo!();
}

fn lua_setmetatable(_lua: &mut State) -> Result<usize, RuaErr> {
    todo!();
}

fn lua_next(_lua: &mut State) -> Result<usize, RuaErr> {
    todo!();
}

fn lua_pairs(_lua: &mut State) -> Result<usize, RuaErr> {
    todo!();
}

fn lua_ipairs(_lua: &mut State) -> Result<usize, RuaErr> {
    todo!();
}

fn lua_load(_lua: &mut State) -> Result<usize, RuaErr> {
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
