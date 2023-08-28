use crate::{
    state::{RuntimeErr, State},
    value::RsFunc,
};

fn lua_print(lua: &mut State) -> Result<usize, RuntimeErr> {
    let n = 1;
    // for i in 1..=n {
    //     print!("{}", lua.stack_view().get(i as i32).to_string());
    //     if i > 1 {
    //         print!("\t");
    //     }
    // }

    for i in 1..=6 {
        print!("{}", lua.stack_view().get(-1 + i as i32).to_string());
        if i > 1 {
            print!("\t");
        }
    }
    println!("");
    Ok(0)
}

fn lua_assert(lua: &mut State) -> Result<usize, RuntimeErr> {
    todo!();
}

fn lua_error(lua: &mut State) -> Result<usize, RuntimeErr> {
    todo!();
}

fn lua_pcall(lua: &mut State) -> Result<usize, RuntimeErr> {
    todo!();
}

fn lua_xpcall(lua: &mut State) -> Result<usize, RuntimeErr> {
    todo!();
}

fn lua_getmetatable(lua: &mut State) -> Result<usize, RuntimeErr> {
    todo!();
}

fn lua_setmetatable(lua: &mut State) -> Result<usize, RuntimeErr> {
    todo!();
}

fn lua_next(lua: &mut State) -> Result<usize, RuntimeErr> {
    todo!();
}

fn lua_pairs(lua: &mut State) -> Result<usize, RuntimeErr> {
    todo!();
}

fn lua_ipairs(lua: &mut State) -> Result<usize, RuntimeErr> {
    todo!();
}

fn lua_load(lua: &mut State) -> Result<usize, RuntimeErr> {
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
