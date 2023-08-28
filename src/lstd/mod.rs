use crate::value::RsFunc;

use self::base::BASE_LIBS;

pub mod base;
// pub mod coroutine;
// pub mod debug;
// pub mod math;
// pub mod os;
// pub mod package;
// pub mod string;
// pub mod table;

pub const SOME: [&[(&str, RsFunc)]; 1] = [&BASE_LIBS];
