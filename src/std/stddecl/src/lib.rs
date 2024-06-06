extern crate proc_macro;

use proc_macro::TokenStream;
use quote::{format_ident, quote};
use syn::{parse_macro_input, ItemMod};

// fn entry_format(name: String) -> String {
//     format!("lua_{}_entry", name)
// }

// #[proc_macro]
// pub fn entry(item: TokenStream) -> TokenStream {
//     let module_name = parse_macro_input!(item as syn::LitStr);
//     return quote! { stringify!(entry_format(module_name.to_string())) }.into();
// }

/// Introduce `luaopen_#mod_name` to current file. All function in mod will be treated as `RsFunc` and added to vm on open.
#[proc_macro_attribute]
pub fn ruastd(_attr: TokenStream, item: TokenStream) -> TokenStream {
    let input = parse_macro_input!(item as ItemMod);
    let module_name = input.ident.clone();

    let mut fns = Vec::new();

    if let None = input.content {
        return quote! {
            #input
        }
        .into();
    }

    for item in &input.content.as_ref().unwrap().1 {
        if let syn::Item::Fn(ref function) = item {
            fns.push(function.sig.ident.clone());
        }
    }

    let entry = format_ident!("luaopen_{}", module_name);
    let rua_entrys = format_ident!("__rua_{}_names", module_name);
    let rua_entrycnt_id = format_ident!("__rua_{}_num", module_name);
    let entrycnt = fns.len();

    let output = quote! {
        pub #input

        #[no_mangle]
        #[used]
        static #rua_entrys: [&'static str; #entrycnt] = [#(stringify!(#fns)),*];

        #[no_mangle]
        #[used]
        static #rua_entrycnt_id: u32 = #entrycnt as u32;

        use rua::{state::VM, value::{RsFunc, Value}, InterpretError};
        use crate::#module_name::*;
        #[no_mangle]
        pub extern "C" fn #entry (vm: &mut VM) -> u32 {
            for (name, ptr) in [#(stringify!(#fns)),*].iter().zip([#(#fns),*].iter()) {
                // println!("Loading {} in {}", name, file!());
                let key = if name.ends_with("_") {
                    &name[..name.len()-1]
                } else {
                    name
                };
                let key: Value = vm.new_str(key);
                vm.set_global(key, Value::from(*ptr));
            }
            #rua_entrycnt_id
        }
    };

    output.into()
}
