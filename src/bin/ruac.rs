use rua::{
    ast::{AstDumper, ConstantFoldPass, DumpPrecison, TransformPass},
    codegen::{ChunkDumper, CodeGen},
    parser::Parser,
    RuaErr, StaticErr,
};

struct CliArg {
    input: String,
    list: usize,
    parse_only: bool,
    strip_debug: bool,
    dump_stmt: bool,
    optimize: bool,
    output: String,
}

impl CliArg {
    fn parse() -> Self {
        let (mut list, mut parse_only, mut strip_debug, mut dump_stmt, mut optimize) =
            (0, false, false, false, false);
        let (output, mut src) = ("luac.out".to_string(), String::new());
        for arg in std::env::args().skip(1) {
            match arg.as_str() {
                "-l" => list += 1,
                "-p" => {
                    parse_only = true;
                }
                "-s" => {
                    strip_debug = true;
                }
                "-d" => {
                    dump_stmt = true;
                }
                "-v" => {
                    println!("luac-rs <unknown> <other info>")
                }
                "-O" => {
                    optimize = true;
                }
                // "-o" => {
                //     output = std::env::args().unwrap_or("luac.out".to_string());
                // }
                unknown if unknown.starts_with('-') => {
                    let help = [
                        "Usage: luac [options] [filenames]",
                        "Try 'luac -h' for more information.",
                        "  -l       list (use -l -l for full listing)",
                        "  -o name  output to file 'name' (default is \"luac.out\")",
                        "  -p       parse only",
                        "  -s       strip debug information",
                        "  -d       dump ast in statement precison",
                        "  -v       show version information",
                    ]
                    .join("\n");
                    println!("luac: unrecognized option '{}'", arg);
                    println!("{help}");
                    std::process::exit(1);
                }
                _ => {
                    src = arg;
                }
            }
        }

        CliArg {
            input: src,
            list,
            parse_only,
            strip_debug,
            dump_stmt,
            optimize,
            output,
        }
    }
}

fn main() -> Result<(), RuaErr> {
    let args = CliArg::parse();

    let src = std::fs::read_to_string(&args.input).map_err(RuaErr::IOErr)?;

    let mut ast =
        Parser::parse(&src, Some(args.input)).map_err(|se| StaticErr::SyntaxErr(Box::new(se)))?;

    if args.parse_only {
        return Ok(());
    }

    if args.dump_stmt {
        let mut outer = std::io::BufWriter::new(std::io::stdout());
        return AstDumper::dump(&ast, DumpPrecison::Statement, &mut outer, false)
            .map_err(RuaErr::IOErr);
    };

    if args.optimize {
        // TODO:
        // Use OptimizeScheduler insted of ConstantFoldPass

        let mut cfp = ConstantFoldPass::new();
        cfp.walk(&mut ast);
    }

    let chunk = CodeGen::generate(ast, args.strip_debug)
        .map_err(|ce| StaticErr::CodeGenErr(Box::new(ce)))?;

    if args.list > 0 {
        if args.list == 1 {
            println!("{}", chunk);
        } else if args.list >= 2 {
            println!("{:?}", chunk);
        }
    }

    let mut ruacout =
        std::io::BufWriter::new(std::fs::File::create(&args.output).map_err(RuaErr::IOErr)?);
    ChunkDumper::dump(&chunk, &mut ruacout).map_err(RuaErr::IOErr)?;

    Ok(())
}
