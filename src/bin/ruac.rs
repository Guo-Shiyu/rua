use rua::{
    ast::{dump_ast, DumpPrecison},
    codegen::{dump_chunk, CodeGen},
    heap::Heap,
    parser::Parser,
    passes, InterpretError,
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

fn main() -> Result<(), InterpretError> {
    let args = CliArg::parse();

    let src = std::fs::read_to_string(&args.input).map_err(InterpretError::IOErr)?;

    let mut ast = Parser::parse(&src, Some(args.input))?;

    if args.parse_only {
        return Ok(());
    }

    if args.dump_stmt {
        let mut outer = std::io::BufWriter::new(std::io::stdout());
        return dump_ast(&ast, DumpPrecison::Statement, &mut outer, false)
            .map_err(InterpretError::IOErr);
    };

    if args.optimize {
        // TODO:
        // Use OptimizeScheduler insted of ConstantFoldPass

        passes::constant_fold(&mut ast);
    }

    let mut heap = Heap::default();

    let chunk = CodeGen::codegen(ast, args.strip_debug, &mut heap)?;

    if args.list > 0 {
        if args.list == 1 {
            println!("{}", chunk);
        } else if args.list >= 2 {
            println!("{:?}", chunk);
        }
    }

    let mut ruacout = std::io::BufWriter::new(
        std::fs::File::create(&args.output).map_err(InterpretError::IOErr)?,
    );
    dump_chunk(&chunk, &mut ruacout).map_err(InterpretError::IOErr)?;

    Ok(())
}
