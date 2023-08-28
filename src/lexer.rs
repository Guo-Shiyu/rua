use std::num::IntErrorKind;
use std::str::FromStr;
use std::str::Chars;

/// Defined in [Lua 5.4 manual / 3.1 ](https://www.lua.org/manual/5.4/manual.html) 
#[rustfmt::skip]
#[derive(Debug, PartialEq, Clone)]
pub enum Token {

    // keywords
    And,    Break,  Do,        Else,   Elseif, End, 
    False,  For,    Function,  Goto,   If,     In, 
    Local,  Nil,    Not,       Or,     Repeat, Return, 
    Then,   True,   Until,     While, 

    // +     -        *       /      %        ^       # 
    Add,     Minus,   Mul,    Div,   Mod,     Pow,    Len,

    // &     ~        |       <<     >>       // 
    BitAnd,  BitXor,  BitOr,  Shl,   Shr,     IDiv, 

    // ==    ~=       <=      >=     <        >       =
    Eq,      Neq,     LE,     GE,    Less,    Great,  Assign,

    // (     )        {       }      [        ]       ::
    LP,      RP,      LB,     RB,    LS,      RS,     Follow,

    // ;     :        ,       .      ..       ... 
    Semi,    Colon,   Comma,  Dot,   Concat,  Dots, 

    // others
    Integer(i64),
    Float(f64),
    Ident(String),
    Literal(String),

    // end 
    Eof,
}

impl Token {
    pub fn is_ident(&self) -> bool {
        matches!(self, Token::Ident(_))
    }

    pub fn is_number(&self) -> bool {
        matches!(self, Token::Integer(_) | Token::Float(_))
    }

    pub fn is_keyword(&self) -> bool {
        match self {
            Token::And
            | Token::Break
            | Token::Do
            | Token::Else
            | Token::Elseif
            | Token::End
            | Token::False
            | Token::For
            | Token::Function
            | Token::Goto
            | Token::If
            | Token::In
            | Token::Local
            | Token::Nil
            | Token::Not
            | Token::Or
            | Token::Repeat
            | Token::Return
            | Token::Then
            | Token::True
            | Token::Until
            | Token::While => true,
            _ => false,
        }
    }

    pub fn is_eof(&self) -> bool {
        matches!(self, Token::Eof)
    }
}

/// A low-level peekable scannner iterate the input character sequence.
///
/// Next 'n' th character can be peeked via `first` and `second` method.
/// Position can be shifted forward via `eat` and `eat_n` method.   
///
/// ref: https://github.com/rust-lang/rust/blob/master/compiler/rustc_lexer/src/cursor.rs
pub struct Scanner<'a> {
    chars: Chars<'a>,
}
//
impl<'a> Scanner<'a> {
    pub fn new(input: &'a str) -> Scanner<'a> {
        Scanner {
            chars: input.chars(),
        }
    }

    /// Peek the next `nth(n)` char without consuming any input.
    pub fn peek(&self, n: usize) -> Option<char> {
        let mut it = self.chars.clone();
        let mut ch = None;
        (0..n).for_each(|_| ch = it.next());
        ch
    }

    pub fn first(&self) -> char {
        self.peek(1).unwrap_or_default()
    }

    pub fn second(&self) -> char {
        self.peek(2).unwrap_or_default()
    }

    /// Checks if there is nothing more to consume.
    pub fn is_eof(&self) -> bool {
        self.chars.as_str().is_empty()
    }
}

impl<'a> Scanner<'a> {
    /// Consume one character.
    pub fn eat(&mut self) -> Option<char> {
        self.chars.next()
    }

    /// Consume while predicate returns `true` or until the end of input.
    pub fn eat_while(&mut self, mut predicate: impl FnMut(char) -> bool) -> &'a str {
        let record = self.chars.as_str();
        let mut len = 0;
        while predicate(self.first()) && !self.is_eof() {
            self.eat();
            len += 1;
        }
        &record[0..len]
    }

    /// Consume n character or until the end of input.
    pub fn eat_n(&mut self, n: usize) -> &'a str {
        let record = self.chars.as_str();
        let mut i = 0;
        while !self.is_eof() && i < n {
            self.eat();
            i += 1;
        }
        &record[0..n.min(record.len())]
    }
}

/// Lex input sequence to token stream.
///
/// Get next token or an lexical error via `next` method.
/// Get source loaction info via `column`, `line` method.
///
/// The lua language refrence: https://www.lua.org/manual/5.4/manual.html#3
pub struct Lexer<'a> {
    scan: Scanner<'a>,
    line: u32,
    colume: u32,
}

impl Lexer<'_> {
    pub fn new(input: &str) -> Lexer {
        Lexer {
            scan: Scanner::new(input),
            line: 1,
            colume: 1,
        }
    }

    pub fn line(&self) -> u32 {
        self.line
    }

    pub fn column(&self) -> u32 {
        self.colume
    }

    pub fn is_eof(&self) -> bool {
        self.scan.is_eof()
    }
}

impl<'a> Lexer<'a> {
    /// Record n columns.
    fn col_n(&mut self, n: usize) -> &mut Self {
        self.colume += n as u32;
        self
    }

    /// Record single column and eat a character.
    fn bump(&mut self) -> &mut Self {
        self.scan.eat();
        self.col_n(1)
    }

    /// Record new line, reset column to 1.
    fn new_line(&mut self) {
        self.line += 1;
        self.colume = 1;
    }

    /// Get next token or an lexical error.
    pub fn pump(&mut self) -> Result<Token, LexicalError> {
        const WHITESPACES: [char; 4] = [' ', '\t', '\u{013}', '\u{014}'];

        loop {
            if self.scan.is_eof() {
                break Ok(Token::Eof);
            }

            match self.scan.first() {
                wp if WHITESPACES.contains(&wp) => {
                    let whites = self.scan.eat_while(|x| WHITESPACES.contains(&x)).len();
                    self.col_n(whites);
                }

                '\n' => {
                    self.bump().new_line();
                }

                '\r' => {
                    self.scan.eat();
                }

                '[' => {
                    break match self.scan.second() {
                        '=' | '[' => Ok(Token::Literal(self.lex_multiline())),
                        _ => {
                            self.scan.eat();
                            Ok(Token::LS)
                        }
                    }
                }

                '\'' | '\"' => break self.lex_literal(),

                num if num.is_ascii_digit() => break self.lex_number(),
                '-' => {
                    self.bump();
                    if self.scan.first() == '-' {
                        self.bump().lex_comment();
                    } else {
                        break Ok(Token::Minus);
                    }
                }

                // UTF8 identity support :)
                ident if ident.is_alphabetic() || ident == '_' => {
                    let id = self
                        .scan
                        .eat_while(|ch| ch == '_' || ch.is_alphabetic() || ch.is_ascii_digit());
                    self.col_n(id.len());
                    break Ok(Self::keyword_or_ident(id));
                }

                punc if punc.is_ascii_punctuation() => {
                    break if punc == '@' || punc == '`' {
                        Err(LexicalError {
                            reason: format!("invalid punctuatuion character: {}", punc),
                        })
                    } else {
                        self.lex_punctuation(punc)
                    }
                }

                invalid => {
                    break Err(LexicalError {
                        reason: format!("invalid character: {}", invalid),
                    })
                }
            }
        }
    }

    /// Lex all form of number in lua.
    fn lex_number(&mut self) -> Result<Token, LexicalError> {
        let hex = matches!(
            (self.scan.first(), self.scan.second()),
            ('0', 'X') | ('0', 'x')
        );

        let base = if hex {
            self.scan
                .eat_while(|c| c.is_ascii_hexdigit() || c == 'x' || c == 'X' || c == '.')
        } else {
            self.scan.eat_while(|c| c.is_ascii_digit() || c == '.')
        };

        let dot = base.contains('.');

        let e_notation = {
            let peek = self.scan.first();
            peek == 'e' || peek == 'E'
        };

        match (hex, dot) {
            // 0xA.1
            (true, true) |
            // 0xABCp-1 | 0xABC
            (true, false) => {
                let src = &base[2..base.len()];

                let float = {
                    let mut dot = false;
                    let mut n = 0.0;
                    for dig in src.chars() {
                        let mut dvalue = dig as u32;
                        match dig {
                            _ if dig.is_ascii_digit() => dvalue -= '0' as u32,
                            _ if ('a'..='f').contains(&dig) => dvalue = dvalue - ('a' as u32) + 10,
                            _ if ('A'..='F').contains(&dig) => dvalue = dvalue - ('A' as u32) + 10,
                            '.' => {
                                dot = true;
                                dvalue = 0
                            }
                            _ =>
                                unreachable!(),
                        }
                        n += dvalue as f64 * if dot { 1.0 / 16.0 } else { 16.0 };
                    }
                    n
                };
                self.col_n(src.len());

                let exponent_notation = {
                    let peek = self.scan.first();
                    peek == 'p' || peek == 'P'
                };

                if exponent_notation {
                    // 'p'
                    self.bump();
                    let is_positive = match self.scan.first() {
                        '+' => {self.bump(); true},
                        '-' => {self.bump(); false},
                        dig if dig.is_ascii_digit() => true,
                        other => {
                            return Err(LexicalError {
                                reason: format!(
                                    "invalid float exponent nonation: 0x{}p{}",
                                    src, other
                                ),
                            })
                        }
                    };

                    let exp_str = self.scan.eat_while(|c| c.is_ascii_digit());
                    if exp_str.is_empty() {
                        Err(LexicalError {
                            reason: format!(
                                "invalid float exponent nonation: 0x{}p{}",
                                base, if is_positive { '+' } else {'-'}
                            ),
                        })
                    } else {
                        let exp = u32::from_str(exp_str).unwrap();
                        self.col_n(exp_str.len());
                        Ok(Token::Float(float + 2u32.pow(exp) as f64 * if is_positive { 1.0} else { -1.0 }))
                    }
                } else {
                    Ok(Token::Integer(float as i64))
                }

            }

            // 12.0 | 12.0e-2
            (false, true) => {
                if e_notation {
                    let mut copy = base.to_string();
                    copy.push(self.scan.eat().unwrap()); // 'e' | 'E'

                    let exp_str = self.scan.eat_while(|c| c.is_ascii_digit() || c == '+' || c == '-');
                    copy.push_str(exp_str);

                    if let Ok(f) = copy.parse::<f64>() {
                        self.col_n(copy.len());
                        Ok(Token::Float(f))
                    } else {
                        Err(LexicalError {
                            reason: format!("invalid float E nonation: {}", copy),
                        })
                    }
                } else if let Ok(f) = f64::from_str(base) {
                        self.col_n(base.len());
                        Ok(Token::Float(f))
                    } else {
                        Err(LexicalError {
                            reason: format!("invalid float representation: {}", base),
                        })
                    }
            },


            // 123 | 0e12
            (false, false) => {
                if e_notation {
                    let mut copy = base.to_string();
                    copy.push(self.scan.eat().unwrap()); // 'e' | 'E'

                    let exp_str = self.scan.eat_while(|c| c.is_ascii_digit() || c == '+' || c == '-');
                    copy.push_str(exp_str);

                    if let Ok(f) = copy.parse::<f64>() {
                        self.col_n(copy.len());
                        Ok(Token::Float(f))
                    } else {
                        Err(LexicalError {
                            reason: format!("invalid E-float nonation: {}", copy),
                        })
                    }
                }else {
                    self.col_n(base.len());
                    let ret = i64::from_str(base)
                        .map(Token::Integer);

                    if let Err(ref e) = ret {
                        match e.kind() {
                            IntErrorKind::PosOverflow | IntErrorKind::NegOverflow => {
                                return i128::from_str(base)
                                            .map(|i| Token::Float(i as f64))
                                            .map_err(|_| LexicalError {
                                                reason: format!("invalid interger representation: {}", base),
                                            })
                            },
                            _ => return Err(LexicalError {
                                reason: format!("invalid interger representation: {}", base),
                            }),
                        }
                    };

                    ret.map_err(|_| LexicalError {
                        reason: format!("invalid interger representation: {}", base),
                    })
                }
            }
        }
    }

    /// Lex sequence bounded with '\'' / '\"'.
    fn lex_literal(&mut self) -> Result<Token, LexicalError> {
        // ' or "
        let quote = self.scan.first();
        self.bump();

        // reserve space for short string
        let mut lit = String::with_capacity(32);
        loop {
            if self.scan.is_eof() {
                break;
            }

            match self.scan.first() {
                q if q == quote => {
                    self.bump();
                    break;
                }

                '\x00' | '\n' | '\r' => {
                    return Err(LexicalError {
                        reason: format!("unfinished literal near: {quote}{lit}"),
                    })
                }

                '\\' => {
                    self.bump();
                    let cur = self.scan.first();
                    self.bump();

                    if let Some(escape) = match cur {
                        'a' => Some('\u{0007}'),
                        'b' => Some('\u{0008}'),
                        'f' => Some('\u{000C}'),
                        'n' => Some('\u{000A}'),
                        'r' => Some('\u{000D}'),
                        't' => Some('\u{0009}'),
                        'v' => Some('\u{000b}'),
                        '\\' => Some('\u{005c}'),
                        '\'' => Some('\u{0027}'),
                        '"' => Some('\u{0022}'),
                        _ => None,
                    } {
                        lit.push(escape);
                        continue;
                    }

                    match (cur, self.scan.first(), self.scan.second()) {
                        // \xXX, XX is 2 hex digits escape seq
                        ('x', x, y) if x.is_ascii_hexdigit() && y.is_ascii_hexdigit() => {
                            let escape =
                                u8::from_str_radix(self.scan.eat_n(2), 16).map_err(|_| {
                                    LexicalError {
                                        reason: format!("invalid hex escape seq: \\x{x}{y}"),
                                    }
                                })?;
                            lit.push(escape as char);
                            self.col_n(2);
                        }

                        // \ddd, where ddd is a sequence of up to three decimal digits.
                        // if a decimal escape sequence is to be followed by a digit, it must be expressed using exactly three digits.
                        (decimal, _, _) if decimal.is_ascii_digit() => {
                            let subseq = self.scan.eat_while(|c| c.is_ascii_digit());
                            if subseq.is_empty() {
                                lit.push((decimal as u8 - b'0') as char);
                            } else {
                                let escape_len = subseq.len().min(2);
                                let s = &subseq[0..escape_len];
                                let escape = s.parse::<u8>().map_err(|_| LexicalError {
                                    reason: format!("invalid decimal escape seq: \\{s}"),
                                })?;

                                lit.push(escape as char);

                                if escape_len > 3 {
                                    lit.push_str(&subseq[3..]);
                                }
                            }
                            self.col_n(subseq.len());
                        }

                        // '\z' skips the following span of whitespace characters, including line breaks
                        ('z', _, _) => loop {
                            match self.scan.first() {
                                '\n' => {
                                    self.new_line();
                                }
                                '\r' => {
                                    if self.scan.first() == '\n' {
                                        self.scan.eat();
                                    }
                                    self.new_line();
                                }
                                '\t' | ' ' => {}
                                _ => break,
                            }
                            self.bump();
                        },

                        ('u', _, _) => {
                            self.scan.eat(); // '{'
                            let unicode = self.scan.eat_while(|c| c != '}');
                            self.scan.eat(); // '}'
                            self.col_n(2);

                            let codepoint = u32::from_str_radix(unicode, 16)
                                .map(char::from_u32)
                                .unwrap_or(None);

                            if let Some(c) = codepoint {
                                lit.push(c);
                                self.col_n(unicode.len());
                            } else {
                                return Err(LexicalError {
                                    reason: format!(
                                        r##"invalid escape sequence: \u{{{}}}"##,
                                        unicode
                                    ),
                                });
                            }
                        }

                        ('\r', _, _) => {
                            self.bump();
                            self.new_line();
                        }

                        _ => unreachable!(),
                    };
                }

                _ => {
                    lit.push(self.scan.eat().unwrap());
                }
            }
        }
        Ok(Token::Literal(lit))
    }

    /// Lex both single or multi line comment.
    fn lex_comment(&mut self) {
        match (self.scan.first(), self.scan.second()) {
            // multiline comment
            ('[', '[') | ('[', '=') => {
                self.lex_multiline();
            }
            // single line comment
            _ => {
                // Because the next character is '\n' so there is no necessary to record column.
                // This will be slightly faster during parse, but ignore a signle line comment
                // if it is on the end of input string.

                /* let n = */
                self.scan.eat_while(|x| x != '\n'); /* .len() */
                /* self.col_n(n) */
            }
        };
    }

    /// Lex sequence like "\[\[ ... ]]" or "\[==\[ ... ]==]".
    fn lex_multiline(&mut self) -> String {
        let mut pattern = String::with_capacity(16);

        // deal with pattern: "[==["
        pattern.push('[');
        self.scan.eat();
        pattern.push_str(self.scan.eat_while(|c| c == '='));
        pattern.push('[');
        self.scan.eat();

        pattern = pattern.replace('[', "]");
        self.col_n(pattern.len());

        // deal with content
        let mut content = String::with_capacity(256);

        let mut de_pattern = false;
        loop {
            let c = self.scan.eat().unwrap();
            content.push(c);
            if c == '\n' {
                self.new_line();
            } else {
                self.col_n(1);
            }

            if c == ']' {
                de_pattern = true;
            }

            if self.scan.is_eof() {
                break;
            }

            if de_pattern && content.ends_with(pattern.as_str()) {
                content.truncate(content.len() - pattern.len());
                break;
            }
        }
        content
    }

    // /// Lex sequence like "\x06".
    // fn lex_hex_escape(&mut self) -> Result<char, LexicalError> {
    //     let hex = self.scan.eat_n(2);
    //     self.col_n(2);
    //     u8::from_str_radix(hex, 16)
    //         .map(|u| u as char )
    //         .map_err(|_| LexicalError {
    //             reason: format!("invalid hexadecimal espace sequence: \\x{{{}}}", hex),
    //         })
    // }

    /// Lex operator and delimters
    fn lex_punctuation(&mut self, start: char) -> Result<Token, LexicalError> {
        if start == '.' {
            let next = self.scan.second();
            // .25 | .4
            let tk = if next.is_ascii_digit() {
                let base = self
                    .scan
                    .eat_while(|c| c.is_ascii_digit() || c == '.' || c == 'e');
                let float = f64::from_str(base).unwrap();
                Token::Float(float)
            } else if next == '.' {
                let dot_num = self.scan.eat_while(|c| c == '.').len();
                self.col_n(dot_num);
                match dot_num {
                    2 => Token::Concat,
                    3 => Token::Dots,
                    _ => unreachable!(),
                }
            } else {
                self.bump();
                Token::Dot
            };
            return Ok(tk);
        }

        use Token::*;
        self.bump();
        let token = match start {
            // single character operator
            '+' => Add,
            '*' => Mul,
            '%' => Mod,
            '^' => Pow,
            '#' => Len,
            '&' => BitAnd,
            '|' => BitOr,
            '(' => LP,
            ')' => RP,
            '{' => LB,
            '}' => RB,
            ']' => RS,
            ';' => Semi,
            ',' => Comma,
            _ => {
                let tk = match (start, self.scan.first()) {
                    ('~', '=') => Neq,
                    ('~', _) => BitXor,

                    ('<', '<') => Shl,
                    ('<', '=') => LE,
                    ('<', _) => Less,

                    ('>', '>') => Shr,
                    ('>', '=') => GE,
                    ('>', _) => Great,

                    ('=', '=') => Eq,
                    ('=', _) => Assign,

                    (':', ':') => Follow,
                    (':', _) => Colon,

                    ('/', '/') => IDiv,
                    ('/', _) => Div,
                    _ => unreachable!(),
                };

                // eat following characters
                match tk {
                    IDiv | Follow | Eq | GE | Shr | LE | Shl | Neq | Concat => {
                        self.bump();
                    }
                    Dots => {
                        self.col_n(2);
                        self.scan.eat_n(2);
                    }

                    // do nothing
                    _ => (),
                };
                tk
            }
        };
        Ok(token)
    }

    /// specify a string is ident or keyword.
    fn keyword_or_ident(name: &'a str) -> Token {
        use Token::*;
        match name {
            "and" => And,
            "break" => Break,
            "do" => Do,
            "else" => Else,
            "elseif" => Elseif,
            "end" => End,
            "false" => False,
            "for" => For,
            "function" => Function,
            "if" => If,
            "in" => In,
            "local" => Local,
            "nil" => Nil,
            "not" => Not,
            "or" => Or,
            "repeat" => Repeat,
            "return" => Return,
            "then" => Then,
            "true" => True,
            "until" => Until,
            "while" => While,
            _ => Ident(name.to_string()),
        }
    }
}

#[derive(Debug)]
pub struct LexicalError {
    pub reason: String,
}

mod test {
    #[test]
    fn common_float() {
        let floats = [
            "3.0",
            "3.1416",
            "314.16e-2",
            "0.31416E1",
            "34e1",
            ".2e2",
            ".4",
        ];
        for item in floats {
            assert!(item.parse::<f64>().is_ok())
        }
    }

    #[test]
    fn hex_floats() {
        let hex_floats = ["0x0.1E", "0xA23p-4", "0X1.921FB54442D18P+1"];

        for item in hex_floats {
            assert!(item.parse::<f64>().is_err())
        }
    }

    #[test]
    fn interger() {
        let ints = ["0xff", "0xBEBADA"];

        // assert!("0xFF".parse::<i64>().is_ok());
        // assert!(i64::from_str_radix("0xBEBADA", 16).is_ok());

        for item in ints {
            assert!(i64::from_str_radix(item, 16).is_err())
        }
    }
}
