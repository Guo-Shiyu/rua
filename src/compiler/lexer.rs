use std::num::IntErrorKind;
use std::str::FromStr;

use super::scanner::Scanner;
use super::token::Token;

pub struct Lexer<'a> {
    scan: Scanner<'a>,
    line: usize,
    colume: usize,
}

impl<'a> Lexer<'a> {
    pub fn new(input: &'a str) -> Lexer<'a> {
        Lexer {
            scan: Scanner::new(input),
            line: 1,
            colume: 1,
        }
    }

    pub fn line(&self) -> usize {
        self.line
    }

    pub fn column(&self) -> usize {
        self.colume
    }
}

impl<'a> Lexer<'a> {
    fn advance(&mut self, n: usize) -> &mut Self {
        self.colume += n;
        self
    }

    fn new_line(&mut self) {
        self.line += 1;
        self.colume = 1;
    }

    /// Get next token or an error
    pub fn next(&mut self) -> Result<Token, LexicalErr> {
        use Token::*;
        let whitespaces = [' ', '\t', '\u{013}', '\u{014}'];
        loop {
            if self.scan.is_eof() {
                return Ok(Token::Eof);
            }
            match self.scan.first() {
                wp if whitespaces.contains(&wp) => {
                    let count = self.scan.eat_while(|x| whitespaces.contains(&x)).len();
                    self.advance(count);
                    if self.scan.is_eof() {
                        return Ok(Token::Eof);
                    }
                }

                '\n' => {
                    self.scan.eat();
                    self.advance(1).new_line();
                }
                '\r' => {
                    self.scan.eat();
                }

                '[' => match self.scan.second() {
                    '=' | '[' => return Ok(Token::Literial(self.lex_multiline())),
                    _ => {
                        self.scan.eat();
                        return Ok(Token::LS);
                    }
                },

                '\'' | '\"' => return self.lex_literal(),

                num if num.is_ascii_digit() => return self.lex_number(),

                // UTF8 identity support :)
                id if id.is_alphabetic() || id == '_' => {
                    return Ok(self.scan.eat_while(|ch| ch == '_' || ch.is_alphabetic() || ch.is_ascii_digit()))
                        .map(|id| {self.advance(id.len()); id})
                        .map(Self::keyword_or_ident);
                }

                punc if punc.is_ascii_punctuation() => {
                    if ['@', '`'].contains(&punc) {
                        return Err(LexicalErr {
                            reason: format!("invalid character: {}", punc),
                        });
                    }

                    self.scan.eat();
                    self.advance(1);
                    return Ok(match punc {
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
                            let peek = self.scan.first();
                            let token = match punc {
                                '-' => match peek {
                                    '-' => {
                                        // eat "--"
                                        self.scan.eat();
                                        self.advance(1); 
                                        self.lex_comment();
                                        continue;
                                    }
                                    _ => Minus,
                                },
                                '.' => match (self.scan.first(), self.scan.second()) {
                                    ('.', '.') => Dots,
                                    ('.', _) => Concat,
                                    (_, _) => Dot,
                                },
                                '~' => match peek {
                                    '=' => Neq,
                                    _ => BitXor,
                                },
                                '<' => match peek {
                                    '<' => Shl,
                                    '=' => LE,
                                    _ => Less,
                                },
                                '>' => match peek {
                                    '>' => Shr,
                                    '=' => GE,
                                    _ => Great,
                                },
                                '=' => match peek {
                                    '=' => Eq,
                                    _ => Assign,
                                },
                                ':' => match peek {
                                    ':' => Follow,
                                    _ => Colon,
                                },
                                '/' => match peek {
                                    '/' => IDiv,
                                    _ => Div,
                                },

                                _ => unreachable!(),
                            };

                            // eat follows character
                            match token {
                                IDiv | Follow | GE | Shr | LE | Shl | Neq | Concat => {
                                    self.advance(1);
                                    self.scan.eat();
                                }
                                Dots => {
                                    self.advance(2);
                                    self.scan.eat();
                                    self.scan.eat();
                                }
                                _ => (),
                            };
                            token
                        }
                    });
                }

                invalid => {
                    return Err(LexicalErr {
                        reason: format!("invalid character: {}", invalid),
                    })
                }
            }
        }
    }

    /// Lex all form of number in lua.
    fn lex_number(&mut self) -> Result<Token, LexicalErr> {
        let hex = match (self.scan.first(), self.scan.second()) {
            ('0', 'X') | ('0', 'x') => true,
            _ => false,
        };

        let base = if hex {
            self.scan
                .eat_while(|c| c.is_ascii_hexdigit() || c == 'x' || c == 'X' || c == '.')
        } else {
            self.scan.eat_while(|c| c.is_ascii_digit() || c == '.')
        };

        let dot = base.contains('.');

        let e_notation = {
            let peek = self.scan.first();
            if peek == 'e' || peek == 'E' {
                true
            } else {
                false
            }
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
                            _ if '0' <= dig && dig <= '9' => dvalue = dvalue - ('0' as u32),
                            _ if 'a' <= dig && dig <= 'z' => dvalue = dvalue - ('a' as u32) + 10,
                            _ if 'A' <= dig && dig <= 'z' => dvalue = dvalue - ('A' as u32) + 10,
                            '.' => {
                                dot = true;
                                dvalue = 0
                            }
                            _ => unreachable!(),
                        }
                        n += dvalue as f64 * if dot { 1.0 / 16.0 } else { 16.0 };}  
                    n
                };
                self.advance(src.len());

                let exponent_notation = {let peek = self.scan.first();
                if peek == 'p' || peek == 'P' {
                    true
                } else {
                    false
                }};

                if exponent_notation {
                    self.scan.eat(); // 'p'
                    self.advance(1);
                    let is_positive = match self.scan.first() {
                        '+' => {self.scan.eat(); self.advance(1); true},
                        '-' => {self.scan.eat(); self.advance(1); false},
                        dig if dig.is_ascii_digit() => true,
                        other => {
                            return Err(LexicalErr {
                                reason: format!(
                                    "invalid float exponent nonation: 0x{}p{}",
                                    src, other
                                ),
                            })
                        }
                    };
                    
                    let exp_str = self.scan.eat_while(|c| c.is_ascii_digit());
                    if exp_str.len() == 0 {
                        Err(LexicalErr {
                            reason: format!(
                                "invalid float exponent nonation: 0x{}p{}",
                                base, if is_positive { '+' } else {'-'}
                            ),
                        })
                    } else {
                        let exp = u32::from_str(exp_str).unwrap();
                        self.advance(exp_str.len());
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
                        self.advance(copy.len());
                        Ok(Token::Float(f))
                    } else {
                        Err(LexicalErr {
                            reason: format!("invalid float E nonation: {}", copy),
                        })
                    }
                } else {
                    if let Ok(f) = f64::from_str(base) {
                        self.advance(base.len());
                        Ok(Token::Float(f))
                    } else {
                        Err(LexicalErr {
                            reason: format!("invalid float representation: {}", base),
                        })
                    }
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
                        self.advance(copy.len());
                        Ok(Token::Float(f))
                    } else {
                        Err(LexicalErr {
                            reason: format!("invalid E-float nonation: {}", copy),
                        })
                    }
                }else {
                    self.advance(base.len());
                    let ret = i64::from_str(base)
                        .map(|i| Token::Integer(i));

                    
                    if let Err(ref e) = ret {
                        match e.kind() {
                            IntErrorKind::PosOverflow | IntErrorKind::NegOverflow => {
                                return i128::from_str(base)
                                            .map(|i| Token::Float(i as f64))
                                            .map_err(|_| LexicalErr {
                                                reason: format!("invalid interger representation: {}", base),
                                            })
                            },
                            _ => return Err(LexicalErr {
                                reason: format!("invalid interger representation: {}", base),
                            }),
                        }
                    };  

                    ret.map_err(|_| LexicalErr {
                        reason: format!("invalid interger representation: {}", base),
                    })
                }
            }
        }
    }

    /// Lex sequence bounded with '\'' / '\"'.
    fn lex_literal(&mut self) -> Result<Token, LexicalErr> {
        let quote = self.scan.eat().unwrap();

        let mut lit = String::with_capacity(32);
        loop {
            match self.scan.first() {
                '\x00' | '\n' | '\r' => {
                    return Err(LexicalErr {
                        reason: "unfinished literal".to_string(),
                    })
                }
                '\\' => {
                    self.scan.eat();
                    self.advance(1);
                    let cur = self.scan.first();
                    match cur {
                        '\x00' => {}
                        decimal if decimal.is_ascii_digit() => {

                            // read max 3 digis 
                            let mut digit = 0;
                            let dec = self.scan.eat_while(|c| {
                                digit += 1;
                                digit < 3 && c.is_ascii_digit()
                            });

                            self.advance(dec.len());
                            if let Ok(int) = u8::from_str(dec) {
                                debug_assert!(int <= u8::MAX);
                                lit.push(int as char);
                            } else {
                                return Err(LexicalErr {
                                    reason: format!("invalid decimal escape sequence: \\{}", dec),
                                });
                            }
                        }
                        _ => {
                            self.scan.eat();
                            match cur {
                                'a' => lit.push('\u{0007}'),
                                'b' => lit.push('\u{0008}'),
                                'f' => lit.push('\u{000C}'),
                                'n' => lit.push('\u{000A}'),
                                'r' => lit.push('\u{000D}'),
                                't' => lit.push('\u{0009}'),
                                'v' => lit.push('\u{000b}'),
                                '\\' => lit.push('\u{005c}'),
                                '\'' => lit.push('\u{0027}'),
                                '"' => lit.push('\u{0022}'),
                                '\n' => {
                                    self.new_line();
                                }
                                '\r' => {
                                    if self.scan.first() == '\n' {
                                        self.scan.eat();
                                    }
                                    self.new_line();
                                }
                                'x' => {
                                    lit.push(self.lex_hex_escape()?);
                                }
                                'z' => {
                                    loop {
                                        match self.scan.first() {
                                            '\n' => {
                                                self.new_line();
                                            }
                                            '\r' => {
                                                if self.scan.first() == '\n' {
                                                    self.scan.eat();
                                                }
                                                self.new_line();
                                            },
                                            '\t' | ' ' => {}
                                            _ => break
                                        }
                                        self.scan.eat();
                                    }
                                }

                                'u' => {
                                    self.scan.eat(); // '{'
                                    let unicode = self.scan.eat_while(|c| c != '}');
                                    self.scan.eat(); // '}'
                                    self.advance(2);

                                    let codepoint = u32::from_str_radix(unicode, 16)
                                        .map(char::from_u32)
                                        .unwrap_or(None);
                                        
                                        if let Some( c) = codepoint {
                                            lit.push(c);
                                        } else {
                                            return Err(LexicalErr {
                                                reason: format!(
                                                    r##"invalid escape sequence: \u{{{}}}"##,
                                                    unicode
                                                ),
                                            });
                                        }
                                    
                                    
                                }

                                bad => {
                                    return Err(LexicalErr {
                                        reason: format!(
                                            "invalid escape sequence: \\{}",
                                            bad as u32
                                        ),
                                    })
                                }
                            }
                        }
                    }
                }
                _ => {
                    let cur = self.scan.eat().unwrap();
                    if cur == quote {
                        break;
                    }
                    lit.push(cur);
                }
            }
        }
        self.advance(lit.len() + 2);
        lit.shrink_to(lit.len());
        Ok(Token::Literial(lit))
    }

    /// Lex both single or multi line comment.
    fn lex_comment(&mut self) {
        match (self.scan.first(), self.scan.second()) {
            // multiline comment
            ('[', '[') => {
                self.lex_multiline();
            }

            // single line comment
            _ => {
                let count = self.scan.eat_while(|x| x != '\n').len();
                self.advance(count);
            }
        }
    }

    /// Lex sequence like "[[ hello ]]" / "[==[ hello ]==]".
    fn lex_multiline(&mut self) -> String {
        // deal with "[==["
        let mut pattern = String::with_capacity(16);
        pattern.push('[');
        self.scan.eat();
        for _ in 0..self.scan.eat_while(|c| c == '=').len() {
            pattern.push('=');
        }
        pattern.push('[');
        self.scan.eat();
        pattern = pattern.replace("[", "]");
        self.advance(pattern.len());

        // deal with content 
        let mut content = String::with_capacity(256);
        loop {
            let c = self.scan.eat().unwrap();
            content.push(c);
            if c == '\n' {
                self.new_line();
            }
            if content.ends_with(pattern.as_str()) {
                for _ in 0..pattern.len() {
                    content.pop();
                }
                break;
            }
        }

        // fix column at last line 
        self.advance(
            pattern.len() + 
        if content.ends_with('\n') {
            0
        } else {
            content.split_terminator('\n').last().unwrap().len()
        });

        content.shrink_to_fit();
        content
    }

    /// Lex sequence like "\x06".
    fn lex_hex_escape(&mut self) -> Result<char, LexicalErr> {
        let hex = self.scan.eat_n(2);
        u8::from_str_radix(hex, 16)
            .map(|u| { self.advance(2); u as char} )
            .map_err(|_| LexicalErr {
                reason: format!("invalid hexadecimal espace sequence: \\x{{{}}}", hex),
            })
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
pub struct LexicalErr {
    reason: String,
}

impl LexicalErr {
    pub fn reason(&self) -> &str {
        self.reason.as_str()
    }
}

