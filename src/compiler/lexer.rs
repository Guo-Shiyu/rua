use std::num::IntErrorKind;
use std::str::FromStr;

use super::{scanner::Scanner, token::Token};

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
    fn bump(&mut self ) -> &mut Self {
        self.scan.eat();
        self.col_n(1)
    }

    /// Record new line, reset column to 1.
    fn new_line(&mut self) {
        self.line += 1;
        self.colume = 1;
    }

    /// Get next token or an lexical error.
    pub fn next(&mut self) -> Result<Token, LexicalErr> {
        
        const WHITESPACES: [char; 4] = [' ', '\t', '\u{013}', '\u{014}'];
        
        loop {
            if self.scan.is_eof() {
                break Ok(Token::Eof)
            }

            match self.scan.first() {
                '\0' => { self.scan.eat(); }

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
                
                '[' => break match self.scan.second() {
                    '=' | '[' =>  Ok(Token::Literal(self.lex_multiline())),
                    _ => {
                        self.scan.eat();
                        Ok(Token::LS)
                    }
                },
    
                '\'' | '\"' => break self.lex_literal(),
    
                num if num.is_ascii_digit() => break self.lex_number(),
                    '-' => {
                    self.bump();
                        if self.scan.first() == '-' {
                            self.bump().lex_comment();
                        } else {
                            break Ok(Token::Minus)
                        }
                }
    
                // UTF8 identity support :)
                ident if ident.is_alphabetic() || ident == '_' => {
                    let id = self.scan.eat_while(|ch| ch == '_' || ch.is_alphabetic() || ch.is_ascii_digit());
                    self.col_n(id.len());
                    break Ok(Self::keyword_or_ident(id))
                }
                
                punc if punc.is_ascii_punctuation() => {
                    break if punc == '@' || punc == '`'{
                        Err(LexicalErr {
                            reason: format!("invalid punctuatuion character: {}", punc),
                        })
                    } else {
                        self.lex_punctuation(punc)
                    }
                }
    
                invalid => {
                    break Err(LexicalErr {
                        reason: format!("invalid character: {}", invalid),
                    })
                }
            }
        }
    }

    /// Lex all form of number in lua.
    fn lex_number(&mut self) -> Result<Token, LexicalErr> {
        let hex = matches!((self.scan.first(), self.scan.second()), ('0', 'X') | ('0', 'x'));

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
                            _ if ('0'..='9').contains(&dig) => dvalue = dvalue - ('0' as u32),
                            _ if ('a'..='f').contains(&dig) => dvalue = dvalue - ('a' as u32) + 10,
                            _ if ('A'..='F').contains(&dig) => dvalue = dvalue - ('A' as u32) + 10,
                            '.' => {
                                dot = true;
                                dvalue = 0
                            }
                            _ => {
                                let _ = 123; 
                                unreachable!()},
                        }
                        n += dvalue as f64 * if dot { 1.0 / 16.0 } else { 16.0 };}  
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
                            return Err(LexicalErr {
                                reason: format!(
                                    "invalid float exponent nonation: 0x{}p{}",
                                    src, other
                                ),
                            })
                        }
                    };
                    
                    let exp_str = self.scan.eat_while(|c| c.is_ascii_digit());
                    if exp_str.is_empty() {
                        Err(LexicalErr {
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
                        Err(LexicalErr {
                            reason: format!("invalid float E nonation: {}", copy),
                        })
                    }
                } else if let Ok(f) = f64::from_str(base) {
                        self.col_n(base.len());
                        Ok(Token::Float(f))
                    } else {
                        Err(LexicalErr {
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
                        Err(LexicalErr {
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
                q if q == quote => {self.bump(); break},

                '\x00' | '\n' | '\r' => {
                    return Err(LexicalErr {
                        reason: format!("unfinished literal near: {quote}{lit}"),
                    })
                }

                '\\' => {
                    self.bump();
                    let cur = self.scan.first();
                    match cur {
                        // decimal escape seq like \012, \987
                        decimal if decimal.is_ascii_digit() => {
                            // read max 3 digis 
                            let mut digit = 0;
                            let dec = self.scan.eat_while(|c| {
                                digit += 1;
                                digit < 3 && c.is_ascii_digit()
                            });

                            self.col_n(dec.len());
                            if let Ok(int) = u8::from_str(dec) {
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
                                
                                // "abcd\
                                //   efgh"
                                '\n' => {
                                    self.new_line();
                                }
                                '\r' => {
                                    if self.scan.first() == '\n' {
                                        self.scan.eat();
                                    }
                                    self.new_line();
                                }
                                
                                // hex escape like \x789
                                'x' => {
                                    lit.push(self.lex_hex_escape()?);
                                }
                                // '\z' skips the following span of whitespace characters, including line breaks
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
                                    self.col_n(2);

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
                    lit.push(self.scan.eat().unwrap());
                }
            }
        }
        self.col_n(lit.len() + 2);
        lit.shrink_to(lit.len());
        Ok(Token::Literal(lit))
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
                // Because the next character is '\n' so there is no necessary to record column.
                // This will be slightly faster during parse, but ignore a signle line comment 
                // if it is on the end of input string.  

                /* let n = */ self.scan.eat_while(|x| x != '\n'); /* .len() */
                /* self.col_n(n) */ 
            }
            
        };
    }

    /// Lex sequence like "\[\[ ... ]]" / "\[==\[ ... ]==]".
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

        content.shrink_to_fit();
        content
    }

    /// Lex sequence like "\x06".
    fn lex_hex_escape(&mut self) -> Result<char, LexicalErr> {
        let hex = self.scan.eat_n(2);
        self.col_n(2);
        u8::from_str_radix(hex, 16)
            .map(|u| u as char )
            .map_err(|_| LexicalErr {
                reason: format!("invalid hexadecimal espace sequence: \\x{{{}}}", hex),
            })
    }

    /// Lex operator and delimters 
    fn lex_punctuation(&mut self, start:char) -> Result<Token, LexicalErr> {
        if start == '.' {
            let token = match (self.scan.first(), self.scan.second()) {
                ('.', '.') => {self.col_n(2).scan.eat_n(2); Token::Dots},
                ('.', _) => {self.bump(); Token::Concat},
                (_, _) => Token::Dot,
            };
            return Ok(token)
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
                    ('~', _) => BitOr,

                    ('<', '<') => Shl,
                    ('<', '=') => LE,
                    ('<', _) => Less,
                    
                    ('>', '>') => Shr,
                    ('>', '=') => GE,
                    ('>', _) => Great,
                    
                    ('=', '=') => Eq,
                    ('=',_ ) => Assign,
                    
                    (':', ':') => Follow,
                    (':', _) => Colon,
                    
                    ('/', '/') => IDiv,
                    ('/', _) => Div,
                    _ => unreachable!()
                };
                
                // eat following characters
                match tk {
                    IDiv | Follow | GE | Shr | LE | Shl | Neq | Concat => {
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
pub struct LexicalErr {
    pub reason: String,
}
