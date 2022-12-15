use std::str::FromStr;

use super::scanner::Scanner;
use super::token::Token;

pub struct Lexer<'a> {
    src: &'a str,
    scan: Scanner<'a>,
    line: usize,
    colume: usize,
}

impl<'a> Lexer<'a> {
    pub fn new(input: &'a str) -> Lexer<'a> {
        Lexer {
            src: input,
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

    pub fn next(&mut self) -> Result<Token, SyntaxErr> {
        if self.scan.is_eof() {
            return Ok(Token::Eof);
        }

        use Token::*;
        loop {
            match self.scan.first() {
                wp if [' ', '\t', '\u{013}', '\u{014}'].contains(&wp) => {
                    let count = self.scan.eat_while(|x| x == wp).len();
                    self.advance(count);
                }

                '\n' => {
                    self.scan.eat();
                    self.advance(1).new_line();
                }
                '\r' => {
                    self.scan.eat();
                    if self.scan.first() == '\n' {
                        self.scan.eat();
                    }
                    self.new_line();
                }

                num if num.is_ascii_digit() => return self.lex_number(),

                // UTF8 identity support :)
                id if id.is_alphabetic() || id == '_' => {
                    let suffix = self.scan.eat_while(|ch| ch == '_' || ch.is_alphabetic());
                    self.advance(suffix.len());

                    let mut ident = String::with_capacity(16);
                    ident.push(id);
                    ident.push_str(suffix);

                    return Ok(Token::Ident(ident));
                }

                punc if punc.is_ascii_punctuation() => {
                    if ['@', '`'].contains(&punc) {
                        return Err(SyntaxErr::InvalidCharacter { character: punc });
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
                        '[' => LS,
                        ']' => RS,
                        ';' => Semi,
                        ',' => Comma,
                        _ => {
                            let peek = self.scan.first();
                            let token = match punc {
                                '-' => match peek {
                                    '-' => {
                                        self.scan.eat();
                                        return self.lex_comment();
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

                                quote if quote == '\'' || quote == '\"' => {
                                    let lit = self.scan.eat_while(|ch| ch != quote);
                                    self.scan.eat();
                                    self.advance(1);

                                    Literial(String::from_str(lit).unwrap())
                                }

                                _ => unreachable!(),
                            };

                            // eat follows char
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

                invalid => return Err(SyntaxErr::InvalidCharacter { character: invalid }),
            }
        }
    }

    fn lex_number(&mut self) -> Result<Token, SyntaxErr> {
        let is_number_digit = |c: char| c.is_ascii_digit() || c == 'x' || c == 'X' || c == '.';
        let num_str = self.scan.eat_while(is_number_digit);

        if num_str.contains('.') {
            if let Ok(f) = f64::from_str(num_str) {
                Ok(Token::Float(f))
            } else {
                Err(SyntaxErr::BrokenFloat {
                    raw: num_str.to_string(),
                })
            }
        } else {
            let hex = num_str.contains('x') || num_str.contains('X');
            if let Ok(i) = u64::from_str_radix(num_str, if hex { 16 } else { 10 }) {
                Ok(Token::Integer(i))
            } else {
                Err(SyntaxErr::BrokenInteger {
                    raw: num_str.to_string(),
                })
            }
        }
    }

    // fn lex_literal(&mut self) -> Result<Token, SyntaxErr> {
    //     todo!()
    // }

    fn lex_comment(&mut self) -> Result<Token, SyntaxErr> {
        match self.scan.first() {
            // multiline comment
            '[' => {
                self.scan.eat();
                self.scan.eat();
                self.advance(2);

                loop {
                    let count = self.scan.eat_while(|x| x != ']').len();
                    self.advance(count);

                    debug_assert_eq!(self.scan.first(), ']');

                    match self.scan.second() {
                        ']' => {
                            self.scan.eat();
                            self.scan.eat();
                            self.advance(2);
                        }
                        '\n' => {
                            self.scan.eat();
                            self.advance(1).new_line();
                        }
                        '\r' => {
                            self.scan.eat();
                            if self.scan.first() == '\n' {
                                self.scan.eat();
                            }
                            self.new_line();
                        }
                        _ => {}
                    }
                }
            }
            // single line comment
            _ => {}
        };
        self.next()
    }
}

#[derive(Debug)]
pub enum SyntaxErr {
    InvalidCharacter { character: char },
    BrokenHexNumber { raw: String },
    BrokenFloat { raw: String },
    BrokenInteger { raw: String },
}
