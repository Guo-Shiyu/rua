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
