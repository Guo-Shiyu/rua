/// Defined in [Lua 5.4 manual / 3.1 ](https://www.lua.org/manual/5.4/manual.html) 
#[rustfmt::skip]
#[derive(Debug)]
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
    Literial(String),

    // end 
    Eof,
}

impl Token {
    pub fn is_ident(&self) -> bool {
        if let Token::Ident(_) = &self {
            true
        } else {
            false
        }
    }

    pub fn is_number(&self) -> bool {
        match self {
            Token::Integer(_) | Token::Float(_) => true,
            _ => false,
        }
    }

    // pub fn is_operator(&self) -> bool {
    // Token::Add => todo!(),
    //     Token::Minus => todo!(),
    //     Token::Mul => todo!(),
    //     Token::Div => todo!(),
    //     Token::Mod => todo!(),
    //     Token::Pow => todo!(),
    //     Token::Len => todo!(),
    //     Token::BitAnd => todo!(),
    //     Token::BitXor => todo!(),
    //     Token::BitOr => todo!(),
    //     Token::Shl => todo!(),
    //     Token::Shr => todo!(),
    //     Token::IDiv => todo!(),
    //     Token::Eq => todo!(),
    //     Token::Neq => todo!(),
    //     Token::LE => todo!(),
    //     Token::GE => todo!(),
    //     Token::Less => todo!(),
    //     Token::Great => todo!(),
    //     Token::Assign => todo!(),
    //     Token::LP => todo!(),
    //     Token::RP => todo!(),
    //     Token::LB => todo!(),
    //     Token::RB => todo!(),
    //     Token::LS => todo!(),
    //     Token::RS => todo!(),
    //     Token::Follow => todo!(),
    //     Token::Semi => todo!(),
    //     Token::Colon => todo!(),
    //     Token::Comma => todo!(),
    //     Token::Dot => todo!(),
    //     Token::Concat => todo!(),
    //     Token::Dots => todo!(),
    // }

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
        if let Token::Eof = self {
            true
        } else {
            false
        }
    }
}
