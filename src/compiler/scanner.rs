use std::str::Chars;

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
