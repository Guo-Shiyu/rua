use std::str::Chars;

pub struct Scanner<'a> {
    chars: Chars<'a>,
}

impl<'a> Scanner<'a> {
    pub fn new(input: &'a str) -> Scanner<'a> {
        Scanner {
            chars: input.chars(),
        }
    }

    /// Peek the next `nth(n)` char without consuming any input.
    fn peek(&self, n: usize) -> Option<char> {
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

    // /// Remainning character
    // pub fn remainning(&self) -> usize {
    //     self.chars.as_str().len()
    // }
}

impl<'a> Scanner<'a> {
    /// Consume one character.
    pub fn eat(&mut self) -> Option<char> {
        self.chars.next()
    }

    /// Consume while predicate returns `true` or until the end of input.
    pub fn eat_while(&mut self, predicate: impl Fn(char) -> bool) -> &'a str {
        let mut len = 0;
        let record  = self.chars.as_str();
        while predicate(self.first()) && !self.is_eof() {
            self.eat();
            len += 1;
        }
        &record[0..len]
    }
}
