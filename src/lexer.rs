use anyhow::{Result, anyhow};

#[derive(Debug, Clone, PartialEq)]
pub enum Token {
    // Literals
    Number(i64),
    Float(f64),
    String(String),
    Boolean(bool),
    Null,
    
    // Identifiers
    Identifier(String),
    
    // Keywords - Basic
    App,
    Fun,
    Let,
    Const,
    If,
    Else,
    Return,
    While,
    For,
    In,
    True,
    False,
    Break,
    Continue,
    
    // Keywords - Advanced Control Flow
    Match,
    Try,
    Catch,
    Finally,
    Throw,
    Await,
    Async,
    
    // Keywords - Object-Oriented
    Class,
    Trait,
    Impl,
    Public,
    Private,
    Protected,
    Static,
    Extends,
    Interface,
    Abstract,
    Enum,
    Constructor,
    Method,
    Super,
    This,
    SelfKeyword,
    Override,
    Virtual,
    Final,
    
    // Keywords - Threading & Concurrency
    Spawn,
    Thread,
    Lock,
    Mutex,
    Channel,
    
    // Keywords - GPU & System
    Gpu,
    Kernel,
    Cuda,
    OpenCL,
    System,
    Process,
    File,
    
    // Keywords - Math & Science
    Math,
    Stats,
    Matrix,
    Complex,
    
    // Keywords - Memory & Performance
    Unsafe,
    Safe,
    Fast,
    Hypercar,
    
    // Types - Basic
    TypeInt,
    TypeFloat,
    TypeString,
    TypeBool,
    TypeVoid,
    TypeDouble,
    TypeChar,
    TypeByte,
    TypeLong,
    
    // Types - Advanced Numeric
    TypeInt8, TypeInt16, TypeInt32, TypeInt64,
    TypeUInt8, TypeUInt16, TypeUInt32, TypeUInt64,
    TypeFloat32, TypeFloat64,
    TypeDecimal,
    TypeBigInt,
    
    // Types - System & Concurrency
    TypeThread,
    TypeMutex,
    TypeChannel,
    TypeFuture,
    
    // Types - GPU
    TypeGpuBuffer,
    TypeGpuKernel,
    TypeGpuMemory,
    
    // Types - OS
    TypeProcess,
    TypeFile,
    TypeSocket,
    TypeSystemInfo,
    
    // Types - Complex
    TypeArray,
    TypeOptional,
    TypeResult,
    
    // Operators - Basic Arithmetic
    Plus,
    Minus,
    Multiply,
    Divide,
    Modulo,
    Power,
    Root,
    
    // Operators - Safe Arithmetic
    SafeAdd,
    SafeSubtract,
    SafeMultiply,
    SafeDivide,
    
    // Operators - Bitwise
    BitAnd,
    BitOr,
    BitXor,
    BitNot,
    LeftShift,
    RightShift,
    
    // Operators - Assignment
    Assign,
    PlusAssign,
    MinusAssign,
    MultiplyAssign,
    DivideAssign,
    
    // Operators - Comparison
    Equal,
    NotEqual,
    Less,
    Greater,
    LessEqual,
    GreaterEqual,
    
    // Operators - Logical
    And,
    Or,
    Not,
    
    // Operators - String
    Concat,
    Contains,
    
    // Operators - Advanced Math
    Factorial,
    Fibonacci,
    Gcd,
    Lcm,
    Prime,
    
    // Operators - Math Functions
    Sin, Cos, Tan,
    Asin, Acos, Atan,
    Sinh, Cosh, Tanh,
    Log, Log10, Exp,
    Sqrt, Abs,
    Floor, Ceil, Round,
    Min, Max,
    Atan2, Hypot,
    
    // Operators - Type Checking
    IsNaN,
    IsInfinite,
    IsFinite,
    
    // Delimiters
    LeftParen,
    RightParen,
    LeftBrace,
    RightBrace,
    LeftBracket,
    RightBracket,
    Comma,
    Semicolon,
    Colon,
    DoubleColon,
    Arrow,
    FatArrow,
    Dot,
    DotDot,
    Question,
    
    // Special
    Newline,
    Eof,
    
    // Comments
    Comment(String),
}

pub struct Lexer {
    input: Vec<char>,
    position: usize,
    current_char: Option<char>,
}

impl Lexer {
    pub fn new(input: &str) -> Self {
        let chars: Vec<char> = input.chars().collect();
        let current_char = chars.get(0).copied();
        
        Self {
            input: chars,
            position: 0,
            current_char,
        }
    }

    pub fn tokenize(&mut self) -> Result<Vec<Token>> {
        let mut tokens = Vec::new();
        
        while let Some(token) = self.next_token()? {
            if !matches!(token, Token::Comment(_)) { // Skip comments in output
                tokens.push(token);
            }
        }
        
        tokens.push(Token::Eof);
        Ok(tokens)
    }

    fn next_token(&mut self) -> Result<Option<Token>> {
        self.skip_whitespace();
        
        match self.current_char {
            None => Ok(None),
            Some('\n') => {
                self.advance();
                Ok(Some(Token::Newline))
            }
            Some('+') => {
                self.advance();
                if self.current_char == Some('=') {
                    self.advance();
                    Ok(Some(Token::PlusAssign))
                } else if self.current_char == Some('+') {
                    self.advance();
                    Ok(Some(Token::SafeAdd))
                } else {
                    Ok(Some(Token::Plus))
                }
            }
            Some('-') => {
                self.advance();
                if self.current_char == Some('>') {
                    self.advance();
                    Ok(Some(Token::Arrow))
                } else if self.current_char == Some('=') {
                    self.advance();
                    Ok(Some(Token::MinusAssign))
                } else if self.current_char == Some('-') {
                    self.advance();
                    Ok(Some(Token::SafeSubtract))
                } else {
                    Ok(Some(Token::Minus))
                }
            }
            Some('*') => {
                self.advance();
                if self.current_char == Some('=') {
                    self.advance();
                    Ok(Some(Token::MultiplyAssign))
                } else if self.current_char == Some('*') {
                    self.advance();
                    if self.current_char == Some('*') {
                        self.advance();
                        Ok(Some(Token::SafeMultiply))
                    } else {
                        Ok(Some(Token::Power))
                    }
                } else {
                    Ok(Some(Token::Multiply))
                }
            }
            Some('/') => {
                self.advance();
                if self.current_char == Some('/') {
                    // Single line comment
                    self.advance();
                    let comment = self.read_comment();
                    Ok(Some(Token::Comment(comment)))
                } else if self.current_char == Some('=') {
                    self.advance();
                    Ok(Some(Token::DivideAssign))
                } else if self.current_char == Some('/') {
                    self.advance();
                    Ok(Some(Token::SafeDivide))
                } else {
                    Ok(Some(Token::Divide))
                }
            }
            Some('%') => {
                self.advance();
                Ok(Some(Token::Modulo))
            }
            Some('=') => {
                self.advance();
                if self.current_char == Some('=') {
                    self.advance();
                    Ok(Some(Token::Equal))
                } else if self.current_char == Some('>') {
                    self.advance();
                    Ok(Some(Token::FatArrow))
                } else {
                    Ok(Some(Token::Assign))
                }
            }
            Some('!') => {
                self.advance();
                if self.current_char == Some('=') {
                    self.advance();
                    Ok(Some(Token::NotEqual))
                } else {
                    Ok(Some(Token::Not))
                }
            }
            Some('<') => {
                self.advance();
                if self.current_char == Some('=') {
                    self.advance();
                    Ok(Some(Token::LessEqual))
                } else if self.current_char == Some('<') {
                    self.advance();
                    Ok(Some(Token::LeftShift))
                } else {
                    Ok(Some(Token::Less))
                }
            }
            Some('>') => {
                self.advance();
                if self.current_char == Some('=') {
                    self.advance();
                    Ok(Some(Token::GreaterEqual))
                } else if self.current_char == Some('>') {
                    self.advance();
                    Ok(Some(Token::RightShift))
                } else {
                    Ok(Some(Token::Greater))
                }
            }
            Some('&') => {
                self.advance();
                if self.current_char == Some('&') {
                    self.advance();
                    Ok(Some(Token::And))
                } else {
                    Ok(Some(Token::BitAnd))
                }
            }
            Some('|') => {
                self.advance();
                if self.current_char == Some('|') {
                    self.advance();
                    Ok(Some(Token::Or))
                } else {
                    Ok(Some(Token::BitOr))
                }
            }
            Some('^') => {
                self.advance();
                Ok(Some(Token::BitXor))
            }
            Some('~') => {
                self.advance();
                Ok(Some(Token::BitNot))
            }
            Some('?') => {
                self.advance();
                Ok(Some(Token::Question))
            }
            Some('(') => {
                self.advance();
                Ok(Some(Token::LeftParen))
            }
            Some(')') => {
                self.advance();
                Ok(Some(Token::RightParen))
            }
            Some('{') => {
                self.advance();
                Ok(Some(Token::LeftBrace))
            }
            Some('}') => {
                self.advance();
                Ok(Some(Token::RightBrace))
            }
            Some('[') => {
                self.advance();
                Ok(Some(Token::LeftBracket))
            }
            Some(']') => {
                self.advance();
                Ok(Some(Token::RightBracket))
            }
            Some(',') => {
                self.advance();
                Ok(Some(Token::Comma))
            }
            Some(';') => {
                self.advance();
                Ok(Some(Token::Semicolon))
            }
            Some(':') => {
                self.advance();
                if self.current_char == Some(':') {
                    self.advance();
                    Ok(Some(Token::DoubleColon))
                } else {
                    Ok(Some(Token::Colon))
                }
            }
            Some('.') => {
                self.advance();
                if self.current_char == Some('.') {
                    self.advance();
                    Ok(Some(Token::DotDot))
                } else {
                    Ok(Some(Token::Dot))
                }
            }
            Some('"') => {
                let string = self.read_string()?;
                Ok(Some(Token::String(string)))
            }
            Some(c) if c.is_ascii_digit() => {
                let number = self.read_number()?;
                Ok(Some(number))
            }
            Some(c) if c.is_alphabetic() || c == '_' || !c.is_ascii() => {
                let identifier = self.read_identifier();
                let token = self.keyword_or_identifier(identifier);
                Ok(Some(token))
            }
            Some(c) => Err(anyhow!("Unexpected character: {}", c)),
        }
    }

    fn advance(&mut self) {
        self.position += 1;
        self.current_char = self.input.get(self.position).copied();
    }

    fn skip_whitespace(&mut self) {
        while let Some(c) = self.current_char {
            if c.is_whitespace() && c != '\n' {
                self.advance();
            } else {
                break;
            }
        }
    }

    fn read_string(&mut self) -> Result<String> {
        self.advance(); // Skip opening quote
        let mut value = String::new();
        
        while let Some(c) = self.current_char {
            if c == '"' {
                self.advance(); // Skip closing quote
                return Ok(value);
            }
            if c == '\\' {
                self.advance();
                match self.current_char {
                    Some('n') => value.push('\n'),
                    Some('t') => value.push('\t'),
                    Some('r') => value.push('\r'),
                    Some('\\') => value.push('\\'),
                    Some('"') => value.push('"'),
                    Some(c) => value.push(c),
                    None => return Err(anyhow!("Unexpected end of string")),
                }
            } else {
                value.push(c);
            }
            self.advance();
        }
        
        Err(anyhow!("Unterminated string"))
    }

    fn read_number(&mut self) -> Result<Token> {
        let mut value = String::new();
        let mut is_float = false;
        
        while let Some(c) = self.current_char {
            if c.is_ascii_digit() {
                value.push(c);
                self.advance();
            } else if c == '.' && !is_float {
                is_float = true;
                value.push(c);
                self.advance();
            } else {
                break;
            }
        }
        
        if is_float {
            let float_val = value.parse::<f64>()
                .map_err(|_| anyhow!("Invalid float: {}", value))?;
            Ok(Token::Float(float_val))
        } else {
            let int_val = value.parse::<i64>()
                .map_err(|_| anyhow!("Invalid integer: {}", value))?;
            Ok(Token::Number(int_val))
        }
    }

    fn read_identifier(&mut self) -> String {
        let mut value = String::new();
        
        while let Some(c) = self.current_char {
            if c.is_alphanumeric() || c == '_' || !c.is_ascii() {
                value.push(c);
                self.advance();
            } else {
                break;
            }
        }
        
        value
    }

    fn read_comment(&mut self) -> String {
        let mut comment = String::new();
        
        while let Some(c) = self.current_char {
            if c == '\n' {
                break;
            }
            comment.push(c);
            self.advance();
        }
        
        comment
    }

    fn keyword_or_identifier(&self, word: String) -> Token {
        match word.as_str() {
            // Basic keywords
            "app" => Token::App,
            "fun" => Token::Fun,
            "let" => Token::Let,
            "const" => Token::Const,
            "if" => Token::If,
            "else" => Token::Else,
            "return" => Token::Return,
            "while" => Token::While,
            "for" => Token::For,
            "in" => Token::In,
            "true" => Token::True,
            "false" => Token::False,
            "null" => Token::Null,
            "break" => Token::Break,
            "continue" => Token::Continue,
            
            // Logical operators
            "and" => Token::And,
            "or" => Token::Or,
            "not" => Token::Not,
            
            // Advanced control flow
            "match" => Token::Match,
            "try" => Token::Try,
            "catch" => Token::Catch,
            "finally" => Token::Finally,
            "throw" => Token::Throw,
            "await" => Token::Await,
            "async" => Token::Async,
            
            // Object-oriented
            "class" => Token::Class,
            "trait" => Token::Trait,
            "impl" => Token::Impl,
            "public" => Token::Public,
            "private" => Token::Private,
            "protected" => Token::Protected,
            "static" => Token::Static,
            "extends" => Token::Extends,
            "interface" => Token::Interface,
            "abstract" => Token::Abstract,
            "enum" => Token::Enum,
            "constructor" => Token::Constructor,
            "method" => Token::Method,
            "super" => Token::Super,
            "this" => Token::This,
            "self" => Token::SelfKeyword,
            "override" => Token::Override,
            "virtual" => Token::Virtual,
            "final" => Token::Final,
            
            // Threading & Concurrency
            "spawn" => Token::Spawn,
            "thread" => Token::Thread,
            "lock" => Token::Lock,
            "mutex" => Token::Mutex,
            "channel" => Token::Channel,
            
            // GPU & System
            "gpu" => Token::Gpu,
            "kernel" => Token::Kernel,
            "cuda" => Token::Cuda,
            "opencl" => Token::OpenCL,
            "system" => Token::System,
            "process" => Token::Process,
            "file" => Token::File,
            
            // Math & Science
            "math" => Token::Math,
            "stats" => Token::Stats,
            "matrix" => Token::Matrix,
            "complex" => Token::Complex,
            
            // Memory & Performance
            "unsafe" => Token::Unsafe,
            "safe" => Token::Safe,
            "fast" => Token::Fast,
            "hypercar" => Token::Hypercar,
            
            // Basic types
            "i32" | "int" => Token::TypeInt,
            "f64" | "float" => Token::TypeFloat,
            "str" | "string" => Token::TypeString,
            "bool" | "boolean" => Token::TypeBool,
            "void" => Token::TypeVoid,
            "double" => Token::TypeDouble,
            "char" => Token::TypeChar,
            "byte" => Token::TypeByte,
            "long" => Token::TypeLong,
            
            // Advanced numeric types
            "i8" => Token::TypeInt8,
            "i16" => Token::TypeInt16,
            "i32" => Token::TypeInt32,
            "i64" => Token::TypeInt64,
            "u8" => Token::TypeUInt8,
            "u16" => Token::TypeUInt16,
            "u32" => Token::TypeUInt32,
            "u64" => Token::TypeUInt64,
            "f32" => Token::TypeFloat32,
            "f64" => Token::TypeFloat64,
            "decimal" => Token::TypeDecimal,
            "bigint" => Token::TypeBigInt,
            
            // Math functions
            "sin" => Token::Sin,
            "cos" => Token::Cos,
            "tan" => Token::Tan,
            "asin" => Token::Asin,
            "acos" => Token::Acos,
            "atan" => Token::Atan,
            "sinh" => Token::Sinh,
            "cosh" => Token::Cosh,
            "tanh" => Token::Tanh,
            "log" => Token::Log,
            "log10" => Token::Log10,
            "exp" => Token::Exp,
            "sqrt" => Token::Sqrt,
            "abs" => Token::Abs,
            "floor" => Token::Floor,
            "ceil" => Token::Ceil,
            "round" => Token::Round,
            "min" => Token::Min,
            "max" => Token::Max,
            "atan2" => Token::Atan2,
            "hypot" => Token::Hypot,
            
            // Advanced math
            "factorial" => Token::Factorial,
            "fibonacci" => Token::Fibonacci,
            "gcd" => Token::Gcd,
            "lcm" => Token::Lcm,
            "prime" => Token::Prime,
            
            // Type checking
            "isnan" => Token::IsNaN,
            "isinfinite" => Token::IsInfinite,
            "isfinite" => Token::IsFinite,
            
            _ => Token::Identifier(word),
        }
    }
} 