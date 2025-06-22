use std::collections::HashMap;
use std::fmt;
use crate::ast::{Statement, Expression, Block};

/// Advanced Error Handling System for ScaffoldLang
#[derive(Debug, Clone, PartialEq)]
pub enum ScaffoldError {
    RuntimeError(String),
    TypeError(String),
    IndexError(String),
    ValueError(String),
    NameError(String),
    AttributeError(String),
    ZeroDivisionError,
    OverflowError(String),
    UnderflowError(String),
    NetworkError(String),
    FileNotFoundError(String),
    PermissionError(String),
    TimeoutError(String),
    CustomError(String, String),
}

impl fmt::Display for ScaffoldError {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match self {
            ScaffoldError::RuntimeError(msg) => write!(f, "RuntimeError: {}", msg),
            ScaffoldError::TypeError(msg) => write!(f, "TypeError: {}", msg),
            ScaffoldError::IndexError(msg) => write!(f, "IndexError: {}", msg),
            ScaffoldError::ValueError(msg) => write!(f, "ValueError: {}", msg),
            ScaffoldError::NameError(msg) => write!(f, "NameError: {}", msg),
            ScaffoldError::AttributeError(msg) => write!(f, "AttributeError: {}", msg),
            ScaffoldError::ZeroDivisionError => write!(f, "ZeroDivisionError: division by zero"),
            ScaffoldError::OverflowError(msg) => write!(f, "OverflowError: {}", msg),
            ScaffoldError::UnderflowError(msg) => write!(f, "UnderflowError: {}", msg),
            ScaffoldError::NetworkError(msg) => write!(f, "NetworkError: {}", msg),
            ScaffoldError::FileNotFoundError(msg) => write!(f, "FileNotFoundError: {}", msg),
            ScaffoldError::PermissionError(msg) => write!(f, "PermissionError: {}", msg),
            ScaffoldError::TimeoutError(msg) => write!(f, "TimeoutError: {}", msg),
            ScaffoldError::CustomError(error_type, msg) => write!(f, "{}: {}", error_type, msg),
        }
    }
}

impl std::error::Error for ScaffoldError {}

#[derive(Debug, Clone)]
pub struct StackFrame {
    pub function_name: String,
    pub file_name: String,
    pub line_number: usize,
    pub column_number: usize,
}

#[derive(Debug, Clone)]
pub struct ErrorContext {
    pub error: ScaffoldError,
    pub stack_trace: Vec<StackFrame>,
    pub source_location: Option<(String, usize, usize)>,
}

impl ErrorContext {
    pub fn new(error: ScaffoldError) -> Self {
        Self {
            error,
            stack_trace: Vec::new(),
            source_location: None,
        }
    }

    pub fn with_location(mut self, file: String, line: usize, column: usize) -> Self {
        self.source_location = Some((file, line, column));
        self
    }

    pub fn add_frame(mut self, frame: StackFrame) -> Self {
        self.stack_trace.push(frame);
        self
    }

    pub fn format_error(&self) -> String {
        let mut output = format!("{}", self.error);
        
        if let Some((file, line, column)) = &self.source_location {
            output.push_str(&format!("\n  at {}:{}:{}", file, line, column));
        }

        if !self.stack_trace.is_empty() {
            output.push_str("\nStack trace:");
            for frame in &self.stack_trace {
                output.push_str(&format!(
                    "\n  at {} ({}:{}:{})",
                    frame.function_name, frame.file_name, frame.line_number, frame.column_number
                ));
            }
        }

        output
    }
}

pub struct ErrorHandler {
    pub custom_error_types: HashMap<String, String>,
    pub error_stack: Vec<ErrorContext>,
}

impl ErrorHandler {
    pub fn new() -> Self {
        Self {
            custom_error_types: HashMap::new(),
            error_stack: Vec::new(),
        }
    }

    pub fn register_error_type(&mut self, name: String, description: String) {
        self.custom_error_types.insert(name, description);
    }

    pub fn throw_error(&mut self, error: ScaffoldError, context: Option<ErrorContext>) -> Result<(), ErrorContext> {
        let error_context = context.unwrap_or_else(|| ErrorContext::new(error.clone()));
        self.error_stack.push(error_context.clone());
        Err(error_context)
    }

    pub fn clear_error_stack(&mut self) {
        self.error_stack.clear();
    }

    pub fn get_last_error(&self) -> Option<&ErrorContext> {
        self.error_stack.last()
    }

    pub fn has_errors(&self) -> bool {
        !self.error_stack.is_empty()
    }
}
