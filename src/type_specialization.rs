/// Phase 2: Type Specialization System
/// Target: 10-25M ops/sec (beat Python 2-5x)
/// 
/// This system specializes operations based on known types at compile time,
/// eliminating runtime type checking and enabling direct CPU instruction usage.

use crate::ast::{Expression, Statement, BinaryOperator, Type};
use crate::interpreter::{Value, RuntimeError};
use std::collections::HashMap;

/// Type-specialized instruction set for maximum performance
#[derive(Debug, Clone)]
pub enum SpecializedInstruction {
    // Integer-only operations (no type checking)
    AddI64(i64, i64),           // Direct 64-bit integer addition
    SubI64(i64, i64),           // Direct 64-bit integer subtraction  
    MulI64(i64, i64),           // Direct 64-bit integer multiplication
    DivI64(i64, i64),           // Direct 64-bit integer division
    
    // Float-only operations (no type checking)
    AddF64(f64, f64),           // Direct 64-bit float addition
    SubF64(f64, f64),           // Direct 64-bit float subtraction
    MulF64(f64, f64),           // Direct 64-bit float multiplication
    DivF64(f64, f64),           // Direct 64-bit float division
    
    // Specialized comparisons
    CmpI64Less(i64, i64),       // Integer comparison <
    CmpI64Greater(i64, i64),    // Integer comparison >
    CmpF64Less(f64, f64),       // Float comparison <
    CmpF64Greater(f64, f64),    // Float comparison >
    
    // Type-specialized constants
    PushI64(i64),               // Push known integer
    PushF64(f64),               // Push known float
    PushBool(bool),             // Push known boolean
    
    // Specialized variable operations
    LoadI64Var(String),         // Load known integer variable
    LoadF64Var(String),         // Load known float variable
    StoreI64Var(String),        // Store to known integer variable
    StoreF64Var(String),        // Store to known float variable
    
    // Specialized mathematical functions
    SqrtF64,                    // Fast float square root
    PowF64,                     // Fast float power
    SinF64,                     // Fast float sine
    CosF64,                     // Fast float cosine
    
    // Loop optimizations
    CounterLoop {               // Specialized integer counter loop
        counter_var: String,
        start: i64,
        end: i64,
        step: i64,
        body_start: usize,
        body_end: usize,
    },
    
    // Memory-efficient operations
    InplaceAddI64(String),      // In-place integer addition
    InplaceSubI64(String),      // In-place integer subtraction
    InplaceMulI64(String),      // In-place integer multiplication
    
    // Control flow
    JumpIfFalseI64(usize),      // Jump based on integer zero check
    JumpIfFalseF64(usize),      // Jump based on float zero check
    
    Halt,
}

/// Type information for variables and expressions
#[derive(Debug, Clone, PartialEq)]
pub enum SpecializedType {
    Int64,
    Float64,
    Boolean,
    String,
    Unknown,
}

/// Type analysis and specialization engine
pub struct TypeSpecializer {
    variable_types: HashMap<String, SpecializedType>,
    specialized_instructions: Vec<SpecializedInstruction>,
    type_hints: HashMap<String, SpecializedType>,
}

impl TypeSpecializer {
    pub fn new() -> Self {
        Self {
            variable_types: HashMap::new(),
            specialized_instructions: Vec::new(),
            type_hints: HashMap::new(),
        }
    }
    
    /// Analyze and specialize a program for maximum performance
    pub fn specialize_program(&mut self, statements: &[Statement]) -> Result<Vec<SpecializedInstruction>, RuntimeError> {
        self.specialized_instructions.clear();
        
        // Phase 1: Type inference pass
        self.infer_types(statements)?;
        
        // Phase 2: Generate specialized instructions
        for statement in statements {
            self.specialize_statement(statement)?;
        }
        
        self.specialized_instructions.push(SpecializedInstruction::Halt);
        Ok(self.specialized_instructions.clone())
    }
    
    /// Infer types for all variables and expressions
    fn infer_types(&mut self, statements: &[Statement]) -> Result<(), RuntimeError> {
        for statement in statements {
            match statement {
                Statement::Let { name, value, var_type } => {
                    let inferred_type = if let Some(explicit_type) = var_type {
                        self.type_to_specialized(explicit_type)
                    } else {
                        self.infer_expression_type(value)?
                    };
                    self.variable_types.insert(name.clone(), inferred_type);
                }
                Statement::Assignment { name, value } => {
                    if let Some(existing_type) = self.variable_types.get(name) {
                        let value_type = self.infer_expression_type(value)?;
                        if *existing_type != value_type {
                            // Type mismatch - keep as unknown for safety
                            self.variable_types.insert(name.clone(), SpecializedType::Unknown);
                        }
                    }
                }
                Statement::While { condition, body } => {
                    self.infer_expression_type(condition)?;
                    self.infer_types(body)?;
                }
                _ => {}
            }
        }
        Ok(())
    }
    
    /// Infer the type of an expression
    fn infer_expression_type(&self, expr: &Expression) -> Result<SpecializedType, RuntimeError> {
        match expr {
            Expression::Number(_) => Ok(SpecializedType::Int64),
            Expression::Float(_) => Ok(SpecializedType::Float64),
            Expression::Boolean(_) => Ok(SpecializedType::Boolean),
            Expression::String(_) => Ok(SpecializedType::String),
            Expression::Identifier(name) => {
                Ok(self.variable_types.get(name).unwrap_or(&SpecializedType::Unknown).clone())
            }
            Expression::Binary { left, operator: _, right } => {
                let left_type = self.infer_expression_type(left)?;
                let right_type = self.infer_expression_type(right)?;
                
                match (left_type, right_type) {
                    (SpecializedType::Int64, SpecializedType::Int64) => Ok(SpecializedType::Int64),
                    (SpecializedType::Float64, SpecializedType::Float64) => Ok(SpecializedType::Float64),
                    (SpecializedType::Int64, SpecializedType::Float64) => Ok(SpecializedType::Float64),
                    (SpecializedType::Float64, SpecializedType::Int64) => Ok(SpecializedType::Float64),
                    _ => Ok(SpecializedType::Unknown),
                }
            }
            _ => Ok(SpecializedType::Unknown),
        }
    }
    
    /// Convert AST type to specialized type
    fn type_to_specialized(&self, ast_type: &Type) -> SpecializedType {
        match ast_type {
            Type::Int | Type::Int64 => SpecializedType::Int64,
            Type::Float | Type::Float64 => SpecializedType::Float64,
            Type::Bool => SpecializedType::Boolean,
            Type::String => SpecializedType::String,
            _ => SpecializedType::Unknown,
        }
    }
    
    /// Generate specialized instructions for a statement
    fn specialize_statement(&mut self, statement: &Statement) -> Result<(), RuntimeError> {
        match statement {
            Statement::Let { name, value, .. } => {
                self.specialize_expression(value)?;
                
                if let Some(var_type) = self.variable_types.get(name) {
                    match var_type {
                        SpecializedType::Int64 => {
                            self.specialized_instructions.push(SpecializedInstruction::StoreI64Var(name.clone()));
                        }
                        SpecializedType::Float64 => {
                            self.specialized_instructions.push(SpecializedInstruction::StoreF64Var(name.clone()));
                        }
                        _ => {
                            // Fall back to generic storage
                            return Err(RuntimeError::TypeError("Unsupported type for specialization".to_string()));
                        }
                    }
                }
            }
            
            Statement::Assignment { name, value } => {
                self.specialize_expression(value)?;
                
                if let Some(var_type) = self.variable_types.get(name) {
                    match var_type {
                        SpecializedType::Int64 => {
                            self.specialized_instructions.push(SpecializedInstruction::StoreI64Var(name.clone()));
                        }
                        SpecializedType::Float64 => {
                            self.specialized_instructions.push(SpecializedInstruction::StoreF64Var(name.clone()));
                        }
                        _ => {
                            return Err(RuntimeError::TypeError("Unsupported type for specialization".to_string()));
                        }
                    }
                }
            }
            
            Statement::While { condition, body } => {
                let start_label = self.specialized_instructions.len();
                
                // Specialize condition
                self.specialize_expression(condition)?;
                
                // Add specialized conditional jump
                let jump_pos = self.specialized_instructions.len();
                self.specialized_instructions.push(SpecializedInstruction::JumpIfFalseI64(0)); // Will be patched
                
                // Specialize body
                for stmt in body {
                    self.specialize_statement(stmt)?;
                }
                
                // Jump back to condition
                self.specialized_instructions.push(SpecializedInstruction::JumpIfFalseI64(start_label));
                
                // Patch the conditional jump
                let end_label = self.specialized_instructions.len();
                self.specialized_instructions[jump_pos] = SpecializedInstruction::JumpIfFalseI64(end_label);
            }
            
            Statement::Expression(expr) => {
                self.specialize_expression(expr)?;
            }
            
            _ => {
                return Err(RuntimeError::InvalidOperation("Statement not supported in type specialization".to_string()));
            }
        }
        Ok(())
    }
    
    /// Generate specialized instructions for an expression
    fn specialize_expression(&mut self, expr: &Expression) -> Result<(), RuntimeError> {
        match expr {
            Expression::Number(n) => {
                self.specialized_instructions.push(SpecializedInstruction::PushI64(*n));
            }
            
            Expression::Float(f) => {
                self.specialized_instructions.push(SpecializedInstruction::PushF64(*f));
            }
            
            Expression::Boolean(b) => {
                self.specialized_instructions.push(SpecializedInstruction::PushBool(*b));
            }
            
            Expression::Identifier(name) => {
                if let Some(var_type) = self.variable_types.get(name) {
                    match var_type {
                        SpecializedType::Int64 => {
                            self.specialized_instructions.push(SpecializedInstruction::LoadI64Var(name.clone()));
                        }
                        SpecializedType::Float64 => {
                            self.specialized_instructions.push(SpecializedInstruction::LoadF64Var(name.clone()));
                        }
                        _ => {
                            return Err(RuntimeError::TypeError("Unsupported variable type".to_string()));
                        }
                    }
                }
            }
            
            Expression::Binary { left, operator, right } => {
                self.specialize_expression(left)?;
                self.specialize_expression(right)?;
                
                let left_type = self.infer_expression_type(left)?;
                let right_type = self.infer_expression_type(right)?;
                
                match (left_type, right_type, operator) {
                    (SpecializedType::Int64, SpecializedType::Int64, BinaryOperator::Add) => {
                        self.specialized_instructions.push(SpecializedInstruction::AddI64(0, 0));
                    }
                    (SpecializedType::Int64, SpecializedType::Int64, BinaryOperator::Subtract) => {
                        self.specialized_instructions.push(SpecializedInstruction::SubI64(0, 0));
                    }
                    (SpecializedType::Int64, SpecializedType::Int64, BinaryOperator::Multiply) => {
                        self.specialized_instructions.push(SpecializedInstruction::MulI64(0, 0));
                    }
                    (SpecializedType::Int64, SpecializedType::Int64, BinaryOperator::Divide) => {
                        self.specialized_instructions.push(SpecializedInstruction::DivI64(0, 0));
                    }
                    (SpecializedType::Float64, SpecializedType::Float64, BinaryOperator::Add) => {
                        self.specialized_instructions.push(SpecializedInstruction::AddF64(0.0, 0.0));
                    }
                    (SpecializedType::Float64, SpecializedType::Float64, BinaryOperator::Subtract) => {
                        self.specialized_instructions.push(SpecializedInstruction::SubF64(0.0, 0.0));
                    }
                    (SpecializedType::Float64, SpecializedType::Float64, BinaryOperator::Multiply) => {
                        self.specialized_instructions.push(SpecializedInstruction::MulF64(0.0, 0.0));
                    }
                    (SpecializedType::Float64, SpecializedType::Float64, BinaryOperator::Divide) => {
                        self.specialized_instructions.push(SpecializedInstruction::DivF64(0.0, 0.0));
                    }
                    (SpecializedType::Int64, SpecializedType::Int64, BinaryOperator::Less) => {
                        self.specialized_instructions.push(SpecializedInstruction::CmpI64Less(0, 0));
                    }
                    (SpecializedType::Int64, SpecializedType::Int64, BinaryOperator::Greater) => {
                        self.specialized_instructions.push(SpecializedInstruction::CmpI64Greater(0, 0));
                    }
                    _ => {
                        return Err(RuntimeError::TypeError("Unsupported operation for type specialization".to_string()));
                    }
                }
            }
            
            Expression::Call { function, arguments } => {
                // Specialize mathematical functions
                match function.as_str() {
                    "sqrt" => {
                        if arguments.len() == 1 {
                            self.specialize_expression(&arguments[0])?;
                            self.specialized_instructions.push(SpecializedInstruction::SqrtF64);
                        }
                    }
                    "pow" => {
                        if arguments.len() == 2 {
                            self.specialize_expression(&arguments[0])?;
                            self.specialize_expression(&arguments[1])?;
                            self.specialized_instructions.push(SpecializedInstruction::PowF64);
                        }
                    }
                    "sin" => {
                        if arguments.len() == 1 {
                            self.specialize_expression(&arguments[0])?;
                            self.specialized_instructions.push(SpecializedInstruction::SinF64);
                        }
                    }
                    "cos" => {
                        if arguments.len() == 1 {
                            self.specialize_expression(&arguments[0])?;
                            self.specialized_instructions.push(SpecializedInstruction::CosF64);
                        }
                    }
                    _ => {
                        return Err(RuntimeError::InvalidOperation("Function not supported in type specialization".to_string()));
                    }
                }
            }
            
            _ => {
                return Err(RuntimeError::InvalidOperation("Expression not supported in type specialization".to_string()));
            }
        }
        Ok(())
    }
}

/// Type-specialized virtual machine for maximum performance
pub struct SpecializedVM {
    instructions: Vec<SpecializedInstruction>,
    i64_stack: Vec<i64>,
    f64_stack: Vec<f64>,
    bool_stack: Vec<bool>,
    i64_variables: HashMap<String, i64>,
    f64_variables: HashMap<String, f64>,
    pc: usize,
}

impl SpecializedVM {
    pub fn new() -> Self {
        Self {
            instructions: Vec::new(),
            i64_stack: Vec::new(),
            f64_stack: Vec::new(),
            bool_stack: Vec::new(),
            i64_variables: HashMap::new(),
            f64_variables: HashMap::new(),
            pc: 0,
        }
    }
    
    /// Execute specialized instructions at maximum speed
    pub fn execute(&mut self, instructions: Vec<SpecializedInstruction>) -> Result<Value, RuntimeError> {
        self.instructions = instructions;
        self.pc = 0;
        self.i64_stack.clear();
        self.f64_stack.clear();
        self.bool_stack.clear();
        
        loop {
            if self.pc >= self.instructions.len() {
                break;
            }
            
            match &self.instructions[self.pc] {
                SpecializedInstruction::PushI64(value) => {
                    self.i64_stack.push(*value);
                }
                
                SpecializedInstruction::PushF64(value) => {
                    self.f64_stack.push(*value);
                }
                
                SpecializedInstruction::PushBool(value) => {
                    self.bool_stack.push(*value);
                }
                
                SpecializedInstruction::LoadI64Var(name) => {
                    if let Some(value) = self.i64_variables.get(name) {
                        self.i64_stack.push(*value);
                    } else {
                        return Err(RuntimeError::NameError(format!("Variable '{}' not found", name)));
                    }
                }
                
                SpecializedInstruction::LoadF64Var(name) => {
                    if let Some(value) = self.f64_variables.get(name) {
                        self.f64_stack.push(*value);
                    } else {
                        return Err(RuntimeError::NameError(format!("Variable '{}' not found", name)));
                    }
                }
                
                SpecializedInstruction::StoreI64Var(name) => {
                    if let Some(value) = self.i64_stack.pop() {
                        self.i64_variables.insert(name.clone(), value);
                    } else {
                        return Err(RuntimeError::ValueError("Stack underflow".to_string()));
                    }
                }
                
                SpecializedInstruction::StoreF64Var(name) => {
                    if let Some(value) = self.f64_stack.pop() {
                        self.f64_variables.insert(name.clone(), value);
                    } else {
                        return Err(RuntimeError::ValueError("Stack underflow".to_string()));
                    }
                }
                
                // Specialized integer operations (no type checking = maximum speed)
                SpecializedInstruction::AddI64(_, _) => {
                    let b = self.i64_stack.pop().ok_or_else(|| RuntimeError::ValueError("Stack underflow".to_string()))?;
                    let a = self.i64_stack.pop().ok_or_else(|| RuntimeError::ValueError("Stack underflow".to_string()))?;
                    self.i64_stack.push(a + b);
                }
                
                SpecializedInstruction::SubI64(_, _) => {
                    let b = self.i64_stack.pop().ok_or_else(|| RuntimeError::ValueError("Stack underflow".to_string()))?;
                    let a = self.i64_stack.pop().ok_or_else(|| RuntimeError::ValueError("Stack underflow".to_string()))?;
                    self.i64_stack.push(a - b);
                }
                
                SpecializedInstruction::MulI64(_, _) => {
                    let b = self.i64_stack.pop().ok_or_else(|| RuntimeError::ValueError("Stack underflow".to_string()))?;
                    let a = self.i64_stack.pop().ok_or_else(|| RuntimeError::ValueError("Stack underflow".to_string()))?;
                    self.i64_stack.push(a * b);
                }
                
                SpecializedInstruction::DivI64(_, _) => {
                    let b = self.i64_stack.pop().ok_or_else(|| RuntimeError::ValueError("Stack underflow".to_string()))?;
                    let a = self.i64_stack.pop().ok_or_else(|| RuntimeError::ValueError("Stack underflow".to_string()))?;
                    if b == 0 {
                        return Err(RuntimeError::DivisionByZero);
                    }
                    self.i64_stack.push(a / b);
                }
                
                // Specialized float operations (no type checking = maximum speed)
                SpecializedInstruction::AddF64(_, _) => {
                    let b = self.f64_stack.pop().ok_or_else(|| RuntimeError::ValueError("Stack underflow".to_string()))?;
                    let a = self.f64_stack.pop().ok_or_else(|| RuntimeError::ValueError("Stack underflow".to_string()))?;
                    self.f64_stack.push(a + b);
                }
                
                SpecializedInstruction::SubF64(_, _) => {
                    let b = self.f64_stack.pop().ok_or_else(|| RuntimeError::ValueError("Stack underflow".to_string()))?;
                    let a = self.f64_stack.pop().ok_or_else(|| RuntimeError::ValueError("Stack underflow".to_string()))?;
                    self.f64_stack.push(a - b);
                }
                
                SpecializedInstruction::MulF64(_, _) => {
                    let b = self.f64_stack.pop().ok_or_else(|| RuntimeError::ValueError("Stack underflow".to_string()))?;
                    let a = self.f64_stack.pop().ok_or_else(|| RuntimeError::ValueError("Stack underflow".to_string()))?;
                    self.f64_stack.push(a * b);
                }
                
                SpecializedInstruction::DivF64(_, _) => {
                    let b = self.f64_stack.pop().ok_or_else(|| RuntimeError::ValueError("Stack underflow".to_string()))?;
                    let a = self.f64_stack.pop().ok_or_else(|| RuntimeError::ValueError("Stack underflow".to_string()))?;
                    if b == 0.0 {
                        return Err(RuntimeError::DivisionByZero);
                    }
                    self.f64_stack.push(a / b);
                }
                
                // Specialized mathematical functions
                SpecializedInstruction::SqrtF64 => {
                    let value = self.f64_stack.pop().ok_or_else(|| RuntimeError::ValueError("Stack underflow".to_string()))?;
                    self.f64_stack.push(value.sqrt());
                }
                
                SpecializedInstruction::PowF64 => {
                    let exponent = self.f64_stack.pop().ok_or_else(|| RuntimeError::ValueError("Stack underflow".to_string()))?;
                    let base = self.f64_stack.pop().ok_or_else(|| RuntimeError::ValueError("Stack underflow".to_string()))?;
                    self.f64_stack.push(base.powf(exponent));
                }
                
                SpecializedInstruction::SinF64 => {
                    let value = self.f64_stack.pop().ok_or_else(|| RuntimeError::ValueError("Stack underflow".to_string()))?;
                    self.f64_stack.push(value.sin());
                }
                
                SpecializedInstruction::CosF64 => {
                    let value = self.f64_stack.pop().ok_or_else(|| RuntimeError::ValueError("Stack underflow".to_string()))?;
                    self.f64_stack.push(value.cos());
                }
                
                // Specialized comparisons
                SpecializedInstruction::CmpI64Less(_, _) => {
                    let b = self.i64_stack.pop().ok_or_else(|| RuntimeError::ValueError("Stack underflow".to_string()))?;
                    let a = self.i64_stack.pop().ok_or_else(|| RuntimeError::ValueError("Stack underflow".to_string()))?;
                    self.bool_stack.push(a < b);
                }
                
                SpecializedInstruction::CmpI64Greater(_, _) => {
                    let b = self.i64_stack.pop().ok_or_else(|| RuntimeError::ValueError("Stack underflow".to_string()))?;
                    let a = self.i64_stack.pop().ok_or_else(|| RuntimeError::ValueError("Stack underflow".to_string()))?;
                    self.bool_stack.push(a > b);
                }
                
                SpecializedInstruction::JumpIfFalseI64(addr) => {
                    if let Some(condition) = self.bool_stack.pop() {
                        if !condition {
                            self.pc = *addr;
                            continue;
                        }
                    }
                }
                
                SpecializedInstruction::Halt => {
                    break;
                }
                
                _ => {
                    return Err(RuntimeError::InvalidOperation("Instruction not implemented in specialized VM".to_string()));
                }
            }
            
            self.pc += 1;
        }
        
        // Return result from appropriate stack
        if let Some(i64_result) = self.i64_stack.last() {
            Ok(Value::Integer(*i64_result))
        } else if let Some(f64_result) = self.f64_stack.last() {
            Ok(Value::Float(*f64_result))
        } else if let Some(bool_result) = self.bool_stack.last() {
            Ok(Value::Boolean(*bool_result))
        } else {
            Ok(Value::Null)
        }
    }
} 