/// Phase 3: Escape Analysis for Stack Allocation Optimization
/// Target: Automatically detect when objects can be stack-allocated instead of heap-allocated
/// 
/// This system analyzes variable lifetimes and usage patterns to determine when objects
/// can be safely allocated on the stack, eliminating heap allocations and GC pressure.

use std::collections::{HashMap, HashSet};
use crate::ast::{Statement, Expression, BinaryOperator, Function, Block};
use crate::interpreter::{Value, RuntimeError};
use crate::execution_profiler::ExecutionProfiler;

/// Escape analysis engine for determining allocation strategies
pub struct EscapeAnalyzer {
    /// Analysis results for each variable
    pub escape_info: HashMap<String, EscapeInfo>,
    
    /// Function-level analysis
    pub function_analysis: HashMap<String, FunctionEscapeInfo>,
    
    /// Global variables that always escape
    pub global_escapes: HashSet<String>,
    
    /// Statistics for optimization effectiveness
    pub stats: EscapeAnalysisStats,
}

/// Information about whether a variable escapes its scope
#[derive(Debug, Clone)]
pub struct EscapeInfo {
    /// Whether this variable escapes to the heap
    pub escapes: bool,
    
    /// Reason for escape decision
    pub escape_reason: EscapeReason,
    
    /// Lifetime scope of the variable
    pub lifetime: VariableLifetime,
    
    /// Size estimate for stack allocation feasibility
    pub estimated_size: usize,
    
    /// Whether this can be allocated in registers
    pub register_candidate: bool,
    
    /// Optimization potential score (0.0-1.0)
    pub optimization_potential: f64,
}

/// Reasons why a variable might escape
#[derive(Debug, Clone)]
pub enum EscapeReason {
    /// Variable doesn't escape - can be stack allocated
    NoEscape,
    
    /// Returned from function
    ReturnedFromFunction,
    
    /// Assigned to global variable
    AssignedToGlobal,
    
    /// Passed to external function
    PassedToExternal,
    
    /// Address taken (references created)
    AddressTaken,
    
    /// Stored in heap-allocated structure
    StoredInHeapObject,
    
    /// Escapes through closure capture
    ClosureCapture,
    
    /// Size too large for stack
    TooLargeForStack,
    
    /// Lifetime extends beyond function scope
    LongLifetime,
}

/// Variable lifetime information
#[derive(Debug, Clone)]
pub struct VariableLifetime {
    /// Scope depth where variable is defined
    pub definition_scope: usize,
    
    /// Last scope where variable is used
    pub last_use_scope: usize,
    
    /// Whether variable is used after function returns
    pub used_after_return: bool,
    
    /// Estimated lifetime in instructions
    pub instruction_lifetime: usize,
}

/// Function-level escape analysis
#[derive(Debug, Clone)]
pub struct FunctionEscapeInfo {
    /// Parameters that don't escape
    pub non_escaping_params: HashSet<String>,
    
    /// Local variables that can be stack allocated
    pub stack_allocatable_locals: HashSet<String>,
    
    /// Variables that can be allocated in registers
    pub register_allocatable: HashSet<String>,
    
    /// Total stack space needed for optimized allocation
    pub stack_space_needed: usize,
    
    /// Optimization opportunities
    pub optimization_opportunities: Vec<StackAllocationOpportunity>,
}

/// Specific optimization opportunity
#[derive(Debug, Clone)]
pub struct StackAllocationOpportunity {
    pub variable_name: String,
    pub opportunity_type: OptimizationType,
    pub estimated_speedup: f64,
    pub memory_saved: usize,
}

#[derive(Debug, Clone)]
pub enum OptimizationType {
    /// Move from heap to stack
    HeapToStack,
    
    /// Move from stack to registers
    StackToRegister,
    
    /// Eliminate allocation entirely (constant folding)
    EliminateAllocation,
    
    /// Merge multiple allocations
    MergeAllocations,
}

/// Statistics for escape analysis effectiveness
#[derive(Debug, Default)]
pub struct EscapeAnalysisStats {
    pub total_variables_analyzed: usize,
    pub variables_optimized: usize,
    pub heap_allocations_eliminated: usize,
    pub stack_allocations_created: usize,
    pub register_allocations_created: usize,
    pub estimated_memory_saved: usize,
    pub estimated_speedup: f64,
}

impl EscapeAnalyzer {
    pub fn new() -> Self {
        Self {
            escape_info: HashMap::new(),
            function_analysis: HashMap::new(),
            global_escapes: HashSet::new(),
            stats: EscapeAnalysisStats::default(),
        }
    }
    
    /// Analyze escape behavior for an entire function
    pub fn analyze_function(&mut self, function: &Function) -> Result<FunctionEscapeInfo, RuntimeError> {
        println!("🔍 Analyzing escape behavior for function: {}", function.name);
        
        let mut func_info = FunctionEscapeInfo {
            non_escaping_params: HashSet::new(),
            stack_allocatable_locals: HashSet::new(),
            register_allocatable: HashSet::new(),
            stack_space_needed: 0,
            optimization_opportunities: Vec::new(),
        };
        
        // Phase 1: Analyze parameter escape behavior
        self.analyze_parameters(&function.parameters, &mut func_info)?;
        
        // Phase 2: Analyze function body
        self.analyze_block(&function.body, 0, &mut func_info)?;
        
        // Phase 3: Determine optimization opportunities
        self.identify_optimization_opportunities(&mut func_info);
        
        // Phase 4: Calculate stack space requirements
        func_info.stack_space_needed = self.calculate_stack_space(&func_info);
        
        self.function_analysis.insert(function.name.clone(), func_info.clone());
        
        println!("✅ Escape analysis complete: {} optimizations found", 
                func_info.optimization_opportunities.len());
        
        Ok(func_info)
    }
    
    /// Analyze parameter escape behavior
    fn analyze_parameters(
        &mut self,
        parameters: &[crate::ast::Parameter],
        func_info: &mut FunctionEscapeInfo,
    ) -> Result<(), RuntimeError> {
        for param in parameters {
            let escape_info = EscapeInfo {
                escapes: false, // Parameters start as non-escaping
                escape_reason: EscapeReason::NoEscape,
                lifetime: VariableLifetime {
                    definition_scope: 0,
                    last_use_scope: 0,
                    used_after_return: false,
                    instruction_lifetime: 0,
                },
                estimated_size: self.estimate_type_size(&param.param_type),
                register_candidate: self.is_register_candidate(&param.param_type),
                optimization_potential: 1.0,
            };
            
            self.escape_info.insert(param.name.clone(), escape_info);
            
            // Parameters that are simple types and small can be non-escaping
            if self.is_simple_type(&param.param_type) && 
               self.estimate_type_size(&param.param_type) <= 8 {
                func_info.non_escaping_params.insert(param.name.clone());
                
                if self.is_register_candidate(&param.param_type) {
                    func_info.register_allocatable.insert(param.name.clone());
                }
            }
        }
        
        Ok(())
    }
    
    /// Analyze a block of statements
    fn analyze_block(
        &mut self,
        block: &Block,
        scope_depth: usize,
        func_info: &mut FunctionEscapeInfo,
    ) -> Result<(), RuntimeError> {
        for statement in &block.statements {
            self.analyze_statement(statement, scope_depth, func_info)?;
        }
        
        Ok(())
    }
    
    /// Analyze a single statement
    fn analyze_statement(
        &mut self,
        statement: &Statement,
        scope_depth: usize,
        func_info: &mut FunctionEscapeInfo,
    ) -> Result<(), RuntimeError> {
        match statement {
            Statement::Let { name, value, var_type } => {
                self.analyze_variable_declaration(name, value, var_type.as_ref(), scope_depth, func_info)?;
            }
            Statement::Assignment { name, value } => {
                self.analyze_assignment(name, value, scope_depth, func_info)?;
            }
            Statement::While { condition, body } => {
                self.analyze_expression(condition, scope_depth, func_info)?;
                self.analyze_block(body, scope_depth + 1, func_info)?;
            }
            Statement::If { condition, then_branch, else_branch } => {
                self.analyze_expression(condition, scope_depth, func_info)?;
                self.analyze_block(then_branch, scope_depth + 1, func_info)?;
                if let Some(else_block) = else_branch {
                    self.analyze_block(else_block, scope_depth + 1, func_info)?;
                }
            }
            Statement::Return { value } => {
                if let Some(expr) = value {
                    self.analyze_expression(expr, scope_depth, func_info)?;
                    // Mark returned variables as escaping
                    self.mark_expression_as_escaping(expr, EscapeReason::ReturnedFromFunction);
                }
            }
            Statement::Expression(expr) => {
                self.analyze_expression(expr, scope_depth, func_info)?;
            }
            _ => {}
        }
        
        Ok(())
    }
    
    /// Analyze variable declaration
    fn analyze_variable_declaration(
        &mut self,
        name: &str,
        value: &Expression,
        var_type: Option<&crate::ast::Type>,
        scope_depth: usize,
        func_info: &mut FunctionEscapeInfo,
    ) -> Result<(), RuntimeError> {
        // Analyze the value expression
        self.analyze_expression(value, scope_depth, func_info)?;
        
        // Determine if this variable can be stack allocated
        let estimated_size = if let Some(t) = var_type {
            self.estimate_type_size(t)
        } else {
            self.estimate_expression_size(value)
        };
        
        let escapes = self.does_variable_escape(name, value, scope_depth);
        let escape_reason = if escapes {
            self.determine_escape_reason(name, value, scope_depth)
        } else {
            EscapeReason::NoEscape
        };
        
        let lifetime = VariableLifetime {
            definition_scope: scope_depth,
            last_use_scope: scope_depth,
            used_after_return: false,
            instruction_lifetime: self.estimate_instruction_lifetime(value),
        };
        
        let escape_info = EscapeInfo {
            escapes,
            escape_reason,
            lifetime,
            estimated_size,
            register_candidate: self.is_register_candidate_from_size(estimated_size) &&
                              self.is_simple_expression(value),
            optimization_potential: self.calculate_optimization_potential(estimated_size, escapes),
        };
        
        self.escape_info.insert(name.to_string(), escape_info);
        
        // Add to appropriate optimization sets
        if !escapes {
            if estimated_size <= 8 && self.is_simple_expression(value) {
                func_info.register_allocatable.insert(name.to_string());
            } else if estimated_size <= 1024 { // Stack limit
                func_info.stack_allocatable_locals.insert(name.to_string());
            }
        }
        
        self.stats.total_variables_analyzed += 1;
        
        Ok(())
    }
    
    /// Analyze assignment statement
    fn analyze_assignment(
        &mut self,
        name: &str,
        value: &Expression,
        scope_depth: usize,
        func_info: &mut FunctionEscapeInfo,
    ) -> Result<(), RuntimeError> {
        self.analyze_expression(value, scope_depth, func_info)?;
        
        // Update escape info if variable already exists
        let mut should_update = false;
        let mut new_escapes = false;
        let mut new_escape_reason = EscapeReason::NoEscape;
        
        if let Some(info) = self.escape_info.get_mut(name) {
            info.lifetime.last_use_scope = scope_depth;
            should_update = true;
        }
        
        if should_update {
            // Check if assignment causes escape
            let causes_escape = self.does_assignment_cause_escape(name, value);
            if causes_escape {
                new_escapes = true;
                new_escape_reason = self.determine_escape_reason(name, value, scope_depth);
            }
        }
        
        if new_escapes {
            if let Some(info) = self.escape_info.get_mut(name) {
                info.escapes = true;
                info.escape_reason = new_escape_reason;
            }
            
            // Remove from optimization sets
            func_info.register_allocatable.remove(name);
            func_info.stack_allocatable_locals.remove(name);
        }
        
        Ok(())
    }
    
    /// Analyze expression for escape behavior
    fn analyze_expression(
        &mut self,
        expr: &Expression,
        scope_depth: usize,
        func_info: &mut FunctionEscapeInfo,
    ) -> Result<(), RuntimeError> {
        match expr {
            Expression::Identifier(name) => {
                // Update last use scope
                if let Some(info) = self.escape_info.get_mut(name) {
                    info.lifetime.last_use_scope = scope_depth;
                }
            }
            Expression::Binary { left, operator, right } => {
                self.analyze_expression(left, scope_depth, func_info)?;
                self.analyze_expression(right, scope_depth, func_info)?;
                
                // Check for operations that might cause escape
                self.check_binary_operation_escape(left, operator, right);
            }
            Expression::Call { function: _, arguments } => {
                for arg in arguments {
                    self.analyze_expression(arg, scope_depth, func_info)?;
                    // Function calls might cause arguments to escape
                    self.mark_expression_as_escaping(arg, EscapeReason::PassedToExternal);
                }
            }
            Expression::Array(elements) => {
                for element in elements {
                    self.analyze_expression(element, scope_depth, func_info)?;
                }
            }
            _ => {}
        }
        
        Ok(())
    }
    
    /// Determine if a variable escapes its local scope
    fn does_variable_escape(&self, _name: &str, value: &Expression, scope_depth: usize) -> bool {
        // Simple heuristics for escape analysis
        
        // Large objects typically escape
        if self.estimate_expression_size(value) > 1024 {
            return true;
        }
        
        // Objects that outlive their scope escape
        if scope_depth > 0 && self.has_long_lifetime(value) {
            return true;
        }
        
        // Arrays and complex objects often escape
        if self.is_complex_expression(value) {
            return true;
        }
        
        false
    }
    
    /// Determine the reason for escape
    fn determine_escape_reason(&self, _name: &str, value: &Expression, scope_depth: usize) -> EscapeReason {
        if self.estimate_expression_size(value) > 1024 {
            EscapeReason::TooLargeForStack
        } else if scope_depth > 0 && self.has_long_lifetime(value) {
            EscapeReason::LongLifetime
        } else if self.is_complex_expression(value) {
            EscapeReason::StoredInHeapObject
        } else {
            EscapeReason::NoEscape
        }
    }
    
    /// Mark an expression as escaping
    fn mark_expression_as_escaping(&mut self, expr: &Expression, reason: EscapeReason) {
        if let Expression::Identifier(name) = expr {
            if let Some(info) = self.escape_info.get_mut(name) {
                info.escapes = true;
                info.escape_reason = reason;
            }
        }
    }
    
    /// Check if assignment causes variable to escape
    fn does_assignment_cause_escape(&self, _name: &str, value: &Expression) -> bool {
        // If we're assigning a complex expression, it might cause escape
        self.is_complex_expression(value) || self.estimate_expression_size(value) > 1024
    }
    
    /// Check if binary operation causes escape
    fn check_binary_operation_escape(&mut self, _left: &Expression, _operator: &BinaryOperator, _right: &Expression) {
        // Most binary operations don't cause escape for simple types
        // More sophisticated analysis would check for address-taking operations
    }
    
    /// Identify optimization opportunities
    fn identify_optimization_opportunities(&mut self, func_info: &mut FunctionEscapeInfo) {
        for (name, info) in &self.escape_info {
            if !info.escapes {
                let opportunity_type = if info.register_candidate {
                    OptimizationType::StackToRegister
                } else if info.estimated_size <= 1024 {
                    OptimizationType::HeapToStack
                } else {
                    continue;
                };
                
                let opportunity = StackAllocationOpportunity {
                    variable_name: name.clone(),
                    opportunity_type: opportunity_type.clone(),
                    estimated_speedup: self.calculate_speedup_for_optimization(&opportunity_type, info.estimated_size),
                    memory_saved: info.estimated_size,
                };
                
                func_info.optimization_opportunities.push(opportunity);
                self.stats.variables_optimized += 1;
                
                match opportunity_type {
                    OptimizationType::HeapToStack => {
                        self.stats.heap_allocations_eliminated += 1;
                        self.stats.stack_allocations_created += 1;
                    }
                    OptimizationType::StackToRegister => {
                        self.stats.register_allocations_created += 1;
                    }
                    _ => {}
                }
                
                self.stats.estimated_memory_saved += info.estimated_size;
            }
        }
        
        // Calculate overall speedup estimate
        self.stats.estimated_speedup = self.calculate_overall_speedup();
    }
    
    /// Calculate stack space needed for optimized allocation
    fn calculate_stack_space(&self, func_info: &FunctionEscapeInfo) -> usize {
        func_info.stack_allocatable_locals.iter()
            .map(|name| self.escape_info.get(name).map(|info| info.estimated_size).unwrap_or(0))
            .sum()
    }
    
    /// Utility functions for type and expression analysis
    fn estimate_type_size(&self, type_info: &crate::ast::Type) -> usize {
        match type_info {
            crate::ast::Type::Int => 8,
            crate::ast::Type::Float => 8,
            crate::ast::Type::Bool => 1,
            crate::ast::Type::String => 32, // Estimated string overhead
            crate::ast::Type::Void => 0,
            _ => 64, // Default for unknown types
        }
    }
    
    fn estimate_expression_size(&self, expr: &Expression) -> usize {
        match expr {
            Expression::Number(_) => 8,
            Expression::Float(_) => 8,
            Expression::Boolean(_) => 1,
            Expression::String(s) => 32 + s.len(),
            Expression::Array(elements) => 32 + elements.len() * 8, // Estimated array overhead
            _ => 64,
        }
    }
    
    fn is_simple_type(&self, type_info: &crate::ast::Type) -> bool {
        matches!(type_info, 
                crate::ast::Type::Int | 
                crate::ast::Type::Float | 
                crate::ast::Type::Bool)
    }
    
    fn is_register_candidate(&self, type_info: &crate::ast::Type) -> bool {
        self.is_simple_type(type_info) && self.estimate_type_size(type_info) <= 8
    }
    
    fn is_register_candidate_from_size(&self, size: usize) -> bool {
        size <= 8
    }
    
    fn is_simple_expression(&self, expr: &Expression) -> bool {
        matches!(expr, 
                Expression::Number(_) | 
                Expression::Float(_) | 
                Expression::Boolean(_) |
                Expression::Identifier(_))
    }
    
    fn is_complex_expression(&self, expr: &Expression) -> bool {
        matches!(expr, 
                Expression::Array(_) | 
                Expression::Call { .. } |
                Expression::Object { .. })
    }
    
    fn has_long_lifetime(&self, _expr: &Expression) -> bool {
        // Simplified lifetime analysis
        false
    }
    
    fn estimate_instruction_lifetime(&self, _expr: &Expression) -> usize {
        // Simplified instruction count estimate
        10
    }
    
    fn calculate_optimization_potential(&self, size: usize, escapes: bool) -> f64 {
        if escapes {
            0.0
        } else if size <= 8 {
            1.0 // Excellent register candidate
        } else if size <= 64 {
            0.8 // Good stack candidate
        } else if size <= 1024 {
            0.5 // Moderate stack candidate
        } else {
            0.1 // Poor candidate
        }
    }
    
    fn calculate_speedup_for_optimization(&self, opt_type: &OptimizationType, size: usize) -> f64 {
        match opt_type {
            OptimizationType::StackToRegister => 3.0 + (64.0 / size as f64), // Larger speedup for smaller objects
            OptimizationType::HeapToStack => 1.5 + (128.0 / size as f64),
            OptimizationType::EliminateAllocation => 10.0,
            OptimizationType::MergeAllocations => 2.0,
        }
    }
    
    fn calculate_overall_speedup(&self) -> f64 {
        if self.stats.variables_optimized == 0 {
            1.0
        } else {
            1.0 + (self.stats.variables_optimized as f64 * 0.1) // 10% speedup per optimized variable
        }
    }
    
    /// Generate optimization report
    pub fn generate_optimization_report(&self) -> String {
        format!(
            "🔍 Escape Analysis Optimization Report:\n\
             📊 Variables analyzed: {}\n\
             ⚡ Variables optimized: {}\n\
             🗑️ Heap allocations eliminated: {}\n\
             📦 Stack allocations created: {}\n\
             🎯 Register allocations created: {}\n\
             💾 Memory saved: {} bytes\n\
             🚀 Estimated speedup: {:.2}x\n\
             📈 Optimization rate: {:.1}%",
            self.stats.total_variables_analyzed,
            self.stats.variables_optimized,
            self.stats.heap_allocations_eliminated,
            self.stats.stack_allocations_created,
            self.stats.register_allocations_created,
            self.stats.estimated_memory_saved,
            self.stats.estimated_speedup,
            if self.stats.total_variables_analyzed > 0 {
                (self.stats.variables_optimized as f64 / self.stats.total_variables_analyzed as f64) * 100.0
            } else { 0.0 }
        )
    }
    
    /// Apply optimizations to a function based on escape analysis
    pub fn apply_optimizations(&self, function: &Function) -> Result<OptimizedFunction, RuntimeError> {
        if let Some(func_info) = self.function_analysis.get(&function.name) {
            Ok(OptimizedFunction {
                original: function.clone(),
                stack_allocations: func_info.stack_allocatable_locals.clone(),
                register_allocations: func_info.register_allocatable.clone(),
                stack_space_needed: func_info.stack_space_needed,
                optimizations: func_info.optimization_opportunities.clone(),
            })
        } else {
            Err(RuntimeError::InvalidOperation("Function not analyzed".to_string()))
        }
    }
}

/// Optimized function with escape analysis applied
#[derive(Debug, Clone)]
pub struct OptimizedFunction {
    pub original: Function,
    pub stack_allocations: HashSet<String>,
    pub register_allocations: HashSet<String>,
    pub stack_space_needed: usize,
    pub optimizations: Vec<StackAllocationOpportunity>,
}

impl Default for EscapeAnalyzer {
    fn default() -> Self {
        Self::new()
    }
}

/// Integration with execution profiler
impl EscapeAnalyzer {
    pub fn update_profiler_stats(&self, profiler: &mut ExecutionProfiler) {
        profiler.allocation_stats.escape_analysis_opportunities = self.stats.heap_allocations_eliminated as u64;
    }
}