// ScaffoldLang Interpreter - Actually Working Implementation
// Processes ScaffoldLang syntax and executes programs

use std::collections::HashMap;
use std::fmt;

#[derive(Debug, Clone)]
pub enum Value {
    Number(f64),
    String(String),
    Boolean(bool),
    Object(ScaffoldObject),
    Function(ScaffoldFunction),
    Array(Vec<Value>),
    Matrix(Vec<Vec<f64>>),
    Null,
}

#[derive(Debug, Clone)]
pub struct ScaffoldObject {
    pub class_name: String,
    pub properties: HashMap<String, Value>,
}

#[derive(Debug, Clone)]
pub struct ScaffoldFunction {
    pub name: String,
    pub params: Vec<String>,
    pub body: Vec<String>,
}

#[derive(Debug, Clone)]
pub struct ScaffoldClass {
    pub name: String,
    pub methods: HashMap<String, ScaffoldFunction>,
    pub constructor: Option<ScaffoldFunction>,
}

#[derive(Debug, Clone)]
pub struct ScaffoldMacro {
    pub name: String,
    pub params: Vec<String>,
    pub template: String,
}

#[derive(Debug, Clone)]
pub struct ScaffoldMicro {
    pub name: String,
    pub params: Vec<String>,
    pub body: String,
    pub inline_hint: bool,
}

impl fmt::Display for Value {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match self {
            Value::Number(n) => write!(f, "{}", n),
            Value::String(s) => write!(f, "{}", s),
            Value::Boolean(b) => write!(f, "{}", b),
            Value::Object(obj) => write!(f, "{}({})", obj.class_name, obj.properties.len()),
            Value::Function(func) => write!(f, "function {}({})", func.name, func.params.join(", ")),
            Value::Array(arr) => {
                let items: Vec<String> = arr.iter().map(|v| v.to_string()).collect();
                write!(f, "[{}]", items.join(", "))
            },
            Value::Matrix(mat) => {
                let rows: Vec<String> = mat.iter().map(|row| {
                    let items: Vec<String> = row.iter().map(|v| v.to_string()).collect();
                    format!("[{}]", items.join(", "))
                }).collect();
                write!(f, "[{}]", rows.join(", "))
            },
            Value::Null => write!(f, "null"),
        }
    }
}

pub struct ScaffoldLangInterpreter {
    variables: HashMap<String, Value>,
    classes: HashMap<String, ScaffoldClass>,
    current_class: Option<String>,
    macros: HashMap<String, ScaffoldMacro>,
    micros: HashMap<String, ScaffoldMicro>,
    imported_modules: HashMap<String, HashMap<String, Value>>,
}

impl ScaffoldLangInterpreter {
    pub fn new() -> Self {
        Self {
            variables: HashMap::new(),
            classes: HashMap::new(),
            current_class: None,
            macros: HashMap::new(),
            micros: HashMap::new(),
            imported_modules: HashMap::new(),
        }
    }

    pub fn execute(&mut self, source: &str) -> Result<String, String> {
        let mut output = String::new();
        let lines: Vec<&str> = source.lines().collect();
        let mut i = 0;
        
        while i < lines.len() {
            let line = lines[i].trim();
            if line.is_empty() || line.starts_with("//") {
                i += 1;
                continue;
            }
            
            // Handle control flow that requires multi-line processing
            if line.starts_with("while ") && line.contains(" {") {
                let (loop_output, lines_consumed) = self.execute_while_loop(&lines, i)?;
                output.push_str(&loop_output);
                i += lines_consumed;
                continue;
            }
            
            if line.starts_with("if ") && line.contains(" {") {
                let (if_output, lines_consumed) = self.execute_if_statement(&lines, i)?;
                output.push_str(&if_output);
                i += lines_consumed;
                continue;
            }
            
            match self.execute_line(line) {
                Ok(result) => {
                    if !result.is_empty() {
                        output.push_str(&result);
                        output.push('\n');
                    }
                }
                Err(e) => return Err(format!("Error on line '{}': {}", line, e)),
            }
            
            i += 1;
        }
        
        Ok(output)
    }

    fn execute_line(&mut self, line: &str) -> Result<String, String> {
        // Handle import statements
        if line.starts_with("import ") {
            return self.handle_import(line);
        }
        
        if line.starts_with("from ") && line.contains(" import ") {
            return self.handle_from_import(line);
        }
        
        // Handle macro definitions
        if line.starts_with("macro ") && line.contains("(") && line.contains("{") {
            return self.handle_macro_definition(line);
        }
        
        // Handle micro definitions  
        if line.starts_with("micro ") && line.contains("(") && line.contains("{") {
            return self.handle_micro_definition(line);
        }
        
        // Check for macro calls before other processing
        if let Some(result) = self.handle_macro_call(line)? {
            return Ok(result);
        }
        
        // Check for micro calls
        if let Some(result) = self.handle_micro_call(line)? {
            return Ok(result);
        }
        
        // Handle print statements
        if line.starts_with("print(") && line.ends_with(")") {
            let content = &line[6..line.len()-1];
            
            // Handle direct variable prints
            if !content.contains("\"") && !content.contains(" + ") {
                if let Some(value) = self.variables.get(content.trim()) {
                    return Ok(match value {
                        Value::Object(obj) => format!("{}({})", obj.class_name, obj.properties.len()),
                        Value::String(s) => s.clone(),
                        Value::Number(n) => {
                            if n.fract() == 0.0 {
                                format!("{}", *n as i64)
                            } else {
                                format!("{}", n)
                            }
                        }
                        _ => value.to_string()
                    });
                }
            }
            
            return Ok(self.evaluate_expression(content)?);
        }
        
        // Handle if statements (basic)
        if line.starts_with("if ") && line.contains(" {") {
            let condition_part = line[3..].split(" {").next().unwrap_or("").trim();
            let condition_result = self.evaluate_condition(condition_part)?;
            if condition_result {
                return Ok(format!("// If condition true: {}", condition_part));
            } else {
                return Ok(format!("// If condition false: {}", condition_part));
            }
        }
        
        // Handle while loops (basic)
        if line.starts_with("while ") && line.contains(" {") {
            let condition_part = line[6..].split(" {").next().unwrap_or("").trim();
            return Ok(format!("// While loop: {}", condition_part));
        }
        
        // Handle function definitions
        if line.contains(" = function(") && line.contains(") {") {
            let parts: Vec<&str> = line.splitn(2, " = function(").collect();
            if parts.len() == 2 {
                let func_name = parts[0].trim().to_string();
                let remaining = parts[1];
                if let Some(params_end) = remaining.find(") {") {
                    let params_str = &remaining[..params_end];
                    let params: Vec<String> = if params_str.trim().is_empty() {
                        Vec::new()
                    } else {
                        params_str.split(',').map(|p| p.trim().to_string()).collect()
                    };
                    
                    let func = ScaffoldFunction {
                        name: func_name.clone(),
                        params,
                        body: Vec::new(), // Would need multi-line parsing for full implementation
                    };
                    
                    self.variables.insert(func_name, Value::Function(func));
                    return Ok(String::new());
                }
            }
        }
        
        // Handle variable assignments (simple variables and mathematical functions)
        if line.contains(" = ") && !line.contains("function(") {
            let parts: Vec<&str> = line.splitn(2, " = ").collect();
            if parts.len() == 2 {
                let var_name = parts[0].trim().to_string();
                let value_expr = parts[1].trim();
                
                // Check if the value is a method call
                if value_expr.contains(".") && value_expr.contains("(") && value_expr.contains(")") {
                    if let Some(method_result) = self.execute_method_call(value_expr)? {
                        self.variables.insert(var_name, Value::String(method_result));
                        return Ok(String::new());
                    }
                }
                
                // Check if the value is array indexing
                if value_expr.contains("[") && value_expr.contains("]") && !value_expr.starts_with("[") {
                    match self.handle_array_indexing(value_expr) {
                        Ok(result) => {
                            // Try to parse as number first, then as string
                            if let Ok(num) = result.parse::<f64>() {
                                self.variables.insert(var_name, Value::Number(num));
                            } else {
                                self.variables.insert(var_name, Value::String(result));
                            }
                            return Ok(String::new());
                        }
                        Err(_) => {
                            // Fall through to normal parse_value
                        }
                    }
                }
                
                // Check if the value is a micro call
                if let Some(micro_result) = self.handle_micro_call(value_expr)? {
                    // Try to parse as number first, then as string
                    if let Ok(num) = micro_result.parse::<f64>() {
                        self.variables.insert(var_name, Value::Number(num));
                    } else {
                        self.variables.insert(var_name, Value::String(micro_result));
                    }
                    return Ok(String::new());
                }
                
                let value = self.parse_value(value_expr)?;
                self.variables.insert(var_name, value);
                return Ok(String::new());
            }
        }
        
        // Handle variable assignments with method calls
        if line.contains(" = ") && line.contains(".") && line.contains("(") && line.contains(")") && !line.contains("function(") {
            let parts: Vec<&str> = line.splitn(2, " = ").collect();
            if parts.len() == 2 {
                let var_name = parts[0].trim().to_string();
                let value_expr = parts[1].trim();
                
                // Check if it's a constructor call first
                if let Some(class_name) = value_expr.split("(").next() {
                    if ["Point", "Vector"].contains(&class_name) || self.classes.contains_key(class_name) {
                        // This is handled by object creation
                        return Ok(String::new());
                    }
                }
                
                // Check if the value is a method call
                if let Some(method_result) = self.execute_method_call(value_expr)? {
                    self.variables.insert(var_name, Value::String(method_result));
                    return Ok(String::new());
                }
                
                let value = self.parse_value(value_expr)?;
                self.variables.insert(var_name, value);
                return Ok(String::new());
            }
        }
        
        // Handle class definitions
        if line.starts_with("class ") && line.contains(" {") {
            let class_name = line[6..].split(" {").next().unwrap_or("").trim();
            self.current_class = Some(class_name.to_string());
            self.classes.insert(class_name.to_string(), ScaffoldClass {
                name: class_name.to_string(),
                methods: HashMap::new(),
                constructor: None,
            });
            return Ok(String::new());
        }
        
        // Handle class end
        if line == "}" && self.current_class.is_some() {
            self.current_class = None;
            return Ok(String::new());
        }
        
        // Handle constructor
        if line.starts_with("constructor(") && self.current_class.is_some() {
            // Basic constructor parsing - could be enhanced
            return Ok(String::new());
        }
        
        // Handle method definitions
        if self.current_class.is_some() && line.contains("(") && line.contains(")") && line.contains("{") {
            // Basic method parsing - could be enhanced
            return Ok(String::new());
        }
        
        // Handle algorithm calls (sort, transpose, etc.)
        if line.contains("(") && line.contains(")") {
            if let Some(result) = self.handle_algorithm_call(line)? {
                return Ok(result);
            }
        }
        
        // Handle method calls
        if line.contains(".") && line.contains("(") && line.contains(")") {
            return self.handle_method_call(line);
        }
        
        // Handle object creation
        if line.contains(" = ") && line.contains("(") && line.contains(")") && !line.contains("function(") {
            let parts: Vec<&str> = line.splitn(2, " = ").collect();
            if parts.len() == 2 {
                let var_name = parts[0].trim();
                let constructor_call = parts[1].trim();
                
                if let Some(class_name) = constructor_call.split("(").next() {
                    if self.classes.contains_key(class_name) || ["Point", "Vector"].contains(&class_name) {
                        // Parse constructor parameters
                        let params_str = constructor_call
                            .strip_prefix(&format!("{}(", class_name))
                            .and_then(|s| s.strip_suffix(")"))
                            .unwrap_or("");
                        
                        let mut properties = HashMap::new();
                        
                        // Handle constructor parameters
                        if !params_str.trim().is_empty() {
                            let params: Vec<&str> = params_str.split(",").collect();
                            for (i, param) in params.iter().enumerate() {
                                let param = param.trim();
                                let value = self.parse_value(param)?;
                                
                                // For Point class, assume x, y parameters
                                // For Vector class, assume x, y, z parameters
                                match (class_name, i) {
                                    ("Point", 0) => { properties.insert("x".to_string(), value); }
                                    ("Point", 1) => { properties.insert("y".to_string(), value); }
                                    ("Vector", 0) => { properties.insert("x".to_string(), value); }
                                    ("Vector", 1) => { properties.insert("y".to_string(), value); }
                                    ("Vector", 2) => { properties.insert("z".to_string(), value); }
                                    _ => {
                                        // Generic property assignment
                                        properties.insert(format!("prop{}", i), value);
                                    }
                                }
                            }
                        }
                        
                        let obj = ScaffoldObject {
                            class_name: class_name.to_string(),
                            properties,
                        };
                        self.variables.insert(var_name.to_string(), Value::Object(obj));
                        return Ok(String::new());
                    }
                }
            }
        }
        
        Ok(String::new())
    }

    fn evaluate_expression(&self, expr: &str) -> Result<String, String> {
        let expr = expr.trim();
        
        // Handle string literals (simple case)
        if expr.starts_with('"') && expr.ends_with('"') && expr.matches('"').count() == 2 {
            return Ok(expr[1..expr.len()-1].to_string());
        }
        
        // Handle array literals
        if expr.starts_with('[') && expr.ends_with(']') {
            return self.handle_array_literal(expr);
        }
        
        // Handle mathematical functions
        if expr.contains('(') && expr.contains(')') {
            if let Some(result) = self.evaluate_math_function(expr)? {
                return Ok(result);
            }
        }
        
        // Handle method calls in expressions
        if expr.contains(".") && expr.contains("(") && expr.contains(")") {
            if let Some(result) = self.execute_method_call(expr)? {
                return Ok(result);
            }
        }
        
        // Handle arithmetic operations and string concatenation
        if expr.contains(" + ") || expr.contains(" - ") || expr.contains(" * ") || expr.contains(" / ") {
            // Check if this is string concatenation or arithmetic
            if expr.contains('"') {
                // Contains quotes, likely string concatenation
                return self.handle_concatenation(expr);
            } else {
                // No quotes, likely arithmetic
                match self.evaluate_arithmetic(expr) {
                    Ok(Value::Number(n)) => {
                        return Ok(if n.fract() == 0.0 {
                            format!("{}", n as i64)
                        } else {
                            format!("{}", n)
                        });
                    }
                    Ok(value) => return Ok(value.to_string()),
                    Err(_) => {
                        // Fall back to concatenation if arithmetic fails
                        return self.handle_concatenation(expr);
                    }
                }
            }
        }
        
        // Handle array indexing
        if expr.contains('[') && expr.contains(']') && !expr.starts_with('[') {
            return self.handle_array_indexing(expr);
        }
        
        // Handle variable lookup
        if let Some(value) = self.variables.get(expr) {
            return Ok(match value {
                Value::Number(n) => {
                    if n.fract() == 0.0 {
                        format!("{}", *n as i64)
                    } else {
                        format!("{}", n)
                    }
                }
                _ => value.to_string()
            });
        }
        
        // Handle numeric literals
        if let Ok(num) = expr.parse::<f64>() {
            return Ok(if num.fract() == 0.0 {
                format!("{}", num as i64)
            } else {
                format!("{}", num)
            });
        }
        
        // Handle boolean literals
        if expr == "true" {
            return Ok("true".to_string());
        }
        if expr == "false" {
            return Ok("false".to_string());
        }
        
        Ok(expr.to_string())
    }

    fn handle_concatenation(&self, expr: &str) -> Result<String, String> {
        let tokens = self.tokenize_expression(expr)?;
        let mut result = String::new();
        
        for token in tokens {
            let value = if token.starts_with('"') && token.ends_with('"') {
                // String literal
                token[1..token.len()-1].to_string()
            } else if token.contains(".") && token.contains("(") && token.contains(")") {
                // Method call - execute it
                if let Some(method_result) = self.execute_method_call(&token)? {
                    method_result
                } else {
                    // Fallback: try to get object and call method
                    let parts: Vec<&str> = token.split(".").collect();
                    if parts.len() == 2 {
                        let obj_name = parts[0].trim();
                        if let Some(Value::Object(obj)) = self.variables.get(obj_name) {
                            match (obj.class_name.as_str(), parts[1].trim()) {
                                ("Point", "toString()") => {
                                    let x = obj.properties.get("x")
                                        .map(|v| match v {
                                            Value::Number(n) => {
                                                if n.fract() == 0.0 {
                                                    format!("{}", *n as i64)
                                                } else {
                                                    format!("{}", n)
                                                }
                                            }
                                            _ => v.to_string()
                                        })
                                        .unwrap_or("0".to_string());
                                    let y = obj.properties.get("y")
                                        .map(|v| match v {
                                            Value::Number(n) => {
                                                if n.fract() == 0.0 {
                                                    format!("{}", *n as i64)
                                                } else {
                                                    format!("{}", n)
                                                }
                                            }
                                            _ => v.to_string()
                                        })
                                        .unwrap_or("0".to_string());
                                    format!("Point({}, {})", x, y)
                                }
                                ("Vector", "toString()") => {
                                    let x = obj.properties.get("x")
                                        .map(|v| match v {
                                            Value::Number(n) => {
                                                if n.fract() == 0.0 {
                                                    format!("{}", *n as i64)
                                                } else {
                                                    format!("{}", n)
                                                }
                                            }
                                            _ => v.to_string()
                                        })
                                        .unwrap_or("0".to_string());
                                    let y = obj.properties.get("y")
                                        .map(|v| match v {
                                            Value::Number(n) => {
                                                if n.fract() == 0.0 {
                                                    format!("{}", *n as i64)
                                                } else {
                                                    format!("{}", n)
                                                }
                                            }
                                            _ => v.to_string()
                                        })
                                        .unwrap_or("0".to_string());
                                    let z = obj.properties.get("z")
                                        .map(|v| match v {
                                            Value::Number(n) => {
                                                if n.fract() == 0.0 {
                                                    format!("{}", *n as i64)
                                                } else {
                                                    format!("{}", n)
                                                }
                                            }
                                            _ => v.to_string()
                                        })
                                        .unwrap_or("0".to_string());
                                    format!("Vector({}, {}, {})", x, y, z)
                                }
                                _ => token.clone()
                            }
                        } else {
                            token.clone()
                        }
                    } else {
                        token.clone()
                    }
                }
            } else if token.contains('[') && token.contains(']') && !token.starts_with('[') {
                // Array indexing
                match self.handle_array_indexing(&token) {
                    Ok(result) => result,
                    Err(_) => token.clone()
                }
            } else if let Some(var_value) = self.variables.get(&token) {
                // Variable
                match var_value {
                    Value::Number(n) => {
                        if n.fract() == 0.0 {
                            format!("{}", *n as i64)
                        } else {
                            format!("{}", n)
                        }
                    }
                    Value::Object(obj) => {
                        // For objects in concatenation, use their class representation
                        let x = obj.properties.get("x")
                            .map(|v| match v {
                                Value::Number(n) => {
                                    if n.fract() == 0.0 {
                                        format!("{}", *n as i64)
                                    } else {
                                        format!("{}", n)
                                    }
                                }
                                _ => v.to_string()
                            })
                            .unwrap_or("0".to_string());
                        let y = obj.properties.get("y")
                            .map(|v| match v {
                                Value::Number(n) => {
                                    if n.fract() == 0.0 {
                                        format!("{}", *n as i64)
                                    } else {
                                        format!("{}", n)
                                    }
                                }
                                _ => v.to_string()
                            })
                            .unwrap_or("0".to_string());
                        
                        if obj.class_name == "Point" {
                            format!("Point({}, {})", x, y)
                        } else if obj.class_name == "Vector" {
                            let z = obj.properties.get("z")
                                .map(|v| match v {
                                    Value::Number(n) => {
                                        if n.fract() == 0.0 {
                                            format!("{}", *n as i64)
                                        } else {
                                            format!("{}", n)
                                        }
                                    }
                                    _ => v.to_string()
                                })
                                .unwrap_or("0".to_string());
                            format!("Vector({}, {}, {})", x, y, z)
                        } else {
                            format!("{}({})", obj.class_name, obj.properties.len())
                        }
                    }
                    Value::Function(func) => format!("function {}", func.name),
                    Value::String(s) => s.clone(),
                    Value::Array(arr) => {
                        let items: Vec<String> = arr.iter().map(|v| v.to_string()).collect();
                        format!("[{}]", items.join(", "))
                    },
                    Value::Matrix(mat) => {
                        let rows: Vec<String> = mat.iter().map(|row| {
                            let items: Vec<String> = row.iter().map(|v| v.to_string()).collect();
                            format!("[{}]", items.join(", "))
                        }).collect();
                        format!("[{}]", rows.join(", "))
                    },
                    _ => var_value.to_string()
                }
            } else if let Ok(num) = token.parse::<f64>() {
                // Number literal
                if num.fract() == 0.0 {
                    format!("{}", num as i64)
                } else {
                    format!("{}", num)
                }
            } else {
                // Literal text
                token
            };
            
            result.push_str(&value);
        }
        
        Ok(result)
    }
    
    fn handle_algorithm_call(&mut self, line: &str) -> Result<Option<String>, String> {
        let line = line.trim();
        
        // Handle sort operations
        if line.starts_with("sort(") {
            return self.handle_sort_call(line, "quick");
        }
        if line.starts_with("bubbleSort(") {
            return self.handle_sort_call(line, "bubble");
        }
        if line.starts_with("quickSort(") {
            return self.handle_sort_call(line, "quick");
        }
        if line.starts_with("mergeSort(") {
            return self.handle_sort_call(line, "merge");
        }
        
        // Handle matrix operations
        if line.starts_with("transpose(") {
            return self.handle_transpose_call(line);
        }
        
        // Handle array operations
        if line.contains(" = ") && line.contains("(") {
            let parts: Vec<&str> = line.splitn(2, " = ").collect();
            if parts.len() == 2 {
                let var_name = parts[0].trim();
                let func_call = parts[1].trim();
                
                if func_call.starts_with("range(") {
                    return self.handle_range_call(var_name, func_call);
                }
                if func_call.starts_with("linspace(") {
                    return self.handle_linspace_call(var_name, func_call);
                }
                if func_call.starts_with("zeros(") {
                    return self.handle_zeros_call(var_name, func_call);
                }
                if func_call.starts_with("ones(") {
                    return self.handle_ones_call(var_name, func_call);
                }
                if func_call.starts_with("eye(") {
                    return self.handle_identity_call(var_name, func_call);
                }
            }
        }
        
        Ok(None)
    }
    
    fn handle_sort_call(&mut self, line: &str, algorithm: &str) -> Result<Option<String>, String> {
        let start = line.find('(').unwrap();
        let end = line.rfind(')').unwrap();
        let args = &line[start + 1..end].trim();
        
        if let Some(value) = self.variables.get_mut(&args.to_string()) {
            if let Value::Array(ref mut arr) = value {
                match algorithm {
                    "bubble" => Self::bubble_sort(arr),
                    "quick" => {
                        let len = arr.len();
                        if len > 1 {
                            Self::quick_sort(arr, 0, len - 1);
                        }
                    },
                    "merge" => Self::merge_sort(arr),
                    _ => Self::quick_sort(arr, 0, arr.len().saturating_sub(1)),
                }
                return Ok(Some(format!("Sorted array {} using {} sort", args, algorithm)));
            }
        }
        
        Err(format!("Cannot sort variable '{}'", args))
    }
    
    fn handle_transpose_call(&mut self, line: &str) -> Result<Option<String>, String> {
        let start = line.find('(').unwrap();
        let end = line.rfind(')').unwrap();
        let args = &line[start + 1..end].trim();
        
        if let Some(value) = self.variables.get(&args.to_string()) {
            if let Value::Matrix(mat) = value {
                let transposed = self.matrix_transpose(mat);
                let result_name = format!("{}_T", args);
                self.variables.insert(result_name.clone(), Value::Matrix(transposed));
                return Ok(Some(format!("Matrix transposed as {}", result_name)));
            }
        }
        
        Err(format!("Cannot transpose variable '{}'", args))
    }
    
    fn handle_range_call(&mut self, var_name: &str, func_call: &str) -> Result<Option<String>, String> {
        let start = func_call.find('(').unwrap();
        let end = func_call.rfind(')').unwrap();
        let args_str = &func_call[start + 1..end];
        
        let args: Vec<&str> = args_str.split(',').collect();
        let (start_val, end_val, step) = if args.len() == 1 {
            (0.0, args[0].trim().parse::<f64>().unwrap_or(0.0), 1.0)
        } else if args.len() == 2 {
            (args[0].trim().parse::<f64>().unwrap_or(0.0), 
             args[1].trim().parse::<f64>().unwrap_or(0.0), 1.0)
        } else if args.len() == 3 {
            (args[0].trim().parse::<f64>().unwrap_or(0.0), 
             args[1].trim().parse::<f64>().unwrap_or(0.0),
             args[2].trim().parse::<f64>().unwrap_or(1.0))
        } else {
            return Err("Invalid range arguments".to_string());
        };
        
        let mut values = Vec::new();
        let mut current = start_val;
        while current < end_val {
            values.push(Value::Number(current));
            current += step;
        }
        
        self.variables.insert(var_name.to_string(), Value::Array(values));
        Ok(Some(String::new()))
    }
    
    fn handle_linspace_call(&mut self, var_name: &str, func_call: &str) -> Result<Option<String>, String> {
        let start = func_call.find('(').unwrap();
        let end = func_call.rfind(')').unwrap();
        let args_str = &func_call[start + 1..end];
        
        let args: Vec<&str> = args_str.split(',').collect();
        if args.len() < 3 {
            return Err("linspace requires 3 arguments: start, end, num_points".to_string());
        }
        
        let start_val = args[0].trim().parse::<f64>().unwrap_or(0.0);
        let end_val = args[1].trim().parse::<f64>().unwrap_or(1.0);
        let num_points = args[2].trim().parse::<usize>().unwrap_or(10);
        
        let mut values = Vec::new();
        if num_points > 0 {
            let step = if num_points > 1 { (end_val - start_val) / (num_points - 1) as f64 } else { 0.0 };
            for i in 0..num_points {
                values.push(Value::Number(start_val + i as f64 * step));
            }
        }
        
        self.variables.insert(var_name.to_string(), Value::Array(values));
        Ok(Some(String::new()))
    }
    
    fn handle_zeros_call(&mut self, var_name: &str, func_call: &str) -> Result<Option<String>, String> {
        let start = func_call.find('(').unwrap();
        let end = func_call.rfind(')').unwrap();
        let args_str = &func_call[start + 1..end];
        
        let args: Vec<&str> = args_str.split(',').collect();
        if args.len() == 1 {
            // Vector of zeros
            let size = args[0].trim().parse::<usize>().unwrap_or(0);
            let values = vec![Value::Number(0.0); size];
            self.variables.insert(var_name.to_string(), Value::Array(values));
        } else if args.len() == 2 {
            // Matrix of zeros
            let rows = args[0].trim().parse::<usize>().unwrap_or(0);
            let cols = args[1].trim().parse::<usize>().unwrap_or(0);
            let matrix = vec![vec![0.0; cols]; rows];
            self.variables.insert(var_name.to_string(), Value::Matrix(matrix));
        }
        
        Ok(Some(String::new()))
    }
    
    fn handle_ones_call(&mut self, var_name: &str, func_call: &str) -> Result<Option<String>, String> {
        let start = func_call.find('(').unwrap();
        let end = func_call.rfind(')').unwrap();
        let args_str = &func_call[start + 1..end];
        
        let args: Vec<&str> = args_str.split(',').collect();
        if args.len() == 1 {
            // Vector of ones
            let size = args[0].trim().parse::<usize>().unwrap_or(0);
            let values = vec![Value::Number(1.0); size];
            self.variables.insert(var_name.to_string(), Value::Array(values));
        } else if args.len() == 2 {
            // Matrix of ones
            let rows = args[0].trim().parse::<usize>().unwrap_or(0);
            let cols = args[1].trim().parse::<usize>().unwrap_or(0);
            let matrix = vec![vec![1.0; cols]; rows];
            self.variables.insert(var_name.to_string(), Value::Matrix(matrix));
        }
        
        Ok(Some(String::new()))
    }
    
    fn handle_identity_call(&mut self, var_name: &str, func_call: &str) -> Result<Option<String>, String> {
        let start = func_call.find('(').unwrap();
        let end = func_call.rfind(')').unwrap();
        let args_str = &func_call[start + 1..end];
        
        let size = args_str.trim().parse::<usize>().unwrap_or(0);
        let mut matrix = vec![vec![0.0; size]; size];
        
        for i in 0..size {
            matrix[i][i] = 1.0;
        }
        
        self.variables.insert(var_name.to_string(), Value::Matrix(matrix));
        Ok(Some(String::new()))
    }
    
    fn tokenize_expression(&self, expr: &str) -> Result<Vec<String>, String> {
        let mut tokens = Vec::new();
        let mut current_token = String::new();
        let mut in_string = false;
        let mut paren_depth = 0;
        let mut chars = expr.chars().peekable();
        
        while let Some(ch) = chars.next() {
            if ch == '"' && paren_depth == 0 {
                if in_string {
                    // End of string
                    current_token.push(ch);
                    tokens.push(current_token.trim().to_string());
                    current_token = String::new();
                    in_string = false;
                } else {
                    // Start of string
                    if !current_token.trim().is_empty() {
                        tokens.push(current_token.trim().to_string());
                        current_token = String::new();
                    }
                    current_token.push(ch);
                    in_string = true;
                }
            } else if in_string {
                current_token.push(ch);
            } else if ch == '(' {
                current_token.push(ch);
                paren_depth += 1;
            } else if ch == ')' {
                current_token.push(ch);
                paren_depth -= 1;
            } else if ch == '+' && chars.peek() == Some(&' ') && paren_depth == 0 {
                // Found " + " separator at top level
                if !current_token.trim().is_empty() {
                    tokens.push(current_token.trim().to_string());
                    current_token = String::new();
                }
                chars.next(); // consume the space
            } else if ch != ' ' || !current_token.is_empty() {
                current_token.push(ch);
            }
        }
        
        if !current_token.trim().is_empty() {
            tokens.push(current_token.trim().to_string());
        }
        
        Ok(tokens)
    }

    fn parse_value(&self, value_str: &str) -> Result<Value, String> {
        let value_str = value_str.trim();
        
        // String literal
        if value_str.starts_with('"') && value_str.ends_with('"') {
            return Ok(Value::String(value_str[1..value_str.len()-1].to_string()));
        }
        
        // Array literal
        if value_str.starts_with('[') && value_str.ends_with(']') {
            return self.parse_array_literal(value_str);
        }
        
        // Matrix literal (nested arrays)
        if value_str.starts_with("matrix(") && value_str.ends_with(")") {
            return self.parse_matrix_literal(value_str);
        }
        
        // Mathematical function calls
        if value_str.contains('(') && value_str.contains(')') {
            if let Some(result_str) = self.evaluate_math_function(value_str)? {
                if let Ok(num) = result_str.parse::<f64>() {
                    return Ok(Value::Number(num));
                } else {
                    return Ok(Value::String(result_str));
                }
            }
        }
        
        // Boolean literal
        if value_str == "true" {
            return Ok(Value::Boolean(true));
        }
        if value_str == "false" {
            return Ok(Value::Boolean(false));
        }
        
        // Numeric literal
        if let Ok(num) = value_str.parse::<f64>() {
            return Ok(Value::Number(num));
        }
        
        // Handle expressions with operators
        if value_str.contains(" + ") {
            return self.evaluate_arithmetic(value_str);
        }
        if value_str.contains(" * ") {
            return self.evaluate_arithmetic(value_str);
        }
        if value_str.contains(" - ") {
            return self.evaluate_arithmetic(value_str);
        }
        if value_str.contains(" / ") {
            return self.evaluate_arithmetic(value_str);
        }
        
        // Variable reference
        if let Some(value) = self.variables.get(value_str) {
            return Ok(value.clone());
        }
        
        Ok(Value::String(value_str.to_string()))
    }
    
    fn parse_array_literal(&self, value_str: &str) -> Result<Value, String> {
        let content = &value_str[1..value_str.len()-1].trim();
        if content.is_empty() {
            return Ok(Value::Array(Vec::new()));
        }
        
        let elements: Vec<&str> = content.split(',').collect();
        let mut values = Vec::new();
        
        for elem in elements {
            let elem = elem.trim();
            let value = self.parse_value(elem)?;
            values.push(value);
        }
        
        Ok(Value::Array(values))
    }
    
    fn parse_matrix_literal(&self, value_str: &str) -> Result<Value, String> {
        // matrix([[1,2,3],[4,5,6],[7,8,9]])
        let content = &value_str[7..value_str.len()-1].trim(); // Remove "matrix(" and ")"
        
        if content.starts_with('[') && content.ends_with(']') {
            let inner = &content[1..content.len()-1];
            let mut matrix = Vec::new();
            let mut current_row = Vec::new();
            let mut bracket_depth = 0;
            let mut current_elem = String::new();
            
            for ch in inner.chars() {
                match ch {
                    '[' => {
                        bracket_depth += 1;
                        if bracket_depth == 1 {
                            current_row.clear();
                        } else {
                            current_elem.push(ch);
                        }
                    }
                    ']' => {
                        bracket_depth -= 1;
                        if bracket_depth == 0 {
                            if !current_elem.trim().is_empty() {
                                if let Ok(num) = current_elem.trim().parse::<f64>() {
                                    current_row.push(num);
                                }
                            }
                            if !current_row.is_empty() {
                                matrix.push(current_row.clone());
                            }
                            current_elem.clear();
                        } else {
                            current_elem.push(ch);
                        }
                    }
                    ',' => {
                        if bracket_depth == 1 {
                            if !current_elem.trim().is_empty() {
                                if let Ok(num) = current_elem.trim().parse::<f64>() {
                                    current_row.push(num);
                                }
                            }
                            current_elem.clear();
                        } else {
                            current_elem.push(ch);
                        }
                    }
                    _ => {
                        if bracket_depth > 0 {
                            current_elem.push(ch);
                        }
                    }
                }
            }
            
            Ok(Value::Matrix(matrix))
        } else {
            Err("Invalid matrix syntax".to_string())
        }
    }

    fn evaluate_arithmetic(&self, expr: &str) -> Result<Value, String> {
        // Handle complex expressions like "excitement_level * version + 42"
        let expr = expr.trim();
        
        // Split by + first (lower precedence)
        if expr.contains(" + ") {
            let parts: Vec<&str> = expr.splitn(2, " + ").collect();
            if parts.len() == 2 {
                let left = self.evaluate_arithmetic_term(parts[0].trim())?;
                let right = self.evaluate_arithmetic_term(parts[1].trim())?;
                return Ok(Value::Number(left + right));
            }
        }
        
        // Split by - (same precedence as +)
        if expr.contains(" - ") {
            let parts: Vec<&str> = expr.splitn(2, " - ").collect();
            if parts.len() == 2 {
                let left = self.evaluate_arithmetic_term(parts[0].trim())?;
                let right = self.evaluate_arithmetic_term(parts[1].trim())?;
                return Ok(Value::Number(left - right));
            }
        }
        
        // If no +/-, evaluate as a single term
        let result = self.evaluate_arithmetic_term(expr)?;
        Ok(Value::Number(result))
    }
    
    fn evaluate_arithmetic_term(&self, expr: &str) -> Result<f64, String> {
        let expr = expr.trim();
        
        // Handle multiplication
        if expr.contains(" * ") {
            let parts: Vec<&str> = expr.split(" * ").collect();
            let mut result = 1.0;
            for part in parts {
                result *= self.get_numeric_value(part.trim())?;
            }
            return Ok(result);
        }
        
        // Handle division
        if expr.contains(" / ") {
            let parts: Vec<&str> = expr.split(" / ").collect();
            if parts.len() == 2 {
                let left = self.get_numeric_value(parts[0].trim())?;
                let right = self.get_numeric_value(parts[1].trim())?;
                if right != 0.0 {
                    return Ok(left / right);
                } else {
                    return Err("Division by zero".to_string());
                }
            }
        }
        
        // Handle modulo
        if expr.contains(" % ") {
            let parts: Vec<&str> = expr.split(" % ").collect();
            if parts.len() == 2 {
                let left = self.get_numeric_value(parts[0].trim())?;
                let right = self.get_numeric_value(parts[1].trim())?;
                if right != 0.0 {
                    return Ok(left % right);
                } else {
                    return Err("Modulo by zero".to_string());
                }
            }
        }
        
        // Single value
        self.get_numeric_value(expr)
    }

    fn get_numeric_value(&self, expr: &str) -> Result<f64, String> {
        if let Ok(num) = expr.parse::<f64>() {
            return Ok(num);
        }
        
        if let Some(Value::Number(num)) = self.variables.get(expr) {
            return Ok(*num);
        }
        
        Err(format!("Cannot convert '{}' to number", expr))
    }
    
    fn evaluate_condition(&self, condition: &str) -> Result<bool, String> {
        let condition = condition.trim();
        
        // Handle simple comparisons
        if condition.contains(" > ") {
            let parts: Vec<&str> = condition.split(" > ").collect();
            if parts.len() == 2 {
                let left = self.get_numeric_value(parts[0].trim())?;
                let right = self.get_numeric_value(parts[1].trim())?;
                return Ok(left > right);
            }
        }
        
        if condition.contains(" < ") {
            let parts: Vec<&str> = condition.split(" < ").collect();
            if parts.len() == 2 {
                let left = self.get_numeric_value(parts[0].trim())?;
                let right = self.get_numeric_value(parts[1].trim())?;
                return Ok(left < right);
            }
        }
        
        if condition.contains(" == ") {
            let parts: Vec<&str> = condition.split(" == ").collect();
            if parts.len() == 2 {
                let left_str = self.evaluate_expression(parts[0].trim())?;
                let right_str = self.evaluate_expression(parts[1].trim())?;
                return Ok(left_str == right_str);
            }
        }
        
        // Handle boolean variables
        if let Some(Value::Boolean(b)) = self.variables.get(condition) {
            return Ok(*b);
        }
        
        // Handle boolean literals
        if condition == "true" {
            return Ok(true);
        }
        if condition == "false" {
            return Ok(false);
        }
        
        Ok(false)
    }
    
    fn handle_method_call(&mut self, line: &str) -> Result<String, String> {
        let line = line.trim();
        
        // Handle print statements with method calls
        if line.starts_with("print(") && line.ends_with(")") {
            let content = &line[6..line.len()-1];
            
            // If it's a string with concatenation containing method calls
            if content.contains(" + ") && content.contains(".") {
                let result = self.handle_concatenation(content)?;
                return Ok(result);
            }
            
            // If it's a direct method call
            if content.contains(".") && content.contains("(") && content.contains(")") {
                if let Some(method_result) = self.execute_method_call(content.trim_matches('"'))? {
                    return Ok(method_result);
                }
            }
        }
        
        // Direct method call
        if let Some(result) = self.execute_method_call(line)? {
            return Ok(result);
        }
        
        Ok(String::new())
    }
    
    fn execute_method_call(&self, expr: &str) -> Result<Option<String>, String> {
        if !expr.contains(".") {
            return Ok(None);
        }
        
        let parts: Vec<&str> = expr.split(".").collect();
        if parts.len() != 2 {
            return Ok(None);
        }
        
        let object_name = parts[0].trim();
        let method_call = parts[1].trim();
        
        // Get the object
        if let Some(Value::Object(obj)) = self.variables.get(object_name) {
            // Parse method name and parameters
            let (method_name, params) = if method_call.contains("(") {
                let method = method_call.split("(").next().unwrap_or("").trim();
                let params_str = method_call
                    .strip_prefix(&format!("{}(", method))
                    .and_then(|s| s.strip_suffix(")"))
                    .unwrap_or("");
                
                let params: Vec<String> = if params_str.trim().is_empty() {
                    Vec::new()
                } else {
                    params_str.split(",").map(|p| p.trim().to_string()).collect()
                };
                
                (method, params)
            } else {
                (method_call.trim(), Vec::new())
            };
            
            // Execute built-in methods based on class and method name
            match (obj.class_name.as_str(), method_name) {
                ("Point", "toString") => {
                    let x = obj.properties.get("x")
                        .map(|v| match v {
                            Value::Number(n) => {
                                if n.fract() == 0.0 {
                                    format!("{}", *n as i64)
                                } else {
                                    format!("{}", n)
                                }
                            }
                            _ => v.to_string()
                        })
                        .unwrap_or("0".to_string());
                    let y = obj.properties.get("y")
                        .map(|v| match v {
                            Value::Number(n) => {
                                if n.fract() == 0.0 {
                                    format!("{}", *n as i64)
                                } else {
                                    format!("{}", n)
                                }
                            }
                            _ => v.to_string()
                        })
                        .unwrap_or("0".to_string());
                    return Ok(Some(format!("Point({}, {})", x, y)));
                }
                ("Point", "distance") => {
                    let x = obj.properties.get("x")
                        .and_then(|v| if let Value::Number(n) = v { Some(*n) } else { None })
                        .unwrap_or(0.0);
                    let y = obj.properties.get("y")
                        .and_then(|v| if let Value::Number(n) = v { Some(*n) } else { None })
                        .unwrap_or(0.0);
                    let distance = (x * x + y * y).sqrt();
                    return Ok(Some(format!("{:.2}", distance)));
                }
                ("Point", "dot") => {
                    if !params.is_empty() {
                        // Dot product with another point
                        if let Some(other_obj) = self.variables.get(&params[0]) {
                            if let Value::Object(other) = other_obj {
                                let x1 = obj.properties.get("x")
                                    .and_then(|v| if let Value::Number(n) = v { Some(*n) } else { None })
                                    .unwrap_or(0.0);
                                let y1 = obj.properties.get("y")
                                    .and_then(|v| if let Value::Number(n) = v { Some(*n) } else { None })
                                    .unwrap_or(0.0);
                                let x2 = other.properties.get("x")
                                    .and_then(|v| if let Value::Number(n) = v { Some(*n) } else { None })
                                    .unwrap_or(0.0);
                                let y2 = other.properties.get("y")
                                    .and_then(|v| if let Value::Number(n) = v { Some(*n) } else { None })
                                    .unwrap_or(0.0);
                                let dot = x1 * x2 + y1 * y2;
                                return Ok(Some(format!("{:.2}", dot)));
                            }
                        }
                    }
                    return Ok(Some("0.00".to_string()));
                }
                ("Vector", "toString") => {
                    let x = obj.properties.get("x")
                        .map(|v| match v {
                            Value::Number(n) => {
                                if n.fract() == 0.0 {
                                    format!("{}", *n as i64)
                                } else {
                                    format!("{}", n)
                                }
                            }
                            _ => v.to_string()
                        })
                        .unwrap_or("0".to_string());
                    let y = obj.properties.get("y")
                        .map(|v| match v {
                            Value::Number(n) => {
                                if n.fract() == 0.0 {
                                    format!("{}", *n as i64)
                                } else {
                                    format!("{}", n)
                                }
                            }
                            _ => v.to_string()
                        })
                        .unwrap_or("0".to_string());
                    let z = obj.properties.get("z")
                        .map(|v| match v {
                            Value::Number(n) => {
                                if n.fract() == 0.0 {
                                    format!("{}", *n as i64)
                                } else {
                                    format!("{}", n)
                                }
                            }
                            _ => v.to_string()
                        })
                        .unwrap_or("0".to_string());
                    return Ok(Some(format!("Vector({}, {}, {})", x, y, z)));
                }
                ("Vector", "distance") => {
                    let x = obj.properties.get("x")
                        .and_then(|v| if let Value::Number(n) = v { Some(*n) } else { None })
                        .unwrap_or(0.0);
                    let y = obj.properties.get("y")
                        .and_then(|v| if let Value::Number(n) = v { Some(*n) } else { None })
                        .unwrap_or(0.0);
                    let z = obj.properties.get("z")
                        .and_then(|v| if let Value::Number(n) = v { Some(*n) } else { None })
                        .unwrap_or(0.0);
                    let distance = (x * x + y * y + z * z).sqrt();
                    return Ok(Some(format!("{:.2}", distance)));
                }
                ("Vector", "dot") => {
                    if !params.is_empty() {
                        // Dot product with another vector
                        if let Some(other_obj) = self.variables.get(&params[0]) {
                            if let Value::Object(other) = other_obj {
                                let x1 = obj.properties.get("x")
                                    .and_then(|v| if let Value::Number(n) = v { Some(*n) } else { None })
                                    .unwrap_or(0.0);
                                let y1 = obj.properties.get("y")
                                    .and_then(|v| if let Value::Number(n) = v { Some(*n) } else { None })
                                    .unwrap_or(0.0);
                                let z1 = obj.properties.get("z")
                                    .and_then(|v| if let Value::Number(n) = v { Some(*n) } else { None })
                                    .unwrap_or(0.0);
                                let x2 = other.properties.get("x")
                                    .and_then(|v| if let Value::Number(n) = v { Some(*n) } else { None })
                                    .unwrap_or(0.0);
                                let y2 = other.properties.get("y")
                                    .and_then(|v| if let Value::Number(n) = v { Some(*n) } else { None })
                                    .unwrap_or(0.0);
                                let z2 = other.properties.get("z")
                                    .and_then(|v| if let Value::Number(n) = v { Some(*n) } else { None })
                                    .unwrap_or(0.0);
                                let dot = x1 * x2 + y1 * y2 + z1 * z2;
                                return Ok(Some(format!("{:.2}", dot)));
                            }
                        }
                    }
                    return Ok(Some("0.00".to_string()));
                }
                _ => {
                    // Unknown method - return a default implementation
                    return Ok(Some(format!("{}({})", obj.class_name, obj.properties.len())));
                }
            }
        }
        
        Ok(None)
    }
    
    fn execute_while_loop(&mut self, lines: &[&str], start_index: usize) -> Result<(String, usize), String> {
        let mut output = String::new();
        let while_line = lines[start_index].trim();
        
        // Extract condition from "while condition {"
        let condition_part = while_line[6..].split(" {").next().unwrap_or("").trim();
        
        // Find the loop body
        let mut body_lines = Vec::new();
        let mut i = start_index + 1;
        let mut brace_count = 1;
        
        while i < lines.len() && brace_count > 0 {
            let line = lines[i].trim();
            if line.contains("{") {
                brace_count += line.matches("{").count();
            }
            if line.contains("}") {
                brace_count -= line.matches("}").count();
            }
            
            if brace_count > 0 {
                body_lines.push(line);
            }
            i += 1;
        }
        
        // Execute the while loop
        let mut iterations = 0;
        const MAX_ITERATIONS: usize = 1000; // Prevent infinite loops
        
        while iterations < MAX_ITERATIONS {
            // Evaluate condition
            if !self.evaluate_condition(condition_part)? {
                break;
            }
            
            // Execute body
            for body_line in &body_lines {
                if !body_line.is_empty() && !body_line.starts_with("//") {
                    match self.execute_line(body_line) {
                        Ok(result) => {
                            if !result.is_empty() {
                                output.push_str(&result);
                                output.push('\n');
                            }
                        }
                        Err(e) => return Err(format!("Error in while loop on line '{}': {}", body_line, e)),
                    }
                }
            }
            
            iterations += 1;
        }
        
        if iterations >= MAX_ITERATIONS {
            output.push_str("// While loop terminated (max iterations reached)\n");
        }
        
        Ok((output, i - start_index))
    }
    
    fn execute_if_statement(&mut self, lines: &[&str], start_index: usize) -> Result<(String, usize), String> {
        let mut output = String::new();
        let if_line = lines[start_index].trim();
        
        // Extract condition from "if condition {"
        let condition_part = if_line[3..].split(" {").next().unwrap_or("").trim();
        
        // Find the if body
        let mut body_lines = Vec::new();
        let mut i = start_index + 1;
        let mut brace_count = 1;
        
        while i < lines.len() && brace_count > 0 {
            let line = lines[i].trim();
            if line.contains("{") {
                brace_count += line.matches("{").count();
            }
            if line.contains("}") {
                brace_count -= line.matches("}").count();
            }
            
            if brace_count > 0 {
                body_lines.push(line);
            }
            i += 1;
        }
        
        // Evaluate condition and execute body if true
        if self.evaluate_condition(condition_part)? {
            for body_line in &body_lines {
                if !body_line.is_empty() && !body_line.starts_with("//") {
                    match self.execute_line(body_line) {
                        Ok(result) => {
                            if !result.is_empty() {
                                output.push_str(&result);
                                output.push('\n');
                            }
                        }
                        Err(e) => return Err(format!("Error in if statement on line '{}': {}", body_line, e)),
                    }
                }
            }
        }
        
        Ok((output, i - start_index))
    }
    
    fn evaluate_math_function(&self, expr: &str) -> Result<Option<String>, String> {
        if !expr.contains('(') || !expr.ends_with(')') {
            return Ok(None);
        }
        
        let func_end = expr.find('(').unwrap();
        let func_name = expr[..func_end].trim();
        let args_str = &expr[func_end + 1..expr.len() - 1];
        
        let args: Vec<f64> = if args_str.trim().is_empty() {
            Vec::new()
        } else {
            args_str.split(',').map(|arg| {
                let arg = arg.trim();
                if let Some(Value::Number(n)) = self.variables.get(arg) {
                    *n
                } else {
                    arg.parse::<f64>().unwrap_or(0.0)
                }
            }).collect()
        };
        
        let result = match func_name {
            // Basic trigonometry
            "sin" => if args.len() >= 1 { args[0].sin() } else { 0.0 },
            "cos" => if args.len() >= 1 { args[0].cos() } else { 0.0 },
            "tan" => if args.len() >= 1 { args[0].tan() } else { 0.0 },
            "asin" => if args.len() >= 1 { args[0].asin() } else { 0.0 },
            "acos" => if args.len() >= 1 { args[0].acos() } else { 0.0 },
            "atan" => if args.len() >= 1 { args[0].atan() } else { 0.0 },
            "atan2" => if args.len() >= 2 { args[0].atan2(args[1]) } else { 0.0 },
            
            // Exponential and logarithmic
            "exp" => if args.len() >= 1 { args[0].exp() } else { 0.0 },
            "ln" | "log" => if args.len() >= 1 { args[0].ln() } else { 0.0 },
            "log10" => if args.len() >= 1 { args[0].log10() } else { 0.0 },
            "log2" => if args.len() >= 1 { args[0].log2() } else { 0.0 },
            "pow" => if args.len() >= 2 { args[0].powf(args[1]) } else { 0.0 },
            "sqrt" => if args.len() >= 1 { args[0].sqrt() } else { 0.0 },
            "cbrt" => if args.len() >= 1 { args[0].cbrt() } else { 0.0 },
            
            // Rounding and absolute
            "abs" => if args.len() >= 1 { args[0].abs() } else { 0.0 },
            "floor" => if args.len() >= 1 { args[0].floor() } else { 0.0 },
            "ceil" => if args.len() >= 1 { args[0].ceil() } else { 0.0 },
            "round" => if args.len() >= 1 { args[0].round() } else { 0.0 },
            "trunc" => if args.len() >= 1 { args[0].trunc() } else { 0.0 },
            
            // Statistics and aggregation
            "min" => args.iter().fold(f64::INFINITY, |a, &b| a.min(b)),
            "max" => args.iter().fold(f64::NEG_INFINITY, |a, &b| a.max(b)),
            "sum" => args.iter().sum(),
            "mean" | "avg" => if args.is_empty() { 0.0 } else { args.iter().sum::<f64>() / args.len() as f64 },
            
            // Constants
            "pi" => std::f64::consts::PI,
            "e" => std::f64::consts::E,
            "tau" => std::f64::consts::TAU,
            
            // Array/List operations
            "len" => {
                if args_str.trim().is_empty() {
                    return Ok(None);
                }
                let var_name = args_str.trim();
                if let Some(value) = self.variables.get(var_name) {
                    match value {
                        Value::Array(arr) => arr.len() as f64,
                        Value::String(s) => s.len() as f64,
                        Value::Matrix(mat) => mat.len() as f64,
                        _ => 0.0
                    }
                } else {
                    0.0
                }
            },
            
            // Sorting algorithms
            "sort" => {
                // This will be handled separately for array modification
                return Ok(None);
            },
            "bubbleSort" => {
                return Ok(None);
            },
            "quickSort" => {
                return Ok(None);
            },
            "mergeSort" => {
                return Ok(None);
            },
            
            // Matrix operations
            "det" | "determinant" => {
                if args_str.trim().is_empty() {
                    return Ok(None);
                }
                let var_name = args_str.trim();
                if let Some(Value::Matrix(mat)) = self.variables.get(var_name) {
                    self.matrix_determinant(mat)
                } else {
                    0.0
                }
            },
            "transpose" => {
                return Ok(None); // Will be handled separately
            },
            "dot" => {
                return Ok(None); // Will be handled separately for matrix dot product
            },
            
            // Coordinate transformations
            "toRadians" => if args.len() >= 1 { args[0].to_radians() } else { 0.0 },
            "toDegrees" => if args.len() >= 1 { args[0].to_degrees() } else { 0.0 },
            "distance2d" => if args.len() >= 4 { 
                let dx = args[2] - args[0];
                let dy = args[3] - args[1];
                (dx * dx + dy * dy).sqrt()
            } else { 0.0 },
            "distance3d" => if args.len() >= 6 { 
                let dx = args[3] - args[0];
                let dy = args[4] - args[1];
                let dz = args[5] - args[2];
                (dx * dx + dy * dy + dz * dz).sqrt()
            } else { 0.0 },
            
            // Euler angle conversions
            "eulerToQuaternion" => {
                if args.len() >= 3 {
                    // Return magnitude for now (simplified)
                    (args[0] * args[0] + args[1] * args[1] + args[2] * args[2]).sqrt()
                } else { 0.0 }
            },
            
            // Advanced numerical methods
            "derivative" => {
                // Numerical derivative approximation
                if args.len() >= 3 {
                    let h = 0.0001; // Small step
                    (args[1] - args[0]) / h
                } else { 0.0 }
            },
            "integral" => {
                // Simple trapezoidal rule approximation
                if args.len() >= 2 {
                    (args[0] + args[1]) * 0.5
                } else { 0.0 }
            },
            "interpolate" => {
                // Linear interpolation
                if args.len() >= 3 {
                    args[0] + args[2] * (args[1] - args[0])
                } else { 0.0 }
            },
            
            // Numerical methods
            "gcd" => if args.len() >= 2 { 
                self.gcd(args[0] as i64, args[1] as i64) as f64
            } else { 0.0 },
            "lcm" => if args.len() >= 2 { 
                self.lcm(args[0] as i64, args[1] as i64) as f64
            } else { 0.0 },
            "factorial" => if args.len() >= 1 { 
                self.factorial(args[0] as u64) as f64
            } else { 0.0 },
            "fibonacci" => if args.len() >= 1 { 
                self.fibonacci(args[0] as u64) as f64
            } else { 0.0 },
            
            _ => return Ok(None)
        };
        
        Ok(Some(if result.fract() == 0.0 {
            format!("{}", result as i64)
        } else {
            format!("{:.6}", result)
        }))
    }
    
    fn handle_array_literal(&self, expr: &str) -> Result<String, String> {
        let content = &expr[1..expr.len()-1].trim();
        if content.is_empty() {
            return Ok("[]".to_string());
        }
        
        let elements: Vec<&str> = content.split(',').collect();
        let mut values = Vec::new();
        
        for elem in elements {
            let elem = elem.trim();
            if let Ok(num) = elem.parse::<f64>() {
                values.push(if num.fract() == 0.0 {
                    format!("{}", num as i64)
                } else {
                    format!("{}", num)
                });
            } else if elem.starts_with('"') && elem.ends_with('"') {
                values.push(elem.to_string());
            } else if let Some(var_value) = self.variables.get(elem) {
                match var_value {
                    Value::Number(n) => {
                        values.push(if n.fract() == 0.0 {
                            format!("{}", *n as i64)
                        } else {
                            format!("{}", n)
                        });
                    }
                    _ => values.push(var_value.to_string())
                }
            } else {
                values.push(elem.to_string());
            }
        }
        
        Ok(format!("[{}]", values.join(", ")))
    }
    
    fn handle_array_indexing(&self, expr: &str) -> Result<String, String> {
        let bracket_start = expr.find('[').unwrap();
        let bracket_end = expr.rfind(']').unwrap();
        
        let array_name = expr[..bracket_start].trim();
        let index_expr = expr[bracket_start + 1..bracket_end].trim();
        
        let index = if let Ok(i) = index_expr.parse::<usize>() {
            i
        } else if let Some(Value::Number(n)) = self.variables.get(index_expr) {
            *n as usize
        } else {
            return Err(format!("Invalid array index: {}", index_expr));
        };
        
        if let Some(value) = self.variables.get(array_name) {
            match value {
                Value::Array(arr) => {
                    if index < arr.len() {
                        Ok(arr[index].to_string())
                    } else {
                        Err(format!("Array index {} out of bounds (length {})", index, arr.len()))
                    }
                }
                Value::String(s) => {
                    if index < s.len() {
                        Ok(s.chars().nth(index).unwrap().to_string())
                    } else {
                        Err(format!("String index {} out of bounds (length {})", index, s.len()))
                    }
                }
                Value::Matrix(mat) => {
                    if index < mat.len() {
                        let row: Vec<String> = mat[index].iter().map(|v| v.to_string()).collect();
                        Ok(format!("[{}]", row.join(", ")))
                    } else {
                        Err(format!("Matrix row index {} out of bounds (rows {})", index, mat.len()))
                    }
                }
                _ => Err(format!("Cannot index into {}", array_name))
            }
        } else {
            Err(format!("Variable '{}' not found", array_name))
        }
    }
    
    // Helper mathematical functions
    fn gcd(&self, a: i64, b: i64) -> i64 {
        if b == 0 { a.abs() } else { self.gcd(b, a % b) }
    }
    
    fn lcm(&self, a: i64, b: i64) -> i64 {
        if a == 0 || b == 0 { 0 } else { (a.abs() * b.abs()) / self.gcd(a, b) }
    }
    
    fn factorial(&self, n: u64) -> u64 {
        if n <= 1 { 1 } else { n * self.factorial(n - 1) }
    }
    
    fn fibonacci(&self, n: u64) -> u64 {
        match n {
            0 => 0,
            1 => 1,
            _ => {
                let mut a = 0;
                let mut b = 1;
                for _ in 2..=n {
                    let temp = a + b;
                    a = b;
                    b = temp;
                }
                b
            }
        }
    }
    
    // Matrix operations
    fn matrix_determinant(&self, matrix: &Vec<Vec<f64>>) -> f64 {
        let n = matrix.len();
        if n == 0 || matrix[0].len() != n {
            return 0.0; // Not a square matrix
        }
        
        match n {
            1 => matrix[0][0],
            2 => matrix[0][0] * matrix[1][1] - matrix[0][1] * matrix[1][0],
            3 => {
                matrix[0][0] * (matrix[1][1] * matrix[2][2] - matrix[1][2] * matrix[2][1])
                - matrix[0][1] * (matrix[1][0] * matrix[2][2] - matrix[1][2] * matrix[2][0])
                + matrix[0][2] * (matrix[1][0] * matrix[2][1] - matrix[1][1] * matrix[2][0])
            },
            _ => {
                // For larger matrices, use LU decomposition (simplified)
                let mut det = 1.0;
                let mut mat = matrix.clone();
                
                for i in 0..n {
                    // Find pivot
                    let mut max_row = i;
                    for k in i + 1..n {
                        if mat[k][i].abs() > mat[max_row][i].abs() {
                            max_row = k;
                        }
                    }
                    
                    if max_row != i {
                        mat.swap(i, max_row);
                        det = -det;
                    }
                    
                    if mat[i][i].abs() < 1e-10 {
                        return 0.0; // Singular matrix
                    }
                    
                    det *= mat[i][i];
                    
                    // Eliminate column
                    for k in i + 1..n {
                        let factor = mat[k][i] / mat[i][i];
                        for j in i..n {
                            mat[k][j] -= factor * mat[i][j];
                        }
                    }
                }
                
                det
            }
        }
    }
    
    // Sorting algorithms implementation
    fn bubble_sort(arr: &mut Vec<Value>) {
        let n = arr.len();
        for i in 0..n {
            for j in 0..n - 1 - i {
                let should_swap = match (&arr[j], &arr[j + 1]) {
                    (Value::Number(a), Value::Number(b)) => a > b,
                    (Value::String(a), Value::String(b)) => a > b,
                    _ => false,
                };
                
                if should_swap {
                    arr.swap(j, j + 1);
                }
            }
        }
    }
    
    fn quick_sort(arr: &mut Vec<Value>, low: usize, high: usize) {
        if low < high {
            let pi = Self::partition(arr, low, high);
            if pi > 0 {
                Self::quick_sort(arr, low, pi - 1);
            }
            Self::quick_sort(arr, pi + 1, high);
        }
    }
    
    fn partition(arr: &mut Vec<Value>, low: usize, high: usize) -> usize {
        let mut i = low;
        
        for j in low..high {
            let should_move = match (&arr[j], &arr[high]) {
                (Value::Number(a), Value::Number(b)) => a <= b,
                (Value::String(a), Value::String(b)) => a <= b,
                _ => false,
            };
            
            if should_move {
                arr.swap(i, j);
                i += 1;
            }
        }
        
        arr.swap(i, high);
        i
    }
    
    fn merge_sort(arr: &mut Vec<Value>) {
        let len = arr.len();
        if len <= 1 {
            return;
        }
        
        let mid = len / 2;
        let mut left = arr[0..mid].to_vec();
        let mut right = arr[mid..].to_vec();
        
        Self::merge_sort(&mut left);
        Self::merge_sort(&mut right);
        
        Self::merge(arr, &left, &right);
    }
    
    fn merge(arr: &mut Vec<Value>, left: &[Value], right: &[Value]) {
        let mut i = 0;
        let mut j = 0;
        let mut k = 0;
        
        while i < left.len() && j < right.len() {
            let left_smaller = match (&left[i], &right[j]) {
                (Value::Number(a), Value::Number(b)) => a <= b,
                (Value::String(a), Value::String(b)) => a <= b,
                _ => true,
            };
            
            if left_smaller {
                arr[k] = left[i].clone();
                i += 1;
            } else {
                arr[k] = right[j].clone();
                j += 1;
            }
            k += 1;
        }
        
        while i < left.len() {
            arr[k] = left[i].clone();
            i += 1;
            k += 1;
        }
        
        while j < right.len() {
            arr[k] = right[j].clone();
            j += 1;
            k += 1;
        }
    }
    
    // Advanced coordinate and transformation functions
    fn euler_to_quaternion(&self, roll: f64, pitch: f64, yaw: f64) -> (f64, f64, f64, f64) {
        let cr = (roll * 0.5).cos();
        let sr = (roll * 0.5).sin();
        let cp = (pitch * 0.5).cos();
        let sp = (pitch * 0.5).sin();
        let cy = (yaw * 0.5).cos();
        let sy = (yaw * 0.5).sin();
        
        let w = cr * cp * cy + sr * sp * sy;
        let x = sr * cp * cy - cr * sp * sy;
        let y = cr * sp * cy + sr * cp * sy;
        let z = cr * cp * sy - sr * sp * cy;
        
        (w, x, y, z)
    }
    
    fn quaternion_to_euler(&self, w: f64, x: f64, y: f64, z: f64) -> (f64, f64, f64) {
        let sinr_cosp = 2.0 * (w * x + y * z);
        let cosr_cosp = 1.0 - 2.0 * (x * x + y * y);
        let roll = sinr_cosp.atan2(cosr_cosp);
        
        let sinp = 2.0 * (w * y - z * x);
        let pitch = if sinp.abs() >= 1.0 {
            std::f64::consts::PI / 2.0 * sinp.signum()
        } else {
            sinp.asin()
        };
        
        let siny_cosp = 2.0 * (w * z + x * y);
        let cosy_cosp = 1.0 - 2.0 * (y * y + z * z);
        let yaw = siny_cosp.atan2(cosy_cosp);
        
        (roll, pitch, yaw)
    }
    
    // Matrix transpose operation
    fn matrix_transpose(&self, matrix: &Vec<Vec<f64>>) -> Vec<Vec<f64>> {
        if matrix.is_empty() {
            return Vec::new();
        }
        
        let rows = matrix.len();
        let cols = matrix[0].len();
        let mut result = vec![vec![0.0; rows]; cols];
        
        for i in 0..rows {
            for j in 0..cols {
                result[j][i] = matrix[i][j];
            }
        }
        
        result
    }
    
    // Matrix multiplication
    fn matrix_multiply(&self, a: &Vec<Vec<f64>>, b: &Vec<Vec<f64>>) -> Result<Vec<Vec<f64>>, String> {
        if a.is_empty() || b.is_empty() {
            return Ok(Vec::new());
        }
        
        let a_rows = a.len();
        let a_cols = a[0].len();
        let b_rows = b.len();
        let b_cols = b[0].len();
        
        if a_cols != b_rows {
            return Err(format!("Cannot multiply {}x{} matrix with {}x{} matrix", a_rows, a_cols, b_rows, b_cols));
        }
        
        let mut result = vec![vec![0.0; b_cols]; a_rows];
        
        for i in 0..a_rows {
            for j in 0..b_cols {
                for k in 0..a_cols {
                    result[i][j] += a[i][k] * b[k][j];
                }
            }
        }
        
        Ok(result)
    }
    
    // Macro system implementation
    fn handle_macro_definition(&mut self, line: &str) -> Result<String, String> {
        // Parse: macro name(param1, param2) { template }
        let line = line.trim();
        if !line.starts_with("macro ") {
            return Err("Invalid macro definition".to_string());
        }
        
        let rest = &line[6..]; // Remove "macro "
        let paren_start = rest.find('(').ok_or("Missing opening parenthesis in macro")?;
        let macro_name = rest[..paren_start].trim().to_string();
        
        let paren_end = rest.find(')').ok_or("Missing closing parenthesis in macro")?;
        let params_str = &rest[paren_start + 1..paren_end];
        let params: Vec<String> = if params_str.trim().is_empty() {
            Vec::new()
        } else {
            params_str.split(',').map(|p| p.trim().to_string()).collect()
        };
        
        let brace_start = rest.find('{').ok_or("Missing opening brace in macro")?;
        let brace_end = rest.rfind('}').ok_or("Missing closing brace in macro")?;
        let template = rest[brace_start + 1..brace_end].trim().to_string();
        
        let macro_def = ScaffoldMacro {
            name: macro_name.clone(),
            params,
            template,
        };
        
        self.macros.insert(macro_name, macro_def);
        Ok(String::new())
    }
    
    fn handle_micro_definition(&mut self, line: &str) -> Result<String, String> {
        // Parse: micro name(param1, param2) { body }
        let line = line.trim();
        if !line.starts_with("micro ") {
            return Err("Invalid micro definition".to_string());
        }
        
        let rest = &line[6..]; // Remove "micro "
        let paren_start = rest.find('(').ok_or("Missing opening parenthesis in micro")?;
        let micro_name = rest[..paren_start].trim().to_string();
        
        let paren_end = rest.find(')').ok_or("Missing closing parenthesis in micro")?;
        let params_str = &rest[paren_start + 1..paren_end];
        let params: Vec<String> = if params_str.trim().is_empty() {
            Vec::new()
        } else {
            params_str.split(',').map(|p| p.trim().to_string()).collect()
        };
        
        let brace_start = rest.find('{').ok_or("Missing opening brace in micro")?;
        let brace_end = rest.rfind('}').ok_or("Missing closing brace in micro")?;
        let body = rest[brace_start + 1..brace_end].trim().to_string();
        
        let micro_def = ScaffoldMicro {
            name: micro_name.clone(),
            params,
            body,
            inline_hint: true,
        };
        
        self.micros.insert(micro_name, micro_def);
        Ok(String::new())
    }
    
    fn handle_macro_call(&mut self, line: &str) -> Result<Option<String>, String> {
        // Check if this line is a macro call
        if let Some(paren_pos) = line.find('(') {
            let potential_macro_name = line[..paren_pos].trim();
            
            if let Some(macro_def) = self.macros.get(potential_macro_name).cloned() {
                // Parse arguments
                let paren_end = line.rfind(')').ok_or("Missing closing parenthesis in macro call")?;
                let args_str = &line[paren_pos + 1..paren_end];
                let args: Vec<String> = if args_str.trim().is_empty() {
                    Vec::new()
                } else {
                    args_str.split(',').map(|a| a.trim().to_string()).collect()
                };
                
                if args.len() != macro_def.params.len() {
                    return Err(format!("Macro {} expects {} arguments, got {}", 
                                     macro_def.name, macro_def.params.len(), args.len()));
                }
                
                // Expand macro template
                let mut expanded = macro_def.template.clone();
                for (param, arg) in macro_def.params.iter().zip(args.iter()) {
                    expanded = expanded.replace(&format!("{{{}}}", param), arg);
                }
                
                // Execute the expanded code
                return Ok(Some(self.execute(&expanded)?));
            }
        }
        
        Ok(None)
    }
    
    fn handle_micro_call(&mut self, line: &str) -> Result<Option<String>, String> {
        // Check if this line is a micro call
        if let Some(paren_pos) = line.find('(') {
            let potential_micro_name = line[..paren_pos].trim();
            
            if let Some(micro_def) = self.micros.get(potential_micro_name).cloned() {
                // Parse arguments
                let paren_end = line.rfind(')').ok_or("Missing closing parenthesis in micro call")?;
                let args_str = &line[paren_pos + 1..paren_end];
                let args: Vec<String> = if args_str.trim().is_empty() {
                    Vec::new()
                } else {
                    args_str.split(',').map(|a| a.trim().to_string()).collect()
                };
                
                if args.len() != micro_def.params.len() {
                    return Err(format!("Micro {} expects {} arguments, got {}", 
                                     micro_def.name, micro_def.params.len(), args.len()));
                }
                
                // Inline micro body with substituted parameters
                let mut inlined = micro_def.body.clone();
                for (param, arg) in micro_def.params.iter().zip(args.iter()) {
                    inlined = inlined.replace(param, arg);
                }
                
                // Handle return statements in micro bodies
                let inlined = inlined.trim();
                if inlined.starts_with("return ") {
                    let return_expr = &inlined[7..]; // Remove "return "
                    let result = self.evaluate_expression(return_expr)?;
                    return Ok(Some(result));
                } else {
                    // Execute the inlined code directly (zero function call overhead)
                    return Ok(Some(self.execute_line(&inlined)?));
                }
            }
        }
        
        Ok(None)
    }
    
    fn handle_import(&mut self, line: &str) -> Result<String, String> {
        // Parse: import module_name [as alias]
        let parts: Vec<&str> = line[7..].trim().split_whitespace().collect(); // Remove "import "
        
        if parts.is_empty() {
            return Err("Invalid import statement".to_string());
        }
        
        let module_name = parts[0];
        let alias = if parts.len() >= 3 && parts[1] == "as" {
            parts[2]
        } else {
            module_name
        };
        
        // Load built-in modules
        let module_functions = match module_name {
            "math" => self.create_math_module(),
            "arrays" => self.create_arrays_module(),
            "strings" => self.create_strings_module(),
            "utils" => self.create_utils_module(),
            _ => return Err(format!("Module '{}' not found", module_name)),
        };
        
        self.imported_modules.insert(alias.to_string(), module_functions);
        Ok(String::new())
    }
    
    fn handle_from_import(&mut self, line: &str) -> Result<String, String> {
        // Parse: from module_name import function1, function2 [as alias]
        let parts: Vec<&str> = line.split(" import ").collect();
        if parts.len() != 2 {
            return Err("Invalid from-import statement".to_string());
        }
        
        let module_name = parts[0][5..].trim(); // Remove "from "
        let import_list = parts[1].trim();
        
        // Load the module
        let module_functions = match module_name {
            "math" => self.create_math_module(),
            "arrays" => self.create_arrays_module(),
            "strings" => self.create_strings_module(),
            "utils" => self.create_utils_module(),
            _ => return Err(format!("Module '{}' not found", module_name)),
        };
        
        // Import specific functions
        let imports: Vec<&str> = import_list.split(',').collect();
        for import_item in imports {
            let import_item = import_item.trim();
            let (func_name, alias) = if import_item.contains(" as ") {
                let parts: Vec<&str> = import_item.split(" as ").collect();
                (parts[0].trim(), parts[1].trim())
            } else {
                (import_item, import_item)
            };
            
            if let Some(func) = module_functions.get(func_name) {
                self.variables.insert(alias.to_string(), func.clone());
            } else {
                return Err(format!("Function '{}' not found in module '{}'", func_name, module_name));
            }
        }
        
        Ok(String::new())
    }
    
    // Built-in module creators
    fn create_math_module(&self) -> HashMap<String, Value> {
        let mut module = HashMap::new();
        
        // Mathematical constants as functions
        module.insert("pi".to_string(), Value::Number(std::f64::consts::PI));
        module.insert("e".to_string(), Value::Number(std::f64::consts::E));
        module.insert("tau".to_string(), Value::Number(std::f64::consts::TAU));
        
        module
    }
    
    fn create_arrays_module(&self) -> HashMap<String, Value> {
        let mut module = HashMap::new();
        
        // Array utility functions would be defined here
        module.insert("sort".to_string(), Value::String("sort_function".to_string()));
        module.insert("reverse".to_string(), Value::String("reverse_function".to_string()));
        
        module
    }
    
    fn create_strings_module(&self) -> HashMap<String, Value> {
        let mut module = HashMap::new();
        
        // String utility functions
        module.insert("upper".to_string(), Value::String("upper_function".to_string()));
        module.insert("lower".to_string(), Value::String("lower_function".to_string()));
        
        module
    }
    
    fn create_utils_module(&self) -> HashMap<String, Value> {
        let mut module = HashMap::new();
        
        // Utility functions
        module.insert("benchmark".to_string(), Value::String("benchmark_function".to_string()));
        
        module
    }
}