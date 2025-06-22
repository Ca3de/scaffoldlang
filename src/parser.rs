use anyhow::{Result, anyhow};
use crate::lexer::Token;
use crate::ast::*;

pub struct Parser {
    tokens: Vec<Token>,
    position: usize,
}

impl Parser {
    pub fn new(tokens: Vec<Token>) -> Self {
        Self {
            tokens,
            position: 0,
        }
    }

    pub fn parse(&mut self) -> Result<Program> {
        // Skip initial comments and whitespace
        self.skip_whitespace_and_comments();
        
        // Debug: Print current token after skipping
        println!("DEBUG: First token after skipping: {:?}", self.current_token());
        
        // Check if this is a simple script (no app structure)
        if self.is_simple_script() {
            println!("DEBUG: Detected as simple script");
            return self.parse_simple_script();
        }
        
        println!("DEBUG: Detected as OOP structure");
        
        // Parse OOP code (classes, interfaces, enums) without app wrapper
        return self.parse_oop_script();
    }

    fn is_simple_script(&self) -> bool {
        // Check if this contains OOP constructs or is an app
        for token in &self.tokens {
            match token {
                Token::App => return false, // Definitely not a simple script
                Token::Class | Token::Interface | Token::Enum => return false, // OOP constructs
                _ => continue,
            }
        }
        true // No OOP constructs found, treat as simple script
    }

    fn parse_simple_script(&mut self) -> Result<Program> {
        // Parse simple script without wrapping in app structure
        let mut statements = Vec::new();
        
        while !self.is_at_end() {
            self.skip_whitespace_and_comments();
            if self.is_at_end() {
                break;
            }
            
            // Handle expression statements (like function calls at script start)
            if let Some(stmt) = self.parse_statement_or_expression()? {
                statements.push(stmt);
            }
        }
        
        // Create a main function containing all statements
        let main_function = Function {
            name: "main".to_string(),
            parameters: Vec::new(),
            return_type: Type::Void,
            body: Block { statements },
            attributes: Vec::new(),
        };
        
        let app = App {
            name: "main".to_string(),
            functions: vec![main_function],
            methods: Vec::new(),
            classes: Vec::new(),
            traits: Vec::new(),
        };
        
        Ok(Program { 
            apps: vec![app], 
            imports: Vec::new(), 
            config: None 
        })
    }

    fn parse_statement_or_expression(&mut self) -> Result<Option<Statement>> {
        // Try to parse as a statement first
        if let Ok(Some(stmt)) = self.parse_statement() {
            return Ok(Some(stmt));
        }
        
        // If that fails, try to parse as an expression statement
        if let Ok(expr) = self.parse_expression() {
            return Ok(Some(Statement::Expression(expr)));
        }
        
        // If both fail, skip this token and continue
        self.advance();
        Ok(None)
    }

    fn skip_whitespace_and_comments(&mut self) {
        while !self.is_at_end() {
            match self.current_token() {
                Some(Token::Newline) | Some(Token::Comment(_)) => {
                    self.advance();
                }
                _ => break,
            }
        }
    }

    fn current_token(&self) -> Option<&Token> {
        self.tokens.get(self.position)
    }

    fn parse_app(&mut self) -> Result<App> {
        self.consume_token(&Token::App, "Expected 'app'")?;
        
        let name = self.consume_identifier()?;
        self.consume_token(&Token::LeftBrace, "Expected '{'")?;
        
        let mut functions = Vec::new();
        while !matches!(self.current_token(), Some(Token::RightBrace)) && !self.is_at_end() {
            if matches!(self.current_token(), Some(Token::Fun)) {
                functions.push(self.parse_function()?);
            } else {
                self.advance();
            }
        }
        
        self.consume_token(&Token::RightBrace, "Expected '}'")?;
        
        Ok(App { name, functions, methods: Vec::new(), classes: Vec::new(), traits: Vec::new() })
    }

    fn parse_function(&mut self) -> Result<Function> {
        self.consume_token(&Token::Fun, "Expected 'fun'")?;
        
        let name = match self.current_token() {
            Some(Token::Constructor) => {
                self.advance();
                "constructor".to_string()
            }
            _ => self.consume_identifier()?
        };
        
        self.consume_token(&Token::LeftParen, "Expected '('")?;
        
        let mut parameters = Vec::new();
        while !matches!(self.current_token(), Some(Token::RightParen)) && !self.is_at_end() {
            parameters.push(self.parse_parameter()?);
            if matches!(self.current_token(), Some(Token::Comma)) {
                self.advance();
            }
        }
        
        self.consume_token(&Token::RightParen, "Expected ')'")?;
        
        // Optional return type
        let return_type = if matches!(self.current_token(), Some(Token::Arrow)) {
            self.advance();
            self.parse_type()?
        } else {
            Type::Void
        };
        
        self.consume_token(&Token::LeftBrace, "Expected '{'")?;
        let body = self.parse_block()?;
        self.consume_token(&Token::RightBrace, "Expected '}'")?;
        
        Ok(Function {
            name,
            parameters,
            return_type,
            body,
            attributes: Vec::new(),
        })
    }

    fn parse_parameter(&mut self) -> Result<Parameter> {
        let name = self.consume_identifier()?;
        
        let param_type = if matches!(self.current_token(), Some(Token::Colon)) {
            self.advance();
            self.parse_type()?
        } else {
            Type::Custom("auto".to_string()) // Auto-infer type
        };
        
        Ok(Parameter { name, param_type, default_value: None })
    }

    fn parse_type(&mut self) -> Result<Type> {
        match self.current_token() {
            Some(Token::TypeInt) => { self.advance(); Ok(Type::Int) }
            Some(Token::TypeFloat) => { self.advance(); Ok(Type::Float) }
            Some(Token::TypeString) => { self.advance(); Ok(Type::String) }
            Some(Token::TypeBool) => { self.advance(); Ok(Type::Bool) }
            Some(Token::TypeVoid) => { self.advance(); Ok(Type::Void) }
            Some(Token::Identifier(name)) => {
                let name = name.clone();
                self.advance();
                Ok(Type::Custom(name))
            }
            _ => Err(anyhow!("Expected type")),
        }
    }

    fn parse_block(&mut self) -> Result<Block> {
        let mut statements = Vec::new();
        
        while !matches!(self.current_token(), Some(Token::RightBrace)) && !self.is_at_end() {
            self.skip_whitespace_and_comments();
            if matches!(self.current_token(), Some(Token::RightBrace)) {
                break;
            }
            
            if let Some(stmt) = self.parse_statement_or_expression()? {
                statements.push(stmt);
            }
        }
        
        Ok(Block { statements })
    }

    fn parse_statement(&mut self) -> Result<Option<Statement>> {
        match self.current_token() {
            Some(Token::Let) => Ok(Some(self.parse_let_statement()?)),
            Some(Token::Const) => Ok(Some(self.parse_var_statement()?)),
            Some(Token::If) => Ok(Some(self.parse_if_statement()?)),
            Some(Token::While) => Ok(Some(self.parse_while_statement()?)),
            Some(Token::For) => Ok(Some(self.parse_for_statement()?)),
            Some(Token::Return) => Ok(Some(self.parse_return_statement()?)),
            Some(Token::Class) => Ok(Some(self.parse_class_statement()?)),
            Some(Token::Interface) => Ok(Some(self.parse_interface_statement()?)),
            Some(Token::Enum) => Ok(Some(self.parse_enum_statement()?)),
            Some(Token::Identifier(name)) if name == "func" => Ok(Some(self.parse_func_statement()?)),
            Some(Token::Constructor) => Ok(Some(self.parse_constructor_statement()?)),
            Some(Token::Identifier(_)) => {
                // Look ahead to see if this is an assignment, method definition, or expression
                if self.position + 1 < self.tokens.len() {
                    let next_token = self.tokens.get(self.position + 1);
                    if matches!(next_token, Some(Token::Assign)) {
                        return Ok(Some(self.parse_assignment_statement()?));
                    } else if matches!(next_token, Some(Token::LeftParen)) {
                        // This is a method definition: identifier(params) { ... }
                        return Ok(Some(self.parse_method_definition()?));
                    }
                }
                // Otherwise it's an expression statement
                let expr = self.parse_expression()?;
                Ok(Some(Statement::Expression(expr)))
            }
            Some(Token::Newline) => {
                self.advance();
                Ok(None)
            }
            Some(Token::Comment(_)) => {
                self.advance();
                Ok(None)
            }
            _ => {
                // Expression statement
                let expr = self.parse_expression()?;
                Ok(Some(Statement::Expression(expr)))
            }
        }
    }

    fn parse_let_statement(&mut self) -> Result<Statement> {
        self.advance(); // consume 'let'
        
        let name = self.consume_identifier()?;
        
        let var_type = if matches!(self.current_token(), Some(Token::Colon)) {
            self.advance();
            Some(self.parse_type()?)
        } else {
            None
        };
        
        let value = if matches!(self.current_token(), Some(Token::Assign)) {
            self.advance();
            self.parse_expression()?
        } else {
            // Default value for uninitialized variables
            Expression::Number(0)
        };
        
        Ok(Statement::Let { name, var_type, value })
    }

    fn parse_var_statement(&mut self) -> Result<Statement> {
        self.advance(); // consume 'var' or 'const'
        
        let name = self.consume_identifier()?;
        
        let var_type = if matches!(self.current_token(), Some(Token::Colon)) {
            self.advance();
            Some(self.parse_type()?)
        } else {
            None
        };
        
        let value = if matches!(self.current_token(), Some(Token::Assign)) {
            self.advance();
            self.parse_expression()?
        } else {
            // Default value for uninitialized variables
            Expression::Number(0)
        };
        
        Ok(Statement::Var { name, var_type, value })
    }

    fn parse_func_statement(&mut self) -> Result<Statement> {
        self.advance(); // consume 'func'
        
        // Handle special case for constructor
        let name = match self.current_token() {
            Some(Token::Constructor) => {
                self.advance();
                "constructor".to_string()
            }
            _ => self.consume_identifier()?
        };
        
        self.consume_token(&Token::LeftParen, "Expected '('")?;
        
        let mut parameters = Vec::new();
        while !matches!(self.current_token(), Some(Token::RightParen)) && !self.is_at_end() {
            let param_name = self.consume_identifier()?;
            parameters.push(param_name);
            if matches!(self.current_token(), Some(Token::Comma)) {
                self.advance();
            }
        }
        
        self.consume_token(&Token::RightParen, "Expected ')'")?;
        
        let return_type = if matches!(self.current_token(), Some(Token::Arrow)) {
            self.advance();
            Some(self.parse_type()?)
        } else {
            None
        };
        
        self.consume_token(&Token::LeftBrace, "Expected '{'")?;
        let body = self.parse_block_statements()?;
        self.consume_token(&Token::RightBrace, "Expected '}'")?;
        
        Ok(Statement::Function {
            name,
            parameters,
            body,
            return_type,
        })
    }

    fn parse_block_statements(&mut self) -> Result<Vec<Statement>> {
        let mut statements = Vec::new();
        
        while !matches!(self.current_token(), Some(Token::RightBrace)) && !self.is_at_end() {
            self.skip_whitespace_and_comments();
            if matches!(self.current_token(), Some(Token::RightBrace)) {
                break;
            }
            
            if let Some(stmt) = self.parse_statement()? {
                statements.push(stmt);
            }
        }
        
        Ok(statements)
    }

    fn parse_if_statement(&mut self) -> Result<Statement> {
        self.advance(); // consume 'if'
        
        let condition = self.parse_expression()?;
        self.consume_token(&Token::LeftBrace, "Expected '{'")?;
        let then_block = self.parse_block_statements()?;
        self.consume_token(&Token::RightBrace, "Expected '}'")?;
        
        let else_block = if matches!(self.current_token(), Some(Token::Else)) {
            self.advance(); // consume 'else'
            
            // Check if this is an 'else if'
            if matches!(self.current_token(), Some(Token::If)) {
                // Parse the else-if as a nested if statement
                Some(vec![self.parse_if_statement()?])
            } else {
                // Regular else block
                self.consume_token(&Token::LeftBrace, "Expected '{'")?;
                let else_stmts = self.parse_block_statements()?;
                self.consume_token(&Token::RightBrace, "Expected '}'")?;
                Some(else_stmts)
            }
        } else {
            None
        };
        
        Ok(Statement::If {
            condition,
            then_block,
            else_block,
        })
    }

    fn parse_while_statement(&mut self) -> Result<Statement> {
        self.advance(); // consume 'while'
        
        let condition = self.parse_expression()?;
        self.consume_token(&Token::LeftBrace, "Expected '{'")?;
        let body = self.parse_block_statements()?;
        self.consume_token(&Token::RightBrace, "Expected '}'")?;
        
        Ok(Statement::While { condition, body })
    }

    fn parse_for_statement(&mut self) -> Result<Statement> {
        self.advance(); // consume 'for'
        
        let variable = self.consume_identifier()?;
        self.consume_token(&Token::In, "Expected 'in'")?;
        let iterable = self.parse_expression()?;
        self.consume_token(&Token::LeftBrace, "Expected '{'")?;
        let body = self.parse_block_statements()?;
        self.consume_token(&Token::RightBrace, "Expected '}'")?;
        
        Ok(Statement::For { variable, iterable, body })
    }

    fn parse_return_statement(&mut self) -> Result<Statement> {
        self.advance(); // consume 'return'
        
        let value = if matches!(self.current_token(), Some(Token::Newline) | Some(Token::RightBrace)) {
            None
        } else {
            Some(self.parse_expression()?)
        };
        
        Ok(Statement::Return { value })
    }

    fn parse_expression(&mut self) -> Result<Expression> {
        self.parse_logical_or()
    }

    fn parse_logical_or(&mut self) -> Result<Expression> {
        let mut expr = self.parse_logical_and()?;
        
        while matches!(self.current_token(), Some(Token::Or)) {
            let operator = BinaryOperator::Or;
            self.advance();
            let right = self.parse_logical_and()?;
            expr = Expression::Binary {
                left: Box::new(expr),
                operator,
                right: Box::new(right),
            };
        }
        
        Ok(expr)
    }

    fn parse_logical_and(&mut self) -> Result<Expression> {
        let mut expr = self.parse_equality()?;
        
        while matches!(self.current_token(), Some(Token::And)) {
            let operator = BinaryOperator::And;
            self.advance();
            let right = self.parse_equality()?;
            expr = Expression::Binary {
                left: Box::new(expr),
                operator,
                right: Box::new(right),
            };
        }
        
        Ok(expr)
    }

    fn parse_equality(&mut self) -> Result<Expression> {
        let mut expr = self.parse_comparison()?;
        
        while matches!(self.current_token(), Some(Token::Equal) | Some(Token::NotEqual)) {
            let operator = match self.current_token() {
                Some(Token::Equal) => BinaryOperator::Equal,
                Some(Token::NotEqual) => BinaryOperator::NotEqual,
                _ => unreachable!(),
            };
            self.advance();
            let right = self.parse_comparison()?;
            expr = Expression::Binary {
                left: Box::new(expr),
                operator,
                right: Box::new(right),
            };
        }
        
        Ok(expr)
    }

    fn parse_comparison(&mut self) -> Result<Expression> {
        let mut expr = self.parse_term()?;
        
        while matches!(self.current_token(), Some(Token::Greater) | Some(Token::GreaterEqual) | Some(Token::Less) | Some(Token::LessEqual)) {
            let operator = match self.current_token() {
                Some(Token::Greater) => BinaryOperator::Greater,
                Some(Token::GreaterEqual) => BinaryOperator::GreaterEqual,
                Some(Token::Less) => BinaryOperator::Less,
                Some(Token::LessEqual) => BinaryOperator::LessEqual,
                _ => unreachable!(),
            };
            self.advance();
            let right = self.parse_term()?;
            expr = Expression::Binary {
                left: Box::new(expr),
                operator,
                right: Box::new(right),
            };
        }
        
        Ok(expr)
    }

    fn parse_term(&mut self) -> Result<Expression> {
        let mut expr = self.parse_factor()?;
        
        while matches!(self.current_token(), Some(Token::Plus) | Some(Token::Minus)) {
            let operator = match self.current_token() {
                Some(Token::Plus) => BinaryOperator::Add,
                Some(Token::Minus) => BinaryOperator::Subtract,
                _ => unreachable!(),
            };
            self.advance();
            let right = self.parse_factor()?;
            expr = Expression::Binary {
                left: Box::new(expr),
                operator,
                right: Box::new(right),
            };
        }
        
        Ok(expr)
    }

    fn parse_factor(&mut self) -> Result<Expression> {
        let mut expr = self.parse_unary()?;
        
        while matches!(self.current_token(), Some(Token::Multiply) | Some(Token::Divide)) {
            let operator = match self.current_token() {
                Some(Token::Multiply) => BinaryOperator::Multiply,
                Some(Token::Divide) => BinaryOperator::Divide,
                _ => unreachable!(),
            };
            self.advance();
            let right = self.parse_unary()?;
            expr = Expression::Binary {
                left: Box::new(expr),
                operator,
                right: Box::new(right),
            };
        }
        
        Ok(expr)
    }

    fn parse_unary(&mut self) -> Result<Expression> {
        match self.current_token() {
            Some(Token::Minus) => {
                self.advance();
                let operand = self.parse_unary()?;
                Ok(Expression::Unary {
                    operator: UnaryOperator::Minus,
                    operand: Box::new(operand),
                })
            }
            Some(Token::Not) => {
                self.advance();
                let operand = self.parse_unary()?;
                Ok(Expression::Unary {
                    operator: UnaryOperator::Not,
                    operand: Box::new(operand),
                })
            }
            _ => self.parse_call(),
        }
    }

    fn parse_call(&mut self) -> Result<Expression> {
        let mut expr = self.parse_primary()?;
        
        loop {
            match self.current_token() {
                Some(Token::LeftParen) => {
                    // Function call: function() or object.method()
                    self.advance();
                    let mut arguments = Vec::new();
                    
                    while !matches!(self.current_token(), Some(Token::RightParen)) && !self.is_at_end() {
                        arguments.push(self.parse_expression()?);
                        if matches!(self.current_token(), Some(Token::Comma)) {
                            self.advance();
                        }
                    }
                    
                    self.consume_token(&Token::RightParen, "Expected ')'")?;
                    
                    // Check if this is a method call or function call
                    match expr {
                        Expression::Identifier(function_name) => {
                            // Regular function call or constructor call
                            expr = Expression::Call {
                                function: function_name,
                                arguments,
                            };
                        }
                        Expression::FieldAccess { object, field } => {
                            // Method call: object.method()
                            expr = Expression::MethodCall {
                                object,
                                method: field,
                                arguments,
                            };
                        }
                        _ => {
                            return Err(anyhow!("Invalid call expression"));
                        }
                    }
                }
                Some(Token::Dot) => {
                    // Field access: object.field
                    self.advance(); // consume '.'
                    let field = self.consume_identifier()?;
                    expr = Expression::FieldAccess {
                        object: Box::new(expr),
                        field,
                    };
                }
                _ => break,
            }
        }
        
        Ok(expr)
    }

    fn parse_primary(&mut self) -> Result<Expression> {
        match self.current_token() {
            Some(Token::Number(n)) => {
                let value = *n;
                self.advance();
                Ok(Expression::Number(value))
            }
            Some(Token::Float(f)) => {
                let value = *f;
                self.advance();
                Ok(Expression::Float(value))
            }
            Some(Token::String(s)) => {
                let value = s.clone();
                self.advance();
                Ok(Expression::String(value))
            }
            Some(Token::True) => {
                self.advance();
                Ok(Expression::Boolean(true))
            }
            Some(Token::False) => {
                self.advance();
                Ok(Expression::Boolean(false))
            }
            Some(Token::Identifier(name)) => {
                let name = name.clone();
                self.advance();
                Ok(Expression::Identifier(name))
            }
            Some(Token::This) => {
                self.advance();
                Ok(Expression::Identifier("this".to_string()))
            }
            Some(Token::LeftParen) => {
                self.advance();
                let expr = self.parse_expression()?;
                self.consume_token(&Token::RightParen, "Expected ')'")?;
                Ok(expr)
            }
            _ => Err(anyhow!("Unexpected token: {:?}", self.current_token())),
        }
    }

    fn consume_token(&mut self, expected: &Token, message: &str) -> Result<()> {
        if std::mem::discriminant(self.current_token().unwrap_or(&Token::Eof)) == std::mem::discriminant(expected) {
            self.advance();
            Ok(())
        } else {
            Err(anyhow!("{}: expected {:?}, found {:?}", message, expected, self.current_token()))
        }
    }

    fn consume_identifier(&mut self) -> Result<String> {
        match self.current_token() {
            Some(Token::Identifier(name)) => {
                let name = name.clone();
                self.advance();
                Ok(name)
            }
            _ => Err(anyhow!("Expected identifier, found {:?}", self.current_token())),
        }
    }

    fn advance(&mut self) {
        if !self.is_at_end() {
            self.position += 1;
        }
    }

    fn is_at_end(&self) -> bool {
        self.position >= self.tokens.len() || 
        matches!(self.current_token(), Some(Token::Eof))
    }

    fn parse_assignment_statement(&mut self) -> Result<Statement> {
        let name = self.consume_identifier()?;
        self.consume_token(&Token::Assign, "Expected '='")?;
        let value = self.parse_expression()?;
        
        Ok(Statement::Assignment { name, value })
    }

    fn parse_class_statement(&mut self) -> Result<Statement> {
        self.advance(); // consume 'class'
        
        let name = self.consume_identifier()?;
        
        // Check for inheritance
        let parent = if matches!(self.current_token(), Some(Token::Extends)) {
            self.advance(); // consume 'extends'
            Some(self.consume_identifier()?)
        } else {
            None
        };
        
        // Check for interface implementation
        let mut interfaces = Vec::new();
        if matches!(self.current_token(), Some(Token::Impl)) {
            self.advance(); // consume 'impl'
            loop {
                interfaces.push(self.consume_identifier()?);
                if matches!(self.current_token(), Some(Token::Comma)) {
                    self.advance();
                } else {
                    break;
                }
            }
        }
        
        self.consume_token(&Token::LeftBrace, "Expected '{'")?;
        let body = self.parse_block_statements()?;
        self.consume_token(&Token::RightBrace, "Expected '}'")?;
        
        Ok(Statement::Class {
            name,
            parent,
            interfaces,
            body,
        })
    }

    fn parse_interface_statement(&mut self) -> Result<Statement> {
        self.advance(); // consume 'interface'
        
        let name = self.consume_identifier()?;
        
        self.consume_token(&Token::LeftBrace, "Expected '{'")?;
        
        let mut methods = Vec::new();
        while !matches!(self.current_token(), Some(Token::RightBrace)) && !self.is_at_end() {
            self.skip_whitespace_and_comments();
            if matches!(self.current_token(), Some(Token::RightBrace)) {
                break;
            }
            
            // Parse method signature: methodName(params): returnType
            let method_name = self.consume_identifier()?;
            self.consume_token(&Token::LeftParen, "Expected '('")?;
            
            let mut parameters = Vec::new();
            while !matches!(self.current_token(), Some(Token::RightParen)) && !self.is_at_end() {
                let param_name = self.consume_identifier()?;
                parameters.push(param_name);
                if matches!(self.current_token(), Some(Token::Comma)) {
                    self.advance();
                }
            }
            
            self.consume_token(&Token::RightParen, "Expected ')'")?;
            
            let return_type = if matches!(self.current_token(), Some(Token::Colon)) {
                self.advance();
                Some(self.parse_type()?.to_string())
            } else {
                None
            };
            
            methods.push((method_name, parameters, return_type));
        }
        
        self.consume_token(&Token::RightBrace, "Expected '}'")?;
        
        Ok(Statement::Interface { name, methods })
    }

    fn parse_enum_statement(&mut self) -> Result<Statement> {
        self.advance(); // consume 'enum'
        
        let name = self.consume_identifier()?;
        
        self.consume_token(&Token::LeftBrace, "Expected '{'")?;
        
        let mut variants = Vec::new();
        while !matches!(self.current_token(), Some(Token::RightBrace)) && !self.is_at_end() {
            self.skip_whitespace_and_comments();
            if matches!(self.current_token(), Some(Token::RightBrace)) {
                break;
            }
            
            let variant_name = self.consume_identifier()?;
            
            let value = if matches!(self.current_token(), Some(Token::Assign)) {
                self.advance(); // consume '='
                Some(self.parse_expression()?)
            } else {
                None
            };
            
            variants.push((variant_name, value));
            
            if matches!(self.current_token(), Some(Token::Comma)) {
                self.advance();
            }
        }
        
        self.consume_token(&Token::RightBrace, "Expected '}'")?;
        
        Ok(Statement::Enum { name, variants })
    }

    fn parse_constructor_statement(&mut self) -> Result<Statement> {
        self.advance(); // consume 'constructor'
        
        self.consume_token(&Token::LeftParen, "Expected '('")?;
        
        let mut parameters = Vec::new();
        while !matches!(self.current_token(), Some(Token::RightParen)) && !self.is_at_end() {
            let param_name = self.consume_identifier()?;
            
            // Optional type annotation
            let param_type = if matches!(self.current_token(), Some(Token::Colon)) {
                self.advance();
                self.parse_type()?
            } else {
                Type::Custom("auto".to_string())
            };
            
            parameters.push(param_name);
            
            if matches!(self.current_token(), Some(Token::Comma)) {
                self.advance();
            }
        }
        
        self.consume_token(&Token::RightParen, "Expected ')'")?;
        self.consume_token(&Token::LeftBrace, "Expected '{'")?;
        let body = self.parse_block_statements()?;
        self.consume_token(&Token::RightBrace, "Expected '}'")?;
        
        // Return as a function statement for now
        Ok(Statement::Function {
            name: "constructor".to_string(),
            parameters,
            body,
            return_type: Some(Type::Void),
        })
    }

    fn parse_method_definition(&mut self) -> Result<Statement> {
        // Parse method definition: methodName(params) { body }
        let name = self.consume_identifier()?;
        
        self.consume_token(&Token::LeftParen, "Expected '('")?;
        
        let mut parameters = Vec::new();
        while !matches!(self.current_token(), Some(Token::RightParen)) && !self.is_at_end() {
            let param_name = self.consume_identifier()?;
            parameters.push(param_name);
            if matches!(self.current_token(), Some(Token::Comma)) {
                self.advance();
            }
        }
        
        self.consume_token(&Token::RightParen, "Expected ')'")?;
        
        let return_type = if matches!(self.current_token(), Some(Token::Arrow)) {
            self.advance();
            Some(self.parse_type()?)
        } else {
            None
        };
        
        self.consume_token(&Token::LeftBrace, "Expected '{'")?;
        let body = self.parse_block_statements()?;
        self.consume_token(&Token::RightBrace, "Expected '}'")?;
        
        Ok(Statement::Function {
            name,
            parameters,
            body,
            return_type,
        })
    }

    fn parse_oop_script(&mut self) -> Result<Program> {
        // Parse OOP constructs and wrap them in a default app
        let mut statements = Vec::new();
        let mut classes = Vec::new();
        
        while !self.is_at_end() {
            self.skip_whitespace_and_comments();
            if self.is_at_end() {
                break;
            }
            
            if let Some(stmt) = self.parse_statement()? {
                match &stmt {
                    Statement::Class { .. } => {
                        // Classes will be handled by the interpreter
                        statements.push(stmt);
                    }
                    _ => {
                        statements.push(stmt);
                    }
                }
            }
        }
        
        // Create a main function containing all statements
        let main_function = Function {
            name: "main".to_string(),
            parameters: Vec::new(),
            return_type: Type::Void,
            body: Block { statements },
            attributes: Vec::new(),
        };
        
        let app = App {
            name: "main".to_string(),
            functions: vec![main_function],
            methods: Vec::new(),
            classes,
            traits: Vec::new(),
        };
        
        Ok(Program { 
            apps: vec![app], 
            imports: Vec::new(), 
            config: None 
        })
    }
} 