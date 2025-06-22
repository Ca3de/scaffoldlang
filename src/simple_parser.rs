use crate::lexer::Token;
use crate::ast::{Statement, Expression, BinaryOperator, UnaryOperator, Type};
use anyhow::{Result, anyhow};

#[derive(Debug)]
pub struct SimpleParser {
    tokens: Vec<Token>,
    position: usize,
}

impl SimpleParser {
    pub fn new(tokens: Vec<Token>) -> Self {
        Self {
            tokens,
            position: 0,
        }
    }

    pub fn parse(&mut self) -> Result<Vec<Statement>> {
        let mut statements = Vec::new();
        
        while !self.is_at_end() {
            self.skip_whitespace_and_comments();
            if self.is_at_end() {
                break;
            }
            
            if let Some(stmt) = self.parse_statement()? {
                statements.push(stmt);
            }
        }
        
        Ok(statements)
    }

    fn skip_whitespace_and_comments(&mut self) {
        while matches!(self.current_token(), Token::Newline | Token::Comment(_)) {
            self.advance();
            if self.is_at_end() {
                break;
            }
        }
    }

    fn parse_statement(&mut self) -> Result<Option<Statement>> {
        if self.is_at_end() {
            return Ok(None);
        }

        self.skip_whitespace_and_comments();
        
        if self.is_at_end() {
            return Ok(None);
        }

        match self.current_token() {
            Token::Let => Ok(Some(self.parse_let_statement()?)),
            Token::Fun => Ok(Some(self.parse_function_statement()?)),
            Token::If => Ok(Some(self.parse_if_statement()?)),
            Token::While => Ok(Some(self.parse_while_statement()?)),
            Token::For => Ok(Some(self.parse_for_statement()?)),
            Token::Return => Ok(Some(self.parse_return_statement()?)),
            Token::Class => Ok(Some(self.parse_class_statement()?)),
            Token::Enum => Ok(Some(self.parse_enum_statement()?)),
            Token::Interface => Ok(Some(self.parse_interface_statement()?)),
            Token::Identifier(name) => {
                if name == "var" {
                    Ok(Some(self.parse_var_statement()?))
                } else if name == "class" {
                    Ok(Some(self.parse_class_statement()?))
                } else if name == "enum" {
                    Ok(Some(self.parse_enum_statement()?))
                } else if name == "interface" {
                    Ok(Some(self.parse_interface_statement()?))
                } else {
                    // Check if it's an assignment
                    let current_pos = self.position;
                    self.advance(); // Move past identifier
                    
                    if matches!(self.current_token(), Token::Assign) {
                        // Reset position and parse as assignment
                        self.position = current_pos;
                        let name = self.consume_identifier()?;
                        self.consume_token(Token::Assign)?;
                        let value = self.parse_expression()?;
                        Ok(Some(Statement::Assignment { name, value }))
                    } else {
                        // Reset position and parse as expression statement
                        self.position = current_pos;
                        let expr = self.parse_expression()?;
                        Ok(Some(Statement::Expression(expr)))
                    }
                }
            }
            _ => {
                let expr = self.parse_expression()?;
                Ok(Some(Statement::Expression(expr)))
            }
        }
    }

    fn parse_let_statement(&mut self) -> Result<Statement> {
        self.consume_token(Token::Let)?;
        
        let name = self.consume_identifier()?;
        
        // Check for optional type annotation
        let var_type = if matches!(self.current_token(), Token::Colon) {
            self.advance(); // consume ':'
            Some(self.parse_type()?)
        } else {
            None
        };
        
        self.consume_token(Token::Assign)?;
        let value = self.parse_expression()?;
        
        Ok(Statement::Let {
            name,
            var_type,
            value,
        })
    }

    fn parse_var_statement(&mut self) -> Result<Statement> {
        self.consume_identifier()?; // consume "var"
        
        let name = self.consume_identifier()?;
        self.consume_token(Token::Assign)?;
        let value = self.parse_expression()?;
        
        Ok(Statement::Var {
            name,
            var_type: None,
            value,
        })
    }

    fn parse_function_statement(&mut self) -> Result<Statement> {
        self.consume_token(Token::Fun)?; // consume "fun"
        
        let name = self.consume_identifier()?;
        self.consume_token(Token::LeftParen)?;
        
        let mut parameters = Vec::new();
        while !matches!(self.current_token(), Token::RightParen) && !self.is_at_end() {
            parameters.push(self.consume_identifier()?);
            if matches!(self.current_token(), Token::Comma) {
                self.advance();
            }
        }
        
        self.consume_token(Token::RightParen)?;
        self.consume_token(Token::LeftBrace)?;
        
        let body = self.parse_block()?;
        
        self.consume_token(Token::RightBrace)?;
        
        Ok(Statement::Function {
            name,
            parameters,
            body,
            return_type: None,
        })
    }

    fn parse_if_statement(&mut self) -> Result<Statement> {
        self.consume_token(Token::If)?;
        let condition = self.parse_expression()?;
        self.consume_token(Token::LeftBrace)?;
        let then_block = self.parse_block()?;
        self.consume_token(Token::RightBrace)?;
        
        let else_block = if matches!(self.current_token(), Token::Else) {
            self.advance();
            self.consume_token(Token::LeftBrace)?;
            let block = self.parse_block()?;
            self.consume_token(Token::RightBrace)?;
            Some(block)
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
        self.consume_token(Token::While)?;
        let condition = self.parse_expression()?;
        self.consume_token(Token::LeftBrace)?;
        let body = self.parse_block()?;
        self.consume_token(Token::RightBrace)?;
        
        Ok(Statement::While { condition, body })
    }

    fn parse_for_statement(&mut self) -> Result<Statement> {
        self.consume_token(Token::For)?;
        let variable = self.consume_identifier()?;
        self.consume_token(Token::In)?;
        let iterable = self.parse_expression()?;
        self.consume_token(Token::LeftBrace)?;
        let body = self.parse_block()?;
        self.consume_token(Token::RightBrace)?;
        
        Ok(Statement::For {
            variable,
            iterable,
            body,
        })
    }

    fn parse_return_statement(&mut self) -> Result<Statement> {
        self.consume_token(Token::Return)?;
        
        let value = if matches!(self.current_token(), Token::Newline | Token::RightBrace) {
            None
        } else {
            Some(self.parse_expression()?)
        };
        
        Ok(Statement::Return { value })
    }

    fn parse_block(&mut self) -> Result<Vec<Statement>> {
        let mut statements = Vec::new();
        
        while !matches!(self.current_token(), Token::RightBrace) && !self.is_at_end() {
            self.skip_whitespace_and_comments();
            if matches!(self.current_token(), Token::RightBrace) {
                break;
            }
            
            if let Some(stmt) = self.parse_statement()? {
                statements.push(stmt);
            }
        }
        
        Ok(statements)
    }

    fn parse_expression(&mut self) -> Result<Expression> {
        self.parse_logical_or()
    }

    fn parse_logical_or(&mut self) -> Result<Expression> {
        let mut expr = self.parse_logical_and()?;
        
        while matches!(self.current_token(), Token::Or) {
            self.advance();
            let right = self.parse_logical_and()?;
            expr = Expression::Binary {
                left: Box::new(expr),
                operator: BinaryOperator::Or,
                right: Box::new(right),
            };
        }
        
        Ok(expr)
    }

    fn parse_logical_and(&mut self) -> Result<Expression> {
        let mut expr = self.parse_equality()?;
        
        while matches!(self.current_token(), Token::And) {
            self.advance();
            let right = self.parse_equality()?;
            expr = Expression::Binary {
                left: Box::new(expr),
                operator: BinaryOperator::And,
                right: Box::new(right),
            };
        }
        
        Ok(expr)
    }

    fn parse_equality(&mut self) -> Result<Expression> {
        let mut expr = self.parse_comparison()?;
        
        while matches!(self.current_token(), Token::Equal | Token::NotEqual) {
            let op = match self.current_token() {
                Token::Equal => BinaryOperator::Equal,
                Token::NotEqual => BinaryOperator::NotEqual,
                _ => unreachable!(),
            };
            self.advance();
            let right = self.parse_comparison()?;
            expr = Expression::Binary {
                left: Box::new(expr),
                operator: op,
                right: Box::new(right),
            };
        }
        
        Ok(expr)
    }

    fn parse_comparison(&mut self) -> Result<Expression> {
        let mut expr = self.parse_term()?;
        
        while matches!(self.current_token(), Token::Greater | Token::GreaterEqual | Token::Less | Token::LessEqual) {
            let op = match self.current_token() {
                Token::Greater => BinaryOperator::Greater,
                Token::GreaterEqual => BinaryOperator::GreaterEqual,
                Token::Less => BinaryOperator::Less,
                Token::LessEqual => BinaryOperator::LessEqual,
                _ => unreachable!(),
            };
            self.advance();
            let right = self.parse_term()?;
            expr = Expression::Binary {
                left: Box::new(expr),
                operator: op,
                right: Box::new(right),
            };
        }
        
        Ok(expr)
    }

    fn parse_term(&mut self) -> Result<Expression> {
        let mut expr = self.parse_factor()?;
        
        while matches!(self.current_token(), Token::Minus | Token::Plus) {
            let op = match self.current_token() {
                Token::Minus => BinaryOperator::Subtract,
                Token::Plus => BinaryOperator::Add,
                _ => unreachable!(),
            };
            self.advance();
            let right = self.parse_factor()?;
            expr = Expression::Binary {
                left: Box::new(expr),
                operator: op,
                right: Box::new(right),
            };
        }
        
        Ok(expr)
    }

    fn parse_factor(&mut self) -> Result<Expression> {
        let mut expr = self.parse_unary()?;
        
        while matches!(self.current_token(), Token::Divide | Token::Multiply | Token::Modulo) {
            let op = match self.current_token() {
                Token::Divide => BinaryOperator::Divide,
                Token::Multiply => BinaryOperator::Multiply,
                Token::Modulo => BinaryOperator::Modulo,
                _ => unreachable!(),
            };
            self.advance();
            let right = self.parse_unary()?;
            expr = Expression::Binary {
                left: Box::new(expr),
                operator: op,
                right: Box::new(right),
            };
        }
        
        Ok(expr)
    }

    fn parse_unary(&mut self) -> Result<Expression> {
        match self.current_token() {
            Token::Not => {
                self.advance();
                let expr = self.parse_unary()?;
                Ok(Expression::Unary {
                    operator: UnaryOperator::Not,
                    operand: Box::new(expr),
                })
            }
            Token::Minus => {
                self.advance();
                let expr = self.parse_unary()?;
                Ok(Expression::Unary {
                    operator: UnaryOperator::Minus,
                    operand: Box::new(expr),
                })
            }
            _ => self.parse_call(),
        }
    }

    fn parse_call(&mut self) -> Result<Expression> {
        let mut expr = self.parse_primary()?;
        
        while matches!(self.current_token(), Token::LeftParen | Token::Dot) {
            match self.current_token() {
                Token::LeftParen => {
                    self.advance();
                    let mut arguments = Vec::new();
                    
                    while !matches!(self.current_token(), Token::RightParen) && !self.is_at_end() {
                        arguments.push(self.parse_expression()?);
                        if matches!(self.current_token(), Token::Comma) {
                            self.advance();
                        }
                    }
                    
                    self.consume_token(Token::RightParen)?;
                    
                    expr = match expr {
                        Expression::Identifier(name) => Expression::Call {
                            function: name,
                            arguments,
                        },
                        _ => return Err(anyhow!("Invalid function call")),
                    };
                }
                Token::Dot => {
                    self.advance();
                    let method = self.consume_identifier()?;
                    
                    if matches!(self.current_token(), Token::LeftParen) {
                        self.advance();
                        let mut arguments = Vec::new();
                        
                        while !matches!(self.current_token(), Token::RightParen) && !self.is_at_end() {
                            arguments.push(self.parse_expression()?);
                            if matches!(self.current_token(), Token::Comma) {
                                self.advance();
                            }
                        }
                        
                        self.consume_token(Token::RightParen)?;
                        
                        expr = Expression::MethodCall {
                            object: Box::new(expr),
                            method,
                            arguments,
                        };
                    } else {
                        expr = Expression::FieldAccess {
                            object: Box::new(expr),
                            field: method,
                        };
                    }
                }
                _ => break,
            }
        }
        
        Ok(expr)
    }

    fn parse_primary(&mut self) -> Result<Expression> {
        match self.current_token().clone() {
            Token::True => {
                self.advance();
                Ok(Expression::Boolean(true))
            }
            Token::False => {
                self.advance();
                Ok(Expression::Boolean(false))
            }
            Token::Null => {
                self.advance();
                Ok(Expression::Null)
            }
            Token::Number(n) => {
                self.advance();
                Ok(Expression::Number(n))
            }
            Token::Float(f) => {
                self.advance();
                Ok(Expression::Float(f))
            }
            Token::String(s) => {
                self.advance();
                Ok(Expression::String(s))
            }
            Token::Identifier(name) => {
                self.advance();
                Ok(Expression::Identifier(name))
            }
            // Handle mathematical function tokens as identifiers
            Token::Sqrt => {
                self.advance();
                Ok(Expression::Identifier("sqrt".to_string()))
            }
            Token::Abs => {
                self.advance();
                Ok(Expression::Identifier("abs".to_string()))
            }
            Token::Sin => {
                self.advance();
                Ok(Expression::Identifier("sin".to_string()))
            }
            Token::Cos => {
                self.advance();
                Ok(Expression::Identifier("cos".to_string()))
            }
            Token::Tan => {
                self.advance();
                Ok(Expression::Identifier("tan".to_string()))
            }
            Token::Log => {
                self.advance();
                Ok(Expression::Identifier("log".to_string()))
            }
            Token::Exp => {
                self.advance();
                Ok(Expression::Identifier("exp".to_string()))
            }
            Token::Floor => {
                self.advance();
                Ok(Expression::Identifier("floor".to_string()))
            }
            Token::Ceil => {
                self.advance();
                Ok(Expression::Identifier("ceil".to_string()))
            }
            Token::Round => {
                self.advance();
                Ok(Expression::Identifier("round".to_string()))
            }
            Token::LeftParen => {
                self.advance();
                let expr = self.parse_expression()?;
                self.consume_token(Token::RightParen)?;
                Ok(expr)
            }
            _ => Err(anyhow!("Unexpected token: {:?}", self.current_token())),
        }
    }

    fn current_token(&self) -> &Token {
        self.tokens.get(self.position).unwrap_or(&Token::Eof)
    }

    fn advance(&mut self) {
        if self.position < self.tokens.len() {
            self.position += 1;
        }
    }

    fn is_at_end(&self) -> bool {
        self.position >= self.tokens.len() || matches!(self.current_token(), Token::Eof)
    }

    fn consume_token(&mut self, expected: Token) -> Result<()> {
        if std::mem::discriminant(self.current_token()) == std::mem::discriminant(&expected) {
            self.advance();
            Ok(())
        } else {
            Err(anyhow!("Expected {:?}, found {:?}", expected, self.current_token()))
        }
    }

    fn consume_identifier(&mut self) -> Result<String> {
        match self.current_token() {
            Token::Identifier(name) => {
                let name = name.clone();
                self.advance();
                Ok(name)
            }
            _ => Err(anyhow!("Expected identifier, found {:?}", self.current_token())),
        }
    }

    fn parse_type(&mut self) -> Result<Type> {
        match self.current_token() {
            Token::Identifier(type_name) => {
                let name = type_name.clone();
                self.advance();
                match name.as_str() {
                    "int" => Ok(Type::Int),
                    "float" => Ok(Type::Float),
                    "str" => Ok(Type::String),
                    "bool" => Ok(Type::Bool),
                    "double" => Ok(Type::Float64),
                    "char" => Ok(Type::String), // Treat char as string for now
                    "byte" => Ok(Type::UInt8),
                    "long" => Ok(Type::Int64),
                    "void" => Ok(Type::Void),
                    _ => Ok(Type::Custom(name)),
                }
            }
            Token::TypeInt => {
                self.advance();
                Ok(Type::Int)
            }
            Token::TypeFloat => {
                self.advance();
                Ok(Type::Float)
            }
            Token::TypeString => {
                self.advance();
                Ok(Type::String)
            }
            Token::TypeBool => {
                self.advance();
                Ok(Type::Bool)
            }
            Token::TypeDouble => {
                self.advance();
                Ok(Type::Float64)
            }
            Token::TypeChar => {
                self.advance();
                Ok(Type::String) // Treat char as string for now
            }
            Token::TypeByte => {
                self.advance();
                Ok(Type::UInt8)
            }
            Token::TypeLong => {
                self.advance();
                Ok(Type::Int64)
            }
            Token::TypeVoid => {
                self.advance();
                Ok(Type::Void)
            }
            _ => Err(anyhow!("Expected type, found {:?}", self.current_token())),
        }
    }

    fn parse_class_statement(&mut self) -> Result<Statement> {
        self.consume_token(Token::Class)?; // consume "class"
        let name = self.consume_identifier()?;
        
        // Check for inheritance
        let parent = if let Token::Identifier(s) = self.current_token() {
            if s == "extends" {
                self.advance(); // consume "extends"
                Some(self.consume_identifier()?)
            } else {
                None
            }
        } else {
            None
        };
        
        // Check for interface implementation
        let interfaces = if let Token::Identifier(s) = self.current_token() {
            if s == "implements" {
                self.advance(); // consume "implements"
                let mut interfaces = vec![self.consume_identifier()?];
                while matches!(self.current_token(), Token::Comma) {
                    self.advance(); // consume ","
                    interfaces.push(self.consume_identifier()?);
                }
                interfaces
            } else {
                Vec::new()
            }
        } else {
            Vec::new()
        };
        
        self.consume_token(Token::LeftBrace)?;
        let body = self.parse_block()?;
        self.consume_token(Token::RightBrace)?;
        
        Ok(Statement::Class {
            name,
            parent,
            interfaces,
            body,
        })
    }

    fn parse_enum_statement(&mut self) -> Result<Statement> {
        self.consume_token(Token::Enum)?; // consume "enum"
        let name = self.consume_identifier()?;
        
        self.consume_token(Token::LeftBrace)?;
        
        let mut variants = Vec::new();
        while !matches!(self.current_token(), Token::RightBrace) && !self.is_at_end() {
            let variant_name = self.consume_identifier()?;
            let value = if matches!(self.current_token(), Token::Assign) {
                self.advance(); // consume "="
                Some(self.parse_expression()?)
            } else {
                None
            };
            variants.push((variant_name, value));
            
            if matches!(self.current_token(), Token::Comma) {
                self.advance(); // consume ","
            }
        }
        
        self.consume_token(Token::RightBrace)?;
        
        Ok(Statement::Enum {
            name,
            variants,
        })
    }

    fn parse_interface_statement(&mut self) -> Result<Statement> {
        self.consume_token(Token::Interface)?; // consume "interface"
        let name = self.consume_identifier()?;
        
        self.consume_token(Token::LeftBrace)?;
        
        let mut methods = Vec::new();
        while !matches!(self.current_token(), Token::RightBrace) && !self.is_at_end() {
            let method_name = self.consume_identifier()?;
            self.consume_token(Token::LeftParen)?;
            
            let mut parameters = Vec::new();
            while !matches!(self.current_token(), Token::RightParen) && !self.is_at_end() {
                parameters.push(self.consume_identifier()?);
                if matches!(self.current_token(), Token::Comma) {
                    self.advance(); // consume ","
                }
            }
            
            self.consume_token(Token::RightParen)?;
            
            let return_type = if matches!(self.current_token(), Token::Colon) {
                self.advance(); // consume ":"
                Some(format!("{:?}", self.parse_type()?))
            } else {
                None
            };
            
            methods.push((method_name, parameters, return_type));
        }
        
        self.consume_token(Token::RightBrace)?;
        
        Ok(Statement::Interface {
            name,
            methods,
        })
    }
}
