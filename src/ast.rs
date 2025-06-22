#[derive(Debug, Clone, PartialEq)]
pub enum Type {
    // Basic types
    Int,
    Float,
    String,
    Bool,
    Void,
    
    // Advanced numeric types
    Int8, Int16, Int32, Int64,
    UInt8, UInt16, UInt32, UInt64,
    Float32, Float64,
    Decimal,
    BigInt,
    
    // System types
    Thread,
    Mutex,
    Channel,
    Future,
    
    // GPU types
    GpuBuffer,
    GpuKernel,
    GpuMemory,
    
    // OS types
    Process,
    File,
    Socket,
    SystemInfo,
    
    // Complex types
    Array(Box<Type>),
    Optional(Box<Type>),
    Result(Box<Type>, Box<Type>),
    Custom(String),
}

#[derive(Debug, Clone, PartialEq)]
pub struct Program {
    pub apps: Vec<App>,
    pub imports: Vec<Import>,
    pub config: Option<Config>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct Import {
    pub module: String,
    pub items: Vec<ImportItem>,
    pub alias: Option<String>,
    pub is_wildcard: bool, // import module::*
}

#[derive(Debug, Clone, PartialEq)]
pub struct ImportItem {
    pub name: String,
    pub alias: Option<String>,
    pub is_type: bool, // Import type vs value
}

#[derive(Debug, Clone, PartialEq)]
pub struct Library {
    pub name: String,
    pub version: String,
    pub path: String,
    pub exports: Vec<Export>,
    pub dependencies: Vec<Dependency>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct Export {
    pub name: String,
    pub item_type: ExportType,
    pub visibility: Visibility,
}

#[derive(Debug, Clone, PartialEq)]
pub enum ExportType {
    Function(Function),
    Class(Class),
    Trait(Trait),
    Constant(Field),
    Type(Type),
}

#[derive(Debug, Clone, PartialEq)]
pub struct Dependency {
    pub name: String,
    pub version: String,
    pub source: DependencySource,
}

#[derive(Debug, Clone, PartialEq)]
pub enum DependencySource {
    Registry(String), // Package registry
    Git(String),      // Git repository
    Local(String),    // Local path
    System,           // System library
}

#[derive(Debug, Clone, PartialEq)]
pub struct Config {
    pub optimization_level: OptimizationLevel,
    pub target_platform: TargetPlatform,
    pub memory_model: MemoryModel,
}

#[derive(Debug, Clone, PartialEq)]
pub enum OptimizationLevel {
    Debug,
    Release,
    Hypercar, // Maximum optimization
}

#[derive(Debug, Clone, PartialEq)]
pub enum TargetPlatform {
    X86_64,
    ARM64,
    GPU_CUDA,
    GPU_OpenCL,
    WebAssembly,
}

#[derive(Debug, Clone, PartialEq)]
pub enum MemoryModel {
    Safe,
    Fast,
    Hypercar, // Zero-copy, manual management
}

#[derive(Debug, Clone, PartialEq)]
pub struct App {
    pub name: String,
    pub functions: Vec<Function>,
    pub methods: Vec<Method>,
    pub classes: Vec<Class>,
    pub traits: Vec<Trait>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct Class {
    pub name: String,
    pub parent: Option<String>, // Inheritance support
    pub fields: Vec<Field>,
    pub methods: Vec<Method>,
    pub constructors: Vec<Constructor>,
    pub implements: Vec<String>, // Trait implementation
    pub is_abstract: bool,
    pub is_final: bool,
    pub visibility: Visibility,
    pub generic_params: Vec<GenericParameter>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct Constructor {
    pub parameters: Vec<Parameter>,
    pub body: Block,
    pub visibility: Visibility,
    pub calls_super: Option<Vec<Expression>>, // super() call
}

#[derive(Debug, Clone, PartialEq)]
pub struct GenericParameter {
    pub name: String,
    pub bounds: Vec<String>, // Trait bounds
    pub default: Option<Type>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct Trait {
    pub name: String,
    pub methods: Vec<MethodSignature>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct Field {
    pub name: String,
    pub field_type: Type,
    pub visibility: Visibility,
}

#[derive(Debug, Clone, PartialEq)]
pub enum Visibility {
    Public,
    Private,
    Protected,
}

#[derive(Debug, Clone, PartialEq)]
pub struct Method {
    pub name: String,
    pub parameters: Vec<Parameter>,
    pub return_type: Type,
    pub body: Option<Block>, // None for abstract methods
    pub visibility: Visibility,
    pub is_static: bool,
    pub is_async: bool,
    pub is_gpu: bool,
    pub is_abstract: bool,
    pub is_virtual: bool,
    pub is_override: bool,
    pub is_final: bool,
    pub generic_params: Vec<GenericParameter>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct MethodSignature {
    pub name: String,
    pub parameters: Vec<Parameter>,
    pub return_type: Type,
}

#[derive(Debug, Clone, PartialEq)]
pub struct Function {
    pub name: String,
    pub parameters: Vec<Parameter>,
    pub return_type: Type,
    pub body: Block,
    pub attributes: Vec<Attribute>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct Attribute {
    pub name: String,
    pub args: Vec<String>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct Parameter {
    pub name: String,
    pub param_type: Type,
    pub default_value: Option<Expression>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct Block {
    pub statements: Vec<Statement>,
}

#[derive(Debug, Clone, PartialEq)]
pub enum Statement {
    // Basic statements
    Let {
        name: String,
        var_type: Option<Type>,
        value: Expression,
    },
    Var {
        name: String,
        var_type: Option<Type>,
        value: Expression,
    },
    Assignment {
        name: String,
        value: Expression,
    },
    Function {
        name: String,
        parameters: Vec<String>,
        body: Vec<Statement>,
        return_type: Option<Type>,
    },
    
    // Control flow
    If {
        condition: Expression,
        then_block: Vec<Statement>,
        else_block: Option<Vec<Statement>>,
    },
    While {
        condition: Expression,
        body: Vec<Statement>,
    },
    For {
        variable: String,
        iterable: Expression,
        body: Vec<Statement>,
    },
    Match {
        expression: Expression,
        arms: Vec<MatchArm>,
    },
    
    // Error handling
    Try {
        body: Block,
        catch_blocks: Vec<CatchBlock>,
        finally_block: Option<Block>,
    },
    Throw(Expression),
    
    // Threading and concurrency
    Spawn {
        thread_name: Option<String>,
        body: Block,
    },
    Await(Expression),
    Lock {
        mutex: Expression,
        body: Block,
    },
    
    // GPU operations
    GpuKernel {
        name: String,
        grid_size: Expression,
        block_size: Expression,
        body: Block,
    },
    GpuMemcpy {
        destination: Expression,
        source: Expression,
        size: Expression,
    },
    
    // System operations
    SystemCall {
        call: String,
        args: Vec<Expression>,
    },
    FileOperation {
        operation: FileOp,
        path: Expression,
        data: Option<Expression>,
    },
    
    // Basic statements
    Return { value: Option<Expression> },
    Break,
    Continue,
    Expression(Expression),
    
    // OOP statements
    ClassInstantiation {
        class_name: String,
        constructor_args: Vec<Expression>,
        variable_name: String,
    },
    SuperCall {
        method: String,
        arguments: Vec<Expression>,
    },
    InterfaceImplementation {
        interface: String,
        methods: Vec<Method>,
    },

    // Additional OOP statements for parsing
    Class {
        name: String,
        parent: Option<String>,
        interfaces: Vec<String>,
        body: Vec<Statement>,
    },
    Enum {
        name: String,
        variants: Vec<(String, Option<Expression>)>,
    },
    Interface {
        name: String,
        methods: Vec<(String, Vec<String>, Option<String>)>, // (name, parameters, return_type)
    },
}

#[derive(Debug, Clone, PartialEq)]
pub struct MatchArm {
    pub pattern: Pattern,
    pub guard: Option<Expression>,
    pub body: Block,
}

#[derive(Debug, Clone, PartialEq)]
pub enum Pattern {
    Literal(Expression),
    Identifier(String),
    Wildcard,
    Tuple(Vec<Pattern>),
}

#[derive(Debug, Clone, PartialEq)]
pub struct CatchBlock {
    pub exception_type: Option<Type>,
    pub variable: Option<String>,
    pub body: Block,
}

#[derive(Debug, Clone, PartialEq)]
pub enum FileOp {
    Read,
    Write,
    Append,
    Delete,
    Create,
    Exists,
}

#[derive(Debug, Clone, PartialEq)]
pub enum Expression {
    // Literals
    Number(i64),
    Float(f64),
    String(String),
    Boolean(bool),
    Null,
    
    // Variables and access
    Identifier(String),
    FieldAccess {
        object: Box<Expression>,
        field: String,
    },
    
    // Operations
    Binary {
        left: Box<Expression>,
        operator: BinaryOperator,
        right: Box<Expression>,
    },
    Unary {
        operator: UnaryOperator,
        operand: Box<Expression>,
    },
    
    // Function calls
    Call {
        function: String,
        arguments: Vec<Expression>,
    },
    MethodCall {
        object: Box<Expression>,
        method: String,
        arguments: Vec<Expression>,
    },
    
    // Collections
    Array(Vec<Expression>),
}

#[derive(Debug, Clone, PartialEq)]
pub enum BinaryOperator {
    // Arithmetic
    Add, Subtract, Multiply, Divide, Modulo,
    Power,
    
    // Comparison
    Equal, NotEqual,
    Less, Greater, LessEqual, GreaterEqual,
    
    // Logical
    And, Or,
    
    // String operations
    Concat,
}

#[derive(Debug, Clone, PartialEq)]
pub enum UnaryOperator {
    Not, Minus,
}

impl std::fmt::Display for BinaryOperator {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            BinaryOperator::Add => write!(f, "+"),
            BinaryOperator::Subtract => write!(f, "-"),
            BinaryOperator::Multiply => write!(f, "*"),
            BinaryOperator::Divide => write!(f, "/"),
            BinaryOperator::Modulo => write!(f, "%"),
            BinaryOperator::Power => write!(f, "^"),
            BinaryOperator::Equal => write!(f, "=="),
            BinaryOperator::NotEqual => write!(f, "!="),
            BinaryOperator::Less => write!(f, "<"),
            BinaryOperator::Greater => write!(f, ">"),
            BinaryOperator::LessEqual => write!(f, "<="),
            BinaryOperator::GreaterEqual => write!(f, ">="),
            BinaryOperator::And => write!(f, "&&"),
            BinaryOperator::Or => write!(f, "||"),
            BinaryOperator::Concat => write!(f, "++"),
        }
    }
}

impl std::fmt::Display for Type {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Type::Int => write!(f, "int"),
            Type::Float => write!(f, "float"),
            Type::String => write!(f, "str"),
            Type::Bool => write!(f, "bool"),
            Type::Void => write!(f, "void"),
            Type::Int8 => write!(f, "i8"),
            Type::Int16 => write!(f, "i16"),
            Type::Int32 => write!(f, "i32"),
            Type::Int64 => write!(f, "i64"),
            Type::UInt8 => write!(f, "u8"),
            Type::UInt16 => write!(f, "u16"),
            Type::UInt32 => write!(f, "u32"),
            Type::UInt64 => write!(f, "u64"),
            Type::Float32 => write!(f, "f32"),
            Type::Float64 => write!(f, "f64"),
            Type::Decimal => write!(f, "decimal"),
            Type::BigInt => write!(f, "bigint"),
            Type::Thread => write!(f, "Thread"),
            Type::Mutex => write!(f, "Mutex"),
            Type::Channel => write!(f, "Channel"),
            Type::Future => write!(f, "Future"),
            Type::GpuBuffer => write!(f, "GpuBuffer"),
            Type::GpuKernel => write!(f, "GpuKernel"),
            Type::GpuMemory => write!(f, "GpuMemory"),
            Type::Process => write!(f, "Process"),
            Type::File => write!(f, "File"),
            Type::Socket => write!(f, "Socket"),
            Type::SystemInfo => write!(f, "SystemInfo"),
            Type::Array(inner) => write!(f, "Array<{}>", inner),
            Type::Optional(inner) => write!(f, "Optional<{}>", inner),
            Type::Result(ok, err) => write!(f, "Result<{}, {}>", ok, err),
            Type::Custom(name) => write!(f, "{}", name),
        }
    }
}

