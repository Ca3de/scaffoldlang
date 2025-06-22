use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::fs;
use crate::ast::{Program, Function, Statement, Expression};
use crate::parser::Parser;
use crate::lexer::Lexer;

/// Module System for ScaffoldLang
#[derive(Debug, Clone)]
pub struct Module {
    pub name: String,
    pub path: PathBuf,
    pub exports: HashMap<String, ModuleExport>,
    pub imports: HashMap<String, ModuleImport>,
    pub program: Option<Program>,
    pub is_loaded: bool,
    pub dependencies: Vec<String>,
}

#[derive(Debug, Clone)]
pub enum ModuleExport {
    Function(Function),
    Variable(String, Expression),
    Class(String),
    Interface(String),
    Enum(String),
    Module(String),
}

#[derive(Debug, Clone)]
pub struct ModuleImport {
    pub module_name: String,
    pub imported_items: Vec<ImportItem>,
    pub alias: Option<String>,
}

#[derive(Debug, Clone)]
pub enum ImportItem {
    All,
    Named(String),
    Default(String),
    Aliased(String, String),
}

/// Package manager for ScaffoldLang
#[derive(Debug)]
pub struct PackageManager {
    pub modules: HashMap<String, Module>,
    pub search_paths: Vec<PathBuf>,
    pub package_cache: HashMap<String, PathBuf>,
    pub dependency_graph: HashMap<String, Vec<String>>,
}

impl PackageManager {
    pub fn new() -> Self {
        let mut search_paths = Vec::new();
        search_paths.push(PathBuf::from("./"));
        search_paths.push(PathBuf::from("./lib/"));
        search_paths.push(PathBuf::from("./modules/"));
        search_paths.push(PathBuf::from("./packages/"));
        
        Self {
            modules: HashMap::new(),
            search_paths,
            package_cache: HashMap::new(),
            dependency_graph: HashMap::new(),
        }
    }

    /// Load a module by name
    pub fn load_module(&mut self, module_name: &str) -> Result<&Module, ModuleError> {
        if self.modules.contains_key(module_name) {
            return Ok(self.modules.get(module_name).unwrap());
        }

        // Find module file
        let module_path = self.find_module_file(module_name)?;
        
        // Read and parse module
        let source = fs::read_to_string(&module_path)
            .map_err(|e| ModuleError::FileNotFound(format!("Cannot read {}: {}", module_path.display(), e)))?;

        let mut lexer = Lexer::new(&source);
        let tokens = lexer.tokenize().map_err(|e| ModuleError::ParseError(format!("Lexer error: {:?}", e)))?;
        
        let mut parser = Parser::new(tokens);
        let program = parser.parse().map_err(|e| ModuleError::ParseError(format!("Parser error: {:?}", e)))?;

        // Extract exports and imports
        let (exports, imports, dependencies) = self.analyze_module(&program)?;

        let module = Module {
            name: module_name.to_string(),
            path: module_path,
            exports,
            imports,
            program: Some(program),
            is_loaded: true,
            dependencies,
        };

        self.modules.insert(module_name.to_string(), module);
        Ok(self.modules.get(module_name).unwrap())
    }

    /// Find module file in search paths
    fn find_module_file(&self, module_name: &str) -> Result<PathBuf, ModuleError> {
        let possible_extensions = vec!["scaffold", "sl"];
        
        for search_path in &self.search_paths {
            for ext in &possible_extensions {
                let mut file_path = search_path.clone();
                file_path.push(format!("{}.{}", module_name, ext));
                
                if file_path.exists() {
                    return Ok(file_path);
                }
                
                // Also try as directory with index file
                let mut dir_path = search_path.clone();
                dir_path.push(module_name);
                dir_path.push(format!("index.{}", ext));
                
                if dir_path.exists() {
                    return Ok(dir_path);
                }
            }
        }

        Err(ModuleError::ModuleNotFound(format!("Module '{}' not found in search paths", module_name)))
    }

    /// Analyze module to extract exports, imports, and dependencies
    fn analyze_module(&self, program: &Program) -> Result<(HashMap<String, ModuleExport>, HashMap<String, ModuleImport>, Vec<String>), ModuleError> {
        let mut exports = HashMap::new();
        let mut imports = HashMap::new();
        let mut dependencies = Vec::new();

        // Analyze imports
        for import in &program.imports {
            let module_import = ModuleImport {
                module_name: import.module.clone(),
                imported_items: import.items.iter().map(|item| {
                    match item.name.as_str() {
                        "*" => ImportItem::All,
                        _ => ImportItem::Named(item.name.clone()),
                    }
                }).collect(),
                alias: import.alias.clone(),
            };
            
            imports.insert(import.module.clone(), module_import);
            dependencies.push(import.module.clone());
        }

        // Analyze exports from apps
        for app in &program.apps {
            for function in &app.functions {
                if function.attributes.iter().any(|attr| attr.name == "export") {
                    exports.insert(
                        function.name.clone(),
                        ModuleExport::Function(function.clone())
                    );
                }
            }
        }

        Ok((exports, imports, dependencies))
    }

    /// Get module by name
    pub fn get_module(&self, name: &str) -> Option<&Module> {
        self.modules.get(name)
    }

    /// Check if module is loaded
    pub fn is_module_loaded(&self, name: &str) -> bool {
        self.modules.contains_key(name)
    }
}

/// Module system errors
#[derive(Debug)]
pub enum ModuleError {
    ModuleNotFound(String),
    FileNotFound(String),
    ParseError(String),
    CircularDependency(String),
    ExportNotFound(String),
    ImportError(String),
    DependencyError(String),
}

impl std::fmt::Display for ModuleError {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        match self {
            ModuleError::ModuleNotFound(msg) => write!(f, "Module not found: {}", msg),
            ModuleError::FileNotFound(msg) => write!(f, "File not found: {}", msg),
            ModuleError::ParseError(msg) => write!(f, "Parse error: {}", msg),
            ModuleError::CircularDependency(msg) => write!(f, "Circular dependency: {}", msg),
            ModuleError::ExportNotFound(msg) => write!(f, "Export not found: {}", msg),
            ModuleError::ImportError(msg) => write!(f, "Import error: {}", msg),
            ModuleError::DependencyError(msg) => write!(f, "Dependency error: {}", msg),
        }
    }
}

impl std::error::Error for ModuleError {}
