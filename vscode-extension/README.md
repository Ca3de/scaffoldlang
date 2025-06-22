# ScaffoldLang VS Code Extension

🔥 **The hypercar of programming languages - now with VS Code integration!**

## Features

- **🏎️ One-Click Execution**: Press F5 or click the run button to execute ScaffoldLang files
- **⚡ Advanced Syntax Highlighting**: Full support for ScaffoldLang syntax including:
  - Macros and Micros
  - Advanced mathematical functions
  - Matrix and array operations
  - Coordinate transformations
  - Import system
- **🔍 Tokenization**: Debug your code with built-in tokenizer
- **⚙️ Compilation Support**: Compile ScaffoldLang files directly from VS Code
- **🎨 Smart Language Features**: Auto-completion, bracket matching, and code folding

## Quick Start

1. Install the ScaffoldLang extension
2. Open a `.scaffold` or `.sl` file
3. Press **F5** to run or use the run button in the editor
4. View output in the ScaffoldLang output panel

## Commands

- **🏎️ Run ScaffoldLang File** (`F5`): Execute the current ScaffoldLang file
- **⚡ Compile ScaffoldLang File** (`Ctrl+Shift+B`): Compile the current file
- **🔍 Tokenize ScaffoldLang File**: Show tokenization details

## Settings

- `scaffoldlang.executablePath`: Path to the ScaffoldLang executable (default: "scaffoldlang")
- `scaffoldlang.showExecutionTime`: Show execution time in output (default: true)
- `scaffoldlang.clearOutputOnRun`: Clear output panel before running (default: true)

## Supported Features

### 🔥 Core Language Features
- Object-oriented programming with classes and inheritance
- Advanced mathematical functions and operations
- Array and matrix manipulation
- Coordinate system transformations
- Euler angle operations

### ⚡ Metaprogramming
- **Macros**: Compile-time code generation with parameter substitution
- **Micros**: Zero-cost inline functions for ultra-performance
- **Import System**: Module organization and library imports

### 📊 Advanced Data Types
- Matrices with linear algebra operations
- Multi-dimensional arrays
- Coordinate systems (Cartesian, Polar, Spherical)
- Vector operations

### 🧮 Mathematical Functions
- Trigonometric functions (sin, cos, tan, etc.)
- Exponential and logarithmic functions
- Statistical functions
- Numerical methods (GCD, LCM, factorial, fibonacci)

## Example ScaffoldLang Code

```scaffold
// Object-oriented programming
class Person {
    constructor(name, age) {
        this.name = name
        this.age = age
    }
    
    greet() {
        print("Hello, I'm " + this.name)
    }
}

// Advanced math and arrays
numbers = [64, 34, 25, 12, 22, 11, 90]
print("Original array: " + numbers)

quickSort(numbers)
print("Sorted array: " + numbers)

// Matrix operations
matrix1 = [[1, 2], [3, 4]]
matrix2 = [[5, 6], [7, 8]]
result = matrixMultiply(matrix1, matrix2)
print("Matrix multiplication result: " + result)

// Micros for performance
micro square(x) { return x * x }
result = square(5)
print("5 squared = " + result)

// Macros for code generation
macro debug_var(name) { print("DEBUG: {name} = " + {name}) }
x = 42
debug_var(x)
```

## Requirements

- ScaffoldLang compiler installed and accessible in PATH
- VS Code 1.60.0 or higher

## Installation

1. Install ScaffoldLang: Follow the [installation guide](https://github.com/scaffoldlang/scaffoldlang#installation)
2. Install this extension from the VS Code marketplace
3. Start coding with ScaffoldLang!

## Performance

ScaffoldLang maintains **1.06x of C performance** while providing:
- Python-like ease of use
- Advanced metaprogramming capabilities
- Zero-cost abstractions through micros
- Compile-time optimizations through macros

---

**🔥 Happy coding with ScaffoldLang - where performance meets productivity!**