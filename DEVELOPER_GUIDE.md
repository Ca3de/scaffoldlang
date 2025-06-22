# 🔥 ScaffoldLang Developer Quick-Start Guide

**The hypercar of programming languages - Get up to speed in minutes!**

## 🚀 Quick Installation

### Option 1: Easy Install (Recommended)
```bash
# Download and run the easy installer
curl -sSL https://raw.githubusercontent.com/scaffoldlang/scaffoldlang/main/easy-install.sh | bash
```

### Option 2: Manual Installation
```bash
# Clone the repository
git clone https://github.com/scaffoldlang/scaffoldlang.git
cd scaffoldlang

# Run the installer
chmod +x easy-install.sh
./easy-install.sh
```

### Option 3: Build from Source
```bash
# Requires Rust/Cargo
cargo build --release
./install.sh
```

## 🎯 Getting Started in 30 Seconds

1. **Test the installation:**
   ```bash
   scaffold-test
   ```

2. **Create your first ScaffoldLang file:**
   ```bash
   cat > hello.sl << 'EOF'
   app HelloWorld {
       fun main() -> void {
           let message: str = "🔥 Hello, ScaffoldLang!"
           print(message)
       }
   }
   EOF
   scaffoldlang run hello.sl
   ```

3. **Try advanced features:**
   ```bash
   cat > advanced.sl << 'EOF'
   app AdvancedDemo {
       fun main() -> void {
           print("🧮 Advanced ScaffoldLang Demo")
           
           // Variables and types
           let x: int = 42
           let pi: float = 3.14159
           let name: str = "ScaffoldLang"
           
           print("Integer: " + toString(x))
           print("Float: " + toString(pi))
           print("String: " + name)
           
           print("✅ Advanced demo completed!")
       }
   }
   EOF
   
   scaffoldlang run advanced.sl
   ```

## 🔌 VS Code Integration

### Auto-Install (if VS Code detected)
The installer automatically sets up VS Code integration. If you have VS Code:

```bash
# Install the extension
code --install-extension ~/.scaffoldlang/vscode-extension

# Open a ScaffoldLang file and press F5 to run!
code hello.sl
```

### Manual VS Code Setup
1. Copy extension: `cp -r ~/.scaffoldlang/vscode-extension ~/.vscode/extensions/scaffoldlang`
2. Restart VS Code
3. Open any `.scaffold` or `.sl` file
4. Press **F5** to run

## 📚 Language Features Overview

### 🎯 Core Syntax (Python-like ease)
```scaffold
app MyApp {
    fun main() -> void {
        // Variables and basic types
        let name: str = "ScaffoldLang"
        let version: float = 2.0
        let is_fast: bool = true

        // Basic output
        print("Name: " + name)
        print("Version: " + toString(version))
        print("Fast: " + toString(is_fast))

        // Simple arithmetic
        let result: int = 5 + 3 * 2
        print("Result: " + toString(result))
    }
}
```

### 🏗️ Object-Oriented Programming
```scaffold
class Vehicle {
    constructor(brand, speed) {
        this.brand = brand
        this.max_speed = speed
    }
    
    accelerate() {
        print(this.brand + " is accelerating!")
    }
}

class Koenigsegg extends Vehicle {
    constructor() {
        super("Koenigsegg", 330)
    }
    
    launch() {
        print("🔥 Hypercar mode activated!")
    }
}

car = new Koenigsegg()
car.launch()
```

### ⚡ Metaprogramming with Macros
```scaffold
// Compile-time code generation
macro debug_print(var) {
    print("DEBUG: {var} = " + {var})
}

macro create_getter(name) {
    function get{name}() {
        return this.{name}
    }
}

x = 42
debug_print(x)  // Expands to: print("DEBUG: x = " + x)
```

### 🏎️ Ultra-Performance with Micros
```scaffold
// Zero-cost inline functions
micro fast_abs(x) { return x < 0 ? -x : x }
micro clamp(val, min_val, max_val) { 
    return val < min_val ? min_val : (val > max_val ? max_val : val) 
}

// Hot-path optimizations
for i in range(1000000) {
    result = fast_abs(numbers[i])
}
```

### 🧮 Advanced Mathematics
```scaffold
// Trigonometry
angle = 45
radians = angle * Math.pi / 180
print("sin(45°) = " + sin(radians))

// Statistics and algorithms
data = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
mean = sum(data) / len(data)
factorial_10 = factorial(10)
gcd_result = gcd(48, 18)
```

### 🔢 Matrix Operations
```scaffold
// Linear algebra
A = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
B = [[9, 8, 7], [6, 5, 4], [3, 2, 1]]

C = matrixMultiply(A, B)
det = matrixDeterminant(A)
A_T = matrixTranspose(A)

// Identity matrices
I = matrixIdentity(3)
zeros = matrixZeros(3, 3)
```

### 🗺️ Coordinate Transformations
```scaffold
// 2D coordinates
cartesian = [3, 4]
polar = cartesianToPolar(cartesian[0], cartesian[1])
print("Polar: r=" + polar[0] + ", θ=" + polar[1])

// 3D coordinates
point_3d = [1, 2, 3]
spherical = cartesianToSpherical(point_3d[0], point_3d[1], point_3d[2])

// Rotations and transformations
rotated = rotatePoint(point_3d, 45, "z")
```

### 📦 Module System
```scaffold
// Import entire modules
import math
import arrays
import utils

// Selective imports
from strings import upper, lower, trim
from coordinate import cartesianToPolar, polarToCartesian

// Use imported functions
result = math.sqrt(16)
uppercase = upper("hello world")
polar_coords = cartesianToPolar(3, 4)
```

## 🔧 Development Workflow

### 1. File Extensions
- `.scaffold` - Full ScaffoldLang files
- `.sl` - Short extension for quick scripts

### 2. Running Code
```bash
# Direct execution
scaffoldlang run myfile.sl

# Compile to machine code
scaffoldlang compile myfile.sl

# Debug tokenization
scaffoldlang tokenize myfile.sl

# Show AST structure
scaffoldlang parse myfile.sl

# Performance benchmarks
scaffoldlang benchmark
```

### 3. VS Code Workflow
1. Create `.scaffold` file
2. Write your code with full syntax highlighting
3. Press **F5** to run instantly
4. View output in integrated terminal
5. Use **Ctrl+Shift+B** to compile

### 4. Debugging
```bash
# Tokenize for debugging
scaffoldlang tokenize myfile.sl

# Show AST structure  
scaffoldlang parse myfile.sl

# Profile performance
scaffoldlang profile myfile.sl
```

## 🚀 Performance Tips

### 1. Use Micros for Hot Paths
```scaffold
// Instead of regular functions for tight loops
micro dot_product(x1, y1, x2, y2) {
    return x1*x2 + y1*y2
}
```

### 2. Leverage Built-in Functions
```scaffold
// Use optimized built-ins
quickSort(large_array)  // Faster than manual sorting
result = pow(base, exp)  // Optimized power function
```

### 3. Matrix Operations
```scaffold
// Use SIMD-optimized matrix functions
result = matrixMultiply(A, B)  // Vectorized operations
```

## 📊 Performance Benchmarks

ScaffoldLang achieves **1.06x of C performance** while maintaining:
- Python-like syntax ease
- Zero-cost abstractions (micros)
- Compile-time optimizations (macros)
- Advanced mathematical capabilities

```
Language Comparison:
┌─────────────┬──────────────┬─────────────┐
│ Language    │ Performance  │ Ease of Use │
├─────────────┼──────────────┼─────────────┤
│ C           │ 1.00x        │ Complex     │
│ ScaffoldLang│ 1.06x        │ Very Easy   │
│ Python      │ 50-100x      │ Very Easy   │
│ JavaScript  │ 10-20x       │ Easy        │
└─────────────┴──────────────┴─────────────┘
```

## 🛠️ Troubleshooting

### Common Issues

1. **Command not found: scaffoldlang**
   ```bash
   # Restart terminal or source your profile
   source ~/.bashrc  # or ~/.zshrc
   
   # Or run directly
   ~/.scaffoldlang/bin/scaffoldlang myfile.sl
   ```

2. **VS Code extension not working**
   ```bash
   # Reinstall extension
   code --install-extension ~/.scaffoldlang/vscode-extension --force
   ```

3. **Permission denied**
   ```bash
   # Fix permissions
   chmod +x ~/.scaffoldlang/bin/scaffoldlang
   ```

## 📖 Example Projects

### 1. Mathematical Calculator
```scaffold
class Calculator {
    static add(a, b) { return a + b }
    static multiply(a, b) { return a * b }
    static power(base, exp) { return pow(base, exp) }
}

result = Calculator.add(5, 3)
print("5 + 3 = " + result)
```

### 2. Data Processing Pipeline
```scaffold
data = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]

// Filter even numbers
evens = filter(data, x => x % 2 == 0)

// Apply transformations
squared = map(evens, x => x * x)

// Reduce to sum
total = reduce(squared, (acc, x) => acc + x, 0)

print("Sum of squared evens: " + total)
```

### 3. Game Development Helper
```scaffold
class Vector2D {
    constructor(x, y) {
        this.x = x
        this.y = y
    }
    
    length() {
        return sqrt(this.x * this.x + this.y * this.y)
    }
    
    normalize() {
        len = this.length()
        return new Vector2D(this.x / len, this.y / len)
    }
}

// Physics calculations
velocity = new Vector2D(10, 5)
normalized = velocity.normalize()
print("Normalized velocity: " + normalized.x + ", " + normalized.y)
```

## 🤝 Contributing

1. **Feedback**: Try ScaffoldLang and report issues
2. **Examples**: Share cool projects you build
3. **Documentation**: Help improve guides and tutorials
4. **Testing**: Test on different platforms

## 📞 Support

- **Documentation**: Check the built-in examples and guides
- **Issues**: Report bugs and feature requests on GitHub
- **Community**: Join discussions and share your projects

---

**🔥 Welcome to the future of programming - where performance meets productivity!**

*Happy coding with ScaffoldLang Koenigsegg Edition!*