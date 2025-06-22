app AdvancedDemo {
    fun main() -> void {
        print("🧮 ScaffoldLang Advanced Demo")
        print("============================")
        
        // Variables and types
        let x: int = 42
        let pi: float = 3.14159
        let name: str = "ScaffoldLang"
        let fast: bool = true
        
        print("Integer: " + toString(x))
        print("Float: " + toString(pi))
        print("String: " + name)
        print("Boolean: " + toString(fast))
        
        // Simple math
        let result: int = x + 10
        print("Math result: " + toString(result))
        
        print("✅ Demo completed successfully!")
    }
}
