#!/bin/bash
# 🚀 ScaffoldLang Installation Script
# Install ScaffoldLang OOP Programming Language

set -e

echo "🚀 ScaffoldLang Installation Script"
echo "==================================="
echo ""

# Check if Rust is installed
if ! command -v cargo &> /dev/null; then
    echo "❌ Rust/Cargo not found. Please install Rust first:"
    echo "   curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh"
    exit 1
fi

echo "✅ Rust/Cargo found"

# Build ScaffoldLang
echo "🔧 Building ScaffoldLang..."
cargo build --release

if [ $? -eq 0 ]; then
    echo "✅ Build successful!"
else
    echo "❌ Build failed!"
    exit 1
fi

# Create symlink for global access (optional)
echo ""
read -p "🔗 Create global symlink? (y/N): " -n 1 -r
echo ""

if [[ $REPLY =~ ^[Yy]$ ]]; then
    INSTALL_DIR="/usr/local/bin"
    
    if [ -w "$INSTALL_DIR" ]; then
        ln -sf "$(pwd)/target/release/scaffoldlang" "$INSTALL_DIR/scaffoldlang"
        echo "✅ Global symlink created: $INSTALL_DIR/scaffoldlang"
    else
        echo "🔐 Creating symlink with sudo..."
        sudo ln -sf "$(pwd)/target/release/scaffoldlang" "$INSTALL_DIR/scaffoldlang"
        echo "✅ Global symlink created: $INSTALL_DIR/scaffoldlang"
    fi
fi

# Test installation
echo ""
echo "🧪 Testing installation..."
./target/release/scaffoldlang examples/hello_world.sl

echo ""
echo "🎉 ScaffoldLang installed successfully!"
echo ""
echo "📚 Getting Started:"
echo "   • Run: ./target/release/scaffoldlang examples/hello_world.sl"
echo "   • Try: ./target/release/scaffoldlang examples/todo_list.sl"
echo "   • Docs: cat README.md"
echo ""
echo "🚀 Happy coding with ScaffoldLang!"
