#!/bin/bash

# ScaffoldLang Easy Installer
# Supports: macOS, Linux, Windows (WSL)

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
PURPLE='\033[0;35m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

# ScaffoldLang ASCII Art
echo -e "${PURPLE}"
echo "  ____            __  __       _     _  _                    "
echo " / ___|  ___ __ _ / _|/ _| ___ | | __| || |    __ _ _ __   __ _ "
echo " \___ \ / __/ _\` | |_| |_ / _ \| |/ _\` || |   / _\` | '_ \ / _\` |"
echo "  ___) | (_| (_| |  _|  _| (_) | | (_| || |__| (_| | | | | (_| |"
echo " |____/ \___\__,_|_| |_|  \___/|_|\__,_||____|\__,_|_| |_|\__, |"
echo "                                                         |___/ "
echo -e "${NC}"
echo -e "${CYAN}🔥 Ultra-Performance Programming Language${NC}"
echo -e "${CYAN}Python Ease + C Speed + AI/ML Ready${NC}"
echo ""

# Detect OS
OS=""
if [[ "$OSTYPE" == "linux-gnu"* ]]; then
    OS="linux"
elif [[ "$OSTYPE" == "darwin"* ]]; then
    OS="macos"
elif [[ "$OSTYPE" == "cygwin" ]] || [[ "$OSTYPE" == "msys" ]] || [[ "$OSTYPE" == "win32" ]]; then
    OS="windows"
else
    echo -e "${RED}❌ Unsupported operating system: $OSTYPE${NC}"
    exit 1
fi

echo -e "${BLUE}🔍 Detected OS: $OS${NC}"

# Installation directory
INSTALL_DIR="$HOME/.scaffoldlang"
BIN_DIR="$INSTALL_DIR/bin"
EXAMPLES_DIR="$INSTALL_DIR/examples"

echo -e "${BLUE}📁 Installation directory: $INSTALL_DIR${NC}"

# Create directories
echo -e "${YELLOW}📂 Creating directories...${NC}"
mkdir -p "$BIN_DIR"
mkdir -p "$EXAMPLES_DIR"

# Check if we have a local package or need to build from source
if [ -f "scaffoldlang-v2.0.0-complete.tar.gz" ]; then
    echo -e "${GREEN}📦 Using local package...${NC}"
    TEMP_FILE="/tmp/scaffoldlang.tar.gz"
    cp "scaffoldlang-v2.0.0-complete.tar.gz" "$TEMP_FILE"
    
    # Extract package
    echo -e "${YELLOW}📦 Extracting ScaffoldLang...${NC}"
    cd "$INSTALL_DIR"
    tar -xzf "$TEMP_FILE"
    
    # Move files to proper locations
    if [ -f "target/release/scaffoldlang" ]; then
        mv target/release/scaffoldlang "$BIN_DIR/"
        chmod +x "$BIN_DIR/scaffoldlang"
    fi
else
    echo -e "${YELLOW}🔧 Building ScaffoldLang from source...${NC}"
    
    # Check if git is installed
    if ! command -v git &> /dev/null; then
        echo -e "${RED}❌ Git not found. Please install git first.${NC}"
        exit 1
    fi
    
    # Check if Rust is installed
    if ! command -v cargo &> /dev/null; then
        echo -e "${RED}❌ Rust/Cargo not found. Installing Rust...${NC}"
        curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y
        source ~/.cargo/env
    fi
    
    echo -e "${GREEN}✅ Rust/Cargo found${NC}"
    
    # Download the repository
    echo -e "${YELLOW}📦 Downloading ScaffoldLang source...${NC}"
    TEMP_DIR="/tmp/scaffoldlang-source"
    rm -rf "$TEMP_DIR"
    git clone https://github.com/Ca3de/scaffoldlang.git "$TEMP_DIR"
    
    if [ $? -ne 0 ]; then
        echo -e "${RED}❌ Failed to download source code${NC}"
        exit 1
    fi
    
    # Copy source files to install directory
    echo -e "${YELLOW}📁 Setting up ScaffoldLang...${NC}"
    cp -r "$TEMP_DIR/src" "$INSTALL_DIR/"
    cp "$TEMP_DIR/Cargo.toml" "$INSTALL_DIR/"
    cp -r "$TEMP_DIR/vscode-extension" "$INSTALL_DIR/"
    cp -r "$TEMP_DIR/examples" "$INSTALL_DIR/"
    cp "$TEMP_DIR"/*.md "$INSTALL_DIR/" 2>/dev/null || true
    
    # Build ScaffoldLang
    echo -e "${YELLOW}🔨 Building ScaffoldLang (this may take a few minutes)...${NC}"
    cd "$INSTALL_DIR"
    cargo build --release
    
    if [ $? -eq 0 ]; then
        echo -e "${GREEN}✅ Build successful!${NC}"
        # Move binary to bin directory
        mv target/release/scaffoldlang "$BIN_DIR/"
        chmod +x "$BIN_DIR/scaffoldlang"
    else
        echo -e "${RED}❌ Build failed!${NC}"
        exit 1
    fi
    
    # Cleanup temporary directory
    rm -rf "$TEMP_DIR"
fi

# Copy examples if they exist
if [ -d "$INSTALL_DIR/examples" ]; then
    cp -r "$INSTALL_DIR/examples"/* "$EXAMPLES_DIR/" 2>/dev/null || true
fi

# Documentation is already copied in the build process

# Add to PATH
echo -e "${YELLOW}🔧 Setting up PATH...${NC}"

# Determine shell config file
SHELL_CONFIG=""
if [[ "$SHELL" == *"zsh"* ]] && [ -f "$HOME/.zshrc" ]; then
    SHELL_CONFIG="$HOME/.zshrc"
elif [[ "$SHELL" == *"bash"* ]] && [ -f "$HOME/.bashrc" ]; then
    SHELL_CONFIG="$HOME/.bashrc"
elif [ -f "$HOME/.bash_profile" ]; then
    SHELL_CONFIG="$HOME/.bash_profile"
elif [ -f "$HOME/.profile" ]; then
    SHELL_CONFIG="$HOME/.profile"
fi

# Add ScaffoldLang to PATH
SCAFFOLD_PATH_EXPORT="export PATH=\"\$HOME/.scaffoldlang/bin:\$PATH\""

if [ -n "$SHELL_CONFIG" ]; then
    if ! grep -q "scaffoldlang" "$SHELL_CONFIG"; then
        echo "" >> "$SHELL_CONFIG"
        echo "# ScaffoldLang" >> "$SHELL_CONFIG"
        echo "$SCAFFOLD_PATH_EXPORT" >> "$SHELL_CONFIG"
        echo -e "${GREEN}✅ Added ScaffoldLang to PATH in $SHELL_CONFIG${NC}"
    else
        echo -e "${BLUE}ℹ️  ScaffoldLang already in PATH${NC}"
    fi
fi

# Install VS Code extension if available
if command -v code &> /dev/null; then
    echo -e "${YELLOW}🔌 Installing VS Code extension...${NC}"
    if [ -d "vscode-extension" ]; then
        cp -r vscode-extension "$INSTALL_DIR/"
        echo -e "${GREEN}✅ VS Code extension files copied${NC}"
        echo -e "${BLUE}💡 To install: code --install-extension $INSTALL_DIR/vscode-extension${NC}"
    else
        echo -e "${YELLOW}⚠️  VS Code extension not found${NC}"
    fi
else
    echo -e "${YELLOW}⚠️  VS Code not found - skipping extension installation${NC}"
fi

# Create quick test script
cat > "$BIN_DIR/scaffold-test" << 'EOF'
#!/bin/bash
echo "🔥 ScaffoldLang Quick Test"
echo "========================="
echo ""

# Create a simple test file
cat > /tmp/test.sl << 'SLTEST'
app QuickTest {
    fun main() -> void {
        print("🔥 ScaffoldLang is working!")
        print("Python ease + C speed + AI/ML ready")
        
        let x: int = 7
        print("Test variable: " + toString(x))
        
        print("✅ Quick test passed!")
    }
}
SLTEST

# Run the test
echo "Running test file..."
echo ""
scaffoldlang run /tmp/test.sl
EOF

chmod +x "$BIN_DIR/scaffold-test"

# Cleanup
rm -f "$TEMP_FILE" 2>/dev/null || true

# Final instructions
echo ""
echo -e "${GREEN}🎉 INSTALLATION COMPLETE!${NC}"
echo ""
echo -e "${CYAN}📋 QUICK START:${NC}"
echo -e "${YELLOW}1. Restart your terminal (or run: source $SHELL_CONFIG)${NC}"
echo -e "${YELLOW}2. Test installation: ${BLUE}scaffold-test${NC}"
echo -e "${YELLOW}3. Run ScaffoldLang: ${BLUE}scaffoldlang file.sl${NC}"
echo ""
echo -e "${CYAN}📁 EXAMPLES:${NC}"
echo -e "${YELLOW}• Examples directory: ${BLUE}$EXAMPLES_DIR${NC}"
echo -e "${YELLOW}• Documentation: ${BLUE}$INSTALL_DIR/*.md${NC}"
echo ""
echo -e "${CYAN}🔌 VS CODE INTEGRATION:${NC}"
if command -v code &> /dev/null && [ -d "$INSTALL_DIR/vscode-extension" ]; then
    echo -e "${YELLOW}• Install extension: ${BLUE}code --install-extension $INSTALL_DIR/vscode-extension${NC}"
    echo -e "${YELLOW}• Then press F5 in .scaffold files to run!${NC}"
else
    echo -e "${YELLOW}• VS Code extension available in: ${BLUE}$INSTALL_DIR/vscode-extension${NC}"
fi
echo ""
echo -e "${GREEN}Happy coding with ScaffoldLang! 🔥${NC}"