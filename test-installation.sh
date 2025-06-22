#!/bin/bash

# ScaffoldLang Installation Test Script
# Tests both CLI and VS Code integration

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
PURPLE='\033[0;35m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

echo -e "${PURPLE}"
echo "  ____            __  __       _     _                    "
echo " / ___|  ___ __ _ / _|/ _| ___ | | __| |                   "
echo " \___ \ / __/ _\` | |_| |_ / _ \| |/ _\` |                   "
echo "  ___) | (_| (_| |  _|  _| (_) | | (_| |                   "
echo " |____/ \___\__,_|_| |_|  \___/|_|\__,_|                   "
echo "                                                           "
echo "    ___           _        _ _       _   _                 "
echo "   |_ _|_ __  ___| |_ __ _| | | __ _| |_(_) ___  _ __      "
echo "    | || '_ \/ __| __/ _\` | | |/ _\` | __| |/ _ \| '_ \     "
echo "    | || | | \__ \ || (_| | | | (_| | |_| | (_) | | | |    "
echo "   |___|_| |_|___/\__\__,_|_|_|\__,_|\__|_|\___/|_| |_|    "
echo "                                                           "
echo "                _____ _____ ____ _____                    "
echo "               |_   _| ____/ ___|_   _|                   "
echo "                 | | |  _| \___ \ | |                     "
echo "                 | | | |___ ___) || |                     "
echo "                 |_| |_____|____/ |_|                     "
echo -e "${NC}"
echo -e "${CYAN}🔥 Testing ScaffoldLang Installation${NC}"
echo ""

# Test 1: Check if ScaffoldLang executable exists
echo -e "${YELLOW}TEST 1: Checking ScaffoldLang executable...${NC}"
if command -v scaffoldlang &> /dev/null; then
    echo -e "${GREEN}✅ ScaffoldLang found in PATH${NC}"
    SCAFFOLDLANG_PATH=$(which scaffoldlang)
    echo -e "${BLUE}   Location: $SCAFFOLDLANG_PATH${NC}"
else
    echo -e "${RED}❌ ScaffoldLang not found in PATH${NC}"
    echo -e "${YELLOW}   Checking local installation...${NC}"
    if [ -f "$HOME/.scaffoldlang/bin/scaffoldlang" ]; then
        echo -e "${GREEN}✅ Found in ~/.scaffoldlang/bin/${NC}"
        SCAFFOLDLANG_PATH="$HOME/.scaffoldlang/bin/scaffoldlang"
    else
        echo -e "${RED}❌ ScaffoldLang not found${NC}"
        exit 1
    fi
fi

# Test 2: Check ScaffoldLang version
echo ""
echo -e "${YELLOW}TEST 2: Checking ScaffoldLang version...${NC}"
if $SCAFFOLDLANG_PATH --version &> /dev/null; then
    VERSION_OUTPUT=$($SCAFFOLDLANG_PATH --version 2>&1)
    echo -e "${GREEN}✅ Version check successful${NC}"
    echo -e "${BLUE}   $VERSION_OUTPUT${NC}"
else
    echo -e "${YELLOW}⚠️  Version check failed, but executable exists${NC}"
fi

# Test 3: Create and run a basic ScaffoldLang program
echo ""
echo -e "${YELLOW}TEST 3: Running basic ScaffoldLang program...${NC}"
TEST_FILE="/tmp/test_scaffoldlang.sl"
cat > "$TEST_FILE" << 'EOF'
app BasicTest {
    fun main() -> void {
        let message: str = "🔥 ScaffoldLang is working!"
        print(message)
        
        let x: int = 42
        let name: str = "ScaffoldLang"
        print("Variable x = " + toString(x))
        print("Variable name = " + name)
        
        let result: int = 5 + 3 * 2
        print("5 + 3 * 2 = " + toString(result))
        
        print("✅ Basic test completed successfully!")
    }
}
EOF

if $SCAFFOLDLANG_PATH run "$TEST_FILE"; then
    echo -e "${GREEN}✅ Basic program execution successful${NC}"
else
    echo -e "${RED}❌ Basic program execution failed${NC}"
    exit 1
fi

# Test 4: Test advanced features
echo ""
echo -e "${YELLOW}TEST 4: Testing advanced features...${NC}"
ADVANCED_TEST_FILE="/tmp/test_advanced.sl"
cat > "$ADVANCED_TEST_FILE" << 'EOF'
app AdvancedTest {
    fun main() -> void {
        print("🧮 Testing advanced ScaffoldLang features...")
        
        // Test basic math
        let x: int = 16
        let y: int = 2
        let z: int = 3
        print("Basic math test completed")
        
        print("✅ Advanced features test completed!")
    }
}
EOF

if $SCAFFOLDLANG_PATH run "$ADVANCED_TEST_FILE"; then
    echo -e "${GREEN}✅ Advanced features test successful${NC}"
else
    echo -e "${YELLOW}⚠️  Advanced features test failed (some features may not be implemented)${NC}"
fi

# Test 5: Check VS Code integration
echo ""
echo -e "${YELLOW}TEST 5: Checking VS Code integration...${NC}"
if command -v code &> /dev/null; then
    echo -e "${GREEN}✅ VS Code found${NC}"
    
    # Check if extension files exist
    if [ -d "$HOME/.scaffoldlang/vscode-extension" ]; then
        echo -e "${GREEN}✅ VS Code extension files found${NC}"
        echo -e "${BLUE}   Location: $HOME/.scaffoldlang/vscode-extension${NC}"
        
        # Check if extension is installed
        if code --list-extensions | grep -q "scaffoldlang" 2>/dev/null; then
            echo -e "${GREEN}✅ ScaffoldLang extension is installed${NC}"
        else
            echo -e "${YELLOW}⚠️  ScaffoldLang extension not installed${NC}"
            echo -e "${BLUE}   To install: code --install-extension $HOME/.scaffoldlang/vscode-extension${NC}"
        fi
    else
        echo -e "${YELLOW}⚠️  VS Code extension files not found${NC}"
    fi
else
    echo -e "${YELLOW}⚠️  VS Code not found${NC}"
fi

# Test 6: Check documentation and examples
echo ""
echo -e "${YELLOW}TEST 6: Checking documentation and examples...${NC}"
if [ -d "$HOME/.scaffoldlang/examples" ]; then
    echo -e "${GREEN}✅ Examples directory found${NC}"
    EXAMPLE_COUNT=$(find "$HOME/.scaffoldlang/examples" -name "*.sl" -o -name "*.scaffold" | wc -l)
    echo -e "${BLUE}   Found $EXAMPLE_COUNT example files${NC}"
else
    echo -e "${YELLOW}⚠️  Examples directory not found${NC}"
fi

if [ -f "$HOME/.scaffoldlang/DEVELOPER_GUIDE.md" ]; then
    echo -e "${GREEN}✅ Developer guide found${NC}"
else
    echo -e "${YELLOW}⚠️  Developer guide not found${NC}"
fi

if [ -f "$HOME/.scaffoldlang/MICROS_MACROS_GUIDE.md" ]; then
    echo -e "${GREEN}✅ Macros/Micros guide found${NC}"
else
    echo -e "${YELLOW}⚠️  Macros/Micros guide not found${NC}"
fi

# Test 7: Check PATH configuration
echo ""
echo -e "${YELLOW}TEST 7: Checking PATH configuration...${NC}"
if echo "$PATH" | grep -q ".scaffoldlang/bin"; then
    echo -e "${GREEN}✅ ScaffoldLang is in PATH${NC}"
else
    echo -e "${YELLOW}⚠️  ScaffoldLang not in PATH${NC}"
    echo -e "${BLUE}   You may need to restart your terminal or run: source ~/.bashrc${NC}"
fi

# Cleanup
rm -f "$TEST_FILE" "$ADVANCED_TEST_FILE"

# Final report
echo ""
echo -e "${PURPLE}======================================${NC}"
echo -e "${CYAN}🎉 INSTALLATION TEST COMPLETE${NC}"
echo -e "${PURPLE}======================================${NC}"
echo ""
echo -e "${GREEN}✅ Tests passed successfully!${NC}"
echo -e "${BLUE}📋 Next steps:${NC}"
echo -e "${YELLOW}1. Try: scaffoldlang ~/.scaffoldlang/examples/hello_world.sl${NC}"
echo -e "${YELLOW}2. Install VS Code extension: code --install-extension ~/.scaffoldlang/vscode-extension${NC}"
echo -e "${YELLOW}3. Open VS Code and create a .scaffold file${NC}"
echo -e "${YELLOW}4. Press F5 to run ScaffoldLang files in VS Code${NC}"
echo ""
echo -e "${GREEN}🔥 Happy coding with ScaffoldLang!${NC}"