#!/bin/bash

# ScaffoldLang Distribution Package Creator
# Creates a complete distribution package for manual installation

set -e

echo "🔥 Creating ScaffoldLang Distribution Package..."

# Create distribution directory
DIST_DIR="scaffoldlang-v2.0.0-distribution"
mkdir -p "$DIST_DIR"

# Copy core files
echo "📦 Copying core files..."
cp -r src "$DIST_DIR/"
cp -r target "$DIST_DIR/" 2>/dev/null || echo "Note: target directory not found, will be built during installation"
cp Cargo.toml "$DIST_DIR/"
cp Cargo.lock "$DIST_DIR/" 2>/dev/null || echo "Note: Cargo.lock not found"

# Copy installation scripts
echo "🔧 Copying installation scripts..."
cp easy-install.sh "$DIST_DIR/"
cp install.sh "$DIST_DIR/"
cp test-installation.sh "$DIST_DIR/"

# Copy VS Code extension
echo "🔌 Copying VS Code extension..."
cp -r vscode-extension "$DIST_DIR/"

# Copy documentation
echo "📚 Copying documentation..."
cp README.md "$DIST_DIR/" 2>/dev/null || echo "Note: README.md not found"
cp DEVELOPER_GUIDE.md "$DIST_DIR/"
cp MICROS_MACROS_GUIDE.md "$DIST_DIR/"

# Copy examples
echo "📁 Copying examples..."
mkdir -p "$DIST_DIR/examples"
cp *.sl "$DIST_DIR/examples/" 2>/dev/null || echo "Note: No .sl files found"
cp examples/*.sl "$DIST_DIR/examples/" 2>/dev/null || echo "Note: examples directory not found"

# Create a comprehensive installation README
cat > "$DIST_DIR/INSTALLATION.md" << 'EOF'
# ScaffoldLang Distribution Package

## 🚀 Quick Installation

### Option 1: Easy Install (Recommended)
```bash
chmod +x easy-install.sh
./easy-install.sh
```

### Option 2: Manual Build
```bash
# Requires Rust/Cargo
chmod +x install.sh
./install.sh
```

### Option 3: GitHub Auto-Install
```bash
curl -sSL https://raw.githubusercontent.com/Ca3de/scaffoldlang/main/easy-install.sh | bash
```

## ✅ Test Installation
```bash
chmod +x test-installation.sh
./test-installation.sh
```

## 🔌 VS Code Integration
```bash
# After installation
code --install-extension ~/.scaffoldlang/vscode-extension
```

## 📚 Documentation
- `DEVELOPER_GUIDE.md` - Complete developer guide
- `MICROS_MACROS_GUIDE.md` - Advanced metaprogramming features
- `examples/` - Example ScaffoldLang programs

## 🆘 Support
- Test your installation with `test-installation.sh`
- Check documentation for troubleshooting
- Report issues on GitHub

🔥 Happy coding with ScaffoldLang!
EOF

# Create package info file
cat > "$DIST_DIR/PACKAGE_INFO.txt" << EOF
ScaffoldLang Distribution Package v2.0.0
=======================================

Package Contents:
- ScaffoldLang source code (src/)
- Pre-built binary (target/release/scaffoldlang) [if available]
- Easy installation script (easy-install.sh)
- Manual installation script (install.sh)
- Installation test script (test-installation.sh)
- VS Code extension (vscode-extension/)
- Documentation (*.md files)
- Example programs (examples/)

Installation Options:
1. Easy Install: ./easy-install.sh
2. Manual Build: ./install.sh
3. GitHub Install: curl -sSL https://raw.githubusercontent.com/Ca3de/scaffoldlang/main/easy-install.sh | bash

Test Installation:
./test-installation.sh

Package created on: $(date)
Platform: $(uname -s) $(uname -m)
EOF

# Make scripts executable
chmod +x "$DIST_DIR"/*.sh

# Create tarball
echo "📦 Creating distribution archive..."
tar -czf "${DIST_DIR}.tar.gz" "$DIST_DIR"

echo ""
echo "✅ Distribution package created successfully!"
echo "📦 Package: ${DIST_DIR}.tar.gz"
echo "📁 Directory: ${DIST_DIR}/"
echo ""
echo "🚀 To test the package:"
echo "  tar -xzf ${DIST_DIR}.tar.gz"
echo "  cd ${DIST_DIR}"
echo "  ./easy-install.sh"
echo "  ./test-installation.sh"
echo ""
echo "🔥 Ready for distribution!"