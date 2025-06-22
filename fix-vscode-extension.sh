#!/bin/bash

# Fix VS Code Extension Installation
echo "🔧 Creating working VS Code extension..."

EXTENSION_DIR="$HOME/.scaffoldlang/vscode-extension-simple"
rm -rf "$EXTENSION_DIR"
mkdir -p "$EXTENSION_DIR"

# Create a simple, working package.json
cat > "$EXTENSION_DIR/package.json" << 'EOF'
{
  "name": "scaffoldlang-simple",
  "displayName": "ScaffoldLang Simple",
  "description": "ScaffoldLang syntax highlighting and execution",
  "version": "1.0.0",
  "publisher": "scaffoldlang",
  "engines": {
    "vscode": "^1.60.0"
  },
  "categories": ["Programming Languages"],
  "contributes": {
    "languages": [{
      "id": "scaffoldlang",
      "aliases": ["ScaffoldLang", "scaffold"],
      "extensions": [".scaffold", ".sl"],
      "configuration": "./language-configuration.json"
    }],
    "grammars": [{
      "language": "scaffoldlang",
      "scopeName": "source.scaffold",
      "path": "./syntaxes/scaffoldlang.tmLanguage.json"
    }],
    "commands": [
      {
        "command": "scaffoldlang.run",
        "title": "🔥 Run ScaffoldLang File",
        "category": "ScaffoldLang"
      }
    ],
    "keybindings": [
      {
        "command": "scaffoldlang.run",
        "key": "F5",
        "when": "editorTextFocus && resourceExtname == '.sl'"
      }
    ],
    "menus": {
      "editor/context": [
        {
          "when": "resourceExtname == '.sl'",
          "command": "scaffoldlang.run",
          "group": "1_run@1"
        }
      ]
    }
  },
  "activationEvents": [
    "onLanguage:scaffoldlang"
  ],
  "main": "./extension.js"
}
EOF

# Create simple extension.js (no TypeScript needed)
cat > "$EXTENSION_DIR/extension.js" << 'EOF'
const vscode = require('vscode');
const { spawn } = require('child_process');

function activate(context) {
    console.log('ScaffoldLang extension activated');

    let disposable = vscode.commands.registerCommand('scaffoldlang.run', function () {
        const editor = vscode.window.activeTextEditor;
        if (!editor) {
            vscode.window.showErrorMessage('No active ScaffoldLang file');
            return;
        }

        const document = editor.document;
        const filePath = document.fileName;

        if (!filePath.endsWith('.sl') && !filePath.endsWith('.scaffold')) {
            vscode.window.showErrorMessage('Current file is not a ScaffoldLang file');
            return;
        }

        // Save file if needed
        if (document.isDirty) {
            document.save();
        }

        // Create output channel
        const outputChannel = vscode.window.createOutputChannel('ScaffoldLang');
        outputChannel.clear();
        outputChannel.show();
        outputChannel.appendLine('🔥 Running ScaffoldLang file: ' + filePath);
        outputChannel.appendLine('─'.repeat(50));

        // Run ScaffoldLang
        const child = spawn('scaffoldlang', ['run', filePath], {
            cwd: require('path').dirname(filePath)
        });

        child.stdout.on('data', (data) => {
            outputChannel.append(data.toString());
        });

        child.stderr.on('data', (data) => {
            outputChannel.append('Error: ' + data.toString());
        });

        child.on('close', (code) => {
            outputChannel.appendLine('─'.repeat(50));
            if (code === 0) {
                outputChannel.appendLine('✅ Execution completed successfully');
            } else {
                outputChannel.appendLine('❌ Execution failed with code: ' + code);
            }
        });

        child.on('error', (error) => {
            outputChannel.appendLine('❌ Failed to run ScaffoldLang: ' + error.message);
            vscode.window.showErrorMessage('Failed to run ScaffoldLang. Make sure it is installed and in PATH.');
        });
    });

    context.subscriptions.push(disposable);
}

function deactivate() {}

module.exports = {
    activate,
    deactivate
};
EOF

# Create language configuration
cat > "$EXTENSION_DIR/language-configuration.json" << 'EOF'
{
    "comments": {
        "lineComment": "//",
        "blockComment": ["/*", "*/"]
    },
    "brackets": [
        ["{", "}"],
        ["[", "]"],
        ["(", ")"]
    ],
    "autoClosingPairs": [
        ["{", "}"],
        ["[", "]"],
        ["(", ")"],
        ["\"", "\""],
        ["'", "'"]
    ]
}
EOF

# Create syntax highlighting
mkdir -p "$EXTENSION_DIR/syntaxes"
cat > "$EXTENSION_DIR/syntaxes/scaffoldlang.tmLanguage.json" << 'EOF'
{
    "$schema": "https://raw.githubusercontent.com/martinring/tmlanguage/master/tmlanguage.json",
    "name": "ScaffoldLang",
    "scopeName": "source.scaffold",
    "patterns": [
        {
            "name": "comment.line.double-slash.scaffold",
            "match": "//.*$"
        },
        {
            "name": "keyword.control.scaffold",
            "match": "\\b(app|fun|let|if|else|while|for|return|print)\\b"
        },
        {
            "name": "storage.type.scaffold",
            "match": "\\b(void|str|int|float|bool)\\b"
        },
        {
            "name": "string.quoted.double.scaffold",
            "begin": "\"",
            "end": "\"",
            "patterns": [
                {
                    "name": "constant.character.escape.scaffold",
                    "match": "\\\\."
                }
            ]
        },
        {
            "name": "constant.numeric.scaffold",
            "match": "\\b\\d+(\\.\\d+)?\\b"
        }
    ]
}
EOF

echo "✅ Simple VS Code extension created at: $EXTENSION_DIR"
echo ""
echo "🔌 Installing VS Code extension..."

# Install the extension
cd "$EXTENSION_DIR"
code --install-extension . --force

echo ""
echo "✅ VS Code extension installed!"
echo "📝 Now create a test file:"
echo "   echo 'app Test { fun main() -> void { print(\"Hello!\") } }' > test.sl"
echo "   code test.sl"
echo "   # Press F5 to run!"