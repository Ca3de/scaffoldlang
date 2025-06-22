import * as vscode from 'vscode';
import * as path from 'path';
import { spawn, ChildProcess } from 'child_process';

let outputChannel: vscode.OutputChannel;

export function activate(context: vscode.ExtensionContext) {
    console.log('🔥 ScaffoldLang extension activated!');
    
    // Create output channel
    outputChannel = vscode.window.createOutputChannel('ScaffoldLang');
    
    // Register commands
    let runCommand = vscode.commands.registerCommand('scaffoldlang.run', runScaffoldLangFile);
    let compileCommand = vscode.commands.registerCommand('scaffoldlang.compile', compileScaffoldLangFile);
    let tokenizeCommand = vscode.commands.registerCommand('scaffoldlang.tokenize', tokenizeScaffoldLangFile);
    
    context.subscriptions.push(runCommand, compileCommand, tokenizeCommand, outputChannel);
    
    // Show welcome message
    vscode.window.showInformationMessage(
        '🔥 ScaffoldLang Koenigsegg Edition is ready! Press F5 to run .scaffold files.',
        'View Documentation'
    ).then(selection => {
        if (selection === 'View Documentation') {
            vscode.env.openExternal(vscode.Uri.parse('https://github.com/scaffoldlang/scaffoldlang'));
        }
    });
}

export function deactivate() {
    if (outputChannel) {
        outputChannel.dispose();
    }
}

async function runScaffoldLangFile() {
    const editor = vscode.window.activeTextEditor;
    if (!editor) {
        vscode.window.showErrorMessage('No active ScaffoldLang file found');
        return;
    }
    
    const document = editor.document;
    const filePath = document.fileName;
    
    // Check if it's a ScaffoldLang file
    if (!filePath.endsWith('.scaffold') && !filePath.endsWith('.sl')) {
        vscode.window.showErrorMessage('Current file is not a ScaffoldLang file (.scaffold or .sl)');
        return;
    }
    
    // Save the file if it has unsaved changes
    if (document.isDirty) {
        await document.save();
    }
    
    const config = vscode.workspace.getConfiguration('scaffoldlang');
    const executablePath = config.get<string>('executablePath', 'scaffoldlang');
    const showExecutionTime = config.get<boolean>('showExecutionTime', true);
    const clearOutput = config.get<boolean>('clearOutputOnRun', true);
    
    if (clearOutput) {
        outputChannel.clear();
    }
    
    outputChannel.show(true);
    outputChannel.appendLine(`🔥 Running ScaffoldLang file: ${path.basename(filePath)}`);
    outputChannel.appendLine(`📁 Path: ${filePath}`);
    outputChannel.appendLine('─'.repeat(60));
    
    const startTime = Date.now();
    
    try {
        const child = spawn(executablePath, ['run', filePath], {
            cwd: path.dirname(filePath),
            shell: true
        });
        
        child.stdout.on('data', (data) => {
            outputChannel.append(data.toString());
        });
        
        child.stderr.on('data', (data) => {
            outputChannel.append(`❌ Error: ${data.toString()}`);
        });
        
        child.on('close', (code) => {
            const endTime = Date.now();
            const executionTime = endTime - startTime;
            
            outputChannel.appendLine('─'.repeat(60));
            
            if (code === 0) {
                outputChannel.appendLine(`✅ ScaffoldLang execution completed successfully`);
            } else {
                outputChannel.appendLine(`❌ ScaffoldLang execution failed with exit code: ${code}`);
            }
            
            if (showExecutionTime) {
                outputChannel.appendLine(`⏱️  Execution time: ${executionTime}ms`);
            }
            
            outputChannel.appendLine('🔥 ScaffoldLang run finished');
        });
        
        child.on('error', (error) => {
            outputChannel.appendLine(`❌ Failed to start ScaffoldLang: ${error.message}`);
            outputChannel.appendLine('💡 Make sure ScaffoldLang is installed and in your PATH');
            outputChannel.appendLine('💡 Or set the correct path in settings: scaffoldlang.executablePath');
            
            vscode.window.showErrorMessage(
                'Failed to run ScaffoldLang. Check the output panel for details.',
                'Open Settings',
                'Install ScaffoldLang'
            ).then(selection => {
                if (selection === 'Open Settings') {
                    vscode.commands.executeCommand('workbench.action.openSettings', 'scaffoldlang');
                } else if (selection === 'Install ScaffoldLang') {
                    vscode.env.openExternal(vscode.Uri.parse('https://github.com/scaffoldlang/scaffoldlang#installation'));
                }
            });
        });
        
    } catch (error) {
        outputChannel.appendLine(`❌ Unexpected error: ${error}`);
        vscode.window.showErrorMessage('An unexpected error occurred while running ScaffoldLang');
    }
}

async function compileScaffoldLangFile() {
    const editor = vscode.window.activeTextEditor;
    if (!editor) {
        vscode.window.showErrorMessage('No active ScaffoldLang file found');
        return;
    }
    
    const document = editor.document;
    const filePath = document.fileName;
    
    if (!filePath.endsWith('.scaffold') && !filePath.endsWith('.sl')) {
        vscode.window.showErrorMessage('Current file is not a ScaffoldLang file (.scaffold or .sl)');
        return;
    }
    
    if (document.isDirty) {
        await document.save();
    }
    
    const config = vscode.workspace.getConfiguration('scaffoldlang');
    const executablePath = config.get<string>('executablePath', 'scaffoldlang');
    
    outputChannel.clear();
    outputChannel.show(true);
    outputChannel.appendLine(`⚡ Compiling ScaffoldLang file: ${path.basename(filePath)}`);
    outputChannel.appendLine('─'.repeat(60));
    
    try {
        const child = spawn(executablePath, ['compile', filePath], {
            cwd: path.dirname(filePath),
            shell: true
        });
        
        child.stdout.on('data', (data) => {
            outputChannel.append(data.toString());
        });
        
        child.stderr.on('data', (data) => {
            outputChannel.append(`❌ Compile Error: ${data.toString()}`);
        });
        
        child.on('close', (code) => {
            outputChannel.appendLine('─'.repeat(60));
            if (code === 0) {
                outputChannel.appendLine(`✅ Compilation completed successfully`);
                vscode.window.showInformationMessage('🔥 ScaffoldLang file compiled successfully!');
            } else {
                outputChannel.appendLine(`❌ Compilation failed with exit code: ${code}`);
                vscode.window.showErrorMessage('❌ ScaffoldLang compilation failed. Check output for details.');
            }
        });
        
    } catch (error) {
        outputChannel.appendLine(`❌ Compilation error: ${error}`);
        vscode.window.showErrorMessage('Failed to compile ScaffoldLang file');
    }
}

async function tokenizeScaffoldLangFile() {
    const editor = vscode.window.activeTextEditor;
    if (!editor) {
        vscode.window.showErrorMessage('No active ScaffoldLang file found');
        return;
    }
    
    const document = editor.document;
    const filePath = document.fileName;
    
    if (!filePath.endsWith('.scaffold') && !filePath.endsWith('.sl')) {
        vscode.window.showErrorMessage('Current file is not a ScaffoldLang file (.scaffold or .sl)');
        return;
    }
    
    const config = vscode.workspace.getConfiguration('scaffoldlang');
    const executablePath = config.get<string>('executablePath', 'scaffoldlang');
    
    outputChannel.clear();
    outputChannel.show(true);
    outputChannel.appendLine(`🔍 Tokenizing ScaffoldLang file: ${path.basename(filePath)}`);
    outputChannel.appendLine('─'.repeat(60));
    
    try {
        const child = spawn(executablePath, ['tokenize', filePath], {
            cwd: path.dirname(filePath),
            shell: true
        });
        
        child.stdout.on('data', (data) => {
            outputChannel.append(data.toString());
        });
        
        child.stderr.on('data', (data) => {
            outputChannel.append(`❌ Tokenization Error: ${data.toString()}`);
        });
        
        child.on('close', (code) => {
            outputChannel.appendLine('─'.repeat(60));
            if (code === 0) {
                outputChannel.appendLine(`✅ Tokenization completed successfully`);
            } else {
                outputChannel.appendLine(`❌ Tokenization failed with exit code: ${code}`);
            }
        });
        
    } catch (error) {
        outputChannel.appendLine(`❌ Tokenization error: ${error}`);
        vscode.window.showErrorMessage('Failed to tokenize ScaffoldLang file');
    }
}