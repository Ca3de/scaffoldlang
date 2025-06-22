#!/usr/bin/env node

import { Command } from 'commander';
import { generateCommand } from './commands/generate.js';
import { visualCommand } from './commands/visual.js';
import { createCommand } from './commands/create.js';

const program = new Command();

program
  .version('2.0.0')
  .description('🚀 Scaffold-Craft - Smart Project Generation & Visual Builder')
  .addCommand(createCommand)
  .addCommand(generateCommand)
  .addCommand(visualCommand);

// Show help and available commands if no command is provided
if (process.argv.length <= 2) {
  console.log('🚀 Welcome to Scaffold-Craft v2.0\n');
  console.log('Available commands:');
  console.log('  create        - Create a new project (simplified syntax)');
  console.log('  generate (g)  - Generate a new project with smart configuration');
  console.log('  visual (v)    - Launch the Visual Builder web interface\n');
  console.log('Examples:');
  console.log('  scaffold-craft create react-app my-app');
  console.log('  scaffold-craft generate my-app --type react');
  console.log('  scaffold-craft visual --port 3001\n');
  console.log('💡 Most popular: "scaffold-craft create react-app" or "scaffold-craft visual"');
  console.log('Run "scaffold-craft <command> --help" for more information about a command.');
} else {
  program.parse(process.argv);
} 