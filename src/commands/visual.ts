import { Command } from 'commander';
import { spawn } from 'child_process';
import path from 'path';
import fs from 'fs-extra';

export const visualCommand = new Command()
  .name('visual')
  .alias('v')
  .description('🎨 Launch the Visual Builder web interface')
  .option('-p, --port <port>', 'Port to run the visual builder on (default: 3000)', '3000')
  .option('--no-open', 'Don\'t automatically open the browser')
  .action(async (options) => {
    try {
      console.log('🎨 Starting Scaffold-Craft Visual Builder...\n');
      
      // Find the web-platform directory
      const currentDir = process.cwd();
      const possiblePaths = [
        path.join(currentDir, 'web-platform'),
        path.join(__dirname, '../../web-platform'),
        path.join(__dirname, '../../../web-platform')
      ];
      
      let webPlatformPath = '';
      
      for (const p of possiblePaths) {
        if (await fs.pathExists(p)) {
          webPlatformPath = p;
          break;
        }
      }
      
      if (!webPlatformPath) {
        console.log('❌ Could not find web-platform directory.');
        console.log('💡 Make sure you\'re running this command from the Scaffold-Craft project root.');
        process.exit(1);
      }
      
      console.log(`📂 Found web platform at: ${webPlatformPath}`);
      console.log(`🌐 Starting development server on port ${options.port}...\n`);
      
      // Check if dependencies are installed
      const nodeModulesPath = path.join(webPlatformPath, 'node_modules');
      if (!await fs.pathExists(nodeModulesPath)) {
        console.log('📦 Installing dependencies...');
        
        const installProcess = spawn('npm', ['install'], {
          cwd: webPlatformPath,
          stdio: 'inherit',
          shell: true
        });
        
        await new Promise((resolve, reject) => {
          installProcess.on('close', (code) => {
            if (code === 0) {
              resolve(code);
            } else {
              reject(new Error(`npm install failed with code ${code}`));
            }
          });
        });
        
        console.log('✅ Dependencies installed!\n');
      }
      
      // Start the development server
      const devProcess = spawn('npm', ['run', 'dev', '--', '--port', options.port], {
        cwd: webPlatformPath,
        stdio: 'inherit',
        shell: true
      });
      
      // Open browser after a delay if requested
      if (options.open) {
        setTimeout(() => {
          const url = `http://localhost:${options.port}/builder`;
          console.log(`\n🌐 Opening ${url} in your browser...`);
          
          const openCommand = process.platform === 'darwin' ? 'open' : 
                             process.platform === 'win32' ? 'start' : 'xdg-open';
          
          spawn(openCommand, [url], {
            stdio: 'ignore',
            detached: true
          }).unref();
        }, 3000);
      }
      
      // Handle graceful shutdown
      process.on('SIGINT', () => {
        console.log('\n\n👋 Shutting down Visual Builder...');
        devProcess.kill('SIGINT');
        process.exit(0);
      });
      
      process.on('SIGTERM', () => {
        devProcess.kill('SIGTERM');
        process.exit(0);
      });
      
      // Show helpful information
      console.log('\n🎨 Visual Builder is starting...');
      console.log(`   Local:    http://localhost:${options.port}/builder`);
      console.log('\n💡 Features available:');
      console.log('   • Drag & Drop Components');
      console.log('   • Real-time Code Export');
      console.log('   • Smart Layout Engine');
      console.log('   • Advanced Animations');
      console.log('   • Multi-framework Support');
      console.log('\n⚡ Quick Actions:');
      console.log('   • Ctrl+C to stop the server');
      console.log('   • Check the Design tab to start building');
      console.log('   • Use Templates tab for quick project generation');
      
    } catch (error) {
      console.error('\n❌ Error starting visual builder:', error);
      process.exit(1);
    }
  }); 