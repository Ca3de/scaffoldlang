import { Command } from 'commander';
import { generateCommand } from './generate.js';

export const createCommand = new Command()
  .name('create')
  .description('🎯 Create a new project (simplified syntax)')
  .argument('<template>', 'Template to use (react-app, nextjs-app, vue-app, angular-app, svelte-app, express-api)')
  .argument('[project-name]', 'Name of the project to create')
  .option('--typescript', 'Use TypeScript (default for most templates)')
  .option('--javascript', 'Use JavaScript instead of TypeScript')
  .option('--with-design', 'Include visual design components from builder')
  .action(async (template: string, projectName: string, options) => {
    try {
      // Map template aliases to our internal types
      const templateMapping: Record<string, string> = {
        'react-app': 'react',
        'nextjs-app': 'nextjs',
        'vue-app': 'vue',
        'angular-app': 'angular',
        'svelte-app': 'svelte',
        'express-api': 'express',
        'node-api': 'nodejs'
      };
      
      const projectType = templateMapping[template] || template;
      
      if (!templateMapping[template] && !['react', 'vue', 'angular', 'svelte', 'nextjs', 'express', 'nodejs'].includes(template)) {
        console.log('❌ Unknown template:', template);
        console.log('\nAvailable templates:');
        console.log('  react-app     - React with Vite and TypeScript');
        console.log('  nextjs-app    - Next.js with App Router');
        console.log('  vue-app       - Vue 3 with Composition API');
        console.log('  angular-app   - Angular with CLI and TypeScript');
        console.log('  svelte-app    - SvelteKit with TypeScript');
        console.log('  express-api   - Express.js API with TypeScript');
        console.log('\nExample: npx scaffold-craft create react-app my-awesome-app');
        process.exit(1);
      }
      
      // Generate a project name if not provided
      if (!projectName) {
        projectName = `my-${template.replace('-app', '').replace('-api', '')}-project`;
      }
      
      console.log(`🚀 Creating ${template} project: ${projectName}`);
      
      if (options.withDesign) {
        console.log('🎨 This project will include your visual design components');
        console.log('💡 Use "scaffold-craft visual" to design components first');
      }
      
      // Call the generate command with simplified options
      const generateOptions = {
        type: projectType,
        quick: true,
        ...(options.javascript && { typescript: false })
      };
      
      // Execute the generate command programmatically
      await generateCommand.parseAsync([
        'node', 'scaffold-craft', 
        projectName, 
        '--type', projectType, 
        '--quick'
      ], { from: 'user' });
      
    } catch (error) {
      console.error('❌ Error creating project:', error);
      process.exit(1);
    }
  }); 