import { ProjectConfiguration } from '../config/ConfigurationSchema.js';
// import { TemplateStrategy } from '../strategies/TemplateStrategy.js';
// import { TemplateFactory } from '../factories/TemplateFactory.js';
import fs from 'fs-extra';
import path from 'path';

export interface TemplateModification {
  filePath: string;
  operation: 'create' | 'modify' | 'delete';
  content?: string;
  insertions?: Array<{
    after: string;
    content: string;
  }>;
}

export class ConfigurableTemplateEngine {
  
  static async generateProject(config: ProjectConfiguration, outputPath: string): Promise<void> {
    console.log(`\n🎯 Generating ${config.projectType} project: ${config.projectName}`);
    
    // Step 1: Create base directory and structure
    await fs.ensureDir(outputPath);
    
    // Step 2: Generate base template based on project type
    await this.generateBaseTemplate(config, outputPath);
    
    // Step 3: Apply configuration-based modifications
    const modifications = this.generateModifications(config, outputPath);
    
    // Step 4: Execute modifications
    await this.applyModifications(modifications, outputPath);
    
    // Step 5: Generate configuration files
    await this.generateConfigurationFiles(config, outputPath);
    
    // Step 6: Generate package.json with dependencies
    await this.generatePackageJson(config, outputPath);
    
    console.log(`\n✅ Project generated successfully at: ${outputPath}`);
    console.log(`\n📖 Next steps:`);
    console.log(`   cd ${config.projectName}`);
    console.log(`   npm install`);
    
    if (config.database?.type && config.database.type !== 'none') {
      console.log(`   npm run db:setup`);
    }
    
    console.log(`   npm run dev`);
  }
  
  private static async generateBaseTemplate(config: ProjectConfiguration, outputPath: string): Promise<void> {
    // Create basic project structure
    const directories = ['src', 'public'];
    
    if (config.projectType === 'react' || config.projectType === 'vue' || config.projectType === 'svelte') {
      directories.push('src/components', 'src/pages');
    }
    
    if (config.projectType === 'nextjs') {
      directories.push('app', 'components', 'lib');
    }
    
    if (config.projectType === 'express' || config.projectType === 'nodejs') {
      directories.push('src/routes', 'src/middleware', 'src/controllers');
    }
    
    for (const dir of directories) {
      await fs.ensureDir(path.join(outputPath, dir));
    }
    
    // Generate basic files based on project type
    await this.generateBaseFiles(config, outputPath);
  }
  
  private static async generateBaseFiles(config: ProjectConfiguration, outputPath: string): Promise<void> {
    const projectType = config.projectType;
    
    // Generate TypeScript config
    await fs.writeFile(
      path.join(outputPath, 'tsconfig.json'),
      this.generateTsConfig(projectType)
    );
    
    // Generate README
    await fs.writeFile(
      path.join(outputPath, 'README.md'),
      this.generateReadme(config)
    );
    
    // Generate gitignore
    await fs.writeFile(
      path.join(outputPath, '.gitignore'),
      this.generateGitignore()
    );
    
    // Generate main application files based on project type
    switch (projectType) {
      case 'react':
        await this.generateReactFiles(config, outputPath);
        break;
      case 'vue':
        await this.generateVueFiles(config, outputPath);
        break;
      case 'nextjs':
        await this.generateNextJSFiles(config, outputPath);
        break;
      case 'express':
        await this.generateExpressFiles(config, outputPath);
        break;
      case 'svelte':
        await this.generateSvelteFiles(config, outputPath);
        break;
      case 'angular':
        await this.generateAngularFiles(config, outputPath);
        break;
      default:
        await this.generateNodeJSFiles(config, outputPath);
    }
  }
  
  private static async generateReactFiles(config: ProjectConfiguration, outputPath: string): Promise<void> {
    // App.tsx
    await fs.writeFile(
      path.join(outputPath, 'src/App.tsx'),
      `import React from 'react';
${config.uiFramework?.styling === 'tailwind' ? "import './index.css';" : ''}

function App() {
  return (
    <div className="${config.uiFramework?.styling === 'tailwind' ? 'min-h-screen bg-gradient-to-br from-purple-600 via-blue-600 to-teal-600 flex items-center justify-center' : 'App'}">
      <div className="${config.uiFramework?.styling === 'tailwind' ? 'text-center text-white' : ''}">
        <h1 className="${config.uiFramework?.styling === 'tailwind' ? 'text-4xl font-bold mb-4' : ''}">${config.projectName}</h1>
        <p className="${config.uiFramework?.styling === 'tailwind' ? 'text-xl opacity-90' : ''}">Welcome to your new React application!</p>
        <div className="${config.uiFramework?.styling === 'tailwind' ? 'mt-8 space-x-4' : ''}">
          <button className="${config.uiFramework?.styling === 'tailwind' ? 'bg-white text-purple-600 px-6 py-3 rounded-lg font-semibold hover:bg-gray-100 transition-colors' : 'btn'}">
            Get Started
          </button>
        </div>
      </div>
    </div>
  );
}

export default App;`
    );
    
    // main.tsx
    await fs.writeFile(
      path.join(outputPath, 'src/main.tsx'),
      `import React from 'react'
import ReactDOM from 'react-dom/client'
import App from './App.tsx'
${config.uiFramework?.styling === 'tailwind' ? "import './index.css'" : ''}

ReactDOM.createRoot(document.getElementById('root')!).render(
  <React.StrictMode>
    <App />
  </React.StrictMode>,
)`
    );
    
    // index.html
    await fs.writeFile(
      path.join(outputPath, 'index.html'),
      `<!DOCTYPE html>
<html lang="en">
  <head>
    <meta charset="UTF-8" />
    <link rel="icon" type="image/svg+xml" href="/vite.svg" />
    <meta name="viewport" content="width=device-width, initial-scale=1.0" />
    <title>${config.projectName}</title>
  </head>
  <body>
    <div id="root"></div>
    <script type="module" src="/src/main.tsx"></script>
  </body>
</html>`
    );
  }
  
  private static async generateVueFiles(config: ProjectConfiguration, outputPath: string): Promise<void> {
    // App.vue
    await fs.writeFile(
      path.join(outputPath, 'src/App.vue'),
      `<template>
  <div class="${config.uiFramework?.styling === 'tailwind' ? 'min-h-screen bg-gradient-to-br from-purple-600 via-blue-600 to-teal-600 flex items-center justify-center' : 'app'}">
    <div class="${config.uiFramework?.styling === 'tailwind' ? 'text-center text-white' : ''}">
      <h1 class="${config.uiFramework?.styling === 'tailwind' ? 'text-4xl font-bold mb-4' : ''}">${config.projectName}</h1>
      <p class="${config.uiFramework?.styling === 'tailwind' ? 'text-xl opacity-90' : ''}">Welcome to your new Vue application!</p>
      <div class="${config.uiFramework?.styling === 'tailwind' ? 'mt-8 space-x-4' : ''}">
        <button class="${config.uiFramework?.styling === 'tailwind' ? 'bg-white text-purple-600 px-6 py-3 rounded-lg font-semibold hover:bg-gray-100 transition-colors' : 'btn'}">
          Get Started
        </button>
      </div>
    </div>
  </div>
</template>

<script setup lang="ts">
// Your Vue 3 Composition API code here
</script>

<style scoped>
/* Component styles */
</style>`
    );
    
    // main.ts
    await fs.writeFile(
      path.join(outputPath, 'src/main.ts'),
      `import { createApp } from 'vue'
${config.uiFramework?.styling === 'tailwind' ? "import './style.css'" : ''}
import App from './App.vue'

createApp(App).mount('#app')`
    );
  }
  
  private static async generateNextJSFiles(config: ProjectConfiguration, outputPath: string): Promise<void> {
    // app/page.tsx
    await fs.ensureDir(path.join(outputPath, 'app'));
    await fs.writeFile(
      path.join(outputPath, 'app/page.tsx'),
      `export default function Home() {
  return (
    <main className="${config.uiFramework?.styling === 'tailwind' ? 'min-h-screen bg-gradient-to-br from-purple-600 via-blue-600 to-teal-600 flex items-center justify-center' : ''}">
      <div className="${config.uiFramework?.styling === 'tailwind' ? 'text-center text-white' : ''}">
        <h1 className="${config.uiFramework?.styling === 'tailwind' ? 'text-4xl font-bold mb-4' : ''}">${config.projectName}</h1>
        <p className="${config.uiFramework?.styling === 'tailwind' ? 'text-xl opacity-90' : ''}">Welcome to your new Next.js application!</p>
        <div className="${config.uiFramework?.styling === 'tailwind' ? 'mt-8 space-x-4' : ''}">
          <button className="${config.uiFramework?.styling === 'tailwind' ? 'bg-white text-purple-600 px-6 py-3 rounded-lg font-semibold hover:bg-gray-100 transition-colors' : 'btn'}">
            Get Started
          </button>
        </div>
      </div>
    </main>
  )
}`
    );
    
    // app/layout.tsx
    await fs.writeFile(
      path.join(outputPath, 'app/layout.tsx'),
      `import type { Metadata } from 'next'
import { Inter } from 'next/font/google'
${config.uiFramework?.styling === 'tailwind' ? "import './globals.css'" : ''}

const inter = Inter({ subsets: ['latin'] })

export const metadata: Metadata = {
  title: '${config.projectName}',
  description: 'Generated by Scaffold-Craft',
}

export default function RootLayout({
  children,
}: {
  children: React.ReactNode
}) {
  return (
    <html lang="en">
      <body className={inter.className}>{children}</body>
    </html>
  )
}`
    );
  }
  
  private static async generateExpressFiles(config: ProjectConfiguration, outputPath: string): Promise<void> {
    // src/index.ts
    await fs.writeFile(
      path.join(outputPath, 'src/index.ts'),
      `import express from 'express';
import cors from 'cors';
import helmet from 'helmet';
import dotenv from 'dotenv';

dotenv.config();

const app = express();
const PORT = process.env.PORT || 3000;

// Middleware
app.use(helmet());
app.use(cors());
app.use(express.json());
app.use(express.urlencoded({ extended: true }));

// Routes
app.get('/', (req, res) => {
  res.json({
    message: 'Welcome to ${config.projectName} API!',
    version: '1.0.0',
    timestamp: new Date().toISOString()
  });
});

app.get('/health', (req, res) => {
  res.json({ status: 'OK', uptime: process.uptime() });
});

// Error handling
app.use((err: any, req: any, res: any, next: any) => {
  console.error(err.stack);
  res.status(500).json({ error: 'Something went wrong!' });
});

app.listen(PORT, () => {
  console.log(\`🚀 Server running on port \${PORT}\`);
});`
    );
  }
  
  private static async generateSvelteFiles(config: ProjectConfiguration, outputPath: string): Promise<void> {
    // src/App.svelte
    await fs.writeFile(
      path.join(outputPath, 'src/App.svelte'),
      `<script lang="ts">
  // Your Svelte TypeScript code here
</script>

<main class="${config.uiFramework?.styling === 'tailwind' ? 'min-h-screen bg-gradient-to-br from-purple-600 via-blue-600 to-teal-600 flex items-center justify-center' : ''}">
  <div class="${config.uiFramework?.styling === 'tailwind' ? 'text-center text-white' : ''}">
    <h1 class="${config.uiFramework?.styling === 'tailwind' ? 'text-4xl font-bold mb-4' : ''}">${config.projectName}</h1>
    <p class="${config.uiFramework?.styling === 'tailwind' ? 'text-xl opacity-90' : ''}">Welcome to your new Svelte application!</p>
    <div class="${config.uiFramework?.styling === 'tailwind' ? 'mt-8 space-x-4' : ''}">
      <button class="${config.uiFramework?.styling === 'tailwind' ? 'bg-white text-purple-600 px-6 py-3 rounded-lg font-semibold hover:bg-gray-100 transition-colors' : 'btn'}">
        Get Started
      </button>
    </div>
  </div>
</main>

<style>
  /* Component styles */
</style>`
    );
    
    // src/main.ts
    await fs.writeFile(
      path.join(outputPath, 'src/main.ts'),
      `import App from './App.svelte'

const app = new App({
  target: document.getElementById('app')!,
})

export default app`
    );
  }
  
  private static async generateAngularFiles(config: ProjectConfiguration, outputPath: string): Promise<void> {
    // src/main.ts
    await fs.writeFile(
      path.join(outputPath, 'src/main.ts'),
      `import { bootstrapApplication } from '@angular/platform-browser';
import { AppComponent } from './app/app.component';

bootstrapApplication(AppComponent).catch(err => console.error(err));`
    );
    
    // src/app/app.component.ts
    await fs.ensureDir(path.join(outputPath, 'src/app'));
    await fs.writeFile(
      path.join(outputPath, 'src/app/app.component.ts'),
      `import { Component } from '@angular/core';

@Component({
  selector: 'app-root',
  standalone: true,
  template: \`
    <div class="${config.uiFramework?.styling === 'tailwind' ? 'min-h-screen bg-gradient-to-br from-purple-600 via-blue-600 to-teal-600 flex items-center justify-center' : ''}">
      <div class="${config.uiFramework?.styling === 'tailwind' ? 'text-center text-white' : ''}">
        <h1 class="${config.uiFramework?.styling === 'tailwind' ? 'text-4xl font-bold mb-4' : ''}">${config.projectName}</h1>
        <p class="${config.uiFramework?.styling === 'tailwind' ? 'text-xl opacity-90' : ''}">Welcome to your new Angular application!</p>
        <div class="${config.uiFramework?.styling === 'tailwind' ? 'mt-8 space-x-4' : ''}">
          <button class="${config.uiFramework?.styling === 'tailwind' ? 'bg-white text-purple-600 px-6 py-3 rounded-lg font-semibold hover:bg-gray-100 transition-colors' : 'btn'}">
            Get Started
          </button>
        </div>
      </div>
    </div>
  \`
})
export class AppComponent {
  title = '${config.projectName}';
}`
    );
  }
  
  private static async generateNodeJSFiles(config: ProjectConfiguration, outputPath: string): Promise<void> {
    // src/index.ts
    await fs.writeFile(
      path.join(outputPath, 'src/index.ts'),
      `console.log('🚀 Welcome to ${config.projectName}!');

// Your Node.js application code here
export default class App {
  constructor() {
    this.init();
  }
  
  private init() {
    console.log('Initializing ${config.projectName}...');
  }
  
  public start() {
    console.log('Starting ${config.projectName}...');
  }
}

// Start the application
const app = new App();
app.start();`
    );
  }
  
  private static generateModifications(config: ProjectConfiguration, outputPath: string): TemplateModification[] {
    const modifications: TemplateModification[] = [];
    
    // Add CSS files for Tailwind
    if (config.uiFramework?.styling === 'tailwind') {
      modifications.push({
        filePath: config.projectType === 'nextjs' ? 'app/globals.css' : 
                 config.projectType === 'vue' ? 'src/style.css' : 'src/index.css',
        operation: 'create',
        content: this.generateTailwindCSS()
      });
      
      modifications.push({
        filePath: 'tailwind.config.js',
        operation: 'create',
        content: this.generateTailwindConfig()
      });
    }
    
    return modifications;
  }
  
  private static generateTailwindCSS(): string {
    return `@tailwind base;
@tailwind components;
@tailwind utilities;

@layer base {
  body {
    @apply font-sans antialiased;
  }
}

@layer components {
  .btn {
    @apply px-4 py-2 rounded-lg font-medium transition-colors;
  }
  
  .btn-primary {
    @apply bg-blue-600 text-white hover:bg-blue-700;
  }
}`;
  }
  
  private static generateTailwindConfig(): string {
    return `/** @type {import('tailwindcss').Config} */
module.exports = {
  content: [
    './pages/**/*.{js,ts,jsx,tsx,mdx}',
    './components/**/*.{js,ts,jsx,tsx,mdx}',
    './app/**/*.{js,ts,jsx,tsx,mdx}',
    './src/**/*.{js,ts,jsx,tsx,mdx}',
    './index.html',
  ],
  theme: {
    extend: {
      fontFamily: {
        sans: ['Inter', 'system-ui', 'sans-serif'],
      },
    },
  },
  plugins: [],
}`;
  }
  
  private static async applyModifications(modifications: TemplateModification[], outputPath: string): Promise<void> {
    for (const mod of modifications) {
      const fullPath = path.join(outputPath, mod.filePath);
      
      switch (mod.operation) {
        case 'create':
          await fs.ensureDir(path.dirname(fullPath));
          await fs.writeFile(fullPath, mod.content || '');
          break;
          
        case 'modify':
          if (mod.insertions && await fs.pathExists(fullPath)) {
            let content = await fs.readFile(fullPath, 'utf-8');
            
            for (const insertion of mod.insertions) {
              if (insertion.after === '') {
                content = insertion.content + content;
              } else {
                content = content.replace(insertion.after, insertion.after + insertion.content);
              }
            }
            
            await fs.writeFile(fullPath, content);
          }
          break;
          
        case 'delete':
          if (await fs.pathExists(fullPath)) {
            await fs.remove(fullPath);
          }
          break;
      }
    }
  }
  
  private static async generateConfigurationFiles(config: ProjectConfiguration, outputPath: string): Promise<void> {
    // Environment variables template
    const envTemplate = this.generateEnvironmentTemplate(config);
    await fs.writeFile(path.join(outputPath, '.env.example'), envTemplate);
    
    // Configuration summary
    const configSummary = {
      projectName: config.projectName,
      projectType: config.projectType,
      generatedAt: new Date().toISOString(),
      configuration: config
    };
    
    await fs.writeFile(
      path.join(outputPath, 'scaffold-config.json'), 
      JSON.stringify(configSummary, null, 2)
    );
  }
  
  private static async generatePackageJson(config: ProjectConfiguration, outputPath: string): Promise<void> {
    const packageJson = {
      name: config.projectName,
      version: '1.0.0',
      private: true,
      type: 'module',
      scripts: this.getScriptsForProjectType(config),
      dependencies: this.getDependenciesForProjectType(config),
      devDependencies: this.getDevDependenciesForProjectType(config)
    };
    
    await fs.writeJson(path.join(outputPath, 'package.json'), packageJson, { spaces: 2 });
  }
  
  private static getScriptsForProjectType(config: ProjectConfiguration): Record<string, string> {
    const scripts: Record<string, string> = {};
    
    switch (config.projectType) {
      case 'react':
      case 'vue':
      case 'svelte':
        scripts.dev = 'vite';
        scripts.build = 'vite build';
        scripts.preview = 'vite preview';
        break;
      case 'nextjs':
        scripts.dev = 'next dev';
        scripts.build = 'next build';
        scripts.start = 'next start';
        scripts.lint = 'next lint';
        break;
      case 'express':
      case 'nodejs':
        scripts.dev = 'tsx watch src/index.ts';
        scripts.build = 'tsc';
        scripts.start = 'node dist/index.js';
        break;
      case 'angular':
        scripts.ng = 'ng';
        scripts.start = 'ng serve';
        scripts.build = 'ng build';
        scripts.watch = 'ng build --watch --configuration development';
        scripts.test = 'ng test';
        break;
    }
    
    if (config.features?.linting?.eslint) {
      scripts.lint = 'eslint . --ext .ts,.tsx';
      scripts['lint:fix'] = 'eslint . --ext .ts,.tsx --fix';
    }
    
    if (config.features?.testing?.unit) {
      scripts.test = config.features.testing.unit;
      scripts['test:watch'] = `${config.features.testing.unit} --watch`;
    }
    
    return scripts;
  }
  
  private static getDependenciesForProjectType(config: ProjectConfiguration): Record<string, string> {
    const deps: Record<string, string> = {};
    
    switch (config.projectType) {
      case 'react':
        deps.react = '^18.2.0';
        deps['react-dom'] = '^18.2.0';
        break;
      case 'vue':
        deps.vue = '^3.3.0';
        break;
      case 'nextjs':
        deps.next = '^14.0.0';
        deps.react = '^18.2.0';
        deps['react-dom'] = '^18.2.0';
        break;
      case 'express':
        deps.express = '^4.18.0';
        deps.cors = '^2.8.5';
        deps.helmet = '^7.0.0';
        deps.dotenv = '^16.0.0';
        break;
      case 'svelte':
        deps.svelte = '^4.0.0';
        break;
      case 'angular':
        deps['@angular/animations'] = '^17.0.0';
        deps['@angular/common'] = '^17.0.0';
        deps['@angular/compiler'] = '^17.0.0';
        deps['@angular/core'] = '^17.0.0';
        deps['@angular/forms'] = '^17.0.0';
        deps['@angular/platform-browser'] = '^17.0.0';
        deps['@angular/platform-browser-dynamic'] = '^17.0.0';
        deps['@angular/router'] = '^17.0.0';
        deps.rxjs = '~7.8.0';
        deps.tslib = '^2.3.0';
        deps['zone.js'] = '~0.14.0';
        break;
    }
    
    // Add UI framework dependencies
    if (config.uiFramework?.styling === 'tailwind') {
      deps.tailwindcss = '^3.3.0';
    }
    
    return deps;
  }
  
  private static getDevDependenciesForProjectType(config: ProjectConfiguration): Record<string, string> {
    const devDeps: Record<string, string> = {
      typescript: '^5.0.0',
      '@types/node': '^20.0.0'
    };
    
    switch (config.projectType) {
      case 'react':
      case 'vue':
      case 'svelte':
        devDeps.vite = '^5.0.0';
        devDeps['@vitejs/plugin-react'] = '^4.0.0';
        break;
      case 'nextjs':
        devDeps['@types/react'] = '^18.2.0';
        devDeps['@types/react-dom'] = '^18.2.0';
        break;
      case 'express':
      case 'nodejs':
        devDeps.tsx = '^4.0.0';
        devDeps['@types/express'] = '^4.17.0';
        devDeps['@types/cors'] = '^2.8.0';
        break;
      case 'angular':
        devDeps['@angular-devkit/build-angular'] = '^17.0.0';
        devDeps['@angular/cli'] = '~17.0.0';
        devDeps['@angular/compiler-cli'] = '^17.0.0';
        break;
    }
    
    if (config.uiFramework?.styling === 'tailwind') {
      devDeps.tailwindcss = '^3.3.0';
      devDeps.autoprefixer = '^10.4.0';
      devDeps.postcss = '^8.4.0';
    }
    
    return devDeps;
  }
  
  private static generateTsConfig(projectType: string): string {
    const baseConfig = {
      compilerOptions: {
        target: 'ES2020',
        lib: ['ES2020', 'DOM', 'DOM.Iterable'],
        module: 'ESNext',
        skipLibCheck: true,
        moduleResolution: 'bundler',
        allowImportingTsExtensions: true,
        resolveJsonModule: true,
        isolatedModules: true,
        noEmit: true,
        jsx: projectType === 'react' || projectType === 'nextjs' ? 'react-jsx' : undefined,
        strict: true,
        noUnusedLocals: true,
        noUnusedParameters: true,
        noFallthroughCasesInSwitch: true
      },
      include: ['src'],
      references: [{ path: './tsconfig.node.json' }]
    };
    
    return JSON.stringify(baseConfig, null, 2);
  }
  
  private static generateReadme(config: ProjectConfiguration): string {
    return `# ${config.projectName}

This project was generated with [Scaffold-Craft](https://github.com/your-username/scaffold-craft) v2.0.

## Project Configuration

- **Type**: ${config.projectType}
- **Database**: ${config.database?.type || 'None'}
- **Authentication**: ${config.authentication?.provider || 'None'}
- **UI Framework**: ${config.uiFramework?.styling || 'None'}
- **Deployment**: ${config.deployment?.platform || 'None'}

## Getting Started

1. Install dependencies:
   \`\`\`bash
   npm install
   \`\`\`

2. Start the development server:
   \`\`\`bash
   npm run dev
   \`\`\`

## Available Scripts

- \`npm run dev\` - Start development server
- \`npm run build\` - Build for production
- \`npm run start\` - Start production server${config.features?.testing ? '\n- `npm run test` - Run tests' : ''}${config.features?.linting ? '\n- `npm run lint` - Lint code' : ''}

## Environment Variables

Copy \`.env.example\` to \`.env\` and fill in your environment variables.

## Learn More

To learn more about the technologies used in this project:

${config.projectType === 'react' ? '- [React Documentation](https://reactjs.org/)' : ''}
${config.projectType === 'vue' ? '- [Vue.js Documentation](https://vuejs.org/)' : ''}
${config.projectType === 'nextjs' ? '- [Next.js Documentation](https://nextjs.org/docs)' : ''}
${config.projectType === 'angular' ? '- [Angular Documentation](https://angular.io/)' : ''}
${config.projectType === 'svelte' ? '- [Svelte Documentation](https://svelte.dev/)' : ''}
${config.uiFramework?.styling === 'tailwind' ? '- [Tailwind CSS Documentation](https://tailwindcss.com/docs)' : ''}

---

Generated with ❤️ by Scaffold-Craft`;
  }
  
  private static generateGitignore(): string {
    return `# Dependencies
node_modules/
/.pnp
.pnp.js

# Production
/build
/dist
/.next/
/out/

# Runtime data
pids
*.pid
*.seed
*.pid.lock

# Environment variables
.env
.env.local
.env.development.local
.env.test.local
.env.production.local

# Logs
npm-debug.log*
yarn-debug.log*
yarn-error.log*
lerna-debug.log*

# IDE
.vscode/
.idea/
*.swp
*.swo

# OS
.DS_Store
.DS_Store?
._*
.Spotlight-V100
.Trashes
ehthumbs.db
Thumbs.db

# Temporary folders
tmp/
temp/

# Coverage directory used by tools like istanbul
coverage/
*.lcov

# nyc test coverage
.nyc_output

# Database
*.db
*.sqlite`;
  }
  
  private static generateEnvironmentTemplate(config: ProjectConfiguration): string {
    let template = `# ${config.projectName} Environment Variables\n\n`;
    
    if (config.database) {
      switch (config.database.type) {
        case 'postgresql':
          template += '# Database\nDATABASE_URL="postgresql://username:password@localhost:5432/mydb"\n\n';
          break;
        case 'mongodb':
          template += '# Database\nMONGODB_URI="mongodb://localhost:27017/mydb"\n\n';
          break;
        case 'supabase':
          template += '# Supabase\nNEXT_PUBLIC_SUPABASE_URL="your-supabase-url"\nNEXT_PUBLIC_SUPABASE_ANON_KEY="your-anon-key"\n\n';
          break;
      }
    }
    
    if (config.authentication?.provider === 'jwt') {
      template += '# Authentication\nJWT_SECRET="your-super-secret-jwt-key"\n\n';
    }
    
    if (config.authentication?.provider === 'nextauth') {
      template += '# NextAuth\nNEXTAUTH_SECRET="your-nextauth-secret"\nNEXTAUTH_URL="http://localhost:3000"\n\n';
    }
    
    return template;
  }
} 