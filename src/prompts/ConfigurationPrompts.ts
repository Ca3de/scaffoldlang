import inquirer from 'inquirer';
import { ProjectConfiguration, ProjectType, ConfigurationPresets, ConfigurationValidator } from '../config/ConfigurationSchema.js';

export class ConfigurationPrompts {
  
  static async getProjectConfiguration(): Promise<ProjectConfiguration> {
    console.log('\n🚀 Welcome to Scaffold-Craft Smart Configuration!\n');
    
    // Step 1: Basic project info
    const basicConfig = await this.getBasicConfiguration();
    
    // Step 2: Choose configuration approach
    const approach = await this.getConfigurationApproach();
    
    let config: ProjectConfiguration;
    
    if (approach === 'preset') {
      config = await this.getPresetConfiguration(basicConfig);
    } else if (approach === 'guided') {
      config = await this.getGuidedConfiguration(basicConfig);
    } else {
      config = await this.getAdvancedConfiguration(basicConfig);
    }
    
    // Step 3: Validate and confirm
    const validation = ConfigurationValidator.validate(config);
    if (!validation.valid) {
      console.log('\n❌ Configuration issues found:');
      validation.errors.forEach(error => console.log(`   • ${error}`));
      
      const shouldFix = await inquirer.prompt([{
        type: 'confirm',
        name: 'fix',
        message: 'Would you like to fix these issues?',
        default: true
      }]);
      
      if (shouldFix.fix) {
        return this.getProjectConfiguration(); // Restart
      }
    }
    
    // Final confirmation
    await this.showConfigurationSummary(config);
    
    return config;
  }
  
  private static async getBasicConfiguration() {
    return inquirer.prompt([
      {
        type: 'input',
        name: 'projectName',
        message: '📝 What is your project name?',
        validate: (input: string) => {
          if (!input || input.length < 1) return 'Project name is required';
          if (!/^[a-z0-9-]+$/.test(input)) return 'Use lowercase letters, numbers, and hyphens only';
          return true;
        },
        filter: (input: string) => input.toLowerCase().replace(/\s+/g, '-')
      },
      {
        type: 'list',
        name: 'projectType',
        message: '🎯 What type of project are you building?',
        choices: [
          { name: '⚛️  React App (SPA)', value: 'react' },
          { name: '🟢 Vue.js App (SPA)', value: 'vue' },
          { name: '🅰️  Angular App (Enterprise)', value: 'angular' },
          { name: '🔥 Svelte App (Modern & Fast)', value: 'svelte' },
          { name: '▲  Next.js App (Full-Stack React)', value: 'nextjs' },
          { name: '🚂 Express.js API (Backend)', value: 'express' },
          { name: '📦 Node.js App (Backend)', value: 'nodejs' },
          new inquirer.Separator(),
          { name: '🔗 Full-Stack: React + Express', value: 'fullstack-react-express' },
          { name: '🔗 Full-Stack: Next.js + Prisma', value: 'fullstack-nextjs-prisma' },
          { name: '🏗️  Microservices Architecture', value: 'microservices' }
        ]
      }
    ]);
  }
  
  private static async getConfigurationApproach() {
    const { approach } = await inquirer.prompt([{
      type: 'list',
      name: 'approach',
      message: '⚡ How would you like to configure your project?',
      choices: [
        { name: '🎨 Quick Start (Recommended presets)', value: 'preset' },
        { name: '🧭 Guided Setup (Step-by-step choices)', value: 'guided' },
        { name: '🔧 Advanced (Full customization)', value: 'advanced' }
      ]
    }]);
    
    return approach;
  }
  
  private static async getPresetConfiguration(basic: any): Promise<ProjectConfiguration> {
    const { preset } = await inquirer.prompt([{
      type: 'list',
      name: 'preset',
      message: '🎨 Choose a preset configuration:',
      choices: [
        { 
          name: '🚀 Startup MVP (Supabase + Tailwind + Vercel)', 
          value: 'startup-mvp',
          short: 'Perfect for quick MVPs and prototypes'
        },
        { 
          name: '🏢 Enterprise App (PostgreSQL + Auth0 + Testing)', 
          value: 'enterprise-app',
          short: 'Production-ready with enterprise features'
        },
        { 
          name: '👤 Personal Project (SQLite + JWT + Simple)', 
          value: 'personal-project',
          short: 'Lightweight setup for personal projects'
        }
      ]
    }]);
    
    // Create mutable configuration objects
    const presetConfigs: Record<string, Partial<ProjectConfiguration>> = {
      'startup-mvp': {
        database: { type: 'supabase', orm: 'prisma' },
        authentication: { provider: 'supabase', features: { registration: true, socialLogin: ['google'] } },
        uiFramework: { styling: 'tailwind', components: { library: 'headless-ui', icons: 'heroicons' } },
        deployment: { platform: 'vercel', cicd: { provider: 'github-actions' } }
      },
      'enterprise-app': {
        database: { type: 'postgresql', orm: 'prisma', seedData: true },
        authentication: { provider: 'auth0', features: { emailVerification: true, twoFactor: true } },
        uiFramework: { styling: 'material-ui', components: { library: 'none' } },
        features: { 
          testing: { unit: 'jest', e2e: 'playwright', coverage: true },
          linting: { eslint: true, prettier: true, husky: true },
          monitoring: { errorTracking: 'sentry', performance: 'web-vitals' }
        }
      },
      'personal-project': {
        database: { type: 'sqlite', orm: 'prisma' },
        authentication: { provider: 'jwt' },
        uiFramework: { styling: 'tailwind' },
        deployment: { platform: 'netlify' }
      }
    };
    
    return {
      projectName: basic.projectName,
      projectType: basic.projectType,
      ...presetConfigs[preset]
    };
  }
  
  private static async getGuidedConfiguration(basic: any): Promise<ProjectConfiguration> {
    console.log('\n🧭 Let\'s configure your project step by step...\n');
    
    const config: ProjectConfiguration = {
      projectName: basic.projectName,
      projectType: basic.projectType
    };
    
    // Database configuration
    if (this.needsDatabase(basic.projectType)) {
      config.database = await this.getDatabaseConfiguration();
    }
    
    // Authentication configuration
    if (this.needsAuth(basic.projectType)) {
      config.authentication = await this.getAuthConfiguration();
    }
    
    // UI Framework (for frontend projects)
    if (this.needsUIFramework(basic.projectType)) {
      config.uiFramework = await this.getUIFrameworkConfiguration();
    }
    
    // State Management (for frontend projects)
    if (this.needsStateManagement(basic.projectType)) {
      config.stateManagement = await this.getStateManagementConfiguration(basic.projectType);
    }
    
    // Essential features
    config.features = await this.getEssentialFeatures();
    
    // Deployment
    config.deployment = await this.getDeploymentConfiguration();
    
    return config;
  }
  
  private static async getAdvancedConfiguration(basic: any): Promise<ProjectConfiguration> {
    console.log('\n🔧 Advanced configuration - Full control over every option...\n');
    
    // This would include all possible options with detailed choices
    // For now, falling back to guided configuration
    return this.getGuidedConfiguration(basic);
  }
  
  private static async getDatabaseConfiguration() {
    const dbAnswers = await inquirer.prompt([
      {
        type: 'list',
        name: 'type',
        message: '🗃️  Choose your database:',
        choices: [
          { name: '🟦 PostgreSQL (Robust, ACID compliant)', value: 'postgresql' },
          { name: '🍃 MongoDB (Document-based, flexible)', value: 'mongodb' },
          { name: '🪶 SQLite (Lightweight, file-based)', value: 'sqlite' },
          { name: '⚡ Supabase (PostgreSQL + Auth + API)', value: 'supabase' },
          { name: '🌍 PlanetScale (MySQL, serverless)', value: 'planetscale' },
          { name: '❌ No Database', value: 'none' }
        ]
      }
    ]);
    
    if (dbAnswers.type === 'none') {
      return { type: 'none' };
    }
    
    const ormChoices = this.getCompatibleORMs(dbAnswers.type);
    const ormAnswers = await inquirer.prompt([
      {
        type: 'list',
        name: 'orm',
        message: '🔧 Choose your ORM/Database toolkit:',
        choices: ormChoices
      },
      {
        type: 'confirm',
        name: 'seedData',
        message: '🌱 Include sample seed data?',
        default: true
      }
    ]);
    
    return { ...dbAnswers, ...ormAnswers };
  }
  
  private static async getAuthConfiguration() {
    const authAnswers = await inquirer.prompt([
      {
        type: 'list',
        name: 'provider',
        message: '🔐 Choose authentication method:',
        choices: [
          { name: '🎯 JWT (Simple, self-managed)', value: 'jwt' },
          { name: '🔒 Auth0 (Enterprise-grade)', value: 'auth0' },
          { name: '🔥 Firebase Auth (Google ecosystem)', value: 'firebase' },
          { name: '🎭 Clerk (Developer-friendly)', value: 'clerk' },
          { name: '🔑 NextAuth.js (Next.js optimized)', value: 'nextauth' },
          { name: '⚡ Supabase Auth (Full-stack)', value: 'supabase' },
          { name: '❌ No Authentication', value: 'none' }
        ]
      }
    ]);
    
    if (authAnswers.provider === 'none') {
      return { provider: 'none' };
    }
    
    const features = await inquirer.prompt([
      {
        type: 'checkbox',
        name: 'socialLogin',
        message: '📱 Social login providers:',
        choices: [
          { name: 'Google', value: 'google' },
          { name: 'GitHub', value: 'github' },
          { name: 'Twitter', value: 'twitter' },
          { name: 'Facebook', value: 'facebook' }
        ]
      },
      {
        type: 'confirm',
        name: 'emailVerification',
        message: '📧 Include email verification?',
        default: true
      }
    ]);
    
    return { ...authAnswers, features };
  }
  
  private static async getUIFrameworkConfiguration() {
    return inquirer.prompt([
      {
        type: 'list',
        name: 'styling',
        message: '🎨 Choose your styling framework:',
        choices: [
          { name: '🎨 Tailwind CSS (Utility-first)', value: 'tailwind' },
          { name: '📘 Material-UI (Google Design)', value: 'material-ui' },
          { name: '🎯 Chakra UI (Simple & modular)', value: 'chakra-ui' },
          { name: '🐜 Ant Design (Enterprise)', value: 'ant-design' },
          { name: '💅 Styled Components (CSS-in-JS)', value: 'styled-components' },
          { name: '🅱️  Bootstrap (Classic)', value: 'bootstrap' }
        ]
      }
    ]);
  }
  
  private static async getStateManagementConfiguration(projectType: ProjectType) {
    const choices = this.getCompatibleStateManagement(projectType);
    
    return inquirer.prompt([
      {
        type: 'list',
        name: 'type',
        message: '🗃️  Choose state management:',
        choices
      }
    ]);
  }
  
  private static async getEssentialFeatures() {
    return inquirer.prompt([
      {
        type: 'confirm',
        name: 'testing',
        message: '🧪 Include testing setup?',
        default: true
      },
      {
        type: 'confirm',
        name: 'linting',
        message: '🔍 Include linting (ESLint + Prettier)?',
        default: true
      },
      {
        type: 'confirm',
        name: 'docker',
        message: '🐳 Include Docker configuration?',
        default: false
      }
    ]);
  }
  
  private static async getDeploymentConfiguration() {
    return inquirer.prompt([
      {
        type: 'list',
        name: 'platform',
        message: '🚀 Preferred deployment platform:',
        choices: [
          { name: '▲ Vercel (Best for Next.js/React)', value: 'vercel' },
          { name: '🎯 Netlify (Great for static sites)', value: 'netlify' },
          { name: '🚂 Railway (Full-stack friendly)', value: 'railway' },
          { name: '🎪 Render (Simple & reliable)', value: 'render' },
          { name: '☁️  AWS (Enterprise scale)', value: 'aws' },
          { name: '🌊 DigitalOcean (Developer-friendly)', value: 'digitalocean' },
          { name: '❌ No deployment config', value: 'none' }
        ]
      }
    ]);
  }
  
  private static async showConfigurationSummary(config: ProjectConfiguration) {
    console.log('\n📋 Configuration Summary:');
    console.log(`   Project: ${config.projectName}`);
    console.log(`   Type: ${config.projectType}`);
    
    if (config.database) {
      console.log(`   Database: ${config.database.type}${config.database.orm ? ` with ${config.database.orm}` : ''}`);
    }
    
    if (config.authentication) {
      console.log(`   Auth: ${config.authentication.provider}`);
    }
    
    if (config.uiFramework) {
      console.log(`   UI: ${config.uiFramework.styling}`);
    }
    
    if (config.deployment) {
      console.log(`   Deploy: ${config.deployment.platform}`);
    }
    
    const { confirm } = await inquirer.prompt([{
      type: 'confirm',
      name: 'confirm',
      message: '\n✅ Generate project with this configuration?',
      default: true
    }]);
    
    if (!confirm) {
      throw new Error('Configuration cancelled by user');
    }
  }
  
  // Helper methods
  private static needsDatabase(projectType: ProjectType): boolean {
    return !['react', 'vue', 'angular', 'svelte'].includes(projectType);
  }
  
  private static needsAuth(projectType: ProjectType): boolean {
    return projectType !== 'nodejs';
  }
  
  private static needsUIFramework(projectType: ProjectType): boolean {
    return ['react', 'vue', 'angular', 'svelte', 'nextjs'].includes(projectType);
  }
  
  private static needsStateManagement(projectType: ProjectType): boolean {
    return ['react', 'vue', 'angular', 'svelte', 'nextjs'].includes(projectType);
  }
  
  private static getCompatibleORMs(dbType: string) {
    const ormMap: Record<string, Array<{name: string, value: string}>> = {
      postgresql: [
        { name: '🔷 Prisma (Type-safe, modern)', value: 'prisma' },
        { name: '🏗️  TypeORM (Decorator-based)', value: 'typeorm' },
        { name: '🌊 Drizzle (Lightweight)', value: 'drizzle' }
      ],
      mongodb: [
        { name: '🍃 Mongoose (MongoDB native)', value: 'mongoose' },
        { name: '🔷 Prisma (Type-safe)', value: 'prisma' }
      ],
      sqlite: [
        { name: '🔷 Prisma (Recommended)', value: 'prisma' },
        { name: '🌊 Drizzle (Lightweight)', value: 'drizzle' }
      ],
      supabase: [
        { name: '🔷 Prisma (Full TypeScript)', value: 'prisma' },
        { name: '⚡ Supabase Client (Native)', value: 'none' }
      ],
      planetscale: [
        { name: '🔷 Prisma (Recommended)', value: 'prisma' },
        { name: '🌊 Drizzle (Edge-optimized)', value: 'drizzle' }
      ]
    };
    
    return ormMap[dbType] || [{ name: 'None', value: 'none' }];
  }
  
  private static getCompatibleStateManagement(projectType: ProjectType) {
    const stateMap: Record<string, Array<{name: string, value: string}>> = {
      react: [
        { name: '🔧 Redux Toolkit (Recommended)', value: 'redux-toolkit' },
        { name: '🐻 Zustand (Simple)', value: 'zustand' },
        { name: '⚛️  Jotai (Atomic)', value: 'jotai' },
        { name: '❌ No state management', value: 'none' }
      ],
      vue: [
        { name: '🍍 Pinia (Recommended)', value: 'pinia' },
        { name: '🗃️  Vuex (Classic)', value: 'vuex' },
        { name: '❌ No state management', value: 'none' }
      ],
      nextjs: [
        { name: '🔧 Redux Toolkit', value: 'redux-toolkit' },
        { name: '🐻 Zustand', value: 'zustand' },
        { name: '❌ No state management', value: 'none' }
      ]
    };
    
    return stateMap[projectType] || [{ name: 'None', value: 'none' }];
  }
} 