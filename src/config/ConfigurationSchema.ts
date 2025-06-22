export interface ProjectConfiguration {
  // Basic project info
  projectName: string;
  projectType: ProjectType;
  
  // Database configuration
  database?: DatabaseConfig;
  
  // Authentication configuration
  authentication?: AuthConfig;
  
  // UI Framework configuration
  uiFramework?: UIFrameworkConfig;
  
  // State management (for frontend projects)
  stateManagement?: StateManagementConfig;
  
  // Additional features
  features?: FeatureConfig;
  
  // Deployment configuration
  deployment?: DeploymentConfig;
}

export type ProjectType = 
  | 'react'
  | 'vue'
  | 'angular'
  | 'svelte'
  | 'nextjs'
  | 'express'
  | 'nodejs'
  | 'fullstack-react-express'
  | 'fullstack-nextjs-prisma'
  | 'microservices';

export interface DatabaseConfig {
  type: 'postgresql' | 'mongodb' | 'sqlite' | 'mysql' | 'supabase' | 'planetscale' | 'none';
  orm?: 'prisma' | 'mongoose' | 'typeorm' | 'drizzle' | 'none';
  seedData?: boolean;
}

export interface AuthConfig {
  provider: 'jwt' | 'auth0' | 'firebase' | 'clerk' | 'nextauth' | 'supabase' | 'none';
  features?: {
    registration?: boolean;
    emailVerification?: boolean;
    passwordReset?: boolean;
    socialLogin?: Array<'google' | 'github' | 'twitter' | 'facebook'>;
    twoFactor?: boolean;
  };
}

export interface UIFrameworkConfig {
  styling: 'tailwind' | 'material-ui' | 'chakra-ui' | 'ant-design' | 'styled-components' | 'bootstrap' | 'css-modules';
  components?: {
    library?: 'headless-ui' | 'radix-ui' | 'mantine' | 'nextui' | 'none';
    icons?: 'heroicons' | 'lucide' | 'react-icons' | 'phosphor' | 'none';
  };
}

export interface StateManagementConfig {
  type: 'redux-toolkit' | 'zustand' | 'jotai' | 'valtio' | 'pinia' | 'vuex' | 'akita' | 'none';
  middleware?: Array<'persist' | 'devtools' | 'logger'>;
}

export interface FeatureConfig {
  testing?: {
    unit?: 'jest' | 'vitest' | 'none';
    e2e?: 'cypress' | 'playwright' | 'none';
    coverage?: boolean;
  };
  linting?: {
    eslint?: boolean;
    prettier?: boolean;
    husky?: boolean;
    lintStaged?: boolean;
  };
  documentation?: {
    storybook?: boolean;
    typedoc?: boolean;
    readme?: 'basic' | 'comprehensive';
  };
  analytics?: {
    provider?: 'google-analytics' | 'mixpanel' | 'posthog' | 'amplitude' | 'none';
  };
  monitoring?: {
    errorTracking?: 'sentry' | 'bugsnag' | 'rollbar' | 'none';
    performance?: 'web-vitals' | 'lighthouse-ci' | 'none';
  };
  containerization?: {
    docker?: boolean;
    dockerCompose?: boolean;
    kubernetes?: boolean;
  };
}

export interface DeploymentConfig {
  platform: 'vercel' | 'netlify' | 'railway' | 'render' | 'aws' | 'digitalocean' | 'heroku' | 'none';
  cicd?: {
    provider?: 'github-actions' | 'gitlab-ci' | 'circle-ci' | 'none';
    features?: Array<'auto-deploy' | 'testing' | 'security-scan' | 'performance-test'>;
  };
  environment?: {
    staging?: boolean;
    production?: boolean;
    preview?: boolean;
  };
}

// Configuration presets for common setups
export const ConfigurationPresets = {
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
} as const;

// Validation helpers
export class ConfigurationValidator {
  static validate(config: ProjectConfiguration): { valid: boolean; errors: string[] } {
    const errors: string[] = [];
    
    // Validate project name
    if (!config.projectName || config.projectName.length < 1) {
      errors.push('Project name is required');
    }
    
    if (!/^[a-z0-9-]+$/.test(config.projectName)) {
      errors.push('Project name must contain only lowercase letters, numbers, and hyphens');
    }
    
    // Validate database + ORM compatibility
    if (config.database) {
      const { type, orm } = config.database;
      if (type === 'mongodb' && orm && !['mongoose', 'prisma'].includes(orm)) {
        errors.push('MongoDB is only compatible with Mongoose or Prisma ORM');
      }
      if (type === 'sqlite' && orm === 'mongoose') {
        errors.push('SQLite is not compatible with Mongoose ORM');
      }
    }
    
    // Validate framework + state management compatibility
    if (config.stateManagement) {
      const { type } = config.stateManagement;
      if (['pinia', 'vuex'].includes(type) && !['vue'].includes(config.projectType)) {
        errors.push('Pinia and Vuex are only compatible with Vue.js projects');
      }
    }
    
    return { valid: errors.length === 0, errors };
  }
  
  static getCompatibleOptions(projectType: ProjectType): Partial<ProjectConfiguration> {
    switch (projectType) {
      case 'react':
      case 'nextjs':
        return {
          stateManagement: { type: 'redux-toolkit' },
          uiFramework: { styling: 'tailwind', components: { library: 'headless-ui' } }
        };
      case 'vue':
        return {
          stateManagement: { type: 'pinia' },
          uiFramework: { styling: 'tailwind' }
        };
      case 'angular':
        return {
          uiFramework: { styling: 'material-ui' }
        };
      default:
        return {};
    }
  }
} 