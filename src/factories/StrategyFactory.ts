import { ProjectStrategy } from '../strategies/ProjectStrategy.js';
import { NodeProjectStrategy } from '../strategies/NodeProjectStrategy.js';
import { ReactProjectStrategy } from '../strategies/ReactProjectStrategy.js';

export class StrategyFactory {
  static createStrategy(projectType: string): ProjectStrategy {
    switch (projectType) {
      case 'Node.js':
        return new NodeProjectStrategy();
      case 'React':
        return new ReactProjectStrategy();
      default:
        throw new Error('Invalid project type');
    }
  }
} 