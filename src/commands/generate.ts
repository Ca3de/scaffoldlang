import { Command } from 'commander';
import inquirer from 'inquirer';
import { StrategyFactory } from '../factories/StrategyFactory.js';

interface ProjectAnswers {
  projectName: string;
  projectType: string;
}

export const generateCommand = new Command('generate')
  .description('Generate a new project from a template')
  .action(async () => {
    const answers: ProjectAnswers = await inquirer.prompt([
      {
        type: 'input',
        name: 'projectName',
        message: 'What is the name of your project?',
        validate: (input: string) => input ? true : 'Project name cannot be empty',
      },
      {
        type: 'list',
        name: 'projectType',
        message: 'What type of project do you want to create?',
        choices: ['Node.js', 'React'],
      },
    ]);

    try {
      const strategy = StrategyFactory.createStrategy(answers.projectType);
      strategy.generate(answers.projectName);
      console.log(`Project '${answers.projectName}' generated successfully!`);
    } catch (error) {
      if (error instanceof Error) {
        console.error(error.message);
      } else {
        console.error('An unknown error occurred');
      }
    }
  }); 