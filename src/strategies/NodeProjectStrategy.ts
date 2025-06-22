import { ProjectStrategy } from './ProjectStrategy.js';
import * as fs from 'fs';
import * as path from 'path';
import { fileURLToPath } from 'url';

export class NodeProjectStrategy extends ProjectStrategy {
  public generate(projectName: string): void {
    const __filename = fileURLToPath(import.meta.url);
    const __dirname = path.dirname(__filename);
    const templatePath = path.resolve(__dirname, '../templates/node-ts');
    const projectPath = path.join(process.cwd(), projectName);

    console.log(`Generating Node.js project: ${projectName}`);

    // Copy template files
    fs.cpSync(templatePath, projectPath, { recursive: true });

    // Update package.json
    const packageJsonPath = path.join(projectPath, 'package.json');
    let packageJson = fs.readFileSync(packageJsonPath, 'utf8');
    packageJson = packageJson.replace('{{projectName}}', projectName);
    fs.writeFileSync(packageJsonPath, packageJson);
  }
} 