import { ProjectStrategy } from './ProjectStrategy.js';
import * as fs from 'fs';
import * as path from 'path';
import { fileURLToPath } from 'url';

export class AngularProjectStrategy extends ProjectStrategy {
  public generate(projectName: string): void {
    const __filename = fileURLToPath(import.meta.url);
    const __dirname = path.dirname(__filename);
    const templatePath = path.resolve(__dirname, '../templates/angular-ts');
    const projectPath = path.join(process.cwd(), projectName);

    console.log(`Generating Angular project: ${projectName}`);

    // Copy template files
    fs.cpSync(templatePath, projectPath, { recursive: true });

    // Update files with placeholders
    this.updateTemplateFiles(projectPath, projectName);

    console.log(`✅ Angular project '${projectName}' generated successfully!`);
    console.log(`📁 Navigate to: cd ${projectName}`);
    console.log(`📦 Install dependencies: npm install`);
    console.log(`🚀 Start development: npm start`);
    console.log(`🔗 Open browser: http://localhost:4200`);
    console.log(`🧪 Run tests: npm test`);
  }

  private updateTemplateFiles(projectPath: string, projectName: string): void {
    const filesToUpdate = [
      'package.json',
      'src/app/app.component.ts'
    ];

    filesToUpdate.forEach(filePath => {
      const fullPath = path.join(projectPath, filePath);
      if (fs.existsSync(fullPath)) {
        let content = fs.readFileSync(fullPath, 'utf8');
        content = content.replace(/{{projectName}}/g, projectName);
        fs.writeFileSync(fullPath, content);
      }
    });
  }
} 