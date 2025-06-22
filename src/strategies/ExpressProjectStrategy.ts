import { ProjectStrategy } from './ProjectStrategy.js';
import * as fs from 'fs';
import * as path from 'path';
import { fileURLToPath } from 'url';

export class ExpressProjectStrategy extends ProjectStrategy {
  public generate(projectName: string): void {
    const __filename = fileURLToPath(import.meta.url);
    const __dirname = path.dirname(__filename);
    const templatePath = path.resolve(__dirname, '../templates/express-ts');
    const projectPath = path.join(process.cwd(), projectName);

    console.log(`Generating Express.js API project: ${projectName}`);

    // Copy template files
    fs.cpSync(templatePath, projectPath, { recursive: true });

    // Update package.json
    const packageJsonPath = path.join(projectPath, 'package.json');
    let packageJson = fs.readFileSync(packageJsonPath, 'utf8');
    packageJson = packageJson.replace(/{{projectName}}/g, projectName);
    fs.writeFileSync(packageJsonPath, packageJson);

    // Update other files with placeholders
    this.updateTemplateFiles(projectPath, projectName);

    console.log(`✅ Express.js API project '${projectName}' generated successfully!`);
    console.log(`📁 Navigate to: cd ${projectName}`);
    console.log(`📦 Install dependencies: npm install`);
    console.log(`🚀 Start development: npm run dev`);
    console.log(`📊 Health check: http://localhost:3000/health`);
  }

  private updateTemplateFiles(projectPath: string, projectName: string): void {
    const filesToUpdate = [
      'src/index.ts',
      'src/utils/database.ts'
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