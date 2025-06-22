import { ProjectStrategy } from './ProjectStrategy.js';
import * as fs from 'fs';
import * as path from 'path';
import { fileURLToPath } from 'url';

export class ReactProjectStrategy extends ProjectStrategy {
  public generate(projectName: string): void {
    const __filename = fileURLToPath(import.meta.url);
    const __dirname = path.dirname(__filename);
    const templatePath = path.resolve(__dirname, '../templates/react-ts');
    const projectPath = path.join(process.cwd(), projectName);

    console.log(`Generating React project: ${projectName}`);

    // Copy template files
    fs.cpSync(templatePath, projectPath, { recursive: true });

    // --- Update package.json ---
    const packageJsonPath = path.join(projectPath, 'package.json');
    let packageJson = fs.readFileSync(packageJsonPath, 'utf8');
    packageJson = packageJson.replace('{{projectName}}', projectName);
    fs.writeFileSync(packageJsonPath, packageJson);
    
    // --- Update index.html ---
    const indexHtmlPath = path.join(projectPath, 'index.html');
    let indexHtml = fs.readFileSync(indexHtmlPath, 'utf8');
    indexHtml = indexHtml.replace('{{projectName}}', projectName);
    fs.writeFileSync(indexHtmlPath, indexHtml);

    // --- Update App.tsx ---
    const appTsxPath = path.join(projectPath, 'src/App.tsx');
    let appTsx = fs.readFileSync(appTsxPath, 'utf8');
    appTsx = appTsx.replace(/{{projectName}}/g, projectName);
    fs.writeFileSync(appTsxPath, appTsx);

    // --- Update README.md ---
    const readmePath = path.join(projectPath, 'README.md');
    let readme = fs.readFileSync(readmePath, 'utf8');
    readme = readme.replace(/{{projectName}}/g, projectName);
    fs.writeFileSync(readmePath, readme);
  }
}
