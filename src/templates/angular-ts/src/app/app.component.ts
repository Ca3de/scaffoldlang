import { Component } from '@angular/core';

@Component({
  selector: 'app-root',
  template: `
    <main class="container">
      <div class="hero">
        <h1>Welcome to <span class="highlight">{{title}}</span></h1>
        <p class="lead">A modern Angular application built with TypeScript</p>
      </div>

      <div class="features">
        <div class="feature-card">
          <h3>🚀 Angular 17</h3>
          <p>Latest Angular with standalone components and signals</p>
        </div>
        <div class="feature-card">
          <h3>🔧 TypeScript</h3>
          <p>Full type safety and excellent developer experience</p>
        </div>
        <div class="feature-card">
          <h3>📦 Dependency Injection</h3>
          <p>Powerful DI system for scalable applications</p>
        </div>
      </div>

      <div class="actions">
        <button (click)="incrementCounter()" class="btn btn-primary">
          Count: {{ counter }}
        </button>
        <button (click)="resetCounter()" class="btn btn-secondary">
          Reset
        </button>
      </div>
    </main>
  `,
  styles: [`
    .container {
      display: flex;
      flex-direction: column;
      align-items: center;
      justify-content: center;
      min-height: 100vh;
      padding: 2rem;
      text-align: center;
      max-width: 800px;
      margin: 0 auto;
    }

    .hero h1 {
      font-size: 3rem;
      margin-bottom: 1rem;
      color: #2c3e50;
    }

    .highlight {
      color: #e74c3c;
    }

    .lead {
      font-size: 1.2rem;
      color: #7f8c8d;
      margin-bottom: 3rem;
    }

    .features {
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
      gap: 2rem;
      margin-bottom: 3rem;
      width: 100%;
    }

    .feature-card {
      padding: 2rem;
      border: 1px solid #ecf0f1;
      border-radius: 8px;
      box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
      transition: transform 0.2s;
    }

    .feature-card:hover {
      transform: translateY(-2px);
    }

    .feature-card h3 {
      color: #2c3e50;
      margin-bottom: 1rem;
    }

    .actions {
      display: flex;
      gap: 1rem;
    }

    .btn {
      padding: 0.75rem 1.5rem;
      border: none;
      border-radius: 4px;
      font-size: 1rem;
      cursor: pointer;
      transition: all 0.3s;
    }

    .btn-primary {
      background-color: #e74c3c;
      color: white;
    }

    .btn-primary:hover {
      background-color: #c0392b;
    }

    .btn-secondary {
      background-color: #95a5a6;
      color: white;
    }

    .btn-secondary:hover {
      background-color: #7f8c8d;
    }
  `]
})
export class AppComponent {
  title = '{{projectName}}';
  counter = 0;

  incrementCounter(): void {
    this.counter++;
  }

  resetCounter(): void {
    this.counter = 0;
  }
} 