import { useState } from 'react'
import './App.css'

function App() {
  const [count, setCount] = useState(0)

  return (
    <div className="app">
      <header className="app-header">
        <h1>Welcome to <span className="highlight">{{projectName}}</span></h1>
        <p className="subtitle">
          A modern React application built with TypeScript and Vite
        </p>
      </header>

      <main className="app-main">
        <div className="features-grid">
          <FeatureCard
            icon="⚡"
            title="Vite"
            description="Lightning fast build tool with instant HMR"
          />
          <FeatureCard
            icon="🔧"
            title="TypeScript"
            description="Full type safety with excellent developer experience"
          />
          <FeatureCard
            icon="⚛️"
            title="React 18"
            description="Latest React with concurrent features and hooks"
          />
        </div>

        <div className="demo-section">
          <button
            className="counter-btn"
            onClick={() => setCount((count) => count + 1)}
          >
            Count is {count}
          </button>
          <p className="demo-text">
            Edit <code>src/App.tsx</code> and save to test HMR
          </p>
        </div>

        <div className="links-section">
          <a
            href="https://reactjs.org"
            target="_blank"
            rel="noopener noreferrer"
            className="link"
          >
            Learn React
          </a>
          <a
            href="https://vitejs.dev"
            target="_blank"
            rel="noopener noreferrer"
            className="link"
          >
            Learn Vite
          </a>
          <a
            href="https://www.typescriptlang.org/"
            target="_blank"
            rel="noopener noreferrer"
            className="link"
          >
            Learn TypeScript
          </a>
        </div>
      </main>
    </div>
  )
}

function FeatureCard({ icon, title, description }: {
  icon: string;
  title: string;
  description: string;
}) {
  return (
    <div className="feature-card">
      <div className="feature-icon">{icon}</div>
      <h3 className="feature-title">{title}</h3>
      <p className="feature-description">{description}</p>
    </div>
  )
}

export default App 