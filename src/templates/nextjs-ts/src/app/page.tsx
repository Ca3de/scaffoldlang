import Link from 'next/link'

export default function Home() {
  return (
    <main className="flex min-h-screen flex-col items-center justify-center p-24">
      <div className="z-10 max-w-5xl w-full items-center justify-between text-center">
        <h1 className="text-6xl font-bold mb-8">
          Welcome to{' '}
          <span className="text-blue-600">{{projectName}}</span>
        </h1>
        
        <p className="text-xl text-gray-600 mb-12 max-w-2xl mx-auto">
          A modern Next.js application built with TypeScript, Tailwind CSS, and the App Router.
        </p>

        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6 mb-12">
          <FeatureCard
            title="⚡ Next.js 14"
            description="Latest App Router with server components and streaming"
          />
          <FeatureCard
            title="🔧 TypeScript"
            description="Full type safety with excellent developer experience"
          />
          <FeatureCard
            title="🎨 Tailwind CSS"
            description="Utility-first CSS framework for rapid UI development"
          />
        </div>

        <div className="flex gap-4 justify-center">
          <Link
            href="/about"
            className="bg-blue-600 hover:bg-blue-700 text-white px-6 py-3 rounded-lg transition-colors"
          >
            About Page
          </Link>
          <a
            href="https://nextjs.org/docs"
            target="_blank"
            rel="noopener noreferrer"
            className="border border-gray-300 hover:border-gray-400 px-6 py-3 rounded-lg transition-colors"
          >
            Documentation
          </a>
        </div>
      </div>
    </main>
  )
}

function FeatureCard({ title, description }: { title: string; description: string }) {
  return (
    <div className="p-6 border border-gray-200 rounded-lg hover:shadow-md transition-shadow">
      <h3 className="text-xl font-semibold mb-2">{title}</h3>
      <p className="text-gray-600">{description}</p>
    </div>
  )
} 