# {{projectName}}

A modern Vue.js application built with TypeScript and Vite.

## ✨ Features

- **🚀 Vue 3** - Latest Vue with Composition API
- **🔧 TypeScript** - Full type safety and excellent DX
- **⚡ Vite** - Lightning fast build tool with instant HMR
- **🧭 Vue Router** - Client-side routing with multiple views
- **🗃️ Pinia** - Modern state management
- **📱 Responsive** - Mobile-first responsive design

## 🚀 Quick Start

### Prerequisites

- Node.js 16+
- npm or yarn

### Installation

```bash
# Navigate to your project
cd {{projectName}}

# Install dependencies
npm install

# Start development server
npm run dev
```

Your app will be available at `http://localhost:5173`

## 📜 Available Scripts

- `npm run dev` - Start development server
- `npm run build` - Build for production
- `npm run preview` - Preview production build
- `npm run lint` - Run ESLint

## 🏗️ Project Structure

```
{{projectName}}/
├── public/              # Static assets
├── src/
│   ├── components/      # Vue components
│   ├── views/          # Page views
│   ├── router/         # Vue Router configuration
│   ├── stores/         # Pinia stores
│   ├── App.vue         # Root component
│   └── main.ts         # App entry point
├── index.html          # HTML template
├── package.json        # Dependencies and scripts
├── tsconfig.json       # TypeScript configuration
└── vite.config.ts      # Vite configuration
```

## 🎨 Customization

### Adding New Views
1. Create a new file in `src/views/`
2. Add the route to `src/router/index.ts`
3. Add navigation links in `App.vue`

### State Management
- Use Pinia stores in `src/stores/`
- Import and use in components as needed

### Styling
- Scoped styles in Vue components
- Global styles in `src/style.css`

## 🚀 Deployment

### Build for Production
```bash
npm run build
```

### Deploy to Vercel
```bash
npm install -g vercel
vercel
```

### Deploy to Netlify
```bash
npm run build
# Then drag and drop the 'dist' folder to Netlify
```

## 📚 Learn More

- [Vue 3 Documentation](https://vuejs.org/)
- [Vue Router](https://router.vuejs.org/)
- [Pinia](https://pinia.vuejs.org/)
- [Vite Guide](https://vitejs.dev/guide/)

---

**Happy Coding!** 🎉

Built with ❤️ using [Scaffold-Craft](https://github.com/yourusername/scaffold-craft) 