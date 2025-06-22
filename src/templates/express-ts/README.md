# {{projectName}} API

A modern Express.js REST API built with TypeScript, featuring authentication, validation, and MongoDB integration.

## ✨ Features

- **🚀 Express.js** - Fast, minimalist web framework
- **🔧 TypeScript** - Full type safety and excellent DX
- **🔐 Authentication** - JWT-based authentication system
- **✅ Validation** - Request validation with Zod
- **🗃️ MongoDB** - NoSQL database with Mongoose ODM
- **🛡️ Security** - Helmet, CORS, rate limiting
- **📊 Logging** - Morgan HTTP request logger

## 🚀 Quick Start

### Prerequisites

- Node.js 16+
- MongoDB (local or cloud)
- npm or yarn

### Installation

```bash
# Navigate to your project
cd {{projectName}}

# Install dependencies
npm install

# Set up environment variables
cp .env.example .env
# Edit .env with your configuration

# Start development server
npm run dev
```

Your API will be available at `http://localhost:3000`

## 📜 Available Scripts

- `npm run dev` - Start development server with hot reload
- `npm run build` - Build for production
- `npm start` - Start production server
- `npm run lint` - Run ESLint
- `npm test` - Run tests

## 🏗️ API Endpoints

### Health Check
- `GET /health` - API health status

### Authentication
- `POST /api/auth/register` - User registration
- `POST /api/auth/login` - User login

### Users
- `GET /api/users` - Get all users
- `GET /api/users/:id` - Get user by ID

## 🔧 Environment Variables

```bash
# Server Configuration
PORT=3000
NODE_ENV=development

# Database
MONGODB_URI=mongodb://localhost:27017/{{projectName}}

# JWT
JWT_SECRET=your-super-secret-key

# API
API_VERSION=v1
```

## 🏗️ Project Structure

```
{{projectName}}/
├── src/
│   ├── routes/          # API routes
│   ├── middleware/      # Custom middleware
│   ├── models/          # Database models
│   ├── utils/          # Utility functions
│   └── index.ts        # App entry point
├── dist/               # Compiled JavaScript
├── package.json        # Dependencies and scripts
├── tsconfig.json       # TypeScript configuration
└── .env               # Environment variables
```

## 🔐 Authentication

The API uses JWT tokens for authentication:

1. Register: `POST /api/auth/register`
2. Login: `POST /api/auth/login`
3. Include token in Authorization header: `Bearer <token>`

## 🚀 Deployment

### Build for Production
```bash
npm run build
npm start
```

### Deploy to Railway
```bash
npm install -g @railway/cli
railway login
railway init
railway up
```

### Deploy to Heroku
```bash
git init
heroku create {{projectName}}-api
git add .
git commit -m "Initial commit"
git push heroku main
```

## 📚 Learn More

- [Express.js Documentation](https://expressjs.com/)
- [TypeScript Handbook](https://www.typescriptlang.org/docs/)
- [MongoDB Documentation](https://docs.mongodb.com/)
- [Mongoose ODM](https://mongoosejs.com/)

---

**Happy Coding!** 🎉

Built with ❤️ using [Scaffold-Craft](https://github.com/yourusername/scaffold-craft) 