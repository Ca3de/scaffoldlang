import { Router } from 'express';
import { z } from 'zod';
import bcrypt from 'bcryptjs';
import jwt from 'jsonwebtoken';

const router = Router();

// Validation schemas
const loginSchema = z.object({
  email: z.string().email(),
  password: z.string().min(6)
});

const registerSchema = z.object({
  name: z.string().min(2),
  email: z.string().email(),
  password: z.string().min(6)
});

// POST /api/auth/register
router.post('/register', async (req, res) => {
  try {
    const { name, email, password } = registerSchema.parse(req.body);
    
    // Hash password
    const hashedPassword = await bcrypt.hash(password, 12);
    
    // TODO: Save user to database
    // const user = await User.create({ name, email, password: hashedPassword });
    
    // Generate JWT
    const token = jwt.sign(
      { userId: 'temp-id', email },
      process.env.JWT_SECRET || 'fallback-secret',
      { expiresIn: '7d' }
    );
    
    res.status(201).json({
      message: 'User created successfully',
      token,
      user: { name, email }
    });
  } catch (error) {
    if (error instanceof z.ZodError) {
      return res.status(400).json({ error: 'Validation failed', details: error.errors });
    }
    res.status(500).json({ error: 'Registration failed' });
  }
});

// POST /api/auth/login
router.post('/login', async (req, res) => {
  try {
    const { email, password } = loginSchema.parse(req.body);
    
    // TODO: Find user in database
    // const user = await User.findOne({ email });
    // if (!user || !await bcrypt.compare(password, user.password)) {
    //   return res.status(401).json({ error: 'Invalid credentials' });
    // }
    
    // Mock authentication for demo
    const isValidPassword = await bcrypt.compare(password, '$2a$12$example');
    
    const token = jwt.sign(
      { userId: 'temp-id', email },
      process.env.JWT_SECRET || 'fallback-secret',
      { expiresIn: '7d' }
    );
    
    res.json({
      message: 'Login successful',
      token,
      user: { email }
    });
  } catch (error) {
    if (error instanceof z.ZodError) {
      return res.status(400).json({ error: 'Validation failed', details: error.errors });
    }
    res.status(500).json({ error: 'Login failed' });
  }
});

export { router as authRoutes }; 