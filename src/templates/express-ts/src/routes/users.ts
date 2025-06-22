import { Router } from 'express';

const router = Router();

// GET /api/users
router.get('/', (req, res) => {
  res.json({
    message: 'Users endpoint',
    data: [
      { id: 1, name: 'John Doe', email: 'john@example.com' },
      { id: 2, name: 'Jane Smith', email: 'jane@example.com' }
    ]
  });
});

// GET /api/users/:id
router.get('/:id', (req, res) => {
  const { id } = req.params;
  res.json({
    message: `User ${id}`,
    data: { id, name: 'John Doe', email: 'john@example.com' }
  });
});

export { router as userRoutes }; 