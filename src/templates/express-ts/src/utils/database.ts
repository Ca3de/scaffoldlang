import mongoose from 'mongoose';

export const connectDB = async (): Promise<void> => {
  try {
    const mongoUri = process.env.MONGODB_URI || 'mongodb://localhost:27017/{{projectName}}';
    
    await mongoose.connect(mongoUri);
    console.log('📦 MongoDB connected successfully');
  } catch (error) {
    console.error('❌ MongoDB connection failed:', error);
    console.log('ℹ️  To connect to MongoDB:');
    console.log('   1. Install MongoDB locally');
    console.log('   2. Set MONGODB_URI in your .env file');
    console.log('   3. Or use MongoDB Atlas for cloud database');
    // Don't exit process in development
    if (process.env.NODE_ENV === 'production') {
      process.exit(1);
    }
  }
}; 