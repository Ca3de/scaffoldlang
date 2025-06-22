use std::collections::HashMap;
use ndarray::{Array, Array1, Array2, Array3, Axis};
use nalgebra::{DMatrix, DVector};
use rayon::prelude::*;
use crate::interpreter::{Value, RuntimeError};

/// CPU-Optimized AI/ML System for ScaffoldLang
/// Uses ndarray and nalgebra for high-performance mathematical operations

#[derive(Debug, Clone)]
pub enum MLOperation {
    // Tensor Operations
    TensorAdd,
    TensorMultiply,
    TensorSubtract,
    TensorDivide,
    MatrixMultiply,
    
    // Neural Network Operations
    Dense,
    Activation(ActivationType),
    
    // Machine Learning Algorithms
    LinearRegression,
    KMeansClustering,
}

#[derive(Debug, Clone)]
pub enum ActivationType {
    ReLU,
    Sigmoid,
    Tanh,
    Softmax,
    LeakyReLU,
}

/// High-performance ML runtime with CPU optimizations
pub struct MLRuntime {
    pub tensors: HashMap<String, MLTensor>,
    pub models: HashMap<String, MLModel>,
    pub performance_stats: PerformanceStats,
}

#[derive(Debug, Clone)]
pub struct MLTensor {
    pub data: Array2<f64>, // 2D tensor for simplicity
    pub shape: Vec<usize>,
    pub requires_grad: bool,
}

#[derive(Debug, Clone)]
pub struct MLModel {
    pub name: String,
    pub layers: Vec<Layer>,
    pub weights: HashMap<String, Array2<f64>>,
    pub biases: HashMap<String, Array1<f64>>,
    pub is_trained: bool,
}

#[derive(Debug, Clone)]
pub struct Layer {
    pub layer_type: LayerType,
    pub input_size: usize,
    pub output_size: usize,
    pub activation: Option<ActivationType>,
}

#[derive(Debug, Clone)]
pub enum LayerType {
    Dense,
}

#[derive(Debug, Default)]
pub struct PerformanceStats {
    pub tensor_ops_per_second: f64,
    pub matrix_multiplications: u64,
    pub training_iterations: u64,
    pub inference_time_ms: f64,
}

impl MLRuntime {
    pub fn new() -> Self {
        Self {
            tensors: HashMap::new(),
            models: HashMap::new(),
            performance_stats: PerformanceStats::default(),
        }
    }

    /// Create a new tensor
    pub fn create_tensor(&mut self, name: String, shape: Vec<usize>, data: Vec<f64>) -> Result<(), RuntimeError> {
        if shape.len() != 2 {
            return Err(RuntimeError::ValueError("Only 2D tensors supported".to_string()));
        }
        
        let total_size = shape.iter().product::<usize>();
        if data.len() != total_size {
            return Err(RuntimeError::ValueError(format!("Data size {} doesn't match shape {:?}", data.len(), shape)));
        }

        let tensor_data = Array2::from_shape_vec((shape[0], shape[1]), data)
            .map_err(|e| RuntimeError::ValueError(format!("Failed to create tensor: {}", e)))?;

        let tensor = MLTensor {
            data: tensor_data,
            shape,
            requires_grad: false,
        };

        self.tensors.insert(name, tensor);
        Ok(())
    }

    /// Perform tensor addition
    pub fn tensor_add(&mut self, a_name: &str, b_name: &str, result_name: String) -> Result<(), RuntimeError> {
        let a = self.tensors.get(a_name).ok_or_else(|| RuntimeError::NameError(format!("Tensor '{}' not found", a_name)))?;
        let b = self.tensors.get(b_name).ok_or_else(|| RuntimeError::NameError(format!("Tensor '{}' not found", b_name)))?;

        if a.shape != b.shape {
            return Err(RuntimeError::ValueError(format!("Shape mismatch: {:?} vs {:?}", a.shape, b.shape)));
        }

        let result_data = &a.data + &b.data;

        let result_tensor = MLTensor {
            data: result_data.clone(),
            shape: a.shape.clone(),
            requires_grad: a.requires_grad || b.requires_grad,
        };

        self.tensors.insert(result_name, result_tensor);
        self.performance_stats.tensor_ops_per_second += 1.0;
        Ok(())
    }

    /// High-performance matrix multiplication
    pub fn matrix_multiply(&mut self, a_name: &str, b_name: &str, result_name: String) -> Result<(), RuntimeError> {
        let a = self.tensors.get(a_name).ok_or_else(|| RuntimeError::NameError(format!("Tensor '{}' not found", a_name)))?;
        let b = self.tensors.get(b_name).ok_or_else(|| RuntimeError::NameError(format!("Tensor '{}' not found", b_name)))?;

        if a.data.ncols() != b.data.nrows() {
            return Err(RuntimeError::ValueError(format!("Matrix dimension mismatch: {}x{} vs {}x{}", 
                a.data.nrows(), a.data.ncols(), b.data.nrows(), b.data.ncols())));
        }

        let result_data = a.data.dot(&b.data);

        let result_tensor = MLTensor {
            data: result_data.clone(),
            shape: vec![result_data.nrows(), result_data.ncols()],
            requires_grad: a.requires_grad || b.requires_grad,
        };

        self.tensors.insert(result_name, result_tensor);
        self.performance_stats.matrix_multiplications += 1;
        Ok(())
    }

    /// Apply activation function
    pub fn apply_activation(&mut self, tensor_name: &str, activation: ActivationType, result_name: String) -> Result<(), RuntimeError> {
        let tensor = self.tensors.get(tensor_name).ok_or_else(|| RuntimeError::NameError(format!("Tensor '{}' not found", tensor_name)))?;

        let result_data = match activation {
            ActivationType::ReLU => tensor.data.mapv(|x| x.max(0.0)),
            ActivationType::Sigmoid => tensor.data.mapv(|x| 1.0 / (1.0 + (-x).exp())),
            ActivationType::Tanh => tensor.data.mapv(|x| x.tanh()),
            ActivationType::LeakyReLU => tensor.data.mapv(|x| if x > 0.0 { x } else { 0.01 * x }),
            ActivationType::Softmax => {
                let exp_data = tensor.data.mapv(|x| x.exp());
                let sum = exp_data.sum();
                exp_data.mapv(|x| x / sum)
            }
        };

        let result_tensor = MLTensor {
            data: result_data.clone(),
            shape: tensor.shape.clone(),
            requires_grad: tensor.requires_grad,
        };

        self.tensors.insert(result_name, result_tensor);
        Ok(())
    }

    /// Get tensor data as ScaffoldLang values
    pub fn get_tensor_values(&self, name: &str) -> Result<Vec<Value>, RuntimeError> {
        let tensor = self.tensors.get(name).ok_or_else(|| RuntimeError::NameError(format!("Tensor '{}' not found", name)))?;
        
        let values = tensor.data.iter()
            .map(|&x| Value::Float(x))
            .collect();
        
        Ok(values)
    }

    /// Get performance statistics
    pub fn get_performance_stats(&self) -> HashMap<String, Value> {
        let mut stats = HashMap::new();
        stats.insert("tensor_ops_per_second".to_string(), Value::Float(self.performance_stats.tensor_ops_per_second));
        stats.insert("matrix_multiplications".to_string(), Value::Integer(self.performance_stats.matrix_multiplications as i64));
        stats.insert("training_iterations".to_string(), Value::Integer(self.performance_stats.training_iterations as i64));
        stats.insert("inference_time_ms".to_string(), Value::Float(self.performance_stats.inference_time_ms));
        stats
    }
}

/// Standalone mathematical functions
pub fn sigmoid(x: f64) -> f64 {
    1.0 / (1.0 + (-x).exp())
}

pub fn relu(x: f64) -> f64 {
    x.max(0.0)
}

pub fn tanh_activation(x: f64) -> f64 {
    x.tanh()
}
