use std::collections::HashMap;
use std::time::{Duration, Instant};
use serde::{Serialize, Deserialize};
use crate::ast::{Statement, Expression, Type};

/// Ultra-Performance Execution Profiler for Profile-Guided Optimization
/// Collects detailed runtime statistics to enable adaptive recompilation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExecutionProfiler {
    // Hot path analysis
    pub hot_paths: HashMap<String, HotPathStats>,
    pub execution_counts: HashMap<usize, u64>,
    
    // Type usage patterns
    pub type_frequencies: HashMap<String, TypeUsageStats>,
    pub type_conversions: HashMap<String, u64>,
    
    // Branch prediction data
    pub branch_stats: HashMap<usize, BranchPredictionStats>,
    
    // Memory allocation patterns
    pub allocation_stats: AllocationStats,
    
    // Performance timing
    pub timing_data: HashMap<String, TimingStats>,
    
    // Loop optimization data
    pub loop_stats: HashMap<usize, LoopStats>,
    
    // Function call patterns
    pub function_calls: HashMap<String, FunctionCallStats>,
    
    // SIMD opportunities
    pub vectorization_candidates: Vec<VectorizationCandidate>,
    
    // Overall statistics
    pub total_operations: u64,
    pub total_execution_time: Duration,
    pub profiling_overhead: Duration,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HotPathStats {
    pub execution_count: u64,
    pub total_time: Duration,
    pub average_time: Duration,
    pub code_hash: u64,
    pub optimization_level: OptimizationLevel,
    pub should_recompile: bool,
    // Note: Instant doesn't serialize, so we skip this field
    #[serde(skip)]
    pub last_recompilation: Option<Instant>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TypeUsageStats {
    pub usage_count: u64,
    pub operations: HashMap<String, u64>, // add, mul, etc.
    pub dominant_type: Option<String>,
    pub specialization_benefit: f64, // 0.0 - 1.0
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BranchPredictionStats {
    pub total_branches: u64,
    pub taken_count: u64,
    pub not_taken_count: u64,
    pub prediction_accuracy: f64,
    pub misprediction_cost: Duration,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AllocationStats {
    pub total_allocations: u64,
    pub heap_allocations: u64,
    pub stack_allocations: u64,
    pub pool_allocations: u64,
    pub allocation_sizes: HashMap<usize, u64>,
    pub gc_pressure: f64,
    pub escape_analysis_opportunities: u64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TimingStats {
    pub call_count: u64,
    pub total_time: Duration,
    pub min_time: Duration,
    pub max_time: Duration,
    pub average_time: Duration,
    pub percentile_95: Duration,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LoopStats {
    pub iterations: Vec<u64>,
    pub average_iterations: f64,
    pub max_iterations: u64,
    pub unroll_candidate: bool,
    pub vectorization_candidate: bool,
    pub parallelization_candidate: bool,
    pub cache_miss_rate: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FunctionCallStats {
    pub call_count: u64,
    pub inline_candidate: bool,
    pub recursive_depth: u32,
    pub argument_types: HashMap<String, u64>,
    pub return_type_frequency: HashMap<String, u64>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VectorizationCandidate {
    pub location: usize,
    pub operation_type: VectorizationType,
    pub data_size: usize,
    pub expected_speedup: f64,
    pub simd_width: usize, // 4 for AVX2, 8 for AVX-512
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum VectorizationType {
    MathOperations,
    ArrayProcessing,
    TensorOperations,
    StringOperations,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum OptimizationLevel {
    None,
    Basic,
    Aggressive,
    UltraFast,
    ProfileGuided,
}

impl ExecutionProfiler {
    pub fn new() -> Self {
        Self {
            hot_paths: HashMap::new(),
            execution_counts: HashMap::new(),
            type_frequencies: HashMap::new(),
            type_conversions: HashMap::new(),
            branch_stats: HashMap::new(),
            allocation_stats: AllocationStats::new(),
            timing_data: HashMap::new(),
            loop_stats: HashMap::new(),
            function_calls: HashMap::new(),
            vectorization_candidates: Vec::new(),
            total_operations: 0,
            total_execution_time: Duration::new(0, 0),
            profiling_overhead: Duration::new(0, 0),
        }
    }
    
    /// Start profiling a code section
    pub fn start_profiling(&mut self, section_id: &str) -> ProfilingContext {
        let start_time = Instant::now();
        ProfilingContext {
            section_id: section_id.to_string(),
            start_time,
            profiler_start: Instant::now(),
        }
    }
    
    /// End profiling and record results
    pub fn end_profiling(&mut self, context: ProfilingContext) {
        let execution_time = context.start_time.elapsed();
        let total_profiler_time = context.profiler_start.elapsed();
        
        // Calculate profiling overhead safely - profiler time should include execution time
        let profiling_overhead = if total_profiler_time > execution_time {
            total_profiler_time - execution_time
        } else {
            // If for some reason total time is less than execution time, set overhead to zero
            Duration::new(0, 0)
        };
        
        let stats = self.timing_data.entry(context.section_id.clone()).or_insert(TimingStats {
            call_count: 0,
            total_time: Duration::new(0, 0),
            min_time: Duration::new(u64::MAX, 0),
            max_time: Duration::new(0, 0),
            average_time: Duration::new(0, 0),
            percentile_95: Duration::new(0, 0),
        });
        
        stats.call_count += 1;
        stats.total_time += execution_time;
        stats.min_time = stats.min_time.min(execution_time);
        stats.max_time = stats.max_time.max(execution_time);
        stats.average_time = stats.total_time / stats.call_count as u32;
        
        self.total_execution_time += execution_time;
        self.profiling_overhead += profiling_overhead;
        
        // Check if this is a hot path
        let section_id_clone = context.section_id.clone();
        let should_mark_hot = stats.call_count > 1000 || execution_time > Duration::from_millis(10);
        let stats_clone = stats.clone();
        
        if should_mark_hot {
            self.mark_as_hot_path(&section_id_clone, &stats_clone);
        }
    }
    
    /// Record type usage for specialization opportunities
    pub fn record_type_usage(&mut self, type_name: &str, operation: &str) {
        let stats = self.type_frequencies.entry(type_name.to_string()).or_insert(TypeUsageStats {
            usage_count: 0,
            operations: HashMap::new(),
            dominant_type: None,
            specialization_benefit: 0.0,
        });
        
        stats.usage_count += 1;
        *stats.operations.entry(operation.to_string()).or_insert(0) += 1;
        
        // Calculate specialization benefit
        if stats.usage_count > 100 {
            let dominant_op_count = stats.operations.values().max().unwrap_or(&0);
            stats.specialization_benefit = (*dominant_op_count as f64) / (stats.usage_count as f64);
        }
    }
    
    /// Record branch prediction data
    pub fn record_branch(&mut self, branch_id: usize, taken: bool, execution_time: Duration) {
        let stats = self.branch_stats.entry(branch_id).or_insert(BranchPredictionStats {
            total_branches: 0,
            taken_count: 0,
            not_taken_count: 0,
            prediction_accuracy: 0.0,
            misprediction_cost: Duration::new(0, 0),
        });
        
        stats.total_branches += 1;
        if taken {
            stats.taken_count += 1;
        } else {
            stats.not_taken_count += 1;
        }
        
        // Simple prediction accuracy (assume always predict most frequent)
        let taken_ratio = stats.taken_count as f64 / stats.total_branches as f64;
        stats.prediction_accuracy = taken_ratio.max(1.0 - taken_ratio);
        
        // Record potential misprediction cost
        if stats.prediction_accuracy < 0.8 {
            stats.misprediction_cost += execution_time;
        }
    }
    
    /// Record loop execution data
    pub fn record_loop_execution(&mut self, loop_id: usize, iterations: u64, is_vectorizable: bool) {
        let stats = self.loop_stats.entry(loop_id).or_insert(LoopStats {
            iterations: Vec::new(),
            average_iterations: 0.0,
            max_iterations: 0,
            unroll_candidate: false,
            vectorization_candidate: false,
            parallelization_candidate: false,
            cache_miss_rate: 0.0,
        });
        
        stats.iterations.push(iterations);
        stats.max_iterations = stats.max_iterations.max(iterations);
        stats.average_iterations = stats.iterations.iter().sum::<u64>() as f64 / stats.iterations.len() as f64;
        
        // Determine optimization opportunities
        stats.unroll_candidate = iterations < 16 && stats.average_iterations < 8.0;
        stats.vectorization_candidate = is_vectorizable && iterations > 4;
        stats.parallelization_candidate = iterations > 1000;
        
        // Record vectorization candidate
        if stats.vectorization_candidate {
            self.vectorization_candidates.push(VectorizationCandidate {
                location: loop_id,
                operation_type: VectorizationType::MathOperations,
                data_size: iterations as usize,
                expected_speedup: if iterations > 8 { 8.0 } else { 4.0 },
                simd_width: if self.supports_avx512() { 8 } else { 4 },
            });
        }
    }
    
    /// Record memory allocation
    pub fn record_allocation(&mut self, size: usize, allocation_type: AllocationType) {
        self.allocation_stats.total_allocations += 1;
        *self.allocation_stats.allocation_sizes.entry(size).or_insert(0) += 1;
        
        match allocation_type {
            AllocationType::Heap => self.allocation_stats.heap_allocations += 1,
            AllocationType::Stack => self.allocation_stats.stack_allocations += 1,
            AllocationType::Pool => self.allocation_stats.pool_allocations += 1,
        }
        
        // Calculate GC pressure
        if self.allocation_stats.heap_allocations > 1000 {
            self.allocation_stats.gc_pressure = 
                self.allocation_stats.heap_allocations as f64 / self.allocation_stats.total_allocations as f64;
        }
    }
    
    /// Generate optimization recommendations
    pub fn generate_recommendations(&self) -> Vec<OptimizationRecommendation> {
        let mut recommendations = Vec::new();
        
        // Hot path recompilation recommendations
        for (path_id, stats) in &self.hot_paths {
            if stats.should_recompile {
                recommendations.push(OptimizationRecommendation::RecompileHotPath {
                    path_id: path_id.clone(),
                    expected_speedup: 2.0 + (stats.execution_count as f64).log10() * 0.5,
                    optimization_level: OptimizationLevel::ProfileGuided,
                });
            }
        }
        
        // Type specialization recommendations
        for (type_name, stats) in &self.type_frequencies {
            if stats.specialization_benefit > 0.8 {
                recommendations.push(OptimizationRecommendation::SpecializeType {
                    type_name: type_name.clone(),
                    expected_speedup: 1.5 + stats.specialization_benefit,
                });
            }
        }
        
        // Vectorization recommendations
        for candidate in &self.vectorization_candidates {
            if candidate.expected_speedup > 2.0 {
                recommendations.push(OptimizationRecommendation::VectorizeLoop {
                    location: candidate.location,
                    simd_width: candidate.simd_width,
                    expected_speedup: candidate.expected_speedup,
                });
            }
        }
        
        // Memory optimization recommendations
        if self.allocation_stats.gc_pressure > 0.3 {
            recommendations.push(OptimizationRecommendation::UsePoolAllocator {
                expected_speedup: 1.2 + self.allocation_stats.gc_pressure,
            });
        }
        
        if self.allocation_stats.escape_analysis_opportunities > 100 {
            recommendations.push(OptimizationRecommendation::StackAllocateObjects {
                object_count: self.allocation_stats.escape_analysis_opportunities,
                expected_speedup: 1.3,
            });
        }
        
        recommendations
    }
    
    /// Check if this path should be marked as hot
    fn mark_as_hot_path(&mut self, section_id: &str, timing_stats: &TimingStats) {
        let hot_path = HotPathStats {
            execution_count: timing_stats.call_count,
            total_time: timing_stats.total_time,
            average_time: timing_stats.average_time,
            code_hash: self.calculate_code_hash(section_id),
            optimization_level: OptimizationLevel::None,
            should_recompile: timing_stats.call_count > 1000 || timing_stats.total_time > Duration::from_millis(100),
            last_recompilation: None,
        };
        
        self.hot_paths.insert(section_id.to_string(), hot_path);
    }
    
    fn calculate_code_hash(&self, section_id: &str) -> u64 {
        use std::collections::hash_map::DefaultHasher;
        use std::hash::{Hash, Hasher};
        
        let mut hasher = DefaultHasher::new();
        section_id.hash(&mut hasher);
        hasher.finish()
    }
    
    fn supports_avx512(&self) -> bool {
        // In a real implementation, check CPU features
        // For now, assume AVX-512 is available
        true
    }
    
    /// Export profiling data for analysis
    pub fn export_profile_data(&self) -> Result<String, serde_json::Error> {
        serde_json::to_string_pretty(self)
    }
    
    /// Import previously collected profiling data
    pub fn import_profile_data(&mut self, data: &str) -> Result<(), serde_json::Error> {
        let imported: ExecutionProfiler = serde_json::from_str(data)?;
        
        // Merge with existing data
        for (key, value) in imported.hot_paths {
            self.hot_paths.insert(key, value);
        }
        
        for (key, value) in imported.type_frequencies {
            self.type_frequencies.insert(key, value);
        }
        
        // Merge other statistics...
        self.total_operations += imported.total_operations;
        
        Ok(())
    }
}

impl AllocationStats {
    fn new() -> Self {
        Self {
            total_allocations: 0,
            heap_allocations: 0,
            stack_allocations: 0,
            pool_allocations: 0,
            allocation_sizes: HashMap::new(),
            gc_pressure: 0.0,
            escape_analysis_opportunities: 0,
        }
    }
}

// ProfilingContext doesn't need to be serializable - it's just used during execution
pub struct ProfilingContext {
    section_id: String,
    start_time: Instant,
    profiler_start: Instant,
}

#[derive(Debug, Clone)]
pub enum AllocationType {
    Heap,
    Stack,
    Pool,
}

#[derive(Debug, Clone)]
pub enum OptimizationRecommendation {
    RecompileHotPath {
        path_id: String,
        expected_speedup: f64,
        optimization_level: OptimizationLevel,
    },
    SpecializeType {
        type_name: String,
        expected_speedup: f64,
    },
    VectorizeLoop {
        location: usize,
        simd_width: usize,
        expected_speedup: f64,
    },
    UsePoolAllocator {
        expected_speedup: f64,
    },
    StackAllocateObjects {
        object_count: u64,
        expected_speedup: f64,
    },
}

impl std::fmt::Display for OptimizationRecommendation {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            OptimizationRecommendation::RecompileHotPath { path_id, expected_speedup, .. } => {
                write!(f, "🔥 Recompile hot path '{}' for {:.1}x speedup", path_id, expected_speedup)
            }
            OptimizationRecommendation::SpecializeType { type_name, expected_speedup } => {
                write!(f, "⚡ Specialize type '{}' for {:.1}x speedup", type_name, expected_speedup)
            }
            OptimizationRecommendation::VectorizeLoop { location, expected_speedup, simd_width } => {
                write!(f, "🚀 Vectorize loop at {} with {}-wide SIMD for {:.1}x speedup", location, simd_width, expected_speedup)
            }
            OptimizationRecommendation::UsePoolAllocator { expected_speedup } => {
                write!(f, "💾 Use pool allocator for {:.1}x speedup", expected_speedup)
            }
            OptimizationRecommendation::StackAllocateObjects { object_count, expected_speedup } => {
                write!(f, "📦 Stack allocate {} objects for {:.1}x speedup", object_count, expected_speedup)
            }
        }
    }
}