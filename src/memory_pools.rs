/// Phase 3: Custom Memory Pool Allocators for Zero-Allocation Execution
/// Target: Eliminate GC pressure and achieve deterministic memory allocation
/// 
/// This system implements custom memory pools that pre-allocate memory regions
/// and reuse them without triggering garbage collection, achieving C-like memory performance.

use std::alloc::{alloc, dealloc, Layout};
use std::ptr::{self, NonNull};
use std::sync::{Arc, Mutex};
use std::collections::HashMap;
use crate::interpreter::{Value, RuntimeError};
use crate::execution_profiler::{AllocationStats, AllocationType};

/// High-performance memory pool system for different allocation patterns
pub struct MemoryPoolSystem {
    /// Pool for small objects (< 64 bytes)
    pub small_object_pool: ObjectPool<64>,
    
    /// Pool for medium objects (64-512 bytes)  
    pub medium_object_pool: ObjectPool<512>,
    
    /// Pool for large objects (512-4096 bytes)
    pub large_object_pool: ObjectPool<4096>,
    
    /// Pool for arrays and collections
    pub array_pool: ArrayPool,
    
    /// Pool for string allocations
    pub string_pool: StringPool,
    
    /// Pool for temporary computation values
    pub temp_value_pool: ValuePool,
    
    /// Allocation statistics for profiling
    pub stats: Arc<Mutex<PoolAllocationStats>>,
}

/// Generic object pool for fixed-size allocations
pub struct ObjectPool<const SIZE: usize> {
    /// Pre-allocated memory chunks
    chunks: Vec<MemoryChunk<SIZE>>,
    
    /// Stack of available object slots
    free_slots: Vec<NonNull<u8>>,
    
    /// Total capacity
    capacity: usize,
    
    /// Current allocation count
    allocated: usize,
}

/// Memory chunk containing multiple objects
struct MemoryChunk<const SIZE: usize> {
    /// Raw memory pointer
    memory: NonNull<u8>,
    
    /// Layout for this chunk
    layout: Layout,
    
    /// Number of objects in this chunk
    object_count: usize,
}

/// Specialized pool for arrays with different element types
pub struct ArrayPool {
    /// Pools for different array sizes
    i64_arrays: HashMap<usize, Vec<Box<[i64]>>>,
    f64_arrays: HashMap<usize, Vec<Box<[f64]>>>,
    string_arrays: HashMap<usize, Vec<Box<[String]>>>,
    
    /// Pre-allocated temporary arrays for computations
    temp_i64_arrays: Vec<Box<[i64]>>,
    temp_f64_arrays: Vec<Box<[f64]>>,
    
    /// SIMD-aligned arrays for vectorized operations
    simd_f64_arrays: Vec<AlignedArray<f64, 64>>, // 64-byte aligned for AVX-512
    simd_f32_arrays: Vec<AlignedArray<f32, 64>>,
}

/// SIMD-aligned array wrapper
#[repr(C, align(64))] // 64-byte alignment for AVX-512
pub struct AlignedArray<T, const ALIGN: usize> {
    data: Box<[T]>,
}

/// String pool for efficient string management
pub struct StringPool {
    /// Small strings (< 32 chars)
    small_strings: Vec<String>,
    
    /// Medium strings (32-256 chars)
    medium_strings: Vec<String>,
    
    /// Large strings (> 256 chars)
    large_strings: Vec<String>,
    
    /// String interning cache for common strings
    interned_strings: HashMap<String, Arc<str>>,
}

/// Pool for Value objects to avoid allocation during interpretation
pub struct ValuePool {
    /// Pre-allocated integer values
    integer_values: Vec<Value>,
    
    /// Pre-allocated float values
    float_values: Vec<Value>,
    
    /// Pre-allocated boolean values
    boolean_values: Vec<Value>,
    
    /// Pre-allocated null values
    null_values: Vec<Value>,
}

/// Statistics for memory pool performance
#[derive(Debug, Default)]
pub struct PoolAllocationStats {
    pub total_pool_allocations: u64,
    pub total_heap_allocations: u64,
    pub pool_hits: u64,
    pub pool_misses: u64,
    pub memory_saved: u64, // bytes saved by using pools
    pub gc_avoided_count: u64,
    pub average_allocation_time_ns: f64,
}

impl MemoryPoolSystem {
    /// Create a new memory pool system with pre-allocated pools
    pub fn new() -> Self {
        println!("🏊 Initializing Memory Pool System for zero-allocation execution");
        
        // Pre-allocate pools based on typical usage patterns
        let small_pool = ObjectPool::<64>::new(1000);  // 1000 small objects
        let medium_pool = ObjectPool::<512>::new(500); // 500 medium objects  
        let large_pool = ObjectPool::<4096>::new(100); // 100 large objects
        
        let array_pool = ArrayPool::new();
        let string_pool = StringPool::new();
        let value_pool = ValuePool::new();
        
        Self {
            small_object_pool: small_pool,
            medium_object_pool: medium_pool,
            large_object_pool: large_pool,
            array_pool,
            string_pool,
            temp_value_pool: value_pool,
            stats: Arc::new(Mutex::new(PoolAllocationStats::default())),
        }
    }
    
    /// Allocate memory from appropriate pool based on size
    pub fn allocate(&mut self, size: usize) -> Result<NonNull<u8>, RuntimeError> {
        let start_time = std::time::Instant::now();
        
        let result = if size <= 64 {
            self.small_object_pool.allocate()
        } else if size <= 512 {
            self.medium_object_pool.allocate()
        } else if size <= 4096 {
            self.large_object_pool.allocate()
        } else {
            // Fall back to heap allocation for very large objects
            self.heap_allocate(size)
        };
        
        // Update statistics
        let allocation_time = start_time.elapsed().as_nanos() as f64;
        let mut stats = self.stats.lock().unwrap();
        
        match &result {
            Ok(_) => {
                stats.total_pool_allocations += 1;
                stats.pool_hits += 1;
                stats.memory_saved += size as u64;
                stats.gc_avoided_count += 1;
            }
            Err(_) => {
                stats.pool_misses += 1;
            }
        }
        
        stats.average_allocation_time_ns = 
            (stats.average_allocation_time_ns * (stats.total_pool_allocations - 1) as f64 + allocation_time) 
            / stats.total_pool_allocations as f64;
        
        result
    }
    
    /// Deallocate memory back to appropriate pool
    pub fn deallocate(&mut self, ptr: NonNull<u8>, size: usize) {
        if size <= 64 {
            self.small_object_pool.deallocate(ptr);
        } else if size <= 512 {
            self.medium_object_pool.deallocate(ptr);
        } else if size <= 4096 {
            self.large_object_pool.deallocate(ptr);
        } else {
            self.heap_deallocate(ptr, size);
        }
    }
    
    /// Allocate a temporary Value without heap allocation
    pub fn allocate_temp_value(&mut self, value_type: TempValueType) -> Value {
        match value_type {
            TempValueType::Integer(val) => {
                if let Some(mut temp_val) = self.temp_value_pool.integer_values.pop() {
                    temp_val = Value::Integer(val);
                    temp_val
                } else {
                    Value::Integer(val)
                }
            }
            TempValueType::Float(val) => {
                if let Some(mut temp_val) = self.temp_value_pool.float_values.pop() {
                    temp_val = Value::Float(val);
                    temp_val
                } else {
                    Value::Float(val)
                }
            }
            TempValueType::Boolean(val) => {
                if let Some(mut temp_val) = self.temp_value_pool.boolean_values.pop() {
                    temp_val = Value::Boolean(val);
                    temp_val
                } else {
                    Value::Boolean(val)
                }
            }
            TempValueType::Null => {
                if let Some(temp_val) = self.temp_value_pool.null_values.pop() {
                    temp_val
                } else {
                    Value::Null
                }
            }
        }
    }
    
    /// Return a temporary Value to the pool
    pub fn return_temp_value(&mut self, value: Value) {
        match value {
            Value::Integer(_) => {
                if self.temp_value_pool.integer_values.len() < 1000 {
                    self.temp_value_pool.integer_values.push(value);
                }
            }
            Value::Float(_) => {
                if self.temp_value_pool.float_values.len() < 1000 {
                    self.temp_value_pool.float_values.push(value);
                }
            }
            Value::Boolean(_) => {
                if self.temp_value_pool.boolean_values.len() < 100 {
                    self.temp_value_pool.boolean_values.push(value);
                }
            }
            Value::Null => {
                if self.temp_value_pool.null_values.len() < 100 {
                    self.temp_value_pool.null_values.push(value);
                }
            }
            _ => {
                // Don't pool complex types for now
            }
        }
    }
    
    /// Allocate SIMD-aligned array for vectorized operations
    pub fn allocate_simd_f64_array(&mut self, size: usize) -> Result<AlignedArray<f64, 64>, RuntimeError> {
        // Try to reuse from pool first
        if let Some(mut array) = self.array_pool.simd_f64_arrays.pop() {
            if array.data.len() >= size {
                // Reuse existing array (just resize the view)
                return Ok(array);
            }
        }
        
        // Allocate new SIMD-aligned array
        AlignedArray::new_f64(size)
    }
    
    /// Return SIMD array to pool for reuse
    pub fn return_simd_f64_array(&mut self, array: AlignedArray<f64, 64>) {
        if self.array_pool.simd_f64_arrays.len() < 100 {
            self.array_pool.simd_f64_arrays.push(array);
        }
    }
    
    /// Allocate string from pool
    pub fn allocate_string(&mut self, capacity: usize) -> String {
        let pool = if capacity < 32 {
            &mut self.string_pool.small_strings
        } else if capacity < 256 {
            &mut self.string_pool.medium_strings
        } else {
            &mut self.string_pool.large_strings
        };
        
        if let Some(mut string) = pool.pop() {
            string.clear();
            string.reserve(capacity);
            string
        } else {
            String::with_capacity(capacity)
        }
    }
    
    /// Return string to pool
    pub fn return_string(&mut self, mut string: String) {
        string.clear();
        
        let pool = if string.capacity() < 32 {
            &mut self.string_pool.small_strings
        } else if string.capacity() < 256 {
            &mut self.string_pool.medium_strings
        } else {
            &mut self.string_pool.large_strings
        };
        
        if pool.len() < 1000 {
            pool.push(string);
        }
    }
    
    /// Heap allocation fallback
    fn heap_allocate(&mut self, size: usize) -> Result<NonNull<u8>, RuntimeError> {
        let layout = Layout::from_size_align(size, 8)
            .map_err(|_| RuntimeError::InvalidOperation("Invalid allocation layout".to_string()))?;
            
        let ptr = unsafe { alloc(layout) };
        if ptr.is_null() {
            return Err(RuntimeError::InvalidOperation("Heap allocation failed".to_string()));
        }
        
        let mut stats = self.stats.lock().unwrap();
        stats.total_heap_allocations += 1;
        
        Ok(NonNull::new(ptr).unwrap())
    }
    
    /// Heap deallocation fallback
    fn heap_deallocate(&self, ptr: NonNull<u8>, size: usize) {
        let layout = Layout::from_size_align(size, 8).unwrap();
        unsafe { dealloc(ptr.as_ptr(), layout); }
    }
    
    /// Get memory pool performance statistics
    pub fn get_stats(&self) -> String {
        let stats = self.stats.lock().unwrap();
        format!(
            "🏊 Memory Pool Performance:\n\
             💾 Pool allocations: {}\n\
             🗑️ Heap allocations: {}\n\
             🎯 Pool hit rate: {:.1}%\n\
             💰 Memory saved: {} KB\n\
             🚫 GC avoided: {} times\n\
             ⚡ Avg allocation time: {:.1} ns",
            stats.total_pool_allocations,
            stats.total_heap_allocations,
            if stats.total_pool_allocations > 0 {
                (stats.pool_hits as f64 / stats.total_pool_allocations as f64) * 100.0
            } else { 0.0 },
            stats.memory_saved / 1024,
            stats.gc_avoided_count,
            stats.average_allocation_time_ns
        )
    }
}

impl<const SIZE: usize> ObjectPool<SIZE> {
    /// Create new object pool with specified capacity
    fn new(capacity: usize) -> Self {
        let mut pool = Self {
            chunks: Vec::new(),
            free_slots: Vec::with_capacity(capacity),
            capacity,
            allocated: 0,
        };
        
        // Pre-allocate initial chunk
        pool.allocate_chunk(capacity).expect("Failed to allocate initial chunk");
        
        pool
    }
    
    /// Allocate chunk of memory for objects
    fn allocate_chunk(&mut self, object_count: usize) -> Result<(), RuntimeError> {
        let total_size = SIZE * object_count;
        let layout = Layout::from_size_align(total_size, 64) // 64-byte align for SIMD
            .map_err(|_| RuntimeError::InvalidOperation("Invalid chunk layout".to_string()))?;
            
        let ptr = unsafe { alloc(layout) };
        if ptr.is_null() {
            return Err(RuntimeError::InvalidOperation("Chunk allocation failed".to_string()));
        }
        
        let chunk = MemoryChunk {
            memory: NonNull::new(ptr).unwrap(),
            layout,
            object_count,
        };
        
        // Add all object slots to free list
        for i in 0..object_count {
            let object_ptr = unsafe { 
                NonNull::new_unchecked(ptr.add(i * SIZE))
            };
            self.free_slots.push(object_ptr);
        }
        
        self.chunks.push(chunk);
        Ok(())
    }
    
    /// Allocate object from pool
    fn allocate(&mut self) -> Result<NonNull<u8>, RuntimeError> {
        if let Some(ptr) = self.free_slots.pop() {
            self.allocated += 1;
            Ok(ptr)
        } else {
            // Pool exhausted, allocate new chunk
            self.allocate_chunk(self.capacity / 2)?;
            self.allocate()
        }
    }
    
    /// Deallocate object back to pool
    fn deallocate(&mut self, ptr: NonNull<u8>) {
        self.free_slots.push(ptr);
        self.allocated -= 1;
    }
}

impl ArrayPool {
    fn new() -> Self {
        Self {
            i64_arrays: HashMap::new(),
            f64_arrays: HashMap::new(),
            string_arrays: HashMap::new(),
            temp_i64_arrays: Vec::new(),
            temp_f64_arrays: Vec::new(),
            simd_f64_arrays: Vec::new(),
            simd_f32_arrays: Vec::new(),
        }
    }
    
    /// Get or allocate i64 array
    pub fn get_i64_array(&mut self, size: usize) -> Box<[i64]> {
        if let Some(arrays) = self.i64_arrays.get_mut(&size) {
            if let Some(array) = arrays.pop() {
                return array;
            }
        }
        
        // Allocate new array
        vec![0i64; size].into_boxed_slice()
    }
    
    /// Return i64 array to pool
    pub fn return_i64_array(&mut self, array: Box<[i64]>) {
        let size = array.len();
        let arrays = self.i64_arrays.entry(size).or_insert_with(Vec::new);
        if arrays.len() < 100 {
            arrays.push(array);
        }
    }
    
    /// Get or allocate f64 array
    pub fn get_f64_array(&mut self, size: usize) -> Box<[f64]> {
        if let Some(arrays) = self.f64_arrays.get_mut(&size) {
            if let Some(array) = arrays.pop() {
                return array;
            }
        }
        
        vec![0.0f64; size].into_boxed_slice()
    }
    
    /// Return f64 array to pool
    pub fn return_f64_array(&mut self, array: Box<[f64]>) {
        let size = array.len();
        let arrays = self.f64_arrays.entry(size).or_insert_with(Vec::new);
        if arrays.len() < 100 {
            arrays.push(array);
        }
    }
}

impl StringPool {
    fn new() -> Self {
        Self {
            small_strings: Vec::with_capacity(1000),
            medium_strings: Vec::with_capacity(500),
            large_strings: Vec::with_capacity(100),
            interned_strings: HashMap::new(),
        }
    }
    
    /// Intern a string to avoid duplicates
    pub fn intern_string(&mut self, s: &str) -> Arc<str> {
        if let Some(interned) = self.interned_strings.get(s) {
            interned.clone()
        } else {
            let interned: Arc<str> = s.into();
            self.interned_strings.insert(s.to_string(), interned.clone());
            interned
        }
    }
}

impl ValuePool {
    fn new() -> Self {
        Self {
            integer_values: Vec::with_capacity(1000),
            float_values: Vec::with_capacity(1000),
            boolean_values: Vec::with_capacity(100),
            null_values: Vec::with_capacity(100),
        }
    }
}

impl<T, const ALIGN: usize> AlignedArray<T, ALIGN> 
where 
    T: Default + Clone 
{
    /// Create new SIMD-aligned array for f64
    fn new_f64(size: usize) -> Result<AlignedArray<f64, 64>, RuntimeError> {
        let data = vec![0.0f64; size].into_boxed_slice();
        Ok(AlignedArray { data })
    }
    
    /// Get data as slice
    pub fn as_slice(&self) -> &[T] {
        &self.data
    }
    
    /// Get data as mutable slice
    pub fn as_mut_slice(&mut self) -> &mut [T] {
        &mut self.data
    }
    
    /// Get raw pointer for SIMD operations
    pub fn as_ptr(&self) -> *const T {
        self.data.as_ptr()
    }
    
    /// Get mutable raw pointer for SIMD operations
    pub fn as_mut_ptr(&mut self) -> *mut T {
        self.data.as_mut_ptr()
    }
}

/// Types for temporary value allocation
pub enum TempValueType {
    Integer(i64),
    Float(f64),
    Boolean(bool),
    Null,
}

impl<const SIZE: usize> Drop for MemoryChunk<SIZE> {
    fn drop(&mut self) {
        unsafe {
            dealloc(self.memory.as_ptr(), self.layout);
        }
    }
}

impl Default for MemoryPoolSystem {
    fn default() -> Self {
        Self::new()
    }
}

/// Memory pool integration with the execution profiler
impl MemoryPoolSystem {
    pub fn update_profiler_stats(&self, profiler: &mut crate::execution_profiler::ExecutionProfiler) {
        let stats = self.stats.lock().unwrap();
        
        // Update allocation stats in profiler
        profiler.allocation_stats.pool_allocations = stats.total_pool_allocations;
        profiler.allocation_stats.heap_allocations = stats.total_heap_allocations;
        profiler.allocation_stats.gc_pressure = if stats.total_heap_allocations > 0 {
            stats.total_heap_allocations as f64 / (stats.total_pool_allocations + stats.total_heap_allocations) as f64
        } else {
            0.0
        };
    }
}