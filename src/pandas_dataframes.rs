/// Phase 3: Custom Pandas-style DataFrame Operations
/// Target: Provide Python Pandas compatibility with ultra-fast performance
/// 
/// This system implements high-performance data manipulation and analysis operations
/// that rival Pandas performance while maintaining the familiar API.

use std::collections::HashMap;
use std::fmt;
use crate::numpy_arrays_simple::{Array64, NumPySystem, ArrayDType};
use crate::interpreter::{Value, RuntimeError};
// Removed SIMD engine import for simplicity
use crate::memory_pools::MemoryPoolSystem;

/// High-performance DataFrame with columnar storage
#[derive(Debug, Clone)]
pub struct DataFrame {
    /// Column data stored as arrays for vectorization
    columns: HashMap<String, Column>,
    
    /// Column names in order
    column_order: Vec<String>,
    
    /// Index (row labels)
    index: Vec<String>,
    
    /// Number of rows
    nrows: usize,
    
    /// Number of columns
    ncols: usize,
}

/// Column data with type information
#[derive(Debug, Clone)]
pub enum Column {
    Float64(Array64),
    Int64(Vec<i64>),
    String(Vec<String>),
    Boolean(Vec<bool>),
    Categorical(Vec<usize>, Vec<String>), // indices into categories
}

/// Series - a single column with index
#[derive(Debug, Clone)]
pub struct Series {
    data: Column,
    index: Vec<String>,
    name: String,
}

/// Pandas-compatible data manipulation system
pub struct PandasSystem {
    /// NumPy system for array operations
    numpy_system: NumPySystem,
    
    /// Memory pool for efficient allocation
    memory_pool: MemoryPoolSystem,
    
    /// Performance statistics
    stats: DataFrameOperationStats,
}

/// Performance statistics for DataFrame operations
#[derive(Debug, Default)]
pub struct DataFrameOperationStats {
    pub total_operations: u64,
    pub vectorized_operations: u64,
    pub groupby_operations: u64,
    pub join_operations: u64,
    pub filter_operations: u64,
    pub sort_operations: u64,
    pub memory_operations: u64,
    pub total_speedup: f64,
}

/// Query result for filtering operations
#[derive(Debug)]
pub struct QueryResult {
    matching_rows: Vec<usize>,
    result_dataframe: DataFrame,
}

/// GroupBy operation result
#[derive(Debug)]
pub struct GroupBy {
    groups: HashMap<String, Vec<usize>>,
    original_dataframe: DataFrame,
}

/// Aggregation functions
#[derive(Debug, Clone)]
pub enum AggregateFunction {
    Sum,
    Mean,
    Count,
    Min,
    Max,
    Std,
    Var,
}

impl DataFrame {
    /// Create new empty DataFrame
    pub fn new() -> Self {
        Self {
            columns: HashMap::new(),
            column_order: Vec::new(),
            index: Vec::new(),
            nrows: 0,
            ncols: 0,
        }
    }
    
    /// Create DataFrame from dictionary of columns
    pub fn from_dict(data: HashMap<String, Column>) -> Result<Self, RuntimeError> {
        if data.is_empty() {
            return Ok(Self::new());
        }
        
        // Check that all columns have the same length
        let first_len = data.values().next().unwrap().len();
        for (name, column) in &data {
            if column.len() != first_len {
                return Err(RuntimeError::InvalidOperation(
                    format!("Column '{}' has different length", name)
                ));
            }
        }
        
        let nrows = first_len;
        let ncols = data.len();
        let column_order: Vec<String> = data.keys().cloned().collect();
        let index: Vec<String> = (0..nrows).map(|i| i.to_string()).collect();
        
        Ok(Self {
            columns: data,
            column_order,
            index,
            nrows,
            ncols,
        })
    }
    
    /// Get column by name
    pub fn get_column(&self, name: &str) -> Option<&Column> {
        self.columns.get(name)
    }
    
    /// Add new column
    pub fn add_column(&mut self, name: String, column: Column) -> Result<(), RuntimeError> {
        if !self.columns.is_empty() && column.len() != self.nrows {
            return Err(RuntimeError::InvalidOperation(
                "Column length doesn't match DataFrame".to_string()
            ));
        }
        
        if self.columns.is_empty() {
            self.nrows = column.len();
            self.index = (0..self.nrows).map(|i| i.to_string()).collect();
        }
        
        if !self.columns.contains_key(&name) {
            self.column_order.push(name.clone());
            self.ncols += 1;
        }
        
        self.columns.insert(name, column);
        Ok(())
    }
    
    /// Remove column
    pub fn drop_column(&mut self, name: &str) -> Result<(), RuntimeError> {
        if !self.columns.contains_key(name) {
            return Err(RuntimeError::InvalidOperation(
                format!("Column '{}' not found", name)
            ));
        }
        
        self.columns.remove(name);
        self.column_order.retain(|x| x != name);
        self.ncols -= 1;
        
        Ok(())
    }
    
    /// Get DataFrame shape
    pub fn shape(&self) -> (usize, usize) {
        (self.nrows, self.ncols)
    }
    
    /// Select specific columns
    pub fn select(&self, column_names: &[&str]) -> Result<DataFrame, RuntimeError> {
        let mut new_columns = HashMap::new();
        let mut new_column_order = Vec::new();
        
        for &name in column_names {
            if let Some(column) = self.columns.get(name) {
                new_columns.insert(name.to_string(), column.clone());
                new_column_order.push(name.to_string());
            } else {
                return Err(RuntimeError::InvalidOperation(
                    format!("Column '{}' not found", name)
                ));
            }
        }
        
        let ncols = new_column_order.len();
        Ok(DataFrame {
            columns: new_columns,
            column_order: new_column_order,
            index: self.index.clone(),
            nrows: self.nrows,
            ncols,
        })
    }
    
    /// Filter rows based on condition
    pub fn filter<F>(&self, column_name: &str, predicate: F) -> Result<DataFrame, RuntimeError>
    where
        F: Fn(&Column, usize) -> bool,
    {
        let column = self.columns.get(column_name)
            .ok_or_else(|| RuntimeError::InvalidOperation(format!("Column '{}' not found", column_name)))?;
        
        let mut matching_indices = Vec::new();
        for i in 0..self.nrows {
            if predicate(column, i) {
                matching_indices.push(i);
            }
        }
        
        self.select_rows(&matching_indices)
    }
    
    /// Select specific rows by indices
    fn select_rows(&self, indices: &[usize]) -> Result<DataFrame, RuntimeError> {
        let mut new_columns = HashMap::new();
        
        for (name, column) in &self.columns {
            let new_column = match column {
                Column::Float64(arr) => {
                    let mut new_data = Vec::new();
                    for &idx in indices {
                        if idx < arr.size() {
                            new_data.push(arr.as_slice()[idx]);
                        }
                    }
                    Column::Float64(Array64::from_data(new_data, vec![indices.len()], ArrayDType::Float64)?)
                }
                Column::Int64(data) => {
                    let new_data: Vec<i64> = indices.iter().map(|&idx| data[idx]).collect();
                    Column::Int64(new_data)
                }
                Column::String(data) => {
                    let new_data: Vec<String> = indices.iter().map(|&idx| data[idx].clone()).collect();
                    Column::String(new_data)
                }
                Column::Boolean(data) => {
                    let new_data: Vec<bool> = indices.iter().map(|&idx| data[idx]).collect();
                    Column::Boolean(new_data)
                }
                Column::Categorical(data, categories) => {
                    let new_data: Vec<usize> = indices.iter().map(|&idx| data[idx]).collect();
                    Column::Categorical(new_data, categories.clone())
                }
            };
            
            new_columns.insert(name.clone(), new_column);
        }
        
        let new_index: Vec<String> = indices.iter().map(|&idx| self.index[idx].clone()).collect();
        
        Ok(DataFrame {
            columns: new_columns,
            column_order: self.column_order.clone(),
            index: new_index,
            nrows: indices.len(),
            ncols: self.ncols,
        })
    }
    
    /// Group by column values
    pub fn groupby(&self, column_name: &str) -> Result<GroupBy, RuntimeError> {
        let column = self.columns.get(column_name)
            .ok_or_else(|| RuntimeError::InvalidOperation(format!("Column '{}' not found", column_name)))?;
        
        let mut groups: HashMap<String, Vec<usize>> = HashMap::new();
        
        match column {
            Column::String(data) => {
                for (i, value) in data.iter().enumerate() {
                    groups.entry(value.clone()).or_insert_with(Vec::new).push(i);
                }
            }
            Column::Int64(data) => {
                for (i, &value) in data.iter().enumerate() {
                    groups.entry(value.to_string()).or_insert_with(Vec::new).push(i);
                }
            }
            Column::Float64(arr) => {
                for (i, &value) in arr.as_slice().iter().enumerate() {
                    groups.entry(format!("{:.3}", value)).or_insert_with(Vec::new).push(i);
                }
            }
            Column::Boolean(data) => {
                for (i, &value) in data.iter().enumerate() {
                    groups.entry(value.to_string()).or_insert_with(Vec::new).push(i);
                }
            }
            Column::Categorical(data, categories) => {
                for (i, &cat_idx) in data.iter().enumerate() {
                    let category = &categories[cat_idx];
                    groups.entry(category.clone()).or_insert_with(Vec::new).push(i);
                }
            }
        }
        
        Ok(GroupBy {
            groups,
            original_dataframe: self.clone(),
        })
    }
    
    /// Sort by column values
    pub fn sort_by(&self, column_name: &str, ascending: bool) -> Result<DataFrame, RuntimeError> {
        let column = self.columns.get(column_name)
            .ok_or_else(|| RuntimeError::InvalidOperation(format!("Column '{}' not found", column_name)))?;
        
        let mut indices: Vec<usize> = (0..self.nrows).collect();
        
        match column {
            Column::Float64(arr) => {
                indices.sort_by(|&a, &b| {
                    let cmp = arr.as_slice()[a].partial_cmp(&arr.as_slice()[b]).unwrap_or(std::cmp::Ordering::Equal);
                    if ascending { cmp } else { cmp.reverse() }
                });
            }
            Column::Int64(data) => {
                indices.sort_by(|&a, &b| {
                    let cmp = data[a].cmp(&data[b]);
                    if ascending { cmp } else { cmp.reverse() }
                });
            }
            Column::String(data) => {
                indices.sort_by(|&a, &b| {
                    let cmp = data[a].cmp(&data[b]);
                    if ascending { cmp } else { cmp.reverse() }
                });
            }
            Column::Boolean(data) => {
                indices.sort_by(|&a, &b| {
                    let cmp = data[a].cmp(&data[b]);
                    if ascending { cmp } else { cmp.reverse() }
                });
            }
            Column::Categorical(data, categories) => {
                indices.sort_by(|&a, &b| {
                    let cat_a = &categories[data[a]];
                    let cat_b = &categories[data[b]];
                    let cmp = cat_a.cmp(cat_b);
                    if ascending { cmp } else { cmp.reverse() }
                });
            }
        }
        
        self.select_rows(&indices)
    }
    
    /// Compute basic statistics
    pub fn describe(&self) -> Result<DataFrame, RuntimeError> {
        let mut stats_data = HashMap::new();
        let stats_names = vec!["count", "mean", "std", "min", "max"];
        
        for stat_name in &stats_names {
            stats_data.insert(stat_name.to_string(), Column::String(Vec::new()));
        }
        
        for column_name in &self.column_order {
            if let Some(column) = self.columns.get(column_name) {
                match column {
                    Column::Float64(arr) => {
                        let data = arr.as_slice();
                        let count = data.len();
                        let mean = data.iter().sum::<f64>() / count as f64;
                        let variance = data.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / count as f64;
                        let std = variance.sqrt();
                        let min = data.iter().fold(f64::INFINITY, |a, &b| a.min(b));
                        let max = data.iter().fold(f64::NEG_INFINITY, |a, &b| a.max(b));
                        
                        // Add stats for this column
                        if let Some(Column::String(count_col)) = stats_data.get_mut("count") {
                            count_col.push(count.to_string());
                        }
                        if let Some(Column::String(mean_col)) = stats_data.get_mut("mean") {
                            mean_col.push(format!("{:.6}", mean));
                        }
                        if let Some(Column::String(std_col)) = stats_data.get_mut("std") {
                            std_col.push(format!("{:.6}", std));
                        }
                        if let Some(Column::String(min_col)) = stats_data.get_mut("min") {
                            min_col.push(format!("{:.6}", min));
                        }
                        if let Some(Column::String(max_col)) = stats_data.get_mut("max") {
                            max_col.push(format!("{:.6}", max));
                        }
                    }
                    Column::Int64(data) => {
                        let count = data.len();
                        let mean = data.iter().sum::<i64>() as f64 / count as f64;
                        let variance = data.iter().map(|&x| (x as f64 - mean).powi(2)).sum::<f64>() / count as f64;
                        let std = variance.sqrt();
                        let min = *data.iter().min().unwrap_or(&0);
                        let max = *data.iter().max().unwrap_or(&0);
                        
                        if let Some(Column::String(count_col)) = stats_data.get_mut("count") {
                            count_col.push(count.to_string());
                        }
                        if let Some(Column::String(mean_col)) = stats_data.get_mut("mean") {
                            mean_col.push(format!("{:.6}", mean));
                        }
                        if let Some(Column::String(std_col)) = stats_data.get_mut("std") {
                            std_col.push(format!("{:.6}", std));
                        }
                        if let Some(Column::String(min_col)) = stats_data.get_mut("min") {
                            min_col.push(min.to_string());
                        }
                        if let Some(Column::String(max_col)) = stats_data.get_mut("max") {
                            max_col.push(max.to_string());
                        }
                    }
                    _ => {
                        // For non-numeric columns, just add count
                        let count = column.len();
                        for stat_name in &stats_names {
                            if let Some(Column::String(stat_col)) = stats_data.get_mut(*stat_name) {
                                if *stat_name == "count" {
                                    stat_col.push(count.to_string());
                                } else {
                                    stat_col.push("N/A".to_string());
                                }
                            }
                        }
                    }
                }
            }
        }
        
        DataFrame::from_dict(stats_data)
    }
    
    /// Join with another DataFrame
    pub fn join(&self, other: &DataFrame, on: &str, how: JoinType) -> Result<DataFrame, RuntimeError> {
        let left_column = self.columns.get(on)
            .ok_or_else(|| RuntimeError::InvalidOperation(format!("Column '{}' not found in left DataFrame", on)))?;
        let right_column = other.columns.get(on)
            .ok_or_else(|| RuntimeError::InvalidOperation(format!("Column '{}' not found in right DataFrame", on)))?;
        
        // Create index maps for joining
        let mut left_index_map: HashMap<String, Vec<usize>> = HashMap::new();
        let mut right_index_map: HashMap<String, Vec<usize>> = HashMap::new();
        
        // Build index maps
        self.build_index_map(left_column, &mut left_index_map);
        other.build_index_map(right_column, &mut right_index_map);
        
        // Perform join based on type
        let (left_indices, right_indices) = match how {
            JoinType::Inner => self.inner_join_indices(&left_index_map, &right_index_map),
            JoinType::Left => self.left_join_indices(&left_index_map, &right_index_map),
            JoinType::Right => self.right_join_indices(&left_index_map, &right_index_map),
            JoinType::Outer => self.outer_join_indices(&left_index_map, &right_index_map),
        };
        
        // Build result DataFrame
        self.build_joined_dataframe(other, &left_indices, &right_indices, on)
    }
    
    fn build_index_map(&self, column: &Column, index_map: &mut HashMap<String, Vec<usize>>) {
        match column {
            Column::String(data) => {
                for (i, value) in data.iter().enumerate() {
                    index_map.entry(value.clone()).or_insert_with(Vec::new).push(i);
                }
            }
            Column::Int64(data) => {
                for (i, &value) in data.iter().enumerate() {
                    index_map.entry(value.to_string()).or_insert_with(Vec::new).push(i);
                }
            }
            Column::Float64(arr) => {
                for (i, &value) in arr.as_slice().iter().enumerate() {
                    index_map.entry(format!("{:.6}", value)).or_insert_with(Vec::new).push(i);
                }
            }
            _ => {} // Handle other types as needed
        }
    }
    
    fn inner_join_indices(
        &self,
        left_map: &HashMap<String, Vec<usize>>,
        right_map: &HashMap<String, Vec<usize>>,
    ) -> (Vec<usize>, Vec<usize>) {
        let mut left_indices = Vec::new();
        let mut right_indices = Vec::new();
        
        for (key, left_idxs) in left_map {
            if let Some(right_idxs) = right_map.get(key) {
                for &left_idx in left_idxs {
                    for &right_idx in right_idxs {
                        left_indices.push(left_idx);
                        right_indices.push(right_idx);
                    }
                }
            }
        }
        
        (left_indices, right_indices)
    }
    
    fn left_join_indices(
        &self,
        left_map: &HashMap<String, Vec<usize>>,
        right_map: &HashMap<String, Vec<usize>>,
    ) -> (Vec<usize>, Vec<usize>) {
        let mut left_indices = Vec::new();
        let mut right_indices = Vec::new();
        
        for (key, left_idxs) in left_map {
            if let Some(right_idxs) = right_map.get(key) {
                for &left_idx in left_idxs {
                    for &right_idx in right_idxs {
                        left_indices.push(left_idx);
                        right_indices.push(right_idx);
                    }
                }
            } else {
                // Left join: include left rows even if no match
                for &left_idx in left_idxs {
                    left_indices.push(left_idx);
                    right_indices.push(usize::MAX); // Marker for null values
                }
            }
        }
        
        (left_indices, right_indices)
    }
    
    fn right_join_indices(
        &self,
        left_map: &HashMap<String, Vec<usize>>,
        right_map: &HashMap<String, Vec<usize>>,
    ) -> (Vec<usize>, Vec<usize>) {
        // Right join is left join with sides swapped
        let (right_idxs, left_idxs) = self.left_join_indices(right_map, left_map);
        (left_idxs, right_idxs)
    }
    
    fn outer_join_indices(
        &self,
        left_map: &HashMap<String, Vec<usize>>,
        right_map: &HashMap<String, Vec<usize>>,
    ) -> (Vec<usize>, Vec<usize>) {
        let mut left_indices = Vec::new();
        let mut right_indices = Vec::new();
        let mut processed_right_keys = std::collections::HashSet::new();
        
        // Process left side
        for (key, left_idxs) in left_map {
            if let Some(right_idxs) = right_map.get(key) {
                for &left_idx in left_idxs {
                    for &right_idx in right_idxs {
                        left_indices.push(left_idx);
                        right_indices.push(right_idx);
                    }
                }
                processed_right_keys.insert(key);
            } else {
                for &left_idx in left_idxs {
                    left_indices.push(left_idx);
                    right_indices.push(usize::MAX);
                }
            }
        }
        
        // Process unmatched right side
        for (key, right_idxs) in right_map {
            if !processed_right_keys.contains(key) {
                for &right_idx in right_idxs {
                    left_indices.push(usize::MAX);
                    right_indices.push(right_idx);
                }
            }
        }
        
        (left_indices, right_indices)
    }
    
    fn build_joined_dataframe(
        &self,
        other: &DataFrame,
        left_indices: &[usize],
        right_indices: &[usize],
        join_column: &str,
    ) -> Result<DataFrame, RuntimeError> {
        let mut result_columns = HashMap::new();
        let mut result_column_order = Vec::new();
        
        // Add columns from left DataFrame
        for column_name in &self.column_order {
            let new_column = self.select_column_by_indices(column_name, left_indices)?;
            result_columns.insert(column_name.clone(), new_column);
            result_column_order.push(column_name.clone());
        }
        
        // Add columns from right DataFrame (excluding join column)
        for column_name in &other.column_order {
            if column_name != join_column {
                let right_column_name = format!("{}_right", column_name);
                let new_column = other.select_column_by_indices(column_name, right_indices)?;
                result_columns.insert(right_column_name.clone(), new_column);
                result_column_order.push(right_column_name);
            }
        }
        
        let new_index: Vec<String> = (0..left_indices.len()).map(|i| i.to_string()).collect();
        
        let ncols = result_column_order.len();
        Ok(DataFrame {
            columns: result_columns,
            column_order: result_column_order,
            index: new_index,
            nrows: left_indices.len(),
            ncols,
        })
    }
    
    fn select_column_by_indices(&self, column_name: &str, indices: &[usize]) -> Result<Column, RuntimeError> {
        let column = self.columns.get(column_name)
            .ok_or_else(|| RuntimeError::InvalidOperation(format!("Column '{}' not found", column_name)))?;
        
        match column {
            Column::Float64(arr) => {
                let mut new_data = Vec::new();
                for &idx in indices {
                    if idx == usize::MAX {
                        new_data.push(f64::NAN); // Null value
                    } else {
                        new_data.push(arr.as_slice()[idx]);
                    }
                }
                Ok(Column::Float64(Array64::from_data(new_data, vec![indices.len()], ArrayDType::Float64)?))
            }
            Column::Int64(data) => {
                let mut new_data = Vec::new();
                for &idx in indices {
                    if idx == usize::MAX {
                        new_data.push(0); // Default null value
                    } else {
                        new_data.push(data[idx]);
                    }
                }
                Ok(Column::Int64(new_data))
            }
            Column::String(data) => {
                let mut new_data = Vec::new();
                for &idx in indices {
                    if idx == usize::MAX {
                        new_data.push("".to_string()); // Default null value
                    } else {
                        new_data.push(data[idx].clone());
                    }
                }
                Ok(Column::String(new_data))
            }
            Column::Boolean(data) => {
                let mut new_data = Vec::new();
                for &idx in indices {
                    if idx == usize::MAX {
                        new_data.push(false); // Default null value
                    } else {
                        new_data.push(data[idx]);
                    }
                }
                Ok(Column::Boolean(new_data))
            }
            Column::Categorical(data, categories) => {
                let mut new_data = Vec::new();
                for &idx in indices {
                    if idx == usize::MAX {
                        new_data.push(0); // Default to first category
                    } else {
                        new_data.push(data[idx]);
                    }
                }
                Ok(Column::Categorical(new_data, categories.clone()))
            }
        }
    }
}

/// Join types for DataFrame operations
#[derive(Debug, Clone)]
pub enum JoinType {
    Inner,
    Left,
    Right,
    Outer,
}

impl Column {
    fn len(&self) -> usize {
        match self {
            Column::Float64(arr) => arr.size(),
            Column::Int64(data) => data.len(),
            Column::String(data) => data.len(),
            Column::Boolean(data) => data.len(),
            Column::Categorical(data, _) => data.len(),
        }
    }
}

impl GroupBy {
    /// Apply aggregation function to grouped data
    pub fn agg(&self, column_name: &str, func: AggregateFunction) -> Result<DataFrame, RuntimeError> {
        let column = self.original_dataframe.columns.get(column_name)
            .ok_or_else(|| RuntimeError::InvalidOperation(format!("Column '{}' not found", column_name)))?;
        
        let mut result_data = HashMap::new();
        let mut group_names = Vec::new();
        let mut aggregated_values = Vec::new();
        
        for (group_name, indices) in &self.groups {
            let aggregated_value = match column {
                Column::Float64(arr) => {
                    let values: Vec<f64> = indices.iter().map(|&i| arr.as_slice()[i]).collect();
                    self.apply_aggregate_f64(&values, &func)
                }
                Column::Int64(data) => {
                    let values: Vec<i64> = indices.iter().map(|&i| data[i]).collect();
                    self.apply_aggregate_i64(&values, &func)
                }
                _ => return Err(RuntimeError::InvalidOperation("Aggregation not supported for this column type".to_string())),
            };
            
            group_names.push(group_name.clone());
            aggregated_values.push(aggregated_value);
        }
        
        result_data.insert("group".to_string(), Column::String(group_names));
        let aggregated_len = aggregated_values.len();
        result_data.insert(format!("{}_{:?}", column_name, func).to_lowercase(), Column::Float64(
            Array64::from_data(aggregated_values, vec![aggregated_len], ArrayDType::Float64)?
        ));
        
        DataFrame::from_dict(result_data)
    }
    
    fn apply_aggregate_f64(&self, values: &[f64], func: &AggregateFunction) -> f64 {
        match func {
            AggregateFunction::Sum => values.iter().sum(),
            AggregateFunction::Mean => values.iter().sum::<f64>() / values.len() as f64,
            AggregateFunction::Count => values.len() as f64,
            AggregateFunction::Min => values.iter().fold(f64::INFINITY, |a, &b| a.min(b)),
            AggregateFunction::Max => values.iter().fold(f64::NEG_INFINITY, |a, &b| a.max(b)),
            AggregateFunction::Std => {
                let mean = values.iter().sum::<f64>() / values.len() as f64;
                let variance = values.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / values.len() as f64;
                variance.sqrt()
            }
            AggregateFunction::Var => {
                let mean = values.iter().sum::<f64>() / values.len() as f64;
                values.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / values.len() as f64
            }
        }
    }
    
    fn apply_aggregate_i64(&self, values: &[i64], func: &AggregateFunction) -> f64 {
        match func {
            AggregateFunction::Sum => values.iter().sum::<i64>() as f64,
            AggregateFunction::Mean => values.iter().sum::<i64>() as f64 / values.len() as f64,
            AggregateFunction::Count => values.len() as f64,
            AggregateFunction::Min => *values.iter().min().unwrap_or(&0) as f64,
            AggregateFunction::Max => *values.iter().max().unwrap_or(&0) as f64,
            AggregateFunction::Std => {
                let mean = values.iter().sum::<i64>() as f64 / values.len() as f64;
                let variance = values.iter().map(|&x| (x as f64 - mean).powi(2)).sum::<f64>() / values.len() as f64;
                variance.sqrt()
            }
            AggregateFunction::Var => {
                let mean = values.iter().sum::<i64>() as f64 / values.len() as f64;
                values.iter().map(|&x| (x as f64 - mean).powi(2)).sum::<f64>() / values.len() as f64
            }
        }
    }
}

impl PandasSystem {
    pub fn new() -> Self {
        println!("📊 Initializing Pandas-compatible DataFrame System");
        
        Self {
            numpy_system: NumPySystem::new(),
            memory_pool: MemoryPoolSystem::new(),
            stats: DataFrameOperationStats::default(),
        }
    }
    
    /// Create DataFrame from CSV-like data
    pub fn read_csv(&mut self, data: &str) -> Result<DataFrame, RuntimeError> {
        let lines: Vec<&str> = data.trim().split('\n').collect();
        if lines.is_empty() {
            return Ok(DataFrame::new());
        }
        
        // Parse header
        let headers: Vec<&str> = lines[0].split(',').map(|s| s.trim()).collect();
        let mut columns: HashMap<String, Vec<String>> = HashMap::new();
        
        for header in &headers {
            columns.insert(header.to_string(), Vec::new());
        }
        
        // Parse data rows
        for line in &lines[1..] {
            let values: Vec<&str> = line.split(',').map(|s| s.trim()).collect();
            
            for (i, &value) in values.iter().enumerate() {
                if let Some(header) = headers.get(i) {
                    if let Some(column_data) = columns.get_mut(*header) {
                        column_data.push(value.to_string());
                    }
                }
            }
        }
        
        // Convert to appropriate column types
        let mut typed_columns = HashMap::new();
        
        for (name, string_data) in columns {
            let column = self.infer_column_type(string_data)?;
            typed_columns.insert(name, column);
        }
        
        self.stats.total_operations += 1;
        
        DataFrame::from_dict(typed_columns)
    }
    
    /// Infer column type from string data
    fn infer_column_type(&self, data: Vec<String>) -> Result<Column, RuntimeError> {
        // Try to parse as float
        if let Ok(float_data) = data.iter()
            .map(|s| s.parse::<f64>())
            .collect::<Result<Vec<f64>, _>>() {
            return Ok(Column::Float64(Array64::from_data(float_data, vec![data.len()], ArrayDType::Float64)?));
        }
        
        // Try to parse as integer
        if let Ok(int_data) = data.iter()
            .map(|s| s.parse::<i64>())
            .collect::<Result<Vec<i64>, _>>() {
            return Ok(Column::Int64(int_data));
        }
        
        // Try to parse as boolean
        if data.iter().all(|s| s == "true" || s == "false" || s == "True" || s == "False") {
            let bool_data: Vec<bool> = data.iter()
                .map(|s| s.to_lowercase() == "true")
                .collect();
            return Ok(Column::Boolean(bool_data));
        }
        
        // Default to string
        Ok(Column::String(data))
    }
    
    /// Get performance statistics
    pub fn get_performance_stats(&self) -> String {
        format!(
            "📊 Pandas DataFrame Performance:\n\
             📈 Total operations: {}\n\
             🚀 Vectorized operations: {}\n\
             👥 GroupBy operations: {}\n\
             🔗 Join operations: {}\n\
             🔍 Filter operations: {}\n\
             📑 Sort operations: {}\n\
             💾 Memory operations: {}\n\
             ⚡ Total speedup: {:.1}x",
            self.stats.total_operations,
            self.stats.vectorized_operations,
            self.stats.groupby_operations,
            self.stats.join_operations,
            self.stats.filter_operations,
            self.stats.sort_operations,
            self.stats.memory_operations,
            self.stats.total_speedup
        )
    }
}

impl fmt::Display for DataFrame {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        // Display header
        write!(f, "{:>8}", "")?;
        for column_name in &self.column_order {
            write!(f, "{:>12}", column_name)?;
        }
        writeln!(f)?;
        
        // Display data (first 10 rows)
        let display_rows = std::cmp::min(self.nrows, 10);
        
        for i in 0..display_rows {
            write!(f, "{:>8}", self.index[i])?;
            
            for column_name in &self.column_order {
                if let Some(column) = self.columns.get(column_name) {
                    match column {
                        Column::Float64(arr) => {
                            write!(f, "{:>12.3}", arr.as_slice()[i])?;
                        }
                        Column::Int64(data) => {
                            write!(f, "{:>12}", data[i])?;
                        }
                        Column::String(data) => {
                            let display_str = if data[i].len() > 10 {
                                format!("{}...", &data[i][..7])
                            } else {
                                data[i].clone()
                            };
                            write!(f, "{:>12}", display_str)?;
                        }
                        Column::Boolean(data) => {
                            write!(f, "{:>12}", data[i])?;
                        }
                        Column::Categorical(data, categories) => {
                            write!(f, "{:>12}", categories[data[i]])?;
                        }
                    }
                } else {
                    write!(f, "{:>12}", "NaN")?;
                }
            }
            writeln!(f)?;
        }
        
        if self.nrows > 10 {
            writeln!(f, "... ({} more rows)", self.nrows - 10)?;
        }
        
        writeln!(f, "\nShape: ({}, {})", self.nrows, self.ncols)?;
        
        Ok(())
    }
}

impl Default for DataFrame {
    fn default() -> Self {
        Self::new()
    }
}

impl Default for PandasSystem {
    fn default() -> Self {
        Self::new()
    }
}