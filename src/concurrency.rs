use std::collections::HashMap;
use std::sync::{Arc, Mutex};
use std::thread::{self, JoinHandle};
use std::time::Duration;
use crossbeam::channel::{self, Receiver, Sender};
use parking_lot::{Mutex as ParkingMutex, RwLock as ParkingRwLock};
use rayon::prelude::*;
use crate::ast::{Statement, Expression, Block};
use crate::interpreter::{Value, RuntimeError};

/// Concurrency System for ScaffoldLang
#[derive(Debug)]
pub struct ConcurrencyManager {
    pub threads: HashMap<String, ThreadHandle>,
    pub channels: HashMap<String, ChannelPair>,
    pub mutexes: HashMap<String, Arc<ParkingMutex<Value>>>,
    pub rwlocks: HashMap<String, Arc<ParkingRwLock<Value>>>,
    pub thread_pools: HashMap<String, rayon::ThreadPool>,
}

#[derive(Debug)]
pub struct ThreadHandle {
    pub name: String,
    pub handle: Option<JoinHandle<Result<Value, RuntimeError>>>,
    pub is_running: bool,
}

#[derive(Debug)]
pub struct ChannelPair {
    pub sender: Sender<Value>,
    pub receiver: Receiver<Value>,
}

impl ConcurrencyManager {
    pub fn new() -> Self {
        Self {
            threads: HashMap::new(),
            channels: HashMap::new(),
            mutexes: HashMap::new(),
            rwlocks: HashMap::new(),
            thread_pools: HashMap::new(),
        }
    }

    /// Create and start a new thread
    pub fn spawn_thread<F>(&mut self, name: String, func: F) -> Result<(), ConcurrencyError>
    where
        F: FnOnce() -> Result<Value, RuntimeError> + Send + 'static,
    {
        let handle = thread::spawn(func);
        
        let thread_handle = ThreadHandle {
            name: name.clone(),
            handle: Some(handle),
            is_running: true,
        };

        self.threads.insert(name, thread_handle);
        Ok(())
    }

    /// Join a thread and get its result
    pub fn join_thread(&mut self, name: &str) -> Result<Value, ConcurrencyError> {
        if let Some(mut thread_handle) = self.threads.remove(name) {
            if let Some(handle) = thread_handle.handle.take() {
                match handle.join() {
                    Ok(result) => match result {
                        Ok(value) => Ok(value),
                        Err(runtime_error) => Err(ConcurrencyError::ThreadError(format!("Thread '{}' failed: {:?}", name, runtime_error))),
                    },
                    Err(_) => Err(ConcurrencyError::ThreadError(format!("Thread '{}' panicked", name))),
                }
            } else {
                Err(ConcurrencyError::ThreadError(format!("Thread '{}' handle not available", name)))
            }
        } else {
            Err(ConcurrencyError::ThreadNotFound(name.to_string()))
        }
    }

    /// Create a new channel
    pub fn create_channel(&mut self, name: String, capacity: Option<usize>) -> Result<(), ConcurrencyError> {
        let (sender, receiver) = if let Some(cap) = capacity {
            channel::bounded(cap)
        } else {
            channel::unbounded()
        };

        let channel_pair = ChannelPair { sender, receiver };
        self.channels.insert(name, channel_pair);
        Ok(())
    }

    /// Send value through channel
    pub fn send_to_channel(&self, channel_name: &str, value: Value) -> Result<(), ConcurrencyError> {
        if let Some(channel) = self.channels.get(channel_name) {
            channel.sender.send(value)
                .map_err(|_| ConcurrencyError::ChannelError(format!("Failed to send to channel '{}'", channel_name)))?;
            Ok(())
        } else {
            Err(ConcurrencyError::ChannelNotFound(channel_name.to_string()))
        }
    }

    /// Receive value from channel (blocking)
    pub fn receive_from_channel(&self, channel_name: &str) -> Result<Value, ConcurrencyError> {
        if let Some(channel) = self.channels.get(channel_name) {
            channel.receiver.recv()
                .map_err(|_| ConcurrencyError::ChannelError(format!("Failed to receive from channel '{}'", channel_name)))
        } else {
            Err(ConcurrencyError::ChannelNotFound(channel_name.to_string()))
        }
    }

    /// Create a mutex
    pub fn create_mutex(&mut self, name: String, initial_value: Value) -> Result<(), ConcurrencyError> {
        let mutex = Arc::new(ParkingMutex::new(initial_value));
        self.mutexes.insert(name, mutex);
        Ok(())
    }

    /// Lock mutex and execute function
    pub fn with_mutex_lock<F, R>(&self, mutex_name: &str, func: F) -> Result<R, ConcurrencyError>
    where
        F: FnOnce(&mut Value) -> R,
    {
        if let Some(mutex) = self.mutexes.get(mutex_name) {
            let mut guard = mutex.lock();
            Ok(func(&mut *guard))
        } else {
            Err(ConcurrencyError::MutexNotFound(mutex_name.to_string()))
        }
    }

    /// Create a thread pool
    pub fn create_thread_pool(&mut self, name: String, num_threads: usize) -> Result<(), ConcurrencyError> {
        let pool = rayon::ThreadPoolBuilder::new()
            .num_threads(num_threads)
            .build()
            .map_err(|e| ConcurrencyError::ThreadPoolError(format!("Failed to create thread pool '{}': {}", name, e)))?;
        
        self.thread_pools.insert(name, pool);
        Ok(())
    }

    /// Sleep current thread
    pub fn sleep(&self, duration_ms: u64) {
        thread::sleep(Duration::from_millis(duration_ms));
    }

    /// Yield current thread
    pub fn yield_now(&self) {
        thread::yield_now();
    }

    /// Get number of available CPU cores
    pub fn num_cpus(&self) -> usize {
        num_cpus::get()
    }

    /// Check if thread is running
    pub fn is_thread_running(&self, name: &str) -> bool {
        self.threads.get(name).map(|t| t.is_running).unwrap_or(false)
    }

    /// List all threads
    pub fn list_threads(&self) -> Vec<&str> {
        self.threads.keys().map(|s| s.as_str()).collect()
    }

    /// List all channels
    pub fn list_channels(&self) -> Vec<&str> {
        self.channels.keys().map(|s| s.as_str()).collect()
    }

    /// List all mutexes
    pub fn list_mutexes(&self) -> Vec<&str> {
        self.mutexes.keys().map(|s| s.as_str()).collect()
    }
}

/// Concurrency errors
#[derive(Debug)]
pub enum ConcurrencyError {
    ThreadError(String),
    ThreadNotFound(String),
    ChannelError(String),
    ChannelNotFound(String),
    MutexNotFound(String),
    RwLockNotFound(String),
    ThreadPoolError(String),
    ThreadPoolNotFound(String),
    AsyncError(String),
    AsyncTaskNotFound(String),
    DeadlockDetected(String),
    RaceCondition(String),
}

impl std::fmt::Display for ConcurrencyError {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        match self {
            ConcurrencyError::ThreadError(msg) => write!(f, "Thread error: {}", msg),
            ConcurrencyError::ThreadNotFound(name) => write!(f, "Thread not found: {}", name),
            ConcurrencyError::ChannelError(msg) => write!(f, "Channel error: {}", msg),
            ConcurrencyError::ChannelNotFound(name) => write!(f, "Channel not found: {}", name),
            ConcurrencyError::MutexNotFound(name) => write!(f, "Mutex not found: {}", name),
            ConcurrencyError::RwLockNotFound(name) => write!(f, "RwLock not found: {}", name),
            ConcurrencyError::ThreadPoolError(msg) => write!(f, "Thread pool error: {}", msg),
            ConcurrencyError::ThreadPoolNotFound(name) => write!(f, "Thread pool not found: {}", name),
            ConcurrencyError::AsyncError(msg) => write!(f, "Async error: {}", msg),
            ConcurrencyError::AsyncTaskNotFound(name) => write!(f, "Async task not found: {}", name),
            ConcurrencyError::DeadlockDetected(msg) => write!(f, "Deadlock detected: {}", msg),
            ConcurrencyError::RaceCondition(msg) => write!(f, "Race condition: {}", msg),
        }
    }
}

impl std::error::Error for ConcurrencyError {}

/// Async runtime placeholder (simplified)
pub struct AsyncRuntime {
    pub tasks: HashMap<String, String>,
}

impl AsyncRuntime {
    pub fn new() -> Self {
        Self {
            tasks: HashMap::new(),
        }
    }
}
