//! Efficient string handling with interning and dictionary encoding
//!
//! This module provides utilities for memory-efficient handling of string data,
//! which is common in ML datasets.

use std::collections::{HashMap, HashSet};
use std::sync::{Arc, Mutex};

use ml_data_core::buffer::Buffer;
use ml_data_core::error::Result as CoreResult;

/// A string interning cache for deduplication
pub struct StringCache {
    /// Set of unique strings
    strings: HashSet<String>,
    
    /// Estimated memory usage
    memory_usage: usize,
}

impl StringCache {
    /// Create a new string cache
    pub fn new() -> Self {
        Self {
            strings: HashSet::new(),
            memory_usage: 0,
        }
    }
    
    /// Create a new string cache with a pre-allocated capacity
    pub fn with_capacity(capacity: usize) -> Self {
        Self {
            strings: HashSet::with_capacity(capacity),
            memory_usage: 0,
        }
    }
    
    /// Intern a string, returning a reference to the cached version
    pub fn intern(&mut self, s: &str) -> &str {
        if self.strings.contains(s) {
            // String already exists in the cache, return reference
            self.strings.get(s).unwrap()
        } else {
            // Add the string to the cache
            let added_memory = s.len();
            self.memory_usage += added_memory;
            self.strings.insert(s.to_string());
            self.strings.get(s).unwrap()
        }
    }
    
    /// Check if a string is already in the cache
    pub fn contains(&self, s: &str) -> bool {
        self.strings.contains(s)
    }
    
    /// Get the number of unique strings in the cache
    pub fn len(&self) -> usize {
        self.strings.len()
    }
    
    /// Check if the cache is empty
    pub fn is_empty(&self) -> bool {
        self.strings.is_empty()
    }
    
    /// Clear the cache
    pub fn clear(&mut self) {
        self.strings.clear();
        self.memory_usage = 0;
    }
    
    /// Get the estimated memory usage of the cache
    pub fn memory_usage(&self) -> usize {
        self.memory_usage
    }
    
    /// Get all unique strings in the cache
    pub fn unique_strings(&self) -> impl Iterator<Item = &str> {
        self.strings.iter().map(|s| s.as_str())
    }
}

impl Default for StringCache {
    fn default() -> Self {
        Self::new()
    }
}

/// Thread-safe string cache
pub struct ThreadSafeStringCache {
    /// Inner cache protected by a mutex
    inner: Mutex<StringCache>,
}

impl ThreadSafeStringCache {
    /// Create a new thread-safe string cache
    pub fn new() -> Self {
        Self {
            inner: Mutex::new(StringCache::new()),
        }
    }
    
    /// Create a new thread-safe string cache with a pre-allocated capacity
    pub fn with_capacity(capacity: usize) -> Self {
        Self {
            inner: Mutex::new(StringCache::with_capacity(capacity)),
        }
    }
    
    /// Intern a string, returning a copy of the cached version
    pub fn intern(&self, s: &str) -> String {
        let mut cache = self.inner.lock().unwrap();
        cache.intern(s).to_string()
    }
    
    /// Check if a string is already in the cache
    pub fn contains(&self, s: &str) -> bool {
        let cache = self.inner.lock().unwrap();
        cache.contains(s)
    }
    
    /// Get the number of unique strings in the cache
    pub fn len(&self) -> usize {
        let cache = self.inner.lock().unwrap();
        cache.len()
    }
    
    /// Check if the cache is empty
    pub fn is_empty(&self) -> bool {
        let cache = self.inner.lock().unwrap();
        cache.is_empty()
    }
    
    /// Clear the cache
    pub fn clear(&self) {
        let mut cache = self.inner.lock().unwrap();
        cache.clear();
    }
    
    /// Get the estimated memory usage of the cache
    pub fn memory_usage(&self) -> usize {
        let cache = self.inner.lock().unwrap();
        cache.memory_usage()
    }
}

impl Default for ThreadSafeStringCache {
    fn default() -> Self {
        Self::new()
    }
}

/// Dictionary encoding for categorical string values
pub struct StringDictionary {
    /// Mapping from string values to dictionary indices
    value_to_index: HashMap<String, u32>,
    
    /// Mapping from dictionary indices to string values
    index_to_value: Vec<String>,
    
    /// Estimated memory usage
    memory_usage: usize,
}

impl StringDictionary {
    /// Create a new string dictionary
    pub fn new() -> Self {
        Self {
            value_to_index: HashMap::new(),
            index_to_value: Vec::new(),
            memory_usage: 0,
        }
    }
    
    /// Create a new string dictionary with a pre-allocated capacity
    pub fn with_capacity(capacity: usize) -> Self {
        Self {
            value_to_index: HashMap::with_capacity(capacity),
            index_to_value: Vec::with_capacity(capacity),
            memory_usage: 0,
        }
    }
    
    /// Get or insert a string value, returning its dictionary index
    pub fn get_or_insert(&mut self, value: &str) -> u32 {
        if let Some(&index) = self.value_to_index.get(value) {
            index
        } else {
            let index = self.index_to_value.len() as u32;
            let owned_value = value.to_string();
            self.memory_usage += owned_value.len();
            self.value_to_index.insert(owned_value.clone(), index);
            self.index_to_value.push(owned_value);
            index
        }
    }
    
    /// Check if a value exists in the dictionary
    pub fn contains(&self, value: &str) -> bool {
        self.value_to_index.contains_key(value)
    }
    
    /// Get the index for a string value
    pub fn get_index(&self, value: &str) -> Option<u32> {
        self.value_to_index.get(value).copied()
    }
    
    /// Get the string value for a dictionary index
    pub fn get_value(&self, index: u32) -> Option<&str> {
        self.index_to_value.get(index as usize).map(|s| s.as_str())
    }
    
    /// Get the number of entries in the dictionary
    pub fn len(&self) -> usize {
        self.index_to_value.len()
    }
    
    /// Check if the dictionary is empty
    pub fn is_empty(&self) -> bool {
        self.index_to_value.is_empty()
    }
    
    /// Get all values in the dictionary
    pub fn values(&self) -> impl Iterator<Item = &str> {
        self.index_to_value.iter().map(|s| s.as_str())
    }
    
    /// Get the estimated memory usage of the dictionary
    pub fn memory_usage(&self) -> usize {
        self.memory_usage
    }
    
    /// Clear the dictionary
    pub fn clear(&mut self) {
        self.value_to_index.clear();
        self.index_to_value.clear();
        self.memory_usage = 0;
    }
    
    /// Create indices buffer from string values
    pub fn encode_values(&mut self, values: &[&str]) -> CoreResult<Buffer> {
        let mut indices = Vec::with_capacity(values.len());
        
        for &value in values {
            let index = self.get_or_insert(value);
            indices.push(index);
        }
        
        Buffer::from_slice(&indices)
    }
    
    /// Decode indices back to string values
    pub fn decode_indices(&self, indices: &[u32]) -> Vec<String> {
        indices
            .iter()
            .filter_map(|&idx| self.get_value(idx).map(|s| s.to_string()))
            .collect()
    }
}

impl Default for StringDictionary {
    fn default() -> Self {
        Self::new()
    }
}

/// Thread-safe string dictionary
pub struct ThreadSafeStringDictionary {
    /// Inner dictionary protected by a mutex
    inner: Mutex<StringDictionary>,
}

impl ThreadSafeStringDictionary {
    /// Create a new thread-safe string dictionary
    pub fn new() -> Self {
        Self {
            inner: Mutex::new(StringDictionary::new()),
        }
    }
    
    /// Create a new thread-safe string dictionary with a pre-allocated capacity
    pub fn with_capacity(capacity: usize) -> Self {
        Self {
            inner: Mutex::new(StringDictionary::with_capacity(capacity)),
        }
    }
    
    /// Get or insert a string value, returning its dictionary index
    pub fn get_or_insert(&self, value: &str) -> u32 {
        let mut dict = self.inner.lock().unwrap();
        dict.get_or_insert(value)
    }
    
    /// Check if a value exists in the dictionary
    pub fn contains(&self, value: &str) -> bool {
        let dict = self.inner.lock().unwrap();
        dict.contains(value)
    }
    
    /// Get the index for a string value
    pub fn get_index(&self, value: &str) -> Option<u32> {
        let dict = self.inner.lock().unwrap();
        dict.get_index(value)
    }
    
    /// Get the string value for a dictionary index
    pub fn get_value(&self, index: u32) -> Option<String> {
        let dict = self.inner.lock().unwrap();
        dict.get_value(index).map(|s| s.to_string())
    }
    
    /// Get the number of entries in the dictionary
    pub fn len(&self) -> usize {
        let dict = self.inner.lock().unwrap();
        dict.len()
    }
    
    /// Check if the dictionary is empty
    pub fn is_empty(&self) -> bool {
        let dict = self.inner.lock().unwrap();
        dict.is_empty()
    }
    
    /// Get the estimated memory usage of the dictionary
    pub fn memory_usage(&self) -> usize {
        let dict = self.inner.lock().unwrap();
        dict.memory_usage()
    }
    
    /// Create a snapshot of the dictionary values
    pub fn values_snapshot(&self) -> Vec<String> {
        let dict = self.inner.lock().unwrap();
        dict.values().map(|s| s.to_string()).collect()
    }
}

impl Default for ThreadSafeStringDictionary {
    fn default() -> Self {
        Self::new()
    }
}