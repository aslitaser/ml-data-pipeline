//! Text data utilities with efficient string handling
//!
//! This module provides utilities for efficient text processing,
//! including string interning, dictionary encoding, and specialized
//! text storage structures.

use std::fs::File;
use std::io::{BufRead, BufReader};
use std::path::{Path, PathBuf};
use std::sync::Arc;

use ml_data_core::{RecordBatch, RecordBatchSource, Schema, Field, DataType};
use ml_data_core::error::{Error as CoreError, Result as CoreResult};
use ml_data_core::buffer::Buffer;

use crate::error::{Error, Result};
use crate::{DataSource, FileDataSource};
use crate::string_cache::{StringCache, StringDictionary, ThreadSafeStringCache};

/// Options for text reading
#[derive(Debug, Clone)]
pub struct TextReaderOptions {
    /// Whether to use string interning
    pub use_string_interning: bool,
    
    /// Whether to use dictionary encoding
    pub use_dictionary_encoding: bool,
    
    /// Minimum frequency for dictionary encoding
    pub dictionary_min_frequency: usize,
    
    /// Maximum dictionary size
    pub max_dictionary_size: Option<usize>,
    
    /// Whether to validate UTF-8
    pub validate_utf8: bool,
    
    /// Whether to trim whitespace
    pub trim_whitespace: bool,
    
    /// Whether to skip empty lines
    pub skip_empty_lines: bool,
    
    /// Whether to strip BOM (byte order mark)
    pub strip_bom: bool,
    
    /// Batch size for reading
    pub batch_size: usize,
    
    /// Maximum line length
    pub max_line_length: Option<usize>,
}

impl Default for TextReaderOptions {
    fn default() -> Self {
        Self {
            use_string_interning: true,
            use_dictionary_encoding: true,
            dictionary_min_frequency: 10,
            max_dictionary_size: Some(10000),
            validate_utf8: true,
            trim_whitespace: false,
            skip_empty_lines: true,
            strip_bom: true,
            batch_size: 10000,
            max_line_length: Some(1024 * 1024), // 1MB
        }
    }
}

/// A rope-like data structure for efficient string manipulation
/// 
/// A rope is a tree-based data structure that represents a string as a sequence of smaller strings.
/// It allows for efficient manipulation of large strings by avoiding frequent reallocation.
pub struct StringRope {
    /// Total length of the rope
    length: usize,
    
    /// Rope structure (either a leaf or a node)
    structure: RopeStructure,
}

enum RopeStructure {
    /// Leaf node with actual string content
    Leaf(String),
    
    /// Internal node with left and right children
    Node {
        /// Left child
        left: Box<StringRope>,
        
        /// Right child
        right: Box<StringRope>,
    },
}

impl StringRope {
    /// Create a new empty rope
    pub fn new() -> Self {
        Self {
            length: 0,
            structure: RopeStructure::Leaf(String::new()),
        }
    }
    
    /// Create a rope from a string
    pub fn from_string(s: String) -> Self {
        let length = s.len();
        Self {
            length,
            structure: RopeStructure::Leaf(s),
        }
    }
    
    /// Get the length of the rope
    pub fn len(&self) -> usize {
        self.length
    }
    
    /// Check if the rope is empty
    pub fn is_empty(&self) -> bool {
        self.length == 0
    }
    
    /// Concatenate two ropes
    pub fn concat(self, other: Self) -> Self {
        // Only create a node if both ropes are non-empty
        if self.is_empty() {
            return other;
        }
        
        if other.is_empty() {
            return self;
        }
        
        Self {
            length: self.length + other.length,
            structure: RopeStructure::Node {
                left: Box::new(self),
                right: Box::new(other),
            },
        }
    }
    
    /// Convert the rope to a string
    pub fn to_string(&self) -> String {
        let mut result = String::with_capacity(self.length);
        self.append_to_string(&mut result);
        result
    }
    
    /// Append the rope's content to a string
    fn append_to_string(&self, s: &mut String) {
        match &self.structure {
            RopeStructure::Leaf(leaf) => {
                s.push_str(leaf);
            },
            RopeStructure::Node { left, right } => {
                left.append_to_string(s);
                right.append_to_string(s);
            },
        }
    }
    
    /// Split the rope at the given index
    pub fn split(&self, index: usize) -> (Self, Self) {
        if index >= self.length {
            return (self.clone(), Self::new());
        }
        
        if index == 0 {
            return (Self::new(), self.clone());
        }
        
        match &self.structure {
            RopeStructure::Leaf(leaf) => {
                let (left, right) = leaf.split_at(index);
                (
                    Self::from_string(left.to_string()),
                    Self::from_string(right.to_string()),
                )
            },
            RopeStructure::Node { left, right } => {
                if index < left.length {
                    // Split inside the left subtree
                    let (left_left, left_right) = left.split(index);
                    (
                        left_left,
                        left_right.concat(*right.clone()),
                    )
                } else {
                    // Split inside the right subtree
                    let (right_left, right_right) = 
                        right.split(index - left.length);
                    (
                        left.clone().concat(right_left),
                        right_right,
                    )
                }
            },
        }
    }
    
    /// Insert a string at the given index
    pub fn insert(&self, index: usize, s: &str) -> Self {
        let (left, right) = self.split(index);
        left.concat(Self::from_string(s.to_string()))
            .concat(right)
    }
    
    /// Delete a range from the rope
    pub fn delete(&self, start: usize, end: usize) -> Self {
        if start >= end || start >= self.length {
            return self.clone();
        }
        
        let (left, temp) = self.split(start);
        let (_, right) = temp.split(end - start);
        
        left.concat(right)
    }
    
    /// Get a character at the given index
    pub fn char_at(&self, index: usize) -> Option<char> {
        if index >= self.length {
            return None;
        }
        
        match &self.structure {
            RopeStructure::Leaf(leaf) => {
                leaf.chars().nth(index)
            },
            RopeStructure::Node { left, right } => {
                if index < left.length {
                    left.char_at(index)
                } else {
                    right.char_at(index - left.length)
                }
            },
        }
    }
    
    /// Get a substring from the rope
    pub fn substring(&self, start: usize, end: usize) -> String {
        if start >= end || start >= self.length {
            return String::new();
        }
        
        let real_end = end.min(self.length);
        let mut result = String::with_capacity(real_end - start);
        
        self.append_substring_to_string(&mut result, start, real_end);
        
        result
    }
    
    /// Append a substring to a string
    fn append_substring_to_string(&self, s: &mut String, start: usize, end: usize) {
        if start >= end || start >= self.length {
            return;
        }
        
        let real_end = end.min(self.length);
        
        match &self.structure {
            RopeStructure::Leaf(leaf) => {
                let start_char = leaf.char_indices()
                    .nth(start)
                    .map(|(i, _)| i)
                    .unwrap_or(leaf.len());
                    
                let end_char = leaf.char_indices()
                    .nth(real_end)
                    .map(|(i, _)| i)
                    .unwrap_or(leaf.len());
                    
                s.push_str(&leaf[start_char..end_char]);
            },
            RopeStructure::Node { left, right } => {
                if start < left.length {
                    let left_end = real_end.min(left.length);
                    left.append_substring_to_string(s, start, left_end);
                }
                
                if real_end > left.length {
                    let right_start = start.saturating_sub(left.length);
                    let right_end = real_end - left.length;
                    right.append_substring_to_string(s, right_start, right_end);
                }
            },
        }
    }
}

impl Clone for StringRope {
    fn clone(&self) -> Self {
        Self {
            length: self.length,
            structure: match &self.structure {
                RopeStructure::Leaf(leaf) => {
                    RopeStructure::Leaf(leaf.clone())
                },
                RopeStructure::Node { left, right } => {
                    RopeStructure::Node {
                        left: Box::new(left.clone()),
                        right: Box::new(right.clone()),
                    }
                },
            },
        }
    }
}

impl Default for StringRope {
    fn default() -> Self {
        Self::new()
    }
}

/// A text file reader
pub struct TextFileReader {
    /// File path
    path: PathBuf,
    
    /// Reader options
    options: TextReaderOptions,
    
    /// File reader
    reader: BufReader<File>,
    
    /// Schema for the data
    schema: Arc<Schema>,
    
    /// Current line number
    current_line: usize,
    
    /// Whether the reader is exhausted
    exhausted: bool,
    
    /// String cache for interning
    string_cache: Option<ThreadSafeStringCache>,
    
    /// String dictionary for encoding
    string_dictionary: Option<StringDictionary>,
}

impl TextFileReader {
    /// Create a new text file reader
    pub fn new<P: AsRef<Path>>(
        path: P,
        options: TextReaderOptions,
    ) -> Result<Self> {
        let path = path.as_ref().to_path_buf();
        
        // Open the file
        let file = File::open(&path)?;
        let reader = BufReader::new(file);
        
        // Create schema
        let schema = Schema::new(vec![
            Field::new("line_number", DataType::Int64, false),
            Field::new("text", DataType::String, false),
        ]);
        
        // Create string cache if using interning
        let string_cache = if options.use_string_interning {
            Some(ThreadSafeStringCache::new())
        } else {
            None
        };
        
        // Create string dictionary if using dictionary encoding
        let string_dictionary = if options.use_dictionary_encoding {
            Some(StringDictionary::new())
        } else {
            None
        };
        
        Ok(Self {
            path,
            options,
            reader,
            schema: Arc::new(schema),
            current_line: 0,
            exhausted: false,
            string_cache,
            string_dictionary,
        })
    }
    
    /// Read next batch of text lines
    pub fn read_next_batch(&mut self) -> Result<Option<RecordBatch>> {
        if self.exhausted {
            return Ok(None);
        }
        
        let mut lines = Vec::with_capacity(self.options.batch_size);
        let mut line_numbers = Vec::with_capacity(self.options.batch_size);
        
        let mut buffer = String::new();
        let mut lines_read = 0;
        
        while lines_read < self.options.batch_size {
            buffer.clear();
            
            let bytes_read = self.reader.read_line(&mut buffer)?;
            
            if bytes_read == 0 {
                // End of file
                self.exhausted = true;
                break;
            }
            
            // Update line number
            self.current_line += 1;
            
            // Process line
            if self.options.strip_bom && self.current_line == 1 {
                // Strip BOM if present
                if buffer.starts_with('\u{FEFF}') {
                    buffer.drain(..3);
                }
            }
            
            if self.options.trim_whitespace {
                buffer = buffer.trim().to_string();
            }
            
            if self.options.skip_empty_lines && buffer.is_empty() {
                continue;
            }
            
            // Check line length
            if let Some(max_length) = self.options.max_line_length {
                if buffer.len() > max_length {
                    return Err(Error::InvalidArgument(format!(
                        "Line {} exceeds maximum length: {} > {}",
                        self.current_line, buffer.len(), max_length
                    )));
                }
            }
            
            // Apply string interning
            let text = if let Some(cache) = &self.string_cache {
                cache.intern(&buffer)
            } else {
                buffer.clone()
            };
            
            // Apply dictionary encoding
            if let Some(dict) = &mut self.string_dictionary {
                dict.get_or_insert(&text);
            }
            
            lines.push(text);
            line_numbers.push(self.current_line as i64);
            
            lines_read += 1;
        }
        
        if lines.is_empty() {
            return Ok(None);
        }
        
        // Create record batch
        let mut columns = Vec::with_capacity(2);
        
        // Line number column
        let line_number_buffer = Buffer::from_slice(&line_numbers)?;
        let line_number_column = ml_data_core::Column::new(
            Field::new("line_number", DataType::Int64, false),
            line_number_buffer,
        );
        
        columns.push(line_number_column);
        
        // Text column
        if let Some(dict) = &self.string_dictionary {
            // Use dictionary encoding
            let indices: Vec<u32> = lines.iter()
                .map(|line| dict.get_index(line).unwrap_or(0))
                .collect();
                
            let indices_buffer = Buffer::from_slice(&indices)?;
            
            let text_column = ml_data_core::Column::new_dictionary(
                Field::new("text", DataType::String, false),
                indices_buffer,
                dict.len(),
            );
            
            columns.push(text_column);
        } else {
            // Store strings directly
            let offsets = compute_string_offsets(&lines);
            let data = concatenate_strings(&lines);
            
            let offsets_buffer = Buffer::from_slice(&offsets)?;
            let data_buffer = Buffer::from_slice(data.as_bytes())?;
            
            let text_column = ml_data_core::Column::new_with_buffers(
                Field::new("text", DataType::String, false),
                vec![offsets_buffer, data_buffer],
            );
            
            columns.push(text_column);
        }
        
        let batch = RecordBatch::new(self.schema.clone(), columns)?;
        
        Ok(Some(batch))
    }
}

/// Compute string offsets for variable-length string storage
fn compute_string_offsets(strings: &[String]) -> Vec<u32> {
    let mut offsets = Vec::with_capacity(strings.len() + 1);
    let mut current_offset = 0;
    
    offsets.push(current_offset);
    
    for s in strings {
        current_offset += s.len() as u32;
        offsets.push(current_offset);
    }
    
    offsets
}

/// Concatenate strings into a single contiguous buffer
fn concatenate_strings(strings: &[String]) -> String {
    let total_len: usize = strings.iter().map(|s| s.len()).sum();
    let mut result = String::with_capacity(total_len);
    
    for s in strings {
        result.push_str(s);
    }
    
    result
}

impl DataSource for TextFileReader {
    fn schema(&self) -> &Arc<Schema> {
        &self.schema
    }
    
    fn reset(&mut self) -> Result<()> {
        self.reader.seek(std::io::SeekFrom::Start(0))?;
        self.current_line = 0;
        self.exhausted = false;
        Ok(())
    }
    
    fn estimated_rows(&self) -> Option<usize> {
        None // Can't easily estimate lines in a text file
    }
    
    fn memory_usage(&self) -> usize {
        let cache_size = self.string_cache
            .as_ref()
            .map(|c| c.memory_usage())
            .unwrap_or(0);
            
        let dict_size = self.string_dictionary
            .as_ref()
            .map(|d| d.memory_usage())
            .unwrap_or(0);
            
        cache_size + dict_size + 1024 * 1024 // Buffer size
    }
}

impl FileDataSource for TextFileReader {
    fn path(&self) -> &Path {
        &self.path
    }
    
    fn file_size(&self) -> Result<u64> {
        Ok(std::fs::metadata(&self.path)?.len())
    }
    
    fn supports_zero_copy(&self) -> bool {
        false // Text processing requires parsing
    }
    
    fn memory_map(&mut self) -> Result<()> {
        // Not implemented for text files
        Ok(())
    }
}

impl RecordBatchSource for TextFileReader {
    fn schema(&self) -> Arc<Schema> {
        self.schema.clone()
    }
    
    fn next_batch(&mut self, max_batch_size: usize) -> CoreResult<Option<RecordBatch>> {
        // Adjust batch size if needed
        let original_batch_size = self.options.batch_size;
        if max_batch_size > 0 && max_batch_size != self.options.batch_size {
            self.options.batch_size = max_batch_size;
        }
        
        let result = self.read_next_batch()
            .map_err(|e| {
                CoreError::Io(std::io::Error::new(
                    std::io::ErrorKind::Other,
                    format!("Text error: {}", e),
                ))
            });
            
        // Restore original batch size
        self.options.batch_size = original_batch_size;
        
        result
    }
    
    fn row_count_hint(&self) -> Option<usize> {
        self.estimated_rows()
    }
    
    fn memory_usage(&self) -> usize {
        self.memory_usage()
    }
    
    fn reset(&mut self) -> CoreResult<()> {
        self.reset().map_err(|e| {
            CoreError::Io(std::io::Error::new(
                std::io::ErrorKind::Other,
                format!("Text error: {}", e),
            ))
        })
    }
}