//! Common utilities and implementations for data sources

use std::path::{Path, PathBuf};
use std::sync::Arc;

use ml_data_core::{RecordBatch, Schema, DataType, Field};
use ml_data_core::error::Result as CoreResult;

use crate::error::{Error, Result};
use crate::string_cache::StringDictionary;

/// Options for file readers
#[derive(Debug, Clone)]
pub struct ReaderOptions {
    /// Use memory mapping for file access
    pub use_memory_mapping: bool,
    
    /// Batch size for reading records
    pub batch_size: usize,
    
    /// Buffer size for I/O operations
    pub buffer_size: usize,
    
    /// Whether to infer schema from data
    pub infer_schema: bool,
    
    /// Maximum number of rows to read for schema inference
    pub schema_inference_rows: usize,
    
    /// Whether to use dictionary encoding for strings
    pub use_dictionary_encoding: bool,
    
    /// Dictionary encoding threshold (minimum occurrences for dictionary encoding)
    pub dictionary_encoding_threshold: usize,
    
    /// Column indices to include (null = all columns)
    pub projection: Option<Vec<usize>>,
    
    /// Column names to include (null = all columns)
    pub projection_by_name: Option<Vec<String>>,
    
    /// Number of threads to use for parsing
    pub num_threads: Option<usize>,
    
    /// Whether to validate UTF-8 strings
    pub validate_utf8: bool,
}

impl Default for ReaderOptions {
    fn default() -> Self {
        Self {
            use_memory_mapping: true,
            batch_size: 8192,
            buffer_size: 64 * 1024, // 64KB
            infer_schema: true,
            schema_inference_rows: 1000,
            use_dictionary_encoding: true,
            dictionary_encoding_threshold: 10,
            projection: None,
            projection_by_name: None,
            num_threads: None,
            validate_utf8: true,
        }
    }
}

/// File format detection utilities
pub struct FileFormat;

impl FileFormat {
    /// Detect the format of a file based on its extension
    pub fn detect_from_path(path: &Path) -> Option<&'static str> {
        let extension = path.extension()?.to_str()?.to_lowercase();
        
        match extension.as_str() {
            "csv" => Some("csv"),
            "tsv" => Some("csv"), // TSV is just a CSV with tabs
            "parquet" => Some("parquet"),
            "json" => Some("json"),
            "jsonl" | "ndjson" => Some("jsonl"),
            "avro" => Some("avro"),
            "tfrecord" => Some("tfrecord"),
            "arrow" | "arrows" | "ipc" => Some("arrow"),
            // Images
            "jpg" | "jpeg" | "png" | "gif" | "bmp" | "tiff" => Some("image"),
            // Compressed formats
            "gz" | "gzip" => {
                // Look at the file stem
                let stem = path.file_stem()?.to_str()?;
                if let Some(pos) = stem.rfind('.') {
                    let inner_ext = &stem[pos+1..];
                    match inner_ext {
                        "csv" => Some("csv.gz"),
                        "tsv" => Some("csv.gz"),
                        "json" => Some("json.gz"),
                        "parquet" => Some("parquet.gz"),
                        _ => Some("gz"),
                    }
                } else {
                    Some("gz")
                }
            }
            // Default to binary
            _ => None,
        }
    }
    
    /// Detect format by inspecting file content
    pub fn detect_from_content(data: &[u8], path: Option<&Path>) -> Option<&'static str> {
        // Try to detect format from path first
        if let Some(path) = path {
            if let Some(format) = Self::detect_from_path(path) {
                return Some(format);
            }
        }
        
        // Check file signatures
        if data.len() >= 4 {
            // Parquet files start with "PAR1"
            if &data[0..4] == b"PAR1" {
                return Some("parquet");
            }
            
            // Arrow files start with "ARROW1"
            if data.len() >= 6 && &data[0..6] == b"ARROW1" {
                return Some("arrow");
            }
            
            // Gzip files start with 0x1F 0x8B
            if data.len() >= 2 && data[0] == 0x1F && data[1] == 0x8B {
                return Some("gz");
            }
            
            // Check for JSON
            if data[0] == b'{' || data[0] == b'[' {
                // Peek at the first few bytes to see if it looks like JSON
                return Some("json");
            }
            
            // Check for CSV
            if data.iter().take(32).any(|&b| b == b',') {
                return Some("csv");
            }
        }
        
        // Default to binary if we can't detect
        Some("binary")
    }
}

/// Schema inference utilities
pub struct SchemaInference;

impl SchemaInference {
    /// Infer schema from string-based CSV/TSV data
    pub fn infer_from_string_records(records: &[Vec<String>], header: Option<Vec<String>>) -> Result<Schema> {
        if records.is_empty() {
            return Err(Error::InvalidArgument("Cannot infer schema from empty data".into()));
        }
        
        let column_count = records[0].len();
        
        // Check if all records have the same number of columns
        if records.iter().any(|r| r.len() != column_count) {
            return Err(Error::Schema("Records have inconsistent column counts".into()));
        }
        
        // Generate column names
        let column_names = match header {
            Some(header) => {
                if header.len() != column_count {
                    return Err(Error::Schema(format!(
                        "Header has {} columns but data has {} columns",
                        header.len(), column_count
                    )));
                }
                header
            },
            None => (0..column_count).map(|i| format!("column_{}", i)).collect(),
        };
        
        let mut fields = Vec::with_capacity(column_count);
        
        // Process each column
        for col_idx in 0..column_count {
            let column_name = &column_names[col_idx];
            
            // Collect values for this column
            let column_values: Vec<&str> = records.iter()
                .map(|r| r[col_idx].as_str())
                .collect();
            
            // Infer data type for this column
            let data_type = Self::infer_data_type(&column_values);
            
            // Create field
            fields.push(Field::new(column_name, data_type, true));
        }
        
        Ok(Schema::new(fields))
    }
    
    /// Infer data type for a column of string values
    pub fn infer_data_type(values: &[&str]) -> DataType {
        // Count empty values
        let non_empty_values: Vec<&str> = values.iter()
            .filter(|&&s| !s.is_empty())
            .copied()
            .collect();
            
        if non_empty_values.is_empty() {
            // If all values are empty, default to string
            return DataType::String;
        }
        
        // Try to parse as integer
        let all_int = non_empty_values.iter().all(|&s| s.parse::<i64>().is_ok());
        if all_int {
            return DataType::Int64;
        }
        
        // Try to parse as float
        let all_float = non_empty_values.iter().all(|&s| s.parse::<f64>().is_ok());
        if all_float {
            return DataType::Float64;
        }
        
        // Try to parse as boolean
        let all_bool = non_empty_values.iter().all(|&s| {
            let s = s.to_lowercase();
            s == "true" || s == "false" || s == "1" || s == "0" || s == "yes" || s == "no"
        });
        if all_bool {
            return DataType::Boolean;
        }
        
        // Default to string
        DataType::String
    }
}

/// Base implementation for string-based record readers
pub struct StringRecordReader {
    /// Schema for the data
    schema: Arc<Schema>,
    
    /// Dictionary encoders for string columns
    string_dictionaries: Vec<Option<StringDictionary>>,
    
    /// Reader options
    options: ReaderOptions,
}

impl StringRecordReader {
    /// Create a new string record reader
    pub fn new(schema: Arc<Schema>, options: ReaderOptions) -> Self {
        // Initialize string dictionaries for string columns
        let string_dictionaries = if options.use_dictionary_encoding {
            schema.fields().iter()
                .map(|field| {
                    match field.data_type() {
                        DataType::String => Some(StringDictionary::new()),
                        _ => None,
                    }
                })
                .collect()
        } else {
            vec![None; schema.len()]
        };
        
        Self {
            schema,
            string_dictionaries,
            options,
        }
    }
    
    /// Convert string records to a RecordBatch
    pub fn create_batch_from_string_records(&mut self, records: Vec<Vec<String>>) -> CoreResult<RecordBatch> {
        // Implementation would convert strings to typed columns
        // For demonstration, we'll create a blank record batch with the right schema
        RecordBatch::new_empty(self.schema.clone())
    }
    
    /// Get the schema
    pub fn schema(&self) -> &Arc<Schema> {
        &self.schema
    }
    
    /// Get the options
    pub fn options(&self) -> &ReaderOptions {
        &self.options
    }
    
    /// Get a mutable reference to the options
    pub fn options_mut(&mut self) -> &mut ReaderOptions {
        &mut self.options
    }
    
    /// Get the string dictionaries
    pub fn string_dictionaries(&self) -> &[Option<StringDictionary>] {
        &self.string_dictionaries
    }
    
    /// Get estimated memory usage
    pub fn memory_usage(&self) -> usize {
        // Base size plus dictionary sizes
        let dict_size = self.string_dictionaries.iter()
            .filter_map(|d| d.as_ref().map(|d| d.memory_usage()))
            .sum::<usize>();
            
        // Add fixed overhead
        dict_size + 1024
    }
}