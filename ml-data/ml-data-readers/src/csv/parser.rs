//! CSV parser for converting string records to typed columns

use std::sync::Arc;

use ml_data_core::{RecordBatch, Schema, Field, DataType, Column};
use ml_data_core::buffer::Buffer;

use crate::error::{Error, Result};
use crate::string_cache::StringDictionary;

/// CSV parser that converts string records to typed columns
pub struct CsvParser {
    /// Schema for the CSV data
    schema: Arc<Schema>,
}

impl CsvParser {
    /// Create a new CSV parser
    pub fn new(schema: Arc<Schema>) -> Self {
        Self { schema }
    }
    
    /// Create string dictionaries for string columns
    pub fn create_string_dictionaries(&self, use_dictionary: bool) -> Vec<Option<StringDictionary>> {
        if !use_dictionary {
            return vec![None; self.schema.len()];
        }
        
        self.schema.fields().iter()
            .map(|field| {
                match field.data_type() {
                    DataType::String => Some(StringDictionary::new()),
                    _ => None,
                }
            })
            .collect()
    }
    
    /// Parse a batch of string records into a RecordBatch
    pub fn parse_batch(
        &self,
        records: Vec<Vec<String>>,
        string_dictionaries: &mut [Option<StringDictionary>],
        use_dictionary: bool,
    ) -> Result<RecordBatch> {
        if records.is_empty() {
            return Ok(RecordBatch::new_empty(self.schema.clone()));
        }
        
        let num_rows = records.len();
        let num_cols = self.schema.len();
        
        // Check if all records have the expected number of columns
        for (i, record) in records.iter().enumerate() {
            if record.len() != num_cols {
                return Err(Error::Format(format!(
                    "Record at row {} has {} columns, expected {}",
                    i, record.len(), num_cols
                )));
            }
        }
        
        // Parse each column
        let mut columns = Vec::with_capacity(num_cols);
        
        for col_idx in 0..num_cols {
            let field = self.schema.field(col_idx);
            let column = self.parse_column(
                &records, 
                col_idx, 
                field,
                string_dictionaries.get_mut(col_idx).unwrap(),
                use_dictionary,
            )?;
            
            columns.push(column);
        }
        
        // Create record batch
        RecordBatch::new(self.schema.clone(), columns).map_err(Error::Core)
    }
    
    /// Parse a single column from string records
    fn parse_column(
        &self,
        records: &[Vec<String>],
        col_idx: usize,
        field: &Field,
        string_dictionary: &mut Option<StringDictionary>,
        use_dictionary: bool,
    ) -> Result<Column> {
        let num_rows = records.len();
        
        // Extract column values
        let string_values: Vec<&str> = records.iter()
            .map(|record| record[col_idx].as_str())
            .collect();
            
        // Parse based on data type
        match field.data_type() {
            DataType::Int8 => {
                let values: Result<Vec<i8>> = string_values.iter()
                    .map(|&s| s.parse::<i8>().map_err(|_| Error::Format(
                        format!("Failed to parse '{}' as i8", s)
                    )))
                    .collect();
                    
                let values = values?;
                let buffer = Buffer::from_slice(&values)?;
                
                Ok(Column::new(field.clone(), buffer))
            },
            DataType::Int16 => {
                let values: Result<Vec<i16>> = string_values.iter()
                    .map(|&s| s.parse::<i16>().map_err(|_| Error::Format(
                        format!("Failed to parse '{}' as i16", s)
                    )))
                    .collect();
                    
                let values = values?;
                let buffer = Buffer::from_slice(&values)?;
                
                Ok(Column::new(field.clone(), buffer))
            },
            DataType::Int32 => {
                let values: Result<Vec<i32>> = string_values.iter()
                    .map(|&s| s.parse::<i32>().map_err(|_| Error::Format(
                        format!("Failed to parse '{}' as i32", s)
                    )))
                    .collect();
                    
                let values = values?;
                let buffer = Buffer::from_slice(&values)?;
                
                Ok(Column::new(field.clone(), buffer))
            },
            DataType::Int64 => {
                let values: Result<Vec<i64>> = string_values.iter()
                    .map(|&s| s.parse::<i64>().map_err(|_| Error::Format(
                        format!("Failed to parse '{}' as i64", s)
                    )))
                    .collect();
                    
                let values = values?;
                let buffer = Buffer::from_slice(&values)?;
                
                Ok(Column::new(field.clone(), buffer))
            },
            DataType::UInt8 => {
                let values: Result<Vec<u8>> = string_values.iter()
                    .map(|&s| s.parse::<u8>().map_err(|_| Error::Format(
                        format!("Failed to parse '{}' as u8", s)
                    )))
                    .collect();
                    
                let values = values?;
                let buffer = Buffer::from_slice(&values)?;
                
                Ok(Column::new(field.clone(), buffer))
            },
            DataType::UInt16 => {
                let values: Result<Vec<u16>> = string_values.iter()
                    .map(|&s| s.parse::<u16>().map_err(|_| Error::Format(
                        format!("Failed to parse '{}' as u16", s)
                    )))
                    .collect();
                    
                let values = values?;
                let buffer = Buffer::from_slice(&values)?;
                
                Ok(Column::new(field.clone(), buffer))
            },
            DataType::UInt32 => {
                let values: Result<Vec<u32>> = string_values.iter()
                    .map(|&s| s.parse::<u32>().map_err(|_| Error::Format(
                        format!("Failed to parse '{}' as u32", s)
                    )))
                    .collect();
                    
                let values = values?;
                let buffer = Buffer::from_slice(&values)?;
                
                Ok(Column::new(field.clone(), buffer))
            },
            DataType::UInt64 => {
                let values: Result<Vec<u64>> = string_values.iter()
                    .map(|&s| s.parse::<u64>().map_err(|_| Error::Format(
                        format!("Failed to parse '{}' as u64", s)
                    )))
                    .collect();
                    
                let values = values?;
                let buffer = Buffer::from_slice(&values)?;
                
                Ok(Column::new(field.clone(), buffer))
            },
            DataType::Float32 => {
                let values: Result<Vec<f32>> = string_values.iter()
                    .map(|&s| s.parse::<f32>().map_err(|_| Error::Format(
                        format!("Failed to parse '{}' as f32", s)
                    )))
                    .collect();
                    
                let values = values?;
                let buffer = Buffer::from_slice(&values)?;
                
                Ok(Column::new(field.clone(), buffer))
            },
            DataType::Float64 => {
                let values: Result<Vec<f64>> = string_values.iter()
                    .map(|&s| s.parse::<f64>().map_err(|_| Error::Format(
                        format!("Failed to parse '{}' as f64", s)
                    )))
                    .collect();
                    
                let values = values?;
                let buffer = Buffer::from_slice(&values)?;
                
                Ok(Column::new(field.clone(), buffer))
            },
            DataType::Boolean => {
                let values: Result<Vec<bool>> = string_values.iter()
                    .map(|&s| {
                        match s.to_lowercase().as_str() {
                            "true" | "1" | "yes" | "y" | "t" => Ok(true),
                            "false" | "0" | "no" | "n" | "f" => Ok(false),
                            _ => Err(Error::Format(format!("Failed to parse '{}' as boolean", s))),
                        }
                    })
                    .collect();
                    
                let values = values?;
                let buffer = Buffer::from_slice(&values)?;
                
                Ok(Column::new(field.clone(), buffer))
            },
            DataType::String => {
                if use_dictionary && string_dictionary.is_some() {
                    // Dictionary encode strings
                    let dict = string_dictionary.as_mut().unwrap();
                    
                    // Get or insert each string
                    let indices: Vec<u32> = string_values.iter()
                        .map(|&s| dict.get_or_insert(s))
                        .collect();
                        
                    let buffer = Buffer::from_slice(&indices)?;
                    
                    // Create a dictionary-encoded column
                    Ok(Column::new_dictionary(
                        field.clone(),
                        buffer,
                        dict.len(),
                    ))
                } else {
                    // Store strings directly (less memory efficient but simpler)
                    let offsets = self.compute_string_offsets(&string_values);
                    let data = self.concatenate_strings(&string_values);
                    
                    let offsets_buffer = Buffer::from_slice(&offsets)?;
                    let data_buffer = Buffer::from_slice(data.as_bytes())?;
                    
                    Ok(Column::new_with_buffers(
                        field.clone(),
                        vec![offsets_buffer, data_buffer],
                    ))
                }
            },
            // Default for unsupported types - store as strings
            _ => {
                // Store as strings for now
                let offsets = self.compute_string_offsets(&string_values);
                let data = self.concatenate_strings(&string_values);
                
                let offsets_buffer = Buffer::from_slice(&offsets)?;
                let data_buffer = Buffer::from_slice(data.as_bytes())?;
                
                Ok(Column::new_with_buffers(
                    field.clone(),
                    vec![offsets_buffer, data_buffer],
                ))
            }
        }
    }
    
    /// Compute string offsets for variable-length string storage
    fn compute_string_offsets(&self, strings: &[&str]) -> Vec<u32> {
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
    fn concatenate_strings(&self, strings: &[&str]) -> String {
        let total_len: usize = strings.iter().map(|s| s.len()).sum();
        let mut result = String::with_capacity(total_len);
        
        for s in strings {
            result.push_str(s);
        }
        
        result
    }
}