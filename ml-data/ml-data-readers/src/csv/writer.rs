//! CSV writer implementation

use std::fs::File;
use std::io::{BufWriter, Write};
use std::path::Path;

use csv::WriterBuilder;
use ml_data_core::{RecordBatch, Schema, DataType};

use crate::error::{Error, Result};
use crate::string_cache::StringDictionary;

/// Options for CSV writer
#[derive(Debug, Clone)]
pub struct CsvWriterOptions {
    /// Whether to write a header row
    pub write_header: bool,
    
    /// Delimiter character
    pub delimiter: u8,
    
    /// Quote character
    pub quote: u8,
    
    /// Whether to quote all fields
    pub quote_all: bool,
    
    /// Buffer size for writing
    pub buffer_size: usize,
}

impl Default for CsvWriterOptions {
    fn default() -> Self {
        Self {
            write_header: true,
            delimiter: b',',
            quote: b'"',
            quote_all: false,
            buffer_size: 64 * 1024, // 64KB
        }
    }
}

/// CSV writer
pub struct CsvWriter<W: Write> {
    /// Inner CSV writer
    writer: csv::Writer<W>,
    
    /// Schema of the data
    schema: Schema,
    
    /// String dictionaries (for dictionary-encoded columns)
    string_dictionaries: Vec<Option<StringDictionary>>,
    
    /// Writer options
    options: CsvWriterOptions,
}

impl<W: Write> CsvWriter<W> {
    /// Create a new CSV writer
    pub fn new(writer: W, schema: Schema, options: CsvWriterOptions) -> Result<Self> {
        let mut csv_writer = WriterBuilder::new()
            .delimiter(options.delimiter)
            .quote(options.quote)
            .quote_style(if options.quote_all {
                csv::QuoteStyle::Always
            } else {
                csv::QuoteStyle::Necessary
            })
            .from_writer(writer);
            
        // Write header if required
        if options.write_header {
            let header: Vec<String> = schema.fields().iter()
                .map(|f| f.name().to_string())
                .collect();
                
            csv_writer.write_record(&header)?;
        }
        
        // Create string dictionaries for dictionary-encoded columns
        let string_dictionaries = schema.fields().iter()
            .map(|field| {
                match field.data_type() {
                    DataType::Dictionary(_, _) => Some(StringDictionary::new()),
                    _ => None,
                }
            })
            .collect();
            
        Ok(Self {
            writer: csv_writer,
            schema,
            string_dictionaries,
            options,
        })
    }
    
    /// Write a record batch to CSV
    pub fn write_batch(&mut self, batch: &RecordBatch) -> Result<()> {
        let num_rows = batch.num_rows();
        
        // Convert each row to strings
        for row_idx in 0..num_rows {
            let mut row = Vec::with_capacity(self.schema.len());
            
            for col_idx in 0..batch.num_columns() {
                let column = batch.column(col_idx);
                let field = column.field();
                
                // Convert to string based on data type
                let value = self.format_value(column, row_idx, col_idx)?;
                row.push(value);
            }
            
            // Write the row
            self.writer.write_record(&row)?;
        }
        
        self.writer.flush()?;
        Ok(())
    }
    
    /// Format a single value as a string
    fn format_value(&self, column: &ml_data_core::Column, row_idx: usize, col_idx: usize) -> Result<String> {
        let field = column.field();
        
        match field.data_type() {
            DataType::Int8 => {
                let value = column.value::<i8>(row_idx).map_err(Error::Core)?;
                Ok(value.to_string())
            },
            DataType::Int16 => {
                let value = column.value::<i16>(row_idx).map_err(Error::Core)?;
                Ok(value.to_string())
            },
            DataType::Int32 => {
                let value = column.value::<i32>(row_idx).map_err(Error::Core)?;
                Ok(value.to_string())
            },
            DataType::Int64 => {
                let value = column.value::<i64>(row_idx).map_err(Error::Core)?;
                Ok(value.to_string())
            },
            DataType::UInt8 => {
                let value = column.value::<u8>(row_idx).map_err(Error::Core)?;
                Ok(value.to_string())
            },
            DataType::UInt16 => {
                let value = column.value::<u16>(row_idx).map_err(Error::Core)?;
                Ok(value.to_string())
            },
            DataType::UInt32 => {
                let value = column.value::<u32>(row_idx).map_err(Error::Core)?;
                Ok(value.to_string())
            },
            DataType::UInt64 => {
                let value = column.value::<u64>(row_idx).map_err(Error::Core)?;
                Ok(value.to_string())
            },
            DataType::Float32 => {
                let value = column.value::<f32>(row_idx).map_err(Error::Core)?;
                Ok(value.to_string())
            },
            DataType::Float64 => {
                let value = column.value::<f64>(row_idx).map_err(Error::Core)?;
                Ok(value.to_string())
            },
            DataType::Boolean => {
                let value = column.value::<bool>(row_idx).map_err(Error::Core)?;
                Ok(value.to_string())
            },
            DataType::String => {
                // For string columns, we need to extract from string array
                let value = column.value_as_string(row_idx).map_err(Error::Core)?;
                Ok(value)
            },
            DataType::Dictionary(_, _) => {
                if let Some(dict) = &self.string_dictionaries[col_idx] {
                    // Get dictionary index
                    let index = column.value::<u32>(row_idx).map_err(Error::Core)?;
                    
                    // Look up in dictionary
                    if let Some(value) = dict.get_value(index) {
                        Ok(value.to_string())
                    } else {
                        Err(Error::InvalidArgument(
                            format!("Invalid dictionary index: {}", index)
                        ))
                    }
                } else {
                    // Fallback to string representation
                    let value = column.value_as_string(row_idx).map_err(Error::Core)?;
                    Ok(value)
                }
            },
            // Default string representation for other types
            _ => {
                let value = column.value_as_string(row_idx).map_err(Error::Core)?;
                Ok(value)
            }
        }
    }
    
    /// Flush the writer
    pub fn flush(&mut self) -> Result<()> {
        self.writer.flush()?;
        Ok(())
    }
}

/// Create a CSV writer for a file
pub fn create_csv_writer<P: AsRef<Path>>(
    path: P, 
    schema: Schema,
    options: CsvWriterOptions,
) -> Result<CsvWriter<BufWriter<File>>> {
    let file = File::create(path)?;
    let buf_writer = BufWriter::with_capacity(options.buffer_size, file);
    
    CsvWriter::new(buf_writer, schema, options)
}