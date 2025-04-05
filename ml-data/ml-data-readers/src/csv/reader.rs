//! CSV reader implementation

use std::fs::File;
use std::io::{BufRead, BufReader, Read, Seek, SeekFrom};
use std::path::{Path, PathBuf};
use std::sync::Arc;

use csv::{ReaderBuilder, StringRecord};
use ml_data_core::{RecordBatch, RecordBatchSource, Schema, Field, DataType};
use ml_data_core::error::{Error as CoreError, Result as CoreResult};
use ml_data_core::io::{MemoryMappedFile, OpenOptions};

use crate::common::{ReaderOptions, SchemaInference};
use crate::error::{Error, Result};
use crate::{DataSource, DataSourceSeek, FileDataSource};
use crate::string_cache::StringDictionary;

use super::parser::CsvParser;

/// Options for CSV reader
#[derive(Debug, Clone)]
pub struct CsvReaderOptions {
    /// Whether the CSV has a header row
    pub has_header: bool,
    
    /// Delimiter character
    pub delimiter: u8,
    
    /// Quote character
    pub quote: u8,
    
    /// Escape character
    pub escape: Option<u8>,
    
    /// Comment character
    pub comment: Option<u8>,
    
    /// Whether to trim whitespace
    pub trim: bool,
    
    /// Maximum field size in bytes
    pub max_field_size: Option<usize>,
    
    /// Core reader options
    pub reader_options: ReaderOptions,
}

impl Default for CsvReaderOptions {
    fn default() -> Self {
        Self {
            has_header: true,
            delimiter: b',',
            quote: b'"',
            escape: None,
            comment: None,
            trim: false,
            max_field_size: None,
            reader_options: ReaderOptions::default(),
        }
    }
}

/// CSV reader that implements the RecordBatchSource trait
pub struct CsvReader<R: Read> {
    /// Inner CSV reader
    reader: csv::Reader<R>,
    
    /// CSV parser
    parser: CsvParser,
    
    /// Schema of the CSV data
    schema: Arc<Schema>,
    
    /// String dictionaries for categorical columns
    string_dictionaries: Vec<Option<StringDictionary>>,
    
    /// Reader options
    options: CsvReaderOptions,
    
    /// Header row (if available)
    header: Option<StringRecord>,
    
    /// Whether the reader is exhausted
    exhausted: bool,
    
    /// Total bytes read
    bytes_read: usize,
    
    /// Estimated total rows
    estimated_rows: Option<usize>,
}

impl<R: Read> CsvReader<R> {
    /// Create a new CSV reader with schema inference
    pub fn new(reader: R, options: CsvReaderOptions) -> Self {
        // Build the CSV reader
        let mut csv_reader = ReaderBuilder::new()
            .delimiter(options.delimiter)
            .quote(options.quote)
            .has_headers(options.has_header)
            .flexible(true);
            
        if let Some(escape) = options.escape {
            csv_reader = csv_reader.escape(Some(escape));
        }
        
        if let Some(comment) = options.comment {
            csv_reader = csv_reader.comment(Some(comment));
        }
        
        csv_reader = csv_reader.trim(options.trim);
        
        let mut reader = csv_reader.from_reader(reader);
        
        // Read header if available
        let header = if options.has_header {
            // Clone the header from the reader's internal state
            Some(reader.headers().ok().cloned().unwrap_or_default())
        } else {
            None
        };
        
        // For schema inference we need to read some rows
        let mut sample_rows = Vec::new();
        let mut temp_record = StringRecord::new();
        for _ in 0..options.reader_options.schema_inference_rows {
            if reader.read_record(&mut temp_record).unwrap_or(false) {
                // Convert to Vec<String> for easier processing
                let row: Vec<String> = temp_record.iter().map(|s| s.to_string()).collect();
                sample_rows.push(row);
            } else {
                break;
            }
        }
        
        // Infer schema
        let header_strings = header.as_ref().map(|h| {
            h.iter().map(|s| s.to_string()).collect::<Vec<_>>()
        });
        
        let schema = SchemaInference::infer_from_string_records(
            &sample_rows,
            header_strings,
        ).unwrap_or_else(|_| {
            // If schema inference fails, create a schema with all string columns
            let field_count = if !sample_rows.is_empty() {
                sample_rows[0].len()
            } else if let Some(h) = &header {
                h.len()
            } else {
                0
            };
            
            let fields = (0..field_count)
                .map(|i| {
                    let name = if let Some(h) = &header {
                        if i < h.len() {
                            h[i].to_string()
                        } else {
                            format!("column_{}", i)
                        }
                    } else {
                        format!("column_{}", i)
                    };
                    
                    Field::new(&name, DataType::String, true)
                })
                .collect();
                
            Schema::new(fields)
        });
        
        // Estimate total rows based on average row size
        let estimated_rows = if !sample_rows.is_empty() {
            // We can only estimate if we have access to the total file size
            None
        } else {
            None
        };
        
        // Initialize parser with schema
        let parser = CsvParser::new(Arc::new(schema.clone()));
        
        // Initialize string dictionaries
        let string_dictionaries = parser.create_string_dictionaries(
            options.reader_options.use_dictionary_encoding
        );
        
        Self {
            reader,
            parser,
            schema: Arc::new(schema),
            string_dictionaries,
            options,
            header,
            exhausted: false,
            bytes_read: 0,
            estimated_rows,
        }
    }
    
    /// Create a new CSV reader with a provided schema
    pub fn new_with_schema(reader: R, schema: Arc<Schema>, options: CsvReaderOptions) -> Self {
        // Build the CSV reader
        let mut csv_reader = ReaderBuilder::new()
            .delimiter(options.delimiter)
            .quote(options.quote)
            .has_headers(options.has_header)
            .flexible(true);
            
        if let Some(escape) = options.escape {
            csv_reader = csv_reader.escape(Some(escape));
        }
        
        if let Some(comment) = options.comment {
            csv_reader = csv_reader.comment(Some(comment));
        }
        
        csv_reader = csv_reader.trim(options.trim);
        
        let mut reader = csv_reader.from_reader(reader);
        
        // Read header if available
        let header = if options.has_header {
            // Clone the header from the reader's internal state
            Some(reader.headers().ok().cloned().unwrap_or_default())
        } else {
            None
        };
        
        // Initialize parser with schema
        let parser = CsvParser::new(schema.clone());
        
        // Initialize string dictionaries
        let string_dictionaries = parser.create_string_dictionaries(
            options.reader_options.use_dictionary_encoding
        );
        
        Self {
            reader,
            parser,
            schema,
            string_dictionaries,
            options,
            header,
            exhausted: false,
            bytes_read: 0,
            estimated_rows: None,
        }
    }
    
    /// Process a batch of CSV records
    fn process_batch(&mut self, max_records: usize) -> Result<Option<RecordBatch>> {
        if self.exhausted {
            return Ok(None);
        }
        
        let mut records = Vec::with_capacity(max_records);
        let mut record = StringRecord::new();
        
        for _ in 0..max_records {
            if self.reader.read_record(&mut record)? {
                // Convert to Vec<String> for easier processing
                let row: Vec<String> = record.iter().map(|s| s.to_string()).collect();
                records.push(row);
            } else {
                self.exhausted = true;
                break;
            }
        }
        
        if records.is_empty() {
            return Ok(None);
        }
        
        // Parse records into a RecordBatch
        let batch = self.parser.parse_batch(
            records,
            &mut self.string_dictionaries,
            self.options.reader_options.use_dictionary_encoding,
        )?;
        
        Ok(Some(batch))
    }
}

impl<R: Read> RecordBatchSource for CsvReader<R> {
    fn schema(&self) -> Arc<Schema> {
        self.schema.clone()
    }
    
    fn next_batch(&mut self, max_batch_size: usize) -> CoreResult<Option<RecordBatch>> {
        self.process_batch(max_batch_size).map_err(|e| {
            CoreError::Io(std::io::Error::new(
                std::io::ErrorKind::Other,
                format!("CSV error: {}", e),
            ))
        })
    }
    
    fn row_count_hint(&self) -> Option<usize> {
        self.estimated_rows
    }
    
    fn memory_usage(&self) -> usize {
        // Approximate size of string dictionaries
        let dict_size = self.string_dictionaries.iter()
            .filter_map(|d| d.as_ref().map(|dict| dict.memory_usage()))
            .sum::<usize>();
            
        // Add fixed overhead
        dict_size + 8 * 1024
    }
    
    fn reset(&mut self) -> CoreResult<()> {
        Err(CoreError::NotImplemented("Cannot reset a CSV reader".into()))
    }
}

/// Factory for creating CSV readers from files
pub struct CsvReaderFactory {
    /// Path to the CSV file
    path: PathBuf,
    
    /// Reader options
    options: CsvReaderOptions,
    
    /// Schema (if provided)
    schema: Option<Arc<Schema>>,
}

impl CsvReaderFactory {
    /// Create a new CSV reader factory
    pub fn new<P: AsRef<Path>>(path: P, options: CsvReaderOptions) -> Self {
        Self {
            path: path.as_ref().to_path_buf(),
            options,
            schema: None,
        }
    }
    
    /// Create a new CSV reader factory with a provided schema
    pub fn new_with_schema<P: AsRef<Path>>(
        path: P,
        schema: Arc<Schema>,
        options: CsvReaderOptions,
    ) -> Self {
        Self {
            path: path.as_ref().to_path_buf(),
            options,
            schema: Some(schema),
        }
    }
    
    /// Create a new CSV reader
    pub fn create(&self) -> Result<CsvReader<Box<dyn Read>>> {
        // Open the file
        let file: Box<dyn Read> = if self.options.reader_options.use_memory_mapping {
            // Use memory mapping
            let mmap = MemoryMappedFile::open(&self.path)?;
            Box::new(mmap.as_slice())
        } else {
            // Use buffered reading
            let file = File::open(&self.path)?;
            let buffer_size = self.options.reader_options.buffer_size;
            Box::new(BufReader::with_capacity(buffer_size, file))
        };
        
        // Create reader
        if let Some(schema) = &self.schema {
            Ok(CsvReader::new_with_schema(
                file,
                schema.clone(),
                self.options.clone(),
            ))
        } else {
            Ok(CsvReader::new(file, self.options.clone()))
        }
    }
}

/// Seekable CSV reader for file-based sources
pub struct SeekableCsvReader {
    /// File path
    path: PathBuf,
    
    /// File handle
    file: File,
    
    /// Schema of the CSV data
    schema: Arc<Schema>,
    
    /// CSV parser
    parser: CsvParser,
    
    /// Reader options
    options: CsvReaderOptions,
    
    /// Header row position in bytes
    header_position: u64,
    
    /// Header row
    header: Option<StringRecord>,
    
    /// Current position in the file
    position: u64,
    
    /// File size
    file_size: u64,
    
    /// String dictionaries
    string_dictionaries: Vec<Option<StringDictionary>>,
    
    /// Whether the reader is exhausted
    exhausted: bool,
    
    /// Row offsets (if scan completed)
    row_offsets: Vec<u64>,
}

impl SeekableCsvReader {
    /// Create a new seekable CSV reader
    pub fn new<P: AsRef<Path>>(path: P, options: CsvReaderOptions) -> Result<Self> {
        let path = path.as_ref().to_path_buf();
        let file = File::open(&path)?;
        
        let file_size = file.metadata()?.len();
        
        // First, we need to read the header (if any) and infer the schema
        let mut buf_reader = BufReader::new(&file);
        
        // Determine the position after the header
        let header_position = if options.has_header {
            // Read first line
            let mut header_line = String::new();
            let bytes_read = buf_reader.read_line(&mut header_line)?;
            bytes_read as u64
        } else {
            0
        };
        
        // Read sample data for schema inference
        buf_reader.seek(SeekFrom::Start(header_position))?;
        
        let mut csv_reader = ReaderBuilder::new()
            .delimiter(options.delimiter)
            .quote(options.quote)
            .has_headers(false) // We've already skipped the header
            .flexible(true)
            .from_reader(buf_reader);
            
        // Read sample rows
        let mut sample_rows = Vec::new();
        let mut temp_record = StringRecord::new();
        for _ in 0..options.reader_options.schema_inference_rows {
            if csv_reader.read_record(&mut temp_record)? {
                // Convert to Vec<String> for easier processing
                let row: Vec<String> = temp_record.iter().map(|s| s.to_string()).collect();
                sample_rows.push(row);
            } else {
                break;
            }
        }
        
        // Read header if available
        let header = if options.has_header {
            // Reopen the file and read just the header
            let mut header_reader = BufReader::new(File::open(&path)?);
            let mut csv_header_reader = ReaderBuilder::new()
                .delimiter(options.delimiter)
                .quote(options.quote)
                .has_headers(true)
                .from_reader(header_reader);
                
            Some(csv_header_reader.headers()?.clone())
        } else {
            None
        };
        
        // Infer schema from sample rows
        let header_strings = header.as_ref().map(|h| {
            h.iter().map(|s| s.to_string()).collect::<Vec<_>>()
        });
        
        let schema = SchemaInference::infer_from_string_records(
            &sample_rows,
            header_strings,
        )?;
        
        // Initialize parser
        let parser = CsvParser::new(Arc::new(schema.clone()));
        
        // Initialize string dictionaries
        let string_dictionaries = parser.create_string_dictionaries(
            options.reader_options.use_dictionary_encoding
        );
        
        // Reset file position to after header
        let file = File::open(&path)?;
        
        Ok(Self {
            path,
            file,
            schema: Arc::new(schema),
            parser,
            options,
            header_position,
            header,
            position: header_position,
            file_size,
            string_dictionaries,
            exhausted: false,
            row_offsets: Vec::new(),
        })
    }
    
    /// Create a new seekable CSV reader with a provided schema
    pub fn new_with_schema<P: AsRef<Path>>(
        path: P,
        schema: Arc<Schema>,
        options: CsvReaderOptions,
    ) -> Result<Self> {
        let path = path.as_ref().to_path_buf();
        let file = File::open(&path)?;
        
        let file_size = file.metadata()?.len();
        
        // Determine the position after the header
        let header_position = if options.has_header {
            // Read first line
            let mut buf_reader = BufReader::new(&file);
            let mut header_line = String::new();
            let bytes_read = buf_reader.read_line(&mut header_line)?;
            bytes_read as u64
        } else {
            0
        };
        
        // Read header if available
        let header = if options.has_header {
            // Reopen the file and read just the header
            let mut header_reader = BufReader::new(File::open(&path)?);
            let mut csv_header_reader = ReaderBuilder::new()
                .delimiter(options.delimiter)
                .quote(options.quote)
                .has_headers(true)
                .from_reader(header_reader);
                
            Some(csv_header_reader.headers()?.clone())
        } else {
            None
        };
        
        // Initialize parser
        let parser = CsvParser::new(schema.clone());
        
        // Initialize string dictionaries
        let string_dictionaries = parser.create_string_dictionaries(
            options.reader_options.use_dictionary_encoding
        );
        
        // Reset file position to after header
        let file = File::open(&path)?;
        
        Ok(Self {
            path,
            file,
            schema,
            parser,
            options,
            header_position,
            header,
            position: header_position,
            file_size,
            string_dictionaries,
            exhausted: false,
            row_offsets: Vec::new(),
        })
    }
    
    /// Scan the file to build an index of row positions
    pub fn build_row_index(&mut self) -> Result<()> {
        if !self.row_offsets.is_empty() {
            return Ok(());
        }
        
        // Start at beginning of data (after header)
        self.file.seek(SeekFrom::Start(self.header_position))?;
        
        let mut buf_reader = BufReader::new(&self.file);
        let mut csv_reader = ReaderBuilder::new()
            .delimiter(self.options.delimiter)
            .quote(self.options.quote)
            .has_headers(false) // We've already accounted for the header
            .from_reader(buf_reader);
            
        let mut record = StringRecord::new();
        let mut current_pos = self.header_position;
        
        self.row_offsets.push(current_pos); // First row starts at this position
        
        while csv_reader.read_record(&mut record)? {
            // The byte position of this record is the start of the next record
            current_pos += record.as_byte_record().len() as u64;
            self.row_offsets.push(current_pos);
        }
        
        // Reset the file position
        self.file.seek(SeekFrom::Start(self.header_position))?;
        self.position = self.header_position;
        
        Ok(())
    }
    
    /// Seek to a specific row
    pub fn seek_to_row(&mut self, row_index: usize) -> Result<()> {
        // Build row index if needed
        if self.row_offsets.is_empty() {
            self.build_row_index()?;
        }
        
        if row_index >= self.row_offsets.len() {
            return Err(Error::InvalidArgument(format!(
                "Row index {} out of bounds (max: {})",
                row_index,
                self.row_offsets.len() - 1
            )));
        }
        
        // Seek to the position of the row
        let pos = self.row_offsets[row_index];
        self.file.seek(SeekFrom::Start(pos))?;
        self.position = pos;
        self.exhausted = false;
        
        Ok(())
    }
    
    /// Process a batch of CSV records
    fn process_batch(&mut self, max_records: usize) -> Result<Option<RecordBatch>> {
        if self.exhausted {
            return Ok(None);
        }
        
        // Create CSV reader from current position
        let buf_reader = BufReader::new(&mut self.file);
        let mut csv_reader = ReaderBuilder::new()
            .delimiter(self.options.delimiter)
            .quote(self.options.quote)
            .has_headers(false) // We're already positioned after the header
            .flexible(true)
            .from_reader(buf_reader);
            
        let mut records = Vec::with_capacity(max_records);
        let mut record = StringRecord::new();
        
        for _ in 0..max_records {
            if csv_reader.read_record(&mut record)? {
                // Convert to Vec<String> for easier processing
                let row: Vec<String> = record.iter().map(|s| s.to_string()).collect();
                
                // Track position
                self.position += record.as_byte_record().len() as u64;
                
                records.push(row);
            } else {
                self.exhausted = true;
                break;
            }
        }
        
        if records.is_empty() {
            return Ok(None);
        }
        
        // Parse records into a RecordBatch
        let batch = self.parser.parse_batch(
            records,
            &mut self.string_dictionaries,
            self.options.reader_options.use_dictionary_encoding,
        )?;
        
        Ok(Some(batch))
    }
}

impl DataSource for SeekableCsvReader {
    fn schema(&self) -> &Arc<Schema> {
        &self.schema
    }
    
    fn reset(&mut self) -> Result<()> {
        self.file.seek(SeekFrom::Start(self.header_position))?;
        self.position = self.header_position;
        self.exhausted = false;
        Ok(())
    }
    
    fn estimated_rows(&self) -> Option<usize> {
        if !self.row_offsets.is_empty() {
            // If we've already counted the rows, we know exactly
            Some(self.row_offsets.len() - 1)
        } else {
            // Make a rough estimate based on average row size
            None
        }
    }
    
    fn memory_usage(&self) -> usize {
        // Approximate size of string dictionaries
        let dict_size = self.string_dictionaries.iter()
            .filter_map(|d| d.as_ref().map(|dict| dict.memory_usage()))
            .sum::<usize>();
            
        // Add fixed overhead + offset index size
        dict_size + 8 * 1024 + (self.row_offsets.len() * 8)
    }
}

impl DataSourceSeek for SeekableCsvReader {
    fn seek(&mut self, pos: SeekFrom) -> Result<u64> {
        self.position = self.file.seek(pos)?;
        self.exhausted = false;
        Ok(self.position)
    }
    
    fn position(&self) -> Result<u64> {
        Ok(self.position)
    }
    
    fn rewind(&mut self) -> Result<()> {
        self.reset()
    }
}

impl FileDataSource for SeekableCsvReader {
    fn path(&self) -> &Path {
        &self.path
    }
    
    fn file_size(&self) -> Result<u64> {
        Ok(self.file_size)
    }
    
    fn supports_zero_copy(&self) -> bool {
        false // CSV requires parsing, so no zero-copy
    }
}

impl RecordBatchSource for SeekableCsvReader {
    fn schema(&self) -> Arc<Schema> {
        self.schema.clone()
    }
    
    fn next_batch(&mut self, max_batch_size: usize) -> CoreResult<Option<RecordBatch>> {
        self.process_batch(max_batch_size).map_err(|e| {
            CoreError::Io(std::io::Error::new(
                std::io::ErrorKind::Other,
                format!("CSV error: {}", e),
            ))
        })
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
                format!("CSV error: {}", e),
            ))
        })
    }
}