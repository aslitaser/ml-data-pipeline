//! Record batch implementation for columnar data processing

use std::collections::HashMap;
use std::fmt;
use std::sync::Arc;

use bincode;
use serde::{Deserialize, Serialize};

use crate::column::Column;
use crate::error::{Error, Result};
use crate::schema::{DataType, Field, Schema};

/// A collection of columns representing a batch of records in columnar format
#[derive(Debug, Clone)]
pub struct RecordBatch {
    /// Schema describing the data
    schema: Arc<Schema>,
    
    /// Columns in this batch
    columns: Vec<Column>,
    
    /// Number of rows in this batch
    row_count: usize,
    
    /// Optional metadata
    metadata: Option<HashMap<String, String>>,
    
    /// Dictionary values for dictionary-encoded columns
    dictionaries: HashMap<i64, Arc<Column>>,
}

impl RecordBatch {
    /// Create a new record batch with the given schema and columns
    pub fn new(schema: Arc<Schema>, columns: Vec<Column>) -> Result<Self> {
        if columns.len() != schema.fields().len() {
            return Err(Error::InvalidArgument(
                "Number of columns does not match schema".into()
            ));
        }
        
        // Verify columns match schema
        for (i, field) in schema.fields().iter().enumerate() {
            let column = &columns[i];
            
            if column.name() != field.name() {
                return Err(Error::InvalidArgument(format!(
                    "Column name mismatch: expected '{}', got '{}'",
                    field.name(), column.name()
                )));
            }
            
            if !column.data_type().compatible_with(field.data_type()) {
                return Err(Error::InvalidArgument(format!(
                    "Column type mismatch for '{}': expected {:?}, got {:?}",
                    field.name(), field.data_type(), column.data_type()
                )));
            }
        }
        
        // Verify all columns have the same length
        if !columns.is_empty() {
            let row_count = columns[0].len();
            for column in &columns[1..] {
                if column.len() != row_count {
                    return Err(Error::InvalidArgument(
                        "All columns must have the same length".into()
                    ));
                }
            }
            
            Ok(Self {
                schema,
                columns,
                row_count,
                metadata: None,
                dictionaries: HashMap::new(),
            })
        } else {
            // Empty batch
            Ok(Self {
                schema,
                columns,
                row_count: 0,
                metadata: None,
                dictionaries: HashMap::new(),
            })
        }
    }
    
    /// Create a new empty record batch with the given schema
    pub fn empty(schema: Arc<Schema>) -> Self {
        Self {
            schema,
            columns: Vec::new(),
            row_count: 0,
            metadata: None,
            dictionaries: HashMap::new(),
        }
    }
    
    /// Get the schema of this batch
    pub fn schema(&self) -> &Arc<Schema> {
        &self.schema
    }
    
    /// Get the number of rows in this batch
    pub fn row_count(&self) -> usize {
        self.row_count
    }
    
    /// Get the number of columns in this batch
    pub fn column_count(&self) -> usize {
        self.columns.len()
    }
    
    /// Check if this batch is empty
    pub fn is_empty(&self) -> bool {
        self.row_count == 0
    }
    
    /// Get a reference to a column by index
    pub fn column(&self, index: usize) -> Result<&Column> {
        self.columns.get(index).ok_or(Error::IndexOutOfBounds)
    }
    
    /// Get a mutable reference to a column by index
    pub fn column_mut(&mut self, index: usize) -> Result<&mut Column> {
        self.columns.get_mut(index).ok_or(Error::IndexOutOfBounds)
    }
    
    /// Get a reference to a column by name
    pub fn column_by_name(&self, name: &str) -> Result<&Column> {
        let index = self.schema.index_of(name)?;
        self.column(index)
    }
    
    /// Get a mutable reference to a column by name
    pub fn column_by_name_mut(&mut self, name: &str) -> Result<&mut Column> {
        let index = self.schema.index_of(name)?;
        self.column_mut(index)
    }
    
    /// Get all columns
    pub fn columns(&self) -> &[Column] {
        &self.columns
    }
    
    /// Get all columns as mutable
    pub fn columns_mut(&mut self) -> &mut [Column] {
        &mut self.columns
    }
    
    /// Add a dictionary for a dictionary-encoded column
    pub fn add_dictionary(&mut self, id: i64, values: Arc<Column>) {
        self.dictionaries.insert(id, values);
    }
    
    /// Get a dictionary by ID
    pub fn dictionary(&self, id: i64) -> Option<&Arc<Column>> {
        self.dictionaries.get(&id)
    }
    
    /// Get metadata value by key
    pub fn metadata_value(&self, key: &str) -> Option<&str> {
        self.metadata.as_ref().and_then(|m| m.get(key).map(|s| s.as_str()))
    }
    
    /// Set a metadata value
    pub fn set_metadata_value(&mut self, key: &str, value: &str) {
        if self.metadata.is_none() {
            self.metadata = Some(HashMap::new());
        }
        
        if let Some(metadata) = &mut self.metadata {
            metadata.insert(key.to_string(), value.to_string());
        }
    }
    
    /// Get all metadata
    pub fn metadata(&self) -> Option<&HashMap<String, String>> {
        self.metadata.as_ref()
    }
    
    /// Slice this batch to create a view of a range of rows
    pub fn slice(&self, offset: usize, length: usize) -> Result<Self> {
        if offset + length > self.row_count {
            return Err(Error::IndexOutOfBounds);
        }
        
        let mut columns = Vec::with_capacity(self.columns.len());
        
        for column in &self.columns {
            columns.push(column.slice(offset, length)?);
        }
        
        Ok(Self {
            schema: self.schema.clone(),
            columns,
            row_count: length,
            metadata: self.metadata.clone(),
            dictionaries: self.dictionaries.clone(),
        })
    }
    
    /// Create a projection of this batch with only the specified columns
    pub fn project(&self, indices: &[usize]) -> Result<Self> {
        if indices.iter().any(|&i| i >= self.columns.len()) {
            return Err(Error::IndexOutOfBounds);
        }
        
        let mut columns = Vec::with_capacity(indices.len());
        let mut fields = Vec::with_capacity(indices.len());
        
        for &index in indices {
            columns.push(self.columns[index].clone());
            fields.push(Field {
                name: self.schema.field(index).name().to_string(),
                data_type: self.schema.field(index).data_type().clone(),
                nullable: self.schema.field(index).is_nullable(),
                metadata: self.schema.field(index).metadata().cloned(),
            });
        }
        
        let projected_schema = Arc::new(Schema::new(fields));
        
        Ok(Self {
            schema: projected_schema,
            columns,
            row_count: self.row_count,
            metadata: self.metadata.clone(),
            dictionaries: self.dictionaries.clone(),
        })
    }
    
    /// Create a projection of this batch with only the specified column names
    pub fn project_by_names(&self, names: &[&str]) -> Result<Self> {
        let indices = names.iter().map(|&name| self.schema.index_of(name)).collect::<Result<Vec<_>>>()?;
        self.project(&indices)
    }
    
    /// Calculate the total memory usage of this batch in bytes
    pub fn memory_usage(&self) -> usize {
        let mut total = 0;
        
        // Add size of columns
        for column in &self.columns {
            total += column.memory_usage();
        }
        
        // Add dictionaries
        for dictionary in self.dictionaries.values() {
            total += dictionary.memory_usage();
        }
        
        // Add metadata (rough estimate)
        if let Some(metadata) = &self.metadata {
            for (key, value) in metadata {
                total += key.len() + value.len();
            }
        }
        
        total
    }
    
    /// Serialize this batch to a binary format optimized for IPC
    pub fn serialize(&self) -> Result<Vec<u8>> {
        let mut result = Vec::new();
        
        // Serialize schema
        let schema_bytes = bincode::serialize(&self.schema).map_err(Error::Serialization)?;
        bincode::serialize_into(&mut result, &schema_bytes.len()).map_err(Error::Serialization)?;
        result.extend_from_slice(&schema_bytes);
        
        // Serialize row count
        bincode::serialize_into(&mut result, &self.row_count).map_err(Error::Serialization)?;
        
        // Serialize columns
        bincode::serialize_into(&mut result, &self.columns.len()).map_err(Error::Serialization)?;
        for column in &self.columns {
            let column_bytes = column.serialize()?;
            bincode::serialize_into(&mut result, &column_bytes.len()).map_err(Error::Serialization)?;
            result.extend_from_slice(&column_bytes);
        }
        
        // Serialize dictionaries
        bincode::serialize_into(&mut result, &self.dictionaries.len()).map_err(Error::Serialization)?;
        for (id, dictionary) in &self.dictionaries {
            bincode::serialize_into(&mut result, id).map_err(Error::Serialization)?;
            let column_bytes = dictionary.serialize()?;
            bincode::serialize_into(&mut result, &column_bytes.len()).map_err(Error::Serialization)?;
            result.extend_from_slice(&column_bytes);
        }
        
        // Serialize metadata
        bincode::serialize_into(&mut result, &self.metadata).map_err(Error::Serialization)?;
        
        Ok(result)
    }
    
    /// Deserialize a batch from a binary format
    pub fn deserialize(data: &[u8]) -> Result<Self> {
        let mut cursor = std::io::Cursor::new(data);
        
        // Deserialize schema
        let schema_len: usize = bincode::deserialize_from(&mut cursor).map_err(Error::Serialization)?;
        let schema_start = cursor.position() as usize;
        let schema_bytes = &data[schema_start..schema_start + schema_len];
        let schema: Schema = bincode::deserialize(schema_bytes).map_err(Error::Serialization)?;
        cursor.set_position((schema_start + schema_len) as u64);
        
        // Deserialize row count
        let row_count: usize = bincode::deserialize_from(&mut cursor).map_err(Error::Serialization)?;
        
        // Deserialize columns
        let column_count: usize = bincode::deserialize_from(&mut cursor).map_err(Error::Serialization)?;
        let mut columns = Vec::with_capacity(column_count);
        
        for _ in 0..column_count {
            let column_len: usize = bincode::deserialize_from(&mut cursor).map_err(Error::Serialization)?;
            let column_start = cursor.position() as usize;
            let column_bytes = &data[column_start..column_start + column_len];
            let column = Column::deserialize(column_bytes)?;
            columns.push(column);
            cursor.set_position((column_start + column_len) as u64);
        }
        
        // Deserialize dictionaries
        let dict_count: usize = bincode::deserialize_from(&mut cursor).map_err(Error::Serialization)?;
        let mut dictionaries = HashMap::with_capacity(dict_count);
        
        for _ in 0..dict_count {
            let id: i64 = bincode::deserialize_from(&mut cursor).map_err(Error::Serialization)?;
            let dict_len: usize = bincode::deserialize_from(&mut cursor).map_err(Error::Serialization)?;
            let dict_start = cursor.position() as usize;
            let dict_bytes = &data[dict_start..dict_start + dict_len];
            let dictionary = Column::deserialize(dict_bytes)?;
            dictionaries.insert(id, Arc::new(dictionary));
            cursor.set_position((dict_start + dict_len) as u64);
        }
        
        // Deserialize metadata
        let metadata: Option<HashMap<String, String>> = bincode::deserialize_from(&mut cursor).map_err(Error::Serialization)?;
        
        Ok(Self {
            schema: Arc::new(schema),
            columns,
            row_count,
            metadata,
            dictionaries,
        })
    }
}

impl fmt::Display for RecordBatch {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        // Print schema
        writeln!(f, "RecordBatch: {} rows, {} columns", self.row_count, self.columns.len())?;
        writeln!(f, "Schema: {}", self.schema)?;
        
        // Limit number of rows to display
        const MAX_ROWS: usize = 10;
        const MAX_COLS: usize = 5;
        
        // Print column headers
        let display_cols = self.columns.len().min(MAX_COLS);
        
        for i in 0..display_cols {
            if i > 0 {
                write!(f, " | ")?;
            }
            write!(f, "{:15}", self.schema.field(i).name())?;
        }
        
        if display_cols < self.columns.len() {
            write!(f, " | ... ({} more columns)", self.columns.len() - display_cols)?;
        }
        
        writeln!(f)?;
        
        // Print separator
        for i in 0..display_cols {
            if i > 0 {
                write!(f, " | ")?;
            }
            write!(f, "{:-<15}", "")?;
        }
        writeln!(f)?;
        
        // Print rows
        let display_rows = self.row_count.min(MAX_ROWS);
        
        for row in 0..display_rows {
            for col in 0..display_cols {
                if col > 0 {
                    write!(f, " | ")?;
                }
                
                // Format value based on type (simplified)
                match self.columns[col].data_type() {
                    DataType::Int32 => {
                        if self.columns[col].is_null(row) {
                            write!(f, "{:15}", "null")?;
                        } else {
                            let values = unsafe { self.columns[col].typed_data::<i32>() };
                            write!(f, "{:15}", values[row])?;
                        }
                    }
                    DataType::Float32 => {
                        if self.columns[col].is_null(row) {
                            write!(f, "{:15}", "null")?;
                        } else {
                            let values = unsafe { self.columns[col].typed_data::<f32>() };
                            write!(f, "{:15.6}", values[row])?;
                        }
                    }
                    DataType::String => {
                        if self.columns[col].is_null(row) {
                            write!(f, "{:15}", "null")?;
                        } else {
                            // String columns use offsets - this is simplified
                            write!(f, "{:15}", "<string>")?;
                        }
                    }
                    _ => {
                        write!(f, "{:15}", "<complex>")?;
                    }
                }
            }
            writeln!(f)?;
        }
        
        // Indicate if more rows were truncated
        if self.row_count > MAX_ROWS {
            writeln!(f, "... ({} more rows)", self.row_count - MAX_ROWS)?;
        }
        
        Ok(())
    }
}