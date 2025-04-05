//! Schema definition for ML data types

use std::collections::HashMap;
use std::fmt;
use std::sync::Arc;

use serde::{Deserialize, Serialize};

use crate::error::{Error, Result};

/// Data type for column values
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum DataType {
    /// Boolean type (1 bit per value, 8 values per byte)
    Boolean,
    
    /// 8-bit signed integer
    Int8,
    
    /// 16-bit signed integer
    Int16,
    
    /// 32-bit signed integer
    Int32,
    
    /// 64-bit signed integer
    Int64,
    
    /// 8-bit unsigned integer
    UInt8,
    
    /// 16-bit unsigned integer
    UInt16,
    
    /// 32-bit unsigned integer
    UInt32,
    
    /// 64-bit unsigned integer
    UInt64,
    
    /// 16-bit floating point
    Float16,
    
    /// 32-bit floating point
    Float32,
    
    /// 64-bit floating point
    Float64,
    
    /// UTF-8 encoded string
    String,
    
    /// Binary data
    Binary,
    
    /// Fixed-size binary data
    FixedSizeBinary(usize),
    
    /// Timestamp with timezone
    Timestamp(TimeUnit, Option<String>),
    
    /// Date (32-bit representing days since UNIX epoch)
    Date32,
    
    /// Date (64-bit representing milliseconds since UNIX epoch)
    Date64,
    
    /// Time (32-bit representing seconds since midnight)
    Time32(TimeUnit),
    
    /// Time (64-bit representing nanoseconds since midnight)
    Time64(TimeUnit),
    
    /// Decimal value with precision and scale
    Decimal(usize, usize),
    
    /// List of values with a given type
    List(Box<DataType>),
    
    /// Fixed-size list of values with a given type
    FixedSizeList(Box<DataType>, usize),
    
    /// Struct with named fields
    Struct(Vec<Field>),
    
    /// Map from key type to value type
    Map(Box<DataType>, Box<DataType>),
    
    /// Dense multi-dimensional tensor
    Tensor(Box<DataType>, Vec<usize>, Option<Vec<usize>>),
    
    /// Sparse tensor
    SparseTensor(Box<DataType>, Vec<usize>, TensorFormat),
    
    /// Dictionary encoded values
    Dictionary(Box<DataType>, Box<DataType>),
    
    /// Extension type with custom parameters
    Extension(String, Box<DataType>, Vec<u8>),
    
    /// Union type
    Union(Vec<DataType>),
    
    /// Null type (for representing null values only)
    Null,
}

/// Time unit for temporal types
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum TimeUnit {
    /// Second
    Second,
    
    /// Millisecond
    Millisecond,
    
    /// Microsecond
    Microsecond,
    
    /// Nanosecond
    Nanosecond,
}

/// Tensor storage format
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum TensorFormat {
    /// Coordinate format (COO)
    COO,
    
    /// Compressed sparse row format (CSR)
    CSR,
    
    /// Compressed sparse column format (CSC)
    CSC,
    
    /// Block compressed sparse row format
    BSR,
}

impl DataType {
    /// Get the size of this type in bytes
    pub fn size_bytes(&self) -> usize {
        match self {
            DataType::Boolean => 1, // 1 bit per value, but minimum allocation is 1 byte
            DataType::Int8 | DataType::UInt8 => 1,
            DataType::Int16 | DataType::UInt16 | DataType::Float16 => 2,
            DataType::Int32 | DataType::UInt32 | DataType::Float32 | DataType::Date32 | DataType::Time32(_) => 4,
            DataType::Int64 | DataType::UInt64 | DataType::Float64 | DataType::Date64 | DataType::Time64(_) | DataType::Timestamp(_, _) => 8,
            DataType::FixedSizeBinary(size) => *size,
            DataType::FixedSizeList(inner, length) => inner.size_bytes() * length,
            DataType::Decimal(precision, _) => {
                // Each digit needs ~3.3 bits, so we round up to nearest byte
                (precision + 7) / 8
            }
            _ => 0, // Variable-size types have no fixed size
        }
    }
    
    /// Check if this type is a fixed-width type
    pub fn is_fixed_width(&self) -> bool {
        matches!(
            self,
            DataType::Boolean
                | DataType::Int8
                | DataType::Int16
                | DataType::Int32
                | DataType::Int64
                | DataType::UInt8
                | DataType::UInt16
                | DataType::UInt32
                | DataType::UInt64
                | DataType::Float16
                | DataType::Float32
                | DataType::Float64
                | DataType::Date32
                | DataType::Date64
                | DataType::Time32(_)
                | DataType::Time64(_)
                | DataType::Timestamp(_, _)
                | DataType::FixedSizeBinary(_)
                | DataType::FixedSizeList(_, _)
                | DataType::Decimal(_, _)
        )
    }
    
    /// Check if this type is a numeric type
    pub fn is_numeric(&self) -> bool {
        matches!(
            self,
            DataType::Int8
                | DataType::Int16
                | DataType::Int32
                | DataType::Int64
                | DataType::UInt8
                | DataType::UInt16
                | DataType::UInt32
                | DataType::UInt64
                | DataType::Float16
                | DataType::Float32
                | DataType::Float64
                | DataType::Decimal(_, _)
        )
    }
    
    /// Check if this type is compatible with another type
    pub fn compatible_with(&self, other: &DataType) -> bool {
        if self == other {
            return true;
        }
        
        match (self, other) {
            // Numeric type compatibility
            (a, b) if a.is_numeric() && b.is_numeric() => true,
            
            // List compatibility
            (DataType::List(a_type), DataType::List(b_type)) => a_type.compatible_with(b_type),
            (DataType::FixedSizeList(a_type, _), DataType::FixedSizeList(b_type, _)) => a_type.compatible_with(b_type),
            
            // Struct compatibility
            (DataType::Struct(a_fields), DataType::Struct(b_fields)) => {
                if a_fields.len() != b_fields.len() {
                    return false;
                }
                
                a_fields.iter().zip(b_fields.iter()).all(|(a, b)| a.compatible_with(b))
            }
            
            // Dictionary compatibility
            (DataType::Dictionary(a_key, a_value), DataType::Dictionary(b_key, b_value)) => {
                a_key.compatible_with(b_key) && a_value.compatible_with(b_value)
            }
            
            // Tensor compatibility
            (DataType::Tensor(a_type, _, _), DataType::Tensor(b_type, _, _)) => a_type.compatible_with(b_type),
            (DataType::SparseTensor(a_type, _, _), DataType::SparseTensor(b_type, _, _)) => a_type.compatible_with(b_type),
            
            // Map compatibility
            (DataType::Map(a_key, a_value), DataType::Map(b_key, b_value)) => {
                a_key.compatible_with(b_key) && a_value.compatible_with(b_value)
            }
            
            // Extension compatibility
            (DataType::Extension(a_name, a_type, _), DataType::Extension(b_name, b_type, _)) => {
                a_name == b_name && a_type.compatible_with(b_type)
            }
            
            // Anything is compatible with Null
            (_, DataType::Null) | (DataType::Null, _) => true,
            
            // Everything else is incompatible
            _ => false,
        }
    }
}

impl fmt::Display for DataType {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            DataType::Boolean => write!(f, "Boolean"),
            DataType::Int8 => write!(f, "Int8"),
            DataType::Int16 => write!(f, "Int16"),
            DataType::Int32 => write!(f, "Int32"),
            DataType::Int64 => write!(f, "Int64"),
            DataType::UInt8 => write!(f, "UInt8"),
            DataType::UInt16 => write!(f, "UInt16"),
            DataType::UInt32 => write!(f, "UInt32"),
            DataType::UInt64 => write!(f, "UInt64"),
            DataType::Float16 => write!(f, "Float16"),
            DataType::Float32 => write!(f, "Float32"),
            DataType::Float64 => write!(f, "Float64"),
            DataType::String => write!(f, "String"),
            DataType::Binary => write!(f, "Binary"),
            DataType::FixedSizeBinary(size) => write!(f, "FixedSizeBinary({})", size),
            DataType::Timestamp(unit, tz) => {
                if let Some(tz) = tz {
                    write!(f, "Timestamp({}, '{}')", unit, tz)
                } else {
                    write!(f, "Timestamp({})", unit)
                }
            }
            DataType::Date32 => write!(f, "Date32"),
            DataType::Date64 => write!(f, "Date64"),
            DataType::Time32(unit) => write!(f, "Time32({})", unit),
            DataType::Time64(unit) => write!(f, "Time64({})", unit),
            DataType::Decimal(precision, scale) => write!(f, "Decimal({}, {})", precision, scale),
            DataType::List(item_type) => write!(f, "List({})", item_type),
            DataType::FixedSizeList(item_type, size) => write!(f, "FixedSizeList({}, {})", item_type, size),
            DataType::Struct(fields) => {
                write!(f, "Struct({{")?;
                for (i, field) in fields.iter().enumerate() {
                    if i > 0 {
                        write!(f, ", ")?;
                    }
                    write!(f, "{}: {}", field.name, field.data_type)?;
                }
                write!(f, "}})")
            }
            DataType::Map(key_type, value_type) => write!(f, "Map({} -> {})", key_type, value_type),
            DataType::Tensor(item_type, shape, _) => {
                write!(f, "Tensor({}, [", item_type)?;
                for (i, dim) in shape.iter().enumerate() {
                    if i > 0 {
                        write!(f, ", ")?;
                    }
                    write!(f, "{}", dim)?;
                }
                write!(f, "])")
            }
            DataType::SparseTensor(item_type, shape, format) => {
                write!(f, "SparseTensor({}, [", item_type)?;
                for (i, dim) in shape.iter().enumerate() {
                    if i > 0 {
                        write!(f, ", ")?;
                    }
                    write!(f, "{}", dim)?;
                }
                write!(f, "], {:?})", format)
            }
            DataType::Dictionary(key_type, value_type) => write!(f, "Dictionary({}, {})", key_type, value_type),
            DataType::Extension(name, data_type, _) => write!(f, "Extension({}, {})", name, data_type),
            DataType::Union(types) => {
                write!(f, "Union(")?;
                for (i, dtype) in types.iter().enumerate() {
                    if i > 0 {
                        write!(f, " | ")?;
                    }
                    write!(f, "{}", dtype)?;
                }
                write!(f, ")")
            }
            DataType::Null => write!(f, "Null"),
        }
    }
}

impl fmt::Display for TimeUnit {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            TimeUnit::Second => write!(f, "Second"),
            TimeUnit::Millisecond => write!(f, "Millisecond"),
            TimeUnit::Microsecond => write!(f, "Microsecond"),
            TimeUnit::Nanosecond => write!(f, "Nanosecond"),
        }
    }
}

/// A field in a schema, with a name, data type, and nullability
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct Field {
    /// Name of the field
    pub name: String,
    
    /// Data type of the field
    pub data_type: DataType,
    
    /// Whether the field can be null
    pub nullable: bool,
    
    /// Additional metadata
    pub metadata: Option<HashMap<String, String>>,
}

impl Field {
    /// Create a new field
    pub fn new(name: &str, data_type: DataType, nullable: bool) -> Self {
        Self {
            name: name.to_string(),
            data_type,
            nullable,
            metadata: None,
        }
    }
    
    /// Create a new field with metadata
    pub fn with_metadata(name: &str, data_type: DataType, nullable: bool, metadata: HashMap<String, String>) -> Self {
        Self {
            name: name.to_string(),
            data_type,
            nullable,
            metadata: Some(metadata),
        }
    }
    
    /// Get the name of this field
    pub fn name(&self) -> &str {
        &self.name
    }
    
    /// Get the data type of this field
    pub fn data_type(&self) -> &DataType {
        &self.data_type
    }
    
    /// Check if this field is nullable
    pub fn is_nullable(&self) -> bool {
        self.nullable
    }
    
    /// Get the metadata for this field
    pub fn metadata(&self) -> Option<&HashMap<String, String>> {
        self.metadata.as_ref()
    }
    
    /// Get a specific metadata value
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
    
    /// Check if this field is compatible with another field
    pub fn compatible_with(&self, other: &Field) -> bool {
        self.data_type.compatible_with(&other.data_type)
            && (self.nullable || !other.nullable) // If self is non-nullable, other must also be non-nullable
    }
}

impl fmt::Display for Field {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        if self.nullable {
            write!(f, "{}: {} (nullable)", self.name, self.data_type)
        } else {
            write!(f, "{}: {} (non-nullable)", self.name, self.data_type)
        }
    }
}

/// A schema describing a dataset's structure
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Schema {
    /// Fields in this schema
    fields: Vec<Field>,
    
    /// Field indices by name for faster lookup
    #[serde(skip)]
    field_indices: HashMap<String, usize>,
    
    /// Additional metadata
    metadata: Option<HashMap<String, String>>,
}

impl Schema {
    /// Create a new schema with the given fields
    pub fn new(fields: Vec<Field>) -> Self {
        let mut field_indices = HashMap::with_capacity(fields.len());
        for (i, field) in fields.iter().enumerate() {
            field_indices.insert(field.name.clone(), i);
        }
        
        Self {
            fields,
            field_indices,
            metadata: None,
        }
    }
    
    /// Create a new schema with the given fields and metadata
    pub fn with_metadata(fields: Vec<Field>, metadata: HashMap<String, String>) -> Self {
        let mut field_indices = HashMap::with_capacity(fields.len());
        for (i, field) in fields.iter().enumerate() {
            field_indices.insert(field.name.clone(), i);
        }
        
        Self {
            fields,
            field_indices,
            metadata: Some(metadata),
        }
    }
    
    /// Get all fields in this schema
    pub fn fields(&self) -> &[Field] {
        &self.fields
    }
    
    /// Get a field by index
    pub fn field(&self, index: usize) -> &Field {
        &self.fields[index]
    }
    
    /// Get a field by name
    pub fn field_by_name(&self, name: &str) -> Result<&Field> {
        let index = self.index_of(name)?;
        Ok(&self.fields[index])
    }
    
    /// Get the index of a field by name
    pub fn index_of(&self, name: &str) -> Result<usize> {
        self.field_indices.get(name).copied().ok_or_else(|| Error::InvalidArgument(format!("Field not found: {}", name)))
    }
    
    /// Get the number of fields in this schema
    pub fn len(&self) -> usize {
        self.fields.len()
    }
    
    /// Check if this schema is empty
    pub fn is_empty(&self) -> bool {
        self.fields.is_empty()
    }
    
    /// Get the metadata for this schema
    pub fn metadata(&self) -> Option<&HashMap<String, String>> {
        self.metadata.as_ref()
    }
    
    /// Get a specific metadata value
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
    
    /// Create a projection of this schema with only the specified fields
    pub fn project(&self, indices: &[usize]) -> Result<Self> {
        if indices.iter().any(|&i| i >= self.fields.len()) {
            return Err(Error::IndexOutOfBounds);
        }
        
        let fields = indices.iter().map(|&i| self.fields[i].clone()).collect();
        
        Ok(Self::new(fields))
    }
    
    /// Create a projection of this schema with only the specified field names
    pub fn project_by_names(&self, names: &[&str]) -> Result<Self> {
        let indices = names.iter().map(|&name| self.index_of(name)).collect::<Result<Vec<_>>>()?;
        self.project(&indices)
    }
    
    /// Serialize this schema to a binary format
    pub fn serialize(&self) -> Result<Vec<u8>> {
        bincode::serialize(self).map_err(Error::Serialization)
    }
    
    /// Deserialize a schema from a binary format
    pub fn deserialize(data: &[u8]) -> Result<Self> {
        let schema: Self = bincode::deserialize(data).map_err(Error::Serialization)?;
        
        // Rebuild the field indices
        let mut schema = schema;
        schema.field_indices.clear();
        for (i, field) in schema.fields.iter().enumerate() {
            schema.field_indices.insert(field.name.clone(), i);
        }
        
        Ok(schema)
    }
}

impl fmt::Display for Schema {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(f, "Schema: {} fields", self.fields.len())?;
        for field in &self.fields {
            writeln!(f, "  {}", field)?;
        }
        Ok(())
    }
}