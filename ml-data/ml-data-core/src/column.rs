//! Column implementation for storing typed vectors of data

use std::fmt;
use std::sync::Arc;

use bincode;
use bytemuck::Pod;
use serde::{Deserialize, Serialize};

use crate::buffer::Buffer;
use crate::error::{Error, Result};
use crate::schema::DataType;

/// A column of data with a specific type
#[derive(Debug, Clone)]
pub struct Column {
    /// Name of the column
    name: String,
    
    /// Data type of the column
    data_type: DataType,
    
    /// Buffer containing the actual data values
    data: Buffer,
    
    /// Optional buffer for null value bitmap
    nulls: Option<Buffer>,
    
    /// Count of null values in this column
    null_count: usize,
    
    /// Optional buffer for offsets (for variable-length data)
    offsets: Option<Buffer>,
    
    /// Number of logical values in this column
    length: usize,
}

impl Column {
    /// Create a new column with the given name, type, and data
    pub fn new(name: &str, data_type: DataType, data: Buffer, length: usize) -> Self {
        Self {
            name: name.to_string(),
            data_type,
            data,
            nulls: None,
            null_count: 0,
            offsets: None,
            length,
        }
    }
    
    /// Create a new column with nullable values
    pub fn new_nullable(
        name: &str,
        data_type: DataType,
        data: Buffer,
        nulls: Buffer,
        null_count: usize,
        length: usize,
    ) -> Self {
        Self {
            name: name.to_string(),
            data_type,
            data,
            nulls: Some(nulls),
            null_count,
            offsets: None,
            length,
        }
    }
    
    /// Create a new column with variable-length data (e.g., strings, lists)
    pub fn new_with_offsets(
        name: &str,
        data_type: DataType,
        data: Buffer,
        offsets: Buffer,
        length: usize,
    ) -> Self {
        Self {
            name: name.to_string(),
            data_type,
            data,
            nulls: None,
            null_count: 0,
            offsets: Some(offsets),
            length,
        }
    }
    
    /// Create a new column with variable-length nullable data
    pub fn new_with_offsets_nullable(
        name: &str,
        data_type: DataType,
        data: Buffer,
        offsets: Buffer,
        nulls: Buffer,
        null_count: usize,
        length: usize,
    ) -> Self {
        Self {
            name: name.to_string(),
            data_type,
            data,
            nulls: Some(nulls),
            null_count,
            offsets: Some(offsets),
            length,
        }
    }
    
    /// Create a column representing a null column (all values are null)
    pub fn new_null_column(name: &str, data_type: DataType, length: usize) -> Result<Self> {
        // Create a buffer of appropriate size filled with zeros
        let size = data_type.size_bytes() * length;
        let data = Buffer::new_zeroed(size)?;
        
        // Create a null bitmap where all bits are zero (indicating all values are null)
        let null_bitmap_size = (length + 7) / 8; // Ceiling division by 8
        let nulls = Buffer::new_zeroed(null_bitmap_size)?;
        
        Ok(Self {
            name: name.to_string(),
            data_type,
            data,
            nulls: Some(nulls),
            null_count: length,
            offsets: None,
            length,
        })
    }
    
    /// Get the name of this column
    pub fn name(&self) -> &str {
        &self.name
    }
    
    /// Get the data type of this column
    pub fn data_type(&self) -> &DataType {
        &self.data_type
    }
    
    /// Get the length of this column (number of values)
    pub fn len(&self) -> usize {
        self.length
    }
    
    /// Check if this column is empty
    pub fn is_empty(&self) -> bool {
        self.length == 0
    }
    
    /// Get the number of null values in this column
    pub fn null_count(&self) -> usize {
        self.null_count
    }
    
    /// Check if this column has any null values
    pub fn has_nulls(&self) -> bool {
        self.null_count > 0
    }
    
    /// Check if a specific value is null
    pub fn is_null(&self, index: usize) -> bool {
        if let Some(nulls) = &self.nulls {
            if index >= self.length {
                return false;
            }
            
            // Check the bit in the null bitmap
            let byte_index = index / 8;
            let bit_index = index % 8;
            
            if byte_index >= nulls.size() {
                return false;
            }
            
            unsafe {
                let null_byte = *nulls.as_ptr().add(byte_index);
                (null_byte & (1 << bit_index)) == 0
            }
        } else {
            false
        }
    }
    
    /// Get access to the raw data buffer
    pub fn data(&self) -> &Buffer {
        &self.data
    }
    
    /// Get mutable access to the raw data buffer
    pub fn data_mut(&mut self) -> &mut Buffer {
        &mut self.data
    }
    
    /// Get access to the null bitmap buffer
    pub fn nulls(&self) -> Option<&Buffer> {
        self.nulls.as_ref()
    }
    
    /// Get access to the offsets buffer
    pub fn offsets(&self) -> Option<&Buffer> {
        self.offsets.as_ref()
    }
    
    /// Slice this column to create a view of a range of values
    pub fn slice(&self, offset: usize, length: usize) -> Result<Self> {
        if offset + length > self.length {
            return Err(Error::IndexOutOfBounds);
        }
        
        // For fixed-width types, calculate the data buffer slice
        let data_offset = match &self.data_type {
            DataType::Boolean => offset / 8,
            dt if dt.is_fixed_width() => offset * dt.size_bytes(),
            _ => {
                // For variable-width types, get the offset from the offsets buffer
                if let Some(offsets) = &self.offsets {
                    let offsets_slice = unsafe {
                        offsets.as_typed_slice::<u32>()
                    };
                    
                    offsets_slice[offset] as usize
                } else {
                    return Err(Error::InvalidOperation(
                        "Variable-width column without offsets buffer".into()
                    ));
                }
            }
        };
        
        // Calculate the data length
        let data_length = match &self.data_type {
            DataType::Boolean => ((offset + length + 7) / 8) - data_offset,
            dt if dt.is_fixed_width() => length * dt.size_bytes(),
            _ => {
                // For variable-width types, calculate from offsets
                if let Some(offsets) = &self.offsets {
                    let offsets_slice = unsafe {
                        offsets.as_typed_slice::<u32>()
                    };
                    
                    offsets_slice[offset + length] as usize - data_offset
                } else {
                    return Err(Error::InvalidOperation(
                        "Variable-width column without offsets buffer".into()
                    ));
                }
            }
        };
        
        // Slice the data buffer
        let data = self.data.slice(data_offset, data_length)?;
        
        // Slice the nulls buffer if present
        let nulls = if let Some(nulls) = &self.nulls {
            Some(nulls.slice(offset / 8, (length + 7) / 8)?)
        } else {
            None
        };
        
        // Slice the offsets buffer if present
        let offsets = if let Some(offsets) = &self.offsets {
            Some(offsets.slice(
                offset * std::mem::size_of::<u32>(),
                (length + 1) * std::mem::size_of::<u32>()
            )?)
        } else {
            None
        };
        
        // Count nulls in the slice
        let null_count = if let Some(nulls_buf) = &nulls {
            // We need to count nulls in the slice
            let mut count = 0;
            for i in 0..length {
                let byte_index = i / 8;
                let bit_index = i % 8;
                
                if byte_index < nulls_buf.size() {
                    unsafe {
                        let null_byte = *nulls_buf.as_ptr().add(byte_index);
                        if (null_byte & (1 << bit_index)) == 0 {
                            count += 1;
                        }
                    }
                }
            }
            count
        } else {
            0
        };
        
        Ok(Self {
            name: self.name.clone(),
            data_type: self.data_type.clone(),
            data,
            nulls,
            null_count,
            offsets,
            length,
        })
    }
    
    /// Get typed access to a column's data for a specific data type
    ///
    /// # Safety
    ///
    /// The caller must ensure that the requested type T matches the actual data type
    /// of the column.
    pub unsafe fn typed_data<T: Pod>(&self) -> &[T] {
        self.data.as_typed_slice::<T>()
    }
    
    /// Get typed access to a column's offsets
    ///
    /// # Safety
    ///
    /// The caller must ensure that this column has an offsets buffer.
    pub unsafe fn typed_offsets<T: Pod>(&self) -> &[T] {
        self.offsets
            .as_ref()
            .expect("Column has no offsets buffer")
            .as_typed_slice::<T>()
    }
    
    /// Get access to a specific value by index
    /// 
    /// # Safety
    /// 
    /// The caller must ensure that the index is valid and the type T matches the actual data type.
    pub unsafe fn value<T: Pod>(&self, index: usize) -> Option<T> {
        if self.is_null(index) {
            return None;
        }
        
        if self.data_type.is_fixed_width() {
            let values = self.typed_data::<T>();
            Some(values[index])
        } else {
            // For variable-width types like strings, lists, etc.
            // This is a simplified implementation
            None
        }
    }
    
    /// Serialize the column to a binary format optimized for IPC
    pub fn serialize(&self) -> Result<Vec<u8>> {
        let mut result = Vec::new();
        
        // Serialize metadata
        let metadata = ColumnMetadata {
            name: self.name.clone(),
            data_type: self.data_type.clone(),
            length: self.length,
            null_count: self.null_count,
            has_nulls: self.nulls.is_some(),
            has_offsets: self.offsets.is_some(),
        };
        
        bincode::serialize_into(&mut result, &metadata).map_err(Error::Serialization)?;
        
        // Serialize data buffer
        bincode::serialize_into(&mut result, &self.data.size()).map_err(Error::Serialization)?;
        result.extend_from_slice(unsafe {
            std::slice::from_raw_parts(self.data.as_ptr(), self.data.size())
        });
        
        // Serialize nulls buffer if present
        if let Some(nulls) = &self.nulls {
            bincode::serialize_into(&mut result, &nulls.size()).map_err(Error::Serialization)?;
            result.extend_from_slice(unsafe {
                std::slice::from_raw_parts(nulls.as_ptr(), nulls.size())
            });
        }
        
        // Serialize offsets buffer if present
        if let Some(offsets) = &self.offsets {
            bincode::serialize_into(&mut result, &offsets.size()).map_err(Error::Serialization)?;
            result.extend_from_slice(unsafe {
                std::slice::from_raw_parts(offsets.as_ptr(), offsets.size())
            });
        }
        
        Ok(result)
    }
    
    /// Deserialize a column from a binary format
    pub fn deserialize(data: &[u8]) -> Result<Self> {
        let mut cursor = std::io::Cursor::new(data);
        
        // Deserialize metadata
        let metadata: ColumnMetadata = bincode::deserialize_from(&mut cursor).map_err(Error::Serialization)?;
        
        // Deserialize data buffer
        let data_size: usize = bincode::deserialize_from(&mut cursor).map_err(Error::Serialization)?;
        let data_start = cursor.position() as usize;
        let data_bytes = &data[data_start..data_start + data_size];
        let data = unsafe {
            Buffer::from_raw_parts(
                data_bytes.as_ptr() as *mut u8,
                data_size,
                data_size,
                true,
            )
        };
        cursor.set_position((data_start + data_size) as u64);
        
        // Deserialize nulls buffer if present
        let nulls = if metadata.has_nulls {
            let nulls_size: usize = bincode::deserialize_from(&mut cursor).map_err(Error::Serialization)?;
            let nulls_start = cursor.position() as usize;
            let nulls_bytes = &data[nulls_start..nulls_start + nulls_size];
            let nulls = unsafe {
                Buffer::from_raw_parts(
                    nulls_bytes.as_ptr() as *mut u8,
                    nulls_size,
                    nulls_size,
                    true,
                )
            };
            cursor.set_position((nulls_start + nulls_size) as u64);
            Some(nulls)
        } else {
            None
        };
        
        // Deserialize offsets buffer if present
        let offsets = if metadata.has_offsets {
            let offsets_size: usize = bincode::deserialize_from(&mut cursor).map_err(Error::Serialization)?;
            let offsets_start = cursor.position() as usize;
            let offsets_bytes = &data[offsets_start..offsets_start + offsets_size];
            let offsets = unsafe {
                Buffer::from_raw_parts(
                    offsets_bytes.as_ptr() as *mut u8,
                    offsets_size,
                    offsets_size,
                    true,
                )
            };
            Some(offsets)
        } else {
            None
        };
        
        Ok(Self {
            name: metadata.name,
            data_type: metadata.data_type,
            data,
            nulls,
            null_count: metadata.null_count,
            offsets,
            length: metadata.length,
        })
    }
    
    /// Calculate the memory usage of this column
    pub fn memory_usage(&self) -> usize {
        let mut usage = self.data.size();
        
        if let Some(nulls) = &self.nulls {
            usage += nulls.size();
        }
        
        if let Some(offsets) = &self.offsets {
            usage += offsets.size();
        }
        
        usage
    }
    
    /// Create a column from a vector of values
    pub fn from_vec<T: Pod>(name: &str, data_type: DataType, values: Vec<T>) -> Result<Self> {
        let buffer = Buffer::from_slice(&values)?;
        
        Ok(Self {
            name: name.to_string(),
            data_type,
            data: buffer,
            nulls: None,
            null_count: 0,
            offsets: None,
            length: values.len(),
        })
    }
    
    /// Create a column from a vector of optional values
    pub fn from_optional_vec<T: Pod>(name: &str, data_type: DataType, values: Vec<Option<T>>) -> Result<Self> {
        let mut non_null_values = Vec::with_capacity(values.len());
        let mut null_bitmap = vec![0u8; (values.len() + 7) / 8];
        let mut null_count = 0;
        
        for (i, value) in values.iter().enumerate() {
            if let Some(v) = value {
                non_null_values.push(*v);
                // Set bit in null bitmap (1 = valid, 0 = null)
                let byte_index = i / 8;
                let bit_index = i % 8;
                null_bitmap[byte_index] |= 1 << bit_index;
            } else {
                // Still need to add a placeholder value for nulls
                non_null_values.push(unsafe { std::mem::zeroed() });
                null_count += 1;
            }
        }
        
        let data = Buffer::from_slice(&non_null_values)?;
        let nulls = Buffer::from_slice(&null_bitmap)?;
        
        Ok(Self {
            name: name.to_string(),
            data_type,
            data,
            nulls: Some(nulls),
            null_count,
            offsets: None,
            length: values.len(),
        })
    }
}

/// Metadata about a column used for serialization
#[derive(Debug, Serialize, Deserialize)]
struct ColumnMetadata {
    name: String,
    data_type: DataType,
    length: usize,
    null_count: usize,
    has_nulls: bool,
    has_offsets: bool,
}