//! Data source trait hierarchy for different source types
//!
//! This module provides a generalized interface for different data sources.

use std::fs::File;
use std::io::{Read, Seek, SeekFrom};
use std::path::{Path, PathBuf};
use std::sync::Arc;

use ml_data_core::{RecordBatch, Schema, Source, RecordBatchSource};
use ml_data_core::error::Result as CoreResult;
use ml_data_core::io::{FileIO, OpenOptions, MemoryMappedFile};

use crate::error::{Error, Result};

/// A trait for seeking within data sources
pub trait DataSourceSeek {
    /// Seek to a specific position
    fn seek(&mut self, pos: SeekFrom) -> Result<u64>;
    
    /// Get the current position
    fn position(&self) -> Result<u64>;
    
    /// Rewind to the beginning
    fn rewind(&mut self) -> Result<()> {
        self.seek(SeekFrom::Start(0))?;
        Ok(())
    }
}

/// Base trait for all data sources
pub trait DataSource: Send + Sync {
    /// Get the schema of this data source
    fn schema(&self) -> &Arc<Schema>;
    
    /// Reset the data source to start from the beginning
    fn reset(&mut self) -> Result<()>;
    
    /// Estimate the total number of rows (if known)
    fn estimated_rows(&self) -> Option<usize> {
        None
    }
    
    /// Get an estimate of memory usage for this data source
    fn memory_usage(&self) -> usize;
    
    /// Get source metadata
    fn metadata(&self) -> Option<&std::collections::HashMap<String, String>> {
        None
    }
}

/// A file-based data source
pub trait FileDataSource: DataSource + DataSourceSeek {
    /// Get the path to the file
    fn path(&self) -> &Path;
    
    /// Get the size of the file in bytes
    fn file_size(&self) -> Result<u64>;
    
    /// Check if this file source supports zero-copy
    fn supports_zero_copy(&self) -> bool {
        false
    }
    
    /// Memory-map this file for zero-copy access (if supported)
    fn memory_map(&mut self) -> Result<()> {
        Err(Error::Unsupported("Memory mapping not supported for this source".into()))
    }
}

/// A stream-based data source (network, stdin, etc.)
pub trait StreamDataSource: DataSource {
    /// Check if this stream is still active
    fn is_active(&self) -> bool;
    
    /// Get the number of bytes read so far
    fn bytes_read(&self) -> u64;
    
    /// Set a timeout for read operations
    fn set_timeout(&mut self, timeout_ms: u64) -> Result<()>;
}

/// An in-memory data source
pub trait MemoryDataSource: DataSource + DataSourceSeek {
    /// Get a reference to the in-memory data
    fn data(&self) -> &[u8];
    
    /// Get the total size in bytes
    fn size(&self) -> usize {
        self.data().len()
    }
}

/// Base implementation for file data sources
pub struct BaseFileDataSource {
    /// Path to the file
    path: PathBuf,
    
    /// File handle
    file: Box<dyn FileIO>,
    
    /// Current position in the file
    position: u64,
    
    /// Size of the file
    size: u64,
    
    /// Schema of the data
    schema: Arc<Schema>,
    
    /// Is memory-mapped
    memory_mapped: bool,
}

impl BaseFileDataSource {
    /// Create a new file data source
    pub fn new<P: AsRef<Path>>(path: P, schema: Arc<Schema>) -> Result<Self> {
        let path = path.as_ref().to_path_buf();
        let file = OpenOptions::new()
            .read(true)
            .buffer_size(64 * 1024) // 64 KB buffer
            .open(&path)?;
        
        let metadata = std::fs::metadata(&path)?;
        let size = metadata.len();
        
        Ok(Self {
            path,
            file,
            position: 0,
            size,
            schema,
            memory_mapped: false,
        })
    }
    
    /// Create a memory-mapped file data source
    pub fn new_memory_mapped<P: AsRef<Path>>(path: P, schema: Arc<Schema>) -> Result<Self> {
        let path = path.as_ref().to_path_buf();
        let mmap = MemoryMappedFile::open(&path)?;
        
        let size = mmap.size() as u64;
        
        Ok(Self {
            path,
            file: Box::new(mmap),
            position: 0,
            size,
            schema,
            memory_mapped: true,
        })
    }
    
    /// Get a reference to the file object
    pub fn file(&self) -> &dyn FileIO {
        self.file.as_ref()
    }
    
    /// Get a mutable reference to the file object
    pub fn file_mut(&mut self) -> &mut dyn FileIO {
        self.file.as_mut()
    }
    
    /// Check if this file is memory-mapped
    pub fn is_memory_mapped(&self) -> bool {
        self.memory_mapped
    }
    
    /// Memory-map this file
    pub fn memory_map(&mut self) -> Result<()> {
        if self.memory_mapped {
            return Ok(());
        }
        
        // Create new memory-mapped file
        let mmap = MemoryMappedFile::open(&self.path)?;
        let position = self.position;
        
        // Replace file
        self.file = Box::new(mmap);
        self.memory_mapped = true;
        
        // Maintain position
        self.position = position;
        
        Ok(())
    }
}

impl DataSource for BaseFileDataSource {
    fn schema(&self) -> &Arc<Schema> {
        &self.schema
    }
    
    fn reset(&mut self) -> Result<()> {
        self.position = 0;
        Ok(())
    }
    
    fn memory_usage(&self) -> usize {
        if self.memory_mapped {
            self.size as usize
        } else {
            // Just an estimate for the file handle buffer
            64 * 1024
        }
    }
}

impl DataSourceSeek for BaseFileDataSource {
    fn seek(&mut self, pos: SeekFrom) -> Result<u64> {
        if let Some(file) = self.file.as_mut().downcast_mut::<File>() {
            self.position = file.seek(pos)?;
        } else if let Some(mmap) = self.file.as_mut().downcast_mut::<MemoryMappedFile>() {
            // Manual position tracking for memory-mapped files
            match pos {
                SeekFrom::Start(offset) => {
                    if offset > self.size {
                        return Err(Error::Io(std::io::Error::new(
                            std::io::ErrorKind::InvalidInput,
                            "Seek position out of range",
                        )));
                    }
                    self.position = offset;
                }
                SeekFrom::End(offset) => {
                    let offset = if offset < 0 {
                        if offset.abs() as u64 > self.size {
                            return Err(Error::Io(std::io::Error::new(
                                std::io::ErrorKind::InvalidInput,
                                "Seek position out of range",
                            )));
                        }
                        self.size - offset.abs() as u64
                    } else {
                        self.size + offset as u64
                    };
                    self.position = offset;
                }
                SeekFrom::Current(offset) => {
                    let offset = if offset < 0 {
                        if offset.abs() as u64 > self.position {
                            return Err(Error::Io(std::io::Error::new(
                                std::io::ErrorKind::InvalidInput,
                                "Seek position out of range",
                            )));
                        }
                        self.position - offset.abs() as u64
                    } else {
                        self.position + offset as u64
                    };
                    if offset > self.size {
                        return Err(Error::Io(std::io::Error::new(
                            std::io::ErrorKind::InvalidInput,
                            "Seek position out of range",
                        )));
                    }
                    self.position = offset;
                }
            }
        } else {
            return Err(Error::Unsupported("Seek not supported for this file type".into()));
        }
        
        Ok(self.position)
    }
    
    fn position(&self) -> Result<u64> {
        Ok(self.position)
    }
}

impl FileDataSource for BaseFileDataSource {
    fn path(&self) -> &Path {
        &self.path
    }
    
    fn file_size(&self) -> Result<u64> {
        Ok(self.size)
    }
    
    fn supports_zero_copy(&self) -> bool {
        self.memory_mapped
    }
    
    fn memory_map(&mut self) -> Result<()> {
        Self::memory_map(self)
    }
}

/// Base implementation for record batch sources
pub struct RecordBatchSourceAdapter<T> {
    /// Inner data source
    inner: T,
    
    /// If the source is exhausted
    exhausted: bool,
}

impl<T: DataSource> RecordBatchSourceAdapter<T> {
    /// Create a new record batch source adapter
    pub fn new(inner: T) -> Self {
        Self {
            inner,
            exhausted: false,
        }
    }
    
    /// Get a reference to the inner data source
    pub fn inner(&self) -> &T {
        &self.inner
    }
    
    /// Get a mutable reference to the inner data source
    pub fn inner_mut(&mut self) -> &mut T {
        &mut self.inner
    }
    
    /// Check if the source is exhausted
    pub fn is_exhausted(&self) -> bool {
        self.exhausted
    }
    
    /// Mark the source as exhausted
    pub fn set_exhausted(&mut self, exhausted: bool) {
        self.exhausted = exhausted;
    }
}

impl<T: DataSource> RecordBatchSource for RecordBatchSourceAdapter<T> {
    fn schema(&self) -> Arc<Schema> {
        self.inner.schema().clone()
    }
    
    fn next_batch(&mut self, max_batch_size: usize) -> CoreResult<Option<RecordBatch>> {
        // This is a stub - the actual implementation should be provided by derived types
        if self.exhausted {
            return Ok(None);
        }
        
        // For this base adapter, just return None to indicate no more data
        self.exhausted = true;
        Ok(None)
    }
    
    fn row_count_hint(&self) -> Option<usize> {
        self.inner.estimated_rows()
    }
    
    fn memory_usage(&self) -> usize {
        self.inner.memory_usage()
    }
    
    fn reset(&mut self) -> CoreResult<()> {
        self.inner.reset()?;
        self.exhausted = false;
        Ok(())
    }
}