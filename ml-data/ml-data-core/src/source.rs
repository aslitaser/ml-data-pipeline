//! Source trait and implementations for data input

use std::sync::Arc;
use crate::error::Result;
use crate::record_batch::RecordBatch;
use crate::schema::Schema;

/// A source of data for the pipeline
pub trait Source: Send + Sync {
    /// The type of items produced by this source
    type Item;
    
    /// The error type that can be produced by this source
    type Error;
    
    /// Retrieve the next batch of items from this source
    /// Returns None when exhausted
    fn next_batch(&mut self, max_batch_size: usize) -> Result<Option<Vec<Self::Item>>>;
    
    /// Provides a hint about the total number of items (if known)
    fn size_hint(&self) -> Option<usize> { None }
    
    /// Memory usage estimate in bytes
    fn memory_usage(&self) -> usize;
}

/// A record batch source for the pipeline
pub trait RecordBatchSource: Send + Sync {
    /// Get the schema of this source
    fn schema(&self) -> Arc<Schema>;
    
    /// Retrieve the next batch of records from this source
    /// Returns None when exhausted
    fn next_batch(&mut self, max_batch_size: usize) -> Result<Option<RecordBatch>>;
    
    /// Provides a hint about the total number of rows (if known)
    fn row_count_hint(&self) -> Option<usize> { None }
    
    /// Memory usage estimate in bytes
    fn memory_usage(&self) -> usize;
    
    /// Reset the source to start reading from the beginning
    fn reset(&mut self) -> Result<()>;
}

/// A factory for creating sources that can be used in parallel
pub trait SourceFactory: Send + Sync {
    /// The type of source this factory creates
    type Source: Source;
    
    /// Create a new source
    fn create(&self) -> Result<Self::Source>;
    
    /// Get the number of sources that can be created (if known)
    /// This is useful for planning parallel execution
    fn source_count_hint(&self) -> Option<usize> { None }
}