//! Sink trait and implementations for data output

use crate::error::Result;
use crate::record_batch::RecordBatch;

/// A sink that consumes data items
pub trait Sink: Send + Sync {
    /// The type of items this sink consumes
    type Item;
    
    /// The error type that can be produced by this sink
    type Error;
    
    /// Consume a batch of items
    fn consume(&mut self, items: Vec<Self::Item>) -> Result<()>;
    
    /// Flush any buffered items and finalize
    fn flush(&mut self) -> Result<()>;
    
    /// Memory usage estimate in bytes
    fn memory_usage(&self) -> usize;
}

/// A sink that consumes record batches
pub trait RecordBatchSink: Send + Sync {
    /// Consume a record batch
    fn consume(&mut self, batch: RecordBatch) -> Result<()>;
    
    /// Flush any buffered records and finalize
    fn flush(&mut self) -> Result<()>;
    
    /// Memory usage estimate in bytes
    fn memory_usage(&self) -> usize;
}

/// A sink that collects items in memory
pub struct CollectingSink<T> {
    /// The collected items
    items: Vec<T>,
    
    /// Maximum number of items to collect
    max_items: Option<usize>,
}

impl<T> CollectingSink<T> {
    /// Create a new collecting sink with no limit
    pub fn new() -> Self {
        Self {
            items: Vec::new(),
            max_items: None,
        }
    }
    
    /// Create a new collecting sink with a maximum number of items
    pub fn with_capacity(max_items: usize) -> Self {
        Self {
            items: Vec::with_capacity(max_items),
            max_items: Some(max_items),
        }
    }
    
    /// Get the collected items
    pub fn items(&self) -> &[T] {
        &self.items
    }
    
    /// Take ownership of the collected items
    pub fn take_items(self) -> Vec<T> {
        self.items
    }
}

impl<T: Send + Sync> Sink for CollectingSink<T> {
    type Item = T;
    type Error = std::convert::Infallible;
    
    fn consume(&mut self, mut items: Vec<Self::Item>) -> Result<()> {
        if let Some(max) = self.max_items {
            let remaining = max.saturating_sub(self.items.len());
            if remaining < items.len() {
                items.truncate(remaining);
            }
        }
        
        self.items.extend(items);
        Ok(())
    }
    
    fn flush(&mut self) -> Result<()> {
        Ok(())
    }
    
    fn memory_usage(&self) -> usize {
        // Rough estimate based on vec capacity
        self.items.capacity() * std::mem::size_of::<T>()
    }
}

/// A sink that collects record batches in memory
pub struct CollectingBatchSink {
    /// The collected batches
    batches: Vec<RecordBatch>,
    
    /// Maximum number of rows to collect
    max_rows: Option<usize>,
    
    /// Current row count
    row_count: usize,
}

impl CollectingBatchSink {
    /// Create a new collecting batch sink with no limit
    pub fn new() -> Self {
        Self {
            batches: Vec::new(),
            max_rows: None,
            row_count: 0,
        }
    }
    
    /// Create a new collecting batch sink with a maximum number of rows
    pub fn with_max_rows(max_rows: usize) -> Self {
        Self {
            batches: Vec::new(),
            max_rows: Some(max_rows),
            row_count: 0,
        }
    }
    
    /// Get the collected batches
    pub fn batches(&self) -> &[RecordBatch] {
        &self.batches
    }
    
    /// Take ownership of the collected batches
    pub fn take_batches(self) -> Vec<RecordBatch> {
        self.batches
    }
    
    /// Get the total number of rows collected
    pub fn row_count(&self) -> usize {
        self.row_count
    }
}

impl RecordBatchSink for CollectingBatchSink {
    fn consume(&mut self, batch: RecordBatch) -> Result<()> {
        if let Some(max) = self.max_rows {
            if self.row_count >= max {
                return Ok(());
            }
            
            if self.row_count + batch.row_count() > max {
                // Need to slice the batch
                let rows_to_take = max - self.row_count;
                let sliced = batch.slice(0, rows_to_take)?;
                self.batches.push(sliced);
                self.row_count += rows_to_take;
                return Ok(());
            }
        }
        
        self.row_count += batch.row_count();
        self.batches.push(batch);
        Ok(())
    }
    
    fn flush(&mut self) -> Result<()> {
        Ok(())
    }
    
    fn memory_usage(&self) -> usize {
        self.batches.iter().map(|b| b.memory_usage()).sum()
    }
}