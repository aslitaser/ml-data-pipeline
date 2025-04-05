//! Transform trait and implementations for data transformation

use crate::error::Result;
use crate::record_batch::RecordBatch;
use crate::schema::Schema;
use std::sync::Arc;

/// A transformation that processes data items
pub trait Transform: Send + Sync {
    /// The type of input items
    type Input;
    
    /// The type of output items
    type Output;
    
    /// The error type that can be produced by this transform
    type Error;
    
    /// Transform a batch of items
    fn transform(&mut self, items: Vec<Self::Input>) -> Result<Vec<Self::Output>>;
    
    /// Memory usage estimate in bytes
    fn memory_usage(&self) -> usize;
    
    /// Reset internal state (if applicable)
    fn reset(&mut self) { }
}

/// A transformation that processes record batches
pub trait RecordBatchTransform: Send + Sync {
    /// Transform a record batch
    fn transform(&mut self, batch: RecordBatch) -> Result<RecordBatch>;
    
    /// Get the output schema for this transform when applied to the given input schema
    fn output_schema(&self, input_schema: &Schema) -> Result<Arc<Schema>>;
    
    /// Memory usage estimate in bytes
    fn memory_usage(&self) -> usize;
    
    /// Reset internal state (if applicable)
    fn reset(&mut self) -> Result<()> {
        Ok(())
    }
    
    /// Whether this transform preserves the order of records
    fn preserves_order(&self) -> bool {
        true
    }
    
    /// Whether this transform can be executed in parallel
    fn parallelizable(&self) -> bool {
        false
    }
}

/// A chain of transforms that can be executed as a single transform
pub struct TransformChain<T> {
    /// The transforms in this chain
    transforms: Vec<T>,
}

impl<T> TransformChain<T> {
    /// Create a new transform chain
    pub fn new(transforms: Vec<T>) -> Self {
        Self { transforms }
    }
    
    /// Get a reference to the transforms in this chain
    pub fn transforms(&self) -> &[T] {
        &self.transforms
    }
    
    /// Get a mutable reference to the transforms in this chain
    pub fn transforms_mut(&mut self) -> &mut [T] {
        &mut self.transforms
    }
}

impl<I, O, E, T: Transform<Input = I, Output = O, Error = E>> Transform for TransformChain<T> {
    type Input = I;
    type Output = O;
    type Error = E;
    
    fn transform(&mut self, items: Vec<Self::Input>) -> Result<Vec<Self::Output>> {
        let mut current = items;
        
        for transform in &mut self.transforms {
            current = transform.transform(current)?;
        }
        
        Ok(current)
    }
    
    fn memory_usage(&self) -> usize {
        self.transforms.iter().map(|t| t.memory_usage()).sum()
    }
    
    fn reset(&mut self) {
        for transform in &mut self.transforms {
            transform.reset();
        }
    }
}

impl RecordBatchTransform for TransformChain<Box<dyn RecordBatchTransform>> {
    fn transform(&mut self, batch: RecordBatch) -> Result<RecordBatch> {
        let mut current = batch;
        
        for transform in &mut self.transforms {
            current = transform.transform(current)?;
        }
        
        Ok(current)
    }
    
    fn output_schema(&self, input_schema: &Schema) -> Result<Arc<Schema>> {
        let mut current = Arc::new(input_schema.clone());
        
        for transform in &self.transforms {
            current = transform.output_schema(&current)?;
        }
        
        Ok(current)
    }
    
    fn memory_usage(&self) -> usize {
        self.transforms.iter().map(|t| t.memory_usage()).sum()
    }
    
    fn reset(&mut self) -> Result<()> {
        for transform in &mut self.transforms {
            transform.reset()?;
        }
        Ok(())
    }
    
    fn preserves_order(&self) -> bool {
        self.transforms.iter().all(|t| t.preserves_order())
    }
    
    fn parallelizable(&self) -> bool {
        self.transforms.iter().all(|t| t.parallelizable())
    }
}