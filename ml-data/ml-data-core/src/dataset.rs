//! Dataset trait and implementations

use std::sync::Arc;

use crate::error::Result;
use crate::record_batch::RecordBatch;
use crate::schema::Schema;
use crate::source::RecordBatchSource;

/// A dataset represents a collection of data
pub trait Dataset: Send + Sync {
    /// Get the schema of this dataset
    fn schema(&self) -> Arc<Schema>;
    
    /// Get the number of rows in this dataset
    fn row_count(&self) -> Option<usize>;
    
    /// Create a source for scanning this dataset
    fn scan(&self) -> Result<Box<dyn RecordBatchSource>>;
    
    /// Create a batch source with specific projection
    fn scan_with_projection(&self, projection: &[&str]) -> Result<Box<dyn RecordBatchSource>>;
    
    /// Create a source with specific selection conditions (predicate pushdown)
    fn scan_with_filter(&self, filter: Box<dyn Fn(&RecordBatch) -> Result<RecordBatch>>) -> Result<Box<dyn RecordBatchSource>>;
    
    /// Create a cached subset of this dataset for faster access
    fn cache(&self) -> Result<Arc<dyn Dataset>>;
    
    /// Repartition this dataset for more efficient processing
    fn repartition(&self, partition_count: usize) -> Result<Arc<dyn Dataset>>;
    
    /// Create a new dataset with a transformed schema
    fn with_schema(&self, schema: Arc<Schema>) -> Result<Arc<dyn Dataset>>;
}

/// A builder for creating datasets
pub struct DatasetBuilder {
    /// The schema of the dataset
    schema: Option<Arc<Schema>>,
    
    /// The source of data
    source: Option<Box<dyn RecordBatchSource>>,
    
    /// The partition count
    partition_count: Option<usize>,
    
    /// Whether to cache the dataset
    cache: bool,
}

impl DatasetBuilder {
    /// Create a new dataset builder
    pub fn new() -> Self {
        Self {
            schema: None,
            source: None,
            partition_count: None,
            cache: false,
        }
    }
    
    /// Set the schema of the dataset
    pub fn schema(mut self, schema: Arc<Schema>) -> Self {
        self.schema = Some(schema);
        self
    }
    
    /// Set the source of the dataset
    pub fn source(mut self, source: Box<dyn RecordBatchSource>) -> Self {
        self.source = Some(source);
        self
    }
    
    /// Set the partition count
    pub fn partition_count(mut self, count: usize) -> Self {
        self.partition_count = Some(count);
        self
    }
    
    /// Enable caching of the dataset
    pub fn cache(mut self, cache: bool) -> Self {
        self.cache = cache;
        self
    }
    
    /// Build the dataset
    pub fn build(self) -> Result<Arc<dyn Dataset>> {
        let schema = self.schema.ok_or_else(|| {
            crate::error::Error::InvalidArgument("Schema is required to build a dataset".into())
        })?;
        
        let source = self.source.ok_or_else(|| {
            crate::error::Error::InvalidArgument("Source is required to build a dataset".into())
        })?;
        
        let mut dataset: Arc<dyn Dataset> = Arc::new(InMemoryDataset::new(schema, source)?);
        
        if let Some(partition_count) = self.partition_count {
            dataset = dataset.repartition(partition_count)?;
        }
        
        if self.cache {
            dataset = dataset.cache()?;
        }
        
        Ok(dataset)
    }
}

impl Default for DatasetBuilder {
    fn default() -> Self {
        Self::new()
    }
}

/// An in-memory dataset backed by a vector of record batches
pub struct InMemoryDataset {
    /// The schema of the dataset
    schema: Arc<Schema>,
    
    /// The cached batches
    batches: Vec<RecordBatch>,
    
    /// The total row count
    row_count: usize,
}

impl InMemoryDataset {
    /// Create a new in-memory dataset by scanning a source
    pub fn new(schema: Arc<Schema>, mut source: Box<dyn RecordBatchSource>) -> Result<Self> {
        let mut batches = Vec::new();
        let mut row_count = 0;
        
        while let Some(batch) = source.next_batch(1024)? {
            row_count += batch.row_count();
            batches.push(batch);
        }
        
        Ok(Self {
            schema,
            batches,
            row_count,
        })
    }
    
    /// Create a new in-memory dataset from existing batches
    pub fn from_batches(schema: Arc<Schema>, batches: Vec<RecordBatch>) -> Result<Self> {
        let row_count = batches.iter().map(|b| b.row_count()).sum();
        
        Ok(Self {
            schema,
            batches,
            row_count,
        })
    }
    
    /// Get the batches in this dataset
    pub fn batches(&self) -> &[RecordBatch] {
        &self.batches
    }
}

impl Dataset for InMemoryDataset {
    fn schema(&self) -> Arc<Schema> {
        self.schema.clone()
    }
    
    fn row_count(&self) -> Option<usize> {
        Some(self.row_count)
    }
    
    fn scan(&self) -> Result<Box<dyn RecordBatchSource>> {
        Ok(Box::new(InMemoryDatasetScanner::new(self.batches.clone())))
    }
    
    fn scan_with_projection(&self, projection: &[&str]) -> Result<Box<dyn RecordBatchSource>> {
        // Find the indices for the requested fields
        let indices = projection
            .iter()
            .map(|name| self.schema.index_of(name))
            .collect::<Result<Vec<_>>>()?;
        
        // Create projected batches
        let mut projected_batches = Vec::with_capacity(self.batches.len());
        for batch in &self.batches {
            projected_batches.push(batch.project(&indices)?);
        }
        
        Ok(Box::new(InMemoryDatasetScanner::new(projected_batches)))
    }
    
    fn scan_with_filter(&self, filter: Box<dyn Fn(&RecordBatch) -> Result<RecordBatch>>) -> Result<Box<dyn RecordBatchSource>> {
        // Apply filter to each batch
        let mut filtered_batches = Vec::with_capacity(self.batches.len());
        for batch in &self.batches {
            let filtered = filter(batch)?;
            if !filtered.is_empty() {
                filtered_batches.push(filtered);
            }
        }
        
        Ok(Box::new(InMemoryDatasetScanner::new(filtered_batches)))
    }
    
    fn cache(&self) -> Result<Arc<dyn Dataset>> {
        // Already cached, just return self
        Ok(Arc::new(self.clone()))
    }
    
    fn repartition(&self, partition_count: usize) -> Result<Arc<dyn Dataset>> {
        if partition_count == 0 {
            return Err(crate::error::Error::InvalidArgument(
                "Partition count must be greater than 0".into(),
            ));
        }
        
        if self.batches.is_empty() {
            return Ok(Arc::new(self.clone()));
        }
        
        // Calculate target rows per partition
        let target_rows_per_partition = (self.row_count + partition_count - 1) / partition_count;
        
        // Redistribute rows across new partitions
        let mut new_batches = Vec::with_capacity(partition_count);
        let mut current_rows = 0;
        let mut current_batch_rows = Vec::new();
        
        for batch in &self.batches {
            for row_idx in 0..batch.row_count() {
                if current_rows >= target_rows_per_partition && !current_batch_rows.is_empty() {
                    // Finish current partition
                    new_batches.push(batch.project(&current_batch_rows)?);
                    current_batch_rows.clear();
                    current_rows = 0;
                }
                
                // Add row to current partition
                current_batch_rows.push(row_idx);
                current_rows += 1;
            }
        }
        
        // Add final partition if needed
        if !current_batch_rows.is_empty() {
            new_batches.push(self.batches.last().unwrap().project(&current_batch_rows)?);
        }
        
        Ok(Arc::new(InMemoryDataset::from_batches(self.schema.clone(), new_batches)?))
    }
    
    fn with_schema(&self, schema: Arc<Schema>) -> Result<Arc<dyn Dataset>> {
        // Validate that the new schema is compatible with the existing one
        if schema.len() != self.schema.len() {
            return Err(crate::error::Error::SchemaMismatch(
                "New schema has different number of fields".into(),
            ));
        }
        
        // Create a new dataset with the updated schema
        Ok(Arc::new(InMemoryDataset {
            schema,
            batches: self.batches.clone(),
            row_count: self.row_count,
        }))
    }
}

impl Clone for InMemoryDataset {
    fn clone(&self) -> Self {
        Self {
            schema: self.schema.clone(),
            batches: self.batches.clone(),
            row_count: self.row_count,
        }
    }
}

/// A scanner for in-memory datasets
pub struct InMemoryDatasetScanner {
    /// The batches to scan
    batches: Vec<RecordBatch>,
    
    /// The current batch index
    current_index: usize,
}

impl InMemoryDatasetScanner {
    /// Create a new scanner
    pub fn new(batches: Vec<RecordBatch>) -> Self {
        Self {
            batches,
            current_index: 0,
        }
    }
}

impl RecordBatchSource for InMemoryDatasetScanner {
    fn schema(&self) -> Arc<Schema> {
        if let Some(first_batch) = self.batches.first() {
            first_batch.schema().clone()
        } else {
            // Empty schema if no batches
            Arc::new(Schema::new(Vec::new()))
        }
    }
    
    fn next_batch(&mut self, _max_batch_size: usize) -> Result<Option<RecordBatch>> {
        if self.current_index >= self.batches.len() {
            return Ok(None);
        }
        
        let batch = self.batches[self.current_index].clone();
        self.current_index += 1;
        
        Ok(Some(batch))
    }
    
    fn row_count_hint(&self) -> Option<usize> {
        Some(self.batches.iter().map(|b| b.row_count()).sum())
    }
    
    fn memory_usage(&self) -> usize {
        self.batches.iter().map(|b| b.memory_usage()).sum()
    }
    
    fn reset(&mut self) -> Result<()> {
        self.current_index = 0;
        Ok(())
    }
}