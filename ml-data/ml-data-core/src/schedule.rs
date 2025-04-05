//! Pipeline execution scheduling

use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::time::Duration;

use crate::error::{Error, Result};
use crate::memory::MemoryBudget;
use crate::sink::Sink;
use crate::source::Source;
use crate::transform::Transform;

/// Configuration for a pipeline
#[derive(Debug, Clone)]
pub struct PipelineConfig {
    /// Batch size for processing
    pub batch_size: usize,
    
    /// Number of worker threads
    pub worker_threads: usize,
    
    /// Capacity of internal buffers
    pub buffer_capacity: usize,
    
    /// Backpressure threshold (0.0-1.0)
    pub backpressure_threshold: f64,
    
    /// Memory check frequency
    pub memory_check_frequency: Duration,
}

impl Default for PipelineConfig {
    fn default() -> Self {
        Self {
            batch_size: 1024,
            worker_threads: num_cpus::get(),
            buffer_capacity: 10_000,
            backpressure_threshold: 0.8,
            memory_check_frequency: Duration::from_millis(100),
        }
    }
}

/// Statistics from pipeline execution
#[derive(Debug, Clone)]
pub struct PipelineStats {
    /// Number of items processed
    pub items_processed: u64,
    
    /// Number of batches processed
    pub batches_processed: u64,
    
    /// Total execution time
    pub execution_time: Duration,
    
    /// Time spent in source
    pub source_time: Duration,
    
    /// Time spent in transform
    pub transform_time: Duration,
    
    /// Time spent in sink
    pub sink_time: Duration,
    
    /// Peak memory usage
    pub peak_memory: usize,
}

/// Error type for pipeline execution
#[derive(thiserror::Error, Debug)]
pub enum PipelineError {
    /// Error from pipeline source
    #[error("Source error: {0}")]
    Source(String),
    
    /// Error from pipeline transform
    #[error("Transform error: {0}")]
    Transform(String),
    
    /// Error from pipeline sink
    #[error("Sink error: {0}")]
    Sink(String),
    
    /// Pipeline was cancelled
    #[error("Pipeline cancelled")]
    Cancelled,
    
    /// Memory budget exceeded
    #[error("Memory budget exceeded: {0}")]
    MemoryBudgetExceeded(String),
    
    /// Execution error
    #[error("Execution error: {0}")]
    Execution(String),
    
    /// General error
    #[error("{0}")]
    Other(#[from] Error),
}

/// A pipeline that processes data from a source through a transform to a sink
pub struct Pipeline<S, T, K> 
where 
    S: Source,
    T: Transform<Input = S::Item>,
    K: Sink<Item = T::Output>
{
    /// The source of data
    source: S,
    
    /// The transformation to apply
    transform: T,
    
    /// The sink to output to
    sink: K,
    
    /// Memory budget for this pipeline
    budget: Arc<MemoryBudget>,
    
    /// Configuration for this pipeline
    config: PipelineConfig,
    
    /// Whether the pipeline is paused
    paused: Arc<AtomicBool>,
    
    /// Whether the pipeline is cancelled
    cancelled: Arc<AtomicBool>,
}

impl<S, T, K> Pipeline<S, T, K>
where
    S: Source,
    T: Transform<Input = S::Item>,
    K: Sink<Item = T::Output>
{
    /// Create a new pipeline
    pub fn new(source: S, transform: T, sink: K, budget: Arc<MemoryBudget>, config: PipelineConfig) -> Self {
        Self {
            source,
            transform,
            sink,
            budget,
            config,
            paused: Arc::new(AtomicBool::new(false)),
            cancelled: Arc::new(AtomicBool::new(false)),
        }
    }
    
    /// Run the pipeline until completion
    pub fn run(mut self) -> Result<PipelineStats, PipelineError> {
        let start_time = std::time::Instant::now();
        let mut source_time = Duration::default();
        let mut transform_time = Duration::default();
        let mut sink_time = Duration::default();
        
        let mut items_processed = 0;
        let mut batches_processed = 0;
        
        loop {
            // Check if cancelled
            if self.cancelled.load(Ordering::SeqCst) {
                return Err(PipelineError::Cancelled);
            }
            
            // Handle pause if needed
            while self.paused.load(Ordering::SeqCst) {
                if self.cancelled.load(Ordering::SeqCst) {
                    return Err(PipelineError::Cancelled);
                }
                std::thread::sleep(Duration::from_millis(10));
            }
            
            // Check memory pressure
            if self.budget.percent_used() > 95.0 {
                return Err(PipelineError::MemoryBudgetExceeded(
                    format!("Memory usage at {}% of budget", self.budget.percent_used())
                ));
            }
            
            // Apply backpressure if needed
            if self.budget.percent_used() > self.config.backpressure_threshold * 100.0 {
                std::thread::sleep(self.config.memory_check_frequency);
                continue;
            }
            
            // Get next batch from source
            let source_start = std::time::Instant::now();
            let batch = match self.source.next_batch(self.config.batch_size) {
                Ok(Some(batch)) => batch,
                Ok(None) => break, // Source exhausted
                Err(e) => return Err(PipelineError::Source(format!("Source error: {}", e))),
            };
            source_time += source_start.elapsed();
            
            let batch_size = batch.len();
            
            // Transform the batch
            let transform_start = std::time::Instant::now();
            let transformed = match self.transform.transform(batch) {
                Ok(transformed) => transformed,
                Err(e) => return Err(PipelineError::Transform(format!("Transform error: {}", e))),
            };
            transform_time += transform_start.elapsed();
            
            // Send to sink
            let sink_start = std::time::Instant::now();
            if let Err(e) = self.sink.consume(transformed) {
                return Err(PipelineError::Sink(format!("Sink error: {}", e)));
            }
            sink_time += sink_start.elapsed();
            
            // Update stats
            items_processed += batch_size as u64;
            batches_processed += 1;
        }
        
        // Flush the sink
        let sink_start = std::time::Instant::now();
        if let Err(e) = self.sink.flush() {
            return Err(PipelineError::Sink(format!("Sink flush error: {}", e)));
        }
        sink_time += sink_start.elapsed();
        
        let peak_memory = self.budget.stats().peak_usage;
        
        Ok(PipelineStats {
            items_processed,
            batches_processed,
            execution_time: start_time.elapsed(),
            source_time,
            transform_time,
            sink_time,
            peak_memory,
        })
    }
    
    /// Pause the pipeline (finish processing current batches)
    pub fn pause(&self) {
        self.paused.store(true, Ordering::SeqCst);
    }
    
    /// Resume a paused pipeline
    pub fn resume(&self) {
        self.paused.store(false, Ordering::SeqCst);
    }
    
    /// Cancel the pipeline execution
    pub fn cancel(&self) {
        self.cancelled.store(true, Ordering::SeqCst);
    }
    
    /// Get current memory usage statistics
    pub fn memory_stats(&self) -> crate::memory::MemoryStats {
        self.budget.stats()
    }
}