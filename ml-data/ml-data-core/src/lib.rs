//! Core traits, data structures, and abstractions for ML data pipelines
//!
//! This crate provides the foundational components for building memory-efficient
//! data pipelines for machine learning applications. It defines the core traits
//! and interfaces that all other components in the system build upon.

#![warn(missing_docs)]

pub mod buffer;
pub mod column;
pub mod dataset;
pub mod error;
pub mod io;
pub mod memory;
pub mod record_batch;
pub mod schedule;
pub mod schema;
pub mod source;
pub mod transform;
pub mod sink;
pub mod tensor;

// Re-export key types for convenience
pub use buffer::Buffer;
pub use column::Column;
pub use dataset::{Dataset, DatasetBuilder};
pub use error::{Error, Result};
pub use record_batch::RecordBatch;
pub use schema::{DataType, Field, Schema};
pub use source::Source;
pub use transform::Transform;
pub use sink::Sink;
pub use tensor::{DenseTensor, SparseTensor, TensorType};

/// Memory budget and accounting functionality
pub mod budget {
    pub use crate::memory::MemoryBudget;
    pub use crate::memory::MemoryStats;
}

/// Pipeline configuration and execution
pub mod pipeline {
    pub use crate::schedule::Pipeline;
    pub use crate::schedule::PipelineConfig;
    pub use crate::schedule::PipelineError;
    pub use crate::schedule::PipelineStats;
}