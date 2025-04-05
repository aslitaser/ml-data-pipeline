//! Error types for ML data pipelines

use std::io;
use thiserror::Error;

/// Result type for ML data pipeline operations
pub type Result<T> = std::result::Result<T, Error>;

/// Error type for ML data pipeline operations
#[derive(Error, Debug)]
pub enum Error {
    /// IO error during file operations
    #[error("IO error: {0}")]
    Io(#[from] io::Error),

    /// Memory allocation failed
    #[error("Memory allocation failed")]
    MemoryAllocationFailed,

    /// Index out of bounds
    #[error("Index out of bounds")]
    IndexOutOfBounds,

    /// Invalid argument
    #[error("Invalid argument: {0}")]
    InvalidArgument(String),

    /// Invalid operation
    #[error("Invalid operation: {0}")]
    InvalidOperation(String),

    /// Schema mismatch
    #[error("Schema mismatch: {0}")]
    SchemaMismatch(String),

    /// Data type mismatch
    #[error("Data type mismatch: {0}")]
    TypeMismatch(String),

    /// Memory budget exceeded
    #[error("Memory budget exceeded: requested {requested} bytes, available {available} bytes")]
    MemoryBudgetExceeded {
        /// Requested memory in bytes
        requested: usize,
        /// Available memory in bytes
        available: usize,
    },

    /// Serialization error
    #[error("Serialization error: {0}")]
    Serialization(#[from] bincode::Error),

    /// Source exhausted (no more data)
    #[error("Source exhausted")]
    SourceExhausted,

    /// Pipeline execution error
    #[error("Pipeline execution error: {0}")]
    PipelineExecution(String),

    /// Transformation error
    #[error("Transformation error: {0}")]
    TransformationError(String),

    /// Feature not implemented
    #[error("Feature not implemented: {0}")]
    NotImplemented(String),

    /// Layout error (alignment, stride, etc.)
    #[error("Memory layout error: {0}")]
    LayoutError(String),
}