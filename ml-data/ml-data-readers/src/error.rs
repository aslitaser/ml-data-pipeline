//! Error types for data readers

use thiserror::Error;

/// Error type for data readers
#[derive(Error, Debug)]
pub enum Error {
    /// Core library error
    #[error("Core error: {0}")]
    Core(#[from] ml_data_core::error::Error),
    
    /// I/O error
    #[error("I/O error: {0}")]
    Io(#[from] std::io::Error),
    
    /// CSV format error
    #[cfg(feature = "csv")]
    #[error("CSV error: {0}")]
    Csv(#[from] csv::Error),
    
    /// Parquet format error
    #[cfg(feature = "parquet")]
    #[error("Parquet error: {0}")]
    Parquet(String),
    
    /// JSON format error
    #[cfg(feature = "json")]
    #[error("JSON error: {0}")]
    Json(#[from] serde_json::Error),
    
    /// Avro format error
    #[cfg(feature = "avro")]
    #[error("Avro error: {0}")]
    Avro(#[from] apache_avro::Error),
    
    /// SQL error
    #[cfg(feature = "database")]
    #[error("SQL error: {0}")]
    Sql(#[from] sqlx::Error),
    
    /// HTTP error
    #[cfg(feature = "http")]
    #[error("HTTP error: {0}")]
    Http(#[from] reqwest::Error),
    
    /// Schema error
    #[error("Schema error: {0}")]
    Schema(String),
    
    /// Format error
    #[error("Format error: {0}")]
    Format(String),
    
    /// Unsupported operation
    #[error("Unsupported operation: {0}")]
    Unsupported(String),
    
    /// End of data
    #[error("End of data")]
    EndOfData,
    
    /// Invalid argument
    #[error("Invalid argument: {0}")]
    InvalidArgument(String),
    
    /// Other error
    #[error("Error: {0}")]
    Other(String),
}

/// Result type for data readers
pub type Result<T> = std::result::Result<T, Error>;