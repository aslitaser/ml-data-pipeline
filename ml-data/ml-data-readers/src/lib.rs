//! Data source implementations for ML data processing
//! 
//! This crate provides efficient data source implementations for common data formats
//! used in machine learning.

mod error;
mod data_source;
mod factory;
mod string_cache;

#[cfg(feature = "csv")]
pub mod csv;

#[cfg(feature = "parquet")]
pub mod parquet;

#[cfg(feature = "json")]
pub mod json;

#[cfg(feature = "avro")]
pub mod avro;

pub mod common;
pub mod binary;
pub mod text;
pub mod image;
pub mod timeseries;

pub use data_source::{
    DataSource, DataSourceSeek, 
    FileDataSource, StreamDataSource, 
    MemoryDataSource
};
pub use factory::DataSourceFactory;
pub use error::{Error, Result};
pub use string_cache::{StringCache, StringDictionary};

// Re-export core types
pub use ml_data_core::{
    RecordBatch, Schema, Field, DataType,
    Source, RecordBatchSource, SourceFactory,
    error::Result as CoreResult
};