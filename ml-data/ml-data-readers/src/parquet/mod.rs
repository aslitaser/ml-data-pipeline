//! Parquet reader with columnar access and predicate pushdown
//!
//! This module provides a memory-efficient Parquet reader with columnar access,
//! predicate pushdown, and memory mapping capabilities.

mod reader;
mod schema;
mod predicates;
mod writer;

pub use reader::{ParquetReader, ParquetReaderOptions};
pub use predicates::{Predicate, PredicateBuilder, ColumnPredicate};
pub use writer::ParquetWriter;

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Arc;
    use ml_data_core::{RecordBatch, RecordBatchSource, Schema, Field, DataType};
    use crate::error::Result;
    use std::io::Cursor;
    
    #[test]
    fn test_parquet_schema_conversion() {
        let schema = Arc::new(Schema::new(vec![
            Field::new("id", DataType::Int32, false),
            Field::new("name", DataType::String, true),
            Field::new("value", DataType::Float64, false),
            Field::new("valid", DataType::Boolean, true),
        ]));
        
        let parquet_schema = schema::convert_to_parquet_schema(&schema);
        let roundtrip = schema::convert_from_parquet_schema(&parquet_schema);
        
        assert_eq!(schema.len(), roundtrip.len());
        for i in 0..schema.len() {
            assert_eq!(schema.field(i).name(), roundtrip.field(i).name());
            // Note: exact type matching depends on conversion details
        }
    }
}