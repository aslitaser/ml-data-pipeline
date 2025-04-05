//! Arrow IPC format support with zero-copy access
//!
//! This module provides readers and writers for the Arrow IPC format,
//! which is particularly efficient for columnar data exchange.

mod reader;
mod writer;
mod schema;

pub use reader::{ArrowReader, ArrowReaderOptions};
pub use writer::{ArrowWriter, ArrowWriterOptions};

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Arc;
    use std::io::Cursor;
    use ml_data_core::{RecordBatch, RecordBatchSource, Schema, Field, DataType};
    
    #[test]
    fn test_arrow_schema_conversion() {
        let schema = Arc::new(Schema::new(vec![
            Field::new("id", DataType::Int32, false),
            Field::new("name", DataType::String, true),
            Field::new("value", DataType::Float64, false),
            Field::new("valid", DataType::Boolean, true),
        ]));
        
        let arrow_schema = schema::convert_to_arrow_schema(&schema);
        let roundtrip = schema::convert_from_arrow_schema(&arrow_schema);
        
        assert_eq!(schema.len(), roundtrip.len());
        for i in 0..schema.len() {
            assert_eq!(schema.field(i).name(), roundtrip.field(i).name());
            // Note: exact type matching depends on conversion details
        }
    }
}