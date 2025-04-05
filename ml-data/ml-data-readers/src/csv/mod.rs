//! CSV reader with memory-efficient processing
//!
//! This module provides a memory-efficient CSV reader with chunked processing,
//! memory mapping, and dictionary encoding for categorical values.

mod reader;
mod parser;
mod writer;

pub use reader::{CsvReader, CsvReaderOptions};
pub use parser::CsvParser;
pub use writer::CsvWriter;

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Arc;
    use std::io::Cursor;
    use ml_data_core::{RecordBatch, RecordBatchSource, Schema, Field, DataType};
    
    #[test]
    fn test_csv_reader_basic() {
        let csv_data = "\
id,name,value
1,Alice,10.5
2,Bob,20.1
3,Charlie,30.9
";
        
        let cursor = Cursor::new(csv_data.as_bytes());
        
        let mut reader = CsvReader::new(
            cursor,
            CsvReaderOptions {
                has_header: true,
                delimiter: b',',
                ..Default::default()
            },
        );
        
        // Read the first batch
        let batch = reader.next_batch(10).unwrap().unwrap();
        
        assert_eq!(batch.num_rows(), 3);
        assert_eq!(batch.num_columns(), 3);
        
        // Verify schema
        let schema = reader.schema();
        assert_eq!(schema.len(), 3);
        assert_eq!(schema.field(0).name(), "id");
        assert_eq!(schema.field(1).name(), "name");
        assert_eq!(schema.field(2).name(), "value");
    }
    
    #[test]
    fn test_csv_reader_no_header() {
        let csv_data = "\
1,Alice,10.5
2,Bob,20.1
3,Charlie,30.9
";
        
        let cursor = Cursor::new(csv_data.as_bytes());
        
        let schema = Arc::new(Schema::new(vec![
            Field::new("id", DataType::Int64, false),
            Field::new("name", DataType::String, false),
            Field::new("value", DataType::Float64, false),
        ]));
        
        let mut reader = CsvReader::new_with_schema(
            cursor,
            schema.clone(),
            CsvReaderOptions {
                has_header: false,
                delimiter: b',',
                ..Default::default()
            },
        );
        
        // Read the first batch
        let batch = reader.next_batch(10).unwrap().unwrap();
        
        assert_eq!(batch.num_rows(), 3);
        assert_eq!(batch.num_columns(), 3);
        
        // Verify schema
        let reader_schema = reader.schema();
        assert_eq!(reader_schema.len(), 3);
        assert_eq!(reader_schema.field(0).name(), "id");
        assert_eq!(reader_schema.field(1).name(), "name");
        assert_eq!(reader_schema.field(2).name(), "value");
    }
}