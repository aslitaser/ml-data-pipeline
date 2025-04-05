//! Schema conversion between ML data types and Arrow types

use std::collections::HashMap;
use std::sync::Arc;

use parquet::arrow::schema::{
    parquet_to_arrow_schema, 
    parquet_to_arrow_field_type, 
    parquet_to_arrow_schema_with_options
};
use parquet::basic::{ConvertedType, Repetition, Type as PhysicalType};
use parquet::schema::types::{Type as SchemaType, TypePtr};

use ml_data_core::{Schema, Field, DataType, TimeUnit};

// This is somewhat of a roundabout approach, but the most practical way to do
// Arrow schema conversion is to use Parquet's API, since that's what we have
// available as a dependency.

/// Convert ML Schema to Arrow Schema
pub fn convert_to_arrow_schema(schema: &Arc<Schema>) -> TypePtr {
    // First convert to Parquet schema
    let mut fields = Vec::new();
    
    for field in schema.fields() {
        fields.push(convert_field_to_parquet(field));
    }
    
    let parquet_schema = Arc::new(
        SchemaType::group_type_builder("schema")
            .with_fields(&mut fields)
            .build()
            .unwrap()
    );
    
    // Then convert to Arrow schema
    parquet_schema
}

/// Convert Arrow Schema to ML Schema
pub fn convert_from_arrow_schema(arrow_schema: &SchemaType) -> Schema {
    let mut fields = Vec::new();
    
    if let SchemaType::GroupType { fields: schema_fields, .. } = arrow_schema {
        for field in schema_fields {
            fields.push(convert_parquet_to_field(field));
        }
    }
    
    Schema::new(fields)
}

/// Convert ML Field to Parquet Type (for Arrow schema conversion)
fn convert_field_to_parquet(field: &Field) -> TypePtr {
    let name = field.name();
    let nullable = field.is_nullable();
    let repetition = if nullable { 
        Repetition::OPTIONAL 
    } else { 
        Repetition::REQUIRED 
    };
    
    match field.data_type() {
        DataType::Boolean => Arc::new(
            SchemaType::primitive_type_builder(name, PhysicalType::BOOLEAN)
                .with_repetition(repetition)
                .build()
                .unwrap()
        ),
        DataType::Int8 => Arc::new(
            SchemaType::primitive_type_builder(name, PhysicalType::INT32)
                .with_repetition(repetition)
                .with_converted_type(ConvertedType::INT_8)
                .build()
                .unwrap()
        ),
        DataType::Int16 => Arc::new(
            SchemaType::primitive_type_builder(name, PhysicalType::INT32)
                .with_repetition(repetition)
                .with_converted_type(ConvertedType::INT_16)
                .build()
                .unwrap()
        ),
        DataType::Int32 => Arc::new(
            SchemaType::primitive_type_builder(name, PhysicalType::INT32)
                .with_repetition(repetition)
                .build()
                .unwrap()
        ),
        DataType::Int64 => Arc::new(
            SchemaType::primitive_type_builder(name, PhysicalType::INT64)
                .with_repetition(repetition)
                .build()
                .unwrap()
        ),
        DataType::UInt8 => Arc::new(
            SchemaType::primitive_type_builder(name, PhysicalType::INT32)
                .with_repetition(repetition)
                .with_converted_type(ConvertedType::UINT_8)
                .build()
                .unwrap()
        ),
        DataType::UInt16 => Arc::new(
            SchemaType::primitive_type_builder(name, PhysicalType::INT32)
                .with_repetition(repetition)
                .with_converted_type(ConvertedType::UINT_16)
                .build()
                .unwrap()
        ),
        DataType::UInt32 => Arc::new(
            SchemaType::primitive_type_builder(name, PhysicalType::INT32)
                .with_repetition(repetition)
                .with_converted_type(ConvertedType::UINT_32)
                .build()
                .unwrap()
        ),
        DataType::UInt64 => Arc::new(
            SchemaType::primitive_type_builder(name, PhysicalType::INT64)
                .with_repetition(repetition)
                .with_converted_type(ConvertedType::UINT_64)
                .build()
                .unwrap()
        ),
        DataType::Float32 => Arc::new(
            SchemaType::primitive_type_builder(name, PhysicalType::FLOAT)
                .with_repetition(repetition)
                .build()
                .unwrap()
        ),
        DataType::Float64 => Arc::new(
            SchemaType::primitive_type_builder(name, PhysicalType::DOUBLE)
                .with_repetition(repetition)
                .build()
                .unwrap()
        ),
        DataType::String => Arc::new(
            SchemaType::primitive_type_builder(name, PhysicalType::BYTE_ARRAY)
                .with_repetition(repetition)
                .with_converted_type(ConvertedType::UTF8)
                .build()
                .unwrap()
        ),
        DataType::Binary => Arc::new(
            SchemaType::primitive_type_builder(name, PhysicalType::BYTE_ARRAY)
                .with_repetition(repetition)
                .build()
                .unwrap()
        ),
        DataType::Date32 => Arc::new(
            SchemaType::primitive_type_builder(name, PhysicalType::INT32)
                .with_repetition(repetition)
                .with_converted_type(ConvertedType::DATE)
                .build()
                .unwrap()
        ),
        DataType::Date64 => Arc::new(
            SchemaType::primitive_type_builder(name, PhysicalType::INT64)
                .with_repetition(repetition)
                .with_converted_type(ConvertedType::TIMESTAMP_MILLIS)
                .build()
                .unwrap()
        ),
        DataType::Timestamp(unit, _) => {
            let converted = match unit {
                TimeUnit::Millisecond => ConvertedType::TIMESTAMP_MILLIS,
                TimeUnit::Microsecond => ConvertedType::TIMESTAMP_MICROS,
                _ => ConvertedType::TIMESTAMP_MICROS, // Best effort
            };
            
            Arc::new(
                SchemaType::primitive_type_builder(name, PhysicalType::INT64)
                    .with_repetition(repetition)
                    .with_converted_type(converted)
                    .build()
                    .unwrap()
            )
        },
        DataType::List(item_type) => {
            let item_field = Field::new("item", *item_type.clone(), true);
            let item_type = convert_field_to_parquet(&item_field);
            
            Arc::new(
                SchemaType::group_type_builder(name)
                    .with_repetition(repetition)
                    .with_converted_type(ConvertedType::LIST)
                    .with_fields(&mut vec![
                        Arc::new(
                            SchemaType::group_type_builder("list")
                                .with_repetition(Repetition::REPEATED)
                                .with_fields(&mut vec![item_type])
                                .build()
                                .unwrap()
                        )
                    ])
                    .build()
                    .unwrap()
            )
        },
        DataType::Struct(fields) => {
            let mut parquet_fields = Vec::new();
            for field in fields {
                parquet_fields.push(convert_field_to_parquet(field));
            }
            
            Arc::new(
                SchemaType::group_type_builder(name)
                    .with_repetition(repetition)
                    .with_fields(&mut parquet_fields)
                    .build()
                    .unwrap()
            )
        },
        // Default to string for unsupported types
        _ => Arc::new(
            SchemaType::primitive_type_builder(name, PhysicalType::BYTE_ARRAY)
                .with_repetition(repetition)
                .with_converted_type(ConvertedType::UTF8)
                .build()
                .unwrap()
        ),
    }
}

/// Convert Parquet Type to ML Field (from Arrow schema conversion)
fn convert_parquet_to_field(parquet_type: &SchemaType) -> Field {
    let name = parquet_type.name();
    let nullable = match parquet_type.get_basic_info().repetition() {
        Repetition::OPTIONAL => true,
        Repetition::REPEATED => true,
        Repetition::REQUIRED => false,
    };
    
    let data_type = match parquet_type {
        SchemaType::PrimitiveType { 
            physical_type, 
            converted_type, 
            ..
        } => {
            match physical_type {
                PhysicalType::BOOLEAN => DataType::Boolean,
                PhysicalType::INT32 => {
                    match converted_type {
                        Some(ConvertedType::INT_8) => DataType::Int8,
                        Some(ConvertedType::INT_16) => DataType::Int16,
                        Some(ConvertedType::UINT_8) => DataType::UInt8,
                        Some(ConvertedType::UINT_16) => DataType::UInt16,
                        Some(ConvertedType::UINT_32) => DataType::UInt32,
                        Some(ConvertedType::DATE) => DataType::Date32,
                        _ => DataType::Int32,
                    }
                },
                PhysicalType::INT64 => {
                    match converted_type {
                        Some(ConvertedType::UINT_64) => DataType::UInt64,
                        Some(ConvertedType::TIMESTAMP_MILLIS) => {
                            DataType::Timestamp(TimeUnit::Millisecond, None)
                        },
                        Some(ConvertedType::TIMESTAMP_MICROS) => {
                            DataType::Timestamp(TimeUnit::Microsecond, None)
                        },
                        _ => DataType::Int64,
                    }
                },
                PhysicalType::FLOAT => DataType::Float32,
                PhysicalType::DOUBLE => DataType::Float64,
                PhysicalType::BYTE_ARRAY => {
                    match converted_type {
                        Some(ConvertedType::UTF8) => DataType::String,
                        _ => DataType::Binary,
                    }
                },
                PhysicalType::FIXED_LEN_BYTE_ARRAY => {
                    DataType::Binary
                },
                PhysicalType::INT96 => {
                    // INT96 is usually a timestamp in older Parquet files
                    DataType::Timestamp(TimeUnit::Nanosecond, None)
                },
            }
        },
        SchemaType::GroupType { fields, .. } => {
            if is_list_type(parquet_type) {
                // Extract list element type
                if let Some(list_group) = fields.get(0) {
                    if let SchemaType::GroupType { fields: list_fields, .. } = list_group.as_ref() {
                        if let Some(element_type) = list_fields.get(0) {
                            let element_field = convert_parquet_to_field(element_type);
                            DataType::List(Box::new(element_field.data_type().clone()))
                        } else {
                            // Empty list, default to string
                            DataType::List(Box::new(DataType::String))
                        }
                    } else {
                        // Malformed list, default to string
                        DataType::List(Box::new(DataType::String))
                    }
                } else {
                    // Empty list, default to string
                    DataType::List(Box::new(DataType::String))
                }
            } else {
                // Regular struct type
                let struct_fields = fields.iter()
                    .map(|f| convert_parquet_to_field(f))
                    .collect();
                    
                DataType::Struct(struct_fields)
            }
        },
    };
    
    Field::new(name, data_type, nullable)
}

/// Check if a Parquet type is a LIST logical type
fn is_list_type(parquet_type: &SchemaType) -> bool {
    if let Some(ConvertedType::LIST) = parquet_type.get_basic_info().converted_type() {
        return true;
    }
    
    false
}