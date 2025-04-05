//! Schema conversion between ML data types and Parquet types

use std::collections::HashMap;
use std::sync::Arc;

use parquet::basic::{
    ConvertedType, Repetition, Type as PhysicalType,
    LogicalType, TimeUnit as ParquetTimeUnit,
};
use parquet::schema::types::{
    Type as SchemaType, TypePtr, GroupTypeBuilder, PrimitiveTypeBuilder,
};

use ml_data_core::{Schema, Field, DataType, TimeUnit};

/// Convert ML Schema to Parquet Schema Type
pub fn convert_to_parquet_schema(schema: &Arc<Schema>) -> TypePtr {
    let mut fields = Vec::new();
    
    for field in schema.fields() {
        fields.push(convert_field_to_parquet(field));
    }
    
    Arc::new(
        SchemaType::group_type_builder("schema")
            .with_fields(&mut fields)
            .build()
            .unwrap()
    )
}

/// Convert Parquet Schema Type to ML Schema
pub fn convert_from_parquet_schema(schema_type: &SchemaType) -> Schema {
    let mut fields = Vec::new();
    
    if let SchemaType::GroupType { fields: schema_fields, .. } = schema_type {
        for field in schema_fields {
            fields.push(convert_parquet_to_field(field));
        }
    }
    
    Schema::new(fields)
}

/// Convert ML Field to Parquet Type
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
                .with_logical_type(Some(LogicalType::Integer { 
                    bit_width: 8, 
                    is_signed: true 
                }))
                .build()
                .unwrap()
        ),
        DataType::Int16 => Arc::new(
            SchemaType::primitive_type_builder(name, PhysicalType::INT32)
                .with_repetition(repetition)
                .with_converted_type(ConvertedType::INT_16)
                .with_logical_type(Some(LogicalType::Integer { 
                    bit_width: 16, 
                    is_signed: true 
                }))
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
                .with_logical_type(Some(LogicalType::Integer { 
                    bit_width: 8, 
                    is_signed: false 
                }))
                .build()
                .unwrap()
        ),
        DataType::UInt16 => Arc::new(
            SchemaType::primitive_type_builder(name, PhysicalType::INT32)
                .with_repetition(repetition)
                .with_converted_type(ConvertedType::UINT_16)
                .with_logical_type(Some(LogicalType::Integer { 
                    bit_width: 16, 
                    is_signed: false 
                }))
                .build()
                .unwrap()
        ),
        DataType::UInt32 => Arc::new(
            SchemaType::primitive_type_builder(name, PhysicalType::INT32)
                .with_repetition(repetition)
                .with_converted_type(ConvertedType::UINT_32)
                .with_logical_type(Some(LogicalType::Integer { 
                    bit_width: 32, 
                    is_signed: false 
                }))
                .build()
                .unwrap()
        ),
        DataType::UInt64 => Arc::new(
            SchemaType::primitive_type_builder(name, PhysicalType::INT64)
                .with_repetition(repetition)
                .with_converted_type(ConvertedType::UINT_64)
                .with_logical_type(Some(LogicalType::Integer { 
                    bit_width: 64, 
                    is_signed: false 
                }))
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
                .with_logical_type(Some(LogicalType::String))
                .build()
                .unwrap()
        ),
        DataType::Binary => Arc::new(
            SchemaType::primitive_type_builder(name, PhysicalType::BYTE_ARRAY)
                .with_repetition(repetition)
                .build()
                .unwrap()
        ),
        DataType::FixedSizeBinary(size) => Arc::new(
            SchemaType::primitive_type_builder(name, PhysicalType::FIXED_LEN_BYTE_ARRAY)
                .with_repetition(repetition)
                .with_length(*size as i32)
                .build()
                .unwrap()
        ),
        DataType::Date32 => Arc::new(
            SchemaType::primitive_type_builder(name, PhysicalType::INT32)
                .with_repetition(repetition)
                .with_converted_type(ConvertedType::DATE)
                .with_logical_type(Some(LogicalType::Date))
                .build()
                .unwrap()
        ),
        DataType::Date64 => Arc::new(
            SchemaType::primitive_type_builder(name, PhysicalType::INT64)
                .with_repetition(repetition)
                .with_converted_type(ConvertedType::TIMESTAMP_MILLIS)
                .with_logical_type(Some(LogicalType::Timestamp { 
                    is_adjusted_to_utc: false, 
                    unit: ParquetTimeUnit::MILLIS 
                }))
                .build()
                .unwrap()
        ),
        DataType::Timestamp(unit, tz) => {
            let parquet_unit = match unit {
                TimeUnit::Second => ParquetTimeUnit::MILLIS, // Parquet doesn't support seconds directly
                TimeUnit::Millisecond => ParquetTimeUnit::MILLIS,
                TimeUnit::Microsecond => ParquetTimeUnit::MICROS,
                TimeUnit::Nanosecond => ParquetTimeUnit::NANOS,
            };
            
            let is_utc = tz.as_ref().map(|tz| tz == "UTC").unwrap_or(false);
            
            let converted = match unit {
                TimeUnit::Millisecond => ConvertedType::TIMESTAMP_MILLIS,
                TimeUnit::Microsecond => ConvertedType::TIMESTAMP_MICROS,
                _ => ConvertedType::TIMESTAMP_MICROS, // Best effort
            };
            
            Arc::new(
                SchemaType::primitive_type_builder(name, PhysicalType::INT64)
                    .with_repetition(repetition)
                    .with_converted_type(converted)
                    .with_logical_type(Some(LogicalType::Timestamp { 
                        is_adjusted_to_utc: is_utc, 
                        unit: parquet_unit
                    }))
                    .build()
                    .unwrap()
            )
        },
        DataType::Decimal(precision, scale) => Arc::new(
            SchemaType::primitive_type_builder(name, PhysicalType::FIXED_LEN_BYTE_ARRAY)
                .with_repetition(repetition)
                .with_converted_type(ConvertedType::DECIMAL)
                .with_logical_type(Some(LogicalType::Decimal { 
                    precision: *precision as i32, 
                    scale: *scale as i32 
                }))
                .with_precision(*precision as i32)
                .with_scale(*scale as i32)
                .with_length(16) // Max size needed for decimal
                .build()
                .unwrap()
        ),
        DataType::List(item_type) => {
            let item_field = Field::new("item", *item_type.clone(), true);
            let item_type = convert_field_to_parquet(&item_field);
            
            Arc::new(
                SchemaType::group_type_builder(name)
                    .with_repetition(repetition)
                    .with_converted_type(ConvertedType::LIST)
                    .with_logical_type(Some(LogicalType::List))
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
        DataType::Dictionary(key_type, value_type) => {
            // For dictionaries, we'll just represent the value type in Parquet
            let inner_field = Field::new(name, *value_type.clone(), nullable);
            convert_field_to_parquet(&inner_field)
        },
        // Unsupported types - convert to string
        _ => Arc::new(
            SchemaType::primitive_type_builder(name, PhysicalType::BYTE_ARRAY)
                .with_repetition(repetition)
                .with_converted_type(ConvertedType::UTF8)
                .with_logical_type(Some(LogicalType::String))
                .build()
                .unwrap()
        ),
    }
}

/// Convert Parquet Type to ML Field
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
            logical_type, 
            ..
        } => {
            match physical_type {
                PhysicalType::BOOLEAN => DataType::Boolean,
                PhysicalType::INT32 => {
                    if let Some(logical) = logical_type {
                        match logical {
                            LogicalType::Integer { bit_width, is_signed } => {
                                match (bit_width, is_signed) {
                                    (8, true) => DataType::Int8,
                                    (16, true) => DataType::Int16,
                                    (8, false) => DataType::UInt8,
                                    (16, false) => DataType::UInt16,
                                    (32, false) => DataType::UInt32,
                                    _ => DataType::Int32,
                                }
                            },
                            LogicalType::Date => DataType::Date32,
                            _ => {
                                // Check converted type for legacy files
                                match converted_type {
                                    Some(ConvertedType::INT_8) => DataType::Int8,
                                    Some(ConvertedType::INT_16) => DataType::Int16,
                                    Some(ConvertedType::UINT_8) => DataType::UInt8,
                                    Some(ConvertedType::UINT_16) => DataType::UInt16,
                                    Some(ConvertedType::UINT_32) => DataType::UInt32,
                                    Some(ConvertedType::DATE) => DataType::Date32,
                                    _ => DataType::Int32,
                                }
                            }
                        }
                    } else {
                        // Check converted type for legacy files
                        match converted_type {
                            Some(ConvertedType::INT_8) => DataType::Int8,
                            Some(ConvertedType::INT_16) => DataType::Int16,
                            Some(ConvertedType::UINT_8) => DataType::UInt8,
                            Some(ConvertedType::UINT_16) => DataType::UInt16,
                            Some(ConvertedType::UINT_32) => DataType::UInt32,
                            Some(ConvertedType::DATE) => DataType::Date32,
                            _ => DataType::Int32,
                        }
                    }
                },
                PhysicalType::INT64 => {
                    if let Some(logical) = logical_type {
                        match logical {
                            LogicalType::Integer { bit_width: 64, is_signed: false } => {
                                DataType::UInt64
                            },
                            LogicalType::Timestamp { unit, .. } => {
                                let time_unit = match unit {
                                    ParquetTimeUnit::MILLIS => TimeUnit::Millisecond,
                                    ParquetTimeUnit::MICROS => TimeUnit::Microsecond,
                                    ParquetTimeUnit::NANOS => TimeUnit::Nanosecond,
                                };
                                DataType::Timestamp(time_unit, None)
                            },
                            _ => DataType::Int64,
                        }
                    } else {
                        // Check converted type for legacy files
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
                    }
                },
                PhysicalType::FLOAT => DataType::Float32,
                PhysicalType::DOUBLE => DataType::Float64,
                PhysicalType::BYTE_ARRAY => {
                    if let Some(logical) = logical_type {
                        match logical {
                            LogicalType::String => DataType::String,
                            _ => match converted_type {
                                Some(ConvertedType::UTF8) => DataType::String,
                                _ => DataType::Binary,
                            },
                        }
                    } else {
                        match converted_type {
                            Some(ConvertedType::UTF8) => DataType::String,
                            _ => DataType::Binary,
                        }
                    }
                },
                PhysicalType::FIXED_LEN_BYTE_ARRAY => {
                    let length = parquet_type.get_physical_type_length();
                    
                    if let Some(logical) = logical_type {
                        match logical {
                            LogicalType::Decimal { precision, scale } => {
                                DataType::Decimal(*precision as usize, *scale as usize)
                            },
                            _ => {
                                // Check converted type for legacy files
                                match converted_type {
                                    Some(ConvertedType::DECIMAL) => {
                                        let precision = parquet_type.get_precision();
                                        let scale = parquet_type.get_scale();
                                        DataType::Decimal(precision as usize, scale as usize)
                                    },
                                    _ => DataType::FixedSizeBinary(length as usize),
                                }
                            }
                        }
                    } else {
                        // Check converted type for legacy files
                        match converted_type {
                            Some(ConvertedType::DECIMAL) => {
                                let precision = parquet_type.get_precision();
                                let scale = parquet_type.get_scale();
                                DataType::Decimal(precision as usize, scale as usize)
                            },
                            _ => DataType::FixedSizeBinary(length as usize),
                        }
                    }
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
    
    // Create field with metadata
    let mut field = Field::new(name, data_type, nullable);
    
    // Add any metadata from Parquet
    let mut metadata = HashMap::new();
    for (key, value) in parquet_type.get_field_id().key_value_metadata() {
        field.set_metadata_value(key, value);
    }
    
    field
}

/// Check if a Parquet type is a LIST logical type
fn is_list_type(parquet_type: &SchemaType) -> bool {
    if let Some(ConvertedType::LIST) = parquet_type.get_basic_info().converted_type() {
        return true;
    }
    
    if let Some(LogicalType::List) = parquet_type.get_logical_type() {
        return true;
    }
    
    false
}