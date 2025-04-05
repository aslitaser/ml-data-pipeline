//! Predicate pushdown for Parquet readers
//!
//! This module implements predicate pushdown for Parquet readers, allowing
//! filtering at the file level for better performance.

use std::collections::HashMap;
use std::fmt;
use std::sync::Arc;

use parquet::file::metadata::RowGroupMetaData;
use parquet::file::reader::ChunkReader;
use parquet::file::statistics::Statistics as ParquetStats;
use parquet::basic::{Type as PhysicalType, LogicalType};
use parquet::schema::types::Type as SchemaType;

use ml_data_core::{Schema, DataType, Field};

/// Operator for column predicates
#[derive(Debug, Clone, PartialEq)]
pub enum Operator {
    /// Equal
    Eq,
    /// Not equal
    Ne,
    /// Less than
    Lt,
    /// Less than or equal
    Le,
    /// Greater than
    Gt,
    /// Greater than or equal
    Ge,
    /// Is null
    IsNull,
    /// Is not null
    IsNotNull,
    /// In set
    In,
    /// Not in set
    NotIn,
    /// String contains
    Contains,
    /// String starts with
    StartsWith,
    /// String ends with
    EndsWith,
}

impl fmt::Display for Operator {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Operator::Eq => write!(f, "="),
            Operator::Ne => write!(f, "!="),
            Operator::Lt => write!(f, "<"),
            Operator::Le => write!(f, "<="),
            Operator::Gt => write!(f, ">"),
            Operator::Ge => write!(f, ">="),
            Operator::IsNull => write!(f, "IS NULL"),
            Operator::IsNotNull => write!(f, "IS NOT NULL"),
            Operator::In => write!(f, "IN"),
            Operator::NotIn => write!(f, "NOT IN"),
            Operator::Contains => write!(f, "CONTAINS"),
            Operator::StartsWith => write!(f, "STARTS WITH"),
            Operator::EndsWith => write!(f, "ENDS WITH"),
        }
    }
}

/// Scalar value for predicates
#[derive(Debug, Clone, PartialEq)]
pub enum ScalarValue {
    /// Null value
    Null,
    /// Boolean value
    Boolean(bool),
    /// 8-bit integer
    Int8(i8),
    /// 16-bit integer
    Int16(i16),
    /// 32-bit integer
    Int32(i32),
    /// 64-bit integer
    Int64(i64),
    /// 8-bit unsigned integer
    UInt8(u8),
    /// 16-bit unsigned integer
    UInt16(u16),
    /// 32-bit unsigned integer
    UInt32(u32),
    /// 64-bit unsigned integer
    UInt64(u64),
    /// 32-bit float
    Float32(f32),
    /// 64-bit float
    Float64(f64),
    /// String value
    String(String),
    /// Binary value
    Binary(Vec<u8>),
    /// Date (days since epoch)
    Date32(i32),
    /// Date (milliseconds since epoch)
    Date64(i64),
    /// Timestamp (nanoseconds since epoch)
    Timestamp(i64),
    /// List of values
    List(Vec<ScalarValue>),
}

impl ScalarValue {
    /// Convert a value to a ScalarValue based on the data type
    pub fn from_string(value: &str, data_type: &DataType) -> Option<Self> {
        match data_type {
            DataType::Boolean => {
                value.parse::<bool>().ok().map(ScalarValue::Boolean)
            },
            DataType::Int8 => {
                value.parse::<i8>().ok().map(ScalarValue::Int8)
            },
            DataType::Int16 => {
                value.parse::<i16>().ok().map(ScalarValue::Int16)
            },
            DataType::Int32 => {
                value.parse::<i32>().ok().map(ScalarValue::Int32)
            },
            DataType::Int64 => {
                value.parse::<i64>().ok().map(ScalarValue::Int64)
            },
            DataType::UInt8 => {
                value.parse::<u8>().ok().map(ScalarValue::UInt8)
            },
            DataType::UInt16 => {
                value.parse::<u16>().ok().map(ScalarValue::UInt16)
            },
            DataType::UInt32 => {
                value.parse::<u32>().ok().map(ScalarValue::UInt32)
            },
            DataType::UInt64 => {
                value.parse::<u64>().ok().map(ScalarValue::UInt64)
            },
            DataType::Float32 => {
                value.parse::<f32>().ok().map(ScalarValue::Float32)
            },
            DataType::Float64 => {
                value.parse::<f64>().ok().map(ScalarValue::Float64)
            },
            DataType::String => {
                Some(ScalarValue::String(value.to_string()))
            },
            DataType::Date32 => {
                value.parse::<i32>().ok().map(ScalarValue::Date32)
            },
            DataType::Date64 => {
                value.parse::<i64>().ok().map(ScalarValue::Date64)
            },
            DataType::Timestamp(_, _) => {
                value.parse::<i64>().ok().map(ScalarValue::Timestamp)
            },
            _ => None,
        }
    }
    
    /// Get the data type of this scalar value
    pub fn data_type(&self) -> DataType {
        match self {
            ScalarValue::Null => DataType::Null,
            ScalarValue::Boolean(_) => DataType::Boolean,
            ScalarValue::Int8(_) => DataType::Int8,
            ScalarValue::Int16(_) => DataType::Int16,
            ScalarValue::Int32(_) => DataType::Int32,
            ScalarValue::Int64(_) => DataType::Int64,
            ScalarValue::UInt8(_) => DataType::UInt8,
            ScalarValue::UInt16(_) => DataType::UInt16,
            ScalarValue::UInt32(_) => DataType::UInt32,
            ScalarValue::UInt64(_) => DataType::UInt64,
            ScalarValue::Float32(_) => DataType::Float32,
            ScalarValue::Float64(_) => DataType::Float64,
            ScalarValue::String(_) => DataType::String,
            ScalarValue::Binary(_) => DataType::Binary,
            ScalarValue::Date32(_) => DataType::Date32,
            ScalarValue::Date64(_) => DataType::Date64,
            ScalarValue::Timestamp(_) => DataType::Timestamp(ml_data_core::TimeUnit::Nanosecond, None),
            ScalarValue::List(values) => {
                if let Some(first) = values.first() {
                    DataType::List(Box::new(first.data_type()))
                } else {
                    DataType::List(Box::new(DataType::Null))
                }
            },
        }
    }
}

/// Column predicate for filtering data
#[derive(Debug, Clone)]
pub struct ColumnPredicate {
    /// Column name
    pub column: String,
    
    /// Operator
    pub op: Operator,
    
    /// Value for comparison
    pub value: ScalarValue,
    
    /// List of values for IN predicates
    pub value_list: Option<Vec<ScalarValue>>,
}

/// Predicate for filtering data
#[derive(Debug, Clone)]
pub enum Predicate {
    /// Column predicate
    Column(ColumnPredicate),
    
    /// AND of multiple predicates
    And(Vec<Predicate>),
    
    /// OR of multiple predicates
    Or(Vec<Predicate>),
    
    /// NOT of a predicate
    Not(Box<Predicate>),
    
    /// Always true predicate
    AlwaysTrue,
    
    /// Always false predicate
    AlwaysFalse,
}

impl Predicate {
    /// Create a predicate that always evaluates to true
    pub fn always_true() -> Self {
        Predicate::AlwaysTrue
    }
    
    /// Create a predicate that always evaluates to false
    pub fn always_false() -> Self {
        Predicate::AlwaysFalse
    }
    
    /// Combine predicates with AND
    pub fn and(predicates: Vec<Predicate>) -> Self {
        if predicates.is_empty() {
            return Predicate::AlwaysTrue;
        }
        
        if predicates.len() == 1 {
            return predicates.into_iter().next().unwrap();
        }
        
        // Check for any AlwaysFalse predicates
        if predicates.iter().any(|p| matches!(p, Predicate::AlwaysFalse)) {
            return Predicate::AlwaysFalse;
        }
        
        // Filter out AlwaysTrue predicates
        let filtered: Vec<_> = predicates
            .into_iter()
            .filter(|p| !matches!(p, Predicate::AlwaysTrue))
            .collect();
            
        if filtered.is_empty() {
            return Predicate::AlwaysTrue;
        }
        
        if filtered.len() == 1 {
            return filtered.into_iter().next().unwrap();
        }
        
        Predicate::And(filtered)
    }
    
    /// Combine predicates with OR
    pub fn or(predicates: Vec<Predicate>) -> Self {
        if predicates.is_empty() {
            return Predicate::AlwaysFalse;
        }
        
        if predicates.len() == 1 {
            return predicates.into_iter().next().unwrap();
        }
        
        // Check for any AlwaysTrue predicates
        if predicates.iter().any(|p| matches!(p, Predicate::AlwaysTrue)) {
            return Predicate::AlwaysTrue;
        }
        
        // Filter out AlwaysFalse predicates
        let filtered: Vec<_> = predicates
            .into_iter()
            .filter(|p| !matches!(p, Predicate::AlwaysFalse))
            .collect();
            
        if filtered.is_empty() {
            return Predicate::AlwaysFalse;
        }
        
        if filtered.len() == 1 {
            return filtered.into_iter().next().unwrap();
        }
        
        Predicate::Or(filtered)
    }
    
    /// Negate a predicate
    pub fn not(predicate: Predicate) -> Self {
        match predicate {
            Predicate::AlwaysTrue => Predicate::AlwaysFalse,
            Predicate::AlwaysFalse => Predicate::AlwaysTrue,
            Predicate::Not(inner) => *inner,
            _ => Predicate::Not(Box::new(predicate)),
        }
    }
    
    /// Check if this predicate can be pushed down to Parquet
    pub fn can_push_down(&self) -> bool {
        match self {
            Predicate::Column(col_pred) => {
                // Check if the operator is supported for pushdown
                matches!(
                    col_pred.op,
                    Operator::Eq | Operator::Ne | Operator::Lt | Operator::Le |
                    Operator::Gt | Operator::Ge | Operator::IsNull | Operator::IsNotNull
                )
            },
            Predicate::And(predicates) => predicates.iter().all(|p| p.can_push_down()),
            Predicate::Or(predicates) => predicates.iter().all(|p| p.can_push_down()),
            Predicate::Not(inner) => inner.can_push_down(),
            Predicate::AlwaysTrue | Predicate::AlwaysFalse => true,
        }
    }
    
    /// Check if this predicate can skip a row group based on statistics
    pub fn can_skip_row_group(
        &self,
        schema: &Arc<Schema>,
        row_group: &RowGroupMetaData,
        schema_mapping: &HashMap<String, usize>,
    ) -> bool {
        match self {
            Predicate::Column(col_pred) => {
                // Get the column index in the row group
                if let Some(&field_idx) = schema_mapping.get(&col_pred.column) {
                    // Get the field and check its type
                    if let Ok(field) = schema.field_by_name(&col_pred.column) {
                        // Get column statistics from row group
                        let column_chunk = row_group.column(field_idx);
                        
                        if let Some(stats) = column_chunk.statistics() {
                            return evaluate_stats(&col_pred.op, &col_pred.value, stats, field.data_type());
                        }
                    }
                }
                
                // If we can't determine, assume we can't skip
                false
            },
            Predicate::And(predicates) => {
                // Can skip if any predicate can skip
                predicates.iter().any(|p| p.can_skip_row_group(schema, row_group, schema_mapping))
            },
            Predicate::Or(predicates) => {
                // Can skip only if all predicates can skip
                predicates.iter().all(|p| p.can_skip_row_group(schema, row_group, schema_mapping))
            },
            Predicate::Not(inner) => {
                // Negation is tricky - we'll be conservative
                false
            },
            Predicate::AlwaysTrue => false,
            Predicate::AlwaysFalse => true,
        }
    }
}

/// Builder for constructing predicates
pub struct PredicateBuilder {
    /// Schema used for type information
    schema: Arc<Schema>,
}

impl PredicateBuilder {
    /// Create a new predicate builder
    pub fn new(schema: Arc<Schema>) -> Self {
        Self { schema }
    }
    
    /// Create a column = value predicate
    pub fn eq<S: Into<String>, T: Into<String>>(&self, column: S, value: T) -> Predicate {
        self.build_column_predicate(column.into(), Operator::Eq, value.into())
    }
    
    /// Create a column != value predicate
    pub fn ne<S: Into<String>, T: Into<String>>(&self, column: S, value: T) -> Predicate {
        self.build_column_predicate(column.into(), Operator::Ne, value.into())
    }
    
    /// Create a column < value predicate
    pub fn lt<S: Into<String>, T: Into<String>>(&self, column: S, value: T) -> Predicate {
        self.build_column_predicate(column.into(), Operator::Lt, value.into())
    }
    
    /// Create a column <= value predicate
    pub fn le<S: Into<String>, T: Into<String>>(&self, column: S, value: T) -> Predicate {
        self.build_column_predicate(column.into(), Operator::Le, value.into())
    }
    
    /// Create a column > value predicate
    pub fn gt<S: Into<String>, T: Into<String>>(&self, column: S, value: T) -> Predicate {
        self.build_column_predicate(column.into(), Operator::Gt, value.into())
    }
    
    /// Create a column >= value predicate
    pub fn ge<S: Into<String>, T: Into<String>>(&self, column: S, value: T) -> Predicate {
        self.build_column_predicate(column.into(), Operator::Ge, value.into())
    }
    
    /// Create a column IS NULL predicate
    pub fn is_null<S: Into<String>>(&self, column: S) -> Predicate {
        let column = column.into();
        
        Predicate::Column(ColumnPredicate {
            column,
            op: Operator::IsNull,
            value: ScalarValue::Null,
            value_list: None,
        })
    }
    
    /// Create a column IS NOT NULL predicate
    pub fn is_not_null<S: Into<String>>(&self, column: S) -> Predicate {
        let column = column.into();
        
        Predicate::Column(ColumnPredicate {
            column,
            op: Operator::IsNotNull,
            value: ScalarValue::Null,
            value_list: None,
        })
    }
    
    /// Create a column IN (values) predicate
    pub fn r#in<S: Into<String>, T: Into<String>>(&self, column: S, values: Vec<T>) -> Predicate {
        let column = column.into();
        
        // Try to convert all values based on the column type
        if let Ok(field) = self.schema.field_by_name(&column) {
            let data_type = field.data_type();
            
            let value_list: Vec<_> = values
                .into_iter()
                .filter_map(|v| ScalarValue::from_string(&v.into(), data_type))
                .collect();
                
            if value_list.is_empty() {
                // Empty IN list never matches anything
                return Predicate::AlwaysFalse;
            }
            
            return Predicate::Column(ColumnPredicate {
                column,
                op: Operator::In,
                value: ScalarValue::Null, // Placeholder
                value_list: Some(value_list),
            });
        }
        
        // If column not found, return predicate that always fails
        Predicate::AlwaysFalse
    }
    
    /// Create a column NOT IN (values) predicate
    pub fn not_in<S: Into<String>, T: Into<String>>(&self, column: S, values: Vec<T>) -> Predicate {
        let column = column.into();
        
        // Try to convert all values based on the column type
        if let Ok(field) = self.schema.field_by_name(&column) {
            let data_type = field.data_type();
            
            let value_list: Vec<_> = values
                .into_iter()
                .filter_map(|v| ScalarValue::from_string(&v.into(), data_type))
                .collect();
                
            if value_list.is_empty() {
                // Empty NOT IN list matches everything
                return Predicate::AlwaysTrue;
            }
            
            return Predicate::Column(ColumnPredicate {
                column,
                op: Operator::NotIn,
                value: ScalarValue::Null, // Placeholder
                value_list: Some(value_list),
            });
        }
        
        // If column not found, return predicate that always fails
        Predicate::AlwaysFalse
    }
    
    /// Create a column CONTAINS value predicate (string starts with)
    pub fn contains<S: Into<String>, T: Into<String>>(&self, column: S, value: T) -> Predicate {
        let column = column.into();
        let value_str = value.into();
        
        // This is only valid for string columns
        if let Ok(field) = self.schema.field_by_name(&column) {
            if let DataType::String = field.data_type() {
                return Predicate::Column(ColumnPredicate {
                    column,
                    op: Operator::Contains,
                    value: ScalarValue::String(value_str),
                    value_list: None,
                });
            }
        }
        
        // If not a string column, return predicate that always fails
        Predicate::AlwaysFalse
    }
    
    /// Create a column STARTS WITH value predicate
    pub fn starts_with<S: Into<String>, T: Into<String>>(&self, column: S, value: T) -> Predicate {
        let column = column.into();
        let value_str = value.into();
        
        // This is only valid for string columns
        if let Ok(field) = self.schema.field_by_name(&column) {
            if let DataType::String = field.data_type() {
                return Predicate::Column(ColumnPredicate {
                    column,
                    op: Operator::StartsWith,
                    value: ScalarValue::String(value_str),
                    value_list: None,
                });
            }
        }
        
        // If not a string column, return predicate that always fails
        Predicate::AlwaysFalse
    }
    
    /// Create a column ENDS WITH value predicate
    pub fn ends_with<S: Into<String>, T: Into<String>>(&self, column: S, value: T) -> Predicate {
        let column = column.into();
        let value_str = value.into();
        
        // This is only valid for string columns
        if let Ok(field) = self.schema.field_by_name(&column) {
            if let DataType::String = field.data_type() {
                return Predicate::Column(ColumnPredicate {
                    column,
                    op: Operator::EndsWith,
                    value: ScalarValue::String(value_str),
                    value_list: None,
                });
            }
        }
        
        // If not a string column, return predicate that always fails
        Predicate::AlwaysFalse
    }
    
    /// Build a column predicate
    fn build_column_predicate(
        &self,
        column: String,
        op: Operator,
        value_str: String,
    ) -> Predicate {
        // Try to convert the value based on the column type
        if let Ok(field) = self.schema.field_by_name(&column) {
            let data_type = field.data_type();
            
            if let Some(value) = ScalarValue::from_string(&value_str, data_type) {
                return Predicate::Column(ColumnPredicate {
                    column,
                    op,
                    value,
                    value_list: None,
                });
            }
        }
        
        // If column not found or value can't be converted, return predicate that always fails
        Predicate::AlwaysFalse
    }
}

/// Evaluate predicate against column statistics
fn evaluate_stats(
    op: &Operator,
    value: &ScalarValue,
    stats: &ParquetStats,
    data_type: &DataType,
) -> bool {
    let min_value = stats.min_bytes();
    let max_value = stats.max_bytes();
    
    // If statistics are not available, we can't skip
    if min_value.is_empty() || max_value.is_empty() {
        return false;
    }
    
    match op {
        Operator::Eq => {
            // If value < min or value > max, we can skip
            compare_bytes(value, min_value, data_type, &Operator::Lt) ||
            compare_bytes(value, max_value, data_type, &Operator::Gt)
        },
        Operator::Ne => {
            // If min = max and value = min, we can skip
            min_value == max_value &&
            compare_bytes(value, min_value, data_type, &Operator::Eq)
        },
        Operator::Lt => {
            // If value <= min, we can skip
            compare_bytes(value, min_value, data_type, &Operator::Le)
        },
        Operator::Le => {
            // If value < min, we can skip
            compare_bytes(value, min_value, data_type, &Operator::Lt)
        },
        Operator::Gt => {
            // If value >= max, we can skip
            compare_bytes(value, max_value, data_type, &Operator::Ge)
        },
        Operator::Ge => {
            // If value > max, we can skip
            compare_bytes(value, max_value, data_type, &Operator::Gt)
        },
        Operator::IsNull => {
            // If null count is 0, we can skip
            stats.null_count() == 0
        },
        Operator::IsNotNull => {
            // If all values are null, we can skip
            stats.null_count() == stats.num_values()
        },
        // Other operators can't be pushed down to statistics level
        _ => false,
    }
}

/// Compare value bytes based on data type
fn compare_bytes(
    value: &ScalarValue,
    bytes: &[u8],
    data_type: &DataType,
    op: &Operator,
) -> bool {
    match data_type {
        DataType::Boolean => {
            if let ScalarValue::Boolean(v) = value {
                let byte_value = bytes[0];
                match op {
                    Operator::Eq => (*v as u8) == byte_value,
                    Operator::Ne => (*v as u8) != byte_value,
                    Operator::Lt => (*v as u8) < byte_value,
                    Operator::Le => (*v as u8) <= byte_value,
                    Operator::Gt => (*v as u8) > byte_value,
                    Operator::Ge => (*v as u8) >= byte_value,
                    _ => false,
                }
            } else {
                false
            }
        },
        DataType::Int8 | DataType::Int16 | DataType::Int32 | DataType::Int64 => {
            // Convert to i64 for comparison
            let byte_value = if bytes.len() == 4 {
                i32::from_le_bytes([bytes[0], bytes[1], bytes[2], bytes[3]]) as i64
            } else if bytes.len() == 8 {
                i64::from_le_bytes([bytes[0], bytes[1], bytes[2], bytes[3], 
                                   bytes[4], bytes[5], bytes[6], bytes[7]])
            } else if bytes.len() == 2 {
                i16::from_le_bytes([bytes[0], bytes[1]]) as i64
            } else if bytes.len() == 1 {
                bytes[0] as i8 as i64
            } else {
                return false;
            };
            
            let value_i64 = match value {
                ScalarValue::Int8(v) => *v as i64,
                ScalarValue::Int16(v) => *v as i64,
                ScalarValue::Int32(v) => *v as i64,
                ScalarValue::Int64(v) => *v,
                _ => return false,
            };
            
            match op {
                Operator::Eq => value_i64 == byte_value,
                Operator::Ne => value_i64 != byte_value,
                Operator::Lt => value_i64 < byte_value,
                Operator::Le => value_i64 <= byte_value,
                Operator::Gt => value_i64 > byte_value,
                Operator::Ge => value_i64 >= byte_value,
                _ => false,
            }
        },
        DataType::UInt8 | DataType::UInt16 | DataType::UInt32 | DataType::UInt64 => {
            // Convert to u64 for comparison
            let byte_value = if bytes.len() == 4 {
                u32::from_le_bytes([bytes[0], bytes[1], bytes[2], bytes[3]]) as u64
            } else if bytes.len() == 8 {
                u64::from_le_bytes([bytes[0], bytes[1], bytes[2], bytes[3], 
                                   bytes[4], bytes[5], bytes[6], bytes[7]])
            } else if bytes.len() == 2 {
                u16::from_le_bytes([bytes[0], bytes[1]]) as u64
            } else if bytes.len() == 1 {
                bytes[0] as u64
            } else {
                return false;
            };
            
            let value_u64 = match value {
                ScalarValue::UInt8(v) => *v as u64,
                ScalarValue::UInt16(v) => *v as u64,
                ScalarValue::UInt32(v) => *v as u64,
                ScalarValue::UInt64(v) => *v,
                _ => return false,
            };
            
            match op {
                Operator::Eq => value_u64 == byte_value,
                Operator::Ne => value_u64 != byte_value,
                Operator::Lt => value_u64 < byte_value,
                Operator::Le => value_u64 <= byte_value,
                Operator::Gt => value_u64 > byte_value,
                Operator::Ge => value_u64 >= byte_value,
                _ => false,
            }
        },
        DataType::Float32 => {
            if let ScalarValue::Float32(v) = value {
                if bytes.len() == 4 {
                    let byte_value = f32::from_le_bytes([bytes[0], bytes[1], bytes[2], bytes[3]]);
                    
                    match op {
                        Operator::Eq => (*v - byte_value).abs() < std::f32::EPSILON,
                        Operator::Ne => (*v - byte_value).abs() >= std::f32::EPSILON,
                        Operator::Lt => *v < byte_value,
                        Operator::Le => *v <= byte_value,
                        Operator::Gt => *v > byte_value,
                        Operator::Ge => *v >= byte_value,
                        _ => false,
                    }
                } else {
                    false
                }
            } else {
                false
            }
        },
        DataType::Float64 => {
            if let ScalarValue::Float64(v) = value {
                if bytes.len() == 8 {
                    let byte_value = f64::from_le_bytes([bytes[0], bytes[1], bytes[2], bytes[3], 
                                                        bytes[4], bytes[5], bytes[6], bytes[7]]);
                    
                    match op {
                        Operator::Eq => (*v - byte_value).abs() < std::f64::EPSILON,
                        Operator::Ne => (*v - byte_value).abs() >= std::f64::EPSILON,
                        Operator::Lt => *v < byte_value,
                        Operator::Le => *v <= byte_value,
                        Operator::Gt => *v > byte_value,
                        Operator::Ge => *v >= byte_value,
                        _ => false,
                    }
                } else {
                    false
                }
            } else {
                false
            }
        },
        DataType::String => {
            if let ScalarValue::String(v) = value {
                let byte_str = std::str::from_utf8(bytes).unwrap_or("");
                
                match op {
                    Operator::Eq => v == byte_str,
                    Operator::Ne => v != byte_str,
                    Operator::Lt => v < byte_str,
                    Operator::Le => v <= byte_str,
                    Operator::Gt => v > byte_str,
                    Operator::Ge => v >= byte_str,
                    _ => false,
                }
            } else {
                false
            }
        },
        DataType::Date32 => {
            if let ScalarValue::Date32(v) = value {
                if bytes.len() == 4 {
                    let byte_value = i32::from_le_bytes([bytes[0], bytes[1], bytes[2], bytes[3]]);
                    
                    match op {
                        Operator::Eq => *v == byte_value,
                        Operator::Ne => *v != byte_value,
                        Operator::Lt => *v < byte_value,
                        Operator::Le => *v <= byte_value,
                        Operator::Gt => *v > byte_value,
                        Operator::Ge => *v >= byte_value,
                        _ => false,
                    }
                } else {
                    false
                }
            } else {
                false
            }
        },
        DataType::Date64 => {
            if let ScalarValue::Date64(v) = value {
                if bytes.len() == 8 {
                    let byte_value = i64::from_le_bytes([bytes[0], bytes[1], bytes[2], bytes[3], 
                                                        bytes[4], bytes[5], bytes[6], bytes[7]]);
                    
                    match op {
                        Operator::Eq => *v == byte_value,
                        Operator::Ne => *v != byte_value,
                        Operator::Lt => *v < byte_value,
                        Operator::Le => *v <= byte_value,
                        Operator::Gt => *v > byte_value,
                        Operator::Ge => *v >= byte_value,
                        _ => false,
                    }
                } else {
                    false
                }
            } else {
                false
            }
        },
        DataType::Timestamp(_, _) => {
            if let ScalarValue::Timestamp(v) = value {
                if bytes.len() == 8 {
                    let byte_value = i64::from_le_bytes([bytes[0], bytes[1], bytes[2], bytes[3], 
                                                        bytes[4], bytes[5], bytes[6], bytes[7]]);
                    
                    match op {
                        Operator::Eq => *v == byte_value,
                        Operator::Ne => *v != byte_value,
                        Operator::Lt => *v < byte_value,
                        Operator::Le => *v <= byte_value,
                        Operator::Gt => *v > byte_value,
                        Operator::Ge => *v >= byte_value,
                        _ => false,
                    }
                } else {
                    false
                }
            } else {
                false
            }
        },
        // For other types, we can't do statistics-based filtering
        _ => false,
    }
}