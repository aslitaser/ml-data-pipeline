//! Benchmark runner for ML data pipeline components

use ml_data_bench::{BenchConfig, bench_record_batch, compare_with_arrow};

fn main() {
    // Print header
    println!("=== ML Data Pipeline Benchmarks ===");
    
    // Run basic benchmarks
    let config = BenchConfig {
        iterations: 5,
        batch_size: 10,
        num_columns: 10,
        rows_per_batch: 10000,
        warmup_iterations: 2,
        track_memory: true,
        num_threads: 4,
    };
    
    // Run record batch benchmark
    let result = bench_record_batch(&config);
    println!("\nBenchmark: {}", result.name);
    println!("  Total time:   {:?}", result.total_time);
    println!("  Average time: {:?}", result.avg_time);
    println!("  Min time:     {:?}", result.min_time);
    println!("  Max time:     {:?}", result.max_time);
    println!("  Throughput:   {:.2} rows/sec", result.throughput);
    
    if let Some(memory) = result.memory_usage {
        println!("  Memory usage: {} bytes", memory);
    }
    
    // Compare with Arrow
    println!("\n=== Comparison with Arrow ===");
    let comparison = compare_with_arrow(&config);
    
    for result in comparison {
        println!("\nImplementation: {}", result.name);
        println!("  Total time:   {:?}", result.total_time);
        println!("  Average time: {:?}", result.avg_time);
        println!("  Throughput:   {:.2} rows/sec", result.throughput);
        
        if let Some(memory) = result.memory_usage {
            println!("  Memory usage: {} bytes", memory);
        }
    }
    
    // Run more complex benchmarks with different configurations
    println!("\n=== Memory Efficiency Tests ===");
    
    // Test with increasing number of columns
    for num_columns in [10, 20, 50, 100] {
        let config = BenchConfig {
            iterations: 3,
            batch_size: 5,
            num_columns,
            rows_per_batch: 10000,
            warmup_iterations: 1,
            track_memory: true,
            num_threads: 4,
        };
        
        let result = bench_record_batch(&config);
        println!("\nColumns: {}", num_columns);
        println!("  Average time: {:?}", result.avg_time);
        println!("  Throughput:   {:.2} rows/sec", result.throughput);
        
        if let Some(memory) = result.memory_usage {
            println!("  Memory usage: {} bytes", memory);
        }
    }
    
    // Test with increasing number of rows
    for rows_per_batch in [1000, 10000, 100000, 1000000] {
        let config = BenchConfig {
            iterations: 3,
            batch_size: 5,
            num_columns: 10,
            rows_per_batch,
            warmup_iterations: 1,
            track_memory: true,
            num_threads: 4,
        };
        
        let result = bench_record_batch(&config);
        println!("\nRows: {}", rows_per_batch);
        println!("  Average time: {:?}", result.avg_time);
        println!("  Throughput:   {:.2} rows/sec", result.throughput);
        
        if let Some(memory) = result.memory_usage {
            println!("  Memory usage: {} bytes", memory);
            println!("  Bytes per row: {:.2} bytes/row", 
                memory as f64 / (rows_per_batch as f64 * config.batch_size as f64));
        }
    }
}