# ML Data Pipeline

A memory-efficient machine learning data pipeline framework built with Rust.

## Features

- **Memory Efficiency**: Zero-copy operations, buffer pooling, and memory mapping
- **Modular Architecture**: Extensible plugin system for custom data sources, transforms, and sinks
- **Type Safety**: Robust type system for machine learning data
- **Arrow Integration**: Seamless interoperability with Apache Arrow
- **Python Bindings**: Use the pipeline in Python with native performance
- **Parallelism**: Multi-threaded processing with adaptive resource management

## Project Structure

The project is organized as a Rust workspace with the following crates:

- **ml-data-core**: Core traits, data structures, and abstractions
- **ml-data-readers**: Implementations of various data sources
- **ml-data-transforms**: Data transformation implementations
- **ml-data-shuffle**: Disk-backed shuffle implementations
- **ml-data-parallel**: Parallelism and concurrency features
- **ml-data-python**: Python bindings using PyO3
- **ml-data-arrow**: Arrow integration components
- **ml-data-bench**: Comprehensive benchmarking suite

## Getting Started

### Prerequisites

- Rust toolchain (1.65+)
- Cargo

For Python bindings:
- Python 3.8+
- maturin

### Installation

```bash
# Clone the repository
git clone https://github.com/your-username/ml-data.git
cd ml-data

# Build the project
cargo build --release

# Run tests
cargo test

# Build Python bindings
cd ml-data-python
maturin develop --release
```

### Usage Example

```rust
use ml_data_core::{
    source::Source,
    transform::Transform,
    sink::Sink,
    memory::MemoryBudget,
    schedule::Pipeline,
    schedule::PipelineConfig,
};
use ml_data_readers::CsvSource;
use ml_data_transforms::categorical::OneHotEncoder;
use std::sync::Arc;

// Create a source
let source = CsvSource::new("data.csv", b',', true, 1024).unwrap();

// Create a transform
let transform = OneHotEncoder::new("category_column", 100);

// Create a sink
let sink = CollectingSink::new();

// Set memory budget
let budget = Arc::new(MemoryBudget::new(1_000_000_000)); // 1GB

// Configure pipeline
let config = PipelineConfig {
    batch_size: 10000,
    worker_threads: 4,
    buffer_capacity: 100,
    backpressure_threshold: 0.8,
    memory_check_frequency: Duration::from_millis(100),
};

// Create and run pipeline
let pipeline = Pipeline::new(source, transform, sink, budget, config);
let stats = pipeline.run().expect("Pipeline execution failed");

println!("Processed {} items in {:?}", stats.items_processed, stats.execution_time);
```

## Documentation

For detailed documentation, see:

- [Architecture Overview](docs/ARCHITECTURE.md)
- [API Reference](docs/API.md)
- [Performance Guide](docs/PERFORMANCE.md)
- [Examples](examples/)

## Contributing

Contributions are welcome! Please see [CONTRIBUTING.md](CONTRIBUTING.md) for details.

## License

This project is licensed under either of:

- Apache License, Version 2.0, ([LICENSE-APACHE](LICENSE-APACHE) or http://www.apache.org/licenses/LICENSE-2.0)
- MIT license ([LICENSE-MIT](LICENSE-MIT) or http://opensource.org/licenses/MIT)

at your option.