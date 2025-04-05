//! Factory pattern for creating data sources

use std::path::{Path, PathBuf};
use std::sync::Arc;

use ml_data_core::{RecordBatchSource, SourceFactory};
use ml_data_core::error::Result as CoreResult;

use crate::error::Result;

/// A factory for creating data sources
pub trait DataSourceFactory: Send + Sync {
    /// The type of source this factory creates
    type Source;
    
    /// Create a new source instance
    fn create_source(&self) -> Result<Self::Source>;
    
    /// Get the number of sources that can be created (if known)
    fn source_count_hint(&self) -> Option<usize> {
        None
    }
}

/// Configuration for file data sources
pub struct FileSourceConfig {
    /// Base directory for file paths
    pub base_dir: Option<PathBuf>,
    
    /// Whether to memory-map files
    pub memory_map: bool,
    
    /// Buffer size for reading
    pub buffer_size: Option<usize>,
    
    /// File chunk size for parallel processing
    pub chunk_size: Option<usize>,
}

impl Default for FileSourceConfig {
    fn default() -> Self {
        Self {
            base_dir: None,
            memory_map: false,
            buffer_size: Some(64 * 1024), // 64KB default
            chunk_size: None,
        }
    }
}

/// Factory for creating file data sources
pub struct FileSourceFactory<T, F> {
    /// File paths to read
    paths: Vec<PathBuf>,
    
    /// Configuration
    config: FileSourceConfig,
    
    /// Source creation function
    source_creator: F,
    
    /// Phantom type for the source type
    _phantom: std::marker::PhantomData<T>,
}

impl<T, F> FileSourceFactory<T, F>
where
    F: Fn(&Path, &FileSourceConfig) -> Result<T> + Send + Sync,
    T: RecordBatchSource,
{
    /// Create a new file source factory
    pub fn new<P: AsRef<Path>>(
        paths: Vec<P>,
        config: FileSourceConfig,
        source_creator: F,
    ) -> Self {
        let paths = paths.into_iter()
            .map(|p| p.as_ref().to_path_buf())
            .collect();
        
        Self {
            paths,
            config,
            source_creator,
            _phantom: std::marker::PhantomData,
        }
    }
    
    /// Get the number of files
    pub fn file_count(&self) -> usize {
        self.paths.len()
    }
    
    /// Get the file paths
    pub fn paths(&self) -> &[PathBuf] {
        &self.paths
    }
    
    /// Get the configuration
    pub fn config(&self) -> &FileSourceConfig {
        &self.config
    }
}

impl<T, F> SourceFactory for FileSourceFactory<T, F>
where
    F: Fn(&Path, &FileSourceConfig) -> Result<T> + Send + Sync,
    T: RecordBatchSource,
{
    type Source = T;
    
    fn create(&self) -> CoreResult<Self::Source> {
        // Select a file path - for now just use the first one
        // In a more advanced implementation, we would distribute paths
        // among parallel executions
        if self.paths.is_empty() {
            return Err(ml_data_core::error::Error::InvalidOperation(
                "No file paths available".into()
            ));
        }
        
        let path = &self.paths[0];
        
        // Create the source
        (self.source_creator)(path, &self.config).map_err(|e| {
            ml_data_core::error::Error::Io(std::io::Error::new(
                std::io::ErrorKind::Other,
                format!("Failed to create source: {}", e),
            ))
        })
    }
    
    fn source_count_hint(&self) -> Option<usize> {
        Some(self.paths.len())
    }
}

/// Factory for creating a single data source
pub struct SingleSourceFactory<T> {
    /// The source creator function
    source_creator: Box<dyn Fn() -> Result<T> + Send + Sync>,
}

impl<T> SingleSourceFactory<T> {
    /// Create a new single source factory
    pub fn new<F>(source_creator: F) -> Self
    where
        F: Fn() -> Result<T> + Send + Sync + 'static,
    {
        Self {
            source_creator: Box::new(source_creator),
        }
    }
}

impl<T: RecordBatchSource> SourceFactory for SingleSourceFactory<T> {
    type Source = T;
    
    fn create(&self) -> CoreResult<Self::Source> {
        (self.source_creator)().map_err(|e| {
            ml_data_core::error::Error::Io(std::io::Error::new(
                std::io::ErrorKind::Other,
                format!("Failed to create source: {}", e),
            ))
        })
    }
    
    fn source_count_hint(&self) -> Option<usize> {
        Some(1)
    }
}