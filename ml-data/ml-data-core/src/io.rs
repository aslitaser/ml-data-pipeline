//! I/O utilities for efficient data access

use std::fs::File;
use std::io::{self, BufReader, BufWriter, Read, Seek, SeekFrom, Write};
use std::path::{Path, PathBuf};
use std::sync::Arc;

use memmap2::{Mmap, MmapMut, MmapOptions};

use crate::error::{Error, Result};

/// Memory-mapped file for zero-copy I/O
pub struct MemoryMappedFile {
    /// The memory map
    mmap: Mmap,
    
    /// The path to the file
    path: PathBuf,
    
    /// The size of the file in bytes
    size: usize,
}

impl MemoryMappedFile {
    /// Open a file for memory-mapped reading
    pub fn open<P: AsRef<Path>>(path: P) -> Result<Self> {
        let path = path.as_ref().to_path_buf();
        let file = File::open(&path).map_err(Error::Io)?;
        let size = file.metadata().map_err(Error::Io)?.len() as usize;
        
        let mmap = unsafe { MmapOptions::new().map(&file).map_err(Error::Io)? };
        
        Ok(Self {
            mmap,
            path,
            size,
        })
    }
    
    /// Get a slice of the memory-mapped file
    pub fn as_slice(&self) -> &[u8] {
        &self.mmap
    }
    
    /// Get a subslice of the memory-mapped file
    pub fn slice(&self, offset: usize, length: usize) -> Result<&[u8]> {
        if offset + length > self.size {
            return Err(Error::IndexOutOfBounds);
        }
        
        Ok(&self.mmap[offset..offset + length])
    }
    
    /// Get the path to the file
    pub fn path(&self) -> &Path {
        &self.path
    }
    
    /// Get the size of the file
    pub fn size(&self) -> usize {
        self.size
    }
}

/// Mutable memory-mapped file for zero-copy I/O
pub struct MemoryMappedFileMut {
    /// The mutable memory map
    mmap: MmapMut,
    
    /// The path to the file
    path: PathBuf,
    
    /// The size of the file in bytes
    size: usize,
}

impl MemoryMappedFileMut {
    /// Open a file for memory-mapped reading and writing
    pub fn open<P: AsRef<Path>>(path: P) -> Result<Self> {
        let path = path.as_ref().to_path_buf();
        let file = OpenOptions::new().read(true).write(true).open(&path).map_err(Error::Io)?;
        let size = file.metadata().map_err(Error::Io)?.len() as usize;
        
        let mmap = unsafe { MmapOptions::new().map_mut(&file).map_err(Error::Io)? };
        
        Ok(Self {
            mmap,
            path,
            size,
        })
    }
    
    /// Create a new memory-mapped file with the given size
    pub fn create<P: AsRef<Path>>(path: P, size: usize) -> Result<Self> {
        let path = path.as_ref().to_path_buf();
        let file = OpenOptions::new()
            .read(true)
            .write(true)
            .create(true)
            .truncate(true)
            .open(&path)
            .map_err(Error::Io)?;
        
        // Set the file size
        file.set_len(size as u64).map_err(Error::Io)?;
        
        let mmap = unsafe { MmapOptions::new().map_mut(&file).map_err(Error::Io)? };
        
        Ok(Self {
            mmap,
            path,
            size,
        })
    }
    
    /// Get a mutable slice of the memory-mapped file
    pub fn as_mut_slice(&mut self) -> &mut [u8] {
        &mut self.mmap
    }
    
    /// Get a mutable subslice of the memory-mapped file
    pub fn slice_mut(&mut self, offset: usize, length: usize) -> Result<&mut [u8]> {
        if offset + length > self.size {
            return Err(Error::IndexOutOfBounds);
        }
        
        Ok(&mut self.mmap[offset..offset + length])
    }
    
    /// Get the path to the file
    pub fn path(&self) -> &Path {
        &self.path
    }
    
    /// Get the size of the file
    pub fn size(&self) -> usize {
        self.size
    }
    
    /// Flush changes to disk
    pub fn flush(&mut self) -> Result<()> {
        self.mmap.flush().map_err(Error::Io)
    }
    
    /// Resize the file
    pub fn resize(&mut self, new_size: usize) -> Result<()> {
        // Need to close and reopen the file
        drop(std::mem::take(&mut self.mmap));
        
        let file = OpenOptions::new()
            .read(true)
            .write(true)
            .open(&self.path)
            .map_err(Error::Io)?;
        
        // Set the new file size
        file.set_len(new_size as u64).map_err(Error::Io)?;
        
        // Remap the file
        self.mmap = unsafe { MmapOptions::new().map_mut(&file).map_err(Error::Io)? };
        self.size = new_size;
        
        Ok(())
    }
}

/// Enhanced file opening options
pub struct OpenOptions {
    /// Inner standard library open options
    inner: std::fs::OpenOptions,
    
    /// Buffer size for buffered I/O
    buffer_size: Option<usize>,
    
    /// Whether to use memory mapping
    memory_map: bool,
    
    /// Whether to memory map with write access
    memory_map_mut: bool,
    
    /// Whether to use direct I/O (unbuffered)
    direct_io: bool,
}

impl OpenOptions {
    /// Create a new set of options
    pub fn new() -> Self {
        Self {
            inner: std::fs::OpenOptions::new(),
            buffer_size: None,
            memory_map: false,
            memory_map_mut: false,
            direct_io: false,
        }
    }
    
    /// Set read access
    pub fn read(mut self, read: bool) -> Self {
        self.inner.read(read);
        self
    }
    
    /// Set write access
    pub fn write(mut self, write: bool) -> Self {
        self.inner.write(write);
        self
    }
    
    /// Set append mode
    pub fn append(mut self, append: bool) -> Self {
        self.inner.append(append);
        self
    }
    
    /// Set truncate mode
    pub fn truncate(mut self, truncate: bool) -> Self {
        self.inner.truncate(truncate);
        self
    }
    
    /// Set create mode
    pub fn create(mut self, create: bool) -> Self {
        self.inner.create(create);
        self
    }
    
    /// Set create_new mode
    pub fn create_new(mut self, create_new: bool) -> Self {
        self.inner.create_new(create_new);
        self
    }
    
    /// Set buffer size for buffered I/O
    pub fn buffer_size(mut self, buffer_size: usize) -> Self {
        self.buffer_size = Some(buffer_size);
        self
    }
    
    /// Set memory mapping mode
    pub fn memory_map(mut self, memory_map: bool) -> Self {
        self.memory_map = memory_map;
        self
    }
    
    /// Set mutable memory mapping mode
    pub fn memory_map_mut(mut self, memory_map_mut: bool) -> Self {
        self.memory_map_mut = memory_map_mut;
        self
    }
    
    /// Set direct I/O mode (unbuffered)
    pub fn direct_io(mut self, direct_io: bool) -> Self {
        self.direct_io = direct_io;
        self
    }
    
    /// Open a file with these options
    pub fn open<P: AsRef<Path>>(self, path: P) -> Result<Box<dyn FileIO>> {
        let path = path.as_ref();
        
        if self.memory_map {
            if self.memory_map_mut {
                let file = MemoryMappedFileMut::open(path)?;
                Ok(Box::new(file))
            } else {
                let file = MemoryMappedFile::open(path)?;
                Ok(Box::new(file))
            }
        } else if self.direct_io {
            #[cfg(target_os = "linux")]
            {
                let file = self.inner.custom_flags(libc::O_DIRECT).open(path).map_err(Error::Io)?;
                Ok(Box::new(DirectFile { file }))
            }
            #[cfg(not(target_os = "linux"))]
            {
                return Err(Error::NotImplemented("Direct I/O is only supported on Linux".into()));
            }
        } else {
            let file = self.inner.open(path).map_err(Error::Io)?;
            
            if let Some(buffer_size) = self.buffer_size {
                if self.inner.read {
                    if self.inner.write || self.inner.append {
                        Ok(Box::new(BufferedRwFile {
                            reader: BufReader::with_capacity(buffer_size, file.try_clone().map_err(Error::Io)?),
                            writer: BufWriter::with_capacity(buffer_size, file),
                        }))
                    } else {
                        Ok(Box::new(BufReader::with_capacity(buffer_size, file)))
                    }
                } else if self.inner.write || self.inner.append {
                    Ok(Box::new(BufWriter::with_capacity(buffer_size, file)))
                } else {
                    Ok(Box::new(file))
                }
            } else {
                Ok(Box::new(file))
            }
        }
    }
}

impl Default for OpenOptions {
    fn default() -> Self {
        Self::new()
    }
}

/// Trait for file I/O operations
pub trait FileIO {}

impl FileIO for File {}
impl FileIO for BufReader<File> {}
impl FileIO for BufWriter<File> {}
impl FileIO for MemoryMappedFile {}
impl FileIO for MemoryMappedFileMut {}

/// Buffered file for both reading and writing
pub struct BufferedRwFile {
    /// Reader part
    reader: BufReader<File>,
    
    /// Writer part
    writer: BufWriter<File>,
}

impl Read for BufferedRwFile {
    fn read(&mut self, buf: &mut [u8]) -> io::Result<usize> {
        // Flush writer to ensure consistency
        self.writer.flush()?;
        self.reader.read(buf)
    }
}

impl Write for BufferedRwFile {
    fn write(&mut self, buf: &[u8]) -> io::Result<usize> {
        self.writer.write(buf)
    }
    
    fn flush(&mut self) -> io::Result<()> {
        self.writer.flush()
    }
}

impl Seek for BufferedRwFile {
    fn seek(&mut self, pos: SeekFrom) -> io::Result<u64> {
        // Flush writer to ensure consistency
        self.writer.flush()?;
        
        // Seek both reader and writer
        let pos = self.reader.seek(pos)?;
        self.writer.seek(SeekFrom::Start(pos))?;
        
        Ok(pos)
    }
}

impl FileIO for BufferedRwFile {}

/// Direct I/O file (unbuffered)
#[cfg(target_os = "linux")]
pub struct DirectFile {
    /// Inner file
    file: File,
}

#[cfg(target_os = "linux")]
impl Read for DirectFile {
    fn read(&mut self, buf: &mut [u8]) -> io::Result<usize> {
        // For direct I/O, the buffer must be aligned to a multiple of the block size
        if (buf.as_ptr() as usize) % 512 != 0 || buf.len() % 512 != 0 {
            return Err(io::Error::new(
                io::ErrorKind::InvalidInput,
                "Buffer must be aligned to a 512-byte boundary and have a length that's a multiple of 512 for direct I/O",
            ));
        }
        
        self.file.read(buf)
    }
}

#[cfg(target_os = "linux")]
impl Write for DirectFile {
    fn write(&mut self, buf: &[u8]) -> io::Result<usize> {
        // For direct I/O, the buffer must be aligned to a multiple of the block size
        if (buf.as_ptr() as usize) % 512 != 0 || buf.len() % 512 != 0 {
            return Err(io::Error::new(
                io::ErrorKind::InvalidInput,
                "Buffer must be aligned to a 512-byte boundary and have a length that's a multiple of 512 for direct I/O",
            ));
        }
        
        self.file.write(buf)
    }
    
    fn flush(&mut self) -> io::Result<()> {
        self.file.flush()
    }
}

#[cfg(target_os = "linux")]
impl Seek for DirectFile {
    fn seek(&mut self, pos: SeekFrom) -> io::Result<u64> {
        self.file.seek(pos)
    }
}

#[cfg(target_os = "linux")]
impl FileIO for DirectFile {}

/// Asynchronous I/O utilities
pub mod async_io {
    use std::path::Path;
    use std::sync::Arc;
    
    use crate::error::Result;
    
    /// Trait for asynchronous file I/O operations
    pub trait AsyncFileIO: Send + Sync {
        /// Read data asynchronously
        async fn read(&self, offset: u64, len: usize) -> Result<Vec<u8>>;
        
        /// Write data asynchronously
        async fn write(&self, offset: u64, data: &[u8]) -> Result<()>;
        
        /// Flush changes to disk
        async fn flush(&self) -> Result<()>;
        
        /// Get the size of the file
        async fn size(&self) -> Result<u64>;
    }
    
    /// Asynchronous memory-mapped file
    pub struct AsyncMemoryMappedFile {
        /// Inner memory mapped file
        inner: Arc<tokio::sync::Mutex<super::MemoryMappedFile>>,
    }
    
    impl AsyncMemoryMappedFile {
        /// Open a file for asynchronous memory-mapped I/O
        pub async fn open<P: AsRef<Path>>(path: P) -> Result<Self> {
            let file = super::MemoryMappedFile::open(path)?;
            
            Ok(Self {
                inner: Arc::new(tokio::sync::Mutex::new(file)),
            })
        }
    }
    
    impl AsyncFileIO for AsyncMemoryMappedFile {
        async fn read(&self, offset: u64, len: usize) -> Result<Vec<u8>> {
            let guard = self.inner.lock().await;
            let data = guard.slice(offset as usize, len)?;
            Ok(data.to_vec())
        }
        
        async fn write(&self, _offset: u64, _data: &[u8]) -> Result<()> {
            Err(crate::error::Error::InvalidOperation(
                "Cannot write to a read-only memory-mapped file".into()
            ))
        }
        
        async fn flush(&self) -> Result<()> {
            Ok(())
        }
        
        async fn size(&self) -> Result<u64> {
            let guard = self.inner.lock().await;
            Ok(guard.size() as u64)
        }
    }
    
    /// Asynchronous mutable memory-mapped file
    pub struct AsyncMemoryMappedFileMut {
        /// Inner mutable memory mapped file
        inner: Arc<tokio::sync::Mutex<super::MemoryMappedFileMut>>,
    }
    
    impl AsyncMemoryMappedFileMut {
        /// Open a file for asynchronous memory-mapped I/O with write access
        pub async fn open<P: AsRef<Path>>(path: P) -> Result<Self> {
            let file = super::MemoryMappedFileMut::open(path)?;
            
            Ok(Self {
                inner: Arc::new(tokio::sync::Mutex::new(file)),
            })
        }
        
        /// Create a new file for asynchronous memory-mapped I/O with write access
        pub async fn create<P: AsRef<Path>>(path: P, size: usize) -> Result<Self> {
            let file = super::MemoryMappedFileMut::create(path, size)?;
            
            Ok(Self {
                inner: Arc::new(tokio::sync::Mutex::new(file)),
            })
        }
    }
    
    impl AsyncFileIO for AsyncMemoryMappedFileMut {
        async fn read(&self, offset: u64, len: usize) -> Result<Vec<u8>> {
            let guard = self.inner.lock().await;
            let data = guard.slice(offset as usize, len)?;
            Ok(data.to_vec())
        }
        
        async fn write(&self, offset: u64, data: &[u8]) -> Result<()> {
            let mut guard = self.inner.lock().await;
            let slice = guard.slice_mut(offset as usize, data.len())?;
            slice.copy_from_slice(data);
            Ok(())
        }
        
        async fn flush(&self) -> Result<()> {
            let mut guard = self.inner.lock().await;
            guard.flush()
        }
        
        async fn size(&self) -> Result<u64> {
            let guard = self.inner.lock().await;
            Ok(guard.size() as u64)
        }
    }
}