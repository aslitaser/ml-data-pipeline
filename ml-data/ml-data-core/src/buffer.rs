//! Memory buffer implementation with reference counting and zero-copy operations

use std::alloc::{alloc_zeroed, dealloc, Layout};
use std::fmt;
use std::marker::PhantomData;
use std::mem::{align_of, size_of};
use std::ops::{Deref, Range};
use std::ptr::NonNull;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;

use bytemuck::{Pod, Zeroable};
use memmap2::{MmapMut, MmapOptions};

use crate::error::{Error, Result};
use crate::memory::MemoryPool;

/// Constant for typical OS page size (4KB)
const PAGE_SIZE: usize = 4096;

/// Alignment for optimal SIMD operations (typically 32 or 64 bytes for AVX/AVX-512)
const SIMD_ALIGNMENT: usize = 64;

/// Buffer holding raw bytes with reference counting and memory management
#[derive(Debug)]
pub struct Buffer {
    /// Raw pointer to the allocated memory
    ptr: NonNull<u8>,
    
    /// Size of the buffer in bytes
    size: usize,
    
    /// Capacity of the buffer in bytes
    capacity: usize,
    
    /// Reference count for this buffer
    ref_count: Arc<AtomicUsize>,
    
    /// Memory pool this buffer was allocated from, if any
    memory_pool: Option<Arc<dyn MemoryPool>>,
    
    /// Layout used for allocation/deallocation
    layout: Layout,
    
    /// Whether this buffer owns its memory (false for slices/views)
    owns_memory: bool,
}

impl Buffer {
    /// Create a new empty buffer with the given capacity
    pub fn new(capacity: usize) -> Result<Self> {
        let layout = Layout::from_size_align(capacity.max(1), SIMD_ALIGNMENT)
            .map_err(|_| Error::LayoutError("Invalid buffer layout".into()))?;
        
        // Safety: layout is guaranteed to be properly aligned and capacity > 0
        let ptr = unsafe { alloc_zeroed(layout) };
        let ptr = NonNull::new(ptr).ok_or(Error::MemoryAllocationFailed)?;
        
        Ok(Self {
            ptr,
            size: 0,
            capacity,
            ref_count: Arc::new(AtomicUsize::new(1)),
            memory_pool: None,
            layout,
            owns_memory: true,
        })
    }
    
    /// Create a new buffer of specified size initialized with zeros
    pub fn new_zeroed(size: usize) -> Result<Self> {
        let mut buffer = Self::new(size)?;
        buffer.size = size;
        Ok(buffer)
    }
    
    /// Create a buffer from a memory pool
    pub fn new_from_pool(capacity: usize, pool: Arc<dyn MemoryPool>) -> Result<Self> {
        let layout = Layout::from_size_align(capacity.max(1), SIMD_ALIGNMENT)
            .map_err(|_| Error::LayoutError("Invalid buffer layout".into()))?;
        
        let ptr = pool.allocate(layout.size(), layout.align())?;
        
        Ok(Self {
            ptr,
            size: 0,
            capacity,
            ref_count: Arc::new(AtomicUsize::new(1)),
            memory_pool: Some(pool),
            layout,
            owns_memory: true,
        })
    }
    
    /// Create a buffer from raw parts
    /// 
    /// # Safety
    /// 
    /// The caller must ensure that:
    /// - `ptr` points to a valid memory region of at least `capacity` bytes
    /// - The memory is aligned to SIMD_ALIGNMENT
    /// - The memory remains valid for the lifetime of the buffer
    pub unsafe fn from_raw_parts(
        ptr: *mut u8,
        size: usize,
        capacity: usize,
        owns_memory: bool,
    ) -> Self {
        let ptr = NonNull::new(ptr).expect("Null pointer in from_raw_parts");
        
        let layout = Layout::from_size_align_unchecked(capacity, SIMD_ALIGNMENT);
        
        Self {
            ptr,
            size,
            capacity,
            ref_count: Arc::new(AtomicUsize::new(1)),
            memory_pool: None,
            layout,
            owns_memory,
        }
    }
    
    /// Create a memory-mapped buffer from a file
    pub fn new_mmap(path: &str, size: usize) -> Result<Self> {
        use std::fs::OpenOptions;
        
        let file = OpenOptions::new()
            .read(true)
            .write(true)
            .create(true)
            .open(path)
            .map_err(Error::Io)?;
            
        // Ensure the file is the right size
        file.set_len(size as u64).map_err(Error::Io)?;
        
        let mmap = unsafe {
            MmapOptions::new()
                .len(size)
                .map_mut(&file)
                .map_err(Error::Io)?
        };
        
        let ptr = mmap.as_mut_ptr();
        
        // We need to leak the mmap object to keep the memory valid
        // It will be cleaned up when the buffer is dropped if it owns the memory
        std::mem::forget(mmap);
        
        unsafe { 
            Ok(Self::from_raw_parts(ptr, size, size, true))
        }
    }
    
    /// Create a slice (view) of this buffer
    pub fn slice(&self, offset: usize, length: usize) -> Result<Self> {
        if offset + length > self.size {
            return Err(Error::IndexOutOfBounds);
        }
        
        let ptr = unsafe { self.ptr.as_ptr().add(offset) };
        
        // Increment reference count
        self.ref_count.fetch_add(1, Ordering::SeqCst);
        
        unsafe {
            Ok(Self {
                ptr: NonNull::new(ptr).unwrap(),
                size: length,
                capacity: length,
                ref_count: self.ref_count.clone(),
                memory_pool: self.memory_pool.clone(),
                layout: self.layout,
                owns_memory: false,
            })
        }
    }
    
    /// Get a slice of this buffer as a typed array
    /// 
    /// # Safety
    /// 
    /// The caller must ensure that:
    /// - T is the correct type for the data in the buffer
    /// - The buffer contains a whole number of T elements
    pub unsafe fn as_typed_slice<T: Pod>(&self) -> &[T] {
        let elem_size = size_of::<T>();
        let count = self.size / elem_size;
        
        std::slice::from_raw_parts(self.ptr.as_ptr() as *const T, count)
    }
    
    /// Get a mutable slice of this buffer as a typed array
    /// 
    /// # Safety
    /// 
    /// The caller must ensure that:
    /// - T is the correct type for the data in the buffer
    /// - The buffer contains a whole number of T elements
    /// - The buffer is not shared through other slices
    pub unsafe fn as_typed_slice_mut<T: Pod>(&mut self) -> &mut [T] {
        debug_assert_eq!(
            self.ref_count.load(Ordering::SeqCst), 
            1, 
            "Attempted to get mutable access to a shared buffer"
        );
        
        let elem_size = size_of::<T>();
        let count = self.size / elem_size;
        
        std::slice::from_raw_parts_mut(self.ptr.as_ptr() as *mut T, count)
    }
    
    /// Get a pointer to the buffer data
    pub fn as_ptr(&self) -> *const u8 {
        self.ptr.as_ptr()
    }
    
    /// Get a mutable pointer to the buffer data
    pub fn as_mut_ptr(&mut self) -> *mut u8 {
        debug_assert_eq!(
            self.ref_count.load(Ordering::SeqCst), 
            1, 
            "Attempted to get mutable access to a shared buffer"
        );
        
        self.ptr.as_ptr()
    }
    
    /// Get the size of the buffer
    pub fn size(&self) -> usize {
        self.size
    }
    
    /// Get the capacity of the buffer
    pub fn capacity(&self) -> usize {
        self.capacity
    }
    
    /// Check if the buffer is empty
    pub fn is_empty(&self) -> bool {
        self.size == 0
    }
    
    /// Resize the buffer
    pub fn resize(&mut self, new_size: usize) -> Result<()> {
        debug_assert_eq!(
            self.ref_count.load(Ordering::SeqCst), 
            1, 
            "Attempted to resize a shared buffer"
        );
        
        if new_size <= self.capacity {
            self.size = new_size;
            return Ok(());
        }
        
        // Need to reallocate
        if !self.owns_memory {
            return Err(Error::InvalidOperation("Cannot resize a buffer view".into()));
        }
        
        let new_capacity = new_size.max(self.capacity * 2);
        let new_layout = Layout::from_size_align(new_capacity, SIMD_ALIGNMENT)
            .map_err(|_| Error::LayoutError("Invalid buffer layout for resize".into()))?;
        
        let new_ptr = if let Some(pool) = &self.memory_pool {
            let new_ptr = pool.reallocate(self.ptr, self.layout.size(), new_layout.size(), new_layout.align())?;
            new_ptr
        } else {
            // Standard reallocation
            let new_ptr = unsafe {
                let new_ptr = alloc_zeroed(new_layout);
                if !new_ptr.is_null() {
                    // Copy existing data
                    std::ptr::copy_nonoverlapping(
                        self.ptr.as_ptr(),
                        new_ptr,
                        self.size
                    );
                }
                
                // Free old memory
                dealloc(self.ptr.as_ptr(), self.layout);
                
                new_ptr
            };
            
            NonNull::new(new_ptr).ok_or(Error::MemoryAllocationFailed)?
        };
        
        self.ptr = new_ptr;
        self.size = new_size;
        self.capacity = new_capacity;
        self.layout = new_layout;
        
        Ok(())
    }
    
    /// Check if this buffer is a view of another buffer
    pub fn is_view(&self) -> bool {
        !self.owns_memory
    }
    
    /// Get the current reference count
    pub fn ref_count(&self) -> usize {
        self.ref_count.load(Ordering::SeqCst)
    }
    
    /// Create a new buffer by copying the data from a slice
    pub fn from_slice<T: Pod>(data: &[T]) -> Result<Self> {
        let size = data.len() * size_of::<T>();
        let mut buffer = Self::new(size)?;
        
        unsafe {
            std::ptr::copy_nonoverlapping(
                data.as_ptr() as *const u8,
                buffer.ptr.as_ptr(),
                size
            );
        }
        
        buffer.size = size;
        Ok(buffer)
    }
}

impl Drop for Buffer {
    fn drop(&mut self) {
        if self.ref_count.fetch_sub(1, Ordering::SeqCst) == 1 {
            // Last reference, deallocate if we own the memory
            if self.owns_memory {
                if let Some(pool) = &self.memory_pool {
                    pool.deallocate(self.ptr, self.layout.size(), self.layout.align());
                } else {
                    unsafe {
                        dealloc(self.ptr.as_ptr(), self.layout);
                    }
                }
            }
        }
    }
}

impl Clone for Buffer {
    fn clone(&self) -> Self {
        // Increment the reference count
        self.ref_count.fetch_add(1, Ordering::SeqCst);
        
        Self {
            ptr: self.ptr,
            size: self.size,
            capacity: self.capacity,
            ref_count: self.ref_count.clone(),
            memory_pool: self.memory_pool.clone(),
            layout: self.layout,
            owns_memory: false, // Clones don't own the memory
        }
    }
}

unsafe impl Send for Buffer {}
unsafe impl Sync for Buffer {}