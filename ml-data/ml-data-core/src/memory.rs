//! Memory management utilities for efficient data handling

use std::alloc::{alloc, alloc_zeroed, dealloc, Layout};
use std::collections::HashMap;
use std::ptr::NonNull;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Mutex, RwLock};
use std::time::Instant;

use crate::error::{Error, Result};

/// Interface for memory pool implementations
pub trait MemoryPool: Send + Sync {
    /// Allocate memory with the given size and alignment
    fn allocate(&self, size: usize, alignment: usize) -> Result<NonNull<u8>>;
    
    /// Reallocate memory, potentially growing or shrinking the allocation
    fn reallocate(
        &self,
        ptr: NonNull<u8>,
        old_size: usize,
        new_size: usize,
        alignment: usize,
    ) -> Result<NonNull<u8>>;
    
    /// Deallocate previously allocated memory
    fn deallocate(&self, ptr: NonNull<u8>, size: usize, alignment: usize);
    
    /// Get memory usage statistics for this pool
    fn usage_stats(&self) -> MemoryPoolStats;
}

/// Memory usage statistics for a memory pool
#[derive(Debug, Clone)]
pub struct MemoryPoolStats {
    /// Total allocated memory in bytes
    pub allocated_bytes: usize,
    
    /// Total deallocated memory in bytes
    pub deallocated_bytes: usize,
    
    /// Current outstanding allocations in bytes
    pub current_bytes: usize,
    
    /// Peak memory usage in bytes
    pub peak_bytes: usize,
    
    /// Number of active allocations
    pub allocation_count: usize,
    
    /// Total number of allocations performed
    pub total_allocations: usize,
    
    /// Total number of deallocations performed
    pub total_deallocations: usize,
}

/// Default system memory pool that uses the global allocator
pub struct SystemMemoryPool {
    /// Stats for this memory pool
    stats: Mutex<MemoryPoolStats>,
}

impl SystemMemoryPool {
    /// Create a new system memory pool
    pub fn new() -> Self {
        Self {
            stats: Mutex::new(MemoryPoolStats {
                allocated_bytes: 0,
                deallocated_bytes: 0,
                current_bytes: 0,
                peak_bytes: 0,
                allocation_count: 0,
                total_allocations: 0,
                total_deallocations: 0,
            }),
        }
    }
}

impl Default for SystemMemoryPool {
    fn default() -> Self {
        Self::new()
    }
}

impl MemoryPool for SystemMemoryPool {
    fn allocate(&self, size: usize, alignment: usize) -> Result<NonNull<u8>> {
        let layout = Layout::from_size_align(size, alignment)
            .map_err(|_| Error::LayoutError("Invalid memory layout".into()))?;
        
        let ptr = unsafe { alloc_zeroed(layout) };
        let ptr = NonNull::new(ptr).ok_or(Error::MemoryAllocationFailed)?;
        
        // Update stats
        let mut stats = self.stats.lock().unwrap();
        stats.allocated_bytes += size;
        stats.current_bytes += size;
        stats.peak_bytes = stats.peak_bytes.max(stats.current_bytes);
        stats.allocation_count += 1;
        stats.total_allocations += 1;
        
        Ok(ptr)
    }
    
    fn reallocate(
        &self,
        ptr: NonNull<u8>,
        old_size: usize,
        new_size: usize,
        alignment: usize,
    ) -> Result<NonNull<u8>> {
        let old_layout = Layout::from_size_align(old_size, alignment)
            .map_err(|_| Error::LayoutError("Invalid old memory layout".into()))?;
        
        let new_layout = Layout::from_size_align(new_size, alignment)
            .map_err(|_| Error::LayoutError("Invalid new memory layout".into()))?;
        
        let new_ptr = unsafe {
            let new_ptr = alloc_zeroed(new_layout);
            if !new_ptr.is_null() {
                // Copy existing data
                std::ptr::copy_nonoverlapping(
                    ptr.as_ptr(),
                    new_ptr,
                    old_size.min(new_size)
                );
                
                // Free old memory
                dealloc(ptr.as_ptr(), old_layout);
            }
            
            new_ptr
        };
        
        let new_ptr = NonNull::new(new_ptr).ok_or(Error::MemoryAllocationFailed)?;
        
        // Update stats
        let mut stats = self.stats.lock().unwrap();
        stats.allocated_bytes += new_size;
        stats.deallocated_bytes += old_size;
        stats.current_bytes = stats.current_bytes.saturating_add(new_size).saturating_sub(old_size);
        stats.peak_bytes = stats.peak_bytes.max(stats.current_bytes);
        stats.total_allocations += 1;
        stats.total_deallocations += 1;
        
        Ok(new_ptr)
    }
    
    fn deallocate(&self, ptr: NonNull<u8>, size: usize, alignment: usize) {
        let layout = Layout::from_size_align(size, alignment).expect("Invalid layout in deallocate");
        
        unsafe {
            dealloc(ptr.as_ptr(), layout);
        }
        
        // Update stats
        let mut stats = self.stats.lock().unwrap();
        stats.deallocated_bytes += size;
        stats.current_bytes = stats.current_bytes.saturating_sub(size);
        stats.allocation_count = stats.allocation_count.saturating_sub(1);
        stats.total_deallocations += 1;
    }
    
    fn usage_stats(&self) -> MemoryPoolStats {
        self.stats.lock().unwrap().clone()
    }
}

/// An arena-based memory pool that allocates memory in large chunks
pub struct ArenaMemoryPool {
    /// The size of each arena in bytes
    arena_size: usize,
    
    /// The current arena being allocated from
    current_arena: Mutex<Vec<u8>>,
    
    /// The current position in the arena
    position: AtomicUsize,
    
    /// Completed arenas that are no longer being allocated from
    completed_arenas: Mutex<Vec<Vec<u8>>>,
    
    /// Stats for this memory pool
    stats: Mutex<MemoryPoolStats>,
}

impl ArenaMemoryPool {
    /// Create a new arena memory pool with the given arena size
    pub fn new(arena_size: usize) -> Self {
        Self {
            arena_size,
            current_arena: Mutex::new(Vec::with_capacity(arena_size)),
            position: AtomicUsize::new(0),
            completed_arenas: Mutex::new(Vec::new()),
            stats: Mutex::new(MemoryPoolStats {
                allocated_bytes: 0,
                deallocated_bytes: 0,
                current_bytes: 0,
                peak_bytes: 0,
                allocation_count: 0,
                total_allocations: 0,
                total_deallocations: 0,
            }),
        }
    }
    
    /// Reset the arena, invalidating all previous allocations
    pub fn reset(&self) {
        let mut current = self.current_arena.lock().unwrap();
        let mut completed = self.completed_arenas.lock().unwrap();
        
        // Move the current arena to completed if it has allocations
        if self.position.load(Ordering::SeqCst) > 0 {
            if !current.is_empty() {
                completed.push(std::mem::take(&mut *current));
            }
        }
        
        // Create a new current arena
        *current = Vec::with_capacity(self.arena_size);
        self.position.store(0, Ordering::SeqCst);
        
        // Clear completed arenas
        completed.clear();
        
        // Reset stats
        let mut stats = self.stats.lock().unwrap();
        stats.current_bytes = 0;
        stats.allocation_count = 0;
    }
}

impl MemoryPool for ArenaMemoryPool {
    fn allocate(&self, size: usize, alignment: usize) -> Result<NonNull<u8>> {
        if size > self.arena_size {
            return Err(Error::InvalidArgument(
                format!("Allocation size {} exceeds arena size {}", size, self.arena_size)
            ));
        }
        
        // Get the current position and align it
        let mut position = self.position.load(Ordering::SeqCst);
        let align_offset = (alignment - (position % alignment)) % alignment;
        position += align_offset;
        
        // Check if we have enough space in the current arena
        if position + size > self.arena_size {
            // Need to create a new arena
            let mut current = self.current_arena.lock().unwrap();
            let mut completed = self.completed_arenas.lock().unwrap();
            
            // Move the current arena to completed if it has allocations
            if position > 0 {
                if !current.is_empty() {
                    completed.push(std::mem::take(&mut *current));
                }
            }
            
            // Create a new current arena
            *current = Vec::with_capacity(self.arena_size);
            unsafe {
                current.set_len(self.arena_size);
            }
            
            // Reset position
            position = 0;
            self.position.store(0, Ordering::SeqCst);
        }
        
        // Allocate from the current arena
        let new_position = position + size;
        self.position.store(new_position, Ordering::SeqCst);
        
        // Get a pointer to the allocated memory
        let ptr = {
            let current = self.current_arena.lock().unwrap();
            unsafe {
                current.as_ptr().add(position) as *mut u8
            }
        };
        
        // Update stats
        let mut stats = self.stats.lock().unwrap();
        stats.allocated_bytes += size;
        stats.current_bytes += size;
        stats.peak_bytes = stats.peak_bytes.max(stats.current_bytes);
        stats.allocation_count += 1;
        stats.total_allocations += 1;
        
        Ok(NonNull::new(ptr).unwrap())
    }
    
    fn reallocate(
        &self,
        _ptr: NonNull<u8>,
        _old_size: usize,
        _new_size: usize,
        _alignment: usize,
    ) -> Result<NonNull<u8>> {
        Err(Error::NotImplemented(
            "ArenaMemoryPool does not support reallocation".into()
        ))
    }
    
    fn deallocate(&self, _ptr: NonNull<u8>, size: usize, _alignment: usize) {
        // Arena allocators don't individually deallocate
        // but we update the stats for consistency
        let mut stats = self.stats.lock().unwrap();
        stats.deallocated_bytes += size;
        stats.allocation_count = stats.allocation_count.saturating_sub(1);
        stats.total_deallocations += 1;
        // Note: current_bytes is not updated as arena memory is only freed on reset
    }
    
    fn usage_stats(&self) -> MemoryPoolStats {
        self.stats.lock().unwrap().clone()
    }
}

/// A memory pool allocator that aligns allocations to page boundaries
pub struct PageAlignedAllocator;

impl PageAlignedAllocator {
    /// Create a new page-aligned allocator
    pub fn new() -> Self {
        Self
    }
    
    /// Allocate memory aligned to page boundaries
    pub fn allocate(&self, size: usize) -> Result<NonNull<u8>> {
        let layout = Layout::from_size_align(size, PAGE_SIZE)
            .map_err(|_| Error::LayoutError("Invalid memory layout for page alignment".into()))?;
        
        let ptr = unsafe { alloc_zeroed(layout) };
        NonNull::new(ptr).ok_or(Error::MemoryAllocationFailed)
    }
    
    /// Deallocate page-aligned memory
    pub fn deallocate(&self, ptr: NonNull<u8>, size: usize) {
        let layout = Layout::from_size_align(size, PAGE_SIZE)
            .expect("Invalid layout in page-aligned deallocate");
        
        unsafe {
            dealloc(ptr.as_ptr(), layout);
        }
    }
}

impl Default for PageAlignedAllocator {
    fn default() -> Self {
        Self::new()
    }
}

/// Memory budget for controlling and tracking memory usage
pub struct MemoryBudget {
    /// The total memory budget in bytes
    pub total_budget: usize,
    
    /// The current memory usage in bytes
    pub current_usage: AtomicUsize,
    
    /// Memory usage statistics
    pub stats: RwLock<MemoryStats>,
}

/// Memory usage statistics
#[derive(Debug, Clone)]
pub struct MemoryStats {
    /// Total allocated memory in bytes
    pub total_allocated: usize,
    
    /// Component-specific memory usage
    pub component_usage: HashMap<String, usize>,
    
    /// Peak memory usage in bytes
    pub peak_usage: usize,
    
    /// Buffer usages (name, current, capacity)
    pub buffer_usages: Vec<(String, usize, usize)>,
    
    /// Last time the stats were updated
    pub last_updated: Instant,
}

impl MemoryBudget {
    /// Create a new memory budget with the given total budget
    pub fn new(total_budget: usize) -> Self {
        Self {
            total_budget,
            current_usage: AtomicUsize::new(0),
            stats: RwLock::new(MemoryStats {
                total_allocated: 0,
                component_usage: HashMap::new(),
                peak_usage: 0,
                buffer_usages: Vec::new(),
                last_updated: Instant::now(),
            }),
        }
    }
    
    /// Try to allocate memory, returns false if exceeds budget
    pub fn try_allocate(&self, bytes: usize) -> bool {
        let mut current = self.current_usage.load(Ordering::SeqCst);
        
        loop {
            if current + bytes > self.total_budget {
                return false;
            }
            
            match self.current_usage.compare_exchange(
                current,
                current + bytes,
                Ordering::SeqCst,
                Ordering::SeqCst,
            ) {
                Ok(_) => {
                    // Update stats
                    let mut stats = self.stats.write().unwrap();
                    stats.total_allocated += bytes;
                    stats.peak_usage = stats.peak_usage.max(current + bytes);
                    stats.last_updated = Instant::now();
                    
                    return true;
                }
                Err(actual) => {
                    current = actual;
                }
            }
        }
    }
    
    /// Allocate memory, returning an error if it exceeds budget
    pub fn allocate(&self, bytes: usize) -> Result<()> {
        if self.try_allocate(bytes) {
            Ok(())
        } else {
            Err(Error::MemoryBudgetExceeded {
                requested: bytes,
                available: self.total_budget - self.current_usage.load(Ordering::SeqCst),
            })
        }
    }
    
    /// Release previously allocated memory
    pub fn release(&self, bytes: usize) {
        let prev = self.current_usage.fetch_sub(bytes, Ordering::SeqCst);
        
        // Update stats
        let mut stats = self.stats.write().unwrap();
        stats.last_updated = Instant::now();
        
        debug_assert!(
            prev >= bytes,
            "Attempted to release more memory than allocated: prev={}, release={}",
            prev,
            bytes
        );
    }
    
    /// Register component memory usage
    pub fn register_component_usage(&self, component: &str, bytes: usize) {
        let mut stats = self.stats.write().unwrap();
        *stats.component_usage.entry(component.to_string()).or_insert(0) = bytes;
        stats.last_updated = Instant::now();
    }
    
    /// Register buffer usage
    pub fn register_buffer_usage(&self, name: &str, current: usize, capacity: usize) {
        let mut stats = self.stats.write().unwrap();
        
        // Remove previous entry for this buffer if it exists
        stats.buffer_usages.retain(|(n, _, _)| n != name);
        
        // Add new entry
        stats.buffer_usages.push((name.to_string(), current, capacity));
        stats.last_updated = Instant::now();
    }
    
    /// Get current memory usage
    pub fn usage(&self) -> usize {
        self.current_usage.load(Ordering::SeqCst)
    }
    
    /// Get percent of budget used
    pub fn percent_used(&self) -> f64 {
        let usage = self.usage() as f64;
        let budget = self.total_budget as f64;
        (usage / budget) * 100.0
    }
    
    /// Get memory statistics
    pub fn stats(&self) -> MemoryStats {
        self.stats.read().unwrap().clone()
    }
}