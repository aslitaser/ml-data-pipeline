//! Tensor implementations for multidimensional data

use std::fmt;
use std::marker::PhantomData;
use std::ops::{Deref, Range};
use std::sync::Arc;

use bytemuck::Pod;

use crate::buffer::Buffer;
use crate::error::{Error, Result};
use crate::schema::TensorFormat;

/// Trait for types that can be used in tensors
pub trait TensorType: Pod + Send + Sync + 'static {}

impl<T: Pod + Send + Sync + 'static> TensorType for T {}

/// A dense multidimensional tensor
pub struct DenseTensor<T: TensorType> {
    /// Raw pointer to the data
    data_ptr: *const T,
    
    /// Shape of the tensor (dimensions)
    shape: Vec<usize>,
    
    /// Strides of the tensor (bytes to skip per dimension)
    strides: Vec<usize>,
    
    /// Total number of elements
    size: usize,
    
    /// Underlying buffer (if owned)
    buffer: Option<Buffer>,
    
    /// Phantom data for the element type
    _phantom: PhantomData<T>,
}

impl<T: TensorType> DenseTensor<T> {
    /// Create a new tensor with the given shape
    pub fn new(shape: Vec<usize>) -> Result<Self> {
        let size = shape.iter().product();
        let strides = Self::compute_strides(&shape);
        let buffer = Buffer::new_zeroed(size * std::mem::size_of::<T>())?;
        
        Ok(Self {
            data_ptr: buffer.as_ptr() as *const T,
            shape,
            strides,
            size,
            buffer: Some(buffer),
            _phantom: PhantomData,
        })
    }
    
    /// Create a tensor from a slice
    pub fn from_slice(data: &[T], shape: Vec<usize>) -> Result<Self> {
        let expected_size: usize = shape.iter().product();
        if data.len() != expected_size {
            return Err(Error::InvalidArgument(format!(
                "Slice length {} does not match shape product {}",
                data.len(),
                expected_size
            )));
        }
        
        let strides = Self::compute_strides(&shape);
        let buffer = Buffer::from_slice(data)?;
        
        Ok(Self {
            data_ptr: buffer.as_ptr() as *const T,
            shape,
            strides,
            size: expected_size,
            buffer: Some(buffer),
            _phantom: PhantomData,
        })
    }
    
    /// Create a tensor from a vector
    pub fn from_vec(data: Vec<T>, shape: Vec<usize>) -> Result<Self> {
        Self::from_slice(&data, shape)
    }
    
    /// Create a tensor with custom strides
    pub fn with_strides(data: &[T], shape: Vec<usize>, strides: Vec<usize>) -> Result<Self> {
        if shape.len() != strides.len() {
            return Err(Error::InvalidArgument(
                "Shape and strides must have the same length".into()
            ));
        }
        
        let expected_size: usize = shape.iter().product();
        if data.len() < expected_size {
            return Err(Error::InvalidArgument(format!(
                "Data length {} is too small for shape product {}",
                data.len(),
                expected_size
            )));
        }
        
        let buffer = Buffer::from_slice(data)?;
        
        Ok(Self {
            data_ptr: buffer.as_ptr() as *const T,
            shape,
            strides,
            size: expected_size,
            buffer: Some(buffer),
            _phantom: PhantomData,
        })
    }
    
    /// Create a tensor from raw parts
    /// 
    /// # Safety
    /// 
    /// The caller must ensure that:
    /// - `data_ptr` points to valid memory of at least `size` elements
    /// - The memory remains valid for the lifetime of the tensor
    /// - The shape and strides are consistent
    pub unsafe fn from_raw_parts(
        data_ptr: *const T,
        shape: Vec<usize>,
        strides: Option<Vec<usize>>,
        size: usize,
        buffer: Option<Buffer>,
    ) -> Result<Self> {
        let strides = match strides {
            Some(s) => {
                if s.len() != shape.len() {
                    return Err(Error::InvalidArgument(
                        "Shape and strides must have the same length".into()
                    ));
                }
                s
            }
            None => Self::compute_strides(&shape),
        };
        
        Ok(Self {
            data_ptr,
            shape,
            strides,
            size,
            buffer,
            _phantom: PhantomData,
        })
    }
    
    /// Compute strides for a given shape
    fn compute_strides(shape: &[usize]) -> Vec<usize> {
        let mut strides = vec![0; shape.len()];
        let element_size = std::mem::size_of::<T>();
        
        let mut stride = element_size;
        for i in (0..shape.len()).rev() {
            strides[i] = stride;
            stride *= shape[i];
        }
        
        strides
    }
    
    /// Get the shape of this tensor
    pub fn shape(&self) -> &[usize] {
        &self.shape
    }
    
    /// Get the strides of this tensor
    pub fn strides(&self) -> &[usize] {
        &self.strides
    }
    
    /// Get the total number of elements in this tensor
    pub fn len(&self) -> usize {
        self.size
    }
    
    /// Check if this tensor is empty
    pub fn is_empty(&self) -> bool {
        self.size == 0
    }
    
    /// Get the number of dimensions in this tensor
    pub fn ndim(&self) -> usize {
        self.shape.len()
    }
    
    /// Get a raw pointer to the data
    pub fn data_ptr(&self) -> *const T {
        self.data_ptr
    }
    
    /// Calculate the size in bytes of this tensor
    pub fn size_bytes(&self) -> usize {
        self.size * std::mem::size_of::<T>()
    }
    
    /// Reshape this tensor to a new shape
    pub fn reshape(&self, new_shape: Vec<usize>) -> Result<Self> {
        let new_size = new_shape.iter().product();
        if new_size != self.size {
            return Err(Error::InvalidArgument(format!(
                "Cannot reshape tensor of size {} to size {}",
                self.size, new_size
            )));
        }
        
        // If tensor is contiguous, we can create a view with the new shape
        if self.is_contiguous() {
            unsafe {
                Self::from_raw_parts(
                    self.data_ptr,
                    new_shape,
                    None, // Compute new strides
                    self.size,
                    self.buffer.clone(),
                )
            }
        } else {
            // Need to create a new contiguous tensor
            let mut new_tensor = Self::new(new_shape)?;
            self.copy_to(&mut new_tensor)?;
            Ok(new_tensor)
        }
    }
    
    /// Check if the tensor is contiguous in memory
    pub fn is_contiguous(&self) -> bool {
        // Check if strides match the expected contiguous strides
        let contiguous_strides = Self::compute_strides(&self.shape);
        self.strides == contiguous_strides
    }
    
    /// Get a slice of the tensor with the given ranges
    pub fn slice(&self, ranges: &[Range<usize>]) -> Result<Self> {
        if ranges.len() != self.shape.len() {
            return Err(Error::InvalidArgument(
                "Number of ranges must match number of dimensions".into()
            ));
        }
        
        // Validate ranges
        for (i, range) in ranges.iter().enumerate() {
            if range.end > self.shape[i] {
                return Err(Error::IndexOutOfBounds);
            }
        }
        
        // Calculate new shape
        let mut new_shape = Vec::with_capacity(self.shape.len());
        for range in ranges {
            new_shape.push(range.end - range.start);
        }
        
        // Calculate offset to start of slice
        let mut offset = 0;
        for (i, range) in ranges.iter().enumerate() {
            offset += range.start * self.strides[i] / std::mem::size_of::<T>();
        }
        
        let new_size = new_shape.iter().product();
        
        unsafe {
            Self::from_raw_parts(
                self.data_ptr.add(offset),
                new_shape,
                Some(self.strides.clone()),
                new_size,
                self.buffer.clone(),
            )
        }
    }
    
    /// Copy this tensor to another tensor
    pub fn copy_to(&self, dst: &mut DenseTensor<T>) -> Result<()> {
        if self.shape != dst.shape {
            return Err(Error::InvalidArgument(
                "Destination tensor must have the same shape".into()
            ));
        }
        
        // Simple case: both tensors are contiguous
        if self.is_contiguous() && dst.is_contiguous() {
            unsafe {
                std::ptr::copy_nonoverlapping(
                    self.data_ptr,
                    dst.data_ptr as *mut T,
                    self.size,
                );
            }
            return Ok(());
        }
        
        // Complex case: need to copy element by element
        // For simplicity, only implement 1D and 2D case
        match self.shape.len() {
            1 => {
                for i in 0..self.shape[0] {
                    unsafe {
                        let src_idx = i * self.strides[0] / std::mem::size_of::<T>();
                        let dst_idx = i * dst.strides[0] / std::mem::size_of::<T>();
                        
                        *((dst.data_ptr as *mut T).add(dst_idx)) = *self.data_ptr.add(src_idx);
                    }
                }
            }
            2 => {
                for i in 0..self.shape[0] {
                    for j in 0..self.shape[1] {
                        unsafe {
                            let src_idx = i * self.strides[0] / std::mem::size_of::<T>()
                                + j * self.strides[1] / std::mem::size_of::<T>();
                            let dst_idx = i * dst.strides[0] / std::mem::size_of::<T>()
                                + j * dst.strides[1] / std::mem::size_of::<T>();
                            
                            *((dst.data_ptr as *mut T).add(dst_idx)) = *self.data_ptr.add(src_idx);
                        }
                    }
                }
            }
            _ => {
                return Err(Error::NotImplemented(
                    "Copying tensors with more than 2 dimensions is not implemented".into()
                ));
            }
        }
        
        Ok(())
    }
    
    /// Get the value at the specified indices
    pub fn get(&self, indices: &[usize]) -> Result<T> {
        if indices.len() != self.shape.len() {
            return Err(Error::InvalidArgument(
                "Number of indices must match number of dimensions".into()
            ));
        }
        
        // Validate indices
        for (i, &idx) in indices.iter().enumerate() {
            if idx >= self.shape[i] {
                return Err(Error::IndexOutOfBounds);
            }
        }
        
        // Calculate offset
        let mut offset = 0;
        for (i, &idx) in indices.iter().enumerate() {
            offset += idx * self.strides[i] / std::mem::size_of::<T>();
        }
        
        unsafe {
            Ok(*self.data_ptr.add(offset))
        }
    }
    
    /// Set the value at the specified indices
    pub fn set(&mut self, indices: &[usize], value: T) -> Result<()> {
        if indices.len() != self.shape.len() {
            return Err(Error::InvalidArgument(
                "Number of indices must match number of dimensions".into()
            ));
        }
        
        // Validate indices
        for (i, &idx) in indices.iter().enumerate() {
            if idx >= self.shape[i] {
                return Err(Error::IndexOutOfBounds);
            }
        }
        
        // Calculate offset
        let mut offset = 0;
        for (i, &idx) in indices.iter().enumerate() {
            offset += idx * self.strides[i] / std::mem::size_of::<T>();
        }
        
        unsafe {
            *((self.data_ptr as *mut T).add(offset)) = value;
        }
        
        Ok(())
    }
}

impl<T: TensorType> Clone for DenseTensor<T> {
    fn clone(&self) -> Self {
        // For owned tensors, clone the buffer
        if let Some(buffer) = &self.buffer {
            let cloned_buffer = buffer.clone();
            let data_ptr = cloned_buffer.as_ptr() as *const T;
            
            Self {
                data_ptr,
                shape: self.shape.clone(),
                strides: self.strides.clone(),
                size: self.size,
                buffer: Some(cloned_buffer),
                _phantom: PhantomData,
            }
        } else {
            // For view tensors, just clone the reference
            Self {
                data_ptr: self.data_ptr,
                shape: self.shape.clone(),
                strides: self.strides.clone(),
                size: self.size,
                buffer: None,
                _phantom: PhantomData,
            }
        }
    }
}

impl<T: TensorType + fmt::Debug> fmt::Debug for DenseTensor<T> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "DenseTensor<{:?}>{{ shape: {:?}, size: {} }}", 
            std::any::type_name::<T>(),
            self.shape,
            self.size
        )
    }
}

/// A sparse tensor representation
pub struct SparseTensor<T: TensorType> {
    /// Pointer to values
    values_ptr: *const T,
    
    /// Number of non-zero values
    nnz: usize,
    
    /// Pointer to indices
    indices_ptr: *const u8,
    
    /// Size of indices in bytes
    indices_size: usize,
    
    /// Tensor shape
    shape: Vec<usize>,
    
    /// Sparse format
    format: TensorFormat,
    
    /// Underlying buffers (values, indices) if owned
    buffers: Option<(Buffer, Buffer)>,
    
    /// Phantom data for the element type
    _phantom: PhantomData<T>,
}

impl<T: TensorType> SparseTensor<T> {
    /// Create a new COO sparse tensor
    pub fn new_coo(shape: Vec<usize>) -> Result<Self> {
        let values_buffer = Buffer::new(0)?;
        let indices_buffer = Buffer::new(0)?;
        
        Ok(Self {
            values_ptr: values_buffer.as_ptr() as *const T,
            nnz: 0,
            indices_ptr: indices_buffer.as_ptr(),
            indices_size: 0,
            shape,
            format: TensorFormat::COO,
            buffers: Some((values_buffer, indices_buffer)),
            _phantom: PhantomData,
        })
    }
    
    /// Create a new CSR sparse tensor
    pub fn new_csr(shape: Vec<usize>) -> Result<Self> {
        if shape.len() != 2 {
            return Err(Error::InvalidArgument(
                "CSR format requires a 2D tensor shape".into()
            ));
        }
        
        let values_buffer = Buffer::new(0)?;
        let indices_buffer = Buffer::new(0)?;
        
        Ok(Self {
            values_ptr: values_buffer.as_ptr() as *const T,
            nnz: 0,
            indices_ptr: indices_buffer.as_ptr(),
            indices_size: 0,
            shape,
            format: TensorFormat::CSR,
            buffers: Some((values_buffer, indices_buffer)),
            _phantom: PhantomData,
        })
    }
    
    /// Create a sparse tensor from raw parts
    /// 
    /// # Safety
    /// 
    /// The caller must ensure that:
    /// - `values_ptr` points to a valid array of `nnz` elements
    /// - `indices_ptr` points to a valid array of appropriate indices format
    /// - The memory remains valid for the lifetime of the tensor
    pub unsafe fn from_raw_parts(
        values_ptr: *const T,
        nnz: usize,
        indices_ptr: *const u8,
        indices_size: usize,
        shape: Vec<usize>,
        format: TensorFormat,
        buffers: Option<(Buffer, Buffer)>,
    ) -> Result<Self> {
        Ok(Self {
            values_ptr,
            nnz,
            indices_ptr,
            indices_size,
            shape,
            format,
            buffers,
            _phantom: PhantomData,
        })
    }
    
    /// Create a COO sparse tensor from values and indices
    pub fn from_coo(
        values: &[T],
        indices: &[usize],
        shape: Vec<usize>,
    ) -> Result<Self> {
        let ndim = shape.len();
        if indices.len() != values.len() * ndim {
            return Err(Error::InvalidArgument(
                "Indices length must be values length times dimensions".into()
            ));
        }
        
        // Check indices are within bounds
        for dim in 0..ndim {
            for i in 0..values.len() {
                let idx = indices[i * ndim + dim];
                if idx >= shape[dim] {
                    return Err(Error::IndexOutOfBounds);
                }
            }
        }
        
        let values_buffer = Buffer::from_slice(values)?;
        let indices_buffer = Buffer::from_slice(indices)?;
        
        Ok(Self {
            values_ptr: values_buffer.as_ptr() as *const T,
            nnz: values.len(),
            indices_ptr: indices_buffer.as_ptr(),
            indices_size: indices_buffer.size(),
            shape,
            format: TensorFormat::COO,
            buffers: Some((values_buffer, indices_buffer)),
            _phantom: PhantomData,
        })
    }
    
    /// Create a CSR sparse tensor from values, column indices, and row pointers
    pub fn from_csr(
        values: &[T],
        col_indices: &[usize],
        row_ptrs: &[usize],
        shape: Vec<usize>,
    ) -> Result<Self> {
        if shape.len() != 2 {
            return Err(Error::InvalidArgument(
                "CSR format requires a 2D tensor shape".into()
            ));
        }
        
        if col_indices.len() != values.len() {
            return Err(Error::InvalidArgument(
                "Column indices length must match values length".into()
            ));
        }
        
        if row_ptrs.len() != shape[0] + 1 {
            return Err(Error::InvalidArgument(
                "Row pointers length must be number of rows + 1".into()
            ));
        }
        
        // Check indices are within bounds
        for &col in col_indices {
            if col >= shape[1] {
                return Err(Error::IndexOutOfBounds);
            }
        }
        
        // Check row pointers are valid
        if row_ptrs[0] != 0 || row_ptrs[shape[0]] != values.len() {
            return Err(Error::InvalidArgument(
                "Row pointers must start with 0 and end with number of values".into()
            ));
        }
        
        let values_buffer = Buffer::from_slice(values)?;
        
        // Pack indices into a single buffer
        let mut indices = Vec::with_capacity(col_indices.len() + row_ptrs.len());
        indices.extend_from_slice(col_indices);
        indices.extend_from_slice(row_ptrs);
        
        let indices_buffer = Buffer::from_slice(&indices)?;
        
        Ok(Self {
            values_ptr: values_buffer.as_ptr() as *const T,
            nnz: values.len(),
            indices_ptr: indices_buffer.as_ptr(),
            indices_size: indices_buffer.size(),
            shape,
            format: TensorFormat::CSR,
            buffers: Some((values_buffer, indices_buffer)),
            _phantom: PhantomData,
        })
    }
    
    /// Get the number of non-zero elements
    pub fn nnz(&self) -> usize {
        self.nnz
    }
    
    /// Get the shape of this tensor
    pub fn shape(&self) -> &[usize] {
        &self.shape
    }
    
    /// Get the format of this tensor
    pub fn format_type(&self) -> TensorFormat {
        self.format
    }
    
    /// Get a raw pointer to the values
    pub fn values_ptr(&self) -> *const T {
        self.values_ptr
    }
    
    /// Get a raw pointer to the indices
    pub fn indices_ptr(&self) -> *const u8 {
        self.indices_ptr
    }
    
    /// Get the size of the values in bytes
    pub fn values_size_bytes(&self) -> usize {
        self.nnz * std::mem::size_of::<T>()
    }
    
    /// Get the size of the indices in bytes
    pub fn indices_size_bytes(&self) -> usize {
        self.indices_size
    }
    
    /// Convert this sparse tensor to a dense tensor
    pub fn to_dense(&self) -> Result<DenseTensor<T>> {
        let mut dense = DenseTensor::new(self.shape.clone())?;
        
        match self.format {
            TensorFormat::COO => {
                let ndim = self.shape.len();
                let indices = unsafe {
                    std::slice::from_raw_parts(
                        self.indices_ptr as *const usize,
                        self.nnz * ndim,
                    )
                };
                
                let values = unsafe {
                    std::slice::from_raw_parts(self.values_ptr, self.nnz)
                };
                
                for i in 0..self.nnz {
                    let mut idx = Vec::with_capacity(ndim);
                    for dim in 0..ndim {
                        idx.push(indices[i * ndim + dim]);
                    }
                    
                    dense.set(&idx, values[i])?;
                }
            }
            TensorFormat::CSR => {
                if self.shape.len() != 2 {
                    return Err(Error::InvalidArgument(
                        "CSR format requires a 2D tensor shape".into()
                    ));
                }
                
                let rows = self.shape[0];
                let cols = self.shape[1];
                
                let indices_len = self.indices_size / std::mem::size_of::<usize>();
                let indices = unsafe {
                    std::slice::from_raw_parts(
                        self.indices_ptr as *const usize,
                        indices_len,
                    )
                };
                
                let col_indices = &indices[0..self.nnz];
                let row_ptrs = &indices[self.nnz..self.nnz + rows + 1];
                
                let values = unsafe {
                    std::slice::from_raw_parts(self.values_ptr, self.nnz)
                };
                
                for row in 0..rows {
                    let start = row_ptrs[row];
                    let end = row_ptrs[row + 1];
                    
                    for i in start..end {
                        let col = col_indices[i];
                        dense.set(&[row, col], values[i])?;
                    }
                }
            }
            _ => {
                return Err(Error::NotImplemented(
                    format!("Conversion from {:?} to dense is not implemented", self.format)
                ));
            }
        }
        
        Ok(dense)
    }
}

impl<T: TensorType> Clone for SparseTensor<T> {
    fn clone(&self) -> Self {
        // For owned tensors, clone the buffers
        if let Some((values_buffer, indices_buffer)) = &self.buffers {
            let cloned_values = values_buffer.clone();
            let cloned_indices = indices_buffer.clone();
            
            Self {
                values_ptr: cloned_values.as_ptr() as *const T,
                nnz: self.nnz,
                indices_ptr: cloned_indices.as_ptr(),
                indices_size: self.indices_size,
                shape: self.shape.clone(),
                format: self.format,
                buffers: Some((cloned_values, cloned_indices)),
                _phantom: PhantomData,
            }
        } else {
            // For view tensors, just clone the references
            Self {
                values_ptr: self.values_ptr,
                nnz: self.nnz,
                indices_ptr: self.indices_ptr,
                indices_size: self.indices_size,
                shape: self.shape.clone(),
                format: self.format,
                buffers: None,
                _phantom: PhantomData,
            }
        }
    }
}

impl<T: TensorType + fmt::Debug> fmt::Debug for SparseTensor<T> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "SparseTensor<{:?}>{{ format: {:?}, shape: {:?}, nnz: {} }}", 
            std::any::type_name::<T>(),
            self.format,
            self.shape,
            self.nnz
        )
    }
}