#![allow(dead_code)]

use std::borrow::Cow;
use std::fmt::{self, Debug, Formatter};
use std::ffi::CStr;
use std::os::raw;
use std::ptr::null_mut;
use std::result;

use cuda_runtime;
pub use cuda_runtime::{cudaError_t, cudaMemcpyKind, dim3};

pub fn usize_to_dim3(x: usize) -> dim3 {
    dim3 {
        x: x as raw::c_uint,
        y: 1,
        z: 1,
    }
}

pub struct Error {
    raw: cudaError_t,
}
impl Debug for Error {
    fn fmt(&self, f: &mut Formatter) -> result::Result<(), fmt::Error> {
        write!(f,
               "{}",
               unsafe { CStr::from_ptr(cuda_runtime::cudaGetErrorString(self.raw)) }
                   .to_string_lossy())
    }
}
impl Error {
    fn cuda_error(&self) -> cudaError_t {
        self.raw
    }
    fn error_name(&self) -> Cow<str> {
        unsafe { CStr::from_ptr(cuda_runtime::cudaGetErrorName(self.raw)) }.to_string_lossy()
    }
    fn error_string(&self) -> Cow<str> {
        unsafe { CStr::from_ptr(cuda_runtime::cudaGetErrorString(self.raw)) }.to_string_lossy()
    }
}

pub type Result<T> = result::Result<T, Error>;

pub fn malloc<T>(size: usize) -> Result<*mut T> {
    let mut ptr: *mut T = null_mut();
    let cuda_error =
        unsafe { cuda_runtime::cudaMalloc(&mut ptr as *mut *mut T as *mut *mut raw::c_void, size) };
    if cuda_error == cuda_runtime::cudaError::cudaSuccess {
        assert_ne!(ptr,
                   null_mut(),
                   "cudaMalloc is succeeded, but returned null pointer!");
        Ok(ptr)
    } else {
        Err(Error { raw: cuda_error })
    }
}

pub fn memcpy<T>(dst: *mut T, src: *const T, size: usize, kind: cudaMemcpyKind) -> Result<()> {
    let cuda_error = unsafe {
        cuda_runtime::cudaMemcpy(dst as *mut raw::c_void, src as *mut raw::c_void, size, kind)
    };
    if cuda_error == cuda_runtime::cudaError::cudaSuccess {
        Ok(())
    } else {
        Err(Error { raw: cuda_error })
    }
}

pub fn free<T>(devptr: *mut T) -> Result<()> {
    let cuda_error = unsafe { cuda_runtime::cudaFree(devptr as *mut raw::c_void) };
    if cuda_error == cuda_runtime::cudaError::cudaSuccess {
        Ok(())
    } else {
        Err(Error { raw: cuda_error })
    }
}

pub fn configure_call(grid_dim: dim3, block_dim: dim3, shared_mem: usize) -> Result<()> {
    let cuda_error =
        unsafe { cuda_runtime::cudaConfigureCall(grid_dim, block_dim, shared_mem, null_mut()) };
    if cuda_error == cuda_runtime::cudaError::cudaSuccess {
        Ok(())
    } else {
        Err(Error { raw: cuda_error })
    }
}

pub fn launch(func: *const raw::c_void,
              grid_dim: dim3,
              block_dim: dim3,
              args: &mut [*mut raw::c_void],
              shared_mem: usize)
              -> Result<()> {
    let cuda_error = unsafe {
        cuda_runtime::cudaLaunchKernel(func,
                                       grid_dim,
                                       block_dim,
                                       args.as_mut_ptr(),
                                       shared_mem,
                                       null_mut())
    };
    if cuda_error == cuda_runtime::cudaError::cudaSuccess {
        Ok(())
    } else {
        Err(Error { raw: cuda_error })
    }
}

pub fn last_error() -> Result<()> {
    let cuda_error = unsafe { cuda_runtime::cudaGetLastError() };
    if cuda_error == cuda_runtime::cudaError::cudaSuccess {
        Ok(())
    } else {
        Err(Error { raw: cuda_error })
    }
}
