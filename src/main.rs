// This code is based on CUDA's sample code, 0_Simple/vectorAdd/
// This software contains source code provided by NVIDIA Corporation.

extern crate libc;

mod cuda_runtime;
mod cuda_ffi;

use std::os::raw::{c_int, c_void};
use cuda_ffi::cudaMemcpyKind;

extern "C" {
    fn vectorAdd(a: *const f32, b: *const f32, c: *mut f32, n: c_int) -> c_void;
//static vectorAdd: *const c_void;
}

fn main() {
    let n: usize = 50000;

    let h_a = vec![1.0f32; n];
    let h_b = vec![2.0f32; n];
    let mut h_c = vec![0.0f32; n];

    let mut d_a: *mut f32 = cuda_ffi::malloc(n * 4).unwrap();
    let mut d_b: *mut f32 = cuda_ffi::malloc(n * 4).unwrap();
    let mut d_c: *mut f32 = cuda_ffi::malloc(n * 4).unwrap();

    cuda_ffi::memcpy(d_a,
                     h_a.as_ptr(),
                     n * 4,
                     cudaMemcpyKind::cudaMemcpyHostToDevice)
        .unwrap();
    cuda_ffi::memcpy(d_b,
                     h_b.as_ptr(),
                     n * 4,
                     cudaMemcpyKind::cudaMemcpyHostToDevice)
        .unwrap();

    let threads_per_block = 256usize;
    let blockdim = cuda_ffi::usize_to_dim3(threads_per_block);
    let griddim = cuda_ffi::usize_to_dim3((n + threads_per_block - 1) / threads_per_block);
    let sharedmem = 0usize;
    let n_int = n as c_int;

    /* Simple way, but use deprecated API. */
    /*
    cuda_ffi::configure_call(griddim, blockdim, sharedmem).unwrap();
    unsafe {
            vectorAdd(d_a, d_b, d_c, n_int);
    }
    cuda_ffi::last_error().unwrap();
    */

    /* Alternative way. It doesn't contains deprecated API, but very complex. */
    cuda_ffi::launch(vectorAdd as *const c_void,
                     griddim,
                     blockdim,
                     &mut [&mut d_a as *mut *mut f32 as *mut c_void,
                           &mut d_b as *mut *mut f32 as *mut c_void,
                           &mut d_c as *mut *mut f32 as *mut c_void,
                           &n_int as *const c_int as *mut c_int as *mut c_void],
                     sharedmem)
        .unwrap();

    cuda_ffi::memcpy(h_c.as_mut_ptr(),
                     d_c,
                     n * 4,
                     cudaMemcpyKind::cudaMemcpyDeviceToHost)
        .unwrap();
    for i in 0..n {
        assert!((h_a[i] + h_b[i] - h_c[i]).abs() < 0.0001);
    }
    println!("OK");

    cuda_ffi::free(d_a).unwrap();
    cuda_ffi::free(d_b).unwrap();
    cuda_ffi::free(d_c).unwrap();
}
