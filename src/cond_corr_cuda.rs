use cudarc::driver::{
    result, sys, CudaContext, CudaFunction, CudaSlice, CudaStream, DriverError, LaunchConfig,
    PushKernelArg,
};
use cudarc::nvrtc::Ptx;
use numpy::PyArray1;
use pyo3::exceptions::{PyRuntimeError, PyValueError};
use pyo3::prelude::*;
use std::sync::Arc;

fn cuda_error(error: DriverError) -> PyErr {
    PyRuntimeError::new_err(format!("CUDA conditional correlation failed: {error}"))
}

pub struct CudaKernel {
    resources: Option<KernelResources>,
}

struct KernelResources {
    function: CudaFunction,
    stream: Arc<CudaStream>,
}

struct CurrentContext(sys::CUcontext);

impl CurrentContext {
    fn save() -> Result<Self, DriverError> {
        Ok(Self(
            result::ctx::get_current()?.unwrap_or(std::ptr::null_mut()),
        ))
    }
}

impl Drop for CurrentContext {
    fn drop(&mut self) {
        // cudarc binds its context for driver calls. Restore the caller's context.
        let _ = unsafe { result::ctx::set_current(self.0) };
    }
}

impl Drop for CudaKernel {
    fn drop(&mut self) {
        let _previous = CurrentContext::save().ok();
        drop(self.resources.take());
    }
}

impl CudaKernel {
    pub fn new() -> PyResult<Self> {
        // cudarc's driver shims expect a loadable library before calling CUDA.
        if !unsafe { sys::is_culib_present() } {
            return Err(PyRuntimeError::new_err(
                "NVIDIA CUDA driver library is unavailable",
            ));
        }
        result::init().map_err(cuda_error)?;
        let _previous = CurrentContext::save().map_err(cuda_error)?;
        let context = CudaContext::new(0).map_err(cuda_error)?;
        let module = context
            .load_module(Ptx::from_src(include_str!("conditional_corr.ptx")))
            .map_err(cuda_error)?;
        let function = module.load_function("cond_corr").map_err(cuda_error)?;
        let stream = context.new_stream().map_err(cuda_error)?;
        Ok(Self {
            resources: Some(KernelResources { function, stream }),
        })
    }

    pub fn add_data(
        &self,
        trajectories: Vec<[&PyArray1<f32>; 3]>,
        ncorr: usize,
        nbins: usize,
        start: f32,
        dx: f32,
    ) -> PyResult<(Vec<f32>, Vec<u64>)> {
        let output_len = ncorr
            .checked_mul(nbins)
            .filter(|&len| len <= i32::MAX as usize && nbins <= i32::MAX as usize)
            .ok_or_else(|| {
                PyValueError::new_err("CUDA conditional correlation output is too large")
            })?;
        if output_len == 0 || trajectories.is_empty() {
            return Ok((Vec::new(), Vec::new()));
        }

        let _previous = CurrentContext::save().map_err(cuda_error)?;
        let resources = self.resources.as_ref().expect("CUDA resources are live");
        let stream = &resources.stream;
        let ncorr = ncorr as i32;
        let nbins = nbins as i32;
        let mut sums = stream.alloc_zeros::<f32>(output_len).map_err(cuda_error)?;
        let mut counts = stream.alloc_zeros::<u64>(output_len).map_err(cuda_error)?;
        let threads = 256;
        let config = LaunchConfig {
            grid_dim: ((ncorr as u32).div_ceil(threads), 1, 1),
            block_dim: (threads, 1, 1),
            shared_mem_bytes: 0,
        };
        for [a, b, cond] in trajectories {
            let len = i32::try_from(a.len()).map_err(|_| {
                PyValueError::new_err("CUDA conditional correlation input is too long")
            })?;
            if len == 0 {
                continue;
            }
            let a = copy_to_device(stream, a).map_err(cuda_error)?;
            let b = copy_to_device(stream, b).map_err(cuda_error)?;
            let cond = copy_to_device(stream, cond).map_err(cuda_error)?;

            // The kernel takes three equal-length inputs and writes only within the
            // two output buffers of ncorr * nbins elements.
            unsafe {
                stream
                    .launch_builder(&resources.function)
                    .arg(&a)
                    .arg(&b)
                    .arg(&cond)
                    .arg(&mut sums)
                    .arg(&mut counts)
                    .arg(&dx)
                    .arg(&start)
                    .arg(&nbins)
                    .arg(&ncorr)
                    .arg(&len)
                    .launch(config)
                    .map_err(cuda_error)?;
            }
        }

        let sums = stream.clone_dtoh(&sums).map_err(cuda_error)?;
        let counts = stream.clone_dtoh(&counts).map_err(cuda_error)?;
        stream.synchronize().map_err(cuda_error)?;
        Ok((sums, counts))
    }
}

fn copy_to_device(
    stream: &Arc<CudaStream>,
    input: &PyArray1<f32>,
) -> Result<CudaSlice<f32>, DriverError> {
    let input = input.readonly();
    match input.as_slice() {
        Ok(slice) => stream.clone_htod(slice),
        Err(_) => {
            let contiguous: Vec<_> = input.as_array().iter().copied().collect();
            stream.clone_htod(&contiguous)
        }
    }
}
