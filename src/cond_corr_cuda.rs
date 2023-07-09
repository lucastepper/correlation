use ndarray as nd;
use numpy as np;
use pyo3::exceptions::{PyRuntimeError, PyValueError};
use pyo3::prelude::*;
use rustacuda::prelude::*;
use std::error::Error;
use std::ffi::CString;

fn into_pyresult<T, E: std::fmt::Debug>(result: Result<T, E>) -> PyResult<T> {
    match result {
        Ok(t) => Ok(t),
        Err(e) => Err(PyRuntimeError::new_err(format!("{:?}", e))),
    }
}

// TODO add the other condition modes
pub fn add_data<'py>(
    data1: Vec<&'py PyAny>,
    data2: Vec<&'py PyAny>,
    cond_data: Vec<&'py PyAny>,
    ncorr: usize,
    nbins: usize,
    start: f32,
    dx: f32,
) -> PyResult<(nd::Array2<f64>, nd::Array2<u64>)> {
    let mut cuda_obj = ConditionalCorrelationCuda::new(ncorr, nbins, start, dx).unwrap();
    for ((dat1, dat2), cond) in data1.iter().zip(data2.iter()).zip(cond_data.iter()) {
        // for some reason, casting from f64 to f32 array fails, check it is f32
        for to_check in [dat1, dat2, cond] {
            let dtype_elems = to_check.getattr("dtype")?.str()?;
            let dtype_elems = dtype_elems.to_str().unwrap_or("");
            if dtype_elems != "float32" {
                return Err(PyValueError::new_err("Dtype needs to be f32"));
            }
        }
        let dat1 = dat1.downcast::<np::PyArray1<f32>>()?;
        let dat1 = unsafe { dat1.as_array() };
        let dat2 = dat2.cast_as::<np::PyArray1<f32>>()?;
        let dat2 = unsafe { dat2.as_array() };
        let cond = cond.cast_as::<np::PyArray1<f32>>()?;
        let cond = unsafe { cond.as_array() };
        into_pyresult(cuda_obj.add_data(dat1, dat2, cond))?;
    }
    into_pyresult(cuda_obj.get_cond_data())
}

pub struct ConditionalCorrelationCuda {
    ncorr: usize,
    start: f32,
    dx: f32,
    nbins: usize,
    conditional_sums: DeviceBuffer<f32>,
    conditional_counts: DeviceBuffer<u64>,
    module: Module,
    stream: Stream,
    _cuda_context: Context,
}
impl ConditionalCorrelationCuda {
    pub fn new(
        ncorr: usize,
        nbins: usize,
        start: f32,
        dx: f32,
    ) -> PyResult<ConditionalCorrelationCuda> {
        into_pyresult(rustacuda::init(CudaFlags::empty()))?;
        let device = into_pyresult(Device::get_device(0))?;
        let _cuda_context = into_pyresult(Context::create_and_push(
            ContextFlags::MAP_HOST | ContextFlags::SCHED_AUTO,
            device,
        ))?;
        let module_data = into_pyresult(CString::new(include_str!("conditional_corr.ptx")))?;
        let module = into_pyresult(Module::load_from_string(&module_data))?;
        let stream = into_pyresult(Stream::new(StreamFlags::NON_BLOCKING, None))?;
        let conditional_sums = unsafe { into_pyresult(DeviceBuffer::zeroed(ncorr * nbins))? };
        let conditional_counts = unsafe { into_pyresult(DeviceBuffer::zeroed(ncorr * nbins))? };
        // init obj
        Ok(ConditionalCorrelationCuda {
            ncorr,
            start,
            dx,
            nbins,
            conditional_sums,
            conditional_counts,
            module,
            stream,
            _cuda_context,
        })
    }

    pub fn add_data(
        &mut self,
        datax: nd::ArrayView1<f32>,
        datay: nd::ArrayView1<f32>,
        cond: nd::ArrayView1<f32>,
    ) -> Result<(), Box<dyn Error>> {
        let module = &self.module;
        let stream = &self.stream;
        // Allocate space on the device and copy numbers to it.
        let threads_per_block = 256 as u32;
        let nblocks = (self.ncorr as u32 + (threads_per_block - 1)) / threads_per_block;
        let mut datax = DeviceBuffer::from_slice(
            datax
                .as_slice()
                .expect("Datax could not be transformed to slice"),
        )?;
        let mut datay = DeviceBuffer::from_slice(
            datay
                .as_slice()
                .expect("Datay could not be transformed to slice"),
        )?;
        let mut cond = DeviceBuffer::from_slice(
            cond.as_slice()
                .expect("cond could not be transformed to slice"),
        )?;
        unsafe {
            // Launch the `cond-corr` function with one block containing one thread on the given stream.
            launch!(module.cond_corr<<<nblocks, threads_per_block, 0, stream>>>(
                datax.as_device_ptr(),
                datay.as_device_ptr(),
                cond.as_device_ptr(),
                self.conditional_sums.as_device_ptr(),
                self.conditional_counts.as_device_ptr(),
                self.dx,
                self.start,
                self.nbins,
                self.ncorr,
                datax.len()
            ))?;
        }
        stream.synchronize()?;
        Ok(())
    }

    pub fn get_cond_data(&mut self) -> Result<(nd::Array2<f64>, nd::Array2<u64>), Box<dyn Error>> {
        let mut sum_host = vec![0.; self.ncorr * self.nbins];
        let mut count_host = vec![0; self.ncorr * self.nbins];
        self.conditional_sums.copy_to(&mut sum_host)?;
        self.conditional_counts.copy_to(&mut count_host)?;
        let sum_host = sum_host.into_iter().map(|x| x as f64).collect();
        let sum_host = nd::Array2::from_shape_vec((self.ncorr, self.nbins), sum_host)?;
        let count_host = nd::Array2::from_shape_vec((self.ncorr, self.nbins), count_host)?;
        Ok((sum_host, count_host))
    }
}
