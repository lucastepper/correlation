#![allow(unused, dead_code)]
use indicatif::ProgressBar;
use ndarray as nd;
use ndarray::parallel::prelude::*;
use ndarray_npy::read_npy;
use num_cpus;
use numpy as np;
use numpy::IntoPyArray;
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::{PyList, PyString};
use rayon;
use rustacuda::prelude::*;
use rustacuda::{launch, memory::DeviceBox};
use std::error::Error;
use std::ffi::CString;

// TODO return the actual index and then check if it is in range
// in cond_corr, such that values outside of range are ignored
fn get_index(x: f64, x_min: f64, dx: f64, nbins: f64) -> usize {
    f64::min(f64::floor((x - x_min) / dx), nbins - 1.) as usize
}

// TODO change this to use nbins and range, as it better transferes
// that bins need to be of the same size
#[pyclass]
struct ConditionalCorrelationCpu {
    bins: Py<np::PyArray1<f64>>,
    ncorr: usize,
    _conditional_sums: nd::Array2<f64>,
    _conditional_counts: nd::Array2<u64>,
}
#[pymethods]
impl ConditionalCorrelationCpu {
    #[new]
    fn new(py: Python<'_>, bins: Py<np::PyArray1<f64>>, ncorr: usize) -> PyResult<Self> {
        // bins should be the edges of the bins, so it has shape nbins + 1
        let len_bins = unsafe { bins.as_ref(py).as_array().len() };
        if len_bins < 2 {
            return Err(PyValueError::new_err("bins must not be empty"));
        }
        let nbins = len_bins - 1;
        let _conditional_sums = nd::Array2::zeros((nbins, ncorr));
        let _conditional_counts = nd::Array2::zeros((nbins, ncorr));
        Ok(Self {
            bins,
            ncorr,
            _conditional_sums,
            _conditional_counts,
        })
    }

    #[getter]
    fn get_bins(&self) -> PyResult<&Py<np::PyArray1<f64>>> {
        Ok(&self.bins)
    }

    #[setter]
    fn set_bins(&mut self, bins: Py<np::PyArray1<f64>>) -> PyResult<()> {
        self.bins = bins;
        Ok(())
    }

    #[args(nthreads = -1)]
    fn add_data<'py>(
        &mut self,
        py: Python<'py>,
        corr_data1: &'py np::PyArray1<f64>,
        corr_data2: &'py np::PyArray1<f64>,
        cond_data: &'py np::PyArray1<f64>,
        nthreads: i32,
    ) -> PyResult<()> {
        let cond_data = unsafe { cond_data.as_array() };
        let corr_data1 = unsafe { corr_data1.as_array() };
        let corr_data2 = unsafe { corr_data2.as_array() };
        let bins = unsafe { self.bins.as_ref(py).as_array() };
        if corr_data1.len() != corr_data2.len() {
            return Err(PyErr::new::<PyValueError, _>(
                "corr_data1 and corr_data2 must be the same length",
            ));
        }
        if cond_data.len() != corr_data1.len() {
            return Err(PyErr::new::<PyValueError, _>(
                "cond_data and corr_data must be the same length",
            ));
        }
        let nbins = (bins.len() - 1) as f64;
        let start = bins[0];
        let dx: f64 = (&bins.slice(nd::s![1..]) - &bins.slice(nd::s![..-1])).sum() / nbins;
        let bar = ProgressBar::new(self.ncorr as u64);
        if nthreads < -1 || nthreads == 0 {
            return Err(PyValueError::new_err("nthreads must be -1 or positive"));
        }
        let nthreads = match nthreads {
            -1 => num_cpus::get(),
            _ => nthreads as usize,
        };
        rayon::ThreadPoolBuilder::new()
            .num_threads(nthreads)
            .build_global();
        self._conditional_sums
            .axis_iter_mut(nd::Axis(1))
            .into_par_iter()
            .zip(
                self._conditional_counts
                    .axis_iter_mut(nd::Axis(1))
                    .into_par_iter(),
            )
            .enumerate()
            .for_each(|(tcorr, (mut cond_sum, mut cond_count))| {
                for i in 0..(corr_data1.len() - tcorr) {
                    let index = get_index(cond_data[i], start, dx, nbins);
                    cond_sum[index] += corr_data1[i] * corr_data2[i + tcorr];
                    cond_count[index] += 1;
                }
                bar.inc(1);
            });
        bar.finish();
        Ok(())
    }

    fn get_result<'py>(&self, py: Python<'py>) -> PyResult<&'py np::PyArray2<f64>> {
        Ok(
            (self._conditional_counts.mapv(|c| 1. / c as f64) * &self._conditional_sums)
                .into_pyarray(py),
        )
    }
}

struct ConditionalCorrelationCuda {
    ncorr: usize,
    start: f32,
    dx: f32,
    nbins: usize,
    conditional_sums: DeviceBuffer<f32>,
    conditional_counts: DeviceBuffer<f32>,
    module: Module,
    stream: Stream,
    _cuda_context: Context,
}
impl ConditionalCorrelationCuda {
    fn new(
        ncorr: usize,
        nbins: usize,
        range: [f32; 2],
    ) -> Result<ConditionalCorrelationCuda, Box<dyn Error>> {
        rustacuda::init(CudaFlags::empty())?;
        let device = Device::get_device(0)?;
        let _cuda_context =
            Context::create_and_push(ContextFlags::MAP_HOST | ContextFlags::SCHED_AUTO, device)?;
        let module_data = CString::new(include_str!("conditional_corr.ptx"))?;
        let module = Module::load_from_string(&module_data)?;
        let stream = Stream::new(StreamFlags::NON_BLOCKING, None)?;
        let dx = (range[1] - range[0]) / (nbins as f32);
        // let conditional_sums = unsafe{ DeviceBuffer::zeroed(nbins * tcorr)? };
        let conditional_sums = unsafe { DeviceBuffer::zeroed(nbins * ncorr)? };
        let conditional_counts = unsafe { DeviceBuffer::zeroed(nbins * ncorr)? };
        // init obj
        Ok(ConditionalCorrelationCuda {
            ncorr,
            start: range[0],
            dx,
            nbins,
            conditional_sums,
            conditional_counts,
            module,
            stream,
            _cuda_context,
        })
    }

    fn add_data(
        &mut self,
        datax: nd::ArrayView1<f32>,
        datay: nd::ArrayView1<f32>,
        cond: nd::ArrayView1<f32>,
    ) -> Result<(), Box<dyn Error>> {
        let module = &self.module;
        let stream = &self.stream;
        // Allocate space on the device and copy numbers to it.
        let threads_per_block = 256;
        let nblocks = (datax.len() + (threads_per_block - 1)) / threads_per_block;
        println!(
            "Nblocks: {}, Nthreads per block: {}",
            nblocks, threads_per_block
        );
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
        let mut result = DeviceBuffer::from_slice(&vec![0.0f32; datax.len()])?;
        println!("Nbins: {}", self.nbins);
        unsafe {
            // Launch the `cond-corr` function with one block containing one thread on the given stream.
            launch!(module.cond_corr<<<1, 20, 0, stream>>>(
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

    fn get_result(&mut self) -> Result<nd::Array2<f32>, Box<dyn Error>> {
        let stream = &self.stream;
        let module = &self.module;
        let result = unsafe {
            let mut result: DeviceBuffer<f32> = DeviceBuffer::zeroed(self.nbins * self.ncorr)?;
            let nthreads = (self.nbins * self.ncorr) as u32;
            let threads_per_block: u32 = 256;
            let nblocks = (nthreads + (threads_per_block - 1)) / threads_per_block;
            launch!(module.get_result<<<nblocks, threads_per_block, 0, stream>>>(
                self.conditional_sums.as_device_ptr(),
                self.conditional_counts.as_device_ptr(),
                result.as_device_ptr(),
                self.nbins * self.ncorr
            ))?;
            result
        };
        stream.synchronize()?;
        let mut result_host = vec![0.; self.nbins * self.ncorr];
        result.copy_to(&mut result_host)?;
        Ok(nd::Array2::from_shape_vec(
            (self.nbins, self.ncorr),
            result_host,
        )?)
    }
}

#[pymodule]
fn correlation(_py: Python<'_>, m: &PyModule) -> PyResult<()> {
    #[pyfn(m)]
    #[pyo3(name = "conditional_correlation_cuda")]
    fn conditional_correlation_cuda<'py>(
        py: Python<'py>,
        data1: Vec<&str>,
        data2: Vec<&str>,
        cond: Vec<&str>,
        ncorr: usize,
        nbins: usize,
        range: [f32; 2],
    ) -> PyResult<&'py np::PyArray2<f32>> {
        if nbins < 2 {
            return Err(PyValueError::new_err("nbins must larger than 1"));
        }
        let mut cond_corr_cuda_obj = ConditionalCorrelationCuda::new(ncorr, nbins, range).unwrap();
        for ((file1, file2), file_cond) in data1.iter().zip(data2.iter()).zip(cond.iter()) {
            // let datax: nd::Array1<f32> = read_npy(file1).unwrap();
            let datax: nd::Array1<f32> = match read_npy(file1) {
                Ok(arr) => arr,
                Err(e) => {
                    return Err(PyErr::new::<PyValueError, _>(format!(
                        "Tried to load from: '{}', Error: {}",
                        file1, e
                    )))
                }
            };
            let datay: nd::Array1<f32> = match read_npy(file2) {
                Ok(arr) => arr,
                Err(e) => {
                    return Err(PyErr::new::<PyValueError, _>(format!(
                        "Tried to load from: '{}', Error: {}",
                        file2, e
                    )))
                }
            };
            let cond: nd::Array1<f32> = match read_npy(file_cond) {
                Ok(arr) => arr,
                Err(e) => {
                    return Err(PyErr::new::<PyValueError, _>(format!(
                        "Tried to load from: '{}', Error: {}",
                        file_cond, e
                    )))
                }
            };
            cond_corr_cuda_obj.add_data(datax.view(), datay.view(), cond.view());
        }
        let cond_corr = cond_corr_cuda_obj
            .get_result()
            .expect("Something with the cuda calculation went wrong.");
        Ok(unsafe { cond_corr.into_pyarray(py) })
    }

    m.add_class::<ConditionalCorrelationCpu>()?;
    Ok(())
}

#[cfg(test)]
mod test {
    use super::*;

    fn index(x: f32, start: f32, dx: f32) -> f32 {
        f32::floor((x - start) / dx)
    }

    fn cond_ref(
        data1: nd::ArrayView1<f32>,
        data2: nd::ArrayView1<f32>,
        cond: nd::ArrayView1<f32>,
        nbins: usize,
        ncorr: usize,
        start: f32,
        dx: f32,
    ) -> (nd::Array2<f32>, nd::Array2<f32>) {
        let mut cond_count = nd::Array2::zeros((nbins, ncorr));
        let mut cond_sum = nd::Array2::zeros((nbins, ncorr));
        for tcorr in (0..ncorr) {
            for i in (0..data1.len() - tcorr) {
                let idx = index(cond[i], start, dx);
                if idx >= 0. && idx < nbins as f32 {
                    cond_count[[idx as usize, tcorr]] += 1.;
                    cond_sum[[idx as usize, tcorr]] += data1[i] * data2[i + tcorr];
                }
            }
        }
        (cond_sum, cond_count)
    }

    struct TestData {
        data1: nd::Array1<f32>,
        data2: nd::Array1<f32>,
        cond: nd::Array1<f32>,
    }
    impl TestData {
        fn new() -> TestData {
            TestData {
                data1: read_npy("/tmp/data2.npy").unwrap(),
                data2: read_npy("/tmp/data2.npy").unwrap(),
                cond: read_npy("/tmp/cond.npy").unwrap(),
            }
        }
    }

    #[test]
    fn test_conditional_correlation_cuda_verbose() {
        let ncorr = 10;
        let nbins = 2;
        let range = [-1.5, 1.5];
        let dx = (range[1] - range[0]) / (nbins as f32);
        let start = range[0];
        let test_data = TestData::new();
        let (cond_sum_ref, cond_count_ref) = cond_ref(
            test_data.data1.view(),
            test_data.data2.view(),
            test_data.cond.view(),
            nbins,
            ncorr,
            start,
            dx,
        );
        let mut cond_corr_obj_cuda = ConditionalCorrelationCuda::new(ncorr, nbins, range).unwrap();
        cond_corr_obj_cuda.add_data(
            test_data.data1.view(),
            test_data.data2.view(),
            test_data.cond.view(),
        );
        let mut corr_test = cond_corr_obj_cuda.get_result().unwrap();
        let mut cond_count_test = vec![0.; ncorr * nbins];
        cond_corr_obj_cuda
            .conditional_counts
            .copy_to(&mut cond_count_test)
            .unwrap();
        let cond_count_test = nd::Array2::from_shape_vec((nbins, ncorr), cond_count_test).unwrap();
        let mut cond_sum_test = vec![0.; ncorr * nbins];
        cond_corr_obj_cuda
            .conditional_sums
            .copy_to(&mut cond_sum_test)
            .unwrap();
        let cond_sum_test = nd::Array2::from_shape_vec((nbins, ncorr), cond_sum_test).unwrap();
        // test if counts are the same
        for (i, (ref_sample, test_sample)) in cond_count_ref
            .iter()
            .zip(cond_count_test.iter())
            .enumerate()
        {
            if *test_sample != *test_sample {
                println!("\n\tThe ith {} sample is NAN\n", i);
                panic!();
            }
            if (*ref_sample - *test_sample).abs() > 1e-6 {
                println!(
                    "\n\tThe ith {} sample is WRONG: ref: {} test {}\n",
                    i, ref_sample, test_sample
                );
                panic!();
            }
        }
        // test if sums are the same
        for (i, (ref_sample, test_sample)) in
            cond_sum_ref.iter().zip(cond_sum_test.iter()).enumerate()
        {
            if *test_sample != *test_sample {
                println!("\n\tThe ith {} sample is NAN\n", i);
                panic!();
            }
            if (*ref_sample - *test_sample).abs() > 1e-6 {
                println!(
                    "\n\tThe ith {} sample is WRONG: ref: {} test {}\n",
                    i, ref_sample, test_sample
                );
                panic!();
            }
        }
        // // printing for debugging reasons
        // println!("ref sum: {:?}", cond_sum_ref);
        // println!("test sum: {:?}", cond_sum_test);
        // println!("ref count: {:?}", cond_count_ref);
        // println!("ref cond_corr: {:?}", cond_sum_ref / cond_count_ref);
        // panic!();
    }
}
