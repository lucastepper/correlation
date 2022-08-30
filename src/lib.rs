#[macro_use]
extern crate rustacuda;
// use indicatif::ProgressBar;
use ndarray as nd;
// use ndarray::parallel::prelude::*;
use ndarray_npy::read_npy;
// use np::PyArrayDyn;
// use num_cpus;
use numpy as np;
use numpy::IntoPyArray;
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
mod cond_corr_cuda;
mod cond_corr_cpu;


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
        let mut cuda_obj = cond_corr_cuda::ConditionalCorrelationCuda::new(ncorr, nbins, range).unwrap();
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
            let _res = cuda_obj.add_data(datax.view(), datay.view(), cond.view());
        }
        let cond_corr = cuda_obj
            .get_result()
            .expect("Something with the cuda calculation went wrong.");
        Ok(cond_corr.into_pyarray(py))
    }

    m.add_class::<cond_corr_cpu::ConditionalCorrelationCpu>()?;
    Ok(())
}
