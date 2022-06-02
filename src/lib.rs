#![allow(unused, dead_code)]
use indicatif::ProgressBar;
use ndarray as nd;
use ndarray::parallel::prelude::*;
use num_cpus;
use numpy as np;
use numpy::IntoPyArray;
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use rayon;

fn get_index(x: f64, x_min: f64, dx: f64, nbins: f64) -> usize {
    f64::min(f64::floor((x - x_min) / dx), nbins - 1.) as usize
}

#[pyclass]
struct ConditionalCorrelation {
    bins: Py<np::PyArray1<f64>>,
    ncorr: usize,
    _conditional_sums: nd::Array2<f64>,
    _conditional_counts: nd::Array2<u64>,
}
#[pymethods]
impl ConditionalCorrelation {
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

    fn get_results<'py>(&self, py: Python<'py>) -> PyResult<&'py np::PyArray2<f64>> {
        Ok(
            (self._conditional_counts.mapv(|c| 1. / c as f64) * &self._conditional_sums)
                .into_pyarray(py),
        )
    }
}

#[pymodule]
fn correlation(_py: Python<'_>, m: &PyModule) -> PyResult<()> {
    m.add_class::<ConditionalCorrelation>()?;
    Ok(())
}
