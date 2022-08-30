use indicatif::ProgressBar;
use ndarray as nd;
use ndarray::parallel::prelude::*;
use np::PyArrayDyn;
use num_cpus;
use numpy as np;
use numpy::IntoPyArray;
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;


fn get_index(x: f64, x_min: f64, dx: f64) -> f64 {
    f64::floor((x - x_min) / dx)
}

enum CorrMode {
    Start,
    End,
    Both,
}
impl CorrMode {
    fn new(cond_start: bool, cond_end: bool) -> PyResult<CorrMode> {
        if cond_start == true && cond_end == true {
            return Ok(CorrMode::Both);
        }
        if cond_start == true && cond_end == false {
            return Ok(CorrMode::Start);
        }
        if cond_start == false && cond_end == true {
            return Ok(CorrMode::End);
        }
        Err(PyValueError::new_err(
            "Either corr_start or corr_end needs to be True.",
        ))
    }
}

#[pyclass]
pub struct ConditionalCorrelationCpu {
    ncorr: usize,
    start: f64,
    dx: f64,
    nbins: f64,
    cond_sum: Py<PyArrayDyn<f64>>,
    cond_count: Py<PyArrayDyn<u64>>,
    corr_mode: CorrMode,
}
#[pymethods]
impl ConditionalCorrelationCpu {
    #[new]
    #[args(cond_start = false, cond_end = true)]
    fn new(
        ncorr: usize,
        nbins: usize,
        range: [f64; 2],
        cond_start: bool,
        cond_end: bool,
    ) -> PyResult<Self> {
        if nbins < 2 {
            return Err(PyValueError::new_err("bins must not be empty"));
        }
        let dx = (range[1] - range[0]) / (nbins as f64);
        let corr_mode = CorrMode::new(cond_start, cond_end)?;
        let dims = match corr_mode {
            CorrMode::Both => vec![nbins, nbins, ncorr],
            _ => vec![nbins, ncorr],
        };
        let (cond_sum, cond_count) = pyo3::Python::with_gil(|py| {
            let cond_sum: Py<np::PyArrayDyn<f64>> =
                np::PyArray::zeros(py, dims.clone(), false).to_dyn().into();
            let cond_count: Py<np::PyArrayDyn<u64>> =
                np::PyArray::zeros(py, dims, false).to_dyn().into();
            (cond_sum, cond_count)
        });
        Ok(Self {
            ncorr,
            start: range[0],
            dx,
            nbins: nbins as f64,
            cond_sum,
            cond_count,
            corr_mode,
        })
    }

    #[getter]
    fn get_nbins(&self) -> PyResult<usize> {
        Ok(self.nbins as usize)
    }

    #[getter]
    fn get_ncorr(&self) -> PyResult<usize> {
        Ok(self.ncorr)
    }

    #[getter]
    fn get_dx(&self) -> PyResult<f64> {
        Ok(self.dx)
    }

    #[getter]
    fn get_start(&self) -> PyResult<f64> {
        Ok(self.start)
    }

    #[getter]
    fn get_cond_sum<'py>(&'py self, py: Python<'py>) -> PyResult<&'py PyArrayDyn<f64>> {
        Ok(self.cond_sum.as_ref(py))
    }

    #[getter]
    fn get_cond_count<'py>(&'py self, py: Python<'py>) -> PyResult<&'py PyArrayDyn<u64>> {
        Ok(self.cond_count.as_ref(py))
    }

    #[args(nthreads = -1)]
    fn add_data<'py>(
        &mut self,
        py: Python<'py>,
        data1: &'py np::PyArray1<f64>,
        data2: &'py np::PyArray1<f64>,
        cond_data: &'py np::PyArray1<f64>,
        nthreads: i32,
    ) -> PyResult<()> {
        // check input data
        let data1 = unsafe { data1.as_array() };
        let data2 = unsafe { data2.as_array() };
        let cond_data = unsafe { cond_data.as_array() };
        if data1.len() != data2.len() {
            return Err(PyErr::new::<PyValueError, _>(
                "data1 and data2 must be the same length",
            ));
        }
        if cond_data.len() != data1.len() {
            return Err(PyErr::new::<PyValueError, _>(
                "cond_data and data must be the same length",
            ));
        }
        let bar = ProgressBar::new(self.ncorr as u64);
        if nthreads < -1 || nthreads == 0 {
            return Err(PyValueError::new_err("nthreads must be -1 or positive"));
        }
        let nthreads = match nthreads {
            -1 => num_cpus::get(),
            _ => nthreads as usize,
        };
        // make outputs
        let mut cond_sum = unsafe { self.cond_sum.as_ref(py).as_array_mut() };
        let mut cond_count = unsafe { self.cond_count.as_ref(py).as_array_mut() };
        let pool = rayon::ThreadPoolBuilder::new()
            .num_threads(nthreads)
            .build()
            .unwrap();
        // we have three cases, conditioning on the first time, the last time or both.
        // iterate along last dimension, which is the correlation time
        pool.install(|| {
            cond_sum
                .axis_iter_mut(nd::Axis(cond_sum.ndim() - 1))
                .into_par_iter()
                .zip(
                    cond_count
                        .axis_iter_mut(nd::Axis(cond_count.ndim() - 1))
                        .into_par_iter(),
                )
                .enumerate()
                .for_each(|(tcorr, (mut cond_sum, mut cond_count))| {
                    match self.corr_mode {
                        CorrMode::Start => {
                            for i in 0..(data1.len() - tcorr) {
                                let idx = get_index(cond_data[i], self.start, self.dx);
                                if idx >= 0. && idx < self.nbins {
                                    let idx = idx as usize;
                                    cond_sum[idx] += data1[i] * data2[i + tcorr];
                                    cond_count[idx] += 1;
                                }
                            }
                        }
                        CorrMode::End => {
                            for i in 0..(data1.len() - tcorr) {
                                let idx = get_index(cond_data[i + tcorr], self.start, self.dx);
                                if idx >= 0. && idx < self.nbins {
                                    let idx = idx as usize;
                                    cond_sum[idx] += data1[i] * data2[i + tcorr];
                                    cond_count[idx] += 1;
                                }
                            }
                        }
                        _ => {
                            for i in 0..(data1.len() - tcorr) {
                                let idx1 = get_index(cond_data[i], self.start, self.dx);
                                let idx2 = get_index(cond_data[i + tcorr], self.start, self.dx);
                                if idx1 >= 0. && idx1 < self.nbins {
                                    if idx2 >= 0. && idx2 < self.nbins {
                                        let idx1 = idx1 as usize;
                                        let idx2 = idx2 as usize;
                                        cond_sum[[idx1, idx2]] += data1[i] * data2[i + tcorr];
                                        cond_count[[idx1, idx2]] += 1;
                                    }
                                }
                            }
                        }
                    }
                    bar.inc(1);
                });
        });
        bar.finish();
        Ok(())
    }

    fn get_result<'py>(&self, py: Python<'py>) -> PyResult<&'py np::PyArrayDyn<f64>> {
        let cond_sum = unsafe { self.cond_sum.as_ref(py).as_array() };
        let cond_count = unsafe { self.cond_count.as_ref(py).as_array() };
        Ok((cond_count.mapv(|c| 1. / c as f64) * &cond_sum).into_pyarray(py))
    }
}
