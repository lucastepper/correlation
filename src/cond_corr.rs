#[cfg(feature = "cuda")]
use super::cond_corr_cuda;
use super::utils;
use indicatif::ProgressBar;
use ndarray as nd;
use ndarray::parallel::prelude::*;
use np::PyArrayDyn;
use numpy as np;
use numpy::IntoPyArray;
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

fn get_index(x: f64, x_min: f64, dx: f64) -> f64 {
    f64::floor((x - x_min) / dx)
}

#[derive(Debug)]
enum Device {
    Cpu,
    Cuda,
}
impl Device {
    fn new(device: &str) -> PyResult<Device> {
        match device {
            "cpu" => Ok(Device::Cpu),
            "cuda" | "gpu" => Ok(Device::Cuda),
            _ => Err(PyValueError::new_err(format!("Unknown device: {}", device))),
        }
    }
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
pub struct ConditionalCorrelation {
    ncorr: usize,
    nbins: usize,
    start: f64,
    dx: f64,
    cond_sum: Py<PyArrayDyn<f64>>,
    cond_count: Py<PyArrayDyn<u64>>,
    corr_mode: CorrMode,
    device: Device,
}
#[pymethods]
impl ConditionalCorrelation {
    #[new]
    #[args(cond_start = false, cond_end = true, device = "\"cpu\"")]
    fn new(
        ncorr: usize,
        nbins: usize,
        range: [f64; 2],
        cond_start: bool,
        cond_end: bool,
        device: &str,
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
            nbins,
            start: range[0],
            dx,
            cond_sum,
            cond_count,
            corr_mode,
            device: Device::new(device)?,
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
        data1: &PyAny,
        data2: &PyAny,
        cond_data: &PyAny,
        nthreads: i32,
    ) -> PyResult<()> {
        // first cast to list of PyAny so that it works for single or multiple inputs
        let data1 = utils::convert_list_pyany_to_vec_pyany(data1)?;
        let data2 = utils::convert_list_pyany_to_vec_pyany(data2)?;
        let cond_data = utils::convert_list_pyany_to_vec_pyany(cond_data)?;
        if data1.len() != data2.len() || data2.len() != cond_data.len() {
            return Err(PyValueError::new_err("input data needs to be of same len"));
        }
        match self.device {
            Device::Cpu => {
                for i in 0..data1.len() {
                    let dat1 = data1[i].downcast::<np::PyArray1<f64>>()?;
                    let dat2 = data2[i].downcast::<np::PyArray1<f64>>()?;
                    let cond = cond_data[i].downcast::<np::PyArray1<f64>>()?;
                    self._add_data_cpu(py, dat1, dat2, cond, nthreads)?;
                }
            }
            #[cfg(feature = "cuda")]
            Device::Cuda => {
                // use cuda to compute the conditional sums and counts for the trajectories
                let (cond_sum, cond_count) = cond_corr_cuda::add_data(
                    data1,
                    data2,
                    cond_data,
                    self.ncorr,
                    self.nbins,
                    self.start as f32,
                    self.dx as f32,
                )?;
                // add results to state in self
                // TODO fix case were self.cond_* is 3d array
                unsafe {
                    let mut self_cond_sum = self
                        .cond_sum
                        .as_ref(py)
                        .reshape((self.nbins, self.ncorr))?
                        .as_array_mut();
                    let mut self_cond_count = self
                        .cond_count
                        .as_ref(py)
                        .reshape((self.nbins, self.ncorr))?
                        .as_array_mut();
                    self_cond_sum
                        .iter_mut()
                        .zip(cond_sum.iter())
                        .for_each(|(o, x)| *o += *x);
                    self_cond_count
                        .iter_mut()
                        .zip(cond_count.iter())
                        .for_each(|(o, x)| *o += *x);
                }
            }
            #[cfg(not(feature = "cuda"))]
            Device::Cuda => panic!(
                "Invalid device {:?}, build package without cuda support",
                self.device
            ),
        };
        Ok(())
    }

    #[args(nthreads = -1)]
    fn _add_data_cpu<'py>(
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
        let pool = utils::get_pool(nthreads)?;
        let nbins_f64 = self.nbins as f64;
        // make outputs
        let mut cond_sum = unsafe { self.cond_sum.as_ref(py).as_array_mut() };
        let mut cond_count = unsafe { self.cond_count.as_ref(py).as_array_mut() };
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
                                if idx >= 0. && idx < nbins_f64 {
                                    let idx = idx as usize;
                                    cond_sum[idx] += data1[i] * data2[i + tcorr];
                                    cond_count[idx] += 1;
                                }
                            }
                        }
                        CorrMode::End => {
                            for i in 0..(data1.len() - tcorr) {
                                let idx = get_index(cond_data[i + tcorr], self.start, self.dx);
                                if idx >= 0. && idx < nbins_f64 {
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
                                if idx1 >= 0. && idx1 < nbins_f64 {
                                    if idx2 >= 0. && idx2 < nbins_f64 {
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
