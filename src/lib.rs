#[cfg(feature = "cuda")]
#[macro_use]
extern crate rustacuda;
use pyo3::prelude::*;
mod cond_corr;
#[cfg(feature = "cuda")]
mod cond_corr_cuda;
mod utils;

#[pymodule]
fn correlation(_py: Python<'_>, m: &PyModule) -> PyResult<()> {
    m.add_class::<cond_corr::ConditionalCorrelation>()?;
    Ok(())
}
