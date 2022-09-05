use conv::{ApproxFrom, ApproxScheme};
use ndarray as nd;
use ndarray_npy;
use ndarray_npy::{ReadNpyError, ReadNpyExt, ReadableElement};
use num_cpus;
use numpy as np;
use pyo3::exceptions::{PyFileNotFoundError, PyTypeError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::PyList;

// we define constants for all types we try to load .npy files and
// then cast into a given type
// these types are then also the TraitBounds for load_array
const F32: f32 = 0f32;
const F64: f64 = 0f64;
const I32: i32 = 0i32;
const I64: i64 = 0i64;

fn load_npy_from_to<T1: ReadableElement + Clone, T2: ApproxFrom<T1, C>, C: ApproxScheme>(
    path: &str,
    _in_type: T1,
    _out_type: T2,
) -> PyResult<nd::Array1<T2>> {
    // check if file exists
    let reader = match std::fs::File::open(path) {
        Ok(reader) => reader,
        Err(_e) => {
            return Err(PyFileNotFoundError::new_err(format!(
                "File not found: {path}",
            )));
        }
    };
    // try to load into T1 and convert to T2
    match nd::Array1::<T1>::read_npy(reader) {
        Ok(arr) => {
            Ok(arr.mapv(|x| T2::approx_from(x).expect("Conversion error while loading .npy")))
        }
        // if we get here, test if loading failed as the descriptor in
        //  .npy file does not match the type T1 provided.
        Err(e) => {
            let message = match e {
                ReadNpyError::WrongDescriptor(descr) => format!(
                    "Tried to load .npy but encountered unknown descriptor: {:?}.",
                    descr
                ),
                _ => format!("While loading .npy, unknown error: {} occured", e),
            };
            Err(PyValueError::new_err(message))
        }
    }
}

/// load .npy array from given path including type conversion
/// the .npy array can be loaded if it has dtype f64, f32, i64, i32
/// and is converted into the output type using the conv::ApproxFrom.
fn load_array<T: ApproxFrom<f32> + ApproxFrom<f64> + ApproxFrom<i32> + ApproxFrom<i64>>(
    path: &str,
    _out_type: T,
) -> PyResult<nd::Array1<T>> {
    let load_result = load_npy_from_to(path, F32, T::approx_from(0).unwrap());
    let load_result = match load_result {
        Ok(arr) => Ok(arr),
        Err(_e) => load_npy_from_to(path, F64, T::approx_from(0).unwrap()),
    };
    let load_result = match load_result {
        Ok(arr) => Ok(arr),
        Err(_e) => load_npy_from_to(path, I32, T::approx_from(0).unwrap()),
    };
    let load_result = match load_result {
        Ok(arr) => Ok(arr),
        Err(_e) => load_npy_from_to(path, I64, T::approx_from(0).unwrap()),
    };
    load_result
}

// TODO these still need some testing/fixing
/// Get ndarray from PyAny, where PyAny is cast as follows:
///     PyList -> memory copied into ndarray
///     PyArray -> ndarray created that points to the same memory as the PyArray
///     PyString -> npy file loaded into ndarray
#[allow(unused)]
pub fn extract_array_f32<'py>(to_extract: &'py PyAny) -> PyResult<nd::Array1<f32>> {
    let pytype = to_extract.get_type().name().unwrap();
    let array = match pytype {
        "ndarray" => {
            println!("extracting ndarray");
            let dtype_elems = to_extract.getattr("dtype")?.str()?;
            let array = match dtype_elems.to_str().unwrap_or("") {
                "float32" => to_extract.downcast::<np::PyArray1<f32>>()?,
                "float64" => {
                    let to_cast = to_extract.downcast::<np::PyArray1<f64>>()?;
                    to_cast.cast_as::<np::PyArray1<f32>>()?
                }
                _ => {
                    return Err(PyTypeError::new_err(format!(
                        "Could not cast array from {dtype_elems} to PyArray1<f32>"
                    )));
                }
            };
            unsafe { array.as_array().into_owned() }
        }
        "str" => {
            println!("Loading npy file");
            load_array(to_extract.extract::<&str>()?, F32)?
        }
        "list" => {
            println!("extracting list");
            nd::Array1::from_vec(to_extract.extract::<Vec<f32>>()?)
        }
        _ => {
            return Err(PyValueError::new_err(format!(
                "Cannot extract type: {} into ndarray",
                pytype
            )))
        }
    };
    Ok(array)
}

// TODO these still need some testing/fixing
/// Get ndarray from PyAny, where PyAny is cast as follows:
///     PyList -> memory copied into ndarray
///     PyArray -> ndarray created that points to the same memory as the PyArray
///     PyString -> npy file loaded into ndarray
#[allow(unused)]
pub fn extract_array_f64<'py>(to_extract: &'py PyAny) -> PyResult<nd::Array1<f64>> {
    let pytype = to_extract.get_type().name().unwrap();
    let array = match pytype {
        "ndarray" => {
            println!("extracting ndarray");
            let dtype_elems = to_extract.getattr("dtype")?.str()?;
            let array = match dtype_elems.to_str().unwrap_or("") {
                "float32" => {
                    let to_cast = to_extract.downcast::<np::PyArray1<f32>>()?;
                    to_cast.cast_as::<np::PyArray1<f64>>()?
                }
                "floatf64" => to_extract.downcast::<np::PyArray1<f64>>()?,
                _ => {
                    return Err(PyTypeError::new_err(format!(
                        "Could not cast array from {dtype_elems} to PyArray1<f64>"
                    )));
                }
            };
            unsafe { array.as_array().into_owned() }
        }
        "str" => {
            println!("Loading npy file");
            load_array(to_extract.extract::<&str>()?, F64)?
        }
        "list" => {
            println!("extracting list");
            nd::Array1::from_vec(to_extract.extract::<Vec<f64>>()?)
        }
        _ => {
            return Err(PyValueError::new_err(format!(
                "Cannot extract type: {} into ndarray",
                pytype
            )))
        }
    };
    Ok(array)
}

/// Convert a PyAny object to a vector of PyAnys, by eihter extracting list to Vec<PyAny> or
/// wrapping the PyAny in a Vec
pub fn convert_list_pyany_to_vec_pyany<'py>(pyany: &'py PyAny) -> PyResult<Vec<&'py PyAny>> {
    let pytype = pyany.get_type().name()?;
    let vec_pyany = match pytype {
        "list" => {
            let list = pyany.downcast::<PyList>()?;
            list.iter().collect()
        }
        _ => vec![pyany],
    };
    Ok(vec_pyany)
}

/// get a rayon pool for parallel computations. If nthreas == -1
/// get physical number of threads.
pub fn get_pool(nthreads: i32) -> PyResult<rayon::ThreadPool> {
    if nthreads < -1 || nthreads == 0 {
        return Err(PyValueError::new_err("nthreads must be -1 or positive"));
    }
    let nthreads = match nthreads {
        -1 => num_cpus::get(),
        _ => nthreads as usize,
    };
    let pool = rayon::ThreadPoolBuilder::new()
        .num_threads(nthreads)
        .build()
        .unwrap();
    Ok(pool)
}
