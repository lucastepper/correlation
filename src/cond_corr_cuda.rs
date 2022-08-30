use ndarray as nd;
use std::error::Error;
use std::ffi::CString;
use rustacuda::prelude::*;


pub struct ConditionalCorrelationCuda {
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
    pub fn new(
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

    pub fn get_result(&mut self) -> Result<nd::Array2<f32>, Box<dyn Error>> {
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


#[cfg(test)]
mod test {
    use super::*;
    use ndarray_npy::read_npy;

    fn index(x: f32, start: f32, dx: f32) -> f32 {
        f32::floor((x - start) / dx)
    }

    /// Simple reference implementation of the conditional correlation algorithm.
    fn cond_ref(
        data1: nd::ArrayView1<f32>,
        data2: nd::ArrayView1<f32>,
        cond: nd::ArrayView1<f32>,
        nbins: usize,
        ncorr: usize,
        start: f32,
        dx: f32,
        cond_start: bool,
        cond_end: bool,
    ) -> (nd::Array2<f32>, nd::Array2<f32>) {
        // cond both not implemented for this ref func
        assert!(cond_start != cond_end);
        let mut cond_count = nd::Array2::zeros((nbins, ncorr));
        let mut cond_sum = nd::Array2::zeros((nbins, ncorr));
        for tcorr in 0..ncorr {
            for i in 0..data1.len() - tcorr {
                // get the index for condition data either at t=0 or t=\tau
                let idx = match cond_start {
                    true => index(cond[i], start, dx),
                    false => index(cond[i + tcorr], start, dx),
                };
                // check if index is in range of bins and update corr data
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
            true,
            false,
        );
        let mut cond_corr_obj_cuda = ConditionalCorrelationCuda::new(ncorr, nbins, range).unwrap();
        let _res = cond_corr_obj_cuda.add_data(
            test_data.data1.view(),
            test_data.data2.view(),
            test_data.cond.view(),
        );
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
    }
}
