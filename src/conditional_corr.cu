typedef unsigned long long int uint64_cu;

extern "C" __global__ void cond_corr(
    const float* x,
    const float* y,
    const float* cond,
    float* cond_sum,
    uint64_cu* cond_count,
    float dx,
    float start,
    int nbins,
    int ncorr,
    int len
) {
    int tcorr = blockIdx.x*blockDim.x + threadIdx.x;
    if (tcorr < ncorr) {
        // int idx = 0;
        for (int i = 0; i < (len - tcorr); i++) {
            int idx = (int)floor(((cond[i] - start) / dx));
            if (idx >= 0 && idx < nbins) {
                cond_sum[idx * ncorr + tcorr] += x[i] * y[i + tcorr];
                cond_count[idx * ncorr + tcorr] += 1;
            }
        }
    }
}

extern "C" __global__ void get_result(const float* cond_sum, const uint64_cu* cond_count, float* cond_corr, int len) {
    int i = blockIdx.x*blockDim.x + threadIdx.x;
    if (i < len) {
        cond_corr[i] = cond_sum[i] / (float)cond_count[i];
    }
}
