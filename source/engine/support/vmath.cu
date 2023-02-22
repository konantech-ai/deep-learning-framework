#include <stdio.h>
#include <stdlib.h>

#include <cuda.h>
#include <mma.h>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <cmath>

#include "../support/vmath.h"
//#include "../cores/vmath_core.h"
#include "../utils/vexception.h"

// hs.cho
#include <cfloat>
#define PI_L    3.1415926535897931L
#define PI_F    3.141592654f

#ifdef _DEBUG
//#define DEBUG_HOST_MEM
#endif

#ifdef DEBUG_NO_RANDOM
#define NO_RANDOM_HOST
#define NO_RANDOM_CUDA
#endif


static bool debug_supress = false;
static int debug_border = -1;

//--------------------------------------------------------------------------------------------------

#define __static__ static

/*
#define CUDA_CALL(line, funcname, size, args) { \
    unsigned int nthreads = (unsigned int)((size + ms_block_size - 1) / ms_block_size); \
    if (debug_supress && line < debug_border) return; \
    funcname << <nthreads, ms_block_size >> > args; \
    cudaCheck(cudaGetLastError(), #funcname); \
}
*/

#define CUDA_CALL(funcname, device, size, ...) \
	if (device < 0) { \
        funcname##_host(  __VA_ARGS__); \
    } \
	else { \
        unsigned int nthreads = (unsigned int)((size + ms_block_size - 1) / ms_block_size); \
        if (debug_supress && __LINE__ < debug_border) return; \
        funcname##_cuda << <nthreads, ms_block_size >> > (size, ##__VA_ARGS__); \
        cudaCheck(cudaGetLastError(), #funcname, __FILE__, __LINE__); \
    }

//--------------------------------------------------------------------------------------------------

static int ms_block_size = 1024;

static std::random_device ms_rd{};
static std::mt19937 ms_randGen{ ms_rd() };

void VMath::cudaCheck(cudaError_t cuda_ret, string name, string file, int line) {
    static int nth = 0;

    if (0) {
        printf("cudaCheck[%d] => %s()\n", nth, name.c_str());
    }

    nth++;

    static size_t free, total;

    if (cuda_ret != 0) {
        int nDevice = -2;
        cudaMemGetInfo(&free, &total);
        cudaGetDevice(&nDevice);
        string sCudaError = cudaGetErrorString(cuda_ret);
        printf("[Temporal Cuda Error] device = %d, %s (free: %lld, total:%lld) called in %s:%d\n", nDevice, sCudaError.c_str(), free, total, file.c_str(), line);
        VP_THROW1(VERR_CUDA_ERROR, "cuda:"+sCudaError);
    }

    if (0) {
        cudaMemGetInfo(&free, &total);
    }
}

#ifdef NO_RANDOM_HOST
static long ms_no_random_seed = 1234;
static long ms_no_random_coin = ms_no_random_seed;

static float ms_no_random_normal(float mean, float std) {
    ms_no_random_coin = ((((ms_no_random_coin * 214013L + 2531011L) >> 16) & 0x7fff)); // Dewdney Algorithm
    double temp = (double)(int)ms_no_random_coin / 32768.0 + 0.0000001;
    ms_no_random_coin = ((((ms_no_random_coin * 214013L + 2531011L) >> 16) & 0x7fff));
    double temp2 = (double)(int)ms_no_random_coin / 32768.0;
    float temp3 = (float)(::sqrt(-2.0 * ::log(temp)) * cos(2.0 * 3.141592 * temp2)) * std + mean; //Box-Muller Transform
    return temp3;
}

template<typename T> static T ms_no_random_uniform(T start, T end) {
    //ms_no_random_coin = (ms_no_random_coin << 7) % 65521;
    ms_no_random_coin = ((((ms_no_random_coin * 214013L + 2531011L) >> 16) & 0x7fff) % 32768);
    float temp = (float)(int)ms_no_random_coin / 32768.0f;
    float temp2 = temp * ((float)end - (float)start) + (float)start;
    if (typeid(T) == typeid(int)) {
        temp2 += 0.4999f;
        temp2 = (float)floor(temp2);
    }
    return (T)temp2;
}
#endif

//--------------------------------------------------------------------------------------------------
//

#ifdef NO_RANDOM_CUDA
__device__ float dev_no_random_normal(int64 nth, float mean, float std) {
    int ms_no_random_coin = (int)nth;
    ms_no_random_coin = ((((ms_no_random_coin * 214013L + 2531011L) >> 16) & 0x7fff)); // Dewdney Algorithm
    double temp = (double)(int)ms_no_random_coin / 32768.0 + 0.0000001;
    ms_no_random_coin = ((((ms_no_random_coin * 214013L + 2531011L) >> 16) & 0x7fff));
    double temp2 = (double)(int)ms_no_random_coin / 32768.0;
    float temp3 = (float)(::sqrt(-2.0 * ::log(temp)) * cos(2.0 * 3.141592 * temp2)) * std + mean; //Box-Muller Transform
    return temp3;
}

__device__ float dev_no_random_uniform(int64 nth, float start, float end) {
    int ms_no_random_coin = (int)nth;
    //ms_no_random_coin = (ms_no_random_coin << 7) % 65521;
    ms_no_random_coin = ((((ms_no_random_coin * 214013L + 2531011L) >> 16) & 0x7fff) % 32768);
    float temp = (float)(int)ms_no_random_coin / 32768.0f;
    float temp2 = temp * ((float)end - (float)start) + (float)start;
    return temp2;
}
#endif

//--------------------------------------------------------------------------------------------------
//
#ifdef DEBUG_HOST_MEM
static int64 ms_allocHostTotal = 0;
static int64 ms_freeHostTotal = 0;
static map<void*, int64> ms_allocHostSize;
#endif

void* mem_alloc_host(int64 size) {
    void* ptr = malloc(size);
    if (ptr == NULL) VP_THROW(VERR_HOSTMEM_ALLOC_FAILURE);

#ifdef DEBUG_HOST_MEM
    ms_allocHostTotal += size;
    if (ms_allocHostSize.find(ptr) != ms_allocHostSize.end()) {
        VP_THROW(VERR_INVALID_SESSION_HANDLE);
    }
    ms_allocHostSize[ptr] = size;
#endif
    return ptr;
}

#ifdef CUDA_LEAK_CHECK
mutex malloc_mutex;
int alloc_nth = 0;
int64 alloc_acc = 0;
int free_nth = 0;
map<void*, int64> ptrs;
#endif

void* mem_alloc_cuda(int device, int64 size) {
    int nOldDevice;
    void* ptr = NULL;

    if (size % 16 != 0) {
        size = size + 16 - size % 16;
    }

    if (0) {
        static int64 acc_size = 0;
        acc_size += size;
        printf("cudaMalloc(%lld) => acc: %lld bytes\n", size, acc_size);
    }

    VMath::cudaCheck(cudaGetDevice(&nOldDevice), "cudaGetDevice", __FILE__, __LINE__);
    VMath::cudaCheck(cudaSetDevice(device), "cudaSetDevice", __FILE__, __LINE__);
#ifndef CUDA_LEAK_CHECK
    VMath::cudaCheck(cudaMalloc(&ptr, size), "cudaMalloc", __FILE__, __LINE__);
#else
    malloc_mutex.lock();
    alloc_nth++;
    alloc_acc += size;
    cudaError_t cuda_ret = cudaMalloc(&ptr, size);
    //printf("[alloc %d] cudaMalloc(%lld) on device-%d\n", alloc_nth++, size, nDevice);
    ptrs[ptr] = size;
    malloc_mutex.unlock();
    VMath::cudaCheck(cuda_ret, "cudaMalloc");
#endif

    VMath::cudaCheck(cudaSetDevice(nOldDevice), "cudaSetDevice", __FILE__, __LINE__);

    if (ptr == NULL) VP_THROW(VERR_INVALID_POINTER);
    
    return ptr;
}

void* VMath::mem_alloc(int device, int64 size) {
    if (device < 0) return mem_alloc_host(size);
    else return mem_alloc_cuda(device, size);
}

//--------------------------------------------------------------------------------------------------

__static__ void seed_random_host(int64 random_seed) {
#ifdef NO_RANDOM_HOST
    ms_no_random_coin = ms_no_random_seed;
#else
    ms_randGen.seed(1234);
#endif
}

__static__ void seed_random_cuda(int64 random_seed) {
    VP_MEMO(VERR_UNIMPLEMENTEDYET);
}

void VMath::seed_random(int64 random_seed) {
    seed_random_host(random_seed);
    seed_random_cuda(random_seed);
}

//--------------------------------------------------------------------------------------------------

__static__ void mem_free_host(void* ptr) {
#ifdef DEBUG_HOST_MEM
    if (ms_allocHostSize.find(ptr) == ms_allocHostSize.end()) {
        VP_THROW(VERR_INVALID_SESSION_HANDLE);
    }
    int64 size = ms_allocHostSize[ptr];
    ms_allocHostSize.erase(ptr);
    ms_freeHostTotal += size;
#endif
    free(ptr);
}

__static__ void mem_free_cuda(void* ptr) {
#ifndef CUDA_LEAK_CHECK
    VMath::cudaCheck(cudaFree(ptr), "cudaFree", __FILE__, __LINE__);
#else
    malloc_mutex.lock();
    free_nth++;
    ptrs.erase(ptr);
    //ptrs.erase(ptrs.find(ptr));
    cudaError_t cuda_ret = cudaFree(ptr);
    //printf("[free %d] cudaFree() on device-???\n", free_nth++);
    if (cuda_ret != 0) {
        int nnn = 0;
    }
    malloc_mutex.unlock();
    VMath::cudaCheck(cuda_ret, "cudaFree");
#endif
}

void VMath::mem_free(int device, void* ptr) {
    if (device < 0) mem_free_host(ptr);
    else mem_free_cuda(ptr);
}

//--------------------------------------------------------------------------------------------------
//
void VMath::DumpUsage() {
#ifdef DEBUG_HOST_MEM
    printf("Host memory via VMath: %lld alloc, %lld free => %lld left\n", ms_allocHostTotal, ms_freeHostTotal, ms_allocHostTotal - ms_freeHostTotal);
    for (auto& it : ms_allocHostSize) {
        printf("    0x%llx => %lld bytes\n", (int64)it.first, it.second);
    }
#endif
}

//--------------------------------------------------------------------------------------------------
//
void VMath::memcpy_host_to_host(void* dptr, void* sptr, int64 size) {
    memcpy(dptr, sptr, size);
}

void VMath::memcpy_host_to_device(void* dptr, void* sptr, int64 size) {
    cudaCheck(cudaMemcpy(dptr, sptr, size, cudaMemcpyHostToDevice), "cudaMemcpy_host_to_device", __FILE__, __LINE__);
}

void VMath::memcpy_device_to_host(void* dptr, void* sptr, int64 size) {
    cudaCheck(cudaMemcpy(dptr, sptr, size, cudaMemcpyDeviceToHost), "cudaMemcpy_device_to_host", __FILE__, __LINE__);
}

void VMath::memcpy_device_to_device(void* dptr, void* sptr, int64 size) {
    cudaCheck(cudaMemcpy(dptr, sptr, size, cudaMemcpyDeviceToDevice), "cudaMemcpy_device_to_device", __FILE__, __LINE__);
}

//--------------------------------------------------------------------------------------------------
//
float _sigmoid_host(float x) {
    return (x > 0) ? (1.0f / (1.0f + ::expf(-x))) : (::expf(x) / (::expf(x) + 1.0f));
}

__device__ float _sigmoid_cuda(float x) {
    return (x > 0) ? (1.0f / (1.0f + ::expf(-x))) : (::expf(x) / (::expf(x) + 1.0f));
}

float _tanh_host(float x) {
    return 2 * _sigmoid_host(2 * x) - 1;
}

__device__ float _tanh_cuda(float x) {
    return 2 * _sigmoid_cuda(2 * x) - 1;
}

float _sigmoid_derv_with_x_host(float x) {
    float y = _sigmoid_host(x);
    return y * (1.0f - y);
}

__device__ float _sigmoid_derv_with_x_cuda(float x) {
    float y = _sigmoid_cuda(x);
    return y * (1.0f - y);
}

float _sigmoid_derv_with_y_host(float y) {
    return y * (1.0f - y);
}

__device__ float _sigmoid_derv_with_y_cuda(float y) {
    return y * (1.0f - y);
}

float _tanh_derv_with_x_host(float x) {
    float y = _tanh_host(x);
    return (1.0f - y * y);
}

__device__ float _tanh_derv_with_x_cuda(float x) {
    float y = _tanh_cuda(x);
    return (1.0f - y * y);
}

float _tanh_derv_with_y_host(float y) {
    return (1.0f - y * y);
}

__device__ float _tanh_derv_with_y_cuda(float y) {
    return (1.0f - y * y);
}

//--------------------------------------------------------------------------------------------------

void VMath::copy_data(int dest_device, int src_device, void* py, void* px, int64 nbytes) {
    if (dest_device < 0) {
        if (src_device < 0) memcpy_host_to_host(py, px, nbytes);
        else                memcpy_device_to_host(py, px, nbytes);
    }
    else {
        if (src_device < 0) memcpy_host_to_device(py, px, nbytes);
        else                memcpy_device_to_device(py, px, nbytes);
    }
}

//--------------------------------------------------------------------------------------------------

__static__ void accumulate_grad_host(float* py, float* px, int64 ndat) {
    for (int64 nd = 0; nd < ndat; nd++) {
        py[nd] += px[nd];
    }
}

__global__ void accumulate_grad_cuda(int64 size, float* py, float* px, int64 ndat) {
    int64 n = blockIdx.x * blockDim.x + threadIdx.x;

    if (n < size) {
        py[n] += px[n];
    }
}

void VMath::accumulate_grad(int device, float* py, float* px, int64 ndat) {
    int64 size = ndat;
    CUDA_CALL(accumulate_grad, device, size, py, px, ndat);
}

//--------------------------------------------------------------------------------------------------

__static__ void subtract_param_grad_host(float* pp, float* pg, int64 nrow, HYPER hLearningRate) {
    float learningRate = HYPER_ACCESS(hLearningRate);
    for (int64 n = 0; n < nrow; n++) {
        pp[n] -= pg[n] * learningRate;
    }
}

__global__ void subtract_param_grad_cuda(int64 size, float* pp, float* pg, int64 nrow, HYPER hLearningRate) {
    int64 n = blockIdx.x * blockDim.x + threadIdx.x;

    if (n < size) {
        float learningRate = HYPER_ACCESS(hLearningRate);
        pp[n] -= pg[n] * learningRate;
    }
}

void VMath::subtract_param_grad(int device, float* pp, float* pg, int64 nrow, HYPER hLearningRate) {
    int64 size = nrow;
    CUDA_CALL(subtract_param_grad, device, size, pp, pg, nrow, hLearningRate);
}

//--------------------------------------------------------------------------------------------------

__static__ void apply_decay_host(float* pd, float* pp, float* pg, int64 nrow, HYPER hL1Decay, HYPER hL2Decay) {
    float l1Decay = HYPER_ACCESS(hL1Decay);
    float l2Decay = HYPER_ACCESS(hL2Decay);

    for (int64 n = 0; n < nrow; n++) {
        float delta = pg[n];

        if (l2Decay > 0) delta += l2Decay * pp[n];
        if (l1Decay > 0) delta += l1Decay * ((pp[n] > 0) ? 1 : ((pp[n] < 0) ? -1 : 0));

        pd[n] = delta;
    }
}

__global__ void apply_decay_cuda(int64 size, float* pd, float* pp, float* pg, int64 nrow, HYPER hL1Decay, HYPER hL2Decay) {
    int64 n = blockIdx.x * blockDim.x + threadIdx.x;

    if (n < size) {
        float delta = pg[n];

        float l1Decay = HYPER_ACCESS(hL1Decay);
        float l2Decay = HYPER_ACCESS(hL2Decay);

        if (l2Decay > 0) delta += l2Decay * pp[n];
        if (l1Decay > 0) delta += l1Decay * ((pp[n] > 0) ? 1 : ((pp[n] < 0) ? -1 : 0));

        pd[n] = delta;
    }
}

void VMath::apply_decay(int device, float* pd, float* pp, float* pg, int64 nrow, HYPER hL1Decay, HYPER hL2Decay) {
    int64 size = nrow;
    CUDA_CALL(apply_decay, device, size, pd, pp, pg, nrow, hL1Decay, hL2Decay);
}

//--------------------------------------------------------------------------------------------------

__static__ void eval_adam_delta_host(float* pa, float* pg, float* ps, float* pt, float* pn, int64 nrow, HYPER hRo1, HYPER hRo2, HYPER hEpsilon) {
    float ro1 = HYPER_ACCESS(hRo1);
    float ro2 = HYPER_ACCESS(hRo2);
    float epsilon = HYPER_ACCESS(hEpsilon);

    for (int64 n = 0; n < nrow; n++) {
        float delta = pg[n];
        float nstep = pn[n] + 1;

        ps[n] = ro1 * ps[n] + (1 - ro1) * delta;
        pt[n] = ro2 * pt[n] + (1 - ro2) * delta * delta;

        float sterm = ps[n] / (1 - ::powf(ro1, nstep));
        float tterm = pt[n] / (1 - ::powf(ro2, nstep));

        pa[n] = sterm / (::sqrtf(tterm) + epsilon);
        pn[n] = nstep;
    }
}

__global__ void eval_adam_delta_cuda(int64 size, float* pa, float* pg, float* ps, float* pt, float* pn, int64 nrow, HYPER hRo1, HYPER hRo2, HYPER hEpsilon) {
    int64 n = blockIdx.x * blockDim.x + threadIdx.x;

    if (n < size) {
        float ro1 = HYPER_ACCESS(hRo1);
        float ro2 = HYPER_ACCESS(hRo2);
        float epsilon = HYPER_ACCESS(hEpsilon);

        float delta = pg[n];
        float nstep = pn[n] + 1;

        ps[n] = ro1 * ps[n] + (1 - ro1) * delta;
        pt[n] = ro2 * pt[n] + (1 - ro2) * delta * delta;

        float sterm = ps[n] / (1 - ::powf(ro1, nstep));
        float tterm = pt[n] / (1 - ::powf(ro2, nstep));

        pa[n] = sterm / (::sqrtf(tterm) + epsilon);
        pn[n] = nstep;
    }
}

void VMath::eval_adam_delta(int device, float* pa, float* pg, float* ps, float* pt, float* pn, int64 nrow, HYPER hRo1, HYPER hRo2, HYPER hEpsilon) {
    int64 size = nrow;
    CUDA_CALL(eval_adam_delta, device, size, pa, pg, ps, pt, pn, nrow, hRo1, hRo2, hEpsilon);
}

//--------------------------------------------------------------------------------------------------

__static__ void fill_int_host(int* ptr, int64 size, int value) {
    for (int n = 0; n < size; n++) {
        ptr[n] = value;
    }
}

__global__ void fill_int_cuda(int64 size, int* ptr, int64 size2, int value) {
    int64 n = blockIdx.x * blockDim.x + threadIdx.x;

    if (n < size) {
        ptr[n] = value;
    }
}

void VMath::fill_int(int device, int* ptr, int64 size, int value) {
    CUDA_CALL(fill_int, device, size, ptr, size, value);
}

//--------------------------------------------------------------------------------------------------

__static__ void fill_float_host(float* ptr, int64 size, float value) {
    for (int n = 0; n < size; n++) {
        ptr[n] = value;
    }
}

__global__ void fill_float_cuda(int64 size, float* ptr, int64 size2, float value) {
    int64 n = blockIdx.x * blockDim.x + threadIdx.x;

    if (n < size) {
        ptr[n] = value;
    }
}

void VMath::fill_float(int device, float* ptr, int64 size, float value) {
    CUDA_CALL(fill_float, device, size, ptr, size, value);
}

//--------------------------------------------------------------------------------------------------

void VMath::set_zero(int device, void* ptr, int64 size) {
    if (device < 0) {
        memset(ptr, 0, size);
    }
    else {
        cudaMemset(ptr, 0, size);
    }
}

//--------------------------------------------------------------------------------------------------

void VMath::memcpy_from_host(int device, void* py, void* px, int64 size) {
    if (device < 0) {
        memcpy(py, px, size);
    }
    else {
        cudaCheck(cudaMemcpy(py, px, size, cudaMemcpyHostToDevice), "memcpy_from_host", __FILE__, __LINE__);
    }
}

//--------------------------------------------------------------------------------------------------

void VMath::memcpy_to_host(int device, void* py, void* px, int64 size) {
    if (size <= 0) {
        printf("memcpy_to_host(size == 0) called\n");
    }

    if (device < 0) {
        memcpy(py, px, size);
    }
    else {
        cudaCheck(cudaMemcpy(py, px, size, cudaMemcpyDeviceToHost), "memcpy_to_host", __FILE__, __LINE__);
    }
}

//--------------------------------------------------------------------------------------------------

void VMath::memcpy_to_device(int device, void* py, void* px, int64 size) {
    if (device < 0) {
        cudaCheck(cudaMemcpy(py, px, size, cudaMemcpyHostToDevice), "memcpy_to_device", __FILE__, __LINE__);
    }
    else {
        cudaCheck(cudaMemcpy(py, px, size, cudaMemcpyDeviceToDevice), "memcpy_to_device", __FILE__, __LINE__);
    }
}

//--------------------------------------------------------------------------------------------------

__static__ void init_random_normal_host(float* ptr, int64 ndat, float mean, float init_arg) {
#ifndef NO_RANDOM_HOST
    std::normal_distribution<float> coin(mean, init_arg);
#endif

    for (int n = 0; n < ndat; n++) {
#ifdef NO_RANDOM_HOST
        ptr[n] = ms_no_random_normal(mean, init_arg);
#else
        ptr[n] = coin(ms_randGen);
#endif
    }
}

__global__ void init_random_normal_cuda(int64 size, float* ptr, int64 ndat, float mean, float init_arg) {
    int64 n = blockIdx.x * blockDim.x + threadIdx.x;

    if (n < size) {
#ifndef NO_RANDOM_CUDA
        curandState state;

        curand_init(clock64(), n, 0, &state);

        float dice = curand_normal(&state);
#else
        float dice = dev_no_random_normal(n, 0, 1);
#endif

        ptr[n] = (dice * init_arg) + mean;
    }
}

void VMath::init_random_normal(int device, float* ptr, int64 ndat, float mean, float init_arg, bool adaptive) {
    int64 size = ndat;
    CUDA_CALL(init_random_normal, device, size, ptr, ndat, mean, init_arg);

    if (adaptive) {
        float sum = get_sum(device, ptr, ndat);
        float mean = sum / (float)ndat;

        sub(device, ptr, ndat, mean);
    }
}

//--------------------------------------------------------------------------------------------------

__static__ void init_random_uniform_host(float* ptr, int64 ndat, float mean, float init_arg) {
#ifndef NO_RANDOM_HOST
    std::uniform_real_distribution<float> coin(mean - init_arg, mean + init_arg);
#endif

    for (int n = 0; n < ndat; n++) {
#ifdef NO_RANDOM_HOST
        ptr[n] = ms_no_random_uniform(mean - init_arg, mean + init_arg);
#else
        ptr[n] = coin(ms_randGen);
#endif
    }
}

__global__ void init_random_uniform_cuda(int64 size, float* ptr, int64 ndat, float mean, float init_arg) {
    int64 n = blockIdx.x * blockDim.x + threadIdx.x;

    if (n < size) {
#ifndef NO_RANDOM_CUDA
        curandState state;

        curand_init(clock64(), n, 0, &state);

        float dice = curand_uniform(&state);
#else
        float dice = dev_no_random_uniform(n, 0, 1);
#endif

        ptr[n] = dice;
    }
}

void VMath::init_random_uniform(int device, float* ptr, int64 ndat, float mean, float init_arg) {
    int64 size = ndat;
    CUDA_CALL(init_random_uniform, device, size, ptr, ndat, mean, init_arg);
}

//--------------------------------------------------------------------------------------------------

__static__ void sub_host(float* ptr, int64 ndat, float val) {
    for (int64 n = 0; n < ndat; n++) {
        ptr[n] -= val;
    }
}

__global__ void sub_cuda(int64 size, float* ptr, int64 ndat, float val) {
    int64 n = blockIdx.x * blockDim.x + threadIdx.x;

    if (n < size) {
        ptr[n] -= val;
    }
}

void VMath::sub(int device, float* ptr, int64 size, float val) {
    CUDA_CALL(sub, device, size, ptr, size, val);
}

//--------------------------------------------------------------------------------------------------

__static__ float get_sum_host(float* ptr, int64 ndat) {
    float sum = 0.0f;
    for (int n = 0; n < ndat; n++) {
        sum += ptr[n];
    }
    return sum;
}

float VMath::get_sum(int device, float* ptr, int64 size) {
    if (device < 0) return get_sum_host(ptr, size);
    else {
        float* host_buf = (float*)mem_alloc_host(size * sizeof(float));
        cudaCheck(cudaMemcpy(host_buf, ptr, size * sizeof(float), cudaMemcpyDeviceToHost), "cudaMemcpy_sum_cuda", __FILE__, __LINE__);
        float sum = get_sum_host(host_buf, size);
        mem_free_host(host_buf);
        return sum;
    }
}

//--------------------------------------------------------------------------------------------------

void VMath::get_slice(int device, void* py, void* px, int64 nbytes) {
    if (device < 0) {
        cudaCheck(cudaMemcpy(py, px, nbytes, cudaMemcpyHostToDevice), "cudaMemcpy_host_to_device", __FILE__, __LINE__);
    }
    else {
        cudaCheck(cudaMemcpy(py, px, nbytes, cudaMemcpyDeviceToDevice), "cudaMemcpy_device_to_device", __FILE__, __LINE__);
    }
}

//--------------------------------------------------------------------------------------------------

/*
void VMath::copy_slice_from(int device, void* py, void* px, int64 nbytes) {
    if (device < 0) {
        cudaCheck(cudaMemcpy(py, px, nbytes, cudaMemcpyDeviceToDevice), "cudaMemcpy_device_to_device");
    }
    else {
        cudaCheck(cudaMemcpy(py, px, nbytes, cudaMemcpyDeviceToDevice), "cudaMemcpy_device_to_device");
    }
}
*/

//--------------------------------------------------------------------------------------------------

__static__ void copy_host(float* py, float* pa, int64 ndat) {
    for (int64 n = 0; n < ndat; n++) {
        py[n] = pa[n];
    }
}

__global__ void copy_cuda(int64 size, float* py, float* pa, int64 ndat) {
    int64 n = blockIdx.x * blockDim.x + threadIdx.x;

    if (n < size) {
        py[n] = pa[n];
    }
}

void VMath::copy(int device, float* py, float* pa, int64 ndat) {
    int64 size = ndat;
    CUDA_CALL(copy, device, size, py, pa, ndat);
}

//--------------------------------------------------------------------------------------------------

__static__ void minus_host(float* py, int64 ndat) {
    for (int64 n = 0; n < ndat; n++) {
        py[n] *= -1;
    }
}

__global__ void minus_cuda(int64 size, float* py, int64 ndat) {
    int64 n = blockIdx.x * blockDim.x + threadIdx.x;

    if (n < size) {
        py[n] *= -1;
    }
}

void VMath::minus(int device, float* py, int64 ndat) {
    int64 size = ndat;
    CUDA_CALL(minus, device, size, py, ndat);
}

//--------------------------------------------------------------------------------------------------

__static__ void add_host(float* py, float* pa, float* pb, int64 ndat) {
    for (int64 nd = 0; nd < ndat; nd++) {
        py[nd] = pa[nd] + pb[nd];
    }
}

__global__ void add_cuda(int64 size, float* py, float* pa, float* pb, int64 ndat) {
    int64 n = blockIdx.x * blockDim.x + threadIdx.x;

    if (n < size) {
        py[n] = pa[n] + pb[n];
    }
}

void VMath::add(int device, float* py, float* pa, float* pb, int64 ndat) {
    int64 size = ndat;
    CUDA_CALL(add, device, size, py, pa, pb, ndat);
}

//--------------------------------------------------------------------------------------------------

__static__ void add_residual_host(float* py, float* pa, float* pb, int64 ndat, int64 nchn1, int64 nchn2, int64 nrest) {
    int64 ratio = nchn1 / nchn2;

    for (int64 nd = 0, n = 0; nd < ndat; nd++) {
        for (int64 nc = 0; nc < nchn1; nc++) {
            for (int64 nn = 0; nn < nrest; nn++, n++) {
                py[n] = pa[n] + pb[(nd * nchn2 + nc / ratio) * nrest + nn];
            }
        }
    }
}

__global__ void add_residual_cuda(int64 size, float* py, float* pa, float* pb, int64 ndat, int64 nchn1, int64 nchn2, int64 nrest) {
    int64 n = blockIdx.x * blockDim.x + threadIdx.x;

    if (n < size) {
        int64 nd = n / (nchn1 * nrest);
        int64 nc = n / nrest % nchn1;
        int64 nn = n % nrest;

        int64 ratio = nchn1 / nchn2;

        py[n] = pa[n] + pb[(nd * nchn2 + nc / ratio) * nrest + nn];
    }
}

void VMath::add_residual(int device, float* py, float* pa, float* pb, int64 ndat, int64 nchn1, int64 nchn2, int64 nrest) {
    int64 size = ndat * nchn1 * nrest;
    CUDA_CALL(add_residual, device, size, py, pa, pb, ndat, nchn1, nchn2, nrest);
}

//--------------------------------------------------------------------------------------------------

__static__ void add_residual_backward_b_host(float* pgb, float* pgy, int64 ndat, int64 nchn1, int64 nchn2, int64 nrest, bool acc) {
    int64 ratio = nchn1 / nchn2;

    for (int64 nd = 0, n = 0; nd < ndat; nd++) {
        for (int64 nc = 0; nc < nchn2; nc++) {
            for (int64 nn = 0; nn < nrest; nn++, n++) {
                float sum = 0;

                for (int64 k = 0; k < ratio; k++) {
                    sum += pgy[(nd * nchn1 + nc * ratio + k) * nrest + nn];
                }

                if (acc) pgb[n] += sum;
                else pgb[n] = sum;
            }
        }
    }
}

__global__ void add_residual_backward_b_cuda(int64 size, float* pgb, float* pgy, int64 ndat, int64 nchn1, int64 nchn2, int64 nrest, bool acc) {
    int64 n = blockIdx.x * blockDim.x + threadIdx.x;

    if (n < size) {
        int64 nd = n / (nrest * nchn2);
        int64 nc = n / nrest % nchn2;
        int64 nn = n % nrest;

        int64 ratio = nchn1 / nchn2;

        float sum = 0;

        for (int64 k = 0; k < ratio; k++) {
            sum += pgy[(nd * nchn1 + nc * ratio + k) * nrest + nn];
        }

        if (acc) pgb[n] += sum;
        else pgb[n] = sum;
    }
}

void VMath::add_residual_backward_b(int device, float* pgb, float* pgy, int64 ndat, int64 nchn1, int64 nchn2, int64 nrest, bool acc) {
    int64 size = ndat * nchn2 * nrest;
    CUDA_CALL(add_residual_backward_b, device, size, pgb, pgy, ndat, nchn1, nchn2, nrest, acc);
}

//--------------------------------------------------------------------------------------------------

__static__ void add_bias_host(float* py, float* pa, float* pb, int64 nrow, int64 ncol) {
    for (int64 nr = 0; nr < nrow; nr++) {
        for (int64 nc = 0; nc < ncol; nc++) {
            py[nr * ncol + nc] = pa[nr * ncol + nc] + pb[nc];
        }
    }
}

__global__ void add_bias_cuda(int64 size, float* py, float* pa, float* pb, int64 nrow, int64 ncol) {
    int64 n = blockIdx.x * blockDim.x + threadIdx.x;

    if (n < size) {
        int64 nr = n / ncol;
        int64 nc = n % ncol;

        py[n] = pa[nr * ncol + nc] + pb[nc];
    }
}

void VMath::add_bias(int device, float* py, float* pa, float* pb, int64 nrow, int64 ncol) {
    int64 size = nrow * ncol;
    CUDA_CALL(add_bias, device, size, py, pa, pb, nrow, ncol);
}

//--------------------------------------------------------------------------------------------------

__static__ void add_bias_backward_b_host(float* pgb, float* pgy, int64 nrow, int64 ncol, bool acc) {
    for (int64 nc = 0; nc < ncol; nc++) {
        float sum = 0;
        for (int64 nr = 0; nr < nrow; nr++) {
            sum += pgy[nr * ncol + nc];
        }

        if (acc) pgb[nc] += sum;
        else pgb[nc] = sum;
    }
}

__global__ void add_bias_backward_b_cuda(int64 size, float* pgb, float* pgy, int64 nrow, int64 ncol, bool acc) {
    int64 n = blockIdx.x * blockDim.x + threadIdx.x;

    if (n < size) {
        int64 nc = n;

        float sum = 0;
        for (int64 nr = 0; nr < nrow; nr++) {
            sum += pgy[nr * ncol + nc];
        }

        if (acc) pgb[nc] += sum;
        else pgb[nc] = sum;
    }
}

void VMath::add_bias_backward_b(int device, float* pgb, float* pgy, int64 nrow, int64 ncol, bool acc) {
    int64 size = ncol;
    CUDA_CALL(add_bias_backward_b, device, size, pgb, pgy, nrow, ncol, acc);
}

//--------------------------------------------------------------------------------------------------

__static__ void add_2d_bias_host(float* py, float* pa, float* pb, int64 ndat, int64 xchn, int64 xh, int64 xw) {
    for (int64 nd = 0, n = 0; nd < ndat; nd++) {
        for (int64 xn = 0; xn < xchn; xn++) {
            for (int64 nr = 0; nr < xh; nr++) {
                for (int64 nc = 0; nc < xw; nc++) {
                    py[n] = pa[n] + pb[xn];
                }
            }
        }
    }
}

__global__ void add_2d_bias_cuda(int64 size, float* py, float* pa, float* pb, int64 ndat, int64 xchn, int64 xh, int64 xw) {
    int64 n = blockIdx.x * blockDim.x + threadIdx.x;

    if (n < size) {
        int64 xn = n / (xh * xw) % xchn;
        py[n] = pa[n] + pb[xn];
    }
}

void VMath::add_2d_bias(int device, float* py, float* pa, float* pb, int64 ndat, int64 xchn, int64 xh, int64 xw) {
    int64 size = ndat * xchn * xh * xw;
    CUDA_CALL(add_2d_bias, device, size, py, pa, pb, ndat, xchn, xh, xw);
}

//--------------------------------------------------------------------------------------------------

__static__ void add_2d_bias_backward_b_host(float* pgb, float* pgy, int64 ndat, int64 xchn, int64 xh, int64 xw, bool acc) {
    for (int64 nn = 0; nn < xchn; nn++) {
        float sum = 0;
        for (int64 nd = 0; nd < ndat; nd++) {
            for (int64 nr = 0; nr < xh; nr++) {
                for (int64 nc = 0; nc < xw; nc++) {
                    int64 ypos = ((nd * xchn + nn) * xh + nr) * xw + nc;
                    sum += pgy[ypos];
                }
            }
        }

        if (acc) pgb[nn] += sum;
        else pgb[nn] = sum;
    }
}

__global__ void add_2d_bias_backward_b_cuda(int64 size, float* pgb, float* pgy, int64 ndat, int64 xchn, int64 xh, int64 xw, bool acc) {
    int64 n = blockIdx.x * blockDim.x + threadIdx.x;

    if (n < size) {
        int64 nn = n;

        float sum = 0;
        for (int64 nd = 0; nd < ndat; nd++) {
            for (int64 nr = 0; nr < xh; nr++) {
                for (int64 nc = 0; nc < xw; nc++) {
                    int64 ypos = ((nd * xchn + nn) * xh + nr) * xw + nc;
                    sum += pgy[ypos];
                }
            }
        }

        if (acc) pgb[n] += sum;
        else pgb[n] = sum;
    }
}

void VMath::add_2d_bias_backward_b(int device, float* pgb, float* pgy, int64 ndat, int64 xchn, int64 xh, int64 xw, bool acc) {
    int64 size = xchn;
    CUDA_CALL(add_2d_bias_backward_b, device, size, pgb, pgy, ndat, xchn, xh, xw, acc);
}

//--------------------------------------------------------------------------------------------------

__static__ void subtract_host(float* py, float* pa, float* pb, int64 ndat) {
    for (int64 nd = 0; nd < ndat; nd++) {
        py[nd] = pa[nd] - pb[nd];
    }
}

__global__ void subtract_cuda(int64 size, float* py, float* pa, float* pb, int64 ndat) {
    int64 n = blockIdx.x * blockDim.x + threadIdx.x;

    if (n < size) {
        py[n] = pa[n] - pb[n];
    }
}

void VMath::subtract(int device, float* py, float* pa, float* pb, int64 ndat) {
    int64 size = ndat;
    CUDA_CALL(subtract, device, size, py, pa, pb, ndat);
}

//--------------------------------------------------------------------------------------------------

__static__ void subtract_bias_host(float* py, float* pa, float* pb, int64 ndat, int64 nrow, int64 ncol) {
    for (int64 nr = 0; nr < nrow; nr++) {
        for (int64 nc = 0; nc < ncol; nc++) {
            py[nr * ncol + nc] = pa[nr * ncol + nc] - pb[nc];
        }
    }
}

__global__ void subtract_bias_cuda(int64 size, float* py, float* pa, float* pb, int64 ndat, int64 nrow, int64 ncol) {
    int64 n = blockIdx.x * blockDim.x + threadIdx.x;

    if (n < size) {
        int64 nr = n / ncol;
        int64 nc = n % ncol;

        py[n] = pa[nr * ncol + nc] - pb[nc];
    }
}

void VMath::subtract_bias(int device, float* py, float* pa, float* pb, int64 ndat, int64 nrow, int64 ncol) {
    int64 size = nrow * ncol;
    CUDA_CALL(subtract_bias, device, size, py, pa, pb, ndat, nrow, ncol);
}

//--------------------------------------------------------------------------------------------------

__static__ void subtract_backward_b_host(float* pgb, float* pgy, int64 ndat) {
    for (int64 n = 0; n < ndat; n++) {
        pgb[n] = -pgy[n];
    }
}

__global__ void subtract_backward_b_cuda(int64 size, float* pgb, float* pgy, int64 ndat) {
    int64 n = blockIdx.x * blockDim.x + threadIdx.x;

    if (n < size) {
        pgb[n] = -pgy[n];
    }
}

void VMath::subtract_backward_b(int device, float* pgb, float* pgy, int64 ndat) {
    int64 size = ndat;
    CUDA_CALL(subtract_backward_b, device, size, pgb, pgy, ndat);
}

//--------------------------------------------------------------------------------------------------

__static__ void subtract_bias_backward_b_host(float* pgb, float* pgy, int64 ndat, int64 nrow, int64 ncol) {
    for (int64 nc = 0; nc < ncol; nc++) {
        float sum = 0;
        for (int64 nr = 0; nr < nrow; nr++) {
            sum += pgy[nr * ncol + nc];
        }
        pgb[nc] = -sum;
    }
}

__global__ void subtract_bias_backward_b_cuda(int64 size, float* pgb, float* pgy, int64 ndat, int64 nrow, int64 ncol) {
    int64 n = blockIdx.x * blockDim.x + threadIdx.x;

    if (n < size) {
        int64 nc = n;

        float sum = 0;
        for (int64 nr = 0; nr < nrow; nr++) {
            sum += pgy[nr * ncol + nc];
        }
        pgb[n] = -sum;
    }
}

void VMath::subtract_bias_backward_b(int device, float* pgb, float* pgy, int64 ndat, int64 nrow, int64 ncol) {
    int64 size = ncol;
    CUDA_CALL(subtract_bias_backward_b, device, size, pgb, pgy, ndat, nrow, ncol);
}

//--------------------------------------------------------------------------------------------------

__static__ void mult_host(float* py, float* px1, float* px2, int64 left1, int64 left2, int64 mid, int64 right1, int64 right2) {
    // 무지막지한 5중루프 같지만 루프 크기 중 적어도 세 개가 1이다.
    for (int64 nr1 = 0, n = 0; nr1 < left1; nr1++) {
        for (int64 nr2 = 0; nr2 < left2; nr2++) {
            for (int64 nm = 0; nm < mid; nm++) {
                for (int64 nc1 = 0; nc1 < right1; nc1++) {
                    for (int64 nc2 = 0; nc2 < right2; nc2++, n++) {
                        int64 xpos1 = (nr1 * mid + nm) * right1 + nc1;
                        int64 xpos2 = (nr2 * mid + nm) * right2 + nc2;

                        py[n] = px1[xpos1] * px2[xpos2];
                    }
                }
            }
        }
    }
}

__global__ void mult_cuda(int64 size, float* py, float* px1, float* px2, int64 left1, int64 left2, int64 mid, int64 right1, int64 right2) {
    int64 n = blockIdx.x * blockDim.x + threadIdx.x;

    if (n < size) {
        int64 nr1 = n / (left2 * mid * right1 * right2);
        int64 nr2 = n / (mid * right1 * right2) % left2;
        int64 nm = n / (right1 * right2) % mid;
        int64 nc1 = n / right2 % right1;
        int64 nc2 = n % right2;

        int64 xpos1 = (nr1 * mid + nm) * right1 + nc1;
        int64 xpos2 = (nr2 * mid + nm) * right2 + nc2;

        py[n] = px1[xpos1] * px2[xpos2];
    }
}

void VMath::mult(int device, float* py, float* px1, float* px2, int64 left1, int64 left2, int64 mid, int64 right1, int64 right2) {
    int64 size = left1 * left2 * mid * right1 * right2;
    CUDA_CALL(mult, device, size, py, px1, px2, left1, left2, mid, right1, right2);
}

//--------------------------------------------------------------------------------------------------

__static__ void mult_backward_x1_host(float* pgx, float* pgy, float* px2, int64 left1, int64 left2, int64 mid, int64 right1, int64 right2) {
    // 무지막지한 5중루프 같지만 루프 크기 중 적어도 세 개가 1이다.
    for (int64 nr1 = 0, n = 0; nr1 < left1; nr1++) {
        for (int64 nm = 0; nm < mid; nm++) {
            for (int64 nc1 = 0; nc1 < right1; nc1++, n++) {
                float sum = 0;
                for (int64 nr2 = 0; nr2 < left2; nr2++) {
                    for (int64 nc2 = 0; nc2 < right2; nc2++) {
                        int64 xpos2 = (nr2 * mid + nm) * right2 + nc2;
                        int64 ypos = (((nr1 * left2 + nr2) * mid + nm) * right1 + nc1) * right2 + nc2;

                        float gy = pgy ? pgy[ypos] : 1.0f;
                        sum += gy * px2[xpos2];
                    }
                }
                pgx[n] = sum;
            }
        }
    }
}

__global__ void mult_backward_x1_cuda(int64 size, float* pgx, float* pgy, float* px2, int64 left1, int64 left2, int64 mid, int64 right1, int64 right2) {
    int64 n = blockIdx.x * blockDim.x + threadIdx.x;

    if (n < size) {
        int64 nr1 = n / (mid * right1);
        int64 nm = n / right1 % mid;
        int64 nc1 = n % right1;

        float sum = 0;
        for (int64 nr2 = 0; nr2 < left2; nr2++) {
            for (int64 nc2 = 0; nc2 < right2; nc2++) {
                int64 xpos2 = (nr2 * mid + nm) * right2 + nc2;
                int64 ypos = (((nr1 * left2 + nr2) * mid + nm) * right1 + nc1) * right2 + nc2;

                float gy = pgy ? pgy[ypos] : 1.0f;
                sum += gy * px2[xpos2];
            }
        }
        pgx[n] = sum;
    }
}

void VMath::mult_backward_x1(int device, float* pgx, float* pgy, float* px2, int64 left1, int64 left2, int64 mid, int64 right1, int64 right2) {
    int64 size = left1 * mid * right1;
    CUDA_CALL(mult_backward_x1, device, size, pgx, pgy, px2, left1, left2, mid, right1, right2);
}

//--------------------------------------------------------------------------------------------------

__static__ void mult_backward_x2_host(float* pgx, float* pgy, float* px1, int64 left1, int64 left2, int64 mid, int64 right1, int64 right2) {
    // 무지막지한 5중루프 같지만 루프 크기 중 적어도 세 개가 1이다.
    for (int64 nr2 = 0, n = 0; nr2 < left2; nr2++) {
        for (int64 nm = 0; nm < mid; nm++) {
            for (int64 nc2 = 0; nc2 < right2; nc2++, n++) {
                float sum = 0;
                for (int64 nr1 = 0; nr1 < left1; nr1++) {
                    for (int64 nc1 = 0; nc1 < right1; nc1++) {
                        int64 xpos1 = (nr1 * mid + nm) * right1 + nc1;
                        int64 ypos = (((nr1 * left2 + nr2) * mid + nm) * right1 + nc1) * right2 + nc2;

                        float gy = pgy ? pgy[ypos] : 1.0f;
                        sum += gy * px1[xpos1];
                    }
                }
                pgx[n] = sum;
            }
        }
    }
}

__global__ void mult_backward_x2_cuda(int64 size, float* pgx, float* pgy, float* px1, int64 left1, int64 left2, int64 mid, int64 right1, int64 right2) {
    int64 n = blockIdx.x * blockDim.x + threadIdx.x;

    if (n < size) {
        int64 nr2 = n / (mid * right2);
        int64 nm = n / right2 % mid;
        int64 nc2 = n % right2;

        float sum = 0;
        for (int64 nr1 = 0; nr1 < left1; nr1++) {
            for (int64 nc1 = 0; nc1 < right1; nc1++) {
                int64 xpos1 = (nr1 * mid + nm) * right1 + nc1;
                int64 ypos = (((nr1 * left2 + nr2) * mid + nm) * right1 + nc1) * right2 + nc2;

                float gy = pgy ? pgy[ypos] : 1.0f;
                sum += gy * px1[xpos1];
            }
        }
        pgx[n] = sum;
    }
}

void VMath::mult_backward_x2(int device, float* pgx, float* pgy, float* px1, int64 left1, int64 left2, int64 mid, int64 right1, int64 right2) {
    int64 size = left2 * mid * right2;
    CUDA_CALL(mult_backward_x2, device, size, pgx, pgy, px1, left1, left2, mid, right1, right2);
}

//--------------------------------------------------------------------------------------------------

__static__ void mult_se_mask_host(float* py, float* px1, float* px2, int64 ndat, int64 nchn, int64 nheight, int64 nwidth) {
    for (int64 nd = 0, n = 0; nd < ndat; nd++) {
        for (int64 nc = 0; nc < nchn; nc++) {
            for (int64 nh = 0; nh < nheight; nh++) {
                for (int64 nw = 0; nw < nwidth; nw++, n++) {
                    int64 mpos = nd * nchn + nc;
                    py[n] = px1[n] * px2[mpos];
                }
            }
        }
    }
}

__global__ void mult_se_mask_cuda(int64 size, float* py, float* px1, float* px2, int64 ndat, int64 nchn, int64 nheight, int64 nwidth) {
    int64 n = blockIdx.x * blockDim.x + threadIdx.x;

    if (n < size) {
        int64 nd = n / (nchn * nheight * nwidth);
        int64 nc = n / (nheight * nwidth ) % nchn;

        int64 mpos = nd * nchn + nc;
        py[n] = px1[n] * px2[mpos];
    }
}

void VMath::mult_se_mask(int device, float* py, float* px1, float* px2, int64 ndat, int64 nchn, int64 nheight, int64 nwidth) {
    int64 size = ndat * nchn * nheight * nwidth;
    CUDA_CALL(mult_se_mask, device, size, py, px1, px2, ndat, nheight, nwidth, nchn);
}

//--------------------------------------------------------------------------------------------------

__static__ void mult_se_mask_backward_x1_host(float* pgx, float* pgy, float* px2, int64 ndat, int64 nchn, int64 nheight, int64 nwidth) {
    for (int64 nd = 0, n = 0; nd < ndat; nd++) {
        for (int64 nc = 0; nc < nchn; nc++) {
            for (int64 nh = 0; nh < nheight; nh++) {
                for (int64 nw = 0; nw < nwidth; nw++, n++) {
                    int64 mpos = nd * nchn + nc;
                    pgx[n] = pgy[n] * px2[mpos];
                }
            }
        }
    }
}

__global__ void mult_se_mask_backward_x1_cuda(int64 size, float* pgx, float* pgy, float* px2, int64 ndat, int64 nchn, int64 nheight, int64 nwidth) {
    int64 n = blockIdx.x * blockDim.x + threadIdx.x;

    if (n < size) {
        int64 nd = n / (nchn * nheight * nwidth);
        int64 nc = n / (nheight * nwidth) % nchn;

        int64 mpos = nd * nchn + nc;
        pgx[n] = pgy[n] * px2[mpos];
    }
}

void VMath::mult_se_mask_backward_x1(int device, float* pgx, float* pgy, float* px2, int64 ndat, int64 nchn, int64 nheight, int64 nwidth) {
    int64 size = ndat * nchn * nheight * nwidth;
    CUDA_CALL(mult_se_mask_backward_x1, device, size, pgx, pgy, px2, ndat, nchn, nheight, nwidth);
}

//--------------------------------------------------------------------------------------------------

__static__ void mult_se_mask_backward_x2_host(float* pgx, float* pgy, float* px1, int64 ndat, int64 nchn, int64 nheight, int64 nwidth) {
    for (int64 n = 0; n < ndat * nchn; n++) {
        float sum = 0;
        int64 nk = n * nwidth * nheight;
        for (int64 k = 0; k < nwidth * nheight; k++, nk++) {
            sum += px1[nk] * pgy[nk];
        }
        pgx[n] = sum;
    }
}

__global__ void mult_se_mask_backward_x2_cuda(int64 size, float* pgx, float* pgy, float* px1, int64 ndat, int64 nchn, int64 nheight, int64 nwidth) {
    int64 n = blockIdx.x * blockDim.x + threadIdx.x;

    if (n < size) {
        float sum = 0;
        int64 nk = n * nwidth * nheight;
        for (int64 k = 0; k < nwidth * nheight; k++, nk++) {
            sum += px1[nk] * pgy[nk];
        }
        pgx[n] = sum;
    }
}

void VMath::mult_se_mask_backward_x2(int device, float* pgx, float* pgy, float* px1, int64 ndat, int64 nchn, int64 nheight, int64 nwidth) {
    int64 size = ndat * nchn;
    CUDA_CALL(mult_se_mask_backward_x2, device, size, pgx, pgy, px1, ndat, nchn, nheight, nwidth);
}

//--------------------------------------------------------------------------------------------------

__static__ void mult_scalar_host(float* py, int64 ndat, HYPER hCoef) {
    float coef = HYPER_ACCESS(hCoef);

    for (int64 n = 0; n < ndat; n++) {
        py[n] = py[n] * coef;
    }
}

__global__ void mult_scalar_cuda(int64 size, float* py, int64 ndat, HYPER hCoef) {
    int64 n = blockIdx.x * blockDim.x + threadIdx.x;

    if (n < size) {
        float coef = HYPER_ACCESS(hCoef);
        py[n] = py[n] * coef;
    }
}

void VMath::mult_scalar(int device, float* py, int64 ndat, HYPER hCoef) {
    int64 size = ndat;
    CUDA_CALL(mult_scalar, device, size, py, ndat, hCoef);
}

//--------------------------------------------------------------------------------------------------

__static__ void add_mult_scalar_to_host(float* py, float* pa, float* px, int64 ndat, HYPER hCoef) {
    float coef = HYPER_ACCESS(hCoef);

    for (int64 n = 0; n < ndat; n++) {
        py[n] = pa[n] + px[n] * coef;
    }
}

__global__ void add_mult_scalar_to_cuda(int64 size, float* py, float* pa, float* px, int64 ndat, HYPER hCoef) {
    int64 n = blockIdx.x * blockDim.x + threadIdx.x;

    if (n < size) {
        float coef = HYPER_ACCESS(hCoef);
        py[n] = pa[n] + px[n] * coef;
    }
}

void VMath::add_mult_scalar_to(int device, float* py, float* pa, float* px, int64 ndat, HYPER hCoef) {
    int64 size = ndat;
    CUDA_CALL(add_mult_scalar_to, device, size, py, pa, px, ndat, hCoef);
}

//--------------------------------------------------------------------------------------------------

__static__ void sub_mult_scalar_to_host(float* py, float* pa, float* px, int64 ndat, HYPER hCoef) {
    float coef = HYPER_ACCESS(hCoef);

    for (int64 n = 0; n < ndat; n++) {
        py[n] = pa[n] - px[n] * coef;
    }
}

__global__ void sub_mult_scalar_to_cuda(int64 size, float* py, float* pa, float* px, int64 ndat, HYPER hCoef) {
    int64 n = blockIdx.x * blockDim.x + threadIdx.x;

    if (n < size) {
        float coef = HYPER_ACCESS(hCoef);
        py[n] = pa[n] - px[n] * coef;
    }
}

void VMath::sub_mult_scalar_to(int device, float* py, float* pa, float* px, int64 ndat, HYPER hCoef) {
    int64 size = ndat;
    CUDA_CALL(sub_mult_scalar_to, device, size, py, pa, px, ndat, hCoef);
}

//--------------------------------------------------------------------------------------------------

__static__ void acc_sqsum_host(float* pr, float* pg, int64 ndat) {
    for (int64 n = 0; n < ndat; n++) {
        pr[n] = pr[n] + pg[n] * pg[n];
    }
}

__global__ void acc_sqsum_cuda(int64 size, float* pr, float* pg, int64 ndat) {
    int64 n = blockIdx.x * blockDim.x + threadIdx.x;

    if (n < size) {
        pr[n] = pr[n] + pg[n] * pg[n];
    }
}

void VMath::acc_sqsum(int device, float* pr, float* pg, int64 ndat) {
    int64 size = ndat;
    CUDA_CALL(acc_sqsum, device, size, pr, pg, ndat);
}

//--------------------------------------------------------------------------------------------------

__static__ void acc_sqsum_decay_host(float* pr, float* pg, int64 ndat, HYPER hDecay) {
    float decay = HYPER_ACCESS(hDecay);

    for (int64 n = 0; n < ndat; n++) {
        pr[n] = decay * pr[n] + (1 - decay) * pg[n] * pg[n];
    }
}

__global__ void acc_sqsum_decay_cuda(int64 size, float* pr, float* pg, int64 ndat, HYPER hDecay) {
    int64 n = blockIdx.x * blockDim.x + threadIdx.x;

    if (n < size) {
        float decay = HYPER_ACCESS(hDecay);
        pr[n] = decay * pr[n] + (1 - decay) * pg[n] * pg[n];
    }
}

void VMath::acc_sqsum_decay(int device, float* pr, float* pg, int64 ndat, HYPER hDecay) {
    int64 size = ndat;
    CUDA_CALL(acc_sqsum_decay, device, size, pr, pg, ndat, hDecay);
}

//--------------------------------------------------------------------------------------------------

__static__ void adagrad_update_host(float* pn, float* pg, float* pr, int64 ndat, HYPER hLearningRate, HYPER hSigma) {
    float learning_rate = HYPER_ACCESS(hLearningRate);
    float sigma = HYPER_ACCESS(hSigma);

    for (int64 n = 0; n < ndat; n++) {
        float coef = learning_rate / (sigma + ::sqrt(pr[n]));
        pn[n] = -coef * pg[n];
    }
}

__global__ void adagrad_update_cuda(int64 size, float* pn, float* pg, float* pr, int64 ndat, HYPER hLearningRate, HYPER hSigma) {
    int64 n = blockIdx.x * blockDim.x + threadIdx.x;

    if (n < size) {
        float learning_rate = HYPER_ACCESS(hLearningRate);
        float sigma = HYPER_ACCESS(hSigma);

        float coef = learning_rate / (sigma + ::sqrt(pr[n]));
        pn[n] = -coef * pg[n];
    }
}

void VMath::adagrad_update(int device, float* pn, float* pg, float* pr, int64 ndat, HYPER hLearningRate, HYPER hSigma) {
    int64 size = ndat;
    CUDA_CALL(adagrad_update, device, size, pn, pg, pr, ndat, hLearningRate, hSigma);
}

//--------------------------------------------------------------------------------------------------

__static__ void div_host(float* py, float* px1, float* px2, int64 left1, int64 left2, int64 mid, int64 right1, int64 right2) {
    // 무지막지한 5중루프 같지만 루프 크기 중 적어도 세 개가 1이다.
    for (int64 nr1 = 0, n = 0; nr1 < left1; nr1++) {
        for (int64 nr2 = 0; nr2 < left2; nr2++) {
            for (int64 nm = 0; nm < mid; nm++) {
                for (int64 nc1 = 0; nc1 < right1; nc1++) {
                    for (int64 nc2 = 0; nc2 < right2; nc2++, n++) {
                        int64 xpos1 = (nr1 * mid + nm) * right1 + nc1;
                        int64 xpos2 = (nr2 * mid + nm) * right2 + nc2;

                        py[n] = px1[xpos1] / px2[xpos2];
                    }
                }
            }
        }
    }
}

__global__ void div_cuda(int64 size, float* py, float* px1, float* px2, int64 left1, int64 left2, int64 mid, int64 right1, int64 right2) {
    int64 n = blockIdx.x * blockDim.x + threadIdx.x;

    if (n < size) {
        int64 nr1 = n / (left2 * mid * right1 * right2);
        int64 nr2 = n / (mid * right1 * right2) % left2;
        int64 nm = n / (right1 * right2) % mid;
        int64 nc1 = n / right2 % right1;
        int64 nc2 = n % right2;

        int64 xpos1 = (nr1 * mid + nm) * right1 + nc1;
        int64 xpos2 = (nr2 * mid + nm) * right2 + nc2;

        py[n] = px1[xpos1] / px2[xpos2];
    }
}

void VMath::div(int device, float* py, float* px1, float* px2, int64 left1, int64 left2, int64 mid, int64 right1, int64 right2) {
    int64 size = left1 * left2 * mid * right1 * right2;
    CUDA_CALL(div, device, size, py, px1, px2, left1, left2, mid, right1, right2);
}

//--------------------------------------------------------------------------------------------------

__static__ void div_backward_x1_host(float* pgx, float* pgy, float* px2, int64 left1, int64 left2, int64 mid, int64 right1, int64 right2) {
    // 무지막지한 5중루프 같지만 루프 크기 중 적어도 세 개가 1이다.
    for (int64 nr1 = 0, n = 0; nr1 < left1; nr1++) {
        for (int64 nm = 0; nm < mid; nm++) {
            for (int64 nc1 = 0; nc1 < right1; nc1++, n++) {
                float sum = 0;
                for (int64 nr2 = 0; nr2 < left2; nr2++) {
                    for (int64 nc2 = 0; nc2 < right2; nc2++) {
                        int64 xpos2 = (nr2 * mid + nm) * right2 + nc2;
                        int64 ypos = (((nr1 * left2 + nr2) * mid + nm) * right1 + nc1) * right2 + nc2;

                        sum += pgy[ypos] / px2[xpos2];
                    }
                }
                pgx[n] = sum;
            }
        }
    }
}

__global__ void div_backward_x1_cuda(int64 size, float* pgx, float* pgy, float* px2, int64 left1, int64 left2, int64 mid, int64 right1, int64 right2) {
    int64 n = blockIdx.x * blockDim.x + threadIdx.x;

    if (n < size) {
        int64 nr1 = n / (mid * right1);
        int64 nm = n / right1 % mid;
        int64 nc1 = n % right1;

        float sum = 0;
        for (int64 nr2 = 0; nr2 < left2; nr2++) {
            for (int64 nc2 = 0; nc2 < right2; nc2++) {
                int64 xpos2 = (nr2 * mid + nm) * right2 + nc2;
                int64 ypos = (((nr1 * left2 + nr2) * mid + nm) * right1 + nc1) * right2 + nc2;

                sum += pgy[ypos] / px2[xpos2];
            }
        }
        pgx[n] = sum;
    }
}

void VMath::div_backward_x1(int device, float* pgx, float* pgy, float* px2, int64 left1, int64 left2, int64 mid, int64 right1, int64 right2) {
    int64 size = left1 * mid * right1;
    CUDA_CALL(div_backward_x1, device, size, pgx, pgy, px2, left1, left2, mid, right1, right2);
}

//--------------------------------------------------------------------------------------------------

__static__ void div_backward_x2_host(float* pgx, float* pgy, float* px1, float* px2, int64 left1, int64 left2, int64 mid, int64 right1, int64 right2) {
    // 무지막지한 5중루프 같지만 루프 크기 중 적어도 세 개가 1이다.
    for (int64 nr2 = 0, n = 0; nr2 < left2; nr2++) {
        for (int64 nm = 0; nm < mid; nm++) {
            for (int64 nc2 = 0; nc2 < right2; nc2++, n++) {
                float sum = 0;
                for (int64 nr1 = 0; nr1 < left1; nr1++) {
                    for (int64 nc1 = 0; nc1 < right1; nc1++) {
                        int64 xpos1 = (nr1 * mid + nm) * right1 + nc1;
                        int64 ypos = (((nr1 * left2 + nr2) * mid + nm) * right1 + nc1) * right2 + nc2;

                        sum += pgy[ypos] * px1[xpos1];
                    }
                }
                pgx[n] = -sum / (px2[n] * px2[n]);
            }
        }
    }
}

__global__ void div_backward_x2_cuda(int64 size, float* pgx, float* pgy, float* px1, float* px2, int64 left1, int64 left2, int64 mid, int64 right1, int64 right2) {
    int64 n = blockIdx.x * blockDim.x + threadIdx.x;

    if (n < size) {
        int64 nr2 = n / (mid * right2);
        int64 nm = n / right2 % mid;
        int64 nc2 = n % right2;

        float sum = 0;
        for (int64 nr1 = 0; nr1 < left1; nr1++) {
            for (int64 nc1 = 0; nc1 < right1; nc1++) {
                int64 xpos1 = (nr1 * mid + nm) * right1 + nc1;
                int64 ypos = (((nr1 * left2 + nr2) * mid + nm) * right1 + nc1) * right2 + nc2;

                sum += pgy[ypos] * px1[xpos1];
            }
        }
        pgx[n] = -sum / (px2[n] * px2[n]);
    }
}

void VMath::div_backward_x2(int device, float* pgx, float* pgy, float* px1, float* px2, int64 left1, int64 left2, int64 mid, int64 right1, int64 right2) {
    int64 size = left2 * mid * right2;
    CUDA_CALL(div_backward_x2, device, size, pgx, pgy, px1, px2, left1, left2, mid, right1, right2);
}

//--------------------------------------------------------------------------------------------------

__static__ void abs_host(float* py, float* px, int64 ndat) {
    for (int64 nd = 0; nd < ndat; nd++) {
        py[nd] = (px[nd] > 0) ? px[nd] : -px[nd];
    }
}

__global__ void abs_cuda(int64 size, float* py, float* px, int64 ndat) {
    int64 n = blockIdx.x * blockDim.x + threadIdx.x;

    if (n < size) {
        py[n] = (px[n] > 0) ? px[n] : -px[n];
    }
}

void VMath::abs(int device, float* py, float* px, int64 ndat) {
    int64 size = ndat;
    CUDA_CALL(abs, device, size, py, px, ndat);
}

//--------------------------------------------------------------------------------------------------

__static__ void abs_backward_host(float* pgx, float* pgy, float* px, int64 ndat) {
    for (int64 n = 0; n < ndat; n++) {
        pgx[n] = (px[n] > 0) ? pgy[n] : -pgy[n];
    }
}

__global__ void abs_backward_cuda(int64 size, float* pgx, float* pgy, float* px, int64 ndat) {
    int64 n = blockIdx.x * blockDim.x + threadIdx.x;

    if (n < size) {
        pgx[n] = (px[n] > 0) ? pgy[n] : -pgy[n];
    }
}

void VMath::abs_backward(int device, float* pgx, float* pgy, float* px, int64 ndat) {
    int64 size = ndat;
    CUDA_CALL(abs_backward, device, size, pgx, pgy, px, ndat);
}

//--------------------------------------------------------------------------------------------------

__static__ void square_host(float* py, float* px, int64 ndat) {
    for (int64 nd = 0; nd < ndat; nd++) {
        py[nd] = px[nd] * px[nd];
    }
}

__global__ void square_cuda(int64 size, float* py, float* px, int64 ndat) {
    int64 n = blockIdx.x * blockDim.x + threadIdx.x;

    if (n < size) {
        py[n] = px[n] * px[n];
    }
}

void VMath::square(int device, float* py, float* px, int64 ndat) {
    int64 size = ndat;
    CUDA_CALL(square, device, size, py, px, ndat);
}

//--------------------------------------------------------------------------------------------------

__static__ void square_backward_host(float* pgx, float* pgy, float* px, int64 ndat) {
    for (int64 n = 0; n < ndat; n++) {
        pgx[n] = 2.0f * pgy[n] * px[n];
    }
}

__global__ void square_backward_cuda(int64 size, float* pgx, float* pgy, float* px, int64 ndat) {
    int64 n = blockIdx.x * blockDim.x + threadIdx.x;

    if (n < size) {
        pgx[n] = 2.0f * pgy[n] * px[n];
    }
}

void VMath::square_backward(int device, float* pgx, float* pgy, float* px, int64 ndat) {
    int64 size = ndat;
    CUDA_CALL(square_backward, device, size, pgx, pgy, px, ndat);
}

//--------------------------------------------------------------------------------------------------

__static__ void complement_1_host(float* py, float* px, int64 ndat) {
    for (int64 nd = 0; nd < ndat; nd++) {
        py[nd] = 1 - px[nd];
    }
}

__global__ void complement_1_cuda(int64 size, float* py, float* px, int64 ndat) {
    int64 n = blockIdx.x * blockDim.x + threadIdx.x;

    if (n < size) {
        py[n] = 1 - px[n];
    }
}

void VMath::complement_1(int device, float* py, float* px, int64 ndat) {
    int64 size = ndat;
    CUDA_CALL(complement_1, device, size, py, px, ndat);
}

//--------------------------------------------------------------------------------------------------

__static__ void complement_1_backward_host(float* pgx, float* pgy, int64 ndat) {
    for (int64 n = 0; n < ndat; n++) {
        pgx[n] = -pgy[n];
    }
}

__global__ void complement_1_backward_cuda(int64 size, float* pgx, float* pgy, int64 ndat) {
    int64 n = blockIdx.x * blockDim.x + threadIdx.x;

    if (n < size) {
        pgx[n] = -pgy[n];
    }
}

void VMath::complement_1_backward(int device, float* pgx, float* pgy, int64 ndat) {
    int64 size = ndat;
    CUDA_CALL(complement_1_backward, device, size, pgx, pgy, ndat);
}

//--------------------------------------------------------------------------------------------------

__static__ void sqrt_host(float* py, float* px, int64 ndat) {
    for (int64 nd = 0; nd < ndat; nd++) {
        py[nd] = ::sqrtf(px[nd]);
    }
}

__global__ void sqrt_cuda(int64 size, float* py, float* px, int64 ndat) {
    int64 n = blockIdx.x * blockDim.x + threadIdx.x;

    if (n < size) {
        py[n] = ::sqrtf(px[n]);
    }
}

void VMath::sqrt(int device, float* py, float* px, int64 ndat) {
    int64 size = ndat;
    CUDA_CALL(sqrt, device, size, py, px, ndat);
}

//--------------------------------------------------------------------------------------------------

__static__ void sqrt_backward_host(float* pgx, float* pgy, float* px, int64 ndat) {
    for (int64 n = 0; n < ndat; n++) {
        pgx[n] = (0.5f / ::sqrtf(px[n])) * pgy[n];
    }
}

__global__ void sqrt_backward_cuda(int64 size, float* pgx, float* pgy, float* px, int64 ndat) {
    int64 n = blockIdx.x * blockDim.x + threadIdx.x;

    if (n < size) {
        pgx[n] = (0.5f / ::sqrtf(px[n])) * pgy[n];
    }
}

void VMath::sqrt_backward(int device, float* pgx, float* pgy, float* px, int64 ndat) {
    int64 size = ndat;
    CUDA_CALL(sqrt_backward, device, size, pgx, pgy, px, ndat);
}

//--------------------------------------------------------------------------------------------------

__static__ void exp_host(float* py, float* px, int64 ndat) {
    for (int64 n = 0; n < ndat; n++) {
        py[n] = ::expf(px[n]);
    }
}

__global__ void exp_cuda(int64 size, float* py, float* px, int64 ndat) {
    int64 n = blockIdx.x * blockDim.x + threadIdx.x;

    if (n < size) {
        py[n] = ::expf(px[n]);
    }
}

void VMath::exp(int device, float* py, float* px, int64 ndat) {
    int64 size = ndat;
    CUDA_CALL(exp, device, size, py, px, ndat);
}

//--------------------------------------------------------------------------------------------------

__static__ void exp_backward_host(float* pgx, float* pgy, float* px, int64 ndat) {
    for (int64 n = 0; n < ndat; n++) {
        pgx[n] = pgy[n] * ::expf(px[n]);
    }
}

__global__ void exp_backward_cuda(int64 size, float* pgx, float* pgy, float* px, int64 ndat) {
    int64 n = blockIdx.x * blockDim.x + threadIdx.x;

    if (n < size) {
        pgx[n] = pgy[n] * ::expf(px[n]);
    }
}

void VMath::exp_backward(int device, float* pgx, float* pgy, float* px, int64 ndat) {
    int64 size = ndat;
    CUDA_CALL(exp_backward, device, size, pgx, pgy, px, ndat);
}

//--------------------------------------------------------------------------------------------------

__static__ void log_host(float* py, float* px, int64 ndat) {
    for (int64 n = 0; n < ndat; n++) {
        py[n] = ::logf(px[n]);
    }
}

__global__ void log_cuda(int64 size, float* py, float* px, int64 ndat) {
    int64 n = blockIdx.x * blockDim.x + threadIdx.x;

    if (n < size) {
        py[n] = ::logf(px[n]);
    }
}

void VMath::log(int device, float* py, float* px, int64 ndat) {
    int64 size = ndat;
    CUDA_CALL(log, device, size, py, px, ndat);
}

//--------------------------------------------------------------------------------------------------

__static__ void log_backward_host(float* pgx, float* pgy, float* px, int64 ndat) {
    for (int64 n = 0; n < ndat; n++) {
        float x = px[n];
        x = (x > 1e-10f) ? x : (x >= 0) ? 1e-10f : (x > -1e-10f) ? -1e-10f : x; // 0 나누기 오류 방지
        pgx[n] = pgy[n] / x;
    }
}

__global__ void log_backward_cuda(int64 size, float* pgx, float* pgy, float* px, int64 ndat) {
    int64 n = blockIdx.x * blockDim.x + threadIdx.x;

    if (n < size) {
        float x = px[n];
        x = (x > 1e-10f) ? x : (x >= 0) ? 1e-10f : (x > -1e-10f) ? -1e-10f : x; // 0 나누기 오류 방지
        pgx[n] = pgy[n] / x;
    }
}

void VMath::log_backward(int device, float* pgx, float* pgy, float* px, int64 ndat) {
    int64 size = ndat;
    CUDA_CALL(log_backward, device, size, pgx, pgy, px, ndat);
}

//--------------------------------------------------------------------------------------------------

__static__ void maximum_host(float* py, float* px1, float* px2, int64 left1, int64 left2, int64 mid, int64 right1, int64 right2) {
    // 무지막지한 5중루프 같지만 루프 크기 중 적어도 세 개가 1이다.
    for (int64 nr1 = 0, n = 0; nr1 < left1; nr1++) {
        for (int64 nr2 = 0; nr2 < left2; nr2++) {
            for (int64 nm = 0; nm < mid; nm++) {
                for (int64 nc1 = 0; nc1 < right1; nc1++) {
                    for (int64 nc2 = 0; nc2 < right2; nc2++, n++) {
                        int64 xpos1 = (nr1 * mid + nm) * right1 + nc1;
                        int64 xpos2 = (nr2 * mid + nm) * right2 + nc2;

                        float x1 = px1[xpos1];
                        float x2 = px2[xpos2];

                        py[n] = (x1 > x2) ? x1 : x2;
                    }
                }
            }
        }
    }
}

__global__ void maximum_cuda(int64 size, float* py, float* px1, float* px2, int64 left1, int64 left2, int64 mid, int64 right1, int64 right2) {
    int64 n = blockIdx.x * blockDim.x + threadIdx.x;

    if (n < size) {
        int64 nr1 = n / (left2 * mid * right1 * right2);
        int64 nr2 = n / (mid * right1 * right2) % left2;
        int64 nm = n / (right1 * right2) % mid;
        int64 nc1 = n / right2 % right1;
        int64 nc2 = n % right2;

        int64 xpos1 = (nr1 * mid + nm) * right1 + nc1;
        int64 xpos2 = (nr2 * mid + nm) * right2 + nc2;

        float x1 = px1[xpos1];
        float x2 = px2[xpos2];

        py[n] = (x1 > x2) ? x1 : x2;
    }
}

void VMath::maximum(int device, float* py, float* px1, float* px2, int64 left1, int64 left2, int64 mid, int64 right1, int64 right2) {
    int64 size = left1 * left2 * mid * right1 * right2;
    CUDA_CALL(maximum, device, size, py, px1, px2, left1, left2, mid, right1, right2);
}

//--------------------------------------------------------------------------------------------------

__static__ void maximum_backward_x1_host(float* pgx, float* pgy, float* px1, float* px2, int64 left1, int64 left2, int64 mid, int64 right1, int64 right2) {
    // 무지막지한 5중루프 같지만 루프 크기 중 적어도 세 개가 1이다.
    for (int64 nr1 = 0, n = 0; nr1 < left1; nr1++) {
        for (int64 nm = 0; nm < mid; nm++) {
            for (int64 nc1 = 0; nc1 < right1; nc1++, n++) {
                float sum = 0;
                float x1 = px1[n];
                for (int64 nr2 = 0; nr2 < left2; nr2++) {
                    for (int64 nc2 = 0; nc2 < right2; nc2++) {
                        int64 xpos2 = (nr2 * mid + nm) * right2 + nc2;
                        int64 ypos = (((nr1 * left2 + nr2) * mid + nm) * right1 + nc1) * right2 + nc2;

                        float x2 = px2[xpos2];

                        if (x1 > x2) sum += pgy[ypos];
                    }
                }
                pgx[n] = sum;
            }
        }
    }
}

__global__ void maximum_backward_x1_cuda(int64 size, float* pgx, float* pgy, float* px1, float* px2, int64 left1, int64 left2, int64 mid, int64 right1, int64 right2) {
    int64 n = blockIdx.x * blockDim.x + threadIdx.x;

    if (n < size) {
        int64 nr1 = n / (mid * right1);
        int64 nm = n / right1 % mid;
        int64 nc1 = n % right1;

        float sum = 0;
        float x1 = px1[n];
        for (int64 nr2 = 0; nr2 < left2; nr2++) {
            for (int64 nc2 = 0; nc2 < right2; nc2++) {
                int64 xpos2 = (nr2 * mid + nm) * right2 + nc2;
                int64 ypos = (((nr1 * left2 + nr2) * mid + nm) * right1 + nc1) * right2 + nc2;

                float x2 = px2[xpos2];

                if (x1 > x2) sum += pgy[ypos];
            }
        }
        pgx[n] = sum;
    }
}

void VMath::maximum_backward_x1(int device, float* pgx, float* pgy, float* px1, float* px2, int64 left1, int64 left2, int64 mid, int64 right1, int64 right2) {
    int64 size = left1 * mid * right1;
    CUDA_CALL(maximum_backward_x1, device, size, pgx, pgy, px1, px2, left1, left2, mid, right1, right2);
}

//--------------------------------------------------------------------------------------------------

__static__ void maximum_backward_x2_host(float* pgx, float* pgy, float* px1, float* px2, int64 left1, int64 left2, int64 mid, int64 right1, int64 right2) {
    // 무지막지한 5중루프 같지만 루프 크기 중 적어도 세 개가 1이다.
    for (int64 nr2 = 0, n = 0; nr2 < left2; nr2++) {
        for (int64 nm = 0; nm < mid; nm++) {
            for (int64 nc2 = 0; nc2 < right2; nc2++, n++) {
                float sum = 0;
                float x2 = px2[n];
                for (int64 nr1 = 0; nr1 < left1; nr1++) {
                    for (int64 nc1 = 0; nc1 < right1; nc1++) {
                        int64 xpos1 = (nr1 * mid + nm) * right1 + nc1;
                        int64 ypos = (((nr1 * left2 + nr2) * mid + nm) * right1 + nc1) * right2 + nc2;

                        float x1 = px1[xpos1];

                        if (x1 <= x2) sum += pgy[ypos];
                    }
                }
                pgx[n] = sum;
            }
        }
    }
}

__global__ void maximum_backward_x2_cuda(int64 size, float* pgx, float* pgy, float* px1, float* px2, int64 left1, int64 left2, int64 mid, int64 right1, int64 right2) {
    int64 n = blockIdx.x * blockDim.x + threadIdx.x;

    if (n < size) {
        int64 nr2 = n / (mid * right2);
        int64 nm = n / right2 % mid;
        int64 nc2 = n % right2;

        float sum = 0;
        float x2 = px2[n];
        for (int64 nr1 = 0; nr1 < left1; nr1++) {
            for (int64 nc1 = 0; nc1 < right1; nc1++) {
                int64 xpos1 = (nr1 * mid + nm) * right1 + nc1;
                int64 ypos = (((nr1 * left2 + nr2) * mid + nm) * right1 + nc1) * right2 + nc2;

                float x1 = px1[xpos1];

                if (x1 <= x2) sum += pgy[ypos];
            }
        }
        pgx[n] = sum;
    }
}

void VMath::maximum_backward_x2(int device, float* pgx, float* pgy, float* px1, float* px2, int64 left1, int64 left2, int64 mid, int64 right1, int64 right2) {
    int64 size = left2 * mid * right2;
    CUDA_CALL(maximum_backward_x2, device, size, pgx, pgy, px1, px2, left1, left2, mid, right1, right2);
}

//--------------------------------------------------------------------------------------------------

__static__ void minimum_host(float* py, float* px1, float* px2, int64 left1, int64 left2, int64 mid, int64 right1, int64 right2) {
    // 무지막지한 5중루프 같지만 루프 크기 중 적어도 세 개가 1이다.
    for (int64 nr1 = 0, n = 0; nr1 < left1; nr1++) {
        for (int64 nr2 = 0; nr2 < left2; nr2++) {
            for (int64 nm = 0; nm < mid; nm++) {
                for (int64 nc1 = 0; nc1 < right1; nc1++) {
                    for (int64 nc2 = 0; nc2 < right2; nc2++, n++) {
                        int64 xpos1 = (nr1 * mid + nm) * right1 + nc1;
                        int64 xpos2 = (nr2 * mid + nm) * right2 + nc2;

                        float x1 = px1[xpos1];
                        float x2 = px2[xpos2];

                        py[n] = (x1 < x2) ? x1 : x2;
                    }
                }
            }
        }
    }
}

__global__ void minimum_cuda(int64 size, float* py, float* px1, float* px2, int64 left1, int64 left2, int64 mid, int64 right1, int64 right2) {
    int64 n = blockIdx.x * blockDim.x + threadIdx.x;

    if (n < size) {
        int64 nr1 = n / (left2 * mid * right1 * right2);
        int64 nr2 = n / (mid * right1 * right2) % left2;
        int64 nm = n / (right1 * right2) % mid;
        int64 nc1 = n / right2 % right1;
        int64 nc2 = n % right2;

        int64 xpos1 = (nr1 * mid + nm) * right1 + nc1;
        int64 xpos2 = (nr2 * mid + nm) * right2 + nc2;

        float x1 = px1[xpos1];
        float x2 = px2[xpos2];

        py[n] = (x1 < x2) ? x1 : x2;
    }
}

void VMath::minimum(int device, float* py, float* px1, float* px2, int64 left1, int64 left2, int64 mid, int64 right1, int64 right2) {
    int64 size = left1 * left2 * mid * right1 * right2;
    CUDA_CALL(minimum, device, size, py, px1, px2, left1, left2, mid, right1, right2);
}

//--------------------------------------------------------------------------------------------------

__static__ void minimum_backward_x1_host(float* pgx, float* pgy, float* px1, float* px2, int64 left1, int64 left2, int64 mid, int64 right1, int64 right2) {
    // 무지막지한 5중루프 같지만 루프 크기 중 적어도 세 개가 1이다.
    for (int64 nr1 = 0, n = 0; nr1 < left1; nr1++) {
        for (int64 nm = 0; nm < mid; nm++) {
            for (int64 nc1 = 0; nc1 < right1; nc1++, n++) {
                float sum = 0;
                float x1 = px1[n];
                for (int64 nr2 = 0; nr2 < left2; nr2++) {
                    for (int64 nc2 = 0; nc2 < right2; nc2++) {
                        int64 xpos2 = (nr2 * mid + nm) * right2 + nc2;
                        int64 ypos = (((nr1 * left2 + nr2) * mid + nm) * right1 + nc1) * right2 + nc2;

                        float x2 = px2[xpos2];

                        if (x1 < x2) sum += pgy[ypos];
                    }
                }
                pgx[n] = sum;
            }
        }
    }
}

__global__ void minimum_backward_x1_cuda(int64 size, float* pgx, float* pgy, float* px1, float* px2, int64 left1, int64 left2, int64 mid, int64 right1, int64 right2) {
    int64 n = blockIdx.x * blockDim.x + threadIdx.x;

    if (n < size) {
        int64 nr1 = n / (mid * right1);
        int64 nm = n / right1 % mid;
        int64 nc1 = n % right1;

        float sum = 0;
        float x1 = px1[n];
        for (int64 nr2 = 0; nr2 < left2; nr2++) {
            for (int64 nc2 = 0; nc2 < right2; nc2++) {
                int64 xpos2 = (nr2 * mid + nm) * right2 + nc2;
                int64 ypos = (((nr1 * left2 + nr2) * mid + nm) * right1 + nc1) * right2 + nc2;

                float x2 = px2[xpos2];

                if (x1 < x2) sum += pgy[ypos];
            }
        }
        pgx[n] = sum;
    }
}

void VMath::minimum_backward_x1(int device, float* pgx, float* pgy, float* px1, float* px2, int64 left1, int64 left2, int64 mid, int64 right1, int64 right2) {
    int64 size = left1 * mid * right1;
    CUDA_CALL(minimum_backward_x1, device, size, pgx, pgy, px1, px2, left1, left2, mid, right1, right2);
}

//--------------------------------------------------------------------------------------------------

__static__ void minimum_backward_x2_host(float* pgx, float* pgy, float* px1, float* px2, int64 left1, int64 left2, int64 mid, int64 right1, int64 right2) {
    // 무지막지한 5중루프 같지만 루프 크기 중 적어도 세 개가 1이다.
    for (int64 nr2 = 0, n = 0; nr2 < left2; nr2++) {
        for (int64 nm = 0; nm < mid; nm++) {
            for (int64 nc2 = 0; nc2 < right2; nc2++, n++) {
                float sum = 0;
                float x2 = px2[n];
                for (int64 nr1 = 0; nr1 < left1; nr1++) {
                    for (int64 nc1 = 0; nc1 < right1; nc1++) {
                        int64 xpos1 = (nr1 * mid + nm) * right1 + nc1;
                        int64 ypos = (((nr1 * left2 + nr2) * mid + nm) * right1 + nc1) * right2 + nc2;

                        float x1 = px1[xpos1];

                        if (x1 >= x2) sum += pgy[ypos];
                    }
                }
                pgx[n] = sum;
            }
        }
    }
}

__global__ void minimum_backward_x2_cuda(int64 size, float* pgx, float* pgy, float* px1, float* px2, int64 left1, int64 left2, int64 mid, int64 right1, int64 right2) {
    int64 n = blockIdx.x * blockDim.x + threadIdx.x;

    if (n < size) {
        int64 nr2 = n / (mid * right2);
        int64 nm = n / right2 % mid;
        int64 nc2 = n % right2;

        float sum = 0;
        float x2 = px2[n];
        for (int64 nr1 = 0; nr1 < left1; nr1++) {
            for (int64 nc1 = 0; nc1 < right1; nc1++) {
                int64 xpos1 = (nr1 * mid + nm) * right1 + nc1;
                int64 ypos = (((nr1 * left2 + nr2) * mid + nm) * right1 + nc1) * right2 + nc2;

                float x1 = px1[xpos1];

                if (x1 >= x2) sum += pgy[ypos];
            }
        }
        pgx[n] = sum;
    }
}

void VMath::minimum_backward_x2(int device, float* pgx, float* pgy, float* px1, float* px2, int64 left1, int64 left2, int64 mid, int64 right1, int64 right2) {
    int64 size = left2 * mid * right2;
    CUDA_CALL(minimum_backward_x2, device, size, pgx, pgy, px1, px2, left1, left2, mid, right1, right2);
}

//--------------------------------------------------------------------------------------------------

__static__ void _not_host(float* py, float* px, int64 ndat) {
    for (int64 n = 0; n < ndat; n++) {
        py[n] = (px[n] == 0) ? 1.0f : 0.0f;
    }
}

__global__ void _not_cuda(int64 size, float* py, float* px, int64 ndat) {
    int64 n = blockIdx.x * blockDim.x + threadIdx.x;

    if (n < size) {
        py[n] = (px[n] == 0) ? 1.0f : 0.0f;
    }
}

void VMath::_not(int device, float* py, float* px, int64 ndat) {
    int64 size = ndat;
    CUDA_CALL(_not, device, size, py, px, ndat);
}

//--------------------------------------------------------------------------------------------------

__static__ void _and_host(float* py, float* px1, float* px2, int64 left1, int64 left2, int64 mid, int64 right1, int64 right2) {
    // 무지막지한 5중루프 같지만 루프 크기 중 적어도 세 개가 1이다.
    for (int64 nr1 = 0, n = 0; nr1 < left1; nr1++) {
        for (int64 nr2 = 0; nr2 < left2; nr2++) {
            for (int64 nm = 0; nm < mid; nm++) {
                for (int64 nc1 = 0; nc1 < right1; nc1++) {
                    for (int64 nc2 = 0; nc2 < right2; nc2++, n++) {
                        int64 xpos1 = (nr1 * mid + nm) * right1 + nc1;
                        int64 xpos2 = (nr2 * mid + nm) * right2 + nc2;

                        py[n] = (px1[xpos1] != 0) && (px2[xpos2] != 0);
                    }
                }
            }
        }
    }
}

__global__ void _and_cuda(int64 size, float* py, float* px1, float* px2, int64 left1, int64 left2, int64 mid, int64 right1, int64 right2) {
    int64 n = blockIdx.x * blockDim.x + threadIdx.x;

    if (n < size) {
        int64 nr1 = n / (left2 * mid * right1 * right2);
        int64 nr2 = n / (mid * right1 * right2) % left2;
        int64 nm = n / (right1 * right2) % mid;
        int64 nc1 = n / right2 % right1;
        int64 nc2 = n % right2;

        int64 xpos1 = (nr1 * mid + nm) * right1 + nc1;
        int64 xpos2 = (nr2 * mid + nm) * right2 + nc2;

        py[n] = (px1[xpos1] != 0) && (px2[xpos2] != 0);
    }
}

void VMath::_and(int device, float* py, float* px1, float* px2, int64 left1, int64 left2, int64 mid, int64 right1, int64 right2) {
    int64 size = left1 * left2 * mid * right1 * right2;
    CUDA_CALL(_and, device, size, py, px1, px2, left1, left2, mid, right1, right2);
}

//--------------------------------------------------------------------------------------------------

__static__ void _or_host(float* py, float* px1, float* px2, int64 left1, int64 left2, int64 mid, int64 right1, int64 right2) {
    // 무지막지한 5중루프 같지만 루프 크기 중 적어도 세 개가 1이다.
    for (int64 nr1 = 0, n = 0; nr1 < left1; nr1++) {
        for (int64 nr2 = 0; nr2 < left2; nr2++) {
            for (int64 nm = 0; nm < mid; nm++) {
                for (int64 nc1 = 0; nc1 < right1; nc1++) {
                    for (int64 nc2 = 0; nc2 < right2; nc2++, n++) {
                        int64 xpos1 = (nr1 * mid + nm) * right1 + nc1;
                        int64 xpos2 = (nr2 * mid + nm) * right2 + nc2;

                        py[n] = (px1[xpos1] != 0) || (px2[xpos2] != 0);
                    }
                }
            }
        }
    }
}

__global__ void _or_cuda(int64 size, float* py, float* px1, float* px2, int64 left1, int64 left2, int64 mid, int64 right1, int64 right2) {
    int64 n = blockIdx.x * blockDim.x + threadIdx.x;

    if (n < size) {
        int64 nr1 = n / (left2 * mid * right1 * right2);
        int64 nr2 = n / (mid * right1 * right2) % left2;
        int64 nm = n / (right1 * right2) % mid;
        int64 nc1 = n / right2 % right1;
        int64 nc2 = n % right2;

        int64 xpos1 = (nr1 * mid + nm) * right1 + nc1;
        int64 xpos2 = (nr2 * mid + nm) * right2 + nc2;

        py[n] = (px1[xpos1] != 0) || (px2[xpos2] != 0);
    }
}

void VMath::_or(int device, float* py, float* px1, float* px2, int64 left1, int64 left2, int64 mid, int64 right1, int64 right2) {
    int64 size = left1 * left2 * mid * right1 * right2;
    CUDA_CALL(_or, device, size, py, px1, px2, left1, left2, mid, right1, right2);
}

//--------------------------------------------------------------------------------------------------

__static__ void equal_const_host(float* py, float* px, float val, int64 ndat) {
    for (int64 n = 0; n < ndat; n++) {
        py[n] = (px[n] == val) ? 1.0f : 0.0f;
    }
}

__global__ void equal_const_cuda(int64 size, float* py, float* px, float val, int64 ndat) {
    int64 n = blockIdx.x * blockDim.x + threadIdx.x;

    if (n < size) {
        py[n] = (px[n] == val) ? 1.0f : 0.0f;
    }
}

void VMath::equal_const(int device, float* py, float* px, float val, int64 ndat) {
    int64 size = ndat;
    CUDA_CALL(equal_const, device, size, py, px, val, ndat);
}

//--------------------------------------------------------------------------------------------------

__static__ void equal_host(float* py, float* px1, float* px2, int64 left1, int64 left2, int64 mid, int64 right1, int64 right2) {
    // 무지막지한 5중루프 같지만 루프 크기 중 적어도 세 개가 1이다.
    for (int64 nr1 = 0, n = 0; nr1 < left1; nr1++) {
        for (int64 nr2 = 0; nr2 < left2; nr2++) {
            for (int64 nm = 0; nm < mid; nm++) {
                for (int64 nc1 = 0; nc1 < right1; nc1++) {
                    for (int64 nc2 = 0; nc2 < right2; nc2++, n++) {
                        int64 xpos1 = (nr1 * mid + nm) * right1 + nc1;
                        int64 xpos2 = (nr2 * mid + nm) * right2 + nc2;

                        py[n] = (px1[xpos1] == px2[xpos2]) ? 1.0f : 0.0f;
                    }
                }
            }
        }
    }
}

__global__ void equal_cuda(int64 size, float* py, float* px1, float* px2, int64 left1, int64 left2, int64 mid, int64 right1, int64 right2) {
    int64 n = blockIdx.x * blockDim.x + threadIdx.x;

    if (n < size) {
        int64 nr1 = n / (left2 * mid * right1 * right2);
        int64 nr2 = n / (mid * right1 * right2) % left2;
        int64 nm = n / (right1 * right2) % mid;
        int64 nc1 = n / right2 % right1;
        int64 nc2 = n % right2;

        int64 xpos1 = (nr1 * mid + nm) * right1 + nc1;
        int64 xpos2 = (nr2 * mid + nm) * right2 + nc2;

        py[n] = (px1[xpos1] == px2[xpos2]) ? 1.0f : 0.0f;
    }
}

void VMath::equal(int device, float* py, float* px1, float* px2, int64 left1, int64 left2, int64 mid, int64 right1, int64 right2) {
    int64 size = left1 * left2 * mid * right1 * right2;
    CUDA_CALL(equal, device, size, py, px1, px2, left1, left2, mid, right1, right2);
}

//--------------------------------------------------------------------------------------------------

__static__ void greater_than_float_const_host(float* py, float* px, float val, int64 ndat) {
    for (int64 n = 0; n < ndat; n++) {
        py[n] = (px[n] > val) ? 1.0f : 0.0f;
    }
}

__global__ void greater_than_float_const_cuda(int64 size, float* py, float* px, float val, int64 ndat) {
    int64 n = blockIdx.x * blockDim.x + threadIdx.x;

    if (n < size) {
        py[n] = (px[n] > val) ? 1.0f : 0.0f;
    }
}

void VMath::greater_than_float_const(int device, float* py, float* px, float val, int64 ndat) {
    int64 size = ndat;
    CUDA_CALL(greater_than_float_const, device, size, py, px, val, ndat);
}

//--------------------------------------------------------------------------------------------------

__static__ void greater_than_int_const_host(float* py, int* px, int val, int64 ndat) {
    for (int64 n = 0; n < ndat; n++) {
        py[n] = (px[n] > val) ? 1.0f : 0.0f;
    }
}

__global__ void greater_than_int_const_cuda(int64 size, float* py, int* px, int val, int64 ndat) {
    int64 n = blockIdx.x * blockDim.x + threadIdx.x;

    if (n < size) {
        py[n] = (px[n] > val) ? 1.0f : 0.0f;
    }
}

void VMath::greater_than_int_const(int device, float* py, int* px, int val, int64 ndat) {
    int64 size = ndat;
    CUDA_CALL(greater_than_int_const, device, size, py, px, val, ndat);
}

//--------------------------------------------------------------------------------------------------

__static__ void greater_than_host(float* py, float* px1, float* px2, int64 left1, int64 left2, int64 mid, int64 right1, int64 right2) {
    // 무지막지한 5중루프 같지만 루프 크기 중 적어도 세 개가 1이다.
    for (int64 nr1 = 0, n = 0; nr1 < left1; nr1++) {
        for (int64 nr2 = 0; nr2 < left2; nr2++) {
            for (int64 nm = 0; nm < mid; nm++) {
                for (int64 nc1 = 0; nc1 < right1; nc1++) {
                    for (int64 nc2 = 0; nc2 < right2; nc2++, n++) {
                        int64 xpos1 = (nr1 * mid + nm) * right1 + nc1;
                        int64 xpos2 = (nr2 * mid + nm) * right2 + nc2;

                        py[n] = (px1[xpos1] > px2[xpos2]) ? 1.0f : 0.0f;
                    }
                }
            }
        }
    }
}

__global__ void greater_than_cuda(int64 size, float* py, float* px1, float* px2, int64 left1, int64 left2, int64 mid, int64 right1, int64 right2) {
    int64 n = blockIdx.x * blockDim.x + threadIdx.x;

    if (n < size) {
        int64 nr1 = n / (left2 * mid * right1 * right2);
        int64 nr2 = n / (mid * right1 * right2) % left2;
        int64 nm = n / (right1 * right2) % mid;
        int64 nc1 = n / right2 % right1;
        int64 nc2 = n % right2;

        int64 xpos1 = (nr1 * mid + nm) * right1 + nc1;
        int64 xpos2 = (nr2 * mid + nm) * right2 + nc2;

        py[n] = (px1[xpos1] > px2[xpos2]) ? 1.0f : 0.0f;
    }
}

void VMath::greater_than(int device, float* py, float* px1, float* px2, int64 left1, int64 left2, int64 mid, int64 right1, int64 right2) {
    int64 size = left1 * left2 * mid * right1 * right2;
    CUDA_CALL(greater_than, device, size, py, px1, px2, left1, left2, mid, right1, right2);
}

//--------------------------------------------------------------------------------------------------

__static__ void less_than_const_host(float* py, float* px, float val, int64 ndat) {
    for (int64 n = 0; n < ndat; n++) {
        py[n] = (px[n] < val) ? 1.0f : 0.0f;
    }
}

__global__ void less_than_const_cuda(int64 size, float* py, float* px, float val, int64 ndat) {
    int64 n = blockIdx.x * blockDim.x + threadIdx.x;

    if (n < size) {
        py[n] = (px[n] < val) ? 1.0f : 0.0f;
    }
}

void VMath::less_than_const(int device, float* py, float* px, float val, int64 ndat) {
    int64 size = ndat;
    CUDA_CALL(less_than_const, device, size, py, px, val, ndat);
}

//--------------------------------------------------------------------------------------------------

__static__ void less_than_host(float* py, float* px1, float* px2, int64 left1, int64 left2, int64 mid, int64 right1, int64 right2) {
    // 무지막지한 5중루프 같지만 루프 크기 중 적어도 세 개가 1이다.
    for (int64 nr1 = 0, n = 0; nr1 < left1; nr1++) {
        for (int64 nr2 = 0; nr2 < left2; nr2++) {
            for (int64 nm = 0; nm < mid; nm++) {
                for (int64 nc1 = 0; nc1 < right1; nc1++) {
                    for (int64 nc2 = 0; nc2 < right2; nc2++, n++) {
                        int64 xpos1 = (nr1 * mid + nm) * right1 + nc1;
                        int64 xpos2 = (nr2 * mid + nm) * right2 + nc2;

                        py[n] = (px1[xpos1] < px2[xpos2]) ? 1.0f : 0.0f;
                    }
                }
            }
        }
    }
}

__global__ void less_than_cuda(int64 size, float* py, float* px1, float* px2, int64 left1, int64 left2, int64 mid, int64 right1, int64 right2) {
    int64 n = blockIdx.x * blockDim.x + threadIdx.x;

    if (n < size) {
        int64 nr1 = n / (left2 * mid * right1 * right2);
        int64 nr2 = n / (mid * right1 * right2) % left2;
        int64 nm = n / (right1 * right2) % mid;
        int64 nc1 = n / right2 % right1;
        int64 nc2 = n % right2;

        int64 xpos1 = (nr1 * mid + nm) * right1 + nc1;
        int64 xpos2 = (nr2 * mid + nm) * right2 + nc2;

        py[n] = (px1[xpos1] < px2[xpos2]) ? 1.0f : 0.0f;
    }
}

void VMath::less_than(int device, float* py, float* px1, float* px2, int64 left1, int64 left2, int64 mid, int64 right1, int64 right2) {
    int64 size = left1 * left2 * mid * right1 * right2;
    CUDA_CALL(less_than, device, size, py, px1, px2, left1, left2, mid, right1, right2);
}

//--------------------------------------------------------------------------------------------------

__static__ void less_than_cross_host(float* py, float* px1, float* px2, int64 nrow, int64 ncol) {
    for (int64 nr = 0, n = 0; nr < nrow; nr++) {
        for (int64 nc1 = 0; nc1 < ncol; nc1++) {
            for (int64 nc2 = 0; nc2 < ncol; nc2++, n++) {
                int64 xpos1 = nr * ncol + nc1;
                int64 xpos2 = nr * ncol + nc2;

                float diff = px1[xpos1] - px2[xpos2];

                if (diff < 0) py[n] = 1.0f;
                else if (diff > 0) py[n] = 0.0f;
                else if (nc1 < nc2) py[n] = 1.0f;
                else py[n] = 1.0f;
            }
        }
    }
}

__global__ void less_than_cross_cuda(int64 size, float* py, float* px1, float* px2, int64 nrow, int64 ncol) {
    int64 n = blockIdx.x * blockDim.x + threadIdx.x;

    if (n < size) {
        int64 nr = n / (ncol * ncol);
        int64 nc1 = n / ncol % ncol;
        int64 nc2 = n % ncol;

        int64 xpos1 = nr * ncol + nc1;
        int64 xpos2 = nr * ncol + nc2;

        float diff = px1[xpos1] - px2[xpos2];

        if (diff < 0) py[n] = 1.0f;
        else if (diff > 0) py[n] = 0.0f;
        else if (nc1 < nc2) py[n] = 1.0f;
        else py[n] = 0.0f;
    }
}

void VMath::less_than_cross(int device, float* py, float* px1, float* px2, int64 nrow, int64 ncol) {
    int64 size = nrow * ncol * ncol;
    CUDA_CALL(less_than_cross, device, size, py, px1, px2, nrow, ncol);
}

//--------------------------------------------------------------------------------------------------

__static__ void greater_equal_const_host(float* py, float* px, float val, int64 ndat) {
    for (int64 n = 0; n < ndat; n++) {
        py[n] = (px[n] >= val) ? 1.0f : 0.0f;
    }
}

__global__ void greater_equal_const_cuda(int64 size, float* py, float* px, float val, int64 ndat) {
    int64 n = blockIdx.x * blockDim.x + threadIdx.x;

    if (n < size) {
        py[n] = (px[n] >= val) ? 1.0f : 0.0f;
    }
}

void VMath::greater_equal_const(int device, float* py, float* px, float val, int64 ndat) {
    int64 size = ndat;
    CUDA_CALL(greater_equal_const, device, size, py, px, val, ndat);
}

//--------------------------------------------------------------------------------------------------

__static__ void greater_equal_host(float* py, float* px1, float* px2, int64 left1, int64 left2, int64 mid, int64 right1, int64 right2) {
    // 무지막지한 5중루프 같지만 루프 크기 중 적어도 세 개가 1이다.
    for (int64 nr1 = 0, n = 0; nr1 < left1; nr1++) {
        for (int64 nr2 = 0; nr2 < left2; nr2++) {
            for (int64 nm = 0; nm < mid; nm++) {
                for (int64 nc1 = 0; nc1 < right1; nc1++) {
                    for (int64 nc2 = 0; nc2 < right2; nc2++, n++) {
                        int64 xpos1 = (nr1 * mid + nm) * right1 + nc1;
                        int64 xpos2 = (nr2 * mid + nm) * right2 + nc2;

                        py[n] = (px1[xpos1] >= px2[xpos2]) ? 1.0f : 0.0f;
                    }
                }
            }
        }
    }
}

__global__ void greater_equal_cuda(int64 size, float* py, float* px1, float* px2, int64 left1, int64 left2, int64 mid, int64 right1, int64 right2) {
    int64 n = blockIdx.x * blockDim.x + threadIdx.x;

    if (n < size) {
        int64 nr1 = n / (left2 * mid * right1 * right2);
        int64 nr2 = n / (mid * right1 * right2) % left2;
        int64 nm = n / (right1 * right2) % mid;
        int64 nc1 = n / right2 % right1;
        int64 nc2 = n % right2;

        int64 xpos1 = (nr1 * mid + nm) * right1 + nc1;
        int64 xpos2 = (nr2 * mid + nm) * right2 + nc2;

        py[n] = (px1[xpos1] >= px2[xpos2]) ? 1.0f : 0.0f;
    }
}

void VMath::greater_equal(int device, float* py, float* px1, float* px2, int64 left1, int64 left2, int64 mid, int64 right1, int64 right2) {
    int64 size = left1 * left2 * mid * right1 * right2;
    CUDA_CALL(greater_equal, device, size, py, px1, px2, left1, left2, mid, right1, right2);
}

//--------------------------------------------------------------------------------------------------

__static__ void less_equal_const_host(float* py, float* px, float val, int64 ndat) {
    for (int64 n = 0; n < ndat; n++) {
        py[n] = (px[n] <= val) ? 1.0f : 0.0f;
    }
}

__global__ void less_equal_const_cuda(int64 size, float* py, float* px, float val, int64 ndat) {
    int64 n = blockIdx.x * blockDim.x + threadIdx.x;

    if (n < size) {
        py[n] = (px[n] <= val) ? 1.0f : 0.0f;
    }
}

void VMath::less_equal_const(int device, float* py, float* px, float val, int64 ndat) {
    int64 size = ndat;
    CUDA_CALL(less_equal_const, device, size, py, px, val, ndat);
}

//--------------------------------------------------------------------------------------------------

__static__ void less_equal_host(float* py, float* px1, float* px2, int64 left1, int64 left2, int64 mid, int64 right1, int64 right2) {
    // 무지막지한 5중루프 같지만 루프 크기 중 적어도 세 개가 1이다.
    for (int64 nr1 = 0, n = 0; nr1 < left1; nr1++) {
        for (int64 nr2 = 0; nr2 < left2; nr2++) {
            for (int64 nm = 0; nm < mid; nm++) {
                for (int64 nc1 = 0; nc1 < right1; nc1++) {
                    for (int64 nc2 = 0; nc2 < right2; nc2++, n++) {
                        int64 xpos1 = (nr1 * mid + nm) * right1 + nc1;
                        int64 xpos2 = (nr2 * mid + nm) * right2 + nc2;

                        py[n] = (px1[xpos1] <= px2[xpos2]) ? 1.0f : 0.0f;
                    }
                }
            }
        }
    }
}

__global__ void less_equal_cuda(int64 size, float* py, float* px1, float* px2, int64 left1, int64 left2, int64 mid, int64 right1, int64 right2) {
    int64 n = blockIdx.x * blockDim.x + threadIdx.x;

    if (n < size) {
        int64 nr1 = n / (left2 * mid * right1 * right2);
        int64 nr2 = n / (mid * right1 * right2) % left2;
        int64 nm = n / (right1 * right2) % mid;
        int64 nc1 = n / right2 % right1;
        int64 nc2 = n % right2;

        int64 xpos1 = (nr1 * mid + nm) * right1 + nc1;
        int64 xpos2 = (nr2 * mid + nm) * right2 + nc2;

        py[n] = (px1[xpos1] <= px2[xpos2]) ? 1.0f : 0.0f;
    }
}

void VMath::less_equal(int device, float* py, float* px1, float* px2, int64 left1, int64 left2, int64 mid, int64 right1, int64 right2) {
    int64 size = left1 * left2 * mid * right1 * right2;
    CUDA_CALL(less_equal, device, size, py, px1, px2, left1, left2, mid, right1, right2);
}

//--------------------------------------------------------------------------------------------------

#ifdef TENSOR_CORE_TEST
#define WARP_SIZE 32

#define M 16
#define N 16
#define K 16

// GEMM configuration.
#define M_TILES 256
#define N_TILES 256
#define K_TILES 256

#define M_TOTAL (M * M_TILES)
#define N_TOTAL (N * N_TILES)
#define K_TOTAL (K * K_TILES)

//__global__ void WMMAINT8()
using namespace nvcuda;

__host__ void InitMatrix(half* A, half* B, float* C)
{
    for (int i = 0; i < M_TOTAL * K_TOTAL; i++)
        A[i] = __float2half(rand() % 1000 / 1000.0f);
    for (int i = 0; i < K_TOTAL * N_TOTAL; i++)
        B[i] = __float2half(rand() % 1000 / 1000.0f);
    for (int i = 0; i < M_TOTAL * N_TOTAL; i++)
        C[i] = rand() % 1000 / 1000.0f;
}

__global__ void WMMAF16TensorCore(half* A, half* B, float* C, float* D)
{
    int ix = (blockIdx.x * blockDim.x + threadIdx.x) / WARP_SIZE;
    int iy = (blockIdx.y * blockDim.y + threadIdx.y);

    wmma::fragment<wmma::matrix_a, M, N, K, half, wmma::row_major> a_frag;
    wmma::fragment<wmma::matrix_b, M, N, K, half, wmma::col_major> b_frag;
    wmma::fragment<wmma::accumulator, M, N, K, float> ab_frag;
    wmma::fragment<wmma::accumulator, M, N, K, float> c_frag;

    wmma::fill_fragment(ab_frag, 0.0f);

    // AB = A*B
    int a_col, a_row, b_col, b_row, c_col, c_row;
    a_row = ix * M;
    b_row = iy * N;
    for (int k = 0; k < K_TOTAL; k += K) {
        a_col = b_col = k;

        if (a_row < M_TOTAL && a_col < K_TOTAL && b_row < K_TOTAL && b_col < N_TOTAL) {
            // Load the inputs
            wmma::load_matrix_sync(a_frag, A + a_col + a_row * M_TOTAL, M_TOTAL);
            wmma::load_matrix_sync(b_frag, B + b_col + b_col * K_TOTAL, K_TOTAL);

            // Perform the matrix multiplication
            wmma::mma_sync(ab_frag, a_frag, b_frag, ab_frag);
        }
    }

    // D = AB + C
    c_col = b_row;
    c_row = a_row;
    if (c_row < M_TOTAL && c_col < N_TOTAL) {
        wmma::load_matrix_sync(c_frag, C + c_col + c_row * N_TOTAL, N_TOTAL, wmma::mem_row_major);

        for (int i = 0; i < c_frag.num_elements; i++) {
            c_frag.x[i] = ab_frag.x[i] + c_frag.x[i];
        }

        // Store the output
        wmma::store_matrix_sync(D + c_col + c_row * N_TOTAL, c_frag, N_TOTAL, wmma::mem_row_major);
    }
}

cudaError_t CalcWMMA(half* A, half* B, float* C, float* D)
{
    cudaError_t cuda_status;
    dim3 gridDim, blockDim;
    // 16 warps in one block
    blockDim.x = 4 * WARP_SIZE;
    blockDim.y = 4;

    gridDim.x = (M_TOTAL + (M * blockDim.x / WARP_SIZE - 1)) / (M * blockDim.x / WARP_SIZE);
    gridDim.y = (N_TOTAL + N * blockDim.y - 1) / (N * blockDim.y);

    // for Performance Metrics
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);

    WMMAF16TensorCore << <gridDim, blockDim >> > (A, B, C, D);
    cuda_status = cudaDeviceSynchronize();

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);

    // for Performance Metrics
    printf("[+] GPU(with Tensor Cores) Elapsed Time: %f ms\n", milliseconds);
    // references from https://devblogs.nvidia.com/how-implement-performance-metrics-cuda-cc/
    printf("[+] TFLOPS: %.2f\n", ((double)M_TOTAL * N_TOTAL * K_TOTAL * 2) / milliseconds / 1e9);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    return cuda_status;
}

int test_tensore_core_matmul() {
    cudaError_t cuda_status;
    cuda_status = cudaSetDevice(0);
    if (cuda_status != cudaSuccess) {
        printf("cudaSetDevice failed! ");
        return 1;
    }

    // Matrix on device
    half* A;
    half* B;
    float* C;
    float* D;

    // CUDA Unified Memory 
    cudaMallocManaged((void**)&A, sizeof(half) * M_TOTAL * K_TOTAL);
    cudaMallocManaged((void**)&B, sizeof(half) * K_TOTAL * N_TOTAL);
    cudaMallocManaged((void**)&C, sizeof(float) * M_TOTAL * N_TOTAL);
    cudaMallocManaged((void**)&D, sizeof(float) * M_TOTAL * N_TOTAL);

    // Init matrix A B C on host
    //InitHostMatrix(host_A, host_B, host_C);
    printf("[*] Initializing Matrix...\n");
    InitMatrix(A, B, C);
    printf("[+]   A: %d x %d\n", M_TOTAL, K_TOTAL);
    printf("[+]   B: %d x %d\n", K_TOTAL, N_TOTAL);
    printf("[+]   C: %d x %d\n", M_TOTAL, N_TOTAL);

    // computing gemm using tensor core
    printf("[*] Computing D = A * B +C with Tensor Cores...\n");
    // D = A * B +C, D holds the result after ret
    cuda_status = CalcWMMA(A, B, C, D);

    cuda_status = cudaDeviceReset();
    if (cuda_status != cudaSuccess) {
        printf("cudaDeviceReset failed! ");
        return 1;
    }
    // Todo: Add a function to verify the result by using the result of CPU version implementation.

    cudaFree(A);
    cudaFree(B);
    cudaFree(C);
    cudaFree(D);

    return 0;
}
#endif

//--------------------------------------------------------------------------------------------------

/*
__static__ void matmul_host(float* py, float* px, float* pw, int64 ndat, int64 xvec, int64 yvec, bool acc) {
    for (int64 nd = 0; nd < ndat; nd++) {
        for (int64 ny = 0; ny < yvec; ny++) {
            float dotsum = 0;
            for (int64 nx = 0; nx < xvec; nx++) {
                dotsum += px[nd * xvec + nx] * pw[ny * xvec + nx];
            }
            if (acc) py[nd * yvec + ny] += dotsum;
            else py[nd * yvec + ny] = dotsum;
        }
    }
}

__global__ void matmul_cuda(int64 size, float* py, float* px, float* pw, int64 ndat, int64 xvec, int64 yvec, bool acc) {
    int64 n = blockIdx.x * blockDim.x + threadIdx.x;

    if (n < size) {
        int64 nd = (n / yvec) * xvec;
        int64 ny = (n % yvec) * xvec;

        float dotsum = 0;

        for (int64 nx = 0; nx < xvec; nx++) {
            dotsum += px[nd++] * pw[ny++];
        }

        if (acc) py[n] += dotsum;
        else py[n] = dotsum;
    }
}

void VMath::matmul(int device, float* py, float* px, float* pw, int64 ndat, int64 xvec, int64 yvec, bool acc) {
#ifdef TENSOR_CORE_TEST
    int ret = test_tensore_core_matmul();
#endif
    int64 size = ndat * yvec;
    CUDA_CALL(matmul, device, size, py, px, pw, ndat, xvec, yvec, acc);
}

//--------------------------------------------------------------------------------------------------

__static__ void matmul_backward_x_host(float* pgx, float* pgy, float* pw, int64 ndat, int64 xvec, int64 yvec, bool acc) {
    for (int64 nd = 0; nd < ndat; nd++) {
        for (int64 nx = 0; nx < xvec; nx++) {
            float dotsum = 0;

            for (int64 ny = 0; ny < yvec; ny++) {
                dotsum += pgy[nd * yvec + ny] * pw[ny * xvec + nx];
            }
            if (acc) pgx[nd * xvec + nx] += dotsum;
            else pgx[nd * xvec + nx] = dotsum;
        }
    }
}

__global__ void matmul_backward_x_cuda(int64 size, float* pgx, float* pgy, float* pw, int64 ndat, int64 xvec, int64 yvec, bool acc) {
    int64 n = blockIdx.x * blockDim.x + threadIdx.x;

    if (n < size) {
        int64 nd = n / xvec;
        int64 nx = n % xvec;

        float dotsum = 0;

        for (int64 ny = 0; ny < yvec; ny++) {
            dotsum += pgy[nd * yvec + ny] * pw[ny * xvec + nx];
        }

        if (acc) pgx[n] += dotsum;
        else pgx[n] = dotsum;
    }
}

void VMath::matmul_backward_x(int device, float* pgx, float* pgy, float* pw, int64 ndat, int64 xvec, int64 yvec, bool acc) {
    int64 size = ndat * xvec;
    CUDA_CALL(matmul_backward_x, device, size, pgx, pgy, pw, ndat, xvec, yvec, acc);
}

//--------------------------------------------------------------------------------------------------

__static__ void matmul_backward_w_host(float* pgw, float* pgy, float* px, int64 ndat, int64 xvec, int64 yvec, bool acc) {
    for (int64 nx = 0; nx < xvec; nx++) {
        for (int64 ny = 0; ny < yvec; ny++) {
            float dotsum = 0;
            for (int64 nd = 0; nd < ndat; nd++) {
                dotsum += px[nd * xvec + nx] * pgy[nd * yvec + ny];
            }
            if (acc) pgw[ny * xvec + nx] += dotsum;
            else pgw[ny * xvec + nx] = dotsum;
        }
    }
}

__global__ void matmul_backward_w_cuda(int64 size, float* pgw, float* pgy, float* px, int64 ndat, int64 xvec, int64 yvec, bool acc) {
    int64 n = blockIdx.x * blockDim.x + threadIdx.x;

    if (n < size) {
        int64 ny = n / xvec;
        int64 nx = n % xvec;

        float dotsum = 0;
        for (int64 nd = 0; nd < ndat; nd++) {
            dotsum += px[nd * xvec + nx] * pgy[nd * yvec + ny];
        }
        if (acc) pgw[n] += dotsum;
        else  pgw[n] = dotsum;
    }
}

void VMath::matmul_backward_w(int device, float* pgw, float* pgy, float* px, int64 ndat, int64 xvec, int64 yvec, bool acc) {
    int64 size = xvec * yvec;
    CUDA_CALL(matmul_backward_w, device, size, pgw, pgy, px, ndat, xvec, yvec, acc);
}
*/

//--------------------------------------------------------------------------------------------------

__static__ void matmul_host(float* py, float* pw, float* px, int64 yvec, int64 ndat, int64 xvec, bool acc) {
    for (int64 nd = 0, n = 0; nd < ndat; nd++) {
        for (int64 ny = 0; ny < yvec; ny++, n++) {
            float dotsum = 0;

            int64 wpos = ny * xvec;
            int64 xpos = nd * xvec;

            for (int64 nx = 0; nx < xvec; nx++) {
                dotsum += pw[wpos++] * px[xpos++];
            }

            if (acc) py[n] += dotsum;
            else py[n] = dotsum;
        }
    }
}

__global__ void matmul_cuda(int64 size, float* py, float* pw, float* px, int64 yvec, int64 ndat, int64 xvec, bool acc) {
    int64 n = blockIdx.x * blockDim.x + threadIdx.x;

    if (n < size) {
        int64 nd = n / yvec;
        int64 ny = n % yvec;

        float dotsum = 0;

        int64 wpos = ny * xvec;
        int64 xpos = nd * xvec;

        for (int64 nx = 0; nx < xvec; nx++) {
            dotsum += pw[wpos++] * px[xpos++];
        }

        if (acc) py[n] += dotsum;
        else py[n] = dotsum;
    }
}

void VMath::matmul(int device, float* py, float* pw, float* px, int64 yvec, int64 ndat, int64 xvec, bool acc) {
    int64 size = ndat * yvec;
    CUDA_CALL(matmul, device, size, py, pw, px, yvec, ndat, xvec, acc);
}

//--------------------------------------------------------------------------------------------------

__static__ void matmul_backward_x_host(float* pgx, float* pgy, float* pw, int64 yvec, int64 ndat, int64 xvec, bool acc) {
    for (int64 nd = 0, n = 0; nd < ndat; nd++) {
        for (int64 nx = 0; nx < xvec; nx++, n++) {
            float dotsum = 0;

            for (int64 ny = 0; ny < yvec; ny++) {
                int64 wpos = ny * xvec + nx;
                int64 ypos = nd * yvec + ny;

                dotsum += pw[wpos] * pgy[ypos];
            }
            if (acc) pgx[n] += dotsum;
            else pgx[n] = dotsum;
        }
    }
}

__global__ void matmul_backward_x_cuda(int64 size, float* pgx, float* pgy, float* pw, int64 yvec, int64 ndat, int64 xvec, bool acc) {
    int64 n = blockIdx.x * blockDim.x + threadIdx.x;

    if (n < size) {
        int64 nd = n / xvec;
        int64 nx = n % xvec;

        float dotsum = 0;

        for (int64 ny = 0; ny < yvec; ny++) {
            int64 wpos = ny * xvec + nx;
            int64 ypos = nd * yvec + ny;

            dotsum += pw[wpos] * pgy[ypos];
        }
        if (acc) pgx[n] += dotsum;
        else pgx[n] = dotsum;
    }
}

void VMath::matmul_backward_x(int device, float* pgx, float* pgy, float* pw, int64 yvec, int64 ndat, int64 xvec, bool acc) {
    int64 size = ndat * xvec;
    CUDA_CALL(matmul_backward_x, device, size, pgx, pgy, pw, yvec, ndat, xvec, acc);
}

//--------------------------------------------------------------------------------------------------

__static__ void matmul_backward_w_host(float* pgw, float* pgy, float* px, int64 yvec, int64 ndat, int64 xvec, bool acc) {
    for (int64 ny = 0, n = 0; ny < yvec; ny++) {
        for (int64 nx = 0; nx < xvec; nx++, n++) {
            float dotsum = 0;

            for (int64 nd = 0; nd < ndat; nd++) {
                int64 xpos = nd * xvec + nx;
                int64 ypos = nd * yvec + ny;

                dotsum += px[xpos] * pgy[ypos];
            }

            if (acc) pgw[n] += dotsum;
            else pgw[n] = dotsum;
        }
    }
}

__global__ void matmul_backward_w_cuda(int64 size, float* pgw, float* pgy, float* px, int64 yvec, int64 ndat, int64 xvec, bool acc) {
    int64 n = blockIdx.x * blockDim.x + threadIdx.x;

    if (n < size) {
        int64 ny = n / xvec;
        int64 nx = n % xvec;

        float dotsum = 0;

        for (int64 nd = 0; nd < ndat; nd++) {
            int64 xpos = nd * xvec + nx;
            int64 ypos = nd * yvec + ny;

            dotsum += px[xpos] * pgy[ypos];
        }

        if (acc) pgw[n] += dotsum;
        else  pgw[n] = dotsum;
    }
}

void VMath::matmul_backward_w(int device, float* pgw, float* pgy, float* px, int64 yvec, int64 ndat, int64 xvec, bool acc) {
    int64 size = yvec * xvec;
    CUDA_CALL(matmul_backward_w, device, size, pgw, pgy, px, yvec, ndat, xvec, acc);
}

//--------------------------------------------------------------------------------------------------

__static__ void activate_host(float* py, float* px, int64 nrow, int64 ncol, int actFunc, HYPER hLeakyAlpha) {
    float leaky_alpha = HYPER_ACCESS(hLeakyAlpha);

    for (int64 nr = 0, n = 0; nr < nrow; nr++) {
        for (int64 nc = 0; nc < ncol; nc++, n++) {
            float x = px[n];
            float y, t;

            switch ((ActFunc)actFunc) {
            case ActFunc::none:
                y = x;
                break;
            case ActFunc::relu:
                y = (x > 0) ? x : 0;
                break;
            case ActFunc::leaky:
                y = (x > 0) ? x : x * leaky_alpha;
                break;
            case ActFunc::sigmoid:
                y = _sigmoid_host(x);
                break;
            case ActFunc::tanh:
                y = _tanh_host(x);
                break;
            case ActFunc::gelu:
                y = x * 0.5f * (1.0f + ::erff(x / ::sqrtf(2.0f)));
                break;
            case ActFunc::selu:
            {
                float lambda = 1.0507009f;
                float alpha = 1.6732632f;

                y = (x > 0) ? lambda * x : lambda * alpha * (::expf(x) - 1);
            }
                break;
            case ActFunc::mish:
                t = ::logf(1.0f + ::expf(x)) * 2;
                t = (float)((t > 0) ? ((1.0f - ::expf(-t)) / (1.0f + ::expf(-t))) : ((::expf(t) - 1.0f) / (::expf(t) + 1.0f)));
                y = x * t;
                break;
            case ActFunc::swish:
                t = (float)((x > 0) ? (1.0f / (1.0f + ::expf(-x))) : (::expf(x) / (::expf(x) + 1.0f)));
                y = x * t;
                break;
            case ActFunc::softmax:
            {
                float max_term = -FLT_MAX;
                float exp_sum = 0;

                float* pp = px - (n % ncol);

                for (int64 nn = 0; nn < ncol; nn++) {
                    if (pp[nn] > max_term) max_term = pp[nn];
                }

                for (int64 nn = 0; nn < ncol; nn++) {
                    exp_sum += ::expf(pp[nn] - max_term);
                }

                y = ::expf(x - max_term) / exp_sum;
            }
            break;
            default:
                VP_THROW(VERR_CONDITIONAL_STATEMENT);
                break;
            }

            py[n] = y;
        }
    }
}

__global__ void activate_cuda(int64 size, float* py, float* px, int64 nrow, int64 ncol, int actFunc, HYPER hLeakyAlpha) {
    int64 n = blockIdx.x * blockDim.x + threadIdx.x;

    if (n < size) {
        float x = px[n];
        float y, t;
        float leaky_alpha = HYPER_ACCESS(hLeakyAlpha);

        switch ((ActFunc)actFunc) {
        case ActFunc::none:
            y = x;
            break;
        case ActFunc::relu:
            y = (x > 0) ? x : 0;
            break;
        case ActFunc::leaky:
            y = (x > 0) ? x : x * leaky_alpha;
            break;
        case ActFunc::sigmoid:
            y = _sigmoid_cuda(x);
            break;
        case ActFunc::tanh:
            y = _tanh_cuda(x);
            break;
        case ActFunc::gelu:
            y = x * 0.5f * (1.0f + ::erff(x / ::sqrtf(2.0f)));
            break;
        case ActFunc::selu:
        {
            float lambda = 1.0507009f;
            float alpha = 1.6732632f;

            y = (x > 0) ? lambda * x : lambda * alpha * (::expf(x) - 1);
        }
        break;
        case ActFunc::mish:
            t = ::logf(1.0f + ::expf(x)) * 2;
            t = (float)((t > 0) ? ((1.0f - ::expf(-t)) / (1.0f + ::expf(-t))) : ((::expf(t) - 1.0f) / (::expf(t) + 1.0f)));
            y = x * t;
            break;
        case ActFunc::swish:
            t = (float)((x > 0) ? (1.0f / (1.0f + ::expf(-x))) : (::expf(x) / (::expf(x) + 1.0f)));
            y = x * t;
            break;
        case ActFunc::softmax:
        {
            float max_term = -FLT_MAX;
            float exp_sum = 0;

            float* pp = px - (n % ncol);

            for (int64 nn = 0; nn < ncol; nn++) {
                if (pp[nn] > max_term) max_term = pp[nn];
            }

            for (int64 nn = 0; nn < ncol; nn++) {
                exp_sum += ::expf(pp[nn] - max_term);
            }

            y = ::expf(x - max_term) / exp_sum;
        }
            break;
        default:
            if (n == 0) assert(0);
            break;
        }

        py[n] = y;
    }
}

void VMath::activate(int device, float* py, float* px, int64 nrow, int64 ncol, int actFunc, HYPER hLeakyAlpha) {
    int64 size = nrow * ncol;
    CUDA_CALL(activate, device, size, py, px, nrow, ncol, actFunc, hLeakyAlpha);
}

//--------------------------------------------------------------------------------------------------

__static__ void activate_backward_host(float* pgx, float* pgy, float* px, int64 nrow, int64 ncol, int actFunc, HYPER hLeakyAlpha) {
    float leaky_alpha = HYPER_ACCESS(hLeakyAlpha);

    for (int64 nr = 0, n = 0; nr < nrow; nr++) {
        for (int64 nc = 0; nc < ncol; nc++, n++) {
            float gy = pgy[n];
            float x = px[n];
            float gx, y, s, t, u, v;

            switch ((ActFunc)actFunc) {
            case ActFunc::none:
                gx = 1.0f;
                break;
            case ActFunc::relu:
                gx = (x > 0) ? 1.0f : 0.0f;
                break;
            case ActFunc::leaky:
                gx = (x > 0) ? 1.0f : leaky_alpha;
                break;
            case ActFunc::sigmoid:
                y = _sigmoid_host(x);
                gx = y * (1.0f - y);
                break;
            case ActFunc::tanh:
                y = _tanh_host(x);
                gx = 1.0f - y * y;
                break;
            case ActFunc::gelu:
                s = 0.5f * (1.0f + ::erff(x / ::sqrtf(2.0f)));
                t = 0.5f * x * ::sqrtf(2.0f / (float)PI_L) * ::expf(-x * x / 2.0f);
                gx = s + t;
                break;
            case ActFunc::selu:
            {
                float lambda = 1.0507009f;
                float alpha = 1.6732632f;

                gx = (x > 0) ? lambda : lambda * alpha * ::expf(x);
            }
            break;
            case ActFunc::mish:
                s = ::logf(1.0f + ::expf(x));                                                                                   // s = softplus(x)
                t = (x > 0) ? (1.0f / (1.0f + ::expf(-x))) : (::expf(x) / (1.0f + ::expf(x)));                                  // t = softplus'(x)
                s = s * 2.0f;
                u = (float)((s > 0) ? ((1.0f - ::expf(-s)) / (1.0f + ::expf(-s))) : ((::expf(s) - 1.0f) / (::expf(s) + 1.0f))); // u = tanh(softplus(x))
                v = (1.0f - u * u);                                                                                             // w = tan'(softplus(x))
                gx = u + x * v * t;
                break;
            case ActFunc::swish:
                s = _sigmoid_host(x);
                t = s * (1.0f - s);
                gx = s + x * t;
                break;
            default:
                VP_THROW(VERR_CONDITIONAL_STATEMENT);
                break;
            }

            pgx[n] = gx * gy;
        }
    }
}

__global__ void activate_backward_cuda(int64 size, float* pgx, float* pgy, float* px, int64 nrow, int64 ncol, int actFunc, HYPER hLeakyAlpha) {
    int64 n = blockIdx.x * blockDim.x + threadIdx.x;

    if (n < size) {
        float gy = pgy[n];
        float x = px[n];
        float gx, y, s, t, u, v;
        float leaky_alpha = HYPER_ACCESS(hLeakyAlpha);

        switch ((ActFunc)actFunc) {
        case ActFunc::none:
            gx = 1.0f;
            break;
        case ActFunc::relu:
            gx = (x > 0) ? 1.0f : 0.0f;
            break;
        case ActFunc::leaky:
            gx = (x > 0) ? 1.0f : leaky_alpha;
            break;
        case ActFunc::sigmoid:
            y = _sigmoid_cuda(x);
            gx = y * (1.0f - y);
            break;
        case ActFunc::tanh:
            y = _tanh_cuda(x);
            gx = 1.0f - y * y;
            break;
        case ActFunc::gelu:
            s = 0.5f * (1.0 + ::erff(x / ::sqrtf(2.0f)));
            t = 0.5f * x * ::sqrtf(2.0f / (float)PI_L) * ::expf(-x * x / 2.0f);
            gx = s + t;
            break;
        case ActFunc::selu:
        {
            float lambda = 1.0507009f;
            float alpha = 1.6732632f;

            gx = (x > 0) ? lambda : lambda * alpha * ::expf(x);
        }
        break;
        case ActFunc::mish:
            s = ::logf(1.0f + ::expf(x));                                                                                   // s = softplus(x)
            t = (x > 0) ? (1.0f / (1.0f + ::expf(-x))) : (::expf(x) / (1.0f + ::expf(x)));                                  // t = softplus'(x)
            s = s * 2.0f;
            u = (float)((s > 0) ? ((1.0f - ::expf(-s)) / (1.0f + ::expf(-s))) : ((::expf(s) - 1.0f) / (::expf(s) + 1.0f))); // u = tanh(softplus(x))
            v = (1.0f - u * u);                                                                                                // w = tan'(softplus(x))
            gx = u + x * v * t;
            break;
        case ActFunc::swish:
            s = _sigmoid_cuda(x);
            t = s * (1.0f - s);
            gx = s + x * t;
            break;
        case ActFunc::softmax:
        {
            float max_term = -FLT_MAX;
            float exp_sum = 0;
            float Gx = 0;

            int64 nth = n % ncol;

            float* pp = px - nth;
            float* ppgy = pgy - nth;

            for (int64 nn = 0; nn < ncol; nn++) {
                if (pp[nn] > max_term) max_term = pp[nn];
            }

            for (int64 nn = 0; nn < ncol; nn++) {
                exp_sum += ::expf(pp[nn] - max_term);
            }

            float y_i = ::expf(x - max_term) / exp_sum;

            for (int64 nj = 0; nj < ncol; nj++) {
                float y_j = ::expf(pp[nj] - max_term) / exp_sum;
                float round_y_ij = - y_i * y_j;
                if (nj == nth) round_y_ij += y_i;
                Gx += round_y_ij * ppgy[nj];
            }

            pgx[n] = Gx;
            return;
        }
        break;
        default:
            if (n == 0) assert(0);
            break;
        }

        pgx[n] = gx * gy;
    }
}

void VMath::activate_backward(int device, float* pgx, float* pgy, float* px, int64 nrow, int64 ncol, int actFunc, HYPER hLeakyAlpha) {
    int64 size = nrow * ncol;
    CUDA_CALL(activate_backward, device, size, pgx, pgy, px, nrow, ncol, actFunc, hLeakyAlpha);
}

//--------------------------------------------------------------------------------------------------

__static__ void activate_backward_with_y_host(float* pgx, float* pgy, float* py, int64 nrow, int64 ncol, int actFunc, HYPER hLeakyAlpha) {
    float leaky_alpha = HYPER_ACCESS(hLeakyAlpha);

    for (int64 nd = 0, n = 0; nd < nrow; nd++) {
        for (int64 nc = 0; nc < ncol; nc++, n++) {
            float gy = pgy[n];
            float y = py[n];
            float gx;

            switch ((ActFunc)actFunc) {
            case ActFunc::none:
                gx = 1.0f;
                break;
            case ActFunc::relu:
                gx = (y > 0) ? 1.0f : 0;
                break;
            case ActFunc::leaky:
                gx = (y > 0) ? 1.0f : leaky_alpha;
                break;
            case ActFunc::sigmoid:
                gx = y * (1 - y);
                break;
            case ActFunc::tanh:
                gx = 1 - y * y;
                break;
            default:
                VP_THROW(VERR_CONDITIONAL_STATEMENT);
                break;
            }

            pgx[n] = gx * gy;
        }
    }
}

__global__ void activate_backward_with_y_cuda(int64 size, float* pgx, float* pgy, float* py, int64 nrow, int64 ncol, int actFunc, HYPER hLeakyAlpha) {
    int64 n = blockIdx.x * blockDim.x + threadIdx.x;

    if (n < size) {
        //int64 nr = n / ncol;
        //int64 nc = n % ncol;

        float gy = pgy[n];
        float y = py[n];
        float gx;
        float leaky_alpha = HYPER_ACCESS(hLeakyAlpha);

        switch ((ActFunc)actFunc) {
        case ActFunc::none:
            gx = 1.0f;
            break;
        case ActFunc::relu:
            gx = (y > 0) ? 1.0f : 0;
            break;
        case ActFunc::leaky:
            gx = (y > 0) ? 1.0f : leaky_alpha;
            break;
        case ActFunc::sigmoid:
            gx = y * (1 - y);
            break;
        case ActFunc::tanh:
            gx = 1 - y * y;
            break;
        default:
            if (n == 0) assert(0);
            break;
        }

        pgx[n] = gx * gy;
    }
}

void VMath::activate_backward_with_y(int device, float* pgx, float* pgy, float* py, int64 nrow, int64 ncol, int actFunc, HYPER hLeakyAlpha) {
    int64 size = nrow * ncol;
    CUDA_CALL(activate_backward_with_y, device, size, pgx, pgy, py, nrow, ncol, actFunc, hLeakyAlpha);
}

//--------------------------------------------------------------------------------------------------

__static__ void conv2d_host(float* py, float* px, float* pk, int64 ndat,
    int64 ychn, int64 yh, int64 yw,
    int64 xchn, int64 xh, int64 xw,
    int64 kh, int64 kw, int64 group,
    int64 ph, int64 pw, int pmode) {
    int64 kchn = xchn / group;

    for (int64 nd = 0, n = 0; nd < ndat; nd++) {
        for (int64 yn = 0; yn < ychn; yn++) {
            for (int64 yr = 0; yr < yh; yr++) {
                for (int64 yc = 0; yc < yw; yc++, n++) {
                    float sum = 0;

                    for (int64 kr = 0; kr < kh; kr++) {
                        for (int64 kc = 0; kc < kw; kc++) {
                            int64 xr = yr + (kr - ph);
                            int64 xc = yc + (kc - pw);

                            if (xr < 0 || xr >= xh || xc < 0 || xc >= xw) {
                                if (pmode == 0) continue;   // means 'zero' padding
                                continue; // rest case not implemented yet, 구현될 때까지 임시로 텐서 호출단에서 차단 처리함
                            }

                            for (int64 xn = 0; xn < xchn; xn++) {
                                int64 xpos = ((nd * xchn + xn) * xh + xr) * xw + xc;
                                int64 kpos = ((yn * kchn + (xn / group)) * kh + kr) * kw + kc;

                                sum += px[xpos] * pk[kpos];
                            }
                        }
                    }

                    py[n] = sum;
                }
            }
        }
    }
}

__global__ void conv2d_cuda(
    int64 size, float* py, float* px, float* pk, int64 ndat,
    int64 ychn, int64 yh, int64 yw,
    int64 xchn, int64 xh, int64 xw,
    int64 kh, int64 kw, int64 group,
    int64 ph, int64 pw, int pmode) {
    int64 n = blockIdx.x * blockDim.x + threadIdx.x;

    if (n < size) {
        int64 nd = n / (ychn * yh * yw);
        int64 yn = n / (yh * yw) % ychn;
        int64 yr = n / yw % xh;
        int64 yc = n % yw;

        int64 kchn = xchn / group;

        float sum = 0;

        for (int64 kr = 0; kr < kh; kr++) {

            for (int64 kc = 0; kc < kw; kc++) {
                int64 xr = yr + (kr - ph);
                int64 xc = yc + (kc - pw);

                if (xr < 0 || xr >= xh || xc < 0 || xc >= xw) {
                    if (pmode == 0) continue;   // means 'zero' padding
                    continue; // rest case not implemented yet, 구현될 때까지 임시로 텐서 호출단에서 차단 처리함
                }

                for (int64 xn = 0; xn < xchn; xn++) {
                    int64 xpos = ((nd * xchn + xn) * xh + xr) * xw + xc;
                    int64 kpos = ((yn * kchn + (xn/ group)) * kh + kr) * kw + kc;

                    sum += px[xpos] * pk[kpos];
                }
            }
        }

        py[n] = sum;
    }
}

void VMath::conv2d(
    int device, float* py, float* px, float* pk, int64 ndat,  int64 ychn, int64 yh, int64 yw, int64 xchn, int64 xh, int64 xw,
    int64 kh, int64 kw, int64 group, int64 ph, int64 pw, int pmode) {
    int64 size = ndat * ychn * yh * yw;
    CUDA_CALL(conv2d, device, size, py, px, pk, ndat, ychn, yh, yw, xchn, xh, xw, kh, kw, group, ph, pw, pmode);
}

//--------------------------------------------------------------------------------------------------

__static__ void conv2d_backward_x_host(float* pgx, float* pgy, float* pk, int64 ndat, int64 xchn, int64 xh, int64 xw,
    int64 ychn, int64 yh, int64 yw, int64 kh, int64 kw, int64 group, int64 ph, int64 pw) {
    int64 kchn = xchn / group;

    for (int64 nd = 0, n = 0; nd < ndat; nd++) {
        for (int64 xn = 0; xn < xchn; xn++) {
            for (int64 xr = 0; xr < xh; xr++) {
                for (int64 xc = 0; xc < xw; xc++, n++) {
                    float sum = 0;

                    for (int64 kr = 0; kr < kh; kr++) {
                        int64 yr = xr - (kr - ph);
                        if (yr < 0 || yr >= yh) continue;

                        for (int64 kc = 0; kc < kw; kc++) {
                            int64 yc = xc - (kc - pw);
                            if (yc < 0 || yc >= yw) continue;

                            for (int64 yn = 0; yn < ychn; yn++) {
                                int64 ypos = ((nd * ychn + yn) * yh + yr) * yw + yc;
                                int64 kpos = ((yn * kchn + (xn / group)) * kh + kr) * kw + kc;

                                sum += pgy[ypos] * pk[kpos];
                            }
                        }
                    }

                    pgx[n] = sum;
                }
            }
        }
    }
}

__global__ void conv2d_backward_x_cuda(int64 size, float* pgx, float* pgy, float* pk, int64 ndat, int64 xchn, int64 xh, int64 xw,
    int64 ychn, int64 yh, int64 yw, int64 kh, int64 kw, int64 group, int64 ph, int64 pw) {
    int64 n = blockIdx.x * blockDim.x + threadIdx.x;

    if (n < size) {
        int64 nd = n / (xchn * xh * xw);
        int64 xn = n / (xh * xw) % xchn;
        int64 xr = n / xw % xh;
        int64 xc = n % xw;

        int64 kchn = xchn / group;

        float sum = 0;

        for (int64 kr = 0; kr < kh; kr++) {
            int64 yr = xr - (kr - ph);
            if (yr < 0 || yr >= yh) continue;

            for (int64 kc = 0; kc < kw; kc++) {
                int64 yc = xc - (kc - pw);
                if (yc < 0 || yc >= yw) continue;

                for (int64 yn = 0; yn < ychn; yn++) {
                    int64 ypos = ((nd * ychn + yn) * yh + yr) * yw + yc;
                    int64 kpos = ((yn * kchn + (xn / group)) * kh + kr) * kw + kc;

                    sum += pgy[ypos] * pk[kpos];
                }
            }
        }

        pgx[n] = sum;
    }
}

void VMath::conv2d_backward_x(
    int device, float* pgx, float* pgy, float* pk, int64 ndat, int64 xchn, int64 xh, int64 xw,
    int64 ychn, int64 yh, int64 yw, int64 kh, int64 kw, int64 group, int64 ph, int64 pw) {
    int64 size = ndat * xchn * xh * xw;

    CUDA_CALL(conv2d_backward_x, device, size, pgx, pgy, pk, ndat, xchn, xh, xw, ychn, yh, yw, kh, kw, group, ph, pw);
}

//--------------------------------------------------------------------------------------------------

__static__ void conv2d_backward_k_host(
    float* pgk, float* pgy, float* px, int64 ndat, int64 xchn, int64 xh, int64 xw,
    int64 ychn, int64 yh, int64 yw, int64 kh, int64 kw, int64 group, int64 ph, int64 pw) {
    int64 kchn = xchn / group;

    for (int64 kr = 0, n = 0; kr < kh; kr++) {
        for (int64 kc = 0; kc < kw; kc++) {
            for (int64 kn = 0; kn < kchn; kn++) {
                for (int64 yn = 0; yn < ychn; yn++, n++) {
                    float sum = 0;

                    for (int64 xr = 0; xr < xh; xr++) {
                        int64 yr = xr - (kr - ph);
                        if (yr < 0 || yr >= yh) continue;

                        for (int64 xc = 0; xc < xw; xc++) {
                            int64 yc = xc - (kc - yw);
                            if (yc < 0 || yc >= yw) continue;

                            for (int64 nd = 0; nd < ndat; nd++) {
                                for (int64 ng = 0; ng < group; ng++) {
                                    int64 xn = kn * group + ng;
                                    int64 ypos = ((nd * ychn + yn) * yh + yr) * yw + yc;
                                    int64 xpos = ((nd * xchn + xn) * xh + xr) * xw + xc;

                                    sum += pgy[ypos] * px[xpos];
                                }
                            }
                        }
                    }

                    int64 kpos = ((yn * kchn + kn) * kh + kr) * kw + kc;
                    pgk[kpos] = sum;
                }
            }
        }
    }
}

__global__ void conv2d_backward_k_cuda(
    int64 size, float* pgk, float* pgy, float* px, int64 ndat, int64 xchn, int64 xh, int64 xw,
    int64 ychn, int64 yh, int64 yw, int64 kh, int64 kw, int64 group, int64 ph, int64 pw) {
    int64 n = blockIdx.x * blockDim.x + threadIdx.x;

    if (n < size) {
        int64 kchn = xchn / group;

        int64 yn = n / (kchn * kh * kw);
        int64 kn = n / (kh * kw) % kchn;
        int64 kr = n / kw % kh;
        int64 kc = n % kw;

        float sum = 0;

        for (int64 xr = 0; xr < xh; xr++) {
            int64 yr = xr - (kr - ph);
            if (yr < 0 || yr >= yh) continue;

            for (int64 xc = 0; xc < xw; xc++) {
                int64 yc = xc - (kc - pw);
                if (yc < 0 || yc >= yw) continue;

                for (int64 nd = 0; nd < ndat; nd++) {
                    for (int64 ng = 0; ng < group; ng++) {
                        int64 xn = kn * group + ng;
                        int64 ypos = ((nd * ychn + yn) * yh + yr) * yw + yc;
                        int64 xpos = ((nd * xchn + xn) * xh + xr) * xw + xc;

                        sum += pgy[ypos] * px[xpos];
                    }
                }
            }
        }

        pgk[n] = sum;
    }
}

void VMath::conv2d_backward_k(
    int device, float* pgk, float* pgy, float* px, int64 ndat, int64 xchn, int64 xh, int64 xw,
    int64 ychn, int64 yh, int64 yw, int64 kh, int64 kw, int64 group, int64 ph, int64 pw) {
    int64 kchn = xchn / group;
    int64 size = kh * kw * kchn * ychn;

    CUDA_CALL(conv2d_backward_k, device, size, pgk, pgy, px, ndat, xchn, xh, xw, ychn, yh, yw, kh, kw, group, ph, pw);
}

//--------------------------------------------------------------------------------------------------

__static__ void conv2d_transposed_host(float* py, float* px, float* pk, int64 ndat, int64 xchn, int64 xh, int64 xw, int64 ychn, int64 kh, int64 kw, int64 sh, int64 sw) {
    int64 yh = xh * sh;
    int64 yw = xw * sw;

    for (int64 nd = 0, n = 0; nd < ndat; nd++) {
        for (int64 yn = 0; yn < ychn; yn++) {
            for (int64 yr = 0; yr < yh; yr++) {
                for (int64 yc = 0; yc < yw; yc++, n++) {
                    float sum = 0;

                    int64 bh = (kh - 1) / 2, bw = (kw - 1) / 2;

                    for (int64 kr = 0; kr < kh; kr++) {
                        int64 xr = (yr - (kr - bh) - (sh - 1) / 2) / sh;
                        if (xr < 0 || xr >= xh) continue;

                        for (int64 kc = 0; kc < kw; kc++) {
                            int64 xc = (yc - (kc - bw) - (sw - 1) / 2) / sw;
                            if (xc < 0 || xc >= xw) continue;

                            for (int64 xn = 0; xn < xchn; xn++) {
                                int64 xpos = ((nd * xchn + xn) * xh + xr) * xw + xc;
                                int64 kpos = ((yn * xchn + xn) * kh + kr) * kw + kc;

                                sum += px[xpos] * pk[kpos];
                            }
                        }
                    }

                    py[n] = sum;
                }
            }
        }
    }
}

__global__ void conv2d_transposed_cuda(int64 size, float* py, float* px, float* pk, int64 ndat, int64 xchn, int64 xh, int64 xw, int64 ychn, int64 kh, int64 kw, int64 sh, int64 sw) {
    int64 n = blockIdx.x * blockDim.x + threadIdx.x;

    if (n < size) {
        int64 yh = xh * sh;
        int64 yw = xw * sw;

        int64 nd = n / (ychn * yh * yw);
        int64 yn = n / (yh * yw) % ychn;
        int64 yr = n / yw % yh;
        int64 yc = n % yw;

        float sum = 0;

        int64 bh = (kh - 1) / 2, bw = (kw - 1) / 2;

        for (int64 kr = 0; kr < kh; kr++) {
            int64 xr = (yr - (kr - bh) - (sh - 1) / 2) / sh;
            if (xr < 0 || xr >= xh) continue;

            for (int64 kc = 0; kc < kw; kc++) {
                int64 xc = (yc - (kc - bw) - (sw - 1) / 2) / sw;
                if (xc < 0 || xc >= xw) continue;

                for (int64 xn = 0; xn < xchn; xn++) {
                    int64 xpos = ((nd * xchn + xn) * xh + xr) * xw + xc;
                    int64 kpos = ((yn * xchn + xn) * kh + kr) * kw + kc;

                    sum += px[xpos] * pk[kpos];
                }
            }
        }

        py[n] = sum;
    }
}

void VMath::conv2d_transposed(int device, float* py, float* px, float* pk, int64 ndat, int64 xchn, int64 xh, int64 xw, int64 ychn, int64 kh, int64 kw, int64 sh, int64 sw) {
    int64 size = ndat * ychn * xh * sh * xw * sw;
    CUDA_CALL(conv2d_transposed, device, size, py, px, pk, ndat, xchn, xh, xw, ychn, kh, kw, sh, sw);
}

//--------------------------------------------------------------------------------------------------

__static__ void conv2d_transposed_backward_x_host(float* pgx, float* pgy, float* pk, int64 ndat, int64 xchn, int64 xh, int64 xw, int64 ychn, int64 kh, int64 kw, int64 sh, int64 sw) {
    int64 yh = xh * sh;
    int64 yw = xw * sw;

    for (int64 nd = 0, n = 0; nd < ndat; nd++) {
        for (int64 xn = 0; xn < xchn; xn++) {
            for (int64 xr = 0; xr < xh; xr++) {
                for (int64 xc = 0; xc < xw; xc++, n++) {
                    int64 bh = (kh - 1) / 2, bw = (kw - 1) / 2;

                    float sum = 0;

                    for (int64 kr = 0; kr < kh; kr++) {
                        int64 yr = xr * sh + (kr - bh) + (sh - 1) / 2;
                        if (yr < 0 || yr >= xh) continue;

                        for (int64 kc = 0; kc < kw; kc++) {
                            int64 yc = xc * sw + (kc - bw) + (sw - 1) / 2;
                            if (yc < 0 || yc >= xw) continue;

                            for (int64 yn = 0; yn < ychn; yn++) {
                                int64 ypos = ((nd * ychn + yn) * yh + yr) * yw + yc;
                                int64 kpos = ((yn * xchn + xn) * kh + kr) * kw + kc;

                                sum += pgy[ypos] * pk[kpos];
                            }
                        }
                    }

                    pgx[n] = sum;
                }
            }
        }
    }
}

__global__ void conv2d_transposed_backward_x_cuda(int64 size, float* pgx, float* pgy, float* pk, int64 ndat, int64 xchn, int64 xh, int64 xw, int64 ychn, int64 kh, int64 kw, int64 sh, int64 sw) {
    int64 n = blockIdx.x * blockDim.x + threadIdx.x;

    if (n < size) {
        int64 yh = xh * sh;
        int64 yw = xw * sw;

        int64 nd = n / (xchn * xh * xw);
        int64 xn = n / (xh * xw) % xchn;
        int64 xr = n / xw % xh;
        int64 xc = n % xw;

        int64 bh = (kh - 1) / 2, bw = (kw - 1) / 2;

        float sum = 0;

        for (int64 kr = 0; kr < kh; kr++) {
            int64 yr = xr * sh + (kr - bh) + (sh - 1) / 2;
            if (yr < 0 || yr >= xh) continue;

            for (int64 kc = 0; kc < kw; kc++) {
                int64 yc = xc * sw + (kc - bw) + (sw - 1) / 2;
                if (yc < 0 || yc >= xw) continue;

                for (int64 yn = 0; yn < ychn; yn++) {
                    int64 ypos = ((nd * ychn + yn) * yh + yr) * yw + yc;
                    int64 kpos = ((yn * xchn + xn) * kh + kr) * kw + kc;

                    sum += pgy[ypos] * pk[kpos];
                }
            }
        }

        pgx[n] = sum;
    }
}

void VMath::conv2d_transposed_backward_x(int device, float* pgx, float* pgy, float* pk, int64 ndat, int64 xchn, int64 xh, int64 xw, int64 ychn, int64 kh, int64 kw, int64 sh, int64 sw) {
    int64 size = ndat * xchn * xh * xw;

    CUDA_CALL(conv2d_transposed_backward_x, device, size, pgx, pgy, pk, ndat, xchn, xh, xw, ychn, kh, kw, sh, sw);
}

//--------------------------------------------------------------------------------------------------

__static__ void conv2d_transposed_backward_k_host(float* pgk, float* pgy, float* px, int64 ndat, int64 xchn, int64 xh, int64 xw, int64 ychn, int64 kh, int64 kw, int64 sh, int64 sw) {
    int64 yh = xh * sh;
    int64 yw = xw * sw;

    for (int64 kr = 0, n = 0; kr < kh; kr++) {
        for (int64 kc = 0; kc < kw; kc++) {
            for (int64 xn = 0; xn < xchn; xn++) {
                for (int64 yn = 0; yn < ychn; yn++, n++) {
                    int64 bh = (kh - 1) / 2, bw = (kw - 1) / 2;

                    float sum = 0;

                    for (int64 xr = 0; xr < xh; xr++) {
                        int64 yr = xr * sh + (kr - bh) + (sh - 1) / 2;
                        if (yr < 0 || yr >= xh) continue;

                        for (int64 xc = 0; xc < xw; xc++) {
                            int64 yc = xc * sw + (kc - bw) + (sw - 1) / 2;
                            if (yc < 0 || yc >= xw) continue;

                            for (int64 nd = 0; nd < ndat; nd++) {
                                int64 ypos = ((nd * ychn + yn) * yh + yr) * yw + yc;
                                int64 xpos = ((nd * xchn + xn) * xh + xr) * xw + xc;

                                sum += pgy[ypos] * px[xpos];
                            }
                        }
                    }

                    int64 kpos = ((yn * xchn + xn) * kh + kr) * kw + kc;
                    pgk[kpos] = sum;
                }
            }
        }
    }
}

__global__ void conv2d_transposed_backward_k_cuda(int64 size, float* pgk, float* pgy, float* px, int64 ndat, int64 xchn, int64 xh, int64 xw, int64 ychn, int64 kh, int64 kw, int64 sh, int64 sw) {
    int64 n = blockIdx.x * blockDim.x + threadIdx.x;

    if (n < size) {
        int64 yh = xh * sh;
        int64 yw = xw * sw;

        int64 kr = n / (kw * xchn * ychn);
        int64 kc = n / (xchn * ychn) % kw;
        int64 xn = n / ychn % xchn;
        int64 yn = n % ychn;

        int64 bh = (kh - 1) / 2, bw = (kw - 1) / 2;

        float sum = 0;

        for (int64 xr = 0; xr < xh; xr++) {
            int64 yr = xr * sh + (kr - bh) + (sh - 1) / 2;
            if (yr < 0 || yr >= xh) continue;

            for (int64 xc = 0; xc < xw; xc++) {
                int64 yc = xc * sw + (kc - bw) + (sw - 1) / 2;
                if (yc < 0 || yc >= xw) continue;

                for (int64 nd = 0; nd < ndat; nd++) {
                    int64 ypos = ((nd * ychn + yn) * yh + yr) * yw + yc;
                    int64 xpos = ((nd * xchn + xn) * xh + xr) * xw + xc;

                    sum += pgy[ypos] * px[xpos];
                }
            }
        }

        int64 kpos = ((yn * xchn + xn) * kh + kr) * kw + kc;
        pgk[kpos] = sum;
    }
}

void VMath::conv2d_transposed_backward_k(int device, float* pgk, float* pgy, float* px, int64 ndat, int64 xchn, int64 xh, int64 xw, int64 ychn, int64 kh, int64 kw, int64 sh, int64 sw) {
    int64 size = kh * kw * xchn * ychn;

    CUDA_CALL(conv2d_transposed_backward_k, device, size, pgk, pgy, px, ndat, xchn, xh, xw, ychn, kh, kw, sh, sw);
}

//--------------------------------------------------------------------------------------------------

__static__ void conv2d_dilated_host(float* py, float* px, float* pk, int64 ndat, int64 xchn, int64 xh, int64 xw, int64 ychn, int64 kh, int64 kw, int64 gh, int64 gw) {
    for (int64 nd = 0, n = 0; nd < ndat; nd++) {
        for (int64 yn = 0; yn < ychn; yn++) {
            for (int64 yr = 0; yr < xh; yr++) {
                for (int64 yc = 0; yc < xw; yc++, n++) {
                    float sum = 0;

                    int64 bh = (kh - 1) / 2, bw = (kw - 1) / 2;

                    for (int64 kr = 0; kr < kh; kr++) {
                        int64 xr = yr + (kr - bh) * gh;
                        if (xr < 0 || xr >= xh) continue;

                        for (int64 kc = 0; kc < kw; kc++) {
                            int64 xc = yc + (kc - bw) * gw;
                            if (xc < 0 || xc >= xw) continue;

                            for (int64 xn = 0; xn < xchn; xn++) {
                                int64 xpos = ((nd * xchn + xn) * xh + xr) * xw + xc;
                                int64 kpos = ((yn * xchn + xn) * kh + kr) * kw + kc;

                                sum += px[xpos] * pk[kpos];
                            }
                        }
                    }

                    py[n] = sum;
                }
            }
        }
    }
}

__global__ void conv2d_dilated_cuda(int64 size, float* py, float* px, float* pk, int64 ndat, int64 xchn, int64 xh, int64 xw, int64 ychn, int64 kh, int64 kw, int64 gh, int64 gw) {
    int64 n = blockIdx.x * blockDim.x + threadIdx.x;

    if (n < size) {
        int64 nd = n / (ychn * xh * xw);
        int64 yn = n / (xh * xw) % ychn;
        int64 yr = n / xw % xh;
        int64 yc = n % xw;

        int64 bh = (kh - 1) / 2, bw = (kw - 1) / 2;

        float sum = 0;

        for (int64 kr = 0; kr < kh; kr++) {
            int64 xr = yr + (kr - bh) * gh;
            if (xr < 0 || xr >= xh) continue;

            for (int64 kc = 0; kc < kw; kc++) {
                int64 xc = yc + (kc - bw) * gw;
                if (xc < 0 || xc >= xw) continue;

                for (int64 xn = 0; xn < xchn; xn++) {
                    int64 xpos = ((nd * xchn + xn) * xh + xr) * xw + xc;
                    int64 kpos = ((yn * xchn + xn) * kh + kr) * kw + kc;

                    sum += px[xpos] * pk[kpos];
                }
            }
        }

        py[n] = sum;
    }
}

void VMath::conv2d_dilated(int device, float* py, float* px, float* pk, int64 ndat, int64 xchn, int64 xh, int64 xw, int64 ychn, int64 kh, int64 kw, int64 gh, int64 gw) {
    int64 size = ndat * ychn * xh * xw;
    CUDA_CALL(conv2d_dilated, device, size, py, px, pk, ndat, xchn, xh, xw, ychn, kh, kw, gh, gw);
}

//--------------------------------------------------------------------------------------------------

__static__ void conv2d_dilated_backward_x_host(float* pgx, float* pgy, float* pk, int64 ndat, int64 xchn, int64 xh, int64 xw, int64 ychn, int64 kh, int64 kw, int64 gh, int64 gw) {
    for (int64 nd = 0, n = 0; nd < ndat; nd++) {
        for (int64 xn = 0; xn < xchn; xn++) {
            for (int64 xr = 0; xr < xh; xr++) {
                for (int64 xc = 0; xc < xw; xc++, n++) {
                    int64 bh = (kh - 1) / 2, bw = (kw - 1) / 2;

                    float sum = 0;

                    for (int64 kr = 0; kr < kh; kr++) {
                        int64 yr = xr - (kr - bh) * gh;
                        if (yr < 0 || yr >= xh) continue;

                        for (int64 kc = 0; kc < kw; kc++) {
                            int64 yc = xc - (kc - bw) * gw;
                            if (yc < 0 || yc >= xw) continue;

                            for (int64 yn = 0; yn < ychn; yn++) {
                                int64 ypos = ((nd * ychn + yn) * xh + yr) * xw + yc;
                                int64 kpos = ((yn * xchn + xn) * kh + kr) * kw + kc;

                                sum += pgy[ypos] * pk[kpos];
                            }
                        }
                    }

                    pgx[n] = sum;
                }
            }
        }
    }
}

__global__ void conv2d_dilated_backward_x_cuda(int64 size, float* pgx, float* pgy, float* pk, int64 ndat, int64 xchn, int64 xh, int64 xw, int64 ychn, int64 kh, int64 kw, int64 gh, int64 gw) {
    int64 n = blockIdx.x * blockDim.x + threadIdx.x;

    if (n < size) {
        int64 nd = n / (xchn * xh * xw);
        int64 xn = n / (xh * xw ) % xchn;
        int64 xr = n / xw % xh;
        int64 xc = n % xw;

        int64 bh = (kh - 1) / 2, bw = (kw - 1) / 2;

        float sum = 0;

        for (int64 kr = 0; kr < kh; kr++) {
            int64 yr = xr - (kr - bh) * gh;
            if (yr < 0 || yr >= xh) continue;

            for (int64 kc = 0; kc < kw; kc++) {
                int64 yc = xc - (kc - bw) * gw;
                if (yc < 0 || yc >= xw) continue;

                for (int64 yn = 0; yn < ychn; yn++) {
                    int64 ypos = ((nd * ychn + yn) * xh + yr) * xw + yc;
                    int64 kpos = ((yn * xchn + xn) * kh + kr) * kw + kc;

                    sum += pgy[ypos] * pk[kpos];
                }
            }
        }

        pgx[n] = sum;
    }
}

void VMath::conv2d_dilated_backward_x(int device, float* pgx, float* pgy, float* pk, int64 ndat, int64 xchn, int64 xh, int64 xw, int64 ychn, int64 kh, int64 kw, int64 gh, int64 gw) {
    int64 size = ndat * xchn * xh * xw;

    CUDA_CALL(conv2d_dilated_backward_x, device, size, pgx, pgy, pk, ndat, xchn, xh, xw, ychn, kh, kw, gh, gw);
}

//--------------------------------------------------------------------------------------------------

__static__ void conv2d_dilated_backward_k_host(float* pgk, float* pgy, float* px, int64 ndat, int64 xchn, int64 xh, int64 xw, int64 ychn, int64 kh, int64 kw, int64 gh, int64 gw) {
    for (int64 kr = 0, n = 0; kr < kh; kr++) {
        for (int64 kc = 0; kc < kw; kc++) {
            for (int64 xn = 0; xn < xchn; xn++) {
                for (int64 yn = 0; yn < ychn; yn++, n++) {
                    int64 bh = (kh - 1) / 2, bw = (kw - 1) / 2;

                    float sum = 0;

                    for (int64 xr = 0; xr < xh; xr++) {
                        int64 yr = xr - (kr - bh) * gh;
                        if (yr < 0 || yr >= xh) continue;

                        for (int64 xc = 0; xc < xw; xc++) {
                            int64 yc = xc - (kc - bw) * gw;
                            if (yc < 0 || yc >= xw) continue;

                            for (int64 nd = 0; nd < ndat; nd++) {
                                int64 ypos = ((nd * ychn + yn) * xh + yr) * xw + yc;
                                int64 xpos = ((nd * xchn + xn) * xh + xr) * xw + xc;

                                sum += pgy[ypos] * px[xpos];
                            }
                        }
                    }

                    int64 kpos = ((yn * xchn + xn) * kh + kr) * kw + kc;
                    pgk[kpos] = sum;
                }
            }
        }
    }
}

__global__ void conv2d_dilated_backward_k_cuda(int64 size, float* pgk, float* pgy, float* px, int64 ndat, int64 xchn, int64 xh, int64 xw, int64 ychn, int64 kh, int64 kw, int64 gh, int64 gw) {
    int64 n = blockIdx.x * blockDim.x + threadIdx.x;

    if (n < size) {
        int64 kr = n / (kw * xchn * ychn);
        int64 kc = n / (xchn * ychn) % kw;
        int64 xn = n / ychn % xchn;
        int64 yn = n % ychn;

        int64 bh = (kh - 1) / 2, bw = (kw - 1) / 2;

        float sum = 0;

        for (int64 xr = 0; xr < xh; xr++) {
            int64 yr = xr - (kr - bh) * gh;
            if (yr < 0 || yr >= xh) continue;

            for (int64 xc = 0; xc < xw; xc++) {
                int64 yc = xc - (kc - bw) * gw;
                if (yc < 0 || yc >= xw) continue;

                for (int64 nd = 0; nd < ndat; nd++) {
                    int64 ypos = ((nd * ychn + yn) * xh + yr) * xw + yc;
                    int64 xpos = ((nd * xchn + xn) * xh + xr) * xw + xc;

                    sum += pgy[ypos] * px[xpos];
                }
            }
        }

        int64 kpos = ((yn * xchn + xn) * kh + kr) * kw + kc;
        pgk[kpos] = sum;
    }
}

void VMath::conv2d_dilated_backward_k(int device, float* pgk, float* pgy, float* px, int64 ndat, int64 xchn, int64 xh, int64 xw, int64 ychn, int64 kh, int64 kw, int64 gh, int64 gw) {
    int64 size = kh * kw * xchn * ychn;

    CUDA_CALL(conv2d_dilated_backward_k, device, size, pgk, pgy, px, ndat, xchn, xh, xw, ychn, kh, kw, gh, gw);
}

//--------------------------------------------------------------------------------------------------

__static__ void maxpool_host(float* py, float* px, int* pm, int64 ndat, int64 xchn, int64 xh, int64 xw, int64 kh, int64 kw) {
    for (int64 nd = 0, n = 0; nd < ndat; nd++) {
        for (int64 yn = 0; yn < xchn; yn++) {
            for (int64 yr = 0; yr < xh; yr++) {
                for (int64 yc = 0; yc < xw; yc++, n++) {
                    float max = -FLT_MAX;

                    int64 mpos = -1;
                    int64 bh = (kh - 1) / 2, bw = (kw - 1) / 2;

                    for (int64 kr = 0; kr < kh; kr++) {
                        int64 xr = yr + kr - bh;
                        if (xr < 0 || xr >= xh) continue;

                        for (int64 kc = 0; kc < kw; kc++) {
                            int64 xc = yc + kc - bw;
                            if (xc < 0 || xc >= xw) continue;

                            int64 xpos = ((nd * xchn + yn) * xh + xr) * xw + xc;

                            if (px[xpos] > max) {
                                max = px[xpos];
                                mpos = xpos;
                            }
                        }
                    }

                    py[n] = max;
                    pm[n] = (int)mpos;
                }
            }
        }
    }
}

__global__ void maxpool_cuda(int64 size, float* py, float* px, int* pm, int64 ndat, int64 xchn, int64 xh, int64 xw, int64 kh, int64 kw) {
    int64 n = blockIdx.x * blockDim.x + threadIdx.x;

    if (n < size) {
        int64 nd = n / (xchn * xh * xw);
        int64 yn = n / (xh * xw) % xchn;
        int64 yr = n / xw % xh;
        int64 yc = n % xw;

        float max = -FLT_MAX;

        int64 mpos = -1;
        int64 bh = (kh - 1) / 2, bw = (kw - 1) / 2;

        for (int64 kr = 0; kr < kh; kr++) {
            int64 xr = yr + kr - bh;
            if (xr < 0 || xr >= xh) continue;

            for (int64 kc = 0; kc < kw; kc++) {
                int64 xc = yc + kc - bw;
                if (xc < 0 || xc >= xw) continue;

                int64 xpos = ((nd * xchn + yn) * xh + xr) * xw + xc;

                if (px[xpos] > max) {
                    max = px[xpos];
                    mpos = xpos;
                }
            }
        }

        py[n] = max;
        pm[n] = (int)mpos;
    }
}

void VMath::maxpool(int device, float* py, float* px, int* pm, int64 ndat, int64 xchn, int64 xh, int64 xw, int64 kh, int64 kw) {
    int64 size = ndat * xh * xw * xchn;

    CUDA_CALL(maxpool, device, size, py, px, pm, ndat, xchn, xh, xw, kh, kw);
}

//--------------------------------------------------------------------------------------------------

__static__ void maxpool_backward_x_host(float* pgx, float* pgy, int* pm, int64 ndat, int64 xchn, int64 xh, int64 xw, int64 kh, int64 kw) {
    for (int64 nd = 0, n = 0; nd < ndat; nd++) {
        for (int64 xn = 0; xn < xchn; xn++) {
            for (int64 xr = 0; xr < xh; xr++) {
                for (int64 xc = 0; xc < xw; xc++, n++) {
                    float sum = 0;

                    int64 bh = (kh - 1) / 2, bw = (kw - 1) / 2;

                    for (int64 kr = 0; kr < kh; kr++) {
                        int64 yr = xr - kr + bh;
                        if (yr < 0 || yr >= xh) continue;

                        for (int64 kc = 0; kc < kw; kc++) {
                            int64 yc = xc - kc + bw;
                            if (yc < 0 || yc >= xw) continue;

                            int64 ypos = ((nd * xchn + xn) * xh + yr) * xw + yc;

                            if (pm[ypos] == n) sum += pgy[ypos];
                        }
                    }

                    pgx[n] = sum;
                }
            }
        }
    }
}

__global__ void maxpool_backward_x_cuda(int64 size, float* pgx, float* pgy, int* pm, int64 ndat, int64 xchn, int64 xh, int64 xw, int64 kh, int64 kw) {
    int64 n = blockIdx.x * blockDim.x + threadIdx.x;

    if (n < size) {
        int64 nd = n / (xchn * xh * xw);
        int64 xn = n / (xh * xw) % xchn;
        int64 xr = n / xw % xh;
        int64 xc = n % xw;

        float sum = 0;

        int64 bh = (kh - 1) / 2, bw = (kw - 1) / 2;

        for (int64 kr = 0; kr < kh; kr++) {
            int64 yr = xr - kr + bh;
            if (yr < 0 || yr >= xh) continue;

            for (int64 kc = 0; kc < kw; kc++) {
                int64 yc = xc - kc + bw;
                if (yc < 0 || yc >= xw) continue;

                int64 ypos = ((nd * xchn + xn) * xh + yr) * xw + yc;

                if (pm[ypos] == n) sum += pgy[ypos];
            }
        }

        pgx[n] = sum;
    }
}

void VMath::maxpool_backward_x(int device, float* pgx, float* pgy, int* pm, int64 ndat, int64 xchn, int64 xh, int64 xw, int64 kh, int64 kw) {
    if (0) return;

    int64 size = ndat * xchn * xh * xw;

    CUDA_CALL(maxpool_backward_x, device, size, pgx, pgy, pm, ndat, xchn, xh, xw, kh, kw);
}

//--------------------------------------------------------------------------------------------------

__static__ void avgpool_host(float* py, float* px, int* pm, int64 ndat, int64 xchn, int64 xh, int64 xw, int64 kh, int64 kw) {
    for (int64 nd = 0, n = 0; nd < ndat; nd++) {
        for (int64 yn = 0; yn < xchn; yn++) {
            for (int64 yr = 0; yr < xh; yr++) {
                for (int64 yc = 0; yc < xw; yc++, n++) {
                    float sum = 0;

                    int64 count = 0;
                    int64 bh = (kh - 1) / 2, bw = (kw - 1) / 2;

                    for (int64 kr = 0; kr < kh; kr++) {
                        int64 xr = yr + kr - bh;
                        if (xr < 0 || xr >= xh) continue;

                        for (int64 kc = 0; kc < kw; kc++) {
                            int64 xc = yc + kc - bw;
                            if (xc < 0 || xc >= xw) continue;

                            int64 xpos = ((nd * xchn + yn) * xh + xr) * xw + xc;

                            sum += px[xpos];
                            count++;
                        }
                    }

                    py[n] = sum / (float)count;
                    pm[n] = (int)count;
                }
            }
        }
    }
}

__global__ void avgpool_cuda(int64 size, float* py, float* px, int* pm, int64 ndat, int64 xchn, int64 xh, int64 xw, int64 kh, int64 kw) {
    int64 n = blockIdx.x * blockDim.x + threadIdx.x;

    if (n < size) {
        int64 nd = n / (xchn * xh * xw);
        int64 yn = n / (xh * xw ) % xchn;
        int64 yr = n / xw % xh;
        int64 yc = n % xw;

        float sum = 0;

        int64 count = 0;
        int64 bh = (kh - 1) / 2, bw = (kw - 1) / 2;

        for (int64 kr = 0; kr < kh; kr++) {
            int64 xr = yr + kr - bh;
            if (xr < 0 || xr >= xh) continue;

            for (int64 kc = 0; kc < kw; kc++) {
                int64 xc = yc + kc - bw;
                if (xc < 0 || xc >= xw) continue;

                int64 xpos = ((nd * xchn + yn) * xh + xr) * xw + xc;

                sum += px[xpos];
                count++;
            }
        }

        py[n] = sum / (float)count;
        pm[n] = (int)count;
    }
}

void VMath::avgpool(int device, float* py, float* px, int* pm, int64 ndat, int64 xchn, int64 xh, int64 xw, int64 kh, int64 kw) {
    int64 size = ndat * xchn * xh * xw;

    CUDA_CALL(avgpool, device, size, py, px, pm, ndat, xchn, xh, xw, kh, kw);
}

//--------------------------------------------------------------------------------------------------

__static__ void avgpool_backward_x_host(float* pgx, float* pgy, int* pm, int64 ndat, int64 xchn, int64 xh, int64 xw, int64 kh, int64 kw) {
    for (int64 nd = 0, n = 0; nd < ndat; nd++) {
        for (int64 xn = 0; xn < xchn; xn++) {
            for (int64 xr = 0; xr < xh; xr++) {
                for (int64 xc = 0; xc < xw; xc++, n++) {
                    float sum = 0;

                    int64 bh = (kh - 1) / 2, bw = (kw - 1) / 2;

                    for (int64 kr = 0; kr < kh; kr++) {
                        int64 yr = xr - kr + bh;
                        if (yr < 0 || yr >= xh) continue;

                        for (int64 kc = 0; kc < kw; kc++) {
                            int64 yc = xc - kc + bw;
                            if (yc < 0 || yc >= xw) continue;

                            int64 ypos = ((nd * xchn + xn) * xh + yr) * xw + yc;

                            sum += pgy[ypos] / (float)pm[ypos];
                        }
                    }

                    pgx[n] = sum;
                }
            }
        }
    }
}

__global__ void avgpool_backward_x_cuda(int64 size, float* pgx, float* pgy, int* pm, int64 ndat, int64 xchn, int64 xh, int64 xw, int64 kh, int64 kw) {
    int64 n = blockIdx.x * blockDim.x + threadIdx.x;

    if (n < size) {
        int64 nd = n / (xchn * xh * xw);
        int64 xn = n / (xh * xw ) % xchn;
        int64 xr = n / xw % xh;
        int64 xc = n % xw;

        float sum = 0;

        int64 bh = (kh - 1) / 2, bw = (kw - 1) / 2;

        for (int64 kr = 0; kr < kh; kr++) {
            int64 yr = xr - kr + bh;
            if (yr < 0 || yr >= xh) continue;

            for (int64 kc = 0; kc < kw; kc++) {
                int64 yc = xc - kc + bw;
                if (yc < 0 || yc >= xw) continue;

                int64 ypos = ((nd * xchn + xn) * xh + yr) * xw + yc;

                sum += pgy[ypos] / (float)pm[ypos];
            }
        }

        pgx[n] = sum;
    }
}

void VMath::avgpool_backward_x(int device, float* pgx, float* pgy, int* pm, int64 ndat, int64 xchn, int64 xh, int64 xw, int64 kh, int64 kw) {
    int64 size = ndat * xchn * xh * xw;

    CUDA_CALL(avgpool_backward_x, device, size, pgx, pgy, pm, ndat, xchn, xh, xw, kh, kw);
}

//--------------------------------------------------------------------------------------------------

__static__ void globalavg_host(float* py, float* px, int64 ndat, int64 xchn, int64 xh, int64 xw) {
    for (int64 nd = 0, n = 0; nd < ndat; nd++) {
        for (int64 yn = 0; yn < xchn; yn++, n++) {
            float sum = 0;

            for (int64 xr = 0; xr < xh; xr++) {
                for (int64 xc = 0; xc < xw; xc++) {
                    int64 xpos = ((nd * xchn + yn) * xh + xr) * xw + xc;

                    sum += px[xpos];
                }
            }

            py[n] = sum / (float)(xh * xw);
        }
    }
}

__global__ void globalavg_cuda(int64 size, float* py, float* px, int64 ndat, int64 xchn, int64 xh, int64 xw) {
    int64 n = blockIdx.x * blockDim.x + threadIdx.x;

    if (n < size) {
        int64 nd = n / xchn;
        int64 yn = n % xchn;

        float sum = 0;

        for (int64 xr = 0; xr < xh; xr++) {
            for (int64 xc = 0; xc < xw; xc++) {
                int64 xpos = ((nd * xchn + yn) * xh + xr) * xw + xc;

                sum += px[xpos];
            }
        }

        py[n] = sum / (float)(xh * xw);
    }
}

void VMath::globalavg(int device, float* py, float* px, int64 ndat, int64 xchn, int64 xh, int64 xw) {
    int64 size = ndat * xchn;

    CUDA_CALL(globalavg, device, size, py, px, ndat, xchn, xh, xw);
}

//--------------------------------------------------------------------------------------------------

__static__ void globalavg_backward_x_host(float* pgx, float* pgy, int64 ndat, int64 xchn, int64 xh, int64 xw) {
    for (int64 nd = 0, n = 0; nd < ndat; nd++) {
        for (int64 xn = 0; xn < xchn; xn++) {
            for (int64 xr = 0; xr < xh; xr++) {
                for (int64 xc = 0; xc < xw; xc++, n++) {
                    int64 ypos = nd * xchn + xn;
                    pgx[n] = pgy[ypos] / (float)(xh * xw);
                }
            }
        }
    }
}

__global__ void globalavg_backward_x_cuda(int64 size, float* pgx, float* pgy, int64 ndat, int64 xchn, int64 xh, int64 xw) {
    int64 n = blockIdx.x * blockDim.x + threadIdx.x;

    if (n < size) {
        int64 nd = n / (xchn * xh * xw);
        int64 xn = n / (xh * xw) % xchn;
        //int64 xr = n / xw % xh;
        //int64 xc = n % xw;

        int64 ypos = nd * xchn + xn;
        pgx[n] = pgy[ypos] / (float)(xh * xw);
    }
}

void VMath::globalavg_backward_x(int device, float* pgx, float* pgy, int64 ndat, int64 xchn, int64 xh, int64 xw) {
    int64 size = ndat * xchn * xh * xw;

    CUDA_CALL(globalavg_backward_x, device, size, pgx, pgy, ndat, xchn, xh, xw);
}

//--------------------------------------------------------------------------------------------------

__static__ void adaptiveavg_host(float* py, float* px, int64 ndat, int64 xchn, int64 yh, int64 yw, int64 hratio, int64 wratio) {
    for (int64 nd = 0, n = 0; nd < ndat; nd++) {
        for (int64 yn = 0; yn < xchn; yn++) {
            for (int64 rn = 0; rn < yh; rn++) {
                for (int64 cn = 0; cn < yw; cn++, n++) {
                    float sum = 0;

                    for (int64 r = 0; r < hratio; r++) {
                        for (int64 c = 0; c < wratio; c++) {
                            int64 xr = rn * hratio + r;
                            int64 xc = cn * wratio + c;

                            int64 xpos = ((nd * xchn + yn) * yh * hratio + xr) * yw * wratio + xc;

                            sum += px[xpos];
                        }
                    }

                    py[n] = sum / (float)(hratio * wratio);
                }
            }
        }
    }
}

__global__ void adaptiveavg_cuda(int64 size, float* py, float* px, int64 ndat, int64 xchn, int64 yh, int64 yw, int64 hratio, int64 wratio) {
    int64 n = blockIdx.x * blockDim.x + threadIdx.x;

    if (n < size) {
        int64 nd = n / (xchn * yh * yw);
        int64 yn = n / (yh * yw) % xchn;
        int64 rn = n / yw % yh;
        int64 cn = n % yw;

        float sum = 0;

        for (int64 r = 0; r < hratio; r++) {
            for (int64 c = 0; c < wratio; c++) {
                int64 xr = rn * hratio + r;
                int64 xc = cn * wratio + c;

                int64 xpos = ((nd * xchn + yn) * yh * hratio + xr) * yw * wratio + xc;

                sum += px[xpos];
            }
        }

        py[n] = sum / (float)(hratio * wratio);
    }
}

void VMath::adaptiveavg(int device, float* py, float* px, int64 ndat, int64 xchn, int64 yh, int64 yw, int64 hratio, int64 wratio) {
    int64 size = ndat * xchn * yh * yw;
    //return;
    CUDA_CALL(adaptiveavg, device, size, py, px, ndat, xchn, yh, yw, hratio, wratio);
}

//--------------------------------------------------------------------------------------------------

__static__ void adaptiveavg_backward_x_host(float* pgx, float* pgy, int64 ndat, int64 xchn, int64 xheight, int64 xwidth, int64 hratio, int64 wratio) {
    for (int64 nd = 0, n = 0; nd < ndat; nd++) {
        for (int64 xn = 0; xn < xchn; xn++, n++) {
            for (int64 xr = 0; xr < xheight; xr++) {
                for (int64 xc = 0; xc < xwidth; xc++) {
                    int64 yheight = xheight / hratio;
                    int64 ywidth = xwidth / hratio;

                    int64 yr = xr / hratio;
                    int64 yc = xc / wratio;

                    int64 ypos = ((nd * xchn + xn) * yheight + yr) * ywidth + yc;

                    pgx[n] = pgy[ypos] / (float)(hratio * wratio);
                }
            }
        }
    }
}

__global__ void adaptiveavg_backward_x_cuda(int64 size, float* pgx, float* pgy, int64 ndat, int64 xchn, int64 xheight, int64 xwidth, int64 hratio, int64 wratio) {
    int64 n = blockIdx.x * blockDim.x + threadIdx.x;

    if (n < size) {
        int64 nd = n / (xchn * xheight * xwidth);
        int64 xn = n / (xheight * xwidth) % xchn;
        int64 xr = n / xwidth % xheight;
        int64 xc = n % xwidth;

        int64 yheight = xheight / hratio;
        int64 ywidth = xwidth / hratio;

        int64 yr = xr / hratio;
        int64 yc = xc / wratio;

        int64 ypos = ((nd * xchn + xn) * yheight + yr) * ywidth + yc;

        pgx[n] = pgy[ypos] / (float)(hratio * wratio);
    }
}

void VMath::adaptiveavg_backward_x(int device, float* pgx, float* pgy, int64 ndat, int64 xchn, int64 xheight, int64 xwidth, int64 hratio, int64 wratio) {
    int64 size = ndat * xheight * xwidth * xchn;
    CUDA_CALL(adaptiveavg_backward_x, device, size, pgx, pgy, ndat, xchn, xheight, xwidth, hratio, wratio);
}

//--------------------------------------------------------------------------------------------------

__static__ void stride_host(float* py, float* px, int64 ndat, int64 xchn, int64 xh, int64 xw, int64 yh, int64 yw, int64 sh, int64 sw) {
    for (int64 nd = 0, n = 0; nd < ndat; nd++) {
        for (int64 yn = 0; yn < xchn; yn++) {
            for (int64 yr = 0; yr < yh; yr++, n++) {
                for (int64 yc = 0; yc < yw; yc++) {
                    int64 xr = yr * sh + (sh - 1) / 2;
                    int64 xc = yc * sw + (sw - 1) / 2;

                    int64 xpos = ((nd * xchn + yn) * xh + xr) * xw + xc;

                    if ((xr >= 0 && xr < xh) && (xc >= 0 && xc < xw)) py[n] = px[xpos];
                }
            }
        }
    }
}

__global__ void stride_cuda(int64 size, float* py, float* px, int64 ndat, int64 xchn, int64 xh, int64 xw, int64 yh, int64 yw, int64 sh, int64 sw) {
    int64 n = blockIdx.x * blockDim.x + threadIdx.x;

    if (n < size) {
        int64 nd = n / (xchn * yh * yw);
        int64 yn = n / (yh * yw) % xchn;
        int64 yr = n / yw % yh;
        int64 yc = n % yw;

        int64 xr = yr * sh + (sh - 1) / 2;
        int64 xc = yc * sw + (sw - 1) / 2;

        int64 xpos = ((nd * xchn + yn) * xh + xr) * xw + xc;

        if ((xr >= 0 && xr < xh) && (xc >= 0 && xc < xw)) py[n] = px[xpos];
    }
}

void VMath::stride(int device, float* py, float* px, int64 ndat, int64 xchn, int64 xh, int64 xw, int64 yh, int64 yw, int64 sh, int64 sw) {
    int64 size = ndat * xchn * yh * yw;

    CUDA_CALL(stride, device, size, py, px, ndat, xchn, xh, xw, yh, yw, sh, sw);
}

//--------------------------------------------------------------------------------------------------

__static__ void stride_backward_x_host(float* pgx, float* pgy, int64 ndat, int64 xchn, int64 xh, int64 xw, int64 yh, int64 yw, int64 sh, int64 sw) {
    for (int64 nd = 0, n = 0; nd < ndat; nd++) {
        for (int64 xn = 0; xn < xchn; xn++) {
            for (int64 xr = 0; xr < xh; xr++) {
                for (int64 xc = 0; xc < xw; xc++, n++) {
                    int64 rm = (xr - (sh - 1) / 2) % sh;
                    //int64 rm = (xr + (sh - 1)) / 2 % sh;
                    int64 cm = (xc + (sw - 1) / 2) % sw;

                    if (rm == 0 && cm == 0) {
                        int64 yr = (xr - (sh - 1) / 2) / sh;
                        int64 yc = (xc - (sw - 1) / 2) / sw;

                        if ((yr < 0 || yr >= yh || yc < 0 || yc >= yw)) continue;

                        int64 ypos = ((nd * xchn + xn) * yh + yr) * yw + yc;

                        pgx[n] = pgy[ypos];
                    }
                    else {
                        pgx[n] = 0;
                    }
                }
            }
        }
    }
}

__global__ void stride_backward_x_cuda(int64 size, float* pgx, float* pgy, int64 ndat, int64 xchn, int64 xh, int64 xw, int64 yh, int64 yw, int64 sh, int64 sw) {
    int64 n = blockIdx.x * blockDim.x + threadIdx.x;

    if (n < size) {
        int64 nd = n / (xchn * xh * xw);
        int64 xn = n / (xh * xw) % xchn;
        int64 xr = n / xw % xh;
        int64 xc = n % xw;

        pgx[n] = 0;

        //int64 rm = (xr + (sh - 1)) / 2 % sh;
        int64 rm = (xr - (sh - 1) / 2) % sh;
        int64 cm = (xc - (sw - 1) / 2) % sw;

        if (rm == 0 && cm == 0) {
            int64 yr = (xr - (sh - 1) / 2) / sh;
            int64 yc = (xc - (sw - 1) / 2) / sw;

            if ((yr >= 0 && yr < yh && yc >= 0 && yc < yw)) {
                int64 ypos = ((nd * xchn + xn) * yh + yr) * yw + yc;
                pgx[n] = pgy[ypos];
            }
        }
    }
}

void VMath::stride_backward_x(int device, float* pgx, float* pgy, int64 ndat, int64 xchn, int64 xh, int64 xw, int64 yh, int64 yw, int64 sh, int64 sw) {
    int64 size = ndat * xchn * xh * xw;

    CUDA_CALL(stride_backward_x, device, size, pgx, pgy, ndat, xchn, xh, xw, yh, yw, sh, sw);
}

//--------------------------------------------------------------------------------------------------

__static__ void lstm_process_host(float* pr, float* ps, float* ps1, float* pa, int64 ndat, int64 nrec, int64 ninp) {
    for (int64 nd = 0, n = 0; nd < ndat; nd++) {
        for (int64 nr = 0; nr < nrec; nr++, n++) {
            int64 apos = nd * 4 * nrec + nr;

            float state = ps1 ? ps1[n] : 0;

            float input_gate = _sigmoid_host(pa[apos + 0 * nrec]);
            float forget_gate = _sigmoid_host(pa[apos + 1 * nrec]);
            float input_block = _tanh_host(pa[apos + 2 * nrec]);
            float output_gate = _sigmoid_host(pa[apos + 3 * nrec]);

            float new_state = state * forget_gate + input_block * input_gate;
            float new_recur = _tanh_host(new_state) * output_gate;

            ps[n] = new_state;
            pr[n] = new_recur;
        }
    }
}

__global__ void lstm_process_cuda(int64 size, float* pr, float* ps, float* ps1, float* pa, int64 ndat, int64 nrec, int64 ninp) {
    int64 n = blockIdx.x * blockDim.x + threadIdx.x;

    if (n < size) {
        int64 nd = n / nrec;
        int64 nr = n % nrec;

        int64 apos = nd * 4 * nrec + nr;

        float state = ps1 ? ps1[n] : 0;

        float input_gate = _sigmoid_cuda(pa[apos + 0 * nrec]);
        float forget_gate = _sigmoid_cuda(pa[apos + 1 * nrec]);
        float input_block = _tanh_cuda(pa[apos + 2 * nrec]);
        float output_gate = _sigmoid_cuda(pa[apos + 3 * nrec]);

        float new_state = state * forget_gate + input_block * input_gate;
        float new_recur = _tanh_cuda(new_state) * output_gate;

        ps[n] = new_state;
        pr[n] = new_recur;
    }
}

void VMath::lstm_process(int device, float* pr, float* ps, float* ps1, float* pa, int64 ndat, int64 nrec, int64 ninp) {
    int64 size = ndat * nrec;

    CUDA_CALL(lstm_process, device, size, pr, ps, ps1, pa, ndat, nrec, ninp);
}

//--------------------------------------------------------------------------------------------------

__static__ void lstm_process_backward_host(float* pgr, float* pgs, float* pga, float* ps, float* pa, int64 ndat, int64 nrec, int64 ninp) {
    for (int64 nd = 0, n = 0; nd < ndat; nd++) {
        for (int64 nr = 0; nr < nrec; nr++, n++) {
            int64 apos = nd * 4 * nrec + nr;

            float old_state = ps ? ps[n] : 0;

            float input_gate = _sigmoid_host(pa[apos + 0 * nrec]);
            float forget_gate = _sigmoid_host(pa[apos + 1 * nrec]);
            float input_block = _tanh_host(pa[apos + 2 * nrec]);
            float output_gate = _sigmoid_host(pa[apos + 3 * nrec]);

            //float new_state = old_state * forget_gate + input_block * input_gate;
            //float new_recur = _tanh_host(new_state) * output_gate;

            float new_state = old_state * forget_gate + input_block * input_gate;

            float g_new_recur = pgr[n];

            float g_output_gate = _tanh_host(new_state) * g_new_recur;
            float g_tanh_state = output_gate * g_new_recur;
            float g_new_state = _tanh_derv_with_x_host(new_state) * g_tanh_state + pgs[n];

            float g_forget_gate = old_state * g_new_state;
            float g_old_state = forget_gate * g_new_state;

            float g_input_block = input_gate * g_new_state;
            float g_input_gate = input_block * g_new_state;

            pga[apos * 0 + n * nrec] = _sigmoid_derv_with_y_host(input_gate) * g_input_gate;
            pga[apos * 1 + n * nrec] = _sigmoid_derv_with_y_host(forget_gate) * g_forget_gate;
            pga[apos * 2 + n * nrec] = _tanh_derv_with_y_host(input_block) * g_input_block;
            pga[apos * 3 + n * nrec] = _sigmoid_derv_with_y_host(output_gate) * g_output_gate;

            pgs[n] = g_old_state;
        }
    }
}

__global__ void lstm_process_backward_cuda(int64 size, float* pgr, float* pgs, float* pga, float* ps, float* pa, int64 ndat, int64 nrec, int64 ninp) {
    int64 n = blockIdx.x * blockDim.x + threadIdx.x;

    if (n < size) {
        int64 nd = n / nrec;
        int64 nr = n % nrec;

        int64 apos = nd * 4 * nrec + nr;

        float old_state = ps ? ps[n] : 0;

        float input_gate = _sigmoid_cuda(pa[apos + 0 * nrec]);
        float forget_gate = _sigmoid_cuda(pa[apos + 1 * nrec]);
        float input_block = _tanh_cuda(pa[apos + 2 * nrec]);
        float output_gate = _sigmoid_cuda(pa[apos + 3 * nrec]);

        float new_state = old_state * forget_gate + input_block * input_gate;

        float g_new_recur = pgr[n];

        float g_output_gate = _tanh_cuda(new_state) * g_new_recur;
        float g_tanh_state = output_gate * g_new_recur;
        float g_new_state = _tanh_derv_with_x_cuda(new_state) * g_tanh_state + pgs[n];

        float g_forget_gate = old_state * g_new_state;
        float g_old_state = forget_gate * g_new_state;

        float g_input_block = input_gate * g_new_state;
        float g_input_gate = input_block * g_new_state;

        pga[apos + 0 * nrec] = _sigmoid_derv_with_y_cuda(input_gate) * g_input_gate;
        pga[apos + 1 * nrec] = _sigmoid_derv_with_y_cuda(forget_gate) * g_forget_gate;
        pga[apos + 2 * nrec] = _tanh_derv_with_y_cuda(input_block) * g_input_block;
        pga[apos + 3 * nrec] = _sigmoid_derv_with_y_cuda(output_gate) * g_output_gate;

        pgs[n] = g_old_state;
    }
}

void VMath::lstm_process_backward(int device, float* pgr, float* pgs, float* pga, float* ps, float* pa, int64 ndat, int64 nrec, int64 ninp) {
    int64 size = ndat * nrec;

    CUDA_CALL(lstm_process_backward, device, size, pgr, pgs, pga, ps, pa, ndat, nrec, ninp);
}

//--------------------------------------------------------------------------------------------------

__static__ void gru_process_host(float* pr, float* pai, float* par, int64 nt, int64 ntimes, int64 ndat, int64 nrec) {
    for (int64 nd = 0; nd < ndat; nd++) {
        for (int64 ny = 0; ny < nrec; ny++) {
            int64 rpos = nd * nrec + ny;

            int64 rgpos = nd * 3 * nrec + ny;
            int64 zgpos = nd * 3 * nrec + nrec + ny;
            int64 ngpos = nd * 3 * nrec + 2 * nrec + ny;

            float Rt = _sigmoid_host(pai[rgpos] + par[rgpos]);
            float Zt = _sigmoid_host(pai[zgpos] + par[zgpos]);

            float Nt = _tanh_host(pai[ngpos] + Rt * par[ngpos]);

            float Ht = (nt > 0) ? pr[rpos - ndat * nrec] : 0;

            pr[rpos] = (1.0f - Zt) * Nt + Zt * Ht;
        }
    }
}

__global__ void gru_process_cuda(int64 size, float* pr, float* pai, float* par, int64 nt, int64 ntimes, int64 ndat, int64 nrec) {
    int64 n = blockIdx.x * blockDim.x + threadIdx.x;

    if (n < size) {
        int64 nd = n / nrec;
        int64 ny = n % nrec;

        int64 rpos = nd * nrec + ny;

        int64 rgpos = nd * 3 * nrec + ny;
        int64 zgpos = nd * 3 * nrec + nrec + ny;
        int64 ngpos = nd * 3 * nrec + 2 * nrec + ny;

        float Rt = _sigmoid_cuda(pai[rgpos] + par[rgpos]);
        float Zt = _sigmoid_cuda(pai[zgpos] + par[zgpos]);

        float Nt = _tanh_cuda(pai[ngpos] + Rt * par[ngpos]);

        float Ht = (nt > 0) ? pr[rpos - ndat * nrec] : 0;

        pr[rpos] = (1.0f - Zt) * Nt + Zt * Ht;
    }
}

void VMath::gru_process(int device, float* pr, float* pai, float* par, int64 nt, int64 ntimes, int64 ndat, int64 nrec) {
    int64 size = ndat * nrec;

    CUDA_CALL(gru_process, device, size, pr, pai, par, nt, ntimes, ndat, nrec);
}

//--------------------------------------------------------------------------------------------------

__static__ void gru_process_backward_host(float* pgr, float* pgai, float* pgar, float* pr, float* pai, float* par, int64 nt, int64 ntimes, int64 ndat, int64 nrec) {
    for (int64 nd = 0; nd < ndat; nd++) {
        for (int64 ny = 0; ny < nrec; ny++) {
            int64 rpos = nd * nrec + ny;

            int64 rgpos = nd * 3 * nrec + ny;
            int64 zgpos = nd * 3 * nrec + nrec + ny;
            int64 ngpos = nd * 3 * nrec + 2 * nrec + ny;

            float Rt = _sigmoid_host(pai[rgpos] + par[rgpos]);
            float Zt = _sigmoid_host(pai[zgpos] + par[zgpos]);
            float Nt = _tanh_host(pai[ngpos] + Rt * par[ngpos]);

            float Ht_1 = (nt > 0) ? pr[rpos - ndat * nrec] : 0;

            float gHt = pgr[rpos];

            float gNt = (1 - Zt) * gHt;
            float gZt = (Ht_1 - Nt) * gHt;
            float gHt_1 = Zt * gHt;

            float gNt_seed = (1 + Nt) * (1 - Nt) * gNt;

            float gRt = par[ngpos] * gNt_seed;

            pgai[rgpos] = pgar[rgpos] = Rt * (1 - Rt) * gRt;
            pgai[zgpos] = pgar[zgpos] = Zt * (1 - Zt) * gZt;

            pgai[ngpos] = gNt_seed;
            pgar[ngpos] = Rt * gNt_seed;

            pgr[rpos] = gHt_1;
        }
    }
}

__global__ void gru_process_backward_cuda(int64 size, float* pgr, float* pgai, float* pgar, float* pr, float* pai, float* par, int64 nt, int64 ntimes, int64 ndat, int64 nrec) {
    int64 n = blockIdx.x * blockDim.x + threadIdx.x;

    if (n < size) {
        int64 nd = n / nrec;
        int64 ny = n % nrec;

        int64 rpos = nd * nrec + ny;

        int64 rgpos = nd * 3 * nrec + ny;
        int64 zgpos = nd * 3 * nrec + nrec + ny;
        int64 ngpos = nd * 3 * nrec + 2 * nrec + ny;

        float Rt = _sigmoid_cuda(pai[rgpos] + par[rgpos]);
        float Zt = _sigmoid_cuda(pai[zgpos] + par[zgpos]);
        float Nt = _tanh_cuda(pai[ngpos] + Rt * par[ngpos]);

        float Ht_1 = (nt > 0) ? pr[rpos - ndat * nrec] : 0;

        float gHt = pgr[rpos];

        float gNt = (1 - Zt) * gHt;
        float gZt = (Ht_1 - Nt) * gHt;
        float gHt_1 = Zt * gHt;

        float gNt_seed = (1 + Nt) * (1 - Nt) * gNt;

        float gRt = par[ngpos] * gNt_seed;

        pgai[rgpos] = pgar[rgpos] = Rt * (1 - Rt) * gRt;
        pgai[zgpos] = pgar[zgpos] = Zt * (1 - Zt) * gZt;

        pgai[ngpos] = gNt_seed;
        pgar[ngpos] = Rt * gNt_seed;

        pgr[rpos] = gHt_1;
        //pgr[rpos] = Zt * (1 - Zt) * gZt;
    }
}

void VMath::gru_process_backward(int device, float* pgr, float* pgai, float* pgar, float* pr, float* pai, float* par, int64 nt, int64 ntimes, int64 ndat, int64 nrec) {
    int64 size = ndat * nrec;

    CUDA_CALL(gru_process_backward, device, size, pgr, pgai, pgar, pr, pai, par, nt, ntimes, ndat, nrec);
}

/*
//--------------------------------------------------------------------------------------------------

__static__ void rnn_ext_input_host(float* pe, float* px, float* prs, int64 ndat, int64 ntimes, int64 nt, int64 nvec, int64 nrec, bool bInSeq) {
    for (int64 nd = 0, n = 0; nd < ndat; nd++) {
        for (int64 nx = 0; nx < nvec; nx++) {
            int64 xpos = bInSeq ? (nd * ntimes + nt) * nvec + nx : nd * nvec + nx;
            pe[n++] = px[xpos];
        }
        for (int64 ny = 0; ny < nrec; ny++) {
            int64 rpos = (nd * ntimes + nt - 1) * nrec + ny;
            pe[n++] = (nt == 0) ? 0 : prs[rpos];
        }
    }
}

__global__ void rnn_ext_input_cuda(int64 size, float* pe, float* px, float* prs, int64 ndat, int64 ntimes, int64 nt, int64 nvec, int64 nrec, bool bInSeq) {
    int64 n = blockIdx.x * blockDim.x + threadIdx.x;

    if (n < size) {
        int64 nd = n / (nvec + nrec);
        int64 nx = n % (nvec + nrec);
        int64 ny = nx - nvec;

        if (nx < nvec) {
            int64 xpos = bInSeq ? (nd * ntimes + nt) * nvec + nx : nd * nvec + nx;
            pe[n] = px[xpos];
        }
        else {
            int64 rpos = (nd * ntimes + nt - 1) * nrec + ny;
            pe[n] = (nt == 0) ? 0 : prs[rpos];
        }
    }
}

void VMath::rnn_ext_input(int device, float* pe, float* px, float* prs, int64 ndat, int64 ntimes, int64 nt, int64 nvec, int64 nrec, bool bInSeq) {
    int64 size = ndat * (nvec + nrec);

    CUDA_CALL(rnn_ext_input, device, size, pe, px, prs, ndat, ntimes, nt, nvec, nrec, bInSeq);
}

//--------------------------------------------------------------------------------------------------

__static__ void rnn_save_output_host(float* po, float* pr, int64 ndat, int64 ntimes, int64 nt, int64 nrec) {
    for (int64 nd = 0; nd < ndat; nd++) {
        for (int64 ny = 0; ny < nrec; ny++) {
            int64 ypos = (nd * ntimes + nt) * nrec + ny;
            int64 rpos = nd * nrec + ny;
            po[ypos] = pr[rpos];
        }
    }
}

__global__ void rnn_save_output_cuda(int64 size, float* po, float* pr, int64 ndat, int64 ntimes, int64 nt, int64 nrec) {
    int64 n = blockIdx.x * blockDim.x + threadIdx.x;

    if (n < size) {
        int64 nd = n / nrec;
        int64 ny = n % nrec;

        int64 ypos = (nd * ntimes + nt) * nrec + ny;
        int64 rpos = nd * nrec + ny;
        po[ypos] = pr[rpos];
    }
}

void VMath::rnn_save_output(int device, float* po, float* pr, int64 ndat, int64 ntimes, int64 nt, int64 nrec) {
    int64 size = ndat * nrec;

    CUDA_CALL(rnn_save_output, device, size, po, pr, ndat, ntimes, nt, nrec);
}

//--------------------------------------------------------------------------------------------------

__static__ void rnn_fetch_timedata_host(float* pd, float* pds, int64 ndat, int64 ntimes, int64 nt, int64 nrec) {
    for (int64 nd = 0; nd < ndat; nd++) {
        for (int64 ny = 0; ny < nrec; ny++) {
            int64 ypos = (nd * ntimes + nt) * nrec + ny;
            int64 rpos = nd * nrec + ny;

            pd[rpos] = pds[ypos];
        }
    }
}

__global__ void rnn_fetch_timedata_cuda(int64 size, float* pd, float* pds, int64 ndat, int64 ntimes, int64 nt, int64 nrec) {
    int64 n = blockIdx.x * blockDim.x + threadIdx.x;

    if (n < size) {
        int64 nd = n / nrec;
        int64 ny = n % nrec;

        int64 ypos = (nd * ntimes + nt) * nrec + ny;
        int64 rpos = nd * nrec + ny;

        pd[rpos] = pds[ypos];
    }
}

void VMath::rnn_fetch_timedata(int device, float* pd, float* pds, int64 ndat, int64 ntimes, int64 nt, int64 nrec) {
    int64 size = ndat * nrec;

    CUDA_CALL(rnn_fetch_timedata, device, size, pd, pds, ndat, ntimes, nt, nrec);
}

//--------------------------------------------------------------------------------------------------

__static__ void rnn_ext_input_for_backward_host(float* pe, float* px, float* prs, int64 ndat, int64 ntimes, int64 nt, int64 nvec, int64 nrec, bool bInSeq) {
    for (int64 nd = 0, n = 0; nd < ndat; nd++) {
        for (int64 nx = 0; nx < nvec; nx++) {
            int64 xpos = bInSeq ? (nd * ntimes + nt) * nvec + nx : nd * nvec + nx;
            pe[n++] = px[xpos];
        }
        for (int64 ny = 0; ny < nrec; ny++) {
            int64 ypos = (nd * ntimes + (nt - 1)) * nrec + ny;
            pe[n++] = (nt == 0) ? 0 : prs[ypos];
        }
    }
}

__global__ void rnn_ext_input_for_backward_cuda(int64 size, float* pe, float* px, float* prs, int64 ndat, int64 ntimes, int64 nt, int64 nvec, int64 nrec, bool bInSeq) {
    int64 n = blockIdx.x * blockDim.x + threadIdx.x;

    if (n < size) {
        int64 nd = n / (nvec + nrec);
        int64 nx = n % (nvec + nrec);
        int64 ny = nx - nvec;

        if (nx < nvec) {
            int64 xpos = bInSeq ? (nd * ntimes + nt) * nvec + nx : nd * nvec + nx;
            pe[n] = px[xpos];
        }
        else {
            int64 ypos = (nd * ntimes + (nt - 1)) * nrec + ny;
            pe[n] = (nt > 0) ? prs[ypos] : 0;
        }
    }
}

void VMath::rnn_ext_input_for_backward(int device, float* pe, float* px, float* prs, int64 ndat, int64 ntimes, int64 nt, int64 nvec, int64 nrec, bool bInSeq) {
    int64 size = ndat * (nvec + nrec);

    CUDA_CALL(rnn_ext_input_for_backward, device, size, pe, px, prs, ndat, ntimes, nt, nvec, nrec, bInSeq);
}

//--------------------------------------------------------------------------------------------------

__static__ void rnn_ext_split_backward_host(float* pgx, float* pgr, float* pge, int64 ndat, int64 ntimes, int64 nt, int64 nvec, int64 nrec, bool bInSeq) {
    for (int64 nd = 0, n = 0; nd < ndat; nd++) {
        for (int64 nx = 0; nx < nvec; nx++) {
            if (bInSeq) {
                int64 xpos = (nd * ntimes + nt) * nvec + nx;
                pgx[xpos] = pge[n++];
            }
            else {
                int64 xpos = nd * nvec + nx;
                if (nt == ntimes - 1) pgx[xpos] = pge[n];
                else pgx[xpos] += pge[n++];
            }
        }
        for (int64 ny = 0; ny < nrec; ny++) {
            int64 rpos = nd * nrec + ny;
            pgr[rpos] = pge[n++];
        }
    }
}

__global__ void rnn_ext_split_backward_cuda(int64 size, float* pgx, float* pgr, float* pge, int64 ndat, int64 ntimes, int64 nt, int64 nvec, int64 nrec, bool bInSeq) {
    int64 n = blockIdx.x * blockDim.x + threadIdx.x;

    if (n < size) {
        int64 nd = n / (nvec + nrec);
        int64 nx = n % (nvec + nrec);
        int64 ny = nx - nvec;

        if (nx < nvec) {
            if (bInSeq) {
                int64 xpos = (nd * ntimes + nt) * nvec + nx;
                pgx[xpos] = pge[n];
            }
            else {
                int64 xpos = nd * nvec + nx;
                if (nt == ntimes - 1) pgx[xpos] = pge[n];
                else pgx[xpos] += pge[n];
            }
        }
        else {
            //int64 ypos = (nd * ntimes + (nt - 1)) * nrec + ny; // ????
            //pgr[ypos] = pge[n];
            int64 rpos = nd * nrec + ny;
            pgr[rpos] = pge[n++];
        }
    }
}

void VMath::rnn_ext_split_backward(int device, float* pgx, float* pgr, float* pge, int64 ndat, int64 ntimes, int64 nt, int64 nvec, int64 nrec, bool bInSeq) {
    int64 size = ndat * (nvec + nrec);

    CUDA_CALL(rnn_ext_split_backward, device, size, pgx, pgr, pge, ndat, ntimes, nt, nvec, nrec, bInSeq);
}

//--------------------------------------------------------------------------------------------------

__static__ void rnn_load_output_backward_host(float* pgr, float* pgy, int64 ndat, int64 ntimes, int64 nt, int64 nrec, bool bOutSeq) {
    for (int64 nd = 0; nd < ndat; nd++) {
        for (int64 ny = 0; ny < nrec; ny++) {
            int64 ypos = (nd * ntimes + nt) * nrec + ny;
            int64 rpos = nd * nrec + ny;

            if (bOutSeq) {
                pgr[rpos] = (nt == ntimes - 1) ? pgy[ypos] : pgr[rpos] + pgy[ypos];
            }
            else {
                if (nt == ntimes - 1) pgr[rpos] = pgy[rpos];    // 나머지 시간대는 이전에 저장된 다음 시간대 값 사용
            }
        }
    }
}

__global__ void rnn_load_output_backward_cuda(int64 size, float* pgr, float* pgy, int64 ndat, int64 ntimes, int64 nt, int64 nrec, bool bOutSeq) {
    int64 n = blockIdx.x * blockDim.x + threadIdx.x;

    if (n < size) {
        int64 nd = n / nrec;
        int64 ny = n % nrec;

        int64 ypos = (nd * ntimes + nt) * nrec + ny;
        int64 rpos = nd * nrec + ny;

        if (bOutSeq) {
            pgr[rpos] = (nt == ntimes - 1) ? pgy[ypos] : pgr[rpos] + pgy[ypos];
        }
        else {
            if (nt == ntimes - 1) pgr[rpos] = pgy[rpos];    // 나머지 시간대는 이전에 저장된 다음 시간대 값 사용
        }
    }
}

void VMath::rnn_load_output_backward(int device, float* pgr, float* pgy, int64 ndat, int64 ntimes, int64 nt, int64 nrec, bool bOutSeq) {
    int64 size = ndat * nrec;

    CUDA_CALL(rnn_load_output_backward, device, size, pgr, pgy, ndat, ntimes, nt, nrec, bOutSeq);
}

//--------------------------------------------------------------------------------------------------

__static__ void gru_process_rz_host(float* pe, float* prs, float* pas, float* pa2, int64 ndat, int64 ntimes, int64 nt, int64 nvec, int64 nrec) {
    for (int64 nd = 0; nd < ndat; nd++) {
        for (int64 ny = 0; ny < nrec; ny++) {
            int64 rpos = nd * nrec + ny;
            int64 tpos = (nd * ntimes + nt) * nrec + ny;
            int64 arpos = rpos * 2;
            int64 atpos = tpos * 3;
            int64 ppos = (nd * ntimes + (nt - 1)) * nrec + ny;
            int64 epos = nd * (nvec + nrec) + nvec + ny;

            pas[atpos + 0] = pa2[arpos + 0];
            pas[atpos + 1] = pa2[arpos + 1];

            float r_gate = _sigmoid_host(pas[atpos + 0]);
            float old_recur = (nt == 0) ? 0 : prs[ppos];

            pe[epos] = r_gate * old_recur;
        }
    }
}

__global__ void gru_process_rz_cuda(int64 size, float* pe, float* prs, float* pas, float* pa2, int64 ndat, int64 ntimes, int64 nt, int64 nvec, int64 nrec) {
    int64 n = blockIdx.x * blockDim.x + threadIdx.x;

    if (n < size) {
        int64 nd = n / nrec;
        int64 ny = n % nrec;

        int64 rpos = nd * nrec + ny;
        int64 tpos = (nd * ntimes + nt) * nrec + ny;
        int64 arpos = rpos * 2;
        int64 atpos = tpos * 3;
        int64 ppos = (nd * ntimes + (nt - 1)) * nrec + ny;
        int64 epos = nd * (nvec + nrec) + nvec + ny;

        pas[atpos + 0] = pa2[arpos + 0];
        pas[atpos + 1] = pa2[arpos + 1];

        float r_gate = _sigmoid_cuda(pas[atpos + 0]);
        float old_recur = (nt == 0) ? 0 : prs[ppos];

        pe[epos] = r_gate * old_recur;
    }
}

void VMath::gru_process_rz(int device, float* pe, float* prs, float* pas, float* pa2, int64 ndat, int64 ntimes, int64 nt, int64 nvec, int64 nrec) {
    int64 size = ndat * nrec;

    CUDA_CALL(gru_process_rz, device, size, pe, prs, pas, pa2, ndat, ntimes, nt, nvec, nrec);
}

//--------------------------------------------------------------------------------------------------

__static__ void gru_process_h_host(float* prs, float* pas, float* pa1, int64 ndat, int64 ntimes, int64 nt, int64 nrec) {
    for (int64 nd = 0; nd < ndat; nd++) {
        for (int64 ny = 0; ny < nrec; ny++) {
            int64 rpos = nd * nrec + ny;
            int64 tpos = (nd * ntimes + nt) * nrec + ny;
            int64 arpos = rpos;
            int64 atpos = tpos * 3;
            
            int64 ppos = (nd * ntimes + (nt - 1)) * nrec + ny;

            pas[atpos + 2] = pa1[arpos];

            float old_recur = (nt == 0) ? 0 : prs[ppos];

            float z_gate = _sigmoid_host(pas[atpos + 1]);
            float h_block = _tanh_host(pas[atpos + 2]);

            float new_recur = old_recur * z_gate + h_block * (1.0f - z_gate);

            prs[tpos] = new_recur;
        }
    }
}

__global__ void gru_process_h_cuda(int64 size, float* prs, float* pas, float* pa1, int64 ndat, int64 ntimes, int64 nt, int64 nrec) {
    int64 n = blockIdx.x * blockDim.x + threadIdx.x;

    if (n < size) {
        int64 nd = n / nrec;
        int64 ny = n % nrec;

        int64 rpos = nd * nrec + ny;
        int64 tpos = (nd * ntimes + nt) * nrec + ny;
        int64 arpos = rpos;
        int64 atpos = tpos * 3;
        int64 ppos = (nd * ntimes + (nt - 1)) * nrec + ny;

        pas[atpos + 2] = pa1[arpos];

        float old_recur = (nt == 0) ? 0 : prs[ppos];

        float z_gate = _sigmoid_cuda(pas[atpos + 1]);
        float h_block = _tanh_cuda(pas[atpos + 2]);

        float new_recur = old_recur * z_gate + h_block * (1.0f - z_gate);

        prs[tpos] = new_recur;
    }
}

void VMath::gru_process_h(int device, float* prs, float* pas, float* pa1, int64 ndat, int64 ntimes, int64 nt, int64 nrec) {
    int64 size = ndat * nrec;

    CUDA_CALL(gru_process_h, device, size, prs, pas, pa1, ndat, ntimes, nt, nrec);
}

//--------------------------------------------------------------------------------------------------

__static__ void gru_ext_input1_for_backward_host(float* pe, float* px, float* prs, float* pas, int64 ndat, int64 ntimes, int64 nt, int64 nvec, int64 nrec, bool bInSeq) {
    for (int64 nd = 0; nd < ndat; nd++) {
        for (int64 nx = 0; nx < nvec; nx++) {
            int64 xpos = bInSeq ? (nd * ntimes + nt) * nvec + nx : nd * nvec + nx;
            int64 epos = nd * (nvec + nrec) + nx;

            pe[epos] = px[xpos];
        }
        for (int64 ny = 0; ny < nrec; ny++) {
            int64 tpos = (nd * ntimes + nt) * nrec + ny;
            int64 ppos = (nd * ntimes + (nt - 1)) * nrec + ny;
            int64 atpos = tpos * 3;
            int64 epos = nd * (nvec + nrec) + nvec + ny;

            float old_recur = (nt == 0) ? 0 : prs[ppos];
            float r_gate = _sigmoid_host(pas[atpos + 0]);

            pe[epos] = r_gate * old_recur;
        }
    }
}

__global__ void gru_ext_input1_for_backward_cuda(int64 size, float* pe, float* px, float* prs, float* pas, int64 ndat, int64 ntimes, int64 nt, int64 nvec, int64 nrec, bool bInSeq) {
    int64 n = blockIdx.x * blockDim.x + threadIdx.x;

    if (n < size) {
        int64 nd = n / (nvec + nrec);
        int64 nx = n % (nvec + nrec);
        int64 ny = nx - nvec;

        if (nx < nvec) {
            int64 xpos = bInSeq ? (nd * ntimes + nt) * nvec + nx : nd * nvec + nx;
            int64 epos = nd * (nvec + nrec) + nx;

            pe[epos] = px[xpos];
        }
        else {
            int64 tpos = (nd * ntimes + nt) * nrec + ny;
            int64 ppos = (nd * ntimes + (nt - 1)) * nrec + ny;
            int64 atpos = tpos * 3;
            int64 epos = nd * (nvec + nrec) + nvec + ny;

            float old_recur = (nt == 0) ? 0 : prs[ppos];
            float r_gate = _sigmoid_cuda(pas[atpos + 0]);

            pe[epos] = r_gate * old_recur;
        }
    }
}

void VMath::gru_ext_input1_for_backward(int device, float* pe, float* px, float* prs, float* pas, int64 ndat, int64 ntimes, int64 nt, int64 nvec, int64 nrec, bool bInSeq) {
    int64 size = ndat * (nvec + nrec);

    CUDA_CALL(gru_ext_input1_for_backward, device, size, pe, px, prs, pas, ndat, ntimes, nt, nvec, nrec, bInSeq);
}

//--------------------------------------------------------------------------------------------------

__static__ void gru_load_output_backward_host(float* pgr, float* pgy, int64 ndat, int64 ntimes, int64 nt, int64 nrec, bool bOutSeq) {
    for (int64 nd = 0; nd < ndat; nd++) {
        for (int64 ny = 0; ny < nrec; ny++) {
            int64 ypos = (nd * ntimes + nt) * nrec + ny;
            int64 rpos = nd * nrec + ny;

            if (bOutSeq) pgr[rpos] = (nt == ntimes - 1) ? pgy[ypos] : pgr[rpos] + pgy[ypos];
            else if (nt == ntimes - 1) pgr[rpos] = pgy[rpos];    // 나머지 시간대는 이전에 저장된 다음 시간대 값 사용
        }
    }
}

__global__ void gru_load_output_backward_cuda(int64 size, float* pgr, float* pgy, int64 ndat, int64 ntimes, int64 nt, int64 nrec, bool bOutSeq) {
    int64 n = blockIdx.x * blockDim.x + threadIdx.x;

    if (n < size) {
        int64 nd = n / nrec;
        int64 ny = n % nrec;

        int64 ypos = (nd * ntimes + nt) * nrec + ny;
        int64 rpos = nd * nrec + ny;

        if (bOutSeq) pgr[rpos] = (nt == ntimes - 1) ? pgy[ypos] : pgr[rpos] + pgy[ypos];
        else if (nt == ntimes - 1) pgr[rpos] = pgy[rpos];    // 나머지 시간대는 이전에 저장된 다음 시간대 값 사용
    }
}

void VMath::gru_load_output_backward(int device, float* pgr, float* pgy, int64 ndat, int64 ntimes, int64 nt, int64 nrec, bool bOutSeq) {
    int64 size = ndat * nrec;

    CUDA_CALL(gru_load_output_backward, device, size, pgr, pgy, ndat, ntimes, nt, nrec, bOutSeq);
}

//--------------------------------------------------------------------------------------------------

__static__ void gru_process_h_backward_host(float* pgr, float* pga2, float* pga1, float* prs, float* pas, int64 ndat, int64 ntimes, int64 nt, int64 nrec) {
    for (int64 nd = 0; nd < ndat; nd++) {
        for (int64 ny = 0; ny < nrec; ny++) {
            int64 rpos = nd * nrec + ny;
            int64 tpos = (nd * ntimes + nt) * nrec + ny;
            int64 atpos = tpos * 3;
            int64 ppos = (nd * ntimes + (nt - 1)) * nrec + ny;

            float old_recur = (nt == 0) ? 0 : prs[ppos];

            float z_gate = _sigmoid_host(pas[atpos + 1]);
            float h_block = _tanh_host(pas[atpos + 2]);

            float g_new_recur = pgr[rpos];

            float g_old_recur = z_gate * g_new_recur;
            float g_h_block = (1.0f - z_gate) * g_new_recur;
            float g_z_gate = (old_recur - h_block) * g_new_recur;

            pgr[rpos] = g_old_recur;

            pga2[rpos * 2 + 1] = _sigmoid_derv_with_y_host(z_gate) * g_z_gate;
            pga1[rpos] = _tanh_derv_with_y_host(h_block) * g_h_block;
        }
    }
}

__global__ void gru_process_h_backward_cuda(int64 size, float* pgr, float* pga2, float* pga1, float* prs, float* pas, int64 ndat, int64 ntimes, int64 nt, int64 nrec) {
    int64 n = blockIdx.x * blockDim.x + threadIdx.x;

    if (n < size) {
        int64 nd = n / nrec;
        int64 ny = n % nrec;

        int64 rpos = nd * nrec + ny;
        int64 tpos = (nd * ntimes + nt) * nrec + ny;
        int64 atpos = tpos * 3;
        int64 ppos = (nd * ntimes + (nt - 1)) * nrec + ny;

        float old_recur = (nt == 0) ? 0 : prs[ppos];

        float z_gate = _sigmoid_cuda(pas[atpos + 1]);
        float h_block = _tanh_cuda(pas[atpos + 2]);

        float g_new_recur = pgr[rpos];

        float g_old_recur = z_gate * g_new_recur;
        float g_h_block = (1.0f - z_gate) * g_new_recur;
        float g_z_gate = (old_recur - h_block) * g_new_recur;

        pgr[rpos] = g_old_recur;

        pga2[rpos * 2 + 1] = _sigmoid_derv_with_y_cuda(z_gate) * g_z_gate;
        pga1[rpos] = _tanh_derv_with_y_cuda(h_block) * g_h_block;
    }
}

void VMath::gru_process_h_backward(int device, float* pgr, float* pga2, float* pga1, float* prs, float* pas, int64 ndat, int64 ntimes, int64 nt, int64 nrec) {
    int64 size = ndat * nrec;

    CUDA_CALL(gru_process_h_backward, device, size, pgr, pga2, pga1, prs, pas, ndat, ntimes, nt, nrec);
}

//--------------------------------------------------------------------------------------------------

__static__ void gru_process_rz_backward_host(float* pgr, float* pge, float* pga2, float* prs, float* pas, int64 ndat, int64 ntimes, int64 nt, int64 nvec, int64 nrec) {
    for (int64 nd = 0; nd < ndat; nd++) {
        for (int64 ny = 0; ny < nrec; ny++) {
            int64 rpos = nd * nrec + ny;
            int64 tpos = (nd * ntimes + nt) * nrec + ny;
            int64 ppos = (nd * ntimes + (nt - 1)) * nrec + ny;
            int64 arpos = rpos * 2;
            int64 atpos = tpos * 3;
            int64 epos = nd * (nvec + nrec) + nvec + ny;

            float r_gate = _sigmoid_host(pas[atpos + 0]);
            float old_recur = (nt == 0) ? 0 : prs[ppos];

            float g_r_gate = old_recur * pge[epos];
            float g_old_recur = r_gate * pge[epos];

            pga2[arpos + 0] = _sigmoid_derv_with_y_host(r_gate) * g_r_gate;

            pgr[rpos] += g_old_recur;
        }
    }
}

__global__ void gru_process_rz_backward_cuda(int64 size, float* pgr, float* pge, float* pga2, float* prs, float* pas, int64 ndat, int64 ntimes, int64 nt, int64 nvec, int64 nrec) {
    int64 n = blockIdx.x * blockDim.x + threadIdx.x;

    if (n < size) {
        int64 nd = n / nrec;
        int64 ny = n % nrec;

        int64 rpos = nd * nrec + ny;
        int64 tpos = (nd * ntimes + nt) * nrec + ny;
        int64 ppos = (nd * ntimes + (nt - 1)) * nrec + ny;
        int64 arpos = rpos * 2;
        int64 atpos = tpos * 3;
        int64 epos = nd * (nvec + nrec) + nvec + ny;

        float r_gate = _sigmoid_cuda(pas[atpos + 0]);
        float old_recur = (nt == 0) ? 0 : prs[ppos];

        float g_r_gate = old_recur * pge[epos];
        float g_old_recur = r_gate * pge[epos];

        pga2[arpos + 0] = _sigmoid_derv_with_y_cuda(r_gate) * g_r_gate;
        
        pgr[rpos] += g_old_recur;
    }
}

void VMath::gru_process_rz_backward(int device, float* pgr, float* pge, float* pga2, float* prs, float* pas, int64 ndat, int64 ntimes, int64 nt, int64 nvec, int64 nrec) {
    int64 size = ndat * nrec;

    CUDA_CALL(gru_process_rz_backward, device, size, pgr, pge, pga2, prs, pas, ndat, ntimes, nt, nvec, nrec);
}
*/

//--------------------------------------------------------------------------------------------------

__static__ void batchnorm_norm_train_host(float* py, float* px, float* pma, float* pmv, float* pba, float* pbv, int64 ndat, int64 ncol, int64 nrest, HYPER hMomentum, HYPER hEpsilon) {
    float momentum = HYPER_ACCESS(hMomentum);
    float epsilon = HYPER_ACCESS(hEpsilon);

    int64 nskip = (ncol - 1) * nrest;

    for (int64 nc = 0; nc < ncol; nc++) {
        int64 nstart = nc * nrest;

        float sum = 0, sqsum = 0;

        for (int64 nd = nstart; nd < ndat; nd += nskip) {
            for (int64 k = 0; k < nrest; k++) {
                sum += px[nd++];
            }
        }

        float avg = sum / (float)(ndat / ncol);

        for (int64 nd = nstart; nd < ndat; nd += nskip) {
            for (int64 k = 0; k < nrest; k++) {
                float diff = px[nd++] - avg;
                sqsum += diff * diff;
            }
        }

        float var = sqsum / (float)(ndat / ncol);
        float std = ::sqrtf(var + epsilon);

        pba[nc] = avg;
        pbv[nc] = var;

        pma[nc] = pma[nc] * momentum + avg * (1.0f - momentum);
        pmv[nc] = pmv[nc] * momentum + var * (1.0f - momentum);

        for (int64 nd = nstart; nd < ndat; nd += nskip) {
            for (int64 k = 0; k < nrest; k++, nd++) {
                py[nd] = (px[nd] - avg) / std;
            }
        }
    }
}

__static__ void batchnorm_norm_eval_host(float* py, float* px, float* pma, float* pmv, int64 ndat, int64 ncol, int64 nrest, HYPER hEpsilon) {
    float epsilon = HYPER_ACCESS(hEpsilon);

    for (int64 nd = 0; nd < ndat; nd++) {
        int64 nc = nd / nrest % ncol;

        float mavg = pma[nc];
        float mvar = pmv[nc];
        float mstd = ::sqrtf(mvar + epsilon);

        py[nd] = (px[nd] - mavg) / mstd;
    }
}

__global__ void batchnorm_norm_train_cuda(int64 size, float* py, float* px, float* pma, float* pmv, float* pba, float* pbv, int64 ndat, int64 ncol, int64 nrest, HYPER hMomentum, HYPER hEpsilon) {
    int64 n = blockIdx.x * blockDim.x + threadIdx.x;

    if (n < size) {
        int64 nc = n;

        float momentum = HYPER_ACCESS(hMomentum);
        float epsilon = HYPER_ACCESS(hEpsilon);

        int64 nskip = ncol * nrest;
        int64 nstart = nc * nrest;

        float sum = 0, sqsum = 0;

        for (int64 nd = nstart; nd < ndat; nd += nskip) {
            for (int64 k = 0; k < nrest; k++) {
                sum += px[nd + k];
            }
        }

        float avg = sum / (float)(ndat / ncol);

        for (int64 nd = nstart; nd < ndat; nd += nskip) {
            for (int64 k = 0; k < nrest; k++) {
                float diff = px[nd + k] - avg;
                sqsum += diff * diff;
            }
        }

        float var = sqsum / (float)(ndat / ncol);
        float std = ::sqrtf(var + epsilon);

        pba[nc] = avg;
        pbv[nc] = var;

        pma[nc] = pma[nc] * momentum + avg * (1.0f - momentum);
        pmv[nc] = pmv[nc] * momentum + var * (1.0f - momentum);

        for (int64 nd = nstart; nd < ndat; nd += nskip) {
            for (int64 k = 0; k < nrest; k++) {
                py[nd + k] = (px[nd + k] - avg) / std;
            }
        }
    }
}

__global__ void batchnorm_norm_eval_cuda(int64 size, float* py, float* px, float* pma, float* pmv, int64 ndat, int64 ncol, int64 nrest, HYPER hEpsilon) {
    int64 n = blockIdx.x * blockDim.x + threadIdx.x;

    if (n < size) {
        int64 nc = n / nrest % ncol;

        float epsilon = HYPER_ACCESS(hEpsilon);

        float mavg = pma[nc];
        float mvar = pmv[nc];
        float mstd = ::sqrtf(mvar + epsilon);

        py[n] = (px[n] - mavg) / mstd;
    }
}

void VMath::batchnorm_norm(int device, float* py, float* px, float* pma, float* pmv, float* pba, float* pbv, int64 ndat, int64 ncol, int64 nrest, HYPER hMomentum, HYPER hEpsilon, bool train) {
    if (train) {
        int64 size = ncol;
        CUDA_CALL(batchnorm_norm_train, device, size, py, px, pma, pmv, pba, pbv, ndat, ncol, nrest, hMomentum, hEpsilon);
    }
    else {
        int64 size = ndat;
        CUDA_CALL(batchnorm_norm_eval, device, size, py, px, pma, pmv, ndat, ncol, nrest, hEpsilon);
    }
}

//--------------------------------------------------------------------------------------------------

__static__ void batchnorm_scale_host(float* py, float* px, float* pscale, float* pshift, int64 ndat, int64 ncol, int64 nrest) {
    for (int64 nd = 0; nd < ndat; nd++) {
        int64 nc = nd / nrest % ncol;

        float scale = pscale ? pscale[nc] : 1.0f;
        float shift = pshift ? pshift[nc] : 0.0f;
        py[nd] = px[nd] * scale + shift;
    }
}

__global__ void batchnorm_scale_cuda(int64 size, float* py, float* px, float* pscale, float* pshift, int64 ndat, int64 ncol, int64 nrest) {
    int64 n = blockIdx.x * blockDim.x + threadIdx.x;

    if (n < size) {
        int64 nc = n / nrest % ncol;

        float scale = pscale ? pscale[nc] : 1.0f;
        float shift = pshift ? pshift[nc] : 0.0f;

        py[n] = px[n] * scale + shift;
    }
}

void VMath::batchnorm_scale(int device, float* py, float* px, float* pscale, float* pshift, int64 ndat, int64 ncol, int64 nrest) {
    int64 size = ndat;
    CUDA_CALL(batchnorm_scale, device, size, py, px, pscale, pshift, ndat, ncol, nrest);
}

//--------------------------------------------------------------------------------------------------

__static__ void batchnorm_backward_x_host(float* pgx, float* pgy, float* pscale, int64 ndat, int64 ncol, int64 nrest) {
    for (int64 nd = 0; nd < ndat; nd++) {
        int64 nc = nd / nrest % ncol;

        float scale = pscale ? pscale[nc] : 1.0f;
        pgx[nd] = pgy[nd] * scale;
    }
}

__global__ void batchnorm_backward_x_cuda(int64 size, float* pgx, float* pgy, float* pscale, int64 ndat, int64 ncol, int64 nrest) {
    int64 n = blockIdx.x * blockDim.x + threadIdx.x;

    if (n < size) {
        int64 nc = n / nrest % ncol;

        float scale = pscale ? pscale[nc] : 1.0f;

        pgx[n] = pgy[n] * scale;
    }
}

void VMath::batchnorm_backward_x(int device, float* pgx, float* pgy, float* pscale, int64 ndat, int64 ncol, int64 nrest) {
    int64 size = ndat;
    CUDA_CALL(batchnorm_backward_x, device, size, pgx, pgy, pscale, ndat, ncol, nrest);
}

//--------------------------------------------------------------------------------------------------

__static__ void batchnorm_backward_scale_host(float* pgr, float* pgy, float* px, int64 ndat, int64 ncol, int64 nrest) {
    int64 nskip = (ncol - 1) * nrest;

    for (int64 nc = 0; nc < ncol; nc++) {
        int64 nstart = nc * nrest;

        float sum = 0;

        for (int64 nd = nstart; nd < ndat; nd += nskip) {
            for (int64 k = 0; k < nrest; k++, nd++) {
                sum += px[nd] * pgy[nd];
            }
        }

        pgr[nc] = sum;
    }
}

__global__ void batchnorm_backward_scale_cuda(int64 size, float* pgr, float* pgy, float* px, int64 ndat, int64 ncol, int64 nrest) {
    int64 n = blockIdx.x * blockDim.x + threadIdx.x;

    if (n < size) {
        int64 nc = n;

        int64 nskip = (ncol - 1) * nrest;
        int64 nstart = nc * nrest;

        float sum = 0;

        for (int64 nd = nstart; nd < ndat; nd += nskip) {
            for (int64 k = 0; k < nrest; k++, nd++) {
                sum += px[nd] * pgy[nd];
            }
        }

        pgr[nc] = sum;
    }
}

void VMath::batchnorm_backward_scale(int device, float* pgr, float* pgy, float* px, int64 ndat, int64 ncol, int64 nrest) {
    if (pgr == NULL) return;

    int64 size = ncol;
    CUDA_CALL(batchnorm_backward_scale, device, size, pgr, pgy, px, ndat, ncol, nrest);
}

//--------------------------------------------------------------------------------------------------

__static__ void batchnorm_backward_shift_host(float* pgs, float* pgy, int64 ndat, int64 ncol, int64 nrest) {
    int64 nskip = (ncol - 1) * nrest;

    for (int64 nc = 0; nc < ncol; nc++) {
        int64 nstart = nc * nrest;

        float sum = 0;

        for (int64 nd = nstart; nd < ndat; nd += nskip) {
            for (int64 k = 0; k < nrest; k++) {
                sum += pgy[nd];
            }
        }

        pgs[nc] = sum;
    }
}

__global__ void batchnorm_backward_shift_cuda(int64 size, float* pgs, float* pgy, int64 ndat, int64 ncol, int64 nrest) {
    int64 n = blockIdx.x * blockDim.x + threadIdx.x;

    if (n < size) {
        int64 nc = n;

        int64 nskip = (ncol - 1) * nrest;
        int64 nstart = nc * nrest;

        float sum = 0;

        for (int64 nd = nstart; nd < ndat; nd += nskip) {
            for (int64 k = 0; k < nrest; k++) {
                sum += pgy[nd];
            }
        }

        pgs[nc] = sum;
    }
}

void VMath::batchnorm_backward_shift(int device, float* pgs, float* pgy, int64 ndat, int64 ncol, int64 nrest) {
    if (pgs == NULL) return;

    int64 size = ncol;
    CUDA_CALL(batchnorm_backward_shift, device, size, pgs, pgy, ndat, ncol, nrest);
}

//--------------------------------------------------------------------------------------------------

__static__ void batchnorm_backward_norm_host(float* pgx, float* pbv, int64 ndat, int64 ncol, int64 nrest, HYPER hEpsilon) {
    for (int64 nd = 0; nd < ndat; nd++) {
        int64 nc = nd / nrest % ncol;

        float var = pbv[nc];
        float epsilon = HYPER_ACCESS(hEpsilon);

        float std = ::sqrtf(var + epsilon);

        pgx[nd] = pgx[nd] / std;
    }
}

__global__ void batchnorm_backward_norm_cuda(int64 size, float* pgx, float* pbv, int64 ndat, int64 ncol, int64 nrest, HYPER hEpsilon) {
    int64 n = blockIdx.x * blockDim.x + threadIdx.x;

    if (n < size) {
        int64 nc = n / nrest % ncol;

        float var = pbv[nc];
        float epsilon = HYPER_ACCESS(hEpsilon);

        float std = ::sqrtf(var + epsilon);

        pgx[n] = pgx[n] / std;
    }
}

void VMath::batchnorm_backward_norm(int device, float* pgx, float* pbv, int64 ndat, int64 ncol, int64 nrest, HYPER hEpsilon) {
    if (pbv == NULL) return;

    int64 size = ndat;
    CUDA_CALL(batchnorm_backward_norm, device, size, pgx, pbv, ndat, ncol, nrest, hEpsilon);
}

//--------------------------------------------------------------------------------------------------

__static__ void layernorm_host(float* py, float* px, float* ps, int64 nrow, int64 ncol, HYPER hScale) {
    float scale = HYPER_ACCESS(hScale);

    for (int64 nr = 0; nr < nrow; nr++) {
        float sum = 0;
        float dsqsum = 0;

        for (int64 nc = 0; nc < ncol; nc++) {
            int64 xpos = nr * ncol + nc;
            float x = px[xpos];
            sum += x;
        }

        float avg = sum / ncol;

        for (int64 nc = 0; nc < ncol; nc++) {
            int64 xpos = nr * ncol + nc;
            float diff = px[xpos] - avg;
            dsqsum += diff * diff;
        }

        float var = dsqsum / ncol;
        float std = ::sqrtf(var + 1e-10f);

        for (int64 nc = 0; nc < ncol; nc++) {
            int64 xpos = nr * ncol + nc;
            py[xpos] = scale * ((px[xpos] - avg) / std);
        }

        ps[2 * nr + 0] = avg;
        ps[2 * nr + 1] = std;
    }
}

__global__ void layernorm_cuda(int64 size, float* py, float* px, float* ps, int64 nrow, int64 ncol, HYPER hScale) {
    int64 n = blockIdx.x * blockDim.x + threadIdx.x;

    if (n < size) {
        int64 nr = n;

        float sum = 0;
        float dsqsum = 0;

        float scale = HYPER_ACCESS(hScale);

        for (int64 nc = 0; nc < ncol; nc++) {
            int64 xpos = nr * ncol + nc;
            float x = px[xpos];
            sum += x;
        }

        float avg = sum / ncol;

        for (int64 nc = 0; nc < ncol; nc++) {
            int64 xpos = nr * ncol + nc;
            float diff = px[xpos] - avg;
            dsqsum += diff * diff;
        }

        float var = dsqsum / ncol;
        float std = ::sqrtf(var + 1e-10f);

        for (int64 nc = 0; nc < ncol; nc++) {
            int64 xpos = nr * ncol + nc;
            py[xpos] = scale * ((px[xpos] - avg) / std);
        }

        ps[2 * nr + 0] = avg;
        ps[2 * nr + 1] = std;
    }
}

void VMath::layernorm(int device, float* py, float* px, float* ps, int64 nrow, int64 ncol, HYPER hScale) {
    int64 size = nrow;
    CUDA_CALL(layernorm, device, size, py, px, ps, nrow, ncol, hScale);
}

//--------------------------------------------------------------------------------------------------

__static__ void layernorm_backward_host(float* pgx, float* pgy, float* ps, int64 nrow, int64 ncol, HYPER hScale) {
    float scale = HYPER_ACCESS(hScale);

    for (int64 nr = 0, n = 0; nr < nrow; nr++) {
        for (int64 nc = 0; nc < ncol; nc++, n++) {
            float avg = ps[2 * nr + 0];
            float std = ps[2 * nr + 1];

            pgx[n] = scale * ((pgy[n] - avg) / std);
        }
    }
}

__global__ void layernorm_backward_cuda(int64 size, float* pgx, float* pgy, float* ps, int64 nrow, int64 ncol, HYPER hScale) {
    int64 n = blockIdx.x * blockDim.x + threadIdx.x;

    if (n < size) {
        int64 nr = n / ncol;

        float avg = ps[2 * nr + 0];
        float std = ps[2 * nr + 1];
        float scale = HYPER_ACCESS(hScale);

        pgx[n] = scale * ((pgy[n] - avg) / std);
    }
}

void VMath::layernorm_backward(int device, float* pgx, float* pgy, float* ps, int64 nrow, int64 ncol, HYPER hScale) {
    int64 size = nrow * ncol;
    CUDA_CALL(layernorm_backward, device, size, pgx, pgy, ps, nrow, ncol, hScale);
}

//--------------------------------------------------------------------------------------------------

__static__ void dropout_host(float* py, float* px, unsigned char* pm, int64 ndat, HYPER hDropRatio) {
    float drop_ratio = HYPER_ACCESS(hDropRatio);

#ifndef NO_RANDOM_HOST
    std::uniform_real_distribution<float> coin(0, 1);
#endif

    for (int64 nd = 0, n = 0; nd < ndat; nd++, n++) {
#ifdef NO_RANDOM_HOST
        float dice = ms_no_random_uniform<float>(0, 1);
#else
        float dice = coin(ms_randGen);
#endif
        if (dice < drop_ratio) {
            py[n] = 0;
            pm[n] = 0;
        }
        else {
            py[n] = px[n] / (1.0f - drop_ratio);
            pm[n] = 1;
        }
    }
}

__global__ void dropout_cuda(int64 size, float* py, float* px, unsigned char* pm, int64 ndat, HYPER hDropRatio) {
    int64 n = blockIdx.x * blockDim.x + threadIdx.x;

    if (n < size) {
        float drop_ratio = HYPER_ACCESS(hDropRatio);

#ifndef NO_RANDOM_CUDA
        curandState state;

        curand_init(clock64(), n, 0, &state);

        float dice = curand_uniform(&state);
#else
        float dice = dev_no_random_uniform(n, 0, 1);
#endif

        if (dice < drop_ratio) {
            py[n] = 0;
            pm[n] = 0;
        }
        else {
            py[n] = px[n] / (1.0f - drop_ratio);;
            pm[n] = 1;
        }
    }
}

void VMath::dropout(int device, float* py, float* px, unsigned char* pm, int64 ndat, HYPER hDropRatio) {
    int64 size = ndat;
    CUDA_CALL(dropout, device, size, py, px, pm, ndat, hDropRatio);
}

//--------------------------------------------------------------------------------------------------

__static__ void dropout_backward_host(float* pgx, float* pgy, unsigned char* pm, int64 ndat, HYPER hDropRatio) {
    float drop_ratio = HYPER_ACCESS(hDropRatio);

    for (int64 nd = 0, n = 0; nd < ndat; nd++, n++) {
        if (pm[n] == 1) {
            pgx[n] = pgy[n] / (1.0f - drop_ratio);
        }
        else {
            pgx[n] = 0;
        }
    }
}

__global__ void dropout_backward_cuda(int64 size, float* pgx, float* pgy, unsigned char* pm, int64 ndat, HYPER hDropRatio) {
    int64 n = blockIdx.x * blockDim.x + threadIdx.x;

    if (n < size) {
        float drop_ratio = HYPER_ACCESS(hDropRatio);

        if (pm[n] == 1) {
            pgx[n] = pgy[n] / (1.0f - drop_ratio);
        }
        else {
            pgx[n] = 0;
        }
    }
}

void VMath::dropout_backward(int device, float* pgx, float* pgy, unsigned char* pm, int64 ndat, HYPER hDropRatio) {
    int64 size = ndat;
    CUDA_CALL(dropout_backward, device, size, pgx, pgy, pm, ndat, hDropRatio);
}

//--------------------------------------------------------------------------------------------------

__static__ void add_normal_noise_host(float* py, float* px, int64 ndat, HYPER hMean, HYPER hStd) {
    float mean = HYPER_ACCESS(hMean);
    float std = HYPER_ACCESS(hStd);

#ifndef NO_RANDOM_HOST
    std::normal_distribution<float> coin(mean, std);
#endif

    for (int64 nd = 0, n = 0; nd < ndat; nd++, n++) {
#ifdef NO_RANDOM_HOST
        float dice = ms_no_random_normal(mean, std);
#else
        float dice = coin(ms_randGen);
#endif
        py[n] = px[n] + dice;
    }
}

__global__ void add_normal_noise_cuda(int64 size, float* py, float* px, int64 ndat, HYPER hMean, HYPER hStd) {
    int64 n = blockIdx.x * blockDim.x + threadIdx.x;

    if (n < size) {
#ifndef NO_RANDOM_CUDA
        curandState state;

        float mean = HYPER_ACCESS(hMean);
        float std = HYPER_ACCESS(hStd);

        curand_init(clock64(), n, 0, &state);

        float dice = curand_normal(&state);
#else
        float dice = dev_no_random_normal(n, 0, 1);
#endif

        py[n] = px[n] + mean + dice * std;
    }
}

void VMath::add_normal_noise(int device, float* py, float* px, int64 ndat, HYPER hMean, HYPER hStd) {
    int64 size = ndat;
    CUDA_CALL(add_normal_noise, device, size, py, px, ndat, hMean, hStd);
}

//--------------------------------------------------------------------------------------------------

__static__ void add_uniform_noise_host(float* py, float* px, int64 ndat, HYPER hMin, HYPER hMax) {
    float min = HYPER_ACCESS(hMin);
    float max = HYPER_ACCESS(hMax);

#ifndef NO_RANDOM_HOST
    std::uniform_real_distribution<float> coin(min, max);
#endif

    for (int64 nd = 0, n = 0; nd < ndat; nd++, n++) {
#ifdef NO_RANDOM_HOST
        float dice = ms_no_randomuniform(0, 1);
#else
        float dice = coin(ms_randGen);
#endif
        py[n] = px[n] + (max - min) * dice + min;
    }
}

__global__ void add_uniform_noise_cuda(int64 size, float* py, float* px, int64 ndat, HYPER hMin, HYPER hMax) {
    int64 n = blockIdx.x * blockDim.x + threadIdx.x;

    if (n < size) {
#ifndef NO_RANDOM_CUDA
        curandState state;

        float min = HYPER_ACCESS(hMin);
        float max = HYPER_ACCESS(hMax);

        curand_init(clock64(), n, 0, &state);

        float dice = curand_uniform(&state);
#else
        float dice = dev_no_random_uniform(n, 0, 1);
#endif

        py[n] = px[n] + (max - min) * dice + min;
    }
}

void VMath::add_uniform_noise(int device, float* py, float* px, int64 ndat, HYPER hMin, HYPER hMax) {
    int64 size = ndat;
    CUDA_CALL(add_uniform_noise, device, size, py, px, ndat, hMin, hMax);
}

//--------------------------------------------------------------------------------------------------

__static__ void gen_normal_random_host(float* py, int64 ndat, HYPER hMean, HYPER hStd) {
    float mean = HYPER_ACCESS(hMean);
    float std = HYPER_ACCESS(hStd);

#ifndef NO_RANDOM_HOST
    std::normal_distribution<float> coin(mean, std);
#endif

    for (int64 nd = 0, n = 0; nd < ndat; nd++, n++) {
#ifdef NO_RANDOM_HOST
        float dice = ms_no_random_normal(mean, std);
#else
        float dice = coin(ms_randGen);
#endif
        py[n] = dice;
    }
}

__global__ void gen_normal_random_cuda(int64 size, float* py, int64 ndat, HYPER hMean, HYPER hStd) {
    int64 n = blockIdx.x * blockDim.x + threadIdx.x;

    if (n < size) {
#ifndef NO_RANDOM_CUDA
        curandState state;

        float mean = HYPER_ACCESS(hMean);
        float std = HYPER_ACCESS(hStd);

        curand_init(clock64(), n, 0, &state);

        float dice = curand_normal(&state);
#else
        float dice = dev_no_random_normal(n, 0, 1);
#endif

        py[n] = mean + dice * std;
    }
}

void VMath::gen_normal_random(int device, float* py, int64 ndat, HYPER hMean, HYPER hStd) {
    int64 size = ndat;
    CUDA_CALL(gen_normal_random, device, size, py, ndat, hMean, hStd);
}

//--------------------------------------------------------------------------------------------------

__static__ void gen_uniform_random_host(float* py, int64 ndat, HYPER hMin, HYPER hMax) {
    float min = HYPER_ACCESS(hMin);
    float max = HYPER_ACCESS(hMax);

#ifndef NO_RANDOM_HOST
    std::uniform_real_distribution<float> coin(min, max);
#endif

    for (int64 nd = 0, n = 0; nd < ndat; nd++, n++) {
#ifdef NO_RANDOM_HOST
        float dice = ms_no_randomuniform(0, 1);
#else
        float dice = coin(ms_randGen);
#endif
        py[n] = (max - min) * dice + min;
    }
}

__global__ void gen_uniform_random_cuda(int64 size, float* py, int64 ndat, HYPER hMin, HYPER hMax) {
    int64 n = blockIdx.x * blockDim.x + threadIdx.x;

    if (n < size) {
#ifndef NO_RANDOM_CUDA
        curandState state;

        float min = HYPER_ACCESS(hMin);
        float max = HYPER_ACCESS(hMax);

        curand_init(clock64(), n, 0, &state);

        float dice = curand_uniform(&state);
#else
        float dice = dev_no_random_uniform(n, 0, 1);
#endif

        py[n] = (max - min) * dice + min;
    }
}

void VMath::gen_uniform_random(int device, float* py, int64 ndat, HYPER hMin, HYPER hMax) {
    int64 size = ndat;
    CUDA_CALL(gen_uniform_random, device, size, py, ndat, hMin, hMax);
}

//--------------------------------------------------------------------------------------------------

__static__ void round_host(float* py, float* px, int64 ndat, int prec) {
    for (int64 nd = 0, n = 0; nd < ndat; nd++, n++) {
        if (prec == 0) {
            py[n] = ::roundf(px[n]);
        }
        else if (prec > 0) {
            float x = px[n];
            for (int k = 0; k < prec; k++) x /= 10.0f;
            x = ::roundf(x);
            for (int k = 0; k < prec; k++) x *= 10.0f;
            py[n] = x;
        }
        else {
            float x = px[n];
            for (int k = 0; k < prec; k++) x *= 10.0f;
            x = ::roundf(x);
            for (int k = 0; k < prec; k++) x /= 10.0f;
            py[n] = x;
        }
    }
}

__global__ void round_cuda(int64 size, float* py, float* px, int64 ndat, int prec) {
    int64 n = blockIdx.x * blockDim.x + threadIdx.x;

    if (n < size) {
        if (prec == 0) {
            py[n] = ::roundf(px[n]);
        }
        else if (prec > 0) {
            float x = px[n];
            for (int k = 0; k < prec; k++) x /= 10.0f;
            x  = ::roundf(x);
            for (int k = 0; k < prec; k++) x *= 10.0f;
            py[n] = x;
        }
        else {
            float x = px[n];
            for (int k = 0; k < prec; k++) x *= 10.0f;
            x = ::roundf(x);
            for (int k = 0; k < prec; k++) x /= 10.0f;
            py[n] = x;
        }
    }
}

void VMath::round(int device, float* py, float* px, int64 ndat, int prec) {
    int64 size = ndat;
    CUDA_CALL(round, device, size, py, px, ndat, prec);
}

//--------------------------------------------------------------------------------------------------

__static__ void codeconv_host(int* py, float* px, int64 nrow, int64 ncol) {
    for (int64 nr = 0, n = 0; nr < nrow; nr++, n++) {
        int code = 0;
        int unit = 1 << (ncol - 1);

        for (int64 nc = 0; nc < ncol; nc++) {
            int64 xpos = nr * ncol + nc;
            if (px[xpos] > 0.5f) code += unit;
            unit = unit >> 1;
        }
        py[n] = code;
    }
}

__global__ void codeconv_cuda(int64 size, int* py, float* px, int64 nrow, int64 ncol) {
    int64 n = blockIdx.x * blockDim.x + threadIdx.x;

    if (n < size) {
        int64 nr = n;

        int code = 0;
        int unit = 1 << (ncol - 1);

        for (int64 nc = 0; nc < ncol; nc++) {
            int64 xpos = nr * ncol + nc;
            if (px[xpos] > 0.5f) code += unit;
            unit = unit >> 1;
        }
        py[n] = code;
    }
}

void VMath::codeconv(int device, int* py, float* px, int64 nrow, int64 ncol) {
    int64 size = nrow;
    CUDA_CALL(codeconv, device, size, py, px, nrow, ncol);
}

//--------------------------------------------------------------------------------------------------

__static__ void cosinesim_host(float* py, float* px1, float* px2, int64 nrow1, int64 nrow2, int64 ncol) {
    for (int64 nr1 = 0, n = 0; nr1 < nrow1; nr1++) {
        for (int64 nr2 = 0; nr2 < nrow2; nr2++, n++) {
            float aa_sum = 0;
            float bb_sum = 0;
            float ab_sum = 0;

            for (int64 nc = 0; nc < ncol; nc++) {
                int64 pos1 = nr1 * ncol + nc;
                int64 pos2 = nr2 * ncol + nc;
                
                float a = px1[pos1];
                float b = px2[pos2];

                aa_sum += a * a;
                bb_sum += b * b;
                ab_sum += a * b;
            }
            py[n] = ab_sum / (::sqrtf(aa_sum) * ::sqrtf(bb_sum) + 1e-20f);
        }
    }
}

__global__ void cosinesim_cuda(int64 size, float* py, float* px1, float* px2, int64 nrow1, int64 nrow2, int64 ncol) {
    int64 n = blockIdx.x * blockDim.x + threadIdx.x;

    if (n < size) {
        int64 nr1 = n / nrow2;
        int64 nr2 = n % nrow2;

        float aa_sum = 0;
        float bb_sum = 0;
        float ab_sum = 0;

        for (int64 nc = 0; nc < ncol; nc++) {
            int64 pos1 = nr1 * ncol + nc;
            int64 pos2 = nr2 * ncol + nc;

            float a = px1[pos1];
            float b = px2[pos2];

            aa_sum += a * a;
            bb_sum += b * b;
            ab_sum += a * b;
        }

        py[n] = ab_sum / (::sqrtf(aa_sum) * ::sqrtf(bb_sum) + 1e-20f);
    }
}

void VMath::cosinesim(int device, float* py, float* px1, float* px2, int64 nrow1, int64 nrow2, int64 ncol) {
    int64 size = nrow1 * nrow2;
    CUDA_CALL(cosinesim, device, size, py, px1, px2, nrow1, nrow2, ncol);
}

//--------------------------------------------------------------------------------------------------

__static__ void selectntop_host(float* py, float* px, int64 nrow, int64 ncol, int64 ntop) {
    for (int64 nr = 0; nr < nrow; nr++) {
        float* prow = px + nr * ncol;

        for (int64 nc = 0; nc < ncol; nc++) {
            float me = prow[nc];
            int64 me_rank = 0;

            for (int64 nc2 = 0; nc2 < ncol; nc2++) {
                if (me < prow[nc2] || (me == prow[nc2] && nc > nc2)) {
                    me_rank++;
                    if (me_rank >= ntop) break;
                }
            }
            if (me_rank < ntop) {
                int64 ypos = nr * ntop + me_rank;
                py[ypos] = me;
            }
        }
    }
}

__global__ void selectntop_cuda(int64 size, float* py, float* px, int64 nrow, int64 ncol, int64 ntop) {
    int64 n = blockIdx.x * blockDim.x + threadIdx.x;

    if (n < size) {
        int64 nr = n / ncol;
        int64 nc = n % ncol;

        float* prow = px + nr * ncol;
        float me = prow[nc];

        int64 me_rank = 0;

        for (int64 nc2 = 0; nc2 < ncol; nc2++) {
            if (me < prow[nc2] || (me == prow[nc2] && nc > nc2)) {
                me_rank++;
                if (me_rank >= ntop) break;
            }
        }
        if (me_rank < ntop) {
            int64 ypos = nr * ntop + me_rank;
            py[ypos] = me;
        }
    }
}

void VMath::selectntop(int device, float* py, float* px, int64 nrow, int64 ncol, int64 ntop) {
    int64 size = nrow * ncol;
    CUDA_CALL(selectntop, device, size, py, px, nrow, ncol, ntop);
}

//--------------------------------------------------------------------------------------------------

__static__ void selectntoparg_host(int* py, float* px, int64 nrow, int64 ncol, int64 ntop) {
    for (int64 nr = 0; nr < nrow; nr++) {
        float* prow = px + nr * ncol;

        for (int64 nc = 0; nc < ncol; nc++) {
            float me = prow[nc];
            int64 me_rank = 0;

            for (int64 nc2 = 0; nc2 < ncol; nc2++) {
                if (me < prow[nc2] || (me == prow[nc2] && nc > nc2)) {
                    me_rank++;
                    if (me_rank >= ntop) break;
                }
            }
            if (me_rank < ntop) {
                int64 ypos = nr * ntop + me_rank;
                py[ypos] = (int)nc;
            }
        }
    }
}

__global__ void selectntoparg_cuda(int64 size, int* py, float* px, int64 nrow, int64 ncol, int64 ntop) {
    int64 n = blockIdx.x * blockDim.x + threadIdx.x;

    if (n < size) {
        int64 nr = n / ncol;
        int64 nc = n % ncol;

        float* prow = px + nr * ncol;
        float me = prow[nc];

        int64 me_rank = 0;

        for (int64 nc2 = 0; nc2 < ncol; nc2++) {
            if (me < prow[nc2] || (me == prow[nc2] && nc > nc2)) {
                me_rank++;
                if (me_rank >= ntop) break;
            }
        }
        if (me_rank < ntop) {
            int64 ypos = nr * ntop + me_rank;
            py[ypos] = (int)nc;
        }
    }
}

void VMath::selectntoparg(int device, int* py, float* px, int64 nrow, int64 ncol, int64 ntop) {
    int64 size = nrow * ncol;
    CUDA_CALL(selectntoparg, device, size, py, px, nrow, ncol, ntop);
}

//--------------------------------------------------------------------------------------------------

__static__ void embed_host(float* py, int* px, float* pw, int64 ndat, int64 nword, int64 nvec, bool position) {
    for (int64 nr = 0, n = 0; nr < ndat; nr++) {
        for (int64 nv = 0; nv < nvec; nv++, n++) {
            int64 nw = position ? (nr % nword) : px[nr];
            if (nw < 0 || nw >= nword) VP_THROW(VERR_OUT_OF_RANGE);
            int64 wpos = nw * nvec + nv;

            py[n] = pw[wpos];
        }
    }
}

__global__ void embed_cuda(int64 size, float* py, int* px, float* pw, int64 ndat, int64 nword, int64 nvec, bool position) {
    int64 n = blockIdx.x * blockDim.x + threadIdx.x;

    if (n < size) {
        int64 nr = n / nvec;
        int64 nv = n % nvec;

        int64 nw = position ? (nr % nword) : px[nr];

        int64 wpos = nw * nvec + nv;

        py[n] = pw[wpos];
    }
}

void VMath::embed(int device, float* py, int* px, float* pw, int64 ndat, int64 nword, int64 nvec, bool position) {
    int64 size = ndat * nvec;
    CUDA_CALL(embed, device, size, py, px, pw, ndat, nword, nvec, position);
}

//--------------------------------------------------------------------------------------------------

__static__ void embed_backward_w_host(float* pgw, float* pgy, int* px, int64 ndat, int64 nword, int64 nvec, bool position) {
    for (int64 nr = 0; nr < ndat; nr++) {
        for (int64 nv = 0; nv < nvec; nv++) {
            if (position) {
                if (nr < nword) {
                    int64 np = nr % nword;
                    int64 wpos = np * nvec + nv;
                    for (int64 n1 = np; n1 < ndat; n1 += nword) {
                        int64 ypos = n1 * nvec + nv;
                        pgw[wpos] += pgy[ypos];
                    }
                }
            }
            else {
                int64 nw = px[nr];
                int64 wpos = nw * nvec + nv;

                for (int64 n1 = 0; n1 < ndat; n1++) {
                    if (px[n1] != nw) continue;
                    if (n1 < nr) break;

                    int64 ypos = n1 * nvec + nv;

                    pgw[wpos] += pgy[ypos];
                }
            }
        }
    }
}

__global__ void embed_backward_w_cuda(int64 size, float* pgw, float* pgy, int* px, int64 ndat, int64 nword, int64 nvec, bool position) {
    int64 n = blockIdx.x * blockDim.x + threadIdx.x;

    if (n < size) {
        int64 nr = n / nvec;
        int64 nv = n % nvec;

        if (position) {
            if (nr < nword) {
                int64 np = nr % nword;
                int64 wpos = np * nvec + nv;
                for (int64 n1 = np; n1 < ndat; n1 += nword) {
                    int64 ypos = n1 * nvec + nv;
                    pgw[wpos] += pgy[ypos];
                }
            }
            return;
        }
        else {
            int64 nw = px[nr];
            int64 wpos = nw * nvec + nv;

            for (int64 n1 = 0; n1 < ndat; n1++) {
                if (px[n1] != nw) continue;
                if (n1 < nr) break;

                int64 ypos = n1 * nvec + nv;

                pgw[wpos] += pgy[ypos];
            }
        }
    }
}

void VMath::embed_backward_w(int device, float* pgw, float* pgy, int* px, int64 ndat, int64 nword, int64 nvec, bool position) {
    int64 size = ndat * nvec;
    CUDA_CALL(embed_backward_w, device, size, pgw, pgy, px, ndat, nword, nvec, position);
}

//--------------------------------------------------------------------------------------------------

__static__ void mult_on_heads_host(float* pp, float* pK, float* pQ, int64 nbat, int64 ntimes, int64 nvec, int64 nhead, int64 npiece, HYPER hCoef) {
    float coef = HYPER_ACCESS(hCoef);

    for (int64 nb = 0, n = 0; nb < nbat; nb++) {
        for (int64 nq = 0; nq < ntimes; nq++) {
            for (int64 nk = 0; nk < ntimes; nk++) {
                for (int64 nh = 0; nh < nhead; nh++, n++) {
                    int64 kpos = (nb * ntimes + nk) * nvec + nh * npiece;
                    int64 qpos = (nb * ntimes + nq) * nvec + nh * npiece;

                    float sum = 0;

                    for (int64 np = 0; np < npiece; np++) {
                        sum += pK[kpos++] * pQ[qpos++];
                    }

                    pp[n] = sum * coef;
                }
            }
        }
    }
}

__global__ void mult_on_heads_cuda(int64 size, float* pp, float* pK, float* pQ, int64 nbat, int64 ntimes, int64 nvec, int64 nhead, int64 npiece, HYPER hCoef) {
    int64 n = blockIdx.x * blockDim.x + threadIdx.x;

    if (n < size) {
        int64 nb = n / (ntimes * ntimes * nhead);
        int64 nq = n / (ntimes * nhead) % ntimes;
        int64 nk = n / nhead % ntimes;
        int64 nh = n % nhead;

        int64 qpos = (nb * ntimes + nq) * nvec + nh * npiece;
        int64 kpos = (nb * ntimes + nk) * nvec + nh * npiece;

        float coef = HYPER_ACCESS(hCoef);

        float sum = 0;

        for (int64 np = 0; np < npiece; np++) {
            sum += pK[kpos++] * pQ[qpos++];
        }

        pp[n] = sum * coef;
    }
}

void VMath::mult_on_heads(int device, float* pp, float* pK, float* pQ, int64 nbat, int64 ntimes, int64 nvec, int64 nhead, int64 npiece, HYPER hCoef) {
    int64 size = nbat * ntimes * ntimes * nhead;
    CUDA_CALL(mult_on_heads, device, size, pp, pK, pQ, nbat, ntimes, nvec, nhead, npiece, hCoef);
}

//--------------------------------------------------------------------------------------------------

__static__ void mult_on_heads_backward_host(float* pgQK, float* pgp, float* pKQ, int64 nbat, int64 ntimes, int64 nvec, int64 nhead, int64 npiece) {
    for (int64 nb = 0; nb < nbat; nb++) {
        for (int64 nt = 0; nt < ntimes; nt++) { // nq for K, nk for Q
            for (int64 nh = 0; nh < nhead; nh++) {
                for (int64 np = 0; np < npiece; np++) {
                    int64 nq = nt;
                    float sum_q = 0;
                    for (int64 nk = 0; nk < ntimes; nk++) {
                        int64 kpos = (nb * ntimes + nk) * nvec + nh * npiece + np;
                        int64 ypos = ((nb * ntimes + nq) * ntimes + nk) * nhead + nh;

                        sum_q += pgp[ypos] * pKQ[kpos];
                    }
                    int64 qpos = (nb * ntimes + nq) * nvec + nh * npiece + np;
                    pgQK[qpos] = sum_q;
                }
            }
        }
    }
}

__global__ void mult_on_heads_backward_cuda(int64 size, float* pgQK, float* pgp, float* pKQ, int64 nbat, int64 ntimes, int64 nvec, int64 nhead, int64 npiece) {
    int64 n = blockIdx.x * blockDim.x + threadIdx.x;

    if (n < size) {
        int64 nb = n / (ntimes * nhead * npiece) % nbat;
        int64 nt = n / (nhead * npiece) % ntimes;
        int64 nh = n / npiece % nhead;
        int64 np = n % npiece;

        int64 nq = nt;
        float sum_q = 0;
        for (int64 nk = 0; nk < ntimes; nk++) {
            int64 kpos = (nb * ntimes + nk) * nvec + nh * npiece + np;
            int64 ypos = ((nb * ntimes + nq) * ntimes + nk) * nhead + nh;

            sum_q += pgp[ypos] * pKQ[kpos];
        }
        int64 qpos = (nb * ntimes + nq) * nvec + nh * npiece + np;
        pgQK[qpos] = sum_q;
    }
}

void VMath::mult_on_heads_backward(int device, float* pgQK, float* pgp, float* pKQ, int64 nbat, int64 ntimes, int64 nvec, int64 nhead, int64 npiece) {
    int64 size = nbat * ntimes * nhead * npiece;
    CUDA_CALL(mult_on_heads_backward, device, size, pgQK, pgp, pKQ, nbat, ntimes, nvec, nhead, npiece);
}

//--------------------------------------------------------------------------------------------------

__static__ void set_mh_attention_mask_host(float* pp, int64 ndat, int64 ntimes, int64 nhead, bool forward) {
    for (int64 nd = 0; nd < ndat; nd++) {
        for (int64 nh = 0; nh < nhead; nh++) {
            for (int64 nq = 0; nq < ntimes; nq++) {
                for (int64 nk = nq + 1; nk < ntimes; nk++) {
                    int64 ppos = ((nd * ntimes + nq) * ntimes + nk) * nhead + nh;
                    pp[ppos] = forward ? -INFINITY : 0;
                }
            }
        }
    }
}

__global__ void set_mh_attention_mask_cuda(int64 size, float* pp, int64 ndat, int64 ntimes, int64 nhead, bool forward) {
    int64 n = blockIdx.x * blockDim.x + threadIdx.x;

    if (n < size) {
        int64 nq = n / (ntimes * nhead) % ntimes;
        int64 nk = n / nhead % ntimes;

        if (nq < nk) pp[n] = forward ? -INFINITY : 0;
    }
}

void VMath::set_mh_attention_mask(int device, float* pp, int64 ndat, int64 ntimes, int64 nhead, bool forward) {
    int64 size = ndat * ntimes * ntimes * nhead;
    CUDA_CALL(set_mh_attention_mask, device, size, pp, ndat, ntimes, nhead, forward);
}

//--------------------------------------------------------------------------------------------------

__static__ void softmax_direct_on_axis_host(float* pp, int64 nrow, int64 nvec, int64 ncol) {
    for (int64 nr = 0; nr < nrow; nr++) {
        for (int64 nc = 0; nc < ncol; nc++) {
            float max_term = -FLT_MAX;
            float exp_sum = 0;

            int64 ppos0 = nr * nvec * ncol + nc;

            for (int64 nv = 0, ppos = ppos0; nv < nvec; nv++, ppos += ncol) {
                if (pp[ppos] > max_term) max_term = pp[ppos];
            }

            for (int64 nv = 0, ppos = ppos0; nv < nvec; nv++, ppos += ncol) {
                exp_sum += ::expf(pp[ppos] - max_term);
            }

            for (int64 nv = 0, ppos = ppos0; nv < nvec; nv++, ppos += ncol) {
                pp[ppos] = ::expf(pp[ppos] - max_term) / exp_sum;
            }
        }
    }
}

__global__ void softmax_direct_on_axis_cuda(int64 size, float* pp, int64 nrow, int64 nvec, int64 ncol) {
    int64 n = blockIdx.x * blockDim.x + threadIdx.x;

    if (n < size) {
        int64 nr = n / ncol;
        int64 nc = n % ncol;

        float max_term = -FLT_MAX;
        float exp_sum = 0;

        int64 ppos0 = nr * nvec * ncol + nc;

        for (int64 nv = 0, ppos = ppos0; nv < nvec; nv++, ppos += ncol) {
            if (pp[ppos] > max_term) max_term = pp[ppos];
        }

        for (int64 nv = 0, ppos = ppos0; nv < nvec; nv++, ppos += ncol) {
            exp_sum += ::expf(pp[ppos] - max_term);
        }

        for (int64 nv = 0, ppos = ppos0; nv < nvec; nv++, ppos += ncol) {
            pp[ppos] = ::expf(pp[ppos] - max_term) / exp_sum;
        }
    }
}

void VMath::softmax_direct_on_axis(int device, float* pp, int64 nrow, int64 nvec, int64 ncol) {
    int64 size = nrow * ncol;
    CUDA_CALL(softmax_direct_on_axis, device, size, pp, nrow, nvec, ncol);
}

//--------------------------------------------------------------------------------------------------

__static__ void softmax_direct_on_axis_backward_host(float* pgp, float* pp, int64 nbat, int64 ntimes, int64 nhead, HYPER hCoef) {
    float coef = HYPER_ACCESS(hCoef);

    for (int64 nb = 0; nb < nbat; nb++) {
        for (int64 nq = 0; nq < ntimes; nq++) {
            for (int64 nh = 0; nh < nhead; nh++) {
                float sum = 0;

                for (int64 nk = 0; nk < ntimes; nk++) {
                    int64 ppos = ((nb * ntimes + nq) * ntimes + nk) * nhead + nh;

                    sum += pp[ppos] * pgp[ppos];
                }

                for (int64 nk = 0; nk < ntimes; nk++) {
                    int64 ppos = ((nb * ntimes + nq) * ntimes + nk) * nhead + nh;

                    pgp[ppos] = coef * pp[ppos] * (pgp[ppos] - sum);
                }
            }
        }
    }
}

__global__ void softmax_direct_on_axis_backward_cuda(int64 size, float* pgp, float* pp, int64 nbat, int64 ntimes, int64 nhead, HYPER hCoef) {
    int64 n = blockIdx.x * blockDim.x + threadIdx.x;

    if (n < size) {
        int64 nb = n / (ntimes * nhead);
        int64 nq = n / nhead % ntimes;
        int64 nh = n % nhead;

        float coef = HYPER_ACCESS(hCoef);

        float sum = 0;

        for (int64 nk = 0; nk < ntimes; nk++) {
            int64 ppos = ((nb * ntimes + nq) * ntimes + nk) * nhead + nh;

            sum += pp[ppos] * pgp[ppos];
        }

        for (int64 nk = 0; nk < ntimes; nk++) {
            int64 ppos = ((nb * ntimes + nq) * ntimes + nk) * nhead + nh;

            pgp[ppos] = coef * pp[ppos] * (pgp[ppos] - sum);
        }
    }
}

void VMath::softmax_direct_on_axis_backward(int device, float* pgp, float* pp, int64 nbat, int64 ntimes, int64 nhead, HYPER hCoef) {
    int64 size = nbat * ntimes * nhead;
    CUDA_CALL(softmax_direct_on_axis_backward, device, size, pgp, pp, nbat, ntimes, nhead, hCoef);
}

//--------------------------------------------------------------------------------------------------

__static__ void mix_values_host(float* py, float* pp, float* pV, int64 nbat, int64 ntimes, int64 nvec, int64 nhead, int64 npiece) {
    for (int64 nb = 0; nb < nbat; nb++) {
        for (int64 nq = 0; nq < ntimes; nq++) {
            for (int64 nh = 0; nh < nhead; nh++) {
                for (int64 np = 0; np < npiece; np++) {
                    float sum = 0;

                    for (int64 nk = 0; nk < ntimes; nk++) {
                        int64 ppos = ((nb * ntimes + nq) * ntimes + nk) * nhead + nh;
                        int64 vpos = (nb * ntimes + nk) * nvec + nh * npiece + np;

                        sum += pp[ppos] * pV[vpos];
                    }

                    int64 ypos = (((nb * ntimes) + nq) * nhead + nh) * npiece + np;
                    py[ypos] = sum;
                }
            }
        }
    }
}

__global__ void mix_values_cuda(int64 size, float* py, float* pp, float* pV, int64 nbat, int64 ntimes, int64 nvec, int64 nhead, int64 npiece) {
    int64 n = blockIdx.x * blockDim.x + threadIdx.x;

    if (n < size) {
        int64 nb = n / (ntimes * nhead * npiece);
        int64 nq = n / (nhead * npiece) % ntimes;
        int64 nh = n / npiece % nhead;
        int64 np = n % npiece;

        float sum = 0;

        for (int64 nk = 0; nk < ntimes; nk++) {
            int64 ppos = ((nb * ntimes + nq) * ntimes + nk) * nhead + nh;
            int64 vpos = (nb * ntimes + nk) * nvec + nh * npiece + np;

            sum += pp[ppos] * pV[vpos];
        }

        int64 ypos = (((nb * ntimes) + nq) * nhead + nh) * npiece + np;
        py[ypos] = sum;
    }
}

void VMath::mix_values(int device, float* py, float* pp, float* pV, int64 nbat, int64 ntimes, int64 nvec, int64 nhead, int64 npiece) {
    int64 size = nbat * ntimes * nhead * npiece;
    CUDA_CALL(mix_values, device, size, py, pp, pV, nbat, ntimes, nvec, nhead, npiece);
}

//--------------------------------------------------------------------------------------------------

__static__ void mix_values_backward_prop_host(float* pgp, float* pgy, float* pV, int64 nbat, int64 ntimes, int64 nvec, int64 nhead, int64 npiece) {
    for (int64 nb = 0; nb < nbat; nb++) {
        for (int64 nq = 0; nq < ntimes; nq++) {
            for (int64 nk = 0; nk < ntimes; nk++) {
                for (int64 nh = 0; nh < nhead; nh++) {
                    float sum = 0;

                    for (int64 np = 0; np < npiece; np++) {
                        int64 ypos = (((nb * ntimes) + nq) * nhead + nh) * npiece + np;
                        int64 vpos = (nb * ntimes + nk) * nvec + nh * npiece + np;

                        sum += pgy[ypos] * pV[vpos];
                    }

                    int64 ppos = ((nb * ntimes + nq) * ntimes + nk) * nhead + nh;
                    pgp[ppos] = sum;
                }
            }
        }
    }
}

__global__ void mix_values_backward_prop_cuda(int64 size, float* pgp, float* pgy, float* pV, int64 nbat, int64 ntimes, int64 nvec, int64 nhead, int64 npiece) {
    int64 n = blockIdx.x * blockDim.x + threadIdx.x;

    if (n < size) {
        int64 nb = n / (ntimes * ntimes * nhead);
        int64 nq = n / (ntimes * nhead) % ntimes;
        int64 nk = n / nhead % ntimes;
        int64 nh = n % nhead;

        float sum = 0;

        for (int64 np = 0; np < npiece; np++) {
            int64 ypos = (((nb * ntimes) + nq) * nhead + nh) * npiece + np;
            int64 vpos = (nb * ntimes + nk) * nvec + nh * npiece + np;

            sum += pgy[ypos] * pV[vpos];
        }

        int64 ppos = ((nb * ntimes + nq) * ntimes + nk) * nhead + nh;
        pgp[ppos] = sum;
    }
}

void VMath::mix_values_backward_prop(int device, float* pgp, float* pgy, float* pV, int64 nbat, int64 ntimes, int64 nvec, int64 nhead, int64 npiece) {
    int64 size = nbat * ntimes * ntimes * nhead;
    CUDA_CALL(mix_values_backward_prop, device, size, pgp, pgy, pV, nbat, ntimes, nvec, nhead, npiece);
}

//--------------------------------------------------------------------------------------------------

__static__ void mix_values_backward_value_host(float* pgV, float* pgy, float* pp, int64 nbat, int64 ntimes, int64 nvec, int64 nhead, int64 npiece) {
    for (int64 nb = 0; nb < nbat; nb++) {
        for (int64 nk = 0; nk < ntimes; nk++) {
            for (int64 nh = 0; nh < nhead; nh++) {
                for (int64 np = 0; np < npiece; np++) {
                    float sum = 0;

                    for (int64 nq = 0; nq < ntimes; nq++) {
                        int64 ppos = ((nb * ntimes + nq) * ntimes + nk) * nhead + nh;
                        int64 ypos = (((nb * ntimes) + nq) * nhead + nh) * npiece + np;

                        sum += pp[ppos] * pgy[ypos];
                    }

                    int64 vpos = (nb * ntimes + nk) * nvec + nh * npiece + np;
                    pgV[vpos] = sum;
                }
            }
        }
    }
}

__global__ void mix_values_backward_value_cuda(int64 size, float* pgV, float* pgy, float* pp, int64 nbat, int64 ntimes, int64 nvec, int64 nhead, int64 npiece) {
    int64 n = blockIdx.x * blockDim.x + threadIdx.x;

    if (n < size) {
        int64 nb = n / (ntimes * nhead * npiece);
        int64 nk = n / (nhead * npiece) % ntimes;
        int64 nh = n / npiece % nhead;
        int64 np = n % npiece;

        float sum = 0;

        for (int64 nq = 0; nq < ntimes; nq++) {
            int64 ppos = ((nb * ntimes + nq) * ntimes + nk) * nhead + nh;
            int64 ypos = (((nb * ntimes) + nq) * nhead + nh) * npiece + np;

            sum += pp[ppos] * pgy[ypos];
        }

        int64 vpos = (nb * ntimes + nk) * nvec + nh * npiece + np;
        pgV[vpos] = sum;
    }
}

void VMath::mix_values_backward_value(int device, float* pgV, float* pgy, float* pp, int64 nbat, int64 ntimes, int64 nvec, int64 nhead, int64 npiece) {
    int64 size = nbat * ntimes * nhead * npiece;
    CUDA_CALL(mix_values_backward_value, device, size, pgV, pgy, pp, nbat, ntimes, nvec, nhead, npiece);
}

//--------------------------------------------------------------------------------------------------

__static__ void parallel_concat_host(float* py, float* px1, float* px2, int64 nrow, int64 ncol1, int64 ncol2, int64 nrest) {
    for (int64 nr = 0, n = 0; nr < nrow; nr++) {
        for (int64 nc = 0; nc < ncol1; nc++) {
            for (int64 nn = 0; nn < nrest; nn++, n++) {
                py[n] = px1[(nr * ncol1 + nc) * nrest + nn];
            }
        }

        for (int64 nc = 0; nc < ncol2; nc++) {
            for (int64 nn = 0; nn < nrest; nn++, n++) {
                py[n] = px2[(nr * ncol2 + nc) * nrest + nn];
            }
        }
    }
}

__global__ void parallel_concat_cuda(int64 size, float* py, float* px1, float* px2, int64 nrow, int64 ncol1, int64 ncol2, int64 nrest) {
    int64 n = blockIdx.x * blockDim.x + threadIdx.x;

    if (n < size) {
        int64 nr = n / ((ncol1 + ncol2) * nrest);
        int64 nc = n / nrest % (ncol1 + ncol2);
        int64 nn = n % nrest;

        if (nc < ncol1) py[n] = px1[(nr * ncol1 + nc) * nrest + nn];
        else py[n] = px2[(nr * ncol2 + (nc - ncol1)) * nrest + nn];
    }
}

void VMath::parallel_concat(int device, float* py, float* px1, float* px2, int64 nrow, int64 ncol1, int64 ncol2, int64 nrest) {
    int64 size = nrow * (ncol1 + ncol2) * nrest;
    CUDA_CALL(parallel_concat, device, size, py, px1, px2, nrow, ncol1, ncol2, nrest);
}

//--------------------------------------------------------------------------------------------------

__static__ void parallel_concat_backward_x1_host(float* pgx, float* pgy, int64 nrow, int64 ncol1, int64 ncol2, int64 nrest) {
    for (int64 nr = 0, n = 0; nr < nrow; nr++) {
        for (int64 nc = 0; nc < ncol1; nc++) {
            for (int64 nn = 0; nn < nrest; nn++, n++) {
                pgx[n] = pgy[(nr * (ncol1 + ncol2) + nc) * nrest + nn];
            }
        }
    }
}

__global__ void parallel_concat_backward_x1_cuda(int64 size, float* pgx, float* pgy, int64 nrow, int64 ncol1, int64 ncol2, int64 nrest) {
    int64 n = blockIdx.x * blockDim.x + threadIdx.x;

    if (n < size) {
        int64 nr = n / (ncol1 * nrest);
        int64 nc = n / nrest % ncol1;
        int64 nn = n % nrest;

        pgx[n] = pgy[(nr * (ncol1 + ncol2) + nc) * nrest + nn];
    }
}

void VMath::parallel_concat_backward_x1(int device, float* pgx, float* pgy, int64 nrow, int64 ncol1, int64 ncol2, int64 nrest) {
    int64 size = nrow * ncol1 * nrest;
    CUDA_CALL(parallel_concat_backward_x1, device, size, pgx, pgy, nrow, ncol1, ncol2, nrest);
}

//--------------------------------------------------------------------------------------------------

__static__ void parallel_concat_backward_x2_host(float* pgx, float* pgy, int64 nrow, int64 ncol1, int64 ncol2, int64 nrest) {
    for (int64 nr = 0, n = 0; nr < nrow; nr++) {
        for (int64 nc = 0; nc < ncol2; nc++) {
            for (int64 nn = 0; nn < nrest; nn++, n++) {
                pgx[n] = pgy[(nr * (ncol1 + ncol2) + (ncol1 + nc)) * nrest + nn];
            }
        }
    }
}

__global__ void parallel_concat_backward_x2_cuda(int64 size, float* pgx, float* pgy, int64 nrow, int64 ncol1, int64 ncol2, int64 nrest) {
    int64 n = blockIdx.x * blockDim.x + threadIdx.x;

    if (n < size) {
        int64 nr = n / (ncol2 * nrest);
        int64 nc = n / nrest % ncol2;
        int64 nn = n % nrest;

        pgx[n] = pgy[(nr * (ncol1 + ncol2) + (ncol1 + nc)) * nrest + nn];
    }
}

void VMath::parallel_concat_backward_x2(int device, float* pgx, float* pgy, int64 nrow, int64 ncol1, int64 ncol2, int64 nrest) {
    int64 size = nrow * ncol2 * nrest;
    CUDA_CALL(parallel_concat_backward_x2, device, size, pgx, pgy, nrow, ncol1, ncol2, nrest);
}

//--------------------------------------------------------------------------------------------------

__static__ void stack_host(float* py, float* px, int64 nbat, int64 nyrow, int64 nxrow, int64 ncol, int64 nfrom) {
    for (int64 nb = 0; nb < nbat; nb++) {
        for (int64 nr = 0; nr < nxrow; nr++) {
            for (int64 nc = 0; nc < ncol; nc++) {
                int64 xpos = (nb * nxrow + nr) * ncol + nc;
                int64 ypos = (nb * nyrow + (nr + nfrom)) * ncol + nc;

                py[ypos] = px[xpos];
            }
        }
    }
}

__global__ void stack_cuda(int64 size, float* py, float* px, int64 nbat, int64 nyrow, int64 nxrow, int64 ncol, int64 nfrom) {
    int64 n = blockIdx.x * blockDim.x + threadIdx.x;

    if (n < size) {
        int64 nb = n / (nxrow * ncol);
        int64 nr = n / ncol % nxrow;
        int64 nc = n % ncol;

        int64 xpos = (nb * nxrow + nr) * ncol + nc;
        int64 ypos = (nb * nyrow + (nr + nfrom)) * ncol + nc;

        py[ypos] = px[xpos];
    }
}

void VMath::stack(int device, float* py, float* px, int64 nbat, int64 nyrow, int64 nxrow, int64 ncol, int64 nfrom) {
    int64 size = nbat * nxrow * ncol;
    CUDA_CALL(stack, device, size, py, px, nbat, nyrow, nxrow, ncol, nfrom);
}

//--------------------------------------------------------------------------------------------------

__static__ void stack_backward_host(float* pgx, float* pgy, int64 nbat, int64 nyrow, int64 nxrow, int64 ncol, int64 nfrom) {
    for (int64 nb = 0, n = 0; nb < nbat; nb++) {
        for (int64 nr = 0; nr < nxrow; nr++) {
            for (int64 nc = 0; nc < ncol; nc++, n++) {
                int64 ypos = (nb * nyrow + (nr + nfrom)) * ncol + nc;
                pgx[n] = pgy[ypos];
            }
        }
    }
}

__global__ void stack_backward_cuda(int64 size, float* pgx, float* pgy, int64 nbat, int64 nyrow, int64 nxrow, int64 ncol, int64 nfrom) {
    int64 n = blockIdx.x * blockDim.x + threadIdx.x;

    if (n < size) {
        int64 nb = n / (nxrow * ncol);
        int64 nr = n / ncol % nxrow;
        int64 nc = n % ncol;

        int64 ypos = (nb * nyrow + (nr + nfrom)) * ncol + nc;
        pgx[n] = pgy[ypos];
    }
}

void VMath::stack_backward(int device, float* pgx, float* pgy, int64 nbat, int64 nyrow, int64 nxrow, int64 ncol, int64 nfrom) {
    int64 size = nbat * nxrow * ncol;
    CUDA_CALL(stack_backward, device, size, pgx, pgy, nbat, nyrow, nxrow, ncol, nfrom);
}

//--------------------------------------------------------------------------------------------------

__static__ void concat_host(float* py, float* px1, float* px2, int64 nrow, int64 ncol1, int64 ncol2, int64 nrest) {
    for (int64 nr = 0, n = 0; nr < nrow; nr++) {
        for (int64 nc = 0; nc < ncol1; nc++) {
            for (int64 nn = 0; nn < nrest; nn++, n++) {
                int64 xpos = (nr * ncol1 + nc) * nrest + nn;
                py[n] = px1[xpos];
            }
        }
        for (int64 nc = 0; nc < ncol2; nc++) {
            for (int64 nn = 0; nn < nrest; nn++, n++) {
                int64 xpos = (nr * ncol2 + nc) * nrest + nn;
                py[n] = px2[xpos];
            }
        }
    }
}

__global__ void concat_cuda(int64 size, float* py, float* px1, float* px2, int64 nrow, int64 ncol1, int64 ncol2, int64 nrest) {
    int64 n = blockIdx.x * blockDim.x + threadIdx.x;

    if (n < size) {
        int64 nr = n / ((ncol1 + ncol2) * nrest);
        int64 nc = n / nrest  % (ncol1 + ncol2);
        int64 nn = n % nrest;

        if (nc < ncol1) {
            int64 xpos = (nr * ncol1 + nc) * nrest + nn;
            py[n] = px1[xpos];
        }
        else {
            int64 xpos = (nr * ncol2 + (nc - ncol1)) * nrest + nn;
            py[n] = px2[xpos];
        }
    }
}

void VMath::concat(int device, float* py, float* px1, float* px2, int64 nrow, int64 ncol1, int64 ncol2, int64 nrest) {
    int64 size = nrow * (ncol1 + ncol2) * nrest;
    CUDA_CALL(concat, device, size, py, px1, px2, nrow, ncol1, ncol2, nrest);
}

//--------------------------------------------------------------------------------------------------

__static__ void concat_backward_x1_host(float* pgx, float* pgy, int64 nrow, int64 ncol1, int64 ncol2, int64 nrest) {
    for (int64 nr = 0, n = 0; nr < nrow; nr++) {
        for (int64 nc = 0; nc < ncol1; nc++) {
            for (int64 nn = 0; nn < nrest; nn++, n++) {
                int64 ypos = (nr * (ncol1 + ncol2) + nc) * nrest + nn;
                pgx[n] = pgy[ypos];
            }
        }
    }
}

__global__ void concat_backward_x1_cuda(int64 size, float* pgx, float* pgy, int64 nrow, int64 ncol1, int64 ncol2, int64 nrest) {
    int64 n = blockIdx.x * blockDim.x + threadIdx.x;

    if (n < size) {
        int64 nr = n / (ncol1 * nrest);
        int64 nc = n / nrest  % ncol1;
        int64 nn = n % nrest;

        int64 ypos = (nr * (ncol1 + ncol2) + nc) * nrest + nn;
        pgx[n] = pgy[ypos];
    }
}

void VMath::concat_backward_x1(int device, float* pgx, float* pgy, int64 nrow, int64 ncol1, int64 ncol2, int64 nrest) {
    int64 size = nrow * ncol1 * nrest;
    CUDA_CALL(concat_backward_x1, device, size, pgx, pgy, nrow, ncol1, ncol2, nrest);
}

//--------------------------------------------------------------------------------------------------

__static__ void concat_backward_x2_host(float* pgx, float* pgy, int64 nrow, int64 ncol1, int64 ncol2, int64 nrest) {
    for (int64 nr = 0, n = 0; nr < nrow; nr++) {
        for (int64 nc = 0; nc < ncol2; nc++) {
            for (int64 nn = 0; nn < nrest; nn++, n++) {
                int64 ypos = (nr * (ncol1 + ncol2) + (ncol1 + nc)) * nrest + nn;
                pgx[n] = pgy[ypos];
            }
        }
    }
}

__global__ void concat_backward_x2_cuda(int64 size, float* pgx, float* pgy, int64 nrow, int64 ncol1, int64 ncol2, int64 nrest) {
    int64 n = blockIdx.x * blockDim.x + threadIdx.x;

    if (n < size) {
        int64 nr = n / (ncol2 * nrest);
        int64 nc = n / nrest % ncol2;
        int64 nn = n % nrest;

        int64 ypos = (nr * (ncol1 + ncol2) + (ncol1 + nc)) * nrest + nn;
        pgx[n] = pgy[ypos];
    }
}

void VMath::concat_backward_x2(int device, float* pgx, float* pgy, int64 nrow, int64 ncol1, int64 ncol2, int64 nrest) {
    int64 size = nrow * ncol2 * nrest;
    CUDA_CALL(concat_backward_x2, device, size, pgx, pgy, nrow, ncol1, ncol2, nrest);
}

//--------------------------------------------------------------------------------------------------

__static__ void undo_concat_host(float* py1, float* py2, float* px, int64 nrow, int64 ncol1, int64 ncol2) {
    for (int64 nr = 0; nr < nrow; nr++) {
        for (int64 nc = 0; nc < ncol1; nc++) {
            int64 ypos = nr * ncol1 + nc;
            int64 xpos = nr * (ncol1 + ncol2) + nc;
            py1[ypos] = px[xpos];
        }
        for (int64 nc = 0; nc < ncol2; nc++) {
            int64 ypos = nr * ncol2 + nc;
            int64 xpos = nr * (ncol1 + ncol2) + nc;
            py2[ypos] = px[xpos];
        }
    }
}

__global__ void undo_concat_cuda(int64 size, float* py1, float* py2, float* px, int64 nrow, int64 ncol1, int64 ncol2) {
    int64 n = blockIdx.x * blockDim.x + threadIdx.x;

    if (n < size) {
        int64 nr = n / (ncol1 + ncol2);
        int64 nc = n % (ncol1 + ncol2);

        if (nc < ncol1) {
            int64 ypos = nr * ncol1 + nc;
            int64 xpos = nr * (ncol1 + ncol2) + nc;
            py1[ypos] = px[xpos];
        }
        else {
            int64 ypos = nr * ncol2 + (nc - ncol1);
            int64 xpos = nr * (ncol1 + ncol2) + nc;
            py2[ypos] = px[xpos];
        }
    }
}

void VMath::undo_concat(int device, float* py1, float* py2, float* px, int64 nrow, int64 ncol1, int64 ncol2) {
    int64 size = nrow * (ncol1 + ncol2);
    CUDA_CALL(undo_concat, device, size, py1, py2, px, nrow, ncol1, ncol2);
}

//--------------------------------------------------------------------------------------------------

__static__ void count_true_host(int* pc, float* px, int64 nrow, int64 ncol) {
    for (int64 n = 0; n < nrow; n++) {
        int true_cnt = 0;
        for (int64 nc = 0; nc < ncol; nc++) {
            int64 xpos = n * ncol + nc;
            if (px[xpos]) true_cnt++;
        }
        pc[n] = true_cnt;
    }
}

__global__ void count_true_cuda(int64 size, int* pc, float* px, int64 nrow, int64 ncol) {
    int64 n = blockIdx.x * blockDim.x + threadIdx.x;

    if (n < size) {
        int true_cnt = 0;
        for (int64 nc = 0; nc < ncol; nc++) {
            int64 xpos = n * ncol + nc;
            if (px[xpos]) true_cnt++;
        }
        pc[n] = true_cnt;
    }
}

void VMath::count_true(int device, int* pc, float* px, int64 nrow, int64 ncol) {
    int64 size = nrow;
    CUDA_CALL(count_true, device, size, pc, px, nrow, ncol);
}

//--------------------------------------------------------------------------------------------------

__static__ void select_max_host(int* pm, int* pc, int64 nrow) {
    int max_cnt = 0;
    for (int64 nr = 0; nr < nrow; nr++) {
        if (pc[nr] > max_cnt) max_cnt = pc[nr];
    }
    pm[0] = max_cnt;
}

__global__ void select_max_cuda(int64 size, int* pm, int* pc, int64 nrow) {
    int64 n = blockIdx.x * blockDim.x + threadIdx.x;

    if (n < size) {
        int max_cnt = 0;
        for (int64 nr = 0; nr < nrow; nr++) {
            if (pc[nr] > max_cnt) max_cnt = pc[nr];
        }
        pm[0] = max_cnt;
    }
}

void VMath::select_max(int device, int* pm, int* pc, int64 nrow) {
    int64 size = 1;
    CUDA_CALL(select_max, device, size, pm, pc, nrow);
}

//--------------------------------------------------------------------------------------------------

__static__ void to_filter_host(int* py, float* px, int64 nrow, int64 nxcol, int64 nycol) {
    for (int64 n = 0; n < nrow; n++) {
        int nth = 0;
        int64 ybase = n * nycol;

        for (int64 nc = 0; nc < nxcol; nc++) {
            int64 xpos = n * nxcol + nc;
            if (px[xpos]) {
                int64 ypos = ybase + nth++;
                py[ypos] = (int)nc;
            }
        }
        for (; nth < nycol; nth++) {
            int64 ypos = ybase + nth;
            py[ypos] = -1;
        }
    }
}

__global__ void to_filter_cuda(int64 size, int* py, float* px, int64 nrow, int64 nxcol, int64 nycol) {
    int64 n = blockIdx.x * blockDim.x + threadIdx.x;

    if (n < size) {
        int nth = 0;
        int64 ybase = n * nycol;

        for (int64 nc = 0; nc < nxcol; nc++) {
            int64 xpos = n * nxcol + nc;
            if (px[xpos]) {
                int64 ypos = ybase + nth++;
                py[ypos] = (int)nc;
            }
        }
        for (; nth < nycol; nth++) {
            int64 ypos = ybase + nth;
            py[ypos] = -1;
        }
    }
}

void VMath::to_filter(int device, int* py, float* px, int64 nrow, int64 nxcol, int64 nycol) {
    int64 size = nrow;
    CUDA_CALL(to_filter, device, size, py, px, nrow, nxcol, nycol);
}

//--------------------------------------------------------------------------------------------------

__static__ void extract_host(float* py, float* px, int64 nrow, int64 nxvec, int64 index, int64 nyvec, int64 ncol) {
    for (int64 nr = 0, n = 0; nr < nrow; nr++) {
        for (int64 nv = 0; nv < nyvec; nv++) {
            for (int64 nc = 0; nc < ncol; nc++, n++) {
                int64 xpos = (nr * nxvec + (index + nv)) * ncol + nc;
                py[n] = px[xpos];
            }
        }
    }
}

__global__ void extract_cuda(int64 size, float* py, float* px, int64 nrow, int64 nxvec, int64 index, int64 nyvec, int64 ncol) {
    int64 n = blockIdx.x * blockDim.x + threadIdx.x;

    if (n < size) {
        int64 nr = n / (nyvec * ncol);
        int64 nv = n / ncol % nyvec;
        int64 nc = n % ncol;

        int64 xpos = (nr * nxvec + (index + nv)) * ncol + nc;
        py[n] = px[xpos];
    }
}

void VMath::extract(int device, float* py, float* px, int64 nrow, int64 nxvec, int64 index, int64 nyvec, int64 ncol) {
    int64 size = nrow * nyvec * ncol;
    CUDA_CALL(extract, device, size, py, px, nrow, nxvec, index, nyvec, ncol);
}

//--------------------------------------------------------------------------------------------------

__static__ void extract_backward_host(float* pgx, float* pgy, int64 nrow, int64 nxvec, int64 index, int64 nyvec, int64 ncol) {
    for (int64 nr = 0, n = 0; nr < nrow; nr++) {
        for (int64 nv = 0; nv < nxvec; nv++) {
            for (int64 nc = 0; nc < ncol; nc++, n++) {
                if (nv < index || nv >= index + nyvec) pgx[n] = 0;
                else {
                    int64 ypos = (nr * nyvec + (nv - index)) * ncol + nc;
                    pgx[n] = pgy[ypos];
                }
            }
        }
    }
}

__global__ void extract_backward_cuda(int64 size, float* pgx, float* pgy, int64 nrow, int64 nxvec, int64 index, int64 nyvec, int64 ncol) {
    int64 n = blockIdx.x * blockDim.x + threadIdx.x;

    if (n < size) {
        int64 nr = n / (nxvec * ncol);
        int64 nv = n / ncol % nxvec;
        int64 nc = n % ncol;

        if (nv < index || nv >= index + nyvec) pgx[n] = 0;
        else {
            int64 ypos = (nr * nyvec + (nv - index)) * ncol + nc;
            pgx[n] = pgy[ypos];
        }
    }
}

void VMath::extract_backward(int device, float* pgx, float* pgy, int64 nrow, int64 nxvec, int64 index, int64 nyvec, int64 ncol) {
    int64 size = nrow * nxvec * ncol;
    CUDA_CALL(extract_backward, device, size, pgx, pgy, nrow, nxvec, index, nyvec, ncol);
}

//--------------------------------------------------------------------------------------------------

__static__ void subvector_host(float* py, float* px, int64 nrow, int64 ncol, int64 nfrom, int64 ncount) {
    for (int64 nr = 0; nr < nrow; nr++) {
        for (int64 nc = 0; nc < ncount; nc++) {
            int64 xpos = nr * ncol + nfrom + nc;
            int64 ypos = nr * ncount + nc;

            py[ypos] = px[xpos];
        }
    }
}

__global__ void subvector_cuda(int64 size, float* py, float* px, int64 nrow, int64 ncol, int64 nfrom, int64 ncount) {
    int64 n = blockIdx.x * blockDim.x + threadIdx.x;

    if (n < size) {
        int64 nr = n / ncount;
        int64 nc = n % ncount;

        int64 xpos = nr * ncol + nfrom + nc;
        int64 ypos = nr * ncount + nc;

        py[ypos] = px[xpos];
    }
}

void VMath::subvector(int device, float* py, float* px, int64 nrow, int64 ncol, int64 nfrom, int64 ncount) {
    int64 size = nrow * ncount;
    CUDA_CALL(subvector, device, size, py, px, nrow, ncol, nfrom, ncount);
}

//--------------------------------------------------------------------------------------------------

__static__ void subvector_backward_host(float* pgx, float* pgy, int64 nrow, int64 ncol, int64 nfrom, int64 ncount) {
    for (int64 nr = 0, n = 0; nr < nrow; nr++) {
        for (int64 nc = 0; nc < ncol; nc++, n++) {
            if (nc >= nfrom && nc < nfrom - ncount) {
                int64 ypos = nr * ncount + (nc - nfrom);
                pgx[n] = pgy[ypos];
            }
            else pgx[n] = 0;
        }
    }
}

__global__ void subvector_backward_cuda(int64 size, float* pgx, float* pgy, int64 nrow, int64 ncol, int64 nfrom, int64 ncount) {
    int64 n = blockIdx.x * blockDim.x + threadIdx.x;

    if (n < size) {
        int64 nr = n / ncol;
        int64 nc = n % ncol;

        if (nc >= nfrom && nc < nfrom - ncount) {
            int64 ypos = nr * ncount + (nc - nfrom);
            pgx[n] = pgy[ypos];
        }
        else pgx[n] = 0;
    }
}

void VMath::subvector_backward(int device, float* pgx, float* pgy, int64 nrow, int64 ncol, int64 nfrom, int64 ncount) {
    int64 size = nrow * ncol;
    CUDA_CALL(subvector_backward, device, size, pgx, pgy, nrow, ncol, nfrom, ncount);
}

//--------------------------------------------------------------------------------------------------

__static__ void pickup_int_host(int* py, int* px1, int* px2, int64 nbat, int64 nrow, int64 nnom, int64 ncol) {
    for (int64 nb = 0, n = 0; nb < nbat; nb++) {
        for (int64 nr = 0; nr < nrow; nr++) {
            for (int64 nc = 0; nc < ncol; nc++, n++) {
                int nsel = px1[nb * nrow + nr];
                if (nsel == -1) continue;
#ifdef YOLO_DEBUG_TEMPORAL
                nsel = 0;
#else
                VP_THROW(VERR_UNDEFINED);
#endif
                int64 xpos = (nb * nnom + nsel) * ncol + nc;
                py[n] = px2[xpos];
            }
        }
    }
}

__global__ void pickup_int_cuda(int64 size, int* py, int* px1, int* px2, int64 nbat, int64 nrow, int64 nnom, int64 ncol) {
    int64 n = blockIdx.x * blockDim.x + threadIdx.x;

    if (n < size) {
        int64 nb = n / (nrow * ncol);
        int64 nr = (n / ncol) % nrow;
        int64 nc = n % ncol;

        int nsel = px1[nb * nrow + nr];
        if (nsel == -1) return;
        if (nsel < 0 || nsel >= nnom) {
#ifdef YOLO_DEBUG_TEMPORAL
            nsel = 0;
#else
            assert(0);
#endif
        }
        int64 xpos = (nb * nnom + nsel) * ncol + nc;
        py[n] = px2[xpos];
    }
}

void VMath::pickup_int(int device, int* py, int* px1, int* px2, int64 nbat, int64 nrow, int64 nnom, int64 ncol) {
    int64 size = nbat * nrow * ncol;
    CUDA_CALL(pickup_int, device, size, py, px1, px2, nbat, nrow, nnom, ncol);
}

//--------------------------------------------------------------------------------------------------

__static__ void pickup_float_host(float* py, int* px1, float* px2, int64 nbat, int64 nrow, int64 nnom, int64 ncol) {
    for (int64 nb = 0, n = 0; nb < nbat; nb++) {
        for (int64 nr = 0; nr < nrow; nr++) {
            for (int64 nc = 0; nc < ncol; nc++, n++) {
                int nsel = px1[nb * nrow + nr];
                if (nsel == -1) continue;
                if (nsel < 0 || nsel >= nnom) {
#ifdef YOLO_DEBUG_TEMPORAL
                    nsel = 0;
#else
                    VP_THROW(VERR_UNDEFINED);
#endif
                }
                int64 xpos = (nb * nnom + nsel) * ncol + nc;
                py[n] = px2[xpos];
            }
        }
    }
}

__global__ void pickup_float_cuda(int64 size, float* py, int* px1, float* px2, int64 nbat, int64 nrow, int64 nnom, int64 ncol) {
    int64 n = blockIdx.x * blockDim.x + threadIdx.x;

    if (n < size) {
        int64 nb = n / (nrow * ncol);
        int64 nr = (n / ncol) % nrow;
        int64 nc = n % ncol;

        int nsel = px1[nb * nrow + nr];
        if (nsel == -1) return;
        if (nsel < 0 || nsel >= nnom) {
#ifdef YOLO_DEBUG_TEMPORAL
            nsel = 0;
#else
            assert(0);
#endif
        }
        int64 xpos = (nb * nnom + nsel) * ncol + nc;
        py[n] = px2[xpos];
    }
}

void VMath::pickup_float(int device, float* py, int* px1, float* px2, int64 nbat, int64 nrow, int64 nnom, int64 ncol) {
    if (0) return;
    int64 size = nbat * nrow * ncol;
    CUDA_CALL(pickup_float, device, size, py, px1, px2, nbat, nrow, nnom, ncol);
}

//--------------------------------------------------------------------------------------------------

__static__ void pickup_float_backward_host(float* pgx, float* pgy, int* px1, int64 nbat, int64 nrow, int64 nnom, int64 ncol) {
    for (int64 nb = 0, n = 0; nb < nbat; nb++) {
        for (int64 nn = 0; nn < nnom; nn++) {
            for (int64 nc = 0; nc < ncol; nc++, n++) {
                float sum = 0;
                for (int64 nr = 0; nr < nrow; nr++) {
                    int nsel = px1[nb * nrow + nr];
                    if (nsel == nn) {
                        int64 ypos = (nb * nrow + nr) * ncol + nc;
                        sum += pgy[ypos];
                    }
                }
                pgx[n] = sum;
            }
        }
    }
}

__global__ void pickup_float_backward_cuda(int64 size, float* pgx, float* pgy, int* px1, int64 nbat, int64 nrow, int64 nnom, int64 ncol) {
    int64 n = blockIdx.x * blockDim.x + threadIdx.x;

    if (n < size) {
        int64 nb = n / (nnom * ncol);
        int64 nn = (n / ncol) % nnom;
        int64 nc = n % ncol;

        float sum = 0;
        for (int64 nr = 0; nr < nrow; nr++) {
            int nsel = px1[nb * nrow + nr];
            if (nsel == nn) {
                int64 ypos = (nb * nrow + nr) * ncol + nc;
                sum += pgy[ypos];
            }
        }
        pgx[n] = sum;
    }
}

void VMath::pickup_float_backward(int device, float* pgx, float* pgy, int* px1, int64 nbat, int64 nrow, int64 nnom, int64 ncol) {
    int64 size = nbat * nnom * ncol;
    CUDA_CALL(pickup_float_backward, device, size, pgx, pgy, px1, nbat, nrow, nnom, ncol);
}

//--------------------------------------------------------------------------------------------------

__static__ void pickup_static_int_host(int* py, int* px1, int* px2, int64 nbat, int64 nrow, int64 nnom, int64 ncol) {
    for (int64 nb = 0, n = 0; nb < nbat; nb++) {
        for (int64 nr = 0; nr < nrow; nr++) {
            for (int64 nc = 0; nc < ncol; nc++, n++) {
                int nsel = px1[nb * nrow + nr];
                if (nsel == -1) continue;
                if (nsel < 0 || nsel >= nnom) {
#ifdef YOLO_DEBUG_TEMPORAL
                    nsel = 0;
#else
                    VP_THROW(VERR_UNDEFINED);
#endif
                }
                int64 xpos = nsel * ncol + nc;
                py[n] = px2[xpos];
            }
        }
    }
}

__global__ void pickup_static_int_cuda(int64 size, int* py, int* px1, int* px2, int64 nbat, int64 nrow, int64 nnom, int64 ncol) {
    int64 n = blockIdx.x * blockDim.x + threadIdx.x;

    if (n < size) {
        int64 nb = n / (nrow * ncol);
        int64 nr = (n / ncol) % nrow;
        int64 nc = n % ncol;

        int nsel = px1[nb * nrow + nr];
        if (nsel == -1) return;
        if (nsel < 0 || nsel >= nnom) {
#ifdef YOLO_DEBUG_TEMPORAL
            nsel = 0;
#else
            assert(0);
#endif
        }
        int64 xpos = nsel * ncol + nc;
        py[n] = px2[xpos];
    }
}

void VMath::pickup_static_int(int device, int* py, int* px1, int* px2, int64 nbat, int64 nrow, int64 nnom, int64 ncol) {
    int64 size = nbat * nrow * ncol;
    CUDA_CALL(pickup_static_int, device, size, py, px1, px2, nbat, nrow, nnom, ncol);
}

//--------------------------------------------------------------------------------------------------

__static__ void pickup_static_float_host(float* py, int* px1, float* px2, int64 nbat, int64 nrow, int64 nnom, int64 ncol) {
    for (int64 nb = 0, n = 0; nb < nbat; nb++) {
        for (int64 nr = 0; nr < nrow; nr++) {
            for (int64 nc = 0; nc < ncol; nc++, n++) {
                int nsel = px1[nb * nrow + nr];
                if (nsel == -1) continue;
                if (nsel < 0 || nsel >= nnom) {
#ifdef YOLO_DEBUG_TEMPORAL
                    nsel = 0;
#else
                    VP_THROW(VERR_UNDEFINED);
#endif
                }
                int64 xpos = nsel * ncol + nc;
                py[n] = px2[xpos];
            }
        }
    }
}

__global__ void pickup_static_float_cuda(int64 size, float* py, int* px1, float* px2, int64 nbat, int64 nrow, int64 nnom, int64 ncol) {
    int64 n = blockIdx.x * blockDim.x + threadIdx.x;

    if (n < size) {
        int64 nb = n / (nrow * ncol);
        int64 nr = (n / ncol) % nrow;
        int64 nc = n % ncol;

        int nsel = px1[nb * nrow + nr];
        if (nsel == -1) return;
        if (nsel < 0 || nsel >= nnom) {
#ifdef YOLO_DEBUG_TEMPORAL
            nsel = 0;
#else
            assert(0);
#endif
        }
        int64 xpos = nsel * ncol + nc;
        py[n] = px2[xpos];
    }
}

void VMath::pickup_static_float(int device, float* py, int* px1, float* px2, int64 nbat, int64 nrow, int64 nnom, int64 ncol) {
    int64 size = nbat * nrow * ncol;
    CUDA_CALL(pickup_static_float, device, size, py, px1, px2, nbat, nrow, nnom, ncol);
}

//--------------------------------------------------------------------------------------------------

__static__ void iou_cross_xywh_host(float* py, float* px1, float* px2, int64 nbat, int64 nrow, int64 ncol) {
    for (int64 nb = 0, n = 0; nb < nbat; nb++) {
        for (int64 nr = 0; nr < nrow; nr++) {
            for (int64 nc = 0; nc < ncol; nc++, n++) {
                float* p1 = px1 + (nb * nrow + nr) * 4;
                float* p2 = px2 + (nb * ncol + nc) * 4;

                float x1 = p1[0], y1 = p1[1], w1 = p1[2], h1 = p1[3];
                float x2 = p2[0], y2 = p2[1], w2 = p2[2], h2 = p2[3];

                float left1 = x1 - w1 / 2, right1 = x1 + w1 / 2, top1 = y1 - h1 / 2, bottom1 = y1 + h1 / 2;
                float left2 = x2 - w2 / 2, right2 = x2 + w2 / 2, top2 = y2 - h2 / 2, bottom2 = y2 + h2 / 2;

                float left = MAX(left1, left2), right = MIN(right1, right2), top = MAX(top1, top2), bottom = MIN(bottom1, bottom2);

                float iou = 0;

                if (left < right && top < bottom) {
                    float area1 = (right1 - left1) * (bottom1 - top1);
                    float area2 = (right2 - left2) * (bottom2 - top2);
                    float inter_area = (right - left) * (bottom - top);
                    float union_area = area1 + area2 - inter_area;

                    iou = inter_area / union_area;
                }

                py[n] = iou;
            }
        }
    }
}

__global__ void iou_cross_xywh_cuda(int64 size, float* py, float* px1, float* px2, int64 nbat, int64 nrow, int64 ncol) {
    int64 n = blockIdx.x * blockDim.x + threadIdx.x;

    if (n < size) {
        int64 nb = n / (nrow * ncol);
        int64 nr = n / ncol % nrow;
        int64 nc = n % ncol;

        float* p1 = px1 + (nb * nrow + nr) * 4;
        float* p2 = px2 + (nb * ncol + nc) * 4;

        float x1 = p1[0], y1 = p1[1], w1 = p1[2], h1 = p1[3];
        float x2 = p2[0], y2 = p2[1], w2 = p2[2], h2 = p2[3];

        float left1 = x1 - w1 / 2, right1 = x1 + w1 / 2, top1 = y1 - h1 / 2, bottom1 = y1 + h1 / 2;
        float left2 = x2 - w2 / 2, right2 = x2 + w2 / 2, top2 = y2 - h2 / 2, bottom2 = y2 + h2 / 2;

        float left = MAX(left1, left2), right = MIN(right1, right2), top = MAX(top1, top2), bottom = MIN(bottom1, bottom2);

        float iou = 0;

        if (left < right && top < bottom) {
            float area1 = (right1 - left1) * (bottom1 - top1);
            float area2 = (right2 - left2) * (bottom2 - top2);
            float inter_area = (right - left) * (bottom - top);
            float union_area = area1 + area2 - inter_area;

            iou = inter_area / union_area;
        }

        py[n] = iou;
    }
}

void VMath::iou_cross_xywh(int device, float* py, float* px1, float* px2, int64 nbat, int64 nrow, int64 ncol) {
    int64 size = nbat * nrow * ncol;
    CUDA_CALL(iou_cross_xywh, device, size, py, px1, px2, nbat, nrow, ncol);
}

//--------------------------------------------------------------------------------------------------

__static__ void iou_cross_lrtb_host(float* py, float* px1, float* px2, int64 nbat, int64 nrow, int64 ncol) {
    for (int64 nb = 0, n = 0; nb < nbat; nb++) {
        for (int64 nr = 0; nr < nrow; nr++) {
            for (int64 nc = 0; nc < ncol; nc++, n++) {
                float* p1 = px1 + (nb * nrow + nr) * 4;
                float* p2 = px2 + (nb * ncol + nc) * 4;

                float left1 = p1[0], right1 = p1[1], top1 = p1[2], bottom1 = p1[3];
                float left2 = p2[0], right2 = p2[1], top2 = p2[2], bottom2 = p2[3];

                float left = MAX(left1, left2), right = MIN(right1, right2), top = MAX(top1, top2), bottom = MIN(bottom1, bottom2);

                float iou = 0;

                if (left < right && top < bottom) {
                    float area1 = (right1 - left1) * (bottom1 - top1);
                    float area2 = (right2 - left2) * (bottom2 - top2);
                    float inter_area = (right - left) * (bottom - top);
                    float union_area = area1 + area2 - inter_area;

                    iou = inter_area / union_area;
                }

                py[n] = iou;
            }
        }
    }
}

__global__ void iou_cross_lrtb_cuda(int64 size, float* py, float* px1, float* px2, int64 nbat, int64 nrow, int64 ncol) {
    int64 n = blockIdx.x * blockDim.x + threadIdx.x;

    if (n < size) {
        int64 nb = n / (nrow * ncol);
        int64 nr = n / ncol % nrow;
        int64 nc = n % ncol;

        float* p1 = px1 + (nb * nrow + nr) * 4;
        float* p2 = px2 + (nb * ncol + nc) * 4;

        float left1 = p1[0], right1 = p1[1], top1 = p1[2], bottom1 = p1[3];
        float left2 = p2[0], right2 = p2[1], top2 = p2[2], bottom2 = p2[3];

        float left = MAX(left1, left2), right = MIN(right1, right2), top = MAX(top1, top2), bottom = MIN(bottom1, bottom2);

        float iou = 0;

        if (left < right && top < bottom) {
            float area1 = (right1 - left1) * (bottom1 - top1);
            float area2 = (right2 - left2) * (bottom2 - top2);
            float inter_area = (right - left) * (bottom - top);
            float union_area = area1 + area2 - inter_area;

            iou = inter_area / union_area;
        }

        py[n] = iou;
    }
}

void VMath::iou_cross_lrtb(int device, float* py, float* px1, float* px2, int64 nbat, int64 nrow, int64 ncol) {
    int64 size = nbat * nrow * ncol;
    CUDA_CALL(iou_cross_lrtb, device, size, py, px1, px2, nbat, nrow, ncol);
}

//--------------------------------------------------------------------------------------------------

__static__ void iou_loss_host(float* py, float* px1, float* px2, int64 nrow, int op_code) {
    for (int64 n = 0; n < nrow; n++) {
        float* pbox = px1 + n * 4;
        float* tbox = px2 + n * 4;

        float pbx = pbox[0], pby = pbox[1], pbw = pbox[2], pbh = pbox[3];
        float tbx = tbox[0], tby = tbox[1], tbw = tbox[2], tbh = tbox[3];

        float pleft = pbx - pbw / 2, pright = pbx + pbw / 2, ptop = pby - pbh / 2, pbottom = pby + pbh / 2;
        float tleft = tbx - tbw / 2, tright = tbx + tbw / 2, ttop = tby - tbh / 2, tbottom = tby + tbh / 2;

        float left = MAX(pleft, tleft), right = MIN(pright, tright), top = MAX(ptop, ttop), bottom = MIN(pbottom, tbottom);

        float iou = 0;
        float union_area = 0;
        float pred_area = 0;
        float true_area = 0;
        float inter_area = 0;

        if (left < right && top < bottom) {
            pred_area = pbw * pbh;
            true_area = tbw * tbh;
            inter_area = (right - left) * (bottom - top);
            union_area = pred_area + true_area - inter_area;

            iou = inter_area / union_area;

        }

        float iou_loss = 1 - iou;
        py[n] = iou_loss;

        if ((VGraphOpCode)op_code == VGraphOpCode::iou_loss) return;

        float cleft = MIN(pleft, tleft), cright = MAX(pright, tright), ctop = MIN(ptop, ttop), cbottom = MAX(pbottom, tbottom);

        float gterm = 0, dterm = 0, cterm = 0;

        float cover_area = (cright - cleft) * (cbottom - ctop);

        if (cover_area > 0) {
            if ((VGraphOpCode)op_code == VGraphOpCode::giou_loss) {
                gterm = (cover_area - union_area) / cover_area;
                py[n] += gterm;
                return;
            }

            float cover_dist_square = (cright - cleft) * (cright - cleft) + (cbottom - ctop) * (cbottom - ctop);
            float center_dist_square = (pbx - tbx) * (pbx - tbx) + (pby - tby) * (pby - tby);

            dterm = center_dist_square / cover_dist_square;

            if ((VGraphOpCode)op_code == VGraphOpCode::diou_loss) {
                py[n] += dterm;
                return;
            }

            if (iou >= 0.5) {
                float pw_on_ph = pbw / pbh;
                float tw_on_th = tbw / tbh;
                float arc_diff = ::atanf(pw_on_ph) - ::atanf(tw_on_th);
                float V = 4.0f * arc_diff * arc_diff / (PI_F * PI_F);
                float alpha = V / (1 - iou + V);

                cterm = alpha * V;
            }

            float ciou_loss = 1 - iou + dterm + cterm;
            py[n] = ciou_loss;
        }
    }
}

__global__ void iou_loss_cuda(int64 size, float* py, float* px1, float* px2, int64 nrow, int op_code) {
    int64 n = blockIdx.x * blockDim.x + threadIdx.x;

    if (n < size) {
        float* pbox = px1 + n * 4;
        float* tbox = px2 + n * 4;

        float pbx = pbox[0], pby = pbox[1], pbw = pbox[2], pbh = pbox[3];
        float tbx = tbox[0], tby = tbox[1], tbw = tbox[2], tbh = tbox[3];

        float pleft = pbx - pbw / 2, pright = pbx + pbw / 2, ptop = pby - pbh / 2, pbottom = pby + pbh / 2;
        float tleft = tbx - tbw / 2, tright = tbx + tbw / 2, ttop = tby - tbh / 2, tbottom = tby + tbh / 2;

        float left = MAX(pleft, tleft), right = MIN(pright, tright), top = MAX(ptop, ttop), bottom = MIN(pbottom, tbottom);

        float iou = 0;
        float union_area = 0;
        float pred_area = 0;
        float true_area = 0;
        float inter_area = 0;

        if (left < right && top < bottom) {
            pred_area = pbw * pbh;
            true_area = tbw * tbh;
            inter_area = (right - left) * (bottom - top);
            union_area = pred_area + true_area - inter_area;

            iou = inter_area / union_area;

        }

        float iou_loss = 1 - iou;
        py[n] = iou_loss;

        if ((VGraphOpCode)op_code == VGraphOpCode::iou_loss) return;

        float cleft = MIN(pleft, tleft), cright = MAX(pright, tright), ctop = MIN(ptop, ttop), cbottom = MAX(pbottom, tbottom);

        float gterm = 0, dterm = 0, cterm = 0;

        float cover_area = (cright - cleft) * (cbottom - ctop);

        if (cover_area > 0) {
            if ((VGraphOpCode)op_code == VGraphOpCode::giou_loss) {
                gterm = (cover_area - union_area) / cover_area;
                py[n] += gterm;
                return;
            }

            float cover_dist_square = (cright - cleft) * (cright - cleft) + (cbottom - ctop) * (cbottom - ctop);
            float center_dist_square = (pbx - tbx) * (pbx - tbx) + (pby - tby) * (pby - tby);

            dterm = center_dist_square / cover_dist_square;

            if ((VGraphOpCode)op_code == VGraphOpCode::diou_loss) {
                py[n] += dterm;
                return;
            }

            if (iou >= 0.5) {
                float pw_on_ph = pbw / pbh;
                float tw_on_th = tbw / tbh;
                float arc_diff = ::atanf(pw_on_ph) - ::atanf(tw_on_th);
                float V = 4.0f * arc_diff * arc_diff / (PI_F * PI_F);
                float alpha = V / (1 - iou + V);

                cterm = alpha * V;
            }

            float ciou_loss = 1 - iou + dterm + cterm;
            py[n] = ciou_loss;
        }
    }
}

void VMath::iou_loss(int device, float* py, float* px1, float* px2, int64 nrow, int op_code) {
    int64 size = nrow;
    CUDA_CALL(iou_loss, device, size, py, px1, px2, nrow, op_code);
}

//--------------------------------------------------------------------------------------------------

__static__ void iou_loss_backward_host(float* pgx, float* pgy, float* px1, float* px2, int64 nrow, int op_code) {
    for (int64 nr = 0, n = 0; nr < nrow; nr++) {
        for (int64 nc = 0; nc < 4; nc++, n++) {
            float gx = 0;
            float g_loss = pgy[nr];
            if (g_loss == 0) return;

            float* pbox = px1 + nr * 4;
            float* tbox = px2 + nr * 4;

            float px = pbox[0], py = pbox[1], pw = pbox[2], ph = pbox[3];
            float tx = tbox[0], ty = tbox[1], tw = tbox[2], th = tbox[3];

            float pleft = px - pw / 2, pright = px + pw / 2, ptop = py - ph / 2, pbottom = py + ph / 2;
            float tleft = tx - tw / 2, tright = tx + tw / 2, ttop = ty - th / 2, tbottom = ty + th / 2;

            float left = MAX(pleft, tleft), right = MIN(pright, tright), top = MAX(ptop, ttop), bottom = MIN(pbottom, tbottom);

            float iou = 0;
            float union_area = 0;
            float pred_area = 0;
            float true_area = 0;
            float inter_area = 0;

            float g_L_on_iou = -1;

            if (left < right && top < bottom) {
                pred_area = pw * ph;
                true_area = tw * th;
                inter_area = (right - left) * (bottom - top);
                union_area = pred_area + true_area - inter_area;

                iou = inter_area / union_area;
            }

            if ((VGraphOpCode)op_code != VGraphOpCode::iou_loss) {
                float cleft = MIN(pleft, tleft), cright = MAX(pright, tright), ctop = MIN(ptop, ttop), cbottom = MAX(pbottom, tbottom);

                float cover_area = (cright - cleft) * (cbottom - ctop);

                float giou = iou - (cover_area - union_area) / cover_area;

                if (cover_area > 0) {
                    if ((VGraphOpCode)op_code == VGraphOpCode::giou_loss) {
                        float g_L_on_cover_area = -union_area / (cover_area * cover_area);
                        float g_L_on_union_area = 1.0f / cover_area;

                        VP_THROW(VERR_NOT_IMPLEMENTED_YET);
                    }
                    else {
                        float cover_dist_square = (cright - cleft) * (cright - cleft) + (cbottom - ctop) * (cbottom - ctop);
                        float center_dist_square = (px - tx) * (px - tx) + (py - ty) * (py - ty);

                        float dterm = center_dist_square / cover_dist_square;
                        float diou_loss = 1 - iou + dterm;

                        float g_L_on_dterm = 1.0f;

                        float g_dterm_on_center_dist_square = 1.0f / cover_dist_square;
                        float g_dterm_on_cover_dist_square = -dterm / cover_dist_square;

                        float g_L_on_center_dist_square = g_L_on_dterm * g_dterm_on_center_dist_square;
                        float g_L_on_cover_dist_square = g_L_on_dterm * g_dterm_on_cover_dist_square;

                        float g_center_on_px = 2.0f * (px - tx);
                        float g_center_on_py = 2.0f * (py - ty);

                        float g_L_on_px = g_L_on_center_dist_square * g_center_on_px;
                        float g_L_on_py = g_L_on_center_dist_square * g_center_on_py;

                        float g_cover_on_right_minus_left = 2.0f * (cright - cleft);
                        float g_cover_on_bottom_minus_top = 2.0f * (cbottom - ctop);

                        float g_L_on_right_minus_left = g_L_on_cover_dist_square * g_cover_on_right_minus_left;
                        float g_L_on_bottom_minus_top = g_L_on_cover_dist_square * g_cover_on_bottom_minus_top;

                        switch (nc) {
                        case 0:
                            gx += g_L_on_px;
                            if (pleft < tleft) gx -= g_L_on_right_minus_left;
                            if (pright > tright) gx += g_L_on_right_minus_left;
                            break;
                        case 1:
                            gx += g_L_on_py;
                            if (ptop < ttop) gx -= g_L_on_bottom_minus_top;
                            if (pbottom > tbottom) gx += g_L_on_bottom_minus_top;
                            break;
                        case 2:
                            if (pleft < tleft) gx += g_L_on_right_minus_left / 2.0f;
                            if (pright > tright) gx += g_L_on_right_minus_left / 2.0f;
                            break;
                        case 3:
                            if (ptop < ttop) gx += g_L_on_bottom_minus_top / 2.0f;
                            if (pbottom > tbottom) gx += g_L_on_bottom_minus_top / 2.0f;
                            break;
                        default:
                            if (n == 0) VP_THROW(VERR_CONDITIONAL_STATEMENT);
                            break;
                        }

                        if ((VGraphOpCode)op_code == VGraphOpCode::ciou_loss && iou >= 0.5) {
                            float pw_on_ph = pw / ph;
                            float tw_on_th = tw / th;
                            float arc_diff = ::atanf(pw_on_ph) - ::atanf(tw_on_th);
                            float V = 4.0f * arc_diff * arc_diff / (PI_F * PI_F);
                            float alpha = V / (1 - iou + V);

                            float cterm = alpha * V;

                            float ciou_loss = 1 - iou + dterm + cterm;

                            float g_L_on_cterm = 1.0f;

                            float g_cterm_on_V = 2 * alpha - alpha * alpha;
                            float g_L_on_V = g_L_on_cterm * g_cterm_on_V;

                            float g_V_on_arc_diff = 8.0f * arc_diff / (PI_F * PI_F);
                            float g_L_on_arc_diff = g_L_on_V * g_V_on_arc_diff;

                            float g_arc_diff_on_pw_ph = 1.0f / (1.0f + pw_on_ph * pw_on_ph);
                            float g_L_on_pw_ph = g_L_on_arc_diff * g_arc_diff_on_pw_ph;

                            float g_pw_ph_on_pw = 1.0f / ph;
                            float g_pw_ph_on_ph = -pw_on_ph / ph;

                            float g_L_on_pw = g_L_on_pw_ph * g_pw_ph_on_pw;
                            float g_L_on_ph = g_L_on_pw_ph * g_pw_ph_on_ph;

                            if (nc == 2) gx += g_L_on_pw;
                            if (nc == 3) gx -= g_L_on_ph;

                            float g_cterm_on_iou = alpha * alpha;
                            g_L_on_iou += g_L_on_cterm * g_cterm_on_iou;
                        }
                    }
                }
            }

            float g_iou_on_pred_area = (union_area > 0) ? -inter_area / (union_area * union_area) : 0;
            float g_iou_on_inter_area = (union_area > 0) ? (inter_area + union_area) / (union_area * union_area) : 0;

            float g_L_on_pred_area = g_L_on_iou * g_iou_on_pred_area;
            float g_L_on_inter_area = g_L_on_iou * g_iou_on_inter_area;

            float g_right_minus_left = g_L_on_inter_area * (bottom - top);
            float g_bottom_minus_top = g_L_on_inter_area * (right - left);

            switch (nc) {
            case 0:
                if (pleft > tleft) gx -= g_right_minus_left;
                if (pright < tright) gx += g_right_minus_left;
                break;
            case 1:
                if (ptop > ttop) gx -= g_bottom_minus_top;
                if (pbottom < tbottom) gx += g_bottom_minus_top;
                break;
            case 2:
                gx += g_L_on_pred_area * ph;
                if (pleft > tleft) gx += g_right_minus_left / 2.0f;
                if (pright < tright) gx += g_right_minus_left / 2.0f;
                gx *= 2; // darknet과 기울기 비교를 위한 임시처리
                break;
            case 3:
                gx += g_L_on_pred_area * pw;
                if (ptop > ttop) gx += g_bottom_minus_top / 2.0f;
                if (pbottom < tbottom) gx += g_bottom_minus_top / 2.0f;
                gx *= 2; // darknet과 기울기 비교를 위한 임시처리
                break;
            default:
                if (n == 0) VP_THROW(VERR_CONDITIONAL_STATEMENT);
                break;
            }

            pgx[n] += gx * g_loss;
        }
    }
}

__global__ void iou_loss_backward_cuda(int64 size, float* pgx, float* pgy, float* px1, float* px2, int64 nrow, int op_code) {
    int64 n = blockIdx.x * blockDim.x + threadIdx.x;

    if (n < size) {
        int64 nr = n / 4;
        int64 nc = n % 4;

        float gx = 0;
        float g_loss = pgy[nr];
        if (g_loss == 0) return;

        float* pbox = px1 + nr * 4;
        float* tbox = px2 + nr * 4;

        float px = pbox[0], py = pbox[1], pw = pbox[2], ph = pbox[3];
        float tx = tbox[0], ty = tbox[1], tw = tbox[2], th = tbox[3];

        float pleft = px - pw / 2, pright = px + pw / 2, ptop = py - ph / 2, pbottom = py + ph / 2;
        float tleft = tx - tw / 2, tright = tx + tw / 2, ttop = ty - th / 2, tbottom = ty + th / 2;

        float left = MAX(pleft, tleft), right = MIN(pright, tright), top = MAX(ptop, ttop), bottom = MIN(pbottom, tbottom);

        float iou = 0;
        float union_area = 0;
        float pred_area = 0;
        float true_area = 0;
        float inter_area = 0;

        float g_L_on_iou = -1;

        if (left < right && top < bottom) {
            pred_area = pw * ph;
            true_area = tw * th;
            inter_area = (right - left) * (bottom - top);
            union_area = pred_area + true_area - inter_area;

            iou = inter_area / union_area;
        }

        if ((VGraphOpCode)op_code != VGraphOpCode::iou_loss) {
            float cleft = MIN(pleft, tleft), cright = MAX(pright, tright), ctop = MIN(ptop, ttop), cbottom = MAX(pbottom, tbottom);

            float cover_area = (cright - cleft) * (cbottom - ctop);

            float giou = iou - (cover_area - union_area) / cover_area;

            if (cover_area > 0) {
                if ((VGraphOpCode)op_code == VGraphOpCode::giou_loss) {
                    float g_L_on_cover_area = -union_area / (cover_area * cover_area);
                    float g_L_on_union_area = 1.0f / cover_area;

                    assert(0);
                }
                else {
                    float cover_dist_square = (cright - cleft) * (cright - cleft) + (cbottom - ctop) * (cbottom - ctop);
                    float center_dist_square = (px - tx) * (px - tx) + (py - ty) * (py - ty);

                    float dterm = center_dist_square / cover_dist_square;
                    float diou_loss = 1 - iou + dterm;

                    float g_L_on_dterm = 1.0f;

                    float g_dterm_on_center_dist_square = 1.0f / cover_dist_square;
                    float g_dterm_on_cover_dist_square = -dterm / cover_dist_square;

                    float g_L_on_center_dist_square = g_L_on_dterm * g_dterm_on_center_dist_square;
                    float g_L_on_cover_dist_square = g_L_on_dterm * g_dterm_on_cover_dist_square;

                    float g_center_on_px = 2.0f * (px - tx);
                    float g_center_on_py = 2.0f * (py - ty);

                    float g_L_on_px = g_L_on_center_dist_square * g_center_on_px;
                    float g_L_on_py = g_L_on_center_dist_square * g_center_on_py;

                    float g_cover_on_right_minus_left = 2.0f * (cright - cleft);
                    float g_cover_on_bottom_minus_top = 2.0f * (cbottom - ctop);

                    float g_L_on_right_minus_left = g_L_on_cover_dist_square * g_cover_on_right_minus_left;
                    float g_L_on_bottom_minus_top = g_L_on_cover_dist_square * g_cover_on_bottom_minus_top;

                    switch (nc) {
                    case 0:
                        gx += g_L_on_px;
                        if (pleft < tleft) gx -= g_L_on_right_minus_left;
                        if (pright > tright) gx += g_L_on_right_minus_left;
                        break;
                    case 1:
                        gx += g_L_on_py;
                        if (ptop < ttop) gx -= g_L_on_bottom_minus_top;
                        if (pbottom > tbottom) gx += g_L_on_bottom_minus_top;
                        break;
                    case 2:
                        if (pleft < tleft) gx += g_L_on_right_minus_left / 2.0f;
                        if (pright > tright) gx += g_L_on_right_minus_left / 2.0f;
                        break;
                    case 3:
                        if (ptop < ttop) gx += g_L_on_bottom_minus_top / 2.0f;
                        if (pbottom > tbottom) gx += g_L_on_bottom_minus_top / 2.0f;
                        break;
                    default:
                        if (n == 0) assert(0);
                        break;
                    }

                    if ((VGraphOpCode)op_code == VGraphOpCode::ciou_loss && iou >= 0.5) {
                        float pw_on_ph = pw / ph;
                        float tw_on_th = tw / th;
                        float arc_diff = ::atanf(pw_on_ph) - ::atanf(tw_on_th);
                        float V = 4.0f * arc_diff * arc_diff / (PI_F * PI_F);
                        float alpha = V / (1 - iou + V);

                        float cterm = alpha * V;

                        float ciou_loss = 1 - iou + dterm + cterm;

                        float g_L_on_cterm = 1.0f;

                        float g_cterm_on_V = 2 * alpha - alpha * alpha;
                        float g_L_on_V = g_L_on_cterm * g_cterm_on_V;

                        float g_V_on_arc_diff = 8.0f * arc_diff / (PI_F * PI_F);
                        float g_L_on_arc_diff = g_L_on_V * g_V_on_arc_diff;

                        float g_arc_diff_on_pw_ph = 1.0f / (1.0f + pw_on_ph * pw_on_ph);
                        float g_L_on_pw_ph = g_L_on_arc_diff * g_arc_diff_on_pw_ph;

                        float g_pw_ph_on_pw = 1.0f / ph;
                        float g_pw_ph_on_ph = -pw_on_ph / ph;

                        float g_L_on_pw = g_L_on_pw_ph * g_pw_ph_on_pw;
                        float g_L_on_ph = g_L_on_pw_ph * g_pw_ph_on_ph;

                        if (nc == 2) gx += g_L_on_pw;
                        if (nc == 3) gx -= g_L_on_ph;

                        float g_cterm_on_iou = alpha * alpha;
                        g_L_on_iou += g_L_on_cterm * g_cterm_on_iou;
                    }
                }
            }
        }

        float g_iou_on_pred_area = (union_area > 0) ? -inter_area / (union_area * union_area) : 0;
        float g_iou_on_inter_area = (union_area > 0) ? (inter_area + union_area) / (union_area * union_area) : 0;

        float g_L_on_pred_area = g_L_on_iou * g_iou_on_pred_area;
        float g_L_on_inter_area = g_L_on_iou * g_iou_on_inter_area;

        float g_right_minus_left = g_L_on_inter_area * (bottom - top);
        float g_bottom_minus_top = g_L_on_inter_area * (right - left);

        switch (nc) {
        case 0:
            if (pleft > tleft) gx -= g_right_minus_left;
            if (pright < tright) gx += g_right_minus_left;
            break;
        case 1:
            if (ptop > ttop) gx -= g_bottom_minus_top;
            if (pbottom < tbottom) gx += g_bottom_minus_top;
            break;
        case 2:
            gx += g_L_on_pred_area * ph;
            if (pleft > tleft) gx += g_right_minus_left / 2.0f;
            if (pright < tright) gx += g_right_minus_left / 2.0f;
            gx *= 2; // darknet과 기울기 비교를 위한 임시처리
            break;
        case 3:
            gx += g_L_on_pred_area * pw;
            if (ptop > ttop) gx += g_bottom_minus_top / 2.0f;
            if (pbottom < tbottom) gx += g_bottom_minus_top / 2.0f;
            gx *= 2; // darknet과 기울기 비교를 위한 임시처리
            break;
        default:
            if (n == 0) assert(0);
            break;
        }

        pgx[n] += gx * g_loss;
    }
}

void VMath::iou_loss_backward(int device, float* pgx, float* pgy, float* px1, float* px2, int64 nrow, int op_code) {
    int64 size = nrow * 4;
    CUDA_CALL(iou_loss_backward, device, size, pgx, pgy, px2, px1, nrow, op_code);
}

//--------------------------------------------------------------------------------------------------

__static__ void to_boxes_host(float* py, float* px1, float* px2, int64 nrow) {
    for (int64 nr = 0, n = 0; nr < nrow; nr++) {
        py[n++] = px1[nr * 2 + 0] - px2[nr * 2 + 0] / 2.0f;
        py[n++] = px1[nr * 2 + 0] + px2[nr * 2 + 0] / 2.0f;
        py[n++] = px1[nr * 2 + 1] - px2[nr * 2 + 1] / 2.0f;
        py[n++] = px1[nr * 2 + 1] + px2[nr * 2 + 1] / 2.0f;
    }
}

__global__ void to_boxes_cuda(int64 size, float* py, float* px1, float* px2, int64 nrow) {
    int64 n = blockIdx.x * blockDim.x + threadIdx.x;

    if (n < size) {
        int64 nr = n / 4;
        int64 nc = n % 4;

        switch (nc) {
        case 0: py[n] = px1[nr * 2 + 0] - px2[nr * 2 + 0] / 2.0f; break;
        case 1: py[n] = px1[nr * 2 + 0] + px2[nr * 2 + 0] / 2.0f; break;
        case 2: py[n] = px1[nr * 2 + 1] - px2[nr * 2 + 1] / 2.0f; break;
        case 3: py[n] = px1[nr * 2 + 1] + px2[nr * 2 + 1] / 2.0f; break;
        }
    }
}

void VMath::to_boxes(int device, float* py, float* px1, float* px2, int64 nrow) {
    int64 size = nrow * 4;
    CUDA_CALL(to_boxes, device, size, py, px1, px2, nrow);
}

//--------------------------------------------------------------------------------------------------

__static__ void upsample_host(float* py, float* px, int64 ndat, int64 nchn, int64 nyht, int64 nywd, int64 hratio, int64 wratio) {
    int64 nxht = nyht / hratio;
    int64 nxwd = nywd / wratio;

    for (int64 nd = 0, n = 0; nd < ndat; nd++) {
        for (int64 nn = 0; nn < nchn; nn++) {
            for (int64 nh = 0; nh < nyht; nh++) {
                for (int64 nw = 0; nw < nywd; nw++, n++) {
                    int64 xpos = ((nd * nchn + nn) * nxht + nh / hratio) * nxwd + nw / wratio;
                    py[n] = px[xpos];
                }
            }
        }
    }
}

__global__ void upsample_cuda(int64 size, float* py, float* px, int64 ndat, int64 nchn, int64 nyht, int64 nywd, int64 hratio, int64 wratio) {
    int64 n = blockIdx.x * blockDim.x + threadIdx.x;

    if (n < size) {
        int64 nd = n / (nchn * nyht * nywd);
        int64 nn = n / (nyht * nywd ) % nchn;
        int64 nh = n / nywd % nyht;
        int64 nw = n % nywd;

        int64 nxht = nyht / hratio;
        int64 nxwd = nywd / wratio;

        int64 xpos = ((nd * nchn + nn) * nxht + nh / hratio) * nxwd + nw / wratio;
        py[n] = px[xpos];
    }
}

void VMath::upsample(int device, float* py, float* px, int64 ndat, int64 nchn, int64 nyht, int64 nywd, int64 hratio, int64 wratio) {
    int64 size = ndat * nchn * nyht * nywd;
    CUDA_CALL(upsample, device, size, py, px, ndat, nchn, nyht, nywd, hratio, wratio);
}

//--------------------------------------------------------------------------------------------------

__static__ void upsample_backward_host(float* pgx, float* pgy, int64 nbat, int64 nchn, int64 nxht, int64 nxwd, int64 hratio, int64 wratio) {
    int64 nyht = nxht * hratio;
    int64 nywd = nxwd * wratio;

    for (int64 nd = 0, n = 0; nd < nbat; nd++) {
        for (int64 nn = 0; nn < nchn; nn++) {
            for (int64 nh = 0; nh < nxht; nh++) {
                for (int64 nw = 0; nw < nxwd; nw++, n++) {
                    float sum = 0;
                    for (int64 nr = 0; nr < hratio; nr++) {
                        for (int64 nc = 0; nc < wratio; nc++) {
                            int64 ypos = ((nd * nchn + nn) * nyht + nh * hratio + nr) * nywd + nw * wratio + nc;
                            sum += pgy[ypos];
                        }
                    }
                    pgx[n] = sum;
                }
            }
        }
    }
}

__global__ void upsample_backward_cuda(int64 size, float* pgx, float* pgy, int64 nbat, int64 nchn, int64 nxht, int64 nxwd, int64 hratio, int64 wratio) {
    int64 n = blockIdx.x * blockDim.x + threadIdx.x;

    if (n < size) {
        int64 nd = n / (nchn * nxht * nxwd);
        int64 nn = n / (nxht * nxwd) % nchn;
        int64 nh = n / nxwd % nxht;
        int64 nw = n % nxwd;

        int64 nyht = nxht * hratio;
        int64 nywd = nxwd * wratio;

        float sum = 0;
        for (int64 nr = 0; nr < hratio; nr++) {
            for (int64 nc = 0; nc < wratio; nc++) {
                int64 ypos = ((nd * nchn + nn) * nyht + nh * hratio + nr) * nywd + nw * wratio + nc;
                sum += pgy[ypos];
            }
        }
        pgx[n] = sum;
    }
}

void VMath::upsample_backward(int device, float* pgx, float* pgy, int64 nbat, int64 nchn, int64 nxht, int64 nxwd, int64 hratio, int64 wratio) {
    int64 size = nbat * nchn * nxht * nxwd;
    CUDA_CALL(upsample_backward, device, size, pgx, pgy, nbat, nchn, nxht, nxwd, hratio, wratio);
}

//--------------------------------------------------------------------------------------------------

__static__ void sigmoid_host(float* py, float* px, int64 nrow) {
    for (int64 n = 0; n < nrow; n++) {
        float x = px[n];
        py[n] = _sigmoid_host(x);
    }
}

__global__ void sigmoid_cuda(int64 size, float* py, float* px, int64 nrow) {
    int64 n = blockIdx.x * blockDim.x + threadIdx.x;

    if (n < size) {
        float x = px[n];
        py[n] = _sigmoid_cuda(x);
    }
}

void VMath::sigmoid(int device, float* py, float* px, int64 nrow) {
    int64 size = nrow;
    CUDA_CALL(sigmoid, device, size, py, px, nrow);
}

//--------------------------------------------------------------------------------------------------

__static__ void sigmoid_backward_host(float* pgx, float* pgy, float* px, int64 ndat) {
    for (int64 n = 0; n < ndat; n++) {
        float x = px[n];
        float y = _sigmoid_host(x);
        pgx[n] = pgy[n] * y * (1.0f - y);
    }
}

__global__ void sigmoid_backward_cuda(int64 size, float* pgx, float* pgy, float* px, int64 ndat) {
    int64 n = blockIdx.x * blockDim.x + threadIdx.x;

    if (n < size) {
        float x = px[n];
        float y = _sigmoid_cuda(x);
        pgx[n] = pgy[n] * y * (1.0f - y);
    }
}

void VMath::sigmoid_backward(int device, float* pgx, float* pgy, float* px, int64 ndat) {
    int64 size = ndat;
    CUDA_CALL(sigmoid_backward, device, size, pgx, pgy, px, ndat);
}

//--------------------------------------------------------------------------------------------------

__static__ void sigmoid_crossentropy_host(float* py, float* px, float* pz, int64 nrow) {
    for (int64 n = 0; n < nrow; n++) {
        float x = px[n];
        float z = pz[n];

        py[n] = MAX(x, 0) - x * z + ::logf(1.0f + ::expf(-::fabs(x)));
    }
}

__global__ void sigmoid_crossentropy_cuda(int64 size, float* py, float* px, float* pz, int64 nrow) {
    int64 n = blockIdx.x * blockDim.x + threadIdx.x;

    if (n < size) {
        float x = px[n];
        float z = pz[n];

        py[n] = MAX(x, 0) - x * z + ::logf(1.0f + ::expf(-::fabs(x)));
    }
}

void VMath::sigmoid_crossentropy(int device, float* py, float* px, float* pz, int64 nrow) {
    int64 size = nrow;
    CUDA_CALL(sigmoid_crossentropy, device, size, py, px, pz, nrow);
}

//--------------------------------------------------------------------------------------------------

__static__ void sigmoid_crossentropy_backward_x_host(float* pgx, float* pgy, float* px, float* pz, int64 nrow) {
    for (int64 n = 0; n < nrow; n++) {
        float x = px[n];
        float z = pz[n];

        float sigmoid_x = _sigmoid_host(x);

        pgx[n] = pgy[n] * (sigmoid_x - z);
    }
}

__global__ void sigmoid_crossentropy_backward_x_cuda(int64 size, float* pgx, float* pgy, float* px, float* pz, int64 nrow) {
    int64 n = blockIdx.x * blockDim.x + threadIdx.x;

    if (n < size) {
        float x = px[n];
        float z = pz[n];

        float sigmoid_x = _sigmoid_cuda(x);

        pgx[n] = pgy[n] * (sigmoid_x - z);
    }
}

void VMath::sigmoid_crossentropy_backward_x(int device, float* pgx, float* pgy, float* px, float* pz, int64 nrow) {
    int64 size = nrow;
    CUDA_CALL(sigmoid_crossentropy_backward_x, device, size, pgx, pgy, px, pz, nrow);
}

//--------------------------------------------------------------------------------------------------

__static__ void sigmoid_crossentropy_with_logits_host(float* py, float* px, float z, int64 ndat) {
    for (int64 n = 0; n < ndat; n++) {
        float x = px[n];
        py[n] = MAX(x, 0) - x * z + ::logf(1.0f + ::expf(-::fabs(x)));
    }
}

__global__ void sigmoid_crossentropy_with_logits_cuda(int64 size, float* py, float* px, float z, int64 ndat) {
    int64 n = blockIdx.x * blockDim.x + threadIdx.x;

    if (n < size) {
        float x = px[n];
        py[n] = MAX(x, 0) - x * z + ::logf(1.0f + ::expf(-::fabs(x)));
    }
}

void VMath::sigmoid_crossentropy_with_logits(int device, float* py, float* px, float z, int64 ndat) {
    int64 size = ndat;
    CUDA_CALL(sigmoid_crossentropy_with_logits, device, size, py, px, z, ndat);
}

//--------------------------------------------------------------------------------------------------

__static__ void sigmoid_crossentropy_with_logits_backward_host(float* pgx, float* pgy, float* px, float z, int64 ndat) {
    for (int64 n = 0; n < ndat; n++) {
        float prob = _sigmoid_host(px[n]);
        pgx[n] = pgy[n] * (prob - z);
    }
}

__global__ void sigmoid_crossentropy_with_logits_backward_cuda(int64 size, float* pgx, float* pgy, float* px, float z, int64 ndat) {
    int64 n = blockIdx.x * blockDim.x + threadIdx.x;

    if (n < size) {
        float prob = _sigmoid_cuda(px[n]);
        pgx[n] = pgy[n] * (prob - z);
    }
}

void VMath::sigmoid_crossentropy_with_logits_backward(int device, float* pgx, float* pgy, float* px, float z, int64 ndat) {
    int64 size = ndat;
    CUDA_CALL(sigmoid_crossentropy_with_logits_backward, device, size, pgx, pgy, px, z, ndat);
}

//--------------------------------------------------------------------------------------------------

__static__ void sigmoid_crossentropy_with_logits_idx_host(float* py, float* px, int* pz, int64 nrow, int64 ncol) {
    for (int64 nr = 0, n = 0; nr < nrow; nr++) {
        for (int64 nc = 0; nc < ncol; nc++, n++) {
            float x = px[n];
            float z = (pz[nr] == nc) ? 1.0f : 0.0f;
            py[n] = MAX(x, 0) - x * z + ::logf(1.0f + ::expf(-::fabs(x)));
        }
    }
}

__global__ void sigmoid_crossentropy_with_logits_idx_cuda(int64 size, float* py, float* px, int* pz, int64 nrow, int64 ncol) {
    int64 n = blockIdx.x * blockDim.x + threadIdx.x;

    if (n < size) {
        int64 nr = n / ncol;
        int64 nc = n % ncol;

        float x = px[n];
        float z = (pz[nr] == nc) ? 1.0f : 0.0f;
        py[n] = MAX(x, 0) - x * z + ::logf(1.0f + ::expf(-::fabs(x)));
    }
}

void VMath::sigmoid_crossentropy_with_logits_idx(int device, float* py, float* px, int* pz, int64 nrow, int64 ncol) {
    int64 size = nrow * ncol;
    CUDA_CALL(sigmoid_crossentropy_with_logits_idx, device, size, py, px, pz, nrow, ncol);
}

//--------------------------------------------------------------------------------------------------

__static__ void sigmoid_crossentropy_with_logits_idx_backward_host(float* pgx, float* pgy, float* px, int* pz, int64 nrow, int64 ncol) {
    for (int64 nr = 0, n = 0; nr < nrow; nr++) {
        for (int64 nc = 0; nc < ncol; nc++, n++) {
            float prob = _sigmoid_host(px[n]);
            float z = (pz[nr] == nc) ? 1.0f : 0.0f;
            pgx[n] = pgy[n] * (prob - z);
        }
    }
}

__global__ void sigmoid_crossentropy_with_logits_idx_backward_cuda(int64 size, float* pgx, float* pgy, float* px, int* pz, int64 nrow, int64 ncol) {
    int64 n = blockIdx.x * blockDim.x + threadIdx.x;

    if (n < size) {
        int64 nr = n / ncol;
        int64 nc = n % ncol;

        float prob = _sigmoid_cuda(px[n]);
        float z = (pz[nr] == nc) ? 1.0f : 0.0f;
        pgx[n] = pgy[n] * (prob - z);
    }
}

void VMath::sigmoid_crossentropy_with_logits_idx_backward(int device, float* pgx, float* pgy, float* px, int* pz, int64 nrow, int64 ncol) {
    int64 size = nrow * ncol;
    CUDA_CALL(sigmoid_crossentropy_with_logits_idx_backward, device, size, pgx, pgy, px, pz, nrow, ncol);
}

//--------------------------------------------------------------------------------------------------

__static__ void softmax_idx_crossentropy_host(float* py, float* px, int* pc, int64 nrow, int64 ncol) {
    for (int64 n = 0; n < nrow; n++) {
        if (pc[n] < 0 || pc[n] >= ncol) VP_THROW(VERR_OUT_OF_RANGE);

        float max_term = px[n * ncol];
        float exp_sum = 0;

        for (int64 m = 0; m < ncol; m++) {
            if (px[n * ncol + m] > max_term) max_term = px[n * ncol + m];
        }

        for (int64 m = 0; m < ncol; m++) {
            exp_sum += ::expf(px[n * ncol + m] - max_term);
        }

        float prob = ::expf(px[n * ncol + pc[n]] - max_term) / exp_sum;
        float entropy = -::logf(prob + 1.0e-10f);

        py[n] = entropy;
    }
}

__global__ void softmax_idx_crossentropy_cuda(int64 size, float* py, float* px, int* pc, int64 nrow, int64 ncol) {
    int64 n = blockIdx.x * blockDim.x + threadIdx.x;

    if (n < size) {
        if (pc[n] < 0 || pc[n] >= ncol) assert(0);

        float max_term = px[n * ncol];
        float exp_sum = 0;

        for (int64 m = 0; m < ncol; m++) {
            if (px[n * ncol + m] > max_term) max_term = px[n * ncol + m];
        }

        for (int64 m = 0; m < ncol; m++) {
            exp_sum += ::expf(px[n * ncol + m] - max_term);
        }

        float prob = ::expf(px[n * ncol + pc[n]] - max_term) / exp_sum;
        float entropy = -::logf(prob + 1.0e-10f);

        py[n] = entropy;
    }
}

void VMath::softmax_idx_crossentropy(int device, float* py, float* px, int* pc, int64 nrow, int64 ncol) {
    int64 size = nrow;
    CUDA_CALL(softmax_idx_crossentropy, device, size, py, px, pc, nrow, ncol);
}

//--------------------------------------------------------------------------------------------------

__static__ void softmax_i64_idx_crossentropy_host(float* py, float* px, int64* pc, int64 nrow, int64 ncol) {
    for (int64 n = 0; n < nrow; n++) {
        if (pc[n] < 0 || pc[n] >= ncol) VP_THROW(VERR_OUT_OF_RANGE);

        float max_term = px[n * ncol];
        float exp_sum = 0;

        for (int64 m = 0; m < ncol; m++) {
            if (px[n * ncol + m] > max_term) max_term = px[n * ncol + m];
        }

        for (int64 m = 0; m < ncol; m++) {
            exp_sum += ::expf(px[n * ncol + m] - max_term);
        }

        float prob = ::expf(px[n * ncol + pc[n]] - max_term) / exp_sum;
        float entropy = -::logf(prob + 1.0e-10f);

        py[n] = entropy;
    }
}

__global__ void softmax_i64_idx_crossentropy_cuda(int64 size, float* py, float* px, int64* pc, int64 nrow, int64 ncol) {
    int64 n = blockIdx.x * blockDim.x + threadIdx.x;

    if (n < size) {
        if (pc[n] < 0 || pc[n] >= ncol) assert(0);

        float max_term = px[n * ncol];
        float exp_sum = 0;

        for (int64 m = 0; m < ncol; m++) {
            if (px[n * ncol + m] > max_term) max_term = px[n * ncol + m];
        }

        for (int64 m = 0; m < ncol; m++) {
            exp_sum += ::expf(px[n * ncol + m] - max_term);
        }

        float prob = ::expf(px[n * ncol + pc[n]] - max_term) / exp_sum;
        float entropy = -::logf(prob + 1.0e-10f);

        py[n] = entropy;
    }
}

void VMath::softmax_i64_idx_crossentropy(int device, float* py, float* px, int64* pc, int64 nrow, int64 ncol) {
    int64 size = nrow;
    CUDA_CALL(softmax_i64_idx_crossentropy, device, size, py, px, pc, nrow, ncol);
}

//--------------------------------------------------------------------------------------------------

__static__ void softmax_idx_crossentropy_backward_x_host(float* pgx, float* pgy, float* px, int* pz, int64 nrow, int64 ncol) {
    // exp_sum을 구하는 과정을 불필요하게 반복하지 않는 구조
    for (int64 nr = 0; nr < nrow; nr++) {
        float* pgxVec = pgx + nr * ncol;
        float* pxVec = px + nr * ncol;

        float max_term = pxVec[0];
        float exp_sum = 0;

        for (int64 m = 0; m < ncol; m++) {
            if (pxVec[m] > max_term) max_term = pxVec[m];
        }

        for (int64 m = 0; m < ncol; m++) {
            exp_sum += ::expf(pxVec[m] - max_term);
        }

        for (int64 m = 0; m < ncol; m++) {
            pgxVec[m] = ::expf(pxVec[m] - max_term) / exp_sum;
            if (m == pz[nr]) pgxVec[m] -= 1.0f;
            pgxVec[m] *= pgy[nr];
        }
    }
}

__global__ void softmax_idx_crossentropy_backward_x_cuda(int64 size, float* pgx, float* pgy, float* px, int* pz, int64 nrow, int64 ncol) {
    // exp_sum을 구하는 과정을 중복되지만 스레드별로 처리함으로써 대신 호출 구조를 단순화하여 속도를 높이는 구조
    int64 n = blockIdx.x * blockDim.x + threadIdx.x;

    if (n < size) {
        int64 nr = n / ncol;
        int64 nc = n % ncol;

        float* pxVec = px + nr * ncol;

        float max_term = pxVec[0];
        float exp_sum = 0;

        for (int64 m = 0; m < ncol; m++) {
            if (pxVec[m] > max_term) max_term = pxVec[m];
        }

        for (int64 m = 0; m < ncol; m++) {
            exp_sum += ::expf(pxVec[m] - max_term);
        }

        pgx[n] = ::expf(px[n] - max_term) / exp_sum;
        if (nc == pz[nr]) pgx[n] -= 1.0f;
        pgx[n] *= pgy[nr];
    }
}

void VMath::softmax_idx_crossentropy_backward_x(int device, float* pgx, float* pgy, float* px, int* pz, int64 nrow, int64 ncol) {
    int64 size = nrow * ncol;
    CUDA_CALL(softmax_idx_crossentropy_backward_x, device, size, pgx, pgy, px, pz, nrow, ncol);
}

//--------------------------------------------------------------------------------------------------

__static__ void softmax_i64_idx_crossentropy_backward_x_host(float* pgx, float* pgy, float* px, int64* pz, int64 nrow, int64 ncol) {
    // exp_sum을 구하는 과정을 불필요하게 반복하지 않는 구조
    for (int64 nr = 0; nr < nrow; nr++) {
        float* pgxVec = pgx + nr * ncol;
        float* pxVec = px + nr * ncol;

        float max_term = pxVec[0];
        float exp_sum = 0;

        for (int64 m = 0; m < ncol; m++) {
            if (pxVec[m] > max_term) max_term = pxVec[m];
        }

        for (int64 m = 0; m < ncol; m++) {
            exp_sum += ::expf(pxVec[m] - max_term);
        }

        for (int64 m = 0; m < ncol; m++) {
            pgxVec[m] = ::expf(pxVec[m] - max_term) / exp_sum;
            if (m == pz[nr]) pgxVec[m] -= 1.0f;
            pgxVec[m] *= pgy[nr];
        }
    }
}

__global__ void softmax_i64_idx_crossentropy_backward_x_cuda(int64 size, float* pgx, float* pgy, float* px, int64* pz, int64 nrow, int64 ncol) {
    // exp_sum을 구하는 과정을 중복되지만 스레드별로 처리함으로써 대신 호출 구조를 단순화하여 속도를 높이는 구조
    int64 n = blockIdx.x * blockDim.x + threadIdx.x;

    if (n < size) {
        int64 nr = n / ncol;
        int64 nc = n % ncol;

        float* pxVec = px + nr * ncol;

        float max_term = pxVec[0];
        float exp_sum = 0;

        for (int64 m = 0; m < ncol; m++) {
            if (pxVec[m] > max_term) max_term = pxVec[m];
        }

        for (int64 m = 0; m < ncol; m++) {
            exp_sum += ::expf(pxVec[m] - max_term);
        }

        pgx[n] = ::expf(px[n] - max_term) / exp_sum;
        if (nc == pz[nr]) pgx[n] -= 1.0f;
        pgx[n] *= pgy[nr];
    }
}

void VMath::softmax_i64_idx_crossentropy_backward_x(int device, float* pgx, float* pgy, float* px, int64* pz, int64 nrow, int64 ncol) {
    int64 size = nrow * ncol;
    CUDA_CALL(softmax_i64_idx_crossentropy_backward_x, device, size, pgx, pgy, px, pz, nrow, ncol);
}

//--------------------------------------------------------------------------------------------------

__static__ void softmax_idx_crossentropy_pos_idx_host(float* py, float* px, int* pc, int64 nrow, int64 ncol) {
    for (int64 n = 0; n < nrow; n++) {
        if (pc[n] >= ncol) VP_THROW(VERR_UNDEFINED);

        if (pc[n] < 0) {
            py[n] = 0;
            return;
        }

        float max_term = px[n * ncol];
        float exp_sum = 0;

        for (int64 m = 0; m < ncol; m++) {
            if (px[n * ncol + m] > max_term) max_term = px[n * ncol + m];
        }

        for (int64 m = 0; m < ncol; m++) {
            exp_sum += ::expf(px[n * ncol + m] - max_term);
        }

        float prob = ::expf(px[n * ncol + pc[n]] - max_term) / exp_sum;
        float entropy = -::logf(prob + 1.0e-10f);

        py[n] = entropy;
    }
}

__global__ void softmax_idx_crossentropy_pos_idx_cuda(int64 size, float* py, float* px, int* pc, int64 nrow, int64 ncol) {
    int64 n = blockIdx.x * blockDim.x + threadIdx.x;

    if (n < size) {
        if (pc[n] >= ncol) assert(0);

        if (pc[n] < 0) {
            py[n] = 0;
            return;
        }

        float max_term = px[n * ncol];
        float exp_sum = 0;

        for (int64 m = 0; m < ncol; m++) {
            if (px[n * ncol + m] > max_term) max_term = px[n * ncol + m];
        }

        for (int64 m = 0; m < ncol; m++) {
            exp_sum += ::expf(px[n * ncol + m] - max_term);
        }

        float prob = ::expf(px[n * ncol + pc[n]] - max_term) / exp_sum;
        float entropy = -::logf(prob + 1.0e-10f);

        py[n] = entropy;
    }
}

void VMath::softmax_idx_crossentropy_pos_idx(int device, float* py, float* px, int* pc, int64 nrow, int64 ncol) {
    int64 size = nrow;
    CUDA_CALL(softmax_idx_crossentropy_pos_idx, device, size, py, px, pc, nrow, ncol);
}

//--------------------------------------------------------------------------------------------------

__static__ void softmax_idx_crossentropy_pos_idx_backward_x_host(float* pgx, float* pgy, float* px, int* pz, int64 nrow, int64 ncol) {
    // exp_sum을 구하는 과정을 불필요하게 반복하지 않는 구조
    for (int64 nr = 0; nr < nrow; nr++) {
        float* pgxVec = pgx + nr * ncol;
        float* pxVec = px + nr * ncol;

        if (pz[nr] < 0) {
            for (int64 m = 0; m < ncol; m++) {
                pgxVec[m] = 0;
            }
            return;
        }

        float max_term = pxVec[0];
        float exp_sum = 0;

        for (int64 m = 0; m < ncol; m++) {
            if (pxVec[m] > max_term) max_term = pxVec[m];
        }

        for (int64 m = 0; m < ncol; m++) {
            exp_sum += ::expf(pxVec[m] - max_term);
        }

        for (int64 m = 0; m < ncol; m++) {
            pgxVec[m] = ::expf(pxVec[m] - max_term) / exp_sum;
            if (m == pz[nr]) pgxVec[m] -= 1.0f;
            pgxVec[m] *= pgy[nr];
        }
    }
}

__global__ void softmax_idx_crossentropy_pos_idx_backward_x_cuda(int64 size, float* pgx, float* pgy, float* px, int* pz, int64 nrow, int64 ncol) {
    // exp_sum을 구하는 과정을 중복되지만 스레드별로 처리함으로써 대신 호출 구조를 단순화하여 속도를 높이는 구조
    int64 n = blockIdx.x * blockDim.x + threadIdx.x;

    if (n < size) {
        int64 nr = n / ncol;
        int64 nc = n % ncol;

        if (pz[nr] < 0) {
            pgx[n] = 0;
            return;
        }

        float* pxVec = px + nr * ncol;

        float max_term = pxVec[0];
        float exp_sum = 0;

        for (int64 m = 0; m < ncol; m++) {
            if (pxVec[m] > max_term) max_term = pxVec[m];
        }

        for (int64 m = 0; m < ncol; m++) {
            exp_sum += ::expf(pxVec[m] - max_term);
        }

        pgx[n] = ::expf(px[n] - max_term) / exp_sum;
        if (nc == pz[nr]) pgx[n] -= 1.0f;
        pgx[n] *= pgy[nr];
    }
}

void VMath::softmax_idx_crossentropy_pos_idx_backward_x(int device, float* pgx, float* pgy, float* px, int* pz, int64 nrow, int64 ncol) {
    int64 size = nrow * ncol;
    CUDA_CALL(softmax_idx_crossentropy_pos_idx_backward_x, device, size, pgx, pgy, px, pz, nrow, ncol);
}

//--------------------------------------------------------------------------------------------------

__static__ void softmax_crossentropy_host(float* py, float* px, float* pz, int64 nrow, int64 ncol) {
    for (int64 n = 0; n < nrow; n++) {
        float max_term = px[n * ncol];
        float exp_sum = 0;
        float entropy = 0;

        for (int64 m = 0; m < ncol; m++) {
            if (px[n * ncol + m] > max_term) max_term = px[n * ncol + m];
        }

        for (int64 m = 0; m < ncol; m++) {
            exp_sum += ::expf(px[n * ncol + m] - max_term);
        }

        for (int64 m = 0; m < ncol; m++) {
            float prob = ::expf(px[n * ncol + m] - max_term) / exp_sum;
            entropy -= pz[n * ncol + m] * ::logf(prob + 1.0e-10f);
        }

        py[n] = entropy;
    }
}

__global__ void softmax_crossentropy_cuda(int64 size, float* py, float* px, float* pz, int64 nrow, int64 ncol) {
    int64 n = blockIdx.x * blockDim.x + threadIdx.x;

    if (n < size) {
        float max_term = px[n * ncol];
        float exp_sum = 0;
        float entropy = 0;

        for (int64 m = 0; m < ncol; m++) {
            if (px[n * ncol + m] > max_term) max_term = px[n * ncol + m];
        }

        for (int64 m = 0; m < ncol; m++) {
            exp_sum += ::expf(px[n * ncol + m] - max_term);
        }

        for (int64 m = 0; m < ncol; m++) {
            float prob = ::expf(px[n * ncol + m] - max_term) / exp_sum;
            entropy -= pz[n * ncol + m] * ::logf(prob + 1.0e-10f);
        }

        py[n] = entropy;
    }
}

void VMath::softmax_crossentropy(int device, float* py, float* px, float* pz, int64 nrow, int64 ncol) {
    int64 size = nrow;
    CUDA_CALL(softmax_crossentropy, device, size, py, px, pz, nrow, ncol);
}

//--------------------------------------------------------------------------------------------------

__static__ void softmax_crossentropy_backward_x_host(float* pgx, float* pgy, float* px, float* pz, int64 nrow, int64 ncol) {
    for (int64 nr = 0; nr < nrow; nr++) {
        float* pgxVec = pgx + nr * ncol;
        float* pxVec = px + nr * ncol;

        float max_term = pxVec[0];
        float exp_sum = 0;

        for (int64 m = 0; m < ncol; m++) {
            if (pxVec[m] > max_term) max_term = pxVec[m];
        }

        for (int64 m = 0; m < ncol; m++) {
            exp_sum += ::expf(pxVec[m] - max_term);
        }

        for (int64 m = 0; m < ncol; m++) {
            float prob = ::expf(pxVec[m] - max_term) / exp_sum;
            pgxVec[m] = (prob - pz[nr * ncol + m]) * pgy[nr];
        }
    }
}

__global__ void softmax_crossentropy_backward_x_cuda(int64 size, float* pgx, float* pgy, float* px, float* pz, int64 nrow, int64 ncol) {
    // exp_sum을 구하는 과정을 중복되지만 스레드별로 처리함으로써 대신 호출 구조를 단순화하여 속도를 높이는 구조
    int64 n = blockIdx.x * blockDim.x + threadIdx.x;

    if (n < size) {
        int64 nr = n / ncol;

        float* pxVec = px + nr * ncol;

        float max_term = pxVec[0];
        float exp_sum = 0;

        for (int64 m = 0; m < ncol; m++) {
            if (pxVec[m] > max_term) max_term = pxVec[m];
        }

        for (int64 m = 0; m < ncol; m++) {
            exp_sum += ::expf(pxVec[m] - max_term);
        }

        float prob = ::expf(px[n] - max_term) / exp_sum;
        pgx[n] = (prob - pz[n]) * pgy[nr];
    }
}

void VMath::softmax_crossentropy_backward_x(int device, float* pgx, float* pgy, float* px, float* pz, int64 nrow, int64 ncol) {
    int64 size = nrow * ncol;
    CUDA_CALL(softmax_crossentropy_backward_x, device, size, pgx, pgy, px, pz, nrow, ncol);
}

//--------------------------------------------------------------------------------------------------

__static__ void transpose_host(float* py, float* px, int* pn, int64 axis_size, int64 data_size) {
    for (int64 n = 0; n < data_size; n++) {
        int64 xpos = 0;
        int64 ypos = n;

        int64 cood;

        for (int64 m = 0; m < axis_size; m++) {
            int64 y_block_size = pn[m * 3 + 1];
            int64 x_block_size = pn[pn[m * 3] * 3 + 2];

            cood = ypos / y_block_size;
            ypos = ypos % y_block_size;

            xpos += cood * x_block_size;
        }

        py[n] = px[xpos];
    }
}

__global__ void transpose_cuda(int64 size, float* py, float* px, int* pn, int64 axis_size, int64 data_size) {
    int64 n = blockIdx.x * blockDim.x + threadIdx.x;

    if (n < size) {
        int xpos = 0;
        int ypos = n;

        int cood;

        for (int64 m = 0; m < axis_size; m++) {
            int y_block_size = pn[m * 3 + 1];
            int x_block_size = pn[pn[m * 3] * 3 + 2];

            cood = ypos / y_block_size;
            ypos = ypos % y_block_size;

            xpos += cood * x_block_size;
        }

        py[n] = px[xpos];
    }
}

void VMath::transpose(int device, float* py, float* px, int* pn, int64 axis_size, int64 data_size) {
    int64 size = data_size;
    CUDA_CALL(transpose, device, size, py, px, pn, axis_size, data_size);
}

//--------------------------------------------------------------------------------------------------

__static__ void transpose_backward_host(float* pgx, float* pgy, int* pn, int64 axis_size, int64 data_size) {
    for (int64 n = 0; n < data_size; n++) {
        int64 xpos = 0;
        int64 ypos = n;

        int64 cood;

        for (int64 m = 0; m < axis_size; m++) {
            int64 y_block_size = pn[m * 3 + 1];
            int64 x_block_size = pn[pn[m * 3] * 3 + 2];

            cood = ypos / y_block_size;
            ypos = ypos % y_block_size;

            xpos += cood * x_block_size;
        }

        pgx[xpos] = pgy[n];
    }
}

__global__ void transpose_backward_cuda(int64 size, float* pgx, float* pgy, int* pn, int64 axis_size, int64 data_size) {
    int64 n = blockIdx.x * blockDim.x + threadIdx.x;

    if (n < size) {
        int xpos = 0;
        int ypos = n;

        int cood;

        for (int64 m = 0; m < axis_size; m++) {
            int y_block_size = pn[m * 3 + 1];
            int x_block_size = pn[pn[m * 3] * 3 + 2];

            cood = ypos / y_block_size;
            ypos = ypos % y_block_size;

            xpos += cood * x_block_size;
        }

        pgx[xpos] = pgy[n];
    }
}

void VMath::transpose_backward(int device, float* pgx, float* pgy, int* pn, int64 axis_size, int64 data_size) {
    int64 size = data_size;
    CUDA_CALL(transpose_backward, device, size, pgx, pgy, pn, axis_size, data_size);
}

//--------------------------------------------------------------------------------------------------

__static__ void transpose_bin_host(float* py, float* px, int64 nrow, int64 ncol) {
    for (int64 nr = 0, n = 0; nr < nrow; nr++) {
        for (int64 nc = 0; nc < ncol; nc++, n++) {
            int64 xpos = nc * nrow + nr;
            py[n] = px[xpos];
        }
    }
}

__global__ void transpose_bin_cuda(int64 size, float* py, float* px, int64 nrow, int64 ncol) {
    int64 n = blockIdx.x * blockDim.x + threadIdx.x;

    if (n < size) {
        int64 nr = n / ncol;
        int64 nc = n % ncol;

        int64 xpos = nc * nrow + nr;
        py[n] = px[xpos];
    }
}

void VMath::transpose_bin(int device, float* py, float* px, int64 nrow, int64 ncol) {
    int64 size = ncol * nrow;
    CUDA_CALL(transpose_bin, device, size, py, px, nrow, ncol);
}

//--------------------------------------------------------------------------------------------------

__static__ void max_host(float* py, float* px, int64 nrow, int64 ncol) {
    for (int64 n = 0; n < nrow; n++) {
        float max_val = -FLT_MAX;

        for (int64 nc = 0; nc < ncol; nc++) {
            float x = px[n * ncol + nc];
            if (x > max_val) max_val = x;
        }

        py[n] = max_val;
    }
}

__global__ void max_cuda(int64 size, float* py, float* px, int64 nrow, int64 ncol) {
    int64 n = blockIdx.x * blockDim.x + threadIdx.x;

    if (n < size) {
        float max_val = -FLT_MAX;

        for (int64 nc = 0; nc < ncol; nc++) {
            float x = px[n * ncol + nc];
            if (x > max_val) max_val = x;
        }

        py[n] = max_val;
    }
}

void VMath::max(int device, float* py, float* px, int64 nrow, int64 ncol) {
    int64 size = nrow;
    CUDA_CALL(max, device, size, py, px, nrow, ncol);
}

//--------------------------------------------------------------------------------------------------

__static__ void min_host(float* py, float* px, int64 nrow, int64 ncol) {
    for (int64 n = 0; n < nrow; n++) {
        float min_val = FLT_MAX;

        for (int64 nc = 0; nc < ncol; nc++) {
            float x = px[n * ncol + nc];
            if (x < min_val) min_val = x;
        }

        py[n] = min_val;
    }
}

__global__ void min_cuda(int64 size, float* py, float* px, int64 nrow, int64 ncol) {
    int64 n = blockIdx.x * blockDim.x + threadIdx.x;

    if (n < size) {
        float min_val = FLT_MAX;

        for (int64 nc = 0; nc < ncol; nc++) {
            float x = px[n * ncol + nc];
            if (x < min_val) min_val = x;
        }

        py[n] = min_val;
    }
}

void VMath::min(int device, float* py, float* px, int64 nrow, int64 ncol) {
    int64 size = nrow;
    CUDA_CALL(min, device, size, py, px, nrow, ncol);
}

//--------------------------------------------------------------------------------------------------

__static__ void argmax_host(int* py, float* px, int64 nrow, int64 ncol) {
    for (int64 n = 0; n < nrow; n++) {
        float max_val = -FLT_MAX;
        int64 max_pos = -1;

        for (int64 nc = 0; nc < ncol; nc++) {
            float x = px[n * ncol + nc];
            if (x > max_val) {
                max_val = x;
                max_pos = nc;
            }
        }

        py[n] = (int)max_pos;
    }
}

__global__ void argmax_cuda(int64 size, int* py, float* px, int64 nrow, int64 ncol) {
    int64 n = blockIdx.x * blockDim.x + threadIdx.x;

    if (n < size) {
        float max_val = -FLT_MAX;
        int64 max_pos = -1;

        for (int64 nc = 0; nc < ncol; nc++) {
            float x = px[n * ncol + nc];
            if (x > max_val) {
                max_val = x;
                max_pos = nc;
            }
        }

        py[n] = (int)max_pos;
    }
}

void VMath::argmax(int device, int* py, float* px, int64 nrow, int64 ncol) {
    int64 size = nrow;
    CUDA_CALL(argmax, device, size, py, px, nrow, ncol);
}

//--------------------------------------------------------------------------------------------------

__static__ void argmin_host(int* py, float* px, int64 nrow, int64 ncol) {
    for (int64 n = 0; n < nrow; n++) {
        float min_val = FLT_MAX;
        int64 min_pos = -1;

        for (int64 nc = 0; nc < ncol; nc++) {
            float x = px[n * ncol + nc];
            if (x < min_val) {
                min_val = x;
                min_pos = nc;
            }
        }

        py[n] = (int)min_pos;
    }
}

__global__ void argmin_cuda(int64 size, int* py, float* px, int64 nrow, int64 ncol) {
    int64 n = blockIdx.x * blockDim.x + threadIdx.x;

    if (n < size) {
        float min_val = FLT_MAX;
        int64 min_pos = -1;

        for (int64 nc = 0; nc < ncol; nc++) {
            float x = px[n * ncol + nc];
            if (x < min_val) {
                min_val = x;
                min_pos = nc;
            }
        }

        py[n] = (int)min_pos;
    }
}

void VMath::argmin(int device, int* py, float* px, int64 nrow, int64 ncol) {
    int64 size = nrow;
    CUDA_CALL(argmin, device, size, py, px, nrow, ncol);
}

//--------------------------------------------------------------------------------------------------

__static__ void mean_host(float* py, float* px, int64 nrow) {
    float sum = 0;
    for (int64 n = 0; n < nrow; n++) sum += px[n];
    py[0] = sum / (float)nrow;
}

__global__ void mean_cuda(int64 size, float* py, float* px, int64 nrow) {
    int64 n = blockIdx.x * blockDim.x + threadIdx.x;

    if (n < size) {
        float sum = 0;
        for (int64 n = 0; n < nrow; n++) sum += px[n];
        py[0] = sum / (float)nrow;
    }
}

void VMath::mean(int device, float* py, float* px, int64 nrow) {
    int64 size = 1;
    CUDA_CALL(mean, device, size, py, px, nrow);
}

//--------------------------------------------------------------------------------------------------

__static__ void mean_backward_host(float* pgx, float* pgy, int64 nrow) {
    float coef = (pgy ? pgy[0] : 1.0f) / (float)nrow;
    for (int64 n = 0; n < nrow; n++) {
        pgx[n] = coef;
    }
}

__global__ void mean_backward_cuda(int64 size, float* pgx, float* pgy, int64 nrow) {
    int64 n = blockIdx.x * blockDim.x + threadIdx.x;

    if (n < size) {
        float coef = (pgy ? pgy[0] : 1.0f) / (float)nrow;
        pgx[n] = coef;
    }
}

void VMath::mean_backward(int device, float* pgx, float* pgy, int64 nrow) {
    int64 size = nrow;
    CUDA_CALL(mean_backward, device, size, pgx, pgy, nrow);
}

//--------------------------------------------------------------------------------------------------

__static__ void sum_int_host(int* py, int* px, int64 nrow) {
    int sum = 0;
    for (int64 n = 0; n < nrow; n++) sum += px[n];
    py[0] = sum;
}

__global__ void sum_int_cuda(int64 size, int* py, int* px, int64 nrow) {
    int64 n = blockIdx.x * blockDim.x + threadIdx.x;

    if (n < size) {
        int sum = 0;
        for (int64 n = 0; n < nrow; n++) sum += px[n];
        py[0] = sum;
    }
}

void VMath::sum_int(int device, int* py, int* px, int64 nrow) {
    int64 size = 1;
    CUDA_CALL(sum_int, device, size, py, px, nrow);
}

//--------------------------------------------------------------------------------------------------

__static__ void sum_host(float* py, float* px, int64 nrow) {
    float sum = 0;
    for (int64 n = 0; n < nrow; n++) sum += px[n];
    py[0] = sum;
}

__global__ void sum_cuda(int64 size, float* py, float* px, int64 nrow) {
    int64 n = blockIdx.x * blockDim.x + threadIdx.x;

    if (n < size) {
        float sum = 0;
        for (int64 n = 0; n < nrow; n++) sum += px[n];
        py[0] = sum;
    }
}

void VMath::sum(int device, float* py, float* px, int64 nrow) {
    int64 size = 1;
    CUDA_CALL(sum, device, size, py, px, nrow);
}

//--------------------------------------------------------------------------------------------------

__static__ void sum_backward_host(float* pgx, float* pgy, int64 nrow) {
    for (int64 n = 0; n < nrow; n++) {
        pgx[n] = pgy ? pgy[0] : 1.0f;
    }
}

__global__ void sum_backward_cuda(int64 size, float* pgx, float* pgy, int64 nrow) {
    int64 n = blockIdx.x * blockDim.x + threadIdx.x;

    if (n < size) {
        pgx[n] = pgy ? pgy[0] : 1.0f;
    }
}

void VMath::sum_backward(int device, float* pgx, float* pgy, int64 nrow) {
    int64 size = nrow;
    CUDA_CALL(sum_backward, device, size, pgx, pgy, nrow);
}

//--------------------------------------------------------------------------------------------------

__static__ void fft_wave_to_complex_host(float* pbuf, float* pwave, int64 bsize, int64 spec_interval, int64 step_cnt, int64 fft_width, int64 samples_in_data) {
    VP_THROW(VERR_NOT_IMPLEMENTED_YET);
}

__global__ void fft_wave_to_complex_cuda(int64 size, float* pbuf, float* pwave, int64 bsize, int64 spec_interval, int64 step_cnt, int64 fft_width, int64 samples_in_data) {
    int64 n = blockIdx.x * blockDim.x + threadIdx.x;

    if (n < size) {
        int64 nd = n / (step_cnt * fft_width * 2);
        int64 ns = n / (fft_width * 2) % step_cnt;
        int64 nx = (n / 2) % fft_width;

        bool is_real = (n % 2) == 0;

        if (is_real) {
            int64 wpos = nd * samples_in_data + ns * spec_interval + nx;
            pbuf[n] = pwave[wpos];
        }
        else {
            pbuf[n] = 0;
        }
    }
}

void VMath::fft_wave_to_complex(float* pbuf, float* pwave, int64 bsize, int64 spec_interval, int64 step_cnt, int64 fft_width, int64 samples_in_data) {
    int64 size = bsize;
    CUDA_CALL(fft_wave_to_complex, 0, size, pbuf, pwave, bsize, spec_interval, step_cnt, fft_width, samples_in_data);
}

//--------------------------------------------------------------------------------------------------

__static__ void fft_step_split_host(float* pdst, float* psrc, int64 ssize, int64 fft_width, int64 step) {
    VP_THROW(VERR_NOT_IMPLEMENTED_YET);
}

__global__ void fft_step_split_cuda(int64 size, float* pdst, float* psrc, int64 ssize, int64 fft_width, int64 step) {
    int64 n = blockIdx.x * blockDim.x + threadIdx.x;

    if (n < size) {
        int64 nd = n / (fft_width * 2);
        int64 nx = (n / 2) % fft_width;
        bool is_real = (n % 2) == 0;

        int64 stride = fft_width / step;

        int64 pos1 = (nx / stride * (2 * stride)) % fft_width + nx % stride;
        int64 pos2 = pos1 + stride;

        float x1_real = psrc[(nd * fft_width + pos1) * 2];
        float x1_image = psrc[(nd * fft_width + pos1) * 2 + 1];

        float x2_real = psrc[(nd * fft_width + pos2) * 2];
        float x2_image = psrc[(nd * fft_width + pos2) * 2 + 1];

        float theta = -2 * PI_F * (nx / stride * stride) / fft_width;

        float t_real = ::cosf(theta);
        float t_image = ::sinf(theta);

        int64 didx = (nd * fft_width + nx) * 2;

        if (is_real)
            pdst[didx] = x1_real + x2_real * t_real - x2_image * t_image;
        else
            pdst[didx + 1] = x1_image + x2_real * t_image + x2_image * t_real;
    }
}

void VMath::fft_step_split(float* pdst, float* psrc, int64 ssize, int64 fft_width, int64 step) {
    int64 size = ssize;
    CUDA_CALL(fft_step_split, 0, size, pdst, psrc, ssize, fft_width, step);
}

//--------------------------------------------------------------------------------------------------

__static__ void fft_complex_to_abs_mean_host(float* pffts, float* psrc, int64 fsize, int64 fft_width, int64 freq_in_spectrum) {
    VP_THROW(VERR_NOT_IMPLEMENTED_YET);
}

__global__ void fft_complex_to_abs_mean_cuda(int64 size, float* pffts, float* psrc, int64 fsize, int64 fft_width, int64 freq_in_spectrum) {
    int64 n = blockIdx.x * blockDim.x + threadIdx.x;

    if (n < size) {
        int64 nd = n / freq_in_spectrum;
        int64 nx = n % freq_in_spectrum;

        int64 nfreq = fft_width / 2 + 2;

        float log_grid = ::logf((float)nfreq) / freq_in_spectrum;

        float log_start = nx * log_grid;
        float log_end = (nx + 1) * log_grid;

        float fft_start = ::expf(log_start) - 1;
        float fft_end = ::expf(log_end) - 1;

        int nfrom = (int)fft_start;
        int nto = (int)::ceilf(fft_end);

        pffts[n] = (nd == 0) ? (float)nfrom : (float)nto;

        float sum = 0;

        for (int nf = nfrom; nf < nto; nf++) {
            int64 cidx = nd * fft_width + nf;

            float real = psrc[cidx * 2];
            float image = psrc[cidx * 2 + 1];

            float coef = 1.0f;

            if (nf < fft_start) coef -= fft_start - nf;
            if (nf + 1> fft_end) coef -= nf + 1 - fft_end;

            sum += coef * ::sqrtf(real * real + image * image);
        }

        pffts[n] = ::logf(sum + 1e-10f);
    }
}

void VMath::fft_complex_to_abs_mean(float* pffts, float* psrc, int64 fsize, int64 fft_width, int64 freq_in_spectrum) {
    int64 size = fsize;
    CUDA_CALL(fft_complex_to_abs_mean, 0, size, pffts, psrc, fsize, fft_width, freq_in_spectrum);
}

//--------------------------------------------------------------------------------------------------
