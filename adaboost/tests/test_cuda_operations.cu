#ifndef TEST_CUDA_ADABOOST_CORE_OPERATIONS_IMPL_CU
#define TEST_CUDA_ADABOOST_CORE_OPERATIONS_IMPL_CU

#include<gtest/gtest.h>
#include<adaboost/cuda/core/cuda_data_structures.hpp>
#include<adaboost/cuda/utils/cuda_wrappers.hpp>
#include<adaboost/cuda/core/operations.cu>
#include<stdexcept>


__device__ float square_1_in(float x)
{
    return -x*x;
}

__device__  adaboost::cuda::core::func_t<float,float> p_func_here = square_1_in;

TEST(Cuda, Argmax)
{
    adaboost::utils::cuda::cuda_event_t has_happened;
    adaboost::utils::cuda::cuda_event_create(&has_happened);
    adaboost::cuda::core::VectorGPU<float> vec_i(5);
    vec_i.set(0, (float)-1.0);
    vec_i.set(1, (float)0.);
    vec_i.set(2, (float)1.);
    vec_i.set(3, (float)2.);
    vec_i.set(4, (float)3.);
    unsigned int block_size = 2;
    adaboost::utils::cuda::cuda_event_record(has_happened);
    adaboost::utils::cuda::cuda_event_synchronize(has_happened);
    unsigned result_gpu;

    adaboost::cuda::core::func_t <float, float> h_func;
    // adaboost::utils::cuda::cuda_malloc((void**)&h_func, sizeof(func_t <data_type_vec, data_type_ret>));
    
    // cudaMemcpyToSymbol(h_func, &p_func, sizeof(func_t <data_type_vec, data_type_ret>), 0, cudaMemcpyHostToDevice);
    // cudaMemcpyFromSymbol(&h_func, p_func_here, sizeof(adaboost::cuda::core::func_t <float, float>));
    // cudaError_t err = cudaGetLastError();        // Get error code
    // if ( err != cudaSuccess ){
    //     printf("CUDA Error: %s\n", cudaGetErrorString(err));
    //         exit(-1);
    // }
    adaboost::cuda::core::Argmax(square_1_in, vec_i, result_gpu, block_size);
    adaboost::utils::cuda::cuda_event_record(has_happened);
    adaboost::utils::cuda::cuda_event_synchronize(has_happened);
    EXPECT_EQ(4, result_gpu)<<"The arg max value is at 4.";
}
#endif