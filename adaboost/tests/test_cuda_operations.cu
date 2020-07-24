#include<gtest/gtest.h>
#include<adaboost/cuda/core/cuda_data_structures.hpp>
#include<adaboost/cuda/utils/cuda_wrappers.hpp>
#include<adaboost/cuda/core/operations.cu>
#include<stdexcept>


__device__ float square_1(float x)
{
    return x*x;
}

__device__ float square_2(float x)
{
    return -x*x;
}

__device__ const adaboost::cuda::core::func_t<float,float> p_func = square_1;

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
    adaboost::cuda::core::Argmax(p_func, vec_i, result_gpu, block_size);
    adaboost::utils::cuda::cuda_event_record(has_happened);
    adaboost::utils::cuda::cuda_event_synchronize(has_happened);
    EXPECT_EQ(4, result_gpu)<<"The arg max value is at 4.";
}
