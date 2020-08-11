#ifndef TEST_CUDA_ADABOOST_CORE_OPERATIONS_IMPL_CU
#define TEST_CUDA_ADABOOST_CORE_OPERATIONS_IMPL_CU

#include<gtest/gtest.h>
#include<adaboost/cuda/core/cuda_data_structures.hpp>
#include<adaboost/cuda/utils/cuda_wrappers.hpp>
#include<adaboost/cuda/core/operations.hpp>
#include<stdexcept>


TEST(Cuda, Argmax)
{
    adaboost::utils::cuda::cuda_event_t has_happened;
    adaboost::utils::cuda::cuda_event_create(&has_happened);
    adaboost::cuda::core::VectorGPU<float> vec_i(10);
    vec_i.set(0, (float)-1.0);
    vec_i.set(1, (float)0.);
    vec_i.set(2, (float)1.);
    vec_i.set(3, (float)2.);
    vec_i.set(4, (float)3.);
    vec_i.set(5, (float)4.);
    vec_i.set(6, (float)9.);
    vec_i.set(7, (float)12.);
    vec_i.set(8, (float)8.);
    vec_i.set(9, (float)6.);
    unsigned int block_size = 3;
    unsigned int grid_size = 2;
    unsigned int option = 1;
    adaboost::utils::cuda::cuda_event_record(has_happened);
    adaboost::utils::cuda::cuda_event_synchronize(has_happened);
    unsigned result_gpu;
    vec_i.copy_to_device();
    float *val=(float *)(malloc)(sizeof(float));
    adaboost::cuda::core::Argmax(option, vec_i, result_gpu, grid_size, block_size, val);
    free(val);
    adaboost::utils::cuda::cuda_event_record(has_happened);
    adaboost::utils::cuda::cuda_event_synchronize(has_happened);
    EXPECT_EQ(7, result_gpu)<<"The arg max value is at 7.";
}
#endif
