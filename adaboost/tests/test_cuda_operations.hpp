#include<gtest/gtest.h>
#include<string>
#include<adaboost/cuda/core/cuda_data_structures.hpp>
#include<adaboost/cuda/utils/cuda_wrappers.hpp>
#include<adaboost/core/operations.hpp>
#include<adaboost/cuda/core/operations.hpp>
#include<stdexcept> 

float square_1(float x)
{
    return x*x;
}

float square_2(float x)
{
    return -x*x;
}

TEST(Cuda, Argmax)
{
    adaboost::utils::cuda::cuda_event_t has_happened;
    adaboost::utils::cuda::cuda_event_create(&has_happened);

    adaboost::cuda::core::VectorGPU<float> vec_i(5);
    unsigned int block_size = 2;
    vec_i.set(0, -1.0);
    vec_i.set(1, 0);
    vec_i.set(2, 1);
    vec_i.set(3, 2);
    vec_i.set(4, 3);
    float result;
    adaboost::cuda::core::Argmax(&square_1, vec_i, result,0);
    EXPECT_EQ(3, result)<<"The arg max value is at 3.";
}