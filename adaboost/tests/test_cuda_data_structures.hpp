#include<gtest/gtest.h>
#include<string>
#include<adaboost/cuda/cuda_data_structures.hpp>
#include<adaboost/utils/cuda_wrappers.hpp>
#include<stdexcept>

TEST(Cuda, VectorGPU)
{
    adaboost::utils::cuda::cuda_event_t has_happened;
    adaboost::utils::cuda::cuda_event_create(&has_happened);
    adaboost::cuda::core::VectorGPU<float> vec1(1000);
    adaboost::cuda::core::VectorGPU<float> vec2(1000);
    unsigned block_size = 32;
    vec1.fill(1.0, block_size);
    vec2.fill(1.0, block_size);
    adaboost::utils::cuda::cuda_event_record(has_happened);
    adaboost::utils::cuda::cuda_event_synchronize(has_happened);
    float result_gpu;
    product_gpu(vec1, vec2, result_gpu, block_size);
    adaboost::utils::cuda::cuda_event_record(has_happened);
    adaboost::utils::cuda::cuda_event_synchronize(has_happened);
    EXPECT_EQ(result_gpu, 1000.0)<<"Result from product on GPU should be 1000.0";
    vec1.copy_to_host();
    vec2.copy_to_host();
    adaboost::utils::cuda::cuda_event_record(has_happened);
    adaboost::utils::cuda::cuda_event_synchronize(has_happened);
    for(unsigned i = 0; i < 1000; i++)
    {
        std::string msg1 = "All entries of VectorGPU should be 1";
        EXPECT_EQ(1, vec1.at(i))<<msg1;
        EXPECT_EQ(1, vec2.at(i))<<msg1;
    }
    float result;
    product_gpu(vec1, vec2, result);
    EXPECT_EQ(result, 1000.0)<<"Result from product on CPU should be 1000.0";
}
