#include<gtest/gtest.h>
#include<string>
#include<adaboost/cuda/core/cuda_data_structures.hpp>
#include<adaboost/cuda/utils/cuda_wrappers.hpp>
#include<adaboost/core/operations.hpp>
#include<adaboost/cuda/core/operations.hpp>
#include<stdexcept> 

TEST(Cuda, VectorGPU)
{
    adaboost::utils::cuda::cuda_event_t has_happened;
    adaboost::utils::cuda::cuda_event_create(&has_happened);
    adaboost::cuda::core::VectorGPU<float> vec1(1000);
    adaboost::cuda::core::VectorGPU<float> vec2(1000);
    unsigned int block_size = 32;
    adaboost::cuda::core::fill(float(1.0),vec1, (unsigned int)32);
    adaboost::cuda::core::fill(float(1.0),vec2, block_size);
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

TEST(Cuda, MatrixGPU)
{
    adaboost::utils::cuda::cuda_event_t has_happened;
    adaboost::utils::cuda::cuda_event_create(&has_happened);
    adaboost::cuda::core::MatrixGPU<float> mat_f;
    EXPECT_EQ(0, mat_f.get_cols())<<"Number of columns should be 0";
    EXPECT_EQ(0, mat_f.get_rows())<<"Number of rows should be 0.";
    adaboost::cuda::core::MatrixGPU<float> mat1(3, 3), mat2(3, 3), mat3(2, 1);
    mat1.copy_to_device();
    mat2.copy_to_device();
    fill(float(4.0),mat1,0,0);
    fill(float(5.0),mat2,0,0);
    mat2.copy_to_host();
    for(unsigned int i = 0; i < 3; i++)
    {
        for(unsigned int j = 0; j < 3; j++)
        {
            EXPECT_EQ(5.0, mat2.at(i, j))<<"Fail at"<<i<<" "<<j;
        }
    }
    mat2.copy_to_device();
    adaboost::utils::cuda::cuda_event_record(has_happened);
    adaboost::utils::cuda::cuda_event_synchronize(has_happened);
    adaboost::cuda::core::MatrixGPU<float> result1(3, 3);
    result1.copy_to_device();
    adaboost::cuda::core::multiply_gpu(mat1, mat2, result1);
    adaboost::utils::cuda::cuda_event_record(has_happened);
    adaboost::utils::cuda::cuda_event_synchronize(has_happened);
    result1.copy_to_host();
    for(unsigned int i = 0; i < 3; i++)
    {
        for(unsigned int j = 0; j < 3; j++)
        {
            EXPECT_EQ(60.0, result1.at(i, j));
        }
    }
    mat3.set(0, 0, 6.0);
    mat3.set(1, 0, 6.0);
    EXPECT_THROW({
        try
        {
            adaboost::cuda::core::multiply_gpu(mat1, mat3, result1);
        }
        catch(const std::logic_error& e)
        {
            EXPECT_STREQ("Order of matrices don't match.", e.what());
            throw;
        }
    }, std::logic_error);
}

TEST(Cuda, MatricesGPU)
{
    adaboost::utils::cuda::cuda_event_t has_happened;
    adaboost::utils::cuda::cuda_event_create(&has_happened);
    adaboost::cuda::core::MatrixGPU<float> mat_f;
    EXPECT_EQ(0, mat_f.get_cols())<<"Number of columns should be 0";
    EXPECT_EQ(0, mat_f.get_rows())<<"Number of rows should be 0.";
    adaboost::cuda::core::MatrixGPU<float> mat1(3, 3), mat2(3, 3), mat3(2, 1);
    mat1.copy_to_device();
    mat2.copy_to_device();
    adaboost::cuda::core::fill(float(4.0), mat1, 0, 0);
    adaboost::cuda::core::fill(float(5.0), mat2, 0, 0);
    adaboost::utils::cuda::cuda_event_record(has_happened);
    adaboost::utils::cuda::cuda_event_synchronize(has_happened);
    adaboost::cuda::core::MatrixGPU<float> result1(3, 3);
    adaboost::cuda::core::multiply_gpu(mat1, mat2, result1);
    adaboost::utils::cuda::cuda_event_record(has_happened);
    adaboost::utils::cuda::cuda_event_synchronize(has_happened);
    result1.copy_to_host();
    for(unsigned int i = 0; i < 3; i++)
    {
        for(unsigned int j = 0; j < 3; j++)
        {
            EXPECT_EQ(60.0, result1.at(i, j));
        }
    }
    mat3.set(0, 0, 6.0);
    mat3.set(1, 0, 6.0);
    EXPECT_THROW({
        try
        {
            adaboost::cuda::core::multiply_gpu(mat1, mat3, result1);
        }
        catch(const std::logic_error& e)
        {
            EXPECT_STREQ("Order of matrices don't match.", e.what());
            throw;
        }
    }, std::logic_error);

    //Differently sized matrices
    adaboost::cuda::core::MatrixGPU<float> m1(1, 3), m2(3, 4);
    m1.set(0, 0, 3);
    m1.set(0, 1, 4);
    m1.set(0, 2, 2);

    m2.set(0, 0, 13);
    m2.set(0, 1, 9);
    m2.set(0, 2, 7);
    m2.set(0, 3, 15);
    m2.set(1, 0, 8);
    m2.set(1, 1, 7);
    m2.set(1, 2, 4);
    m2.set(1, 3, 6);
    m2.set(2, 0, 6);
    m2.set(2, 1, 4);
    m2.set(2, 2, 0);
    m2.set(2, 3, 3);

    m1.copy_to_device();
    m2.copy_to_device();
    adaboost::utils::cuda::cuda_event_record(has_happened);
    adaboost::utils::cuda::cuda_event_synchronize(has_happened);
    adaboost::cuda::core::MatrixGPU<float> r1(1, 4);
    adaboost::cuda::core::multiply_gpu(m1, m2, r1);
    adaboost::utils::cuda::cuda_event_record(has_happened);
    adaboost::utils::cuda::cuda_event_synchronize(has_happened);
    r1.copy_to_host();
    EXPECT_EQ(83.0, r1.at(0, 0));
    EXPECT_EQ(63.0, r1.at(0, 1));
    EXPECT_EQ(37.0, r1.at(0, 2));
    EXPECT_EQ(75.0, r1.at(0, 3));
}
