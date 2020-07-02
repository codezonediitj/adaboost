#include<gtest/gtest.h>
#include<string>
#include<adaboost/core/data_structures.hpp>
#include<adaboost/core/operations.hpp>
#include<stdexcept>

TEST(Core, Vector)
{
    adaboost::core::Vector<float> vec_f;
    EXPECT_EQ(0, vec_f.get_size())<<"Default size should be 0";
    adaboost::core::Vector<float>
    vec(2), vec1(5), vec2(5);
    std::string msg1 = "The size of vector must be 5";
    EXPECT_EQ(5, vec1.get_size())<<msg1;
    adaboost::core::fill(float(4),vec1);
    for(unsigned int i = 0; i < vec1.get_size(); i++)
    {
        std::string msg2 = "The value must be 4.0";
        EXPECT_EQ(4.0, vec1.at(i))<<msg2;
    }
    vec1.set(2, 6.1);
    fill(3.0F,vec2);
    vec2.set(2, 0);
    float result;
    adaboost::core::product(vec1, vec2, result);
    std::string msg3 = "The product of two vectors must be 48.0";
    EXPECT_EQ(48.0, result)<<msg3;
    EXPECT_THROW({
        try
        {
            adaboost::core::product(vec, vec1, result);
        }
        catch(const std::logic_error& e)
        {
            EXPECT_STREQ("Size of vectors don't match.", e.what());
            throw;
        }
    }, std::logic_error);
}

TEST(Core, Matrices)
{
    adaboost::core::Matrix<float> mat_f;
    EXPECT_EQ(0, mat_f.get_cols())<<"Number of columns should be 0";
    EXPECT_EQ(0, mat_f.get_rows())<<"Number of rows should be 0.";
    adaboost::core::Matrix<float> mat1(3, 3), mat2(3, 3), mat3(2, 1);
    fill(4.0F,mat1);
    fill(5.0F,mat2);
    adaboost::core::Matrix<float> result1(3, 3);
    adaboost::core::multiply(mat1, mat2, result1);
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
            adaboost::core::multiply(mat1, mat3, result1);
        }
        catch(const std::logic_error& e)
        {
            EXPECT_STREQ("Order of matrices don't match.", e.what());
            throw;
        }
    }, std::logic_error);
    adaboost::core::Vector<float> vec(2);
    fill(2.0F,vec);
    adaboost::core::Vector<float> result2(1);
    adaboost::core::multiply(vec, mat3, result2);
    EXPECT_EQ(24, result2.at(0));
    adaboost::core::Vector<float> vec_f(1);
    EXPECT_THROW({
        try
        {
            adaboost::core::multiply(vec_f, mat3, result2);
        }
        catch(const std::logic_error& e)
        {
            EXPECT_STREQ("Orders mismatch in the inputs.", e.what());
            throw;
        }
    }, std::logic_error);
}
