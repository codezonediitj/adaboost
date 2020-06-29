#include<gtest/gtest.h>
#include<string>
#include<adaboost/core/data_structures.hpp>
#include<adaboost/core/operations.hpp>
#include<stdexcept>

float square_1(float x)
{
    return x*x;
}

TEST(Core, Sum)
{
    adaboost::core::Vector<float> vec_f(5);
    fill(2,vec_f);
    float result;
    adaboost::core::Sum(&square_1, vec_f, 0, 9, result);
    EXPECT_EQ(20, result)<<"The sum should be 20.";
}

float square_2(int x)
{
    return -x*x;
}

TEST(Core, Argmax)
{
    adaboost::core::Vector<int> vec_i(5);
    vec_i.set(0, -1.0);
    vec_i.set(1, 0);
    vec_i.set(2, 1);
    vec_i.set(3, 2);
    vec_i.set(4, 3);
    int result;
    adaboost::core::Argmax(&square_2, vec_i, result);
    EXPECT_EQ(0, result)<<"The arg max value is 0.";
}
