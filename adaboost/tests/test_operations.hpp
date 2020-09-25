#include<gtest/gtest.h>
#include<string>
#include<adaboost/core/data_structures.hpp>
#include<adaboost/core/operations.hpp>
#include<adaboost/cuda/core/operations.hpp>
#include<stdexcept>

float square_1(float x)
{
    return x*x;
}

TEST(Core, Sum)
{
    using namespace adaboost::core;
    Vector<float>* vec_f = Vector<float>::create_Vector(5);
    fill(float(2), vec_f);
    float result;
    Sum(&square_1, vec_f, 0, 9, result);
    EXPECT_EQ(20, result)<<"The sum should be 20.";
}

float square_2(int x)
{
    return -x*x;
}

TEST(Core, Argmax)
{
    using namespace adaboost::core;
    Vector<int>* vec_i = Vector<int>::create_Vector(5);
    vec_i->set(0, -1.0);
    vec_i->set(1, 0);
    vec_i->set(2, 1);
    vec_i->set(3, 2);
    vec_i->set(4, 3);
    int result;
    Argmax(&square_2, vec_i, result);
    EXPECT_EQ(0, result)<<"The arg max value is 0.";
}
