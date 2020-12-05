#include<gtest/gtest.h>
#include<string>
#include<adaboost/core/data_structures.hpp>
#include<adaboost/core/operations.hpp>
#include<adaboost/algorithm/naive_decision_stump.hpp>
#include<adaboost/memory_manager.hpp>
#include<stdexcept>

TEST(Algorithm, BinaryNaiveDecisionStump)
{
    using namespace adaboost::core;
    using namespace adaboost::algorithm;
    Matrix<double>* X = Matrix<double>::create_Matrix(2, 5);
    Vector<double>* Y = Vector<double>::create_Vector(5);
    Vector<double>* W = Vector<double>::create_Vector(5);
    adaboost::core::fill<double>(1.0, W);
    X->set(0, 0, 3.0); X->set(1, 0, 3.0); Y->set(0, 0.0);
    X->set(0, 1, 5.0); X->set(1, 1, 5.0); Y->set(1, 1.0);
    X->set(0, 2, 1.0); X->set(1, 2, 5.0); Y->set(2, 1.0);
    X->set(0, 3, 1.0); X->set(1, 3, 1.0); Y->set(3, 1.0);
    X->set(0, 4, 3.0); X->set(1, 4, 1.0); Y->set(4, 1.0);
    BinaryNaiveDecisionStump<double>* weak_classifier = BinaryNaiveDecisionStump<double>::create_BinaryNaiveDecisionStump();
    double best_error = weak_classifier->train(X, Y, W, 3);
    EXPECT_EQ(0.2, best_error)<<"Best training error of BinaryNaiveDecisionStump should be 0.2 for the given testing data";
    adaboost::memory_manager->clear_all();
}
