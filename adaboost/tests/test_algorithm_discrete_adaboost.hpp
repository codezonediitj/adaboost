#include<gtest/gtest.h>
#include<adaboost/core/data_structures.hpp>
#include<adaboost/algorithm/discrete_adaboost.hpp>
#include<adaboost/core/operations.hpp>

TEST(Algorithm, BinaryDiscreteAdaBoost)
{
    using namespace adaboost::core;
    using namespace adaboost::algorithm;
    using namespace std;
    unsigned examples = 10, features = 2;
    Matrix<double>* X = Matrix<double>::create_Matrix(features, examples);
    Vector<double>* Y = Vector<double>::create_Vector(examples);
    Vector<double>* W = Vector<double>::create_Vector(examples);
    adaboost::core::fill(1.0, W);
    X->set(0, 0, 0.5); X->set(1, 0, 1.0); Y->set(0, 1.0);
    X->set(0, 1, 0.4); X->set(1, 1, 1.4); Y->set(1, 1.0);
    X->set(0, 2, 1.1); X->set(1, 2, 2.2); Y->set(2, 1.0);
    X->set(0, 3, 1.2); X->set(1, 3, 2.3); Y->set(3, 1.0);
    X->set(0, 4, 1.3); X->set(1, 4, 2.4); Y->set(4, 1.0);

    X->set(0, 5, 1.4); X->set(1, 5, 1.9); Y->set(5, -1.0);
    X->set(0, 6, 1.5); X->set(1, 6, 1.8); Y->set(6, -1.0);
    X->set(0, 7, 1.6); X->set(1, 7, 1.7); Y->set(7, -1.0);
    X->set(0, 8, 2.1); X->set(1, 8, 1.5); Y->set(8, -1.0);
    X->set(0, 9, 2.3); X->set(1, 9, 2.4); Y->set(9, -1.0);
    BinaryDiscreteAdaBoost<double>* ensemble = BinaryDiscreteAdaBoost<double>::
                                               create_BinaryDiscreteAdaBoost();
    ensemble->train(X, Y, 3, 4, "BinaryNaiveDecisionStump", false);
    Vector<double>* X_test = Vector<double>::create_Vector(2);
    X_test->set(0, 0.5);
    X_test->set(0, 1.0);
    unsigned pred = (unsigned) ensemble->predict(X_test);
    EXPECT_EQ(1.0, pred)<<"(0.5, 1.0) is predicted to belong to class -1, the correct class is 1";
    Vector<double>* preds = ensemble->predict(X);
    for( int idx = 0; idx < Y->get_size(); idx++ )
    {
        EXPECT_EQ(Y->at(idx), preds->at(idx))<<"The predicted class for "<<idx<<"-th training example is wrong";
    }
}
