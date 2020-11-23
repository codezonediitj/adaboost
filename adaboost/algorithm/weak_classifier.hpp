#ifndef ADABOOST_ALGORITHM_WEAK_CLASSIFIER_HPP
#define ADABOOST_ALGORITHM_WEAK_CLASSIFIER_HPP

#include<adaboost/core/data_structures.hpp>

namespace adaboost
{
    namespace algorithm
    {

        using namespace adaboost::core;

        /*
        * Interface for all the binary weak classifiers.
        *
        * @tparam data_type_vector Data type supported by C++.
        */
        template <class data_type>
        class BinaryWeakClassifier : public Base
        {
            public:

                /*
                * This method should be overrided in specific
                * weak classifiers to implement their training
                * algorithms.
                *
                * @param data Matrix<data_type>* The training data.
                *                                The i-th row will contain the i-th feature
                *                                for all the examples across different colums.
                * @param classes Vector<data_type>* The labels for the training data.
                * @param example_weights Vector<data_type>* Weights of different examples in the training data.
                * @param precision data_type The level of precision to be used during training.
                */
                virtual data_type train(Matrix<data_type>* data,
                                        Vector<data_type>* classes,
                                        Vector<data_type>* example_weights,
                                        data_type precision) = 0;

                /*
                * This method should be overrided in specific
                * weak classifiers to implement their prediction
                * algorithms.
                *
                * @param input Vector<data_type>* The feature vector.
                */
                virtual data_type predict(Vector<data_type>* input) = 0;

                /* @overload
                * This method should be overrided in specific
                * weak classifiers to implement their training
                * algorithms.
                *
                * @param input Matrix<data_type>* The i-th column in the matrix
                *                                 should be the i-th feature vector.
                */
                virtual Vector<data_type>* predict(Matrix<data_type>* input) = 0;

        };

    }
}

#endif
