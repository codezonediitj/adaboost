#ifndef ADABOOST_ALGORITHM_ADABOOST_HPP
#define ADABOOST_ALGORITHM_ADABOOST_HPP

#include<adaboost/algorithm/weak_classifier.hpp>
#include<adaboost/core/data_structures.hpp>
#include<adaboost/memory_manager.hpp>
#include<string>

namespace adaboost
{
    namespace algorithm
    {

        using namespace adaboost::core;

        /*
        * Interface for all implementations of two class adaboost algorithms.
        *
        * @tparam data_type Data types supported by C++.
        */
        template <class data_type>
        class BinaryAdaBoost: public Base
        {

            public:

                /*
                * This method should be overrided in specific
                * implementations of training algorithms of two
                * class adaboost algorithms.
                *
                * @param data Matrix<data_type>* The training data.
                *                                The i-th row will contain the i-th feature
                *                                for all the examples across different colums.
                * @param classes Vector<data_type>* The labels for the training data.
                *                                   Should be either +1, or -1.
                * @param num_itrs unsigned The number of iterations.
                * @param precision unsigned The level of precision to be used during training.
                * @param classifier_type std::string The weak classifier that should be used.
                *                                    Optional, by default, "BinaryNaiveDecisionStump".
                * @param record_training_history bool If the training history should be recorded.
                *                                     Optional, by default, true.
                * @param classifier_creater BinaryWeakClassifierFactory<data_type>* The factory to create weak classifiers.
                *                                                                   Optional, by default, NULL.
                */
                virtual data_type train(Matrix<data_type>* data,
                                        Vector<data_type>* classes,
                                        unsigned num_itrs,
                                        unsigned precision,
                                        std::string classifier_type="BinaryNaiveDecisionStump",
                                        bool record_training_history=true,
                                        BinaryWeakClassifierFactory<data_type>* classifier_creator=NULL) = 0;

                /*@overload
                * This method should be overrided in specific
                * implementions of two class adaboost algorithms
                * for their prediction algorithms.
                *
                * @param input Vector<data_type>* The feature vector.
                */
                virtual data_type predict(Vector<data_type>* input) = 0;

                /* @overload
                * This method should be overrided in specific
                * implementions of two class adaboost algorithms
                * for their prediction algorithms.
                *
                * @param input Matrix<data_type>* The i-th column in the matrix
                *                                 should be the i-th feature vector.
                */
                virtual Vector<data_type>* predict(Matrix<data_type>* input) = 0;

        };

    }
}

#endif
