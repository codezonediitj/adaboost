#ifndef ADABOOST_ALGORITHM_WEAK_CLASSIFIER_HPP
#define ADABOOST_ALGORITHM_WEAK_CLASSIFIER_HPP

#include<adaboost/core/data_structures.hpp>

namespace adaboost
{
    namespace algorithm
    {

        using namespace adaboost::core;

        template <class data_type>
        class WeakClassifier: public Base
        {
            public:

                virtual data_type train(Matrix<data_type>* data,
                                        Vector<data_type>* classes,
                                        Vector<data_type>* example_weights,
                                        data_type precision) = 0;

                virtual data_type predict(Vector<data_type>* input) = 0;

                virtual Vector<data_type>* predict(Matrix<data_type>* input) = 0;

        };

    }
}

#endif
