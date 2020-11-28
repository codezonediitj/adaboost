#ifndef ADABOOST_ALGORITHM_ADABOOST_HPP
#define ADABOOST_ALGORITHM_ADABOOST_HPP

#include<data_structures.hpp>
#include<adaboost/memory_manager.hpp>

namespace adaboost
{
    namespace algorithm
    {

        using namespace adaboost::core;

        template <class data_type>
        class BinaryAdaBoost: public Base
        {

            public:

                virtual data_type train(Matrix<data_type>* data,
                                        Vector<data_type>* classes,
                                        unsigned num_itrs,
                                        data_type precision,
                                        WeakClassifierFactory<data_type>* classifierCreator=NULL) = 0;

                virtual data_type predict(Vector<data_type>* input) = 0;

                virtual Vector<data_type>* predict(Matrix<data_type>* input) = 0;

        };

    }
}

#endif
