#ifndef ADABOOST_ALGORITHM_DISCRETE_ADABOOST_HPP
#define ADABOOST_ALGORITHM_DISCRETE_ADABOOST_HPP

#include<adaboost/algorithm/weak_classifier.hpp>
#include<adaboost/algorithm/adaboost.hpp>
#include<data_structures.hpp>

namespace adaboost
{
    namespace algorithm
    {

        using namespace adaboost::core;

        template <class data_type>
        class BinaryDiscreteAdaBoost: public BinaryAdaBoost<data_type>
        {

            private:

                Vector<WeakClassifier<data_type>*>* classifiers;

                Vector<data_type>* pred_weights;

            public:

                static BinaryDiscreteAdaBoost* create_BinaryDiscreteAdaBoost
                (Vector<WeakClassifier<data_type>*>* _classifiers=NULL,
                 Vector<data_type>* _pred_weights=NULL);

                virtual data_type train(Matrix<data_type>* data,
                                        Vector<data_type>* classes,
                                        unsigned num_itrs,
                                        data_type precision,
                                        WeakClassifierFactory<data_type>* classifierCreator=NULL);

                virtual data_type predict(Vector<data_type>* input);

                virtual Vector<data_type>* predict(Matrix<data_type>* input);

                virtual ~BinaryAdaBoost();

            protected:

                BinaryDiscreteAdaBoost
                (Vector<WeakClassifier<data_type>*>* _classifiers,
                 Vector<data_type>* _pred_weights);

        };

    }
}

#endif
