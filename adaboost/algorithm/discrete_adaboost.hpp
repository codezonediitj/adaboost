#ifndef ADABOOST_ALGORITHM_DISCRETE_ADABOOST_HPP
#define ADABOOST_ALGORITHM_DISCRETE_ADABOOST_HPP

#include<adaboost/algorithm/weak_classifier.hpp>
#include<adaboost/algorithm/adaboost.hpp>
#include<adaboost/algorithm/discrete_adaboost.hpp>
#include<string>

namespace adaboost
{
    namespace algorithm
    {

        using namespace adaboost::core;

        template <class data_type>
        class BinaryDiscreteAdaBoost: public BinaryAdaBoost<data_type>
        {

            private:

                BinaryWeakClassifier<data_type>** classifiers;

                Vector<data_type>* pred_weights;

                Vector<data_type>* training_history;

            public:

                static BinaryDiscreteAdaBoost* create_BinaryDiscreteAdaBoost
                (BinaryWeakClassifier<data_type>** _classifiers=NULL,
                 Vector<data_type>* _pred_weights=NULL);

                virtual data_type train(Matrix<data_type>* data,
                                        Vector<data_type>* classes,
                                        unsigned num_itrs,
                                        unsigned precision,
                                        std::string classifier_type="BinaryNaiveDecisionStump",
                                        bool record_training_history=true,
                                        BinaryWeakClassifierFactory<data_type>* classifier_creator=NULL);

                virtual data_type predict(Vector<data_type>* input);

                virtual Vector<data_type>* predict(Matrix<data_type>* input);

                virtual ~BinaryDiscreteAdaBoost();

            protected:

                BinaryDiscreteAdaBoost
                (BinaryWeakClassifier<data_type>** _classifiers,
                 Vector<data_type>* _pred_weights);

                void update_example_weights(Vector<data_type>* weights,
                                            Vector<data_type>* results,
                                            data_type alpha,
                                            Vector<data_type>* classes);

        };

    }
}

#endif
