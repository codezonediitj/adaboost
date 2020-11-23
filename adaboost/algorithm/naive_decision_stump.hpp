#ifndef ADABOOST_ALGORITHM_NAIVE_DECISION_STUMP_HPP
#define ADABOOST_ALGORITHM_NAIVE_DECISION_STUMP_HPP

#include<adaboost/algorithm/weak_classifier.hpp>

namespace adaboost
{
    namespace algorithm
    {

        template <class data_type>
        struct NaiveDecisionStumpProperties
        {

            unsigned feature_index;

            data_type threshold;

            data_type prediction_weight;

            NaiveDecisionStumpProperties();

        };

        template <class data_type>
        class NaiveDecisionStump: public WeakClassifier<data_type>
        {

            private:

                static NaiveDecisionStumpProperties<data_type>* generate_properties
                (NaiveDecisionStumpProperties<data_type>* prev_classifier_information);

                NaiveDecisionStumpProperties<data_type>* classifier_information;

            public:

                static NaiveDecisionStump* create_NaiveDecisionStump
                (NaiveDecisionStumpProperties<data_type>* _classifier_information=NULL);

                virtual data_type train(Matrix<data_type>* data,
                                        Vector<data_type>* classes,
                                        Vector<data_type>* example_weights,
                                        data_type precision);

                virtual data_type predict(Vector<data_type>* input);

                virtual Vector<data_type>* predict(Matrix<data_type>* input);

                virtual ~NaiveDecisionStump();

            protected:

                NaiveDecisionStump
                (NaiveDecisionStumpProperties<data_type>* _classifier_information);

        };

    }
}

#endif
