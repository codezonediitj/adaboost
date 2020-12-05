#ifndef ADABOOST_ALGORITHM_NAIVE_DECISION_STUMP_HPP
#define ADABOOST_ALGORITHM_NAIVE_DECISION_STUMP_HPP

#include<adaboost/algorithm/weak_classifier.hpp>
#include<adaboost/algorithm/discrete_adaboost.hpp>

namespace adaboost
{
    namespace algorithm
    {

        /*
        * Properties of the naive decision stump classifier.
        *
        * @tparam data_type_vector Data type supported by C++.
        */
        template <class data_type>
        struct BinaryNaiveDecisionStumpProperties: public Properties
        {

            //! Index of the feature which should be used to classify the input.
            unsigned feature_index;

            //! The threshold to be used for classification.
            data_type threshold;

            bool direction;

            /*
            * Intialises the data attributes to default values.
            */
            BinaryNaiveDecisionStumpProperties();

        };

        /*
        * Decision stump classifier which use the brute force
        * method in it's training algorithm.
        *
        * @tparam data_type_vector Data type supported by C++.
        */
        template <class data_type>
        class BinaryNaiveDecisionStump: public BinaryWeakClassifier<data_type>
        {

            private:

                /*
                * Calculates the feature index to be used for classification
                * using the properties of the naive decision stump used in the
                * last step of adaboost algorithm.
                *
                * @param prev_classifier_information BinaryNaiveDecisionStumpProperties<data_type>*
                */
                static BinaryNaiveDecisionStumpProperties<data_type>* generate_properties
                (BinaryNaiveDecisionStumpProperties<data_type>* prev_classifier_information);

                //! The properties of the current naive decision stump classifier.
                BinaryNaiveDecisionStumpProperties<data_type>* classifier_information;

            public:

                static BinaryNaiveDecisionStump* create_BinaryNaiveDecisionStump
                (BinaryNaiveDecisionStumpProperties<data_type>* _classifier_information=NULL);

                virtual data_type train(Matrix<data_type>* data,
                                        Vector<data_type>* classes,
                                        Vector<data_type>* example_weights,
                                        data_type precision);

                virtual data_type predict(Vector<data_type>* input);

                virtual Vector<data_type>* predict(Matrix<data_type>* input);

                virtual Properties* get_classifier_information();

                /*
                * Frees memory acquired by the current object.
                */
                virtual ~BinaryNaiveDecisionStump();

            protected:

                /*
                * Initiaises the properties of the given classifier using
                * the properties of the naive decision stump used in the last
                * step of adaboost algorithm.
                *
                * @param _classifier_information BinaryNaiveDecisionStumpProperties<data_type>*
                */
                BinaryNaiveDecisionStump
                (BinaryNaiveDecisionStumpProperties<data_type>* _classifier_information);

        };

    }
}

#endif
