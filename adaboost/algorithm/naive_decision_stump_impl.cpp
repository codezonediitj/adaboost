#ifndef ADABOOST_ALGORITHM_NAIVE_DECISION_STUMP_IMPL_CPP
#define ADABOOST_ALGORITHM_NAIVE_DECISION_STUMP_IMPL_CPP

#include<adaboost/algorithm/naive_decision_stump.hpp>
#include<adaboost/memory_manager.hpp>
#include<adaboost/core/data_structures.hpp>
#include<adaboost/core/operations.hpp>
#include<cmath>

namespace adaboost
{
    namespace algorithm
    {

        using namespace adaboost::core;

        template <class data_type>
        NaiveDecisionStumpProperties<data_type>::
        NaiveDecisionStumpProperties():
        feature_index(0),
        threshold(0)
        {
        }

        template <class data_type>
        NaiveDecisionStump<data_type>*
        NaiveDecisionStump<data_type>::
        create_NaiveDecisionStump
        (NaiveDecisionStumpProperties<data_type>* _classifier_information)
        {
            NaiveDecisionStump<data_type>* naive_decision_stump =
            new NaiveDecisionStump<data_type>(_classifier_information);
            memory_manager->register_object(naive_decision_stump);
            return naive_decision_stump;
        }

        template <class data_type>
        NaiveDecisionStumpProperties<data_type>*
        NaiveDecisionStump<data_type>::
        generate_properties
        (NaiveDecisionStumpProperties<data_type>* prev_classifier_information)
        {
            NaiveDecisionStumpProperties<data_type>* properties =
            new NaiveDecisionStumpProperties<data_type>();
            if(prev_classifier_information != NULL)
            {
                properties->feature_index = prev_classifier_information->feature_index + 1;
            }
            return properties;
        }

        template <class data_type>
        data_type NaiveDecisionStump<data_type>::train
        (Matrix<data_type>* data,
         Vector<data_type>* classes,
         Vector<data_type>* example_weights,
         data_type precision)
        {
            data_type example_weight_sum, train_score;
            unsigned feature_length = data->get_rows();
            unsigned& feature_index = this->classifier_information->feature_index;
            feature_index %= feature_length;
            unsigned num_thresholds = data->get_cols();

            Sum<data_type>(NULL, example_weights,
                           0, example_weights->get_size() - 1,
                           example_weight_sum);

            for( int idx = 0; idx < num_thresholds; idx++ )
            {
                data_type val;
                data_type thresholds[2];
                val = data->at(feature_index, idx);
                thresholds[0] = val - (data_type) pow(10.0, -(precision + 1));
                thresholds[1] = val + (data_type) pow(10.0, -(precision + 1));
                for( int th = 0; th < 2; th++ )
                {
                    data_type curr_threshold = this->classifier_information->threshold;
                    this->classifier_information->threshold = thresholds[th];
                    Vector<data_type>* result = this->predict(data);
                    Vector<data_type>* equals = Vector<data_type>::create_Vector(result->get_size());
                    is_equal<data_type, data_type>(classes, result, equals);
                    data_type new_train_score;
                    product(equals, example_weights, new_train_score);
                    memory_manager->clear_object(equals);
                    if( new_train_score > train_score )
                    {
                        train_score = new_train_score;
                    }
                    else
                    {
                        this->classifier_information->threshold = curr_threshold;
                    }
                }
            }

            return (example_weight_sum - train_score)/example_weight_sum;
        }

        template <class data_type>
        data_type NaiveDecisionStump<data_type>::predict
        (Vector<data_type>* input)
        {
            unsigned feature_length = input->get_size();
            unsigned& feature_index = this->classifier_information->feature_index;
            feature_index %= feature_length;

            return input->at(feature_index) > this->classifier_information->threshold;
        }

        template <class data_type>
        Vector<data_type>* NaiveDecisionStump<data_type>::predict
        (Matrix<data_type>* input)
        {
            unsigned num_inputs = input->get_cols();
            unsigned feature_length = input->get_rows();
            unsigned& feature_index = this->classifier_information->feature_index;
            feature_index %= feature_length;
            Vector<data_type>* result = Vector<data_type>::create_Vector(num_inputs);

            for(int idx = 0; idx < num_inputs; idx++ )
            {
                result->set(idx, (data_type) (input->at(feature_index, idx) >
                                             this->classifier_information->threshold));

            }

            return result;
        }

        template <class data_type>
        NaiveDecisionStump<data_type>::
        NaiveDecisionStump
        (NaiveDecisionStumpProperties<data_type>* _classifier_information):
        classifier_information(generate_properties(_classifier_information))
        {
        }

        template <class data_type>
        NaiveDecisionStump<data_type>::
        ~NaiveDecisionStump()
        {
        }

        #include<adaboost/templates/instantiated_templates_naive_decision_stump.hpp>

    }
}

#endif
