#ifndef ADABOOST_ALGORITHM_NAIVE_DECISION_STUMP_IMPL_CPP
#define ADABOOST_ALGORITHM_NAIVE_DECISION_STUMP_IMPL_CPP

#include<adaboost/algorithm/naive_decision_stump.hpp>
#include<adaboost/memory_manager.hpp>
#include<adaboost/core/data_structures.hpp>
#include<adaboost/core/operations.hpp>
#include<cmath>
#include<iostream>

namespace adaboost
{
    namespace algorithm
    {

        using namespace adaboost::core;

        template <class data_type>
        BinaryNaiveDecisionStumpProperties<data_type>::
        BinaryNaiveDecisionStumpProperties():
        feature_index(0),
        threshold(0),
        direction(true)
        {
        }

        template <class data_type>
        BinaryNaiveDecisionStump<data_type>*
        BinaryNaiveDecisionStump<data_type>::
        create_BinaryNaiveDecisionStump
        (BinaryNaiveDecisionStumpProperties<data_type>* _classifier_information)
        {
            BinaryNaiveDecisionStump<data_type>* naive_decision_stump =
            new BinaryNaiveDecisionStump<data_type>(_classifier_information);
            memory_manager->register_object(naive_decision_stump);
            return naive_decision_stump;
        }

        template <class data_type>
        Properties* BinaryNaiveDecisionStump<data_type>::
        get_classifier_information()
        {
            return this->classifier_information;
        }

        template <class data_type>
        BinaryNaiveDecisionStumpProperties<data_type>*
        BinaryNaiveDecisionStump<data_type>::
        generate_properties
        (BinaryNaiveDecisionStumpProperties<data_type>* prev_classifier_information)
        {
            BinaryNaiveDecisionStumpProperties<data_type>* properties =
            new BinaryNaiveDecisionStumpProperties<data_type>();
            if(prev_classifier_information != NULL)
            {
                properties->feature_index = prev_classifier_information->feature_index + 1;
            }
            return properties;
        }

        template <class data_type>
        data_type BinaryNaiveDecisionStump<data_type>::train
        (Matrix<data_type>* data,
         Vector<data_type>* classes,
         Vector<data_type>* example_weights,
         data_type precision)
        {
            data_type example_weight_sum, train_score = 0.0;
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
                bool directions[2] = {true, false};
                for( unsigned th = 0; th < 2; th++ )
                {
                    for( unsigned dir = 0; dir < 2; dir++ )
                    {
                        data_type curr_threshold = this->classifier_information->threshold;
                        bool curr_dir = this->classifier_information->direction;
                        this->classifier_information->threshold = thresholds[th];
                        this->classifier_information->direction = directions[th];
                        Vector<data_type>* result = this->predict(data);
                        Vector<data_type>* equals = Vector<data_type>::create_Vector(result->get_size());
                        is_equal<data_type, data_type>(classes, result, equals);
                        data_type new_train_score;
                        product(equals, example_weights, new_train_score);
                        memory_manager->clear_object(equals);
                        if( new_train_score > train_score &&
                            new_train_score < example_weight_sum &&
                            new_train_score > example_weight_sum*0.5 )
                        {
                            train_score = new_train_score;
                        }
                        else
                        {
                            this->classifier_information->threshold = curr_threshold;
                            this->classifier_information->direction = curr_dir;
                        }
                    }
                }
            }

            return (example_weight_sum - train_score)/example_weight_sum;
        }

        template <class data_type>
        data_type BinaryNaiveDecisionStump<data_type>::predict
        (Vector<data_type>* input)
        {
            unsigned feature_length = input->get_size();
            unsigned& feature_index = this->classifier_information->feature_index;
            feature_index %= feature_length;

            return this->classifier_information->direction ?
                   (input->at(feature_index) > this->classifier_information->threshold ? 1.0 : -1.0):
                   (input->at(feature_index) < this->classifier_information->threshold ? 1.0 : -1.0);
        }

        template <class data_type>
        Vector<data_type>* BinaryNaiveDecisionStump<data_type>::predict
        (Matrix<data_type>* input)
        {
            unsigned num_inputs = input->get_cols();
            unsigned feature_length = input->get_rows();
            unsigned& feature_index = this->classifier_information->feature_index;
            feature_index %= feature_length;
            Vector<data_type>* result = Vector<data_type>::create_Vector(num_inputs);

            for(int idx = 0; idx < num_inputs; idx++ )
            {
                data_type curr_pred = this->classifier_information->direction ?
                                      (input->at(feature_index, idx) > this->classifier_information->threshold ? 1.0 : -1.0):
                                      (input->at(feature_index, idx) < this->classifier_information->threshold ? 1.0 : -1.0);
                result->set(idx, curr_pred);
            }

            return result;
        }

        template <class data_type>
        BinaryNaiveDecisionStump<data_type>::
        BinaryNaiveDecisionStump
        (BinaryNaiveDecisionStumpProperties<data_type>* _classifier_information):
        classifier_information(generate_properties(_classifier_information))
        {
        }

        template <class data_type>
        BinaryNaiveDecisionStump<data_type>::
        ~BinaryNaiveDecisionStump()
        {
        }

        #include<adaboost/templates/instantiated_templates_naive_decision_stump.hpp>

    }
}

#endif
