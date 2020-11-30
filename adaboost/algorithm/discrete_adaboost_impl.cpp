#ifndef ADABOOST_ALGORITHM_DISCRETE_ADABOOST_IMPL_CPP
#define ADABOOST_ALGORITHM_DISCRETE_ADABOOST_IMPL_CPP

#include<adaboost/algorithm/discrete_adaboost.hpp>
#include<adaboost/algorithm/weak_classifier.hpp>
#include<adaboost/core/data_structures.hpp>
#include<adaboost/memory_manager.hpp>
#include<adaboost/utils/utils.hpp>
#include<cmath>
#include<string>

namespace adaboost
{
    namespace algorithm
    {

        using namespace adaboost::core;

        template <class data_type>
        BinaryDiscreteAdaBoost<data_type>::
        BinaryDiscreteAdaBoost
        (Vector<BinaryWeakClassifier<data_type>*>* _classifiers,
         Vector<data_type>* _pred_weights):
        classifiers(_classifiers),
        pred_weights(_pred_weights),
        training_history(NULL)
        {
        }

        template <class data_type>
        BinaryDiscreteAdaBoost<data_type>*
        BinaryDiscreteAdaBoost<data_type>::
        create_BinaryDiscreteAdaBoost
        (Vector<BinaryWeakClassifier<data_type>*>* _classifiers,
         Vector<data_type>* _pred_weights)
        {
            BinaryDiscreteAdaBoost<data_type>* binaryDiscreteAdaboost =
            new BinaryDiscreteAdaBoost<data_type>(_classifiers, _pred_weights);
            memory_manager->register_object(binaryDiscreteAdaboost);
            return binaryDiscreteAdaboost;
        }

        template <class data_type>
        data_type BinaryDiscreteAdaBoost<data_type>::
        train(Matrix<data_type>* data, Vector<data_type>* classes,
              unsigned num_itrs, data_type precision,
              std::string classifier_type,
              bool record_training_history,
              BinaryWeakClassifierFactory<data_type>* classifier_creator)
        {
            adaboost::utils::check(num_itrs != 0, "Number of iterations should be at least 1.");
            adaboost::utils::check(classes->get_size() != data->get_cols(),
                                   "Number of class labels are not the same as number of featur vectors");
            if( record_training_history )
            {
                this->training_history = Vector<data_type>::create_Vector(num_itrs);
            }
            unsigned num_examples = data->get_cols();
            Vector<data_type>* example_weights = Vector<data_type>::create_Vector(num_examples);
            adaboost::core::fill((data_type) 1.0, example_weights);
            this->classifiers = Vector<BinaryWeakClassifier<data_type>*>::create_Vector(num_itrs);
            this->pred_weights = Vector<data_type*>::create_Vector(num_itrs);
            for( int itr = 0; itr < num_itrs; itr++ )
            {
                BinaryWeakClassifier<data_type>* prev_classifier = itr > 0 ? this->classifiers->at(itr - 1) : NULL;
                BinaryWeakClassifier<data_type>* curr_classifier = classifier_creator->create_WeakClassifier
                                                             (prev_classifier->get_classifier_information(),
                                                             classifier_type);
                this->classifiers->set(itr, curr_classifier);
                data_type best_error = curr_classifier->train(data, classes, example_weights, precision);
                data_type alpha = log((double)((1 - best_error)/best_error);
                this->pred_weights->set(itr, alpha);
                Vector<data_type>* result = curr_classifier->predict(data);
                this->update_example_weights(example_weights, result, alpha);
                if( record_training_history )
                {
                    this->training_history->set(itr, best_error);
                }
            }
        }

        template <class data_type>
        data_type BinaryDiscreteAdaBoost<data_type>::predict
        (Vector<data_type>* input)
        {
            data_type result = 0.0;
            for( unsigned idx = 0; idx < this->classifiers->get_size(); idx++ )
            {
                data_type curr_pred = this->classifiers->at(idx)->predict(input);
                data_type curr_pred_weight = this->pred_weights->at(idx);
                result += curr_pred * curr_pred_weight;
            }
            return result > 0.0 ? 1.0 : 0.0;
        }

        template <class data_type>
        Vector<data_type>* BinaryDiscreteAdaBoost<data_type>::predict
        (Matrix<data_type>* input)
        {
            Vector<data_type>* results;
            adaboost::core::fill(0.0, results);
            for( unsigned idx = 0; idx < this->classifiers->get_size(); idx++ )
            {
                idx = this->classifiers->at(idx)->predict(idx);
            }
        }

        #include<adaboost/templates/instantiated_templates_discrete_adaboost.hpp>

    }
}

#endif
