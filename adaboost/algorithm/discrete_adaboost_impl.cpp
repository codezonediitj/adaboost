#ifndef ADABOOST_ALGORITHM_DISCRETE_ADABOOST_IMPL_CPP
#define ADABOOST_ALGORITHM_DISCRETE_ADABOOST_IMPL_CPP

#include<adaboost/algorithm/discrete_adaboost.hpp>
#include<adaboost/algorithm/weak_classifier.hpp>
#include<adaboost/core/data_structures.hpp>
#include<adaboost/memory_manager.hpp>
#include<adaboost/utils/utils.hpp>

namespace adaboost
{
    namespace algorithm
    {

        using namespace adaboost::core;

        template <class data_type>
        BinaryDiscreteAdaBoost<data_type>::
        BinaryDiscreteAdaBoost
        (Vector<WeakClassifier<data_type>*>* _classifiers,
         Vector<data_type>* _pred_weights):
        classifiers(_classifiers),
        pred_weights(_pred_weights)
        {
        }

        template <class data_type>
        BinaryDiscreteAdaBoost<data_type>*
        BinaryDiscreteAdaBoost<data_type>::
        create_BinaryDiscreteAdaBoost
        (Vector<WeakClassifier<data_type>*>* _classifiers,
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
              WeakClassifierFactory<data_type>* classifier_creator)
        {
            adaboost::utils::check(num_itrs != 0, "Number of iterations should be at least 1.");
            this->classifiers = Vector<WeakClassifier<data_type>*>::create_Vector(num_itrs);
            this->pred_weights = Vector<data_type*>::create_Vector(num_itrs);
            for( int itr = 0; itr < num_itrs; itr++ )
            {
                WeakClassifier<data_type>* prev_classifier = itr > 0 ? this->classifiers->at(itr - 1) : NULL;
                this->classifiers->set(itr, classifier_creator->create_WeakClassifier
                                            (prev_classifier->get_classifier_information()));

            }
        }

    }
}

#endif
