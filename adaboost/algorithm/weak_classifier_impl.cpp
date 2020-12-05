#ifndef ADABOOST_ALGORITHM_WEAK_CLASSIFIER_IMPL_CPP
#define ADABOOST_ALGORITHM_WEAK_CLASSIFIER_IMPL_CPP

#include<adaboost/core/data_structures.hpp>
#include<adaboost/algorithm/weak_classifier.hpp>
#include<adaboost/algorithm/naive_decision_stump.hpp>
#include<adaboost/utils/utils.hpp>
#include<string>

namespace adaboost
{
    namespace algorithm
    {

        using namespace adaboost::core;

        Properties::
        ~Properties()
        {
        }

        template <class data_type>
        BinaryWeakClassifier<data_type>*
        BinaryWeakClassifierFactory<data_type>::
        create_BinaryWeakClassifier
        (Properties* classifier_information,
         std::string classifier_type)
        {
            BinaryWeakClassifier<data_type>* weak_classifier;
            if( classifier_type == "BinaryNaiveDecisionStump" )
            {
                BinaryNaiveDecisionStumpProperties<data_type>* _info =
                dynamic_cast<BinaryNaiveDecisionStumpProperties<data_type>*>(classifier_information);
                weak_classifier = BinaryNaiveDecisionStump<data_type>
                                    ::create_BinaryNaiveDecisionStump(_info);
            }
            else
            {
                std::string msg = "Currently, " + classifier_type + " isn't supported by BinaryAdaBoost.";
                adaboost::utils::check(false, msg);
            }
            return weak_classifier;
        }

        template <class data_type>
        BinaryWeakClassifierFactory<data_type>*
        BinaryWeakClassifierFactory<data_type>::
        create_BinaryWeakClassifierFactory
        ()
        {
            BinaryWeakClassifierFactory<data_type>* factory =
            new BinaryWeakClassifierFactory<data_type>();
            memory_manager->register_object(factory);
            return factory;
        };

        template <class data_type>
        BinaryWeakClassifierFactory<data_type>::
        BinaryWeakClassifierFactory()
        {
        };

        template <class data_type>
        BinaryWeakClassifierFactory<data_type>::
        ~BinaryWeakClassifierFactory()
        {
        };

        #include "adaboost/templates/instantiated_templates_weak_classifier.hpp"

    }
}

#endif
