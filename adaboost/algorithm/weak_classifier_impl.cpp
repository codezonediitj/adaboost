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

        template <class data_type>
        BinaryWeakClassifier<data_type>*
        BinaryWeakClassifierFactory<data_type>::
        create_BinaryWeakClassifier
        (Properties* classifier_information,
         std::string classifier_type)
        {
            BinaryWeakClassifier<data_type>* weak_classifier;
            switch(classifier_type)
            {
                case "BinaryNaiveDecisionStump":

                    weak_classifier = BinaryNaiveDecisionStump<data_type>
                                      ::create_BinaryNaiveDecisionStump(classifier_information);
                    break;

                default:

                    std::string msg = "Currently, " + classifier_type + " isn't supported by BinaryAdaBoost.";
                    check(false, msg);
            }
            return weak_classifier;
        };

        #include "adaboost/templates/instantiated_templates_weak_classifier.hpp"

    }
}

#endif
