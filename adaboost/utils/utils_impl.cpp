#ifndef ADABOOST_UTILS_UTILS_CPP
#define ADABOOST_UTILS_UTILS_CPP

#include<adaboost/utils/utils.hpp>
#include<string>
#include<stdexcept>

namespace adaboost
{
    namespace utils
    {
        void check(bool exp, const std::string& msg)
        {
            if(!exp)
                throw std::logic_error(msg);
        }
    } // namespace utils
} // namspace adaboost

#endif
