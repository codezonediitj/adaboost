#ifndef ADABOOST_UTILS_UTILS_HPP
#define ADABOOST_UTILS_UTILS_HPP

#include<string>
#include<stdexcept>

namespace adaboost
{
    namespace utils
    {
        void check(bool exp, const std::string& msg);
    } // namespace utils
} // namspace adaboost

#endif
