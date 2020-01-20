#ifndef ADABOOST_UTILS_UTILS_HPP
#define ADABOOST_UTILS_UTILS_HPP

#include<string>
#include<stdexcept>

namespace adaboost
{
    namespace utils
    {
        void check(bool exp, std::string msg)
        {
            if(!exp)
                throw std::logic_error(msg);
        }
    } // namespace utils
} // namspace adaboost

#endif
