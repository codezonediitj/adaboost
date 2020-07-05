#ifndef ADABOOST_CORE_OPERATIONS_IMPL_CPP
#define ADABOOST_CORE_OPERATIONS_IMPL_CPP

#include<adaboost/utils/utils.hpp>
#include<adaboost/core/operations.hpp>
#include<iostream>
#include<cmath>

namespace adaboost
{
    namespace core
    {

        template <class data_type_vector>
        void fill(const data_type_vector value, const Vector<data_type_vector>&vec)
        {
            for(unsigned i = 0; i < vec.get_size(); i++)
            {
                vec.get_data_pointer()[i] = value;
            }
        }


        template <class data_type_matrix>
        void fill(data_type_matrix value, const Matrix<data_type_matrix>&mat)
        {
            for(unsigned i = 0; i < mat.get_rows(); i++)
            {
                for(unsigned j = 0; j < mat.get_cols(); j++)
                {
                    mat.get_data_pointer()[i*mat.get_cols() + j] = value;
                }
            }
        }


        template <class data_type>
        void Sum(
        data_type (*func_ptr)(data_type),
        const Vector<data_type>& vec,
        unsigned start,
        unsigned end,
        data_type& result)
        {
            adaboost::utils::check(vec.get_size() >= start,
            "Start is out of range.");
            end = vec.get_size() - 1 < end ?
                    vec.get_size() - 1 : end;
            result = 0;
            for(unsigned i = start; i <= end; i++)
            {
                result += func_ptr(vec.at(i));
            }
        }

        template <class data_type_1, class data_type_2>
        void Argmax(
        data_type_2 (*func_ptr)(data_type_1),
        const Vector<data_type_1>& vec,
        data_type_1& result)
        {
            data_type_2 max_val = func_ptr(vec.at(0));
            data_type_1 arg_max = vec.at(0);
            for(unsigned i = 0; i < vec.get_size(); i++)
            {
                if(max_val < func_ptr(vec.at(i)))
                {
                    max_val = func_ptr(vec.at(i));
                    arg_max = vec.at(i);
                }
            }
            result = arg_max;
        }

        #include "../templates/instantiated_templates_operations.hpp"

    } // namespace core
} // namespace adaboost

#endif
