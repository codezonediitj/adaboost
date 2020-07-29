#ifndef ADABOOST_CORE_OPERATIONS_HPP
#define ADABOOST_CORE_OPERATIONS_HPP

#include<adaboost/core/data_structures.hpp>

namespace adaboost
{
    namespace core
    {

        /*
        * This function fills the vector with a given value.
        *
        * @param value The value with which the vector is to be populated.
        * @param vec The Vector.
        */

        template <class data_type_vector>
        void fill(data_type_vector value, Vector<data_type_vector>&vec);


        /*
        * This function fills the matrix with a given value.
        *
        * @param value The value with which the matrix is to be populated.
        * @param vec The Matrix.
        */

        template <class data_type_matrix>
        void fill(data_type_matrix value, Matrix<data_type_matrix>&mat);


        /*
        * This function computes the sum of
        * elements of the given vector and the
        * given function applied on each element.
        *
        * @tparam data_type_vector Data type of the elements
        *     supported by C++.
        * @param func_ptr Address of the function which
        *     is to be applied on each element.
        * @param vec Vector whose elements are to be
        *     considered.
        * @param start The starting index from where
        *     the sum is to be computed.
        * @param end The ending index i.e., the last
        *     element which is to be considered.
        * @param result The variable which will store
        *     the result.
        */
        template <class data_type>
        void Sum(
        data_type (*func_ptr)(data_type),
        Vector<data_type>& vec,
        unsigned start,
        unsigned end,
        data_type& result);

        /*
        * This function computes the argument which
        * makes the given function maximum.
        *
        * @tparam data_type_1 The data type, supported by C++,
        *     of the elements of the vector.
        * @tparam data_type_2 The data type, supported by C++,
        *     of the value returned by the function at func_ptr.
        * @param func_ptr Address of the function whose value
        *     is to be maximised.
        * @param vec The domain which is to be considered in the
        *     form of a Vector.
        * @param result The value which maximises the function.
        */
        template <class data_type_1, class data_type_2>
        void Argmax(
        data_type_2 (*func_ptr)(data_type_1),
        Vector<data_type_1>& vec,
        data_type_1& result);

    } // namespace core
} // namespace adaboost

#endif
