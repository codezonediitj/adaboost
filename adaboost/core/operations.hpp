#ifndef ADABOOST_CORE_OPERATIONS_HPP
#define ADABOOST_CORE_OPERATIONS_HPP

#include<adaboost/core/data_structures.hpp>

namespace adaboost
{
    namespace core
    {
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
        const Vector<data_type>& vec,
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
        const Vector<data_type_1>& vec,
        data_type_1& result);

        /*
        * @overload
        * Used for taking dot product of two vectors.
        *
        * @param vec1 First vector in dot product.
        * @param vec2 Second vector in dot product.
        * @param result For storing the result.
        */
        template <class data_type_vector>
        void product(const Vector<data_type_vector>& vec1,
                     const Vector<data_type_vector>& vec2,
                     data_type_vector& result);

        /* @overload
        * Used for multiplyng two matrices.
        *
        * @param vec1 First matrix.
        * @param vec2 Second matrix.
        * @param result A matrix for storing the result.
        */
        template <class data_type_matrix>
        void multiply(const Matrix<data_type_matrix>& mat1,
                     const Matrix<data_type_matrix>& mat2,
                     Matrix<data_type_matrix>& result);

        /* @overload
        * Used for multiplying a vector
        * and a matrix.
        *
        * @param vec The vector.
        * @param mat The matrix.
        * @param result A vector for storing the result.
        */
        template <class data_type_vector, class data_type_matrix>
        void multiply(const Vector<data_type_vector>& vec,
                     const Matrix<data_type_matrix>& mat,
                     Vector<data_type_vector>& result);

    } // namespace core
} // namespace adaboost

#endif
